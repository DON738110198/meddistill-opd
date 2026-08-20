from __future__ import annotations

import asyncio
import builtins
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import medical_opd.config as config_module
import medical_opd.evaluation as evaluation_module
from medical_opd.config import ExperimentConfig, load_config
from medical_opd.evaluation import (
    EvalRequest,
    _load_student_tokenizer,
    build_teacher_gate,
    plan_evaluation,
    run_evaluation,
)
from medical_opd.io_utils import atomic_write_json, sha256_file, stable_hash, write_jsonl


class _FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"A": 1, "B": 2, "medical": 3}

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return "\n".join(message["content"] for message in messages)

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return list(range(1, len(text.split()) + 2))


def _eval_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    loaded = load_config()
    config_path = tmp_path / "experiment.toml"
    config_path.write_bytes(loaded.path.read_bytes())
    config = ExperimentConfig(config_path, copy.deepcopy(loaded.raw))
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    atomic_write_json(processed / "manifest.json", {"status": "frozen"})
    atomic_write_json(
        processed / "tokenizer_compatibility.json",
        {
            "student_vocab_sha256": stable_hash(_FakeTokenizer().get_vocab()),
            "vocab_equal": True,
            "special_token_ids_equal": True,
        },
    )
    generated = tmp_path / "reports" / "generated"
    generated.mkdir(parents=True)
    atomic_write_json(
        generated / "preflight_latest.json",
        {
            "status": "ready",
            "config_sha256": sha256_file(config.path),
            "configured_models": {
                "student": config.get("models", "student"),
                "teacher": config.get("models", "teacher"),
            },
        },
    )
    return config


def _eval_row(index: int, *, split: str = "dev") -> dict[str, Any]:
    return {
        "id": f"eval-{split}-{index}",
        "question": f"medical question {index}",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "answer_idx": "A",
        "subject": "medical",
        "source_index": index,
        "source_split": split,
    }


def _prices(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "version": "test-price-v1",
        "items": [
            {
                "display_name": config.get("models", role),
                "prices": {
                    "prefill": {"unit_price": 100},
                    "sample": {"unit_price": 200},
                    "train": {"unit_price": 300},
                },
            }
            for role in ("student", "teacher")
        ],
    }


def test_evaluation_paid_guard_fires_before_manifest_sdk_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load_config()
    config = ExperimentConfig(tmp_path / "missing.toml", copy.deepcopy(loaded.raw))
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path / "missing-root")

    async def unexpected_remote(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("evaluation reached paid async work")

    monkeypatch.setattr(evaluation_module, "_evaluate_async", unexpected_remote)
    request = EvalRequest(
        "M0",
        str(config.get("models", "student")),
        "medical_proxy",
        tmp_path / "eval",
    )

    with pytest.raises(ValueError, match="confirm-paid"):
        run_evaluation(config, request)


def test_blocked_preflight_never_enters_paid_evaluation_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    atomic_write_json(
        tmp_path / "reports" / "generated" / "preflight_latest.json",
        {
            "status": "blocked",
            "config_sha256": sha256_file(config.path),
            "configured_models": {
                "student": config.get("models", "student"),
                "teacher": config.get("models", "teacher"),
            },
        },
    )
    paid_runner_called = False

    async def unexpected_remote(*args: object, **kwargs: object) -> dict[str, str]:
        nonlocal paid_runner_called
        paid_runner_called = True
        raise AssertionError("blocked preflight reached paid evaluation")

    monkeypatch.setattr(evaluation_module, "_evaluate_async", unexpected_remote)
    request = EvalRequest(
        "M0",
        str(config.get("models", "student")),
        "medical_proxy",
        tmp_path / "eval",
        confirm_paid=True,
    )

    with pytest.raises(RuntimeError, match="preflight is blocked or stale"):
        run_evaluation(config, request)
    assert paid_runner_called is False


@pytest.mark.parametrize("role", ["student", "teacher"])
def test_plan_evaluation_never_imports_pytrio_or_creates_paid_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    write_jsonl(
        tmp_path / "data" / "processed" / "eval_medical_proxy.jsonl",
        [_eval_row(index) for index in range(100)],
    )
    monkeypatch.setattr(
        evaluation_module,
        "_load_student_tokenizer",
        lambda config: _FakeTokenizer(),
    )
    monkeypatch.setattr(evaluation_module, "fetch_json", lambda url: _prices(config))
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "pytrio":
            raise AssertionError("plan-eval imported the paid PyTRIO SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    request = EvalRequest(
        model_label="teacher-raw" if role == "teacher" else "M0",
        base_model=str(config.get("models", role)),
        dataset="medical_proxy",
        output_dir=tmp_path / "eval-plan",
    )

    plan = plan_evaluation(config, request)

    assert plan["mode"] == "remote_paid_evaluation"
    assert plan["dataset_count"] == 100
    assert plan["tokenizer_audit_sha256"] == sha256_file(
        tmp_path / "data" / "processed" / "tokenizer_compatibility.json"
    )
    assert plan["rag_corpus_sha256"] is None
    usage = plan["upper_bound_usage"]
    charged_prefix = "teacher" if role == "teacher" else "student"
    other_prefix = "student" if role == "teacher" else "teacher"
    assert usage[f"{charged_prefix}_prefill_tokens"] > 0
    assert usage[f"{charged_prefix}_sample_tokens"] == 100 * 16
    assert usage[f"{other_prefix}_prefill_tokens"] == 0
    assert not request.output_dir.exists()


def test_evaluation_always_loads_frozen_student_tokenizer_for_teacher_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    calls: list[tuple[str, dict[str, Any]]] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, **kwargs: Any) -> _FakeTokenizer:
            calls.append((model, kwargs))
            return _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_AutoTokenizer))

    tokenizer = _load_student_tokenizer(config)

    assert isinstance(tokenizer, _FakeTokenizer)
    assert calls == [
        (
            str(config.get("models", "student")),
            {
                "cache_dir": str(tmp_path.parent / ".cache" / "huggingface" / "hub"),
                "local_files_only": True,
            },
        )
    ]


def _cache_contract(config: ExperimentConfig, request: EvalRequest) -> dict[str, Any]:
    processed = config.root / "data" / "processed"
    dataset_path = processed / evaluation_module.DATASETS[request.dataset]
    return {
        "model_label": request.model_label,
        "base_model": request.base_model,
        "model_path": request.model_path,
        "dataset": request.dataset,
        "dataset_sha256": sha256_file(dataset_path),
        "rag": request.rag,
        "config_sha256": sha256_file(config.path),
        "tokenizer_audit_sha256": sha256_file(processed / "tokenizer_compatibility.json"),
        "rag_corpus_sha256": (
            sha256_file(processed / "rag_corpus.jsonl") if request.rag else None
        ),
    }


def _block_service_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pytrio = SimpleNamespace(
        configure=lambda **kwargs: None,
        ServiceClient=lambda: (_ for _ in ()).throw(
            AssertionError("cache mismatch reached paid ServiceClient")
        ),
    )
    monkeypatch.setitem(sys.modules, "pytrio", fake_pytrio)


@pytest.mark.parametrize("drift", ["dataset", "tokenizer"])
def test_evaluation_cache_rejects_frozen_artifact_drift_before_paid_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    dataset_path = tmp_path / "data" / "processed" / "eval_medical_full.jsonl"
    write_jsonl(dataset_path, [_eval_row(0, split="test")])
    request = EvalRequest(
        "M0",
        str(config.get("models", "student")),
        "medical_full",
        tmp_path / "eval-cache",
        confirm_paid=True,
    )
    request.output_dir.mkdir()
    atomic_write_json(request.output_dir / "request.json", _cache_contract(config, request))
    if drift == "dataset":
        dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        audit = tmp_path / "data" / "processed" / "tokenizer_compatibility.json"
        audit.write_text(audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _block_service_client(monkeypatch)

    with pytest.raises(RuntimeError, match="cache request does not match"):
        asyncio.run(evaluation_module._evaluate_async(config, request))


def test_rag_evaluation_cache_rejects_corpus_drift_before_paid_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    processed = tmp_path / "data" / "processed"
    write_jsonl(processed / "eval_medical_proxy.jsonl", [_eval_row(i) for i in range(100)])
    corpus_path = processed / "rag_corpus.jsonl"
    write_jsonl(
        corpus_path,
        [{"doc_id": f"doc-{i}", "text": f"medical reference {i}"} for i in range(100)],
    )
    request = EvalRequest(
        "M1",
        str(config.get("models", "student")),
        "medical_proxy",
        tmp_path / "eval-rag-cache",
        rag=True,
        confirm_paid=True,
    )
    request.output_dir.mkdir()
    atomic_write_json(request.output_dir / "request.json", _cache_contract(config, request))
    with corpus_path.open("a", encoding="utf-8") as handle:
        handle.write('{"doc_id":"doc-new","text":"changed"}\n')
    _block_service_client(monkeypatch)

    with pytest.raises(RuntimeError, match="cache request does not match"):
        asyncio.run(evaluation_module._evaluate_async(config, request))


def test_tokenizer_audit_mismatch_blocks_before_paid_service_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    write_jsonl(
        tmp_path / "data" / "processed" / "eval_medical_full.jsonl",
        [_eval_row(0, split="test")],
    )
    request = EvalRequest(
        "M0",
        str(config.get("models", "student")),
        "medical_full",
        tmp_path / "eval-tokenizer-gate",
        confirm_paid=True,
    )
    request.output_dir.mkdir()
    atomic_write_json(request.output_dir / "request.json", _cache_contract(config, request))

    def reject_tokenizer(config: ExperimentConfig) -> None:
        raise RuntimeError("local evaluation tokenizer differs from the frozen audit")

    monkeypatch.setattr(evaluation_module, "_load_student_tokenizer", reject_tokenizer)
    _block_service_client(monkeypatch)

    with pytest.raises(RuntimeError, match="tokenizer differs"):
        asyncio.run(evaluation_module._evaluate_async(config, request))


def _write_gate_summaries(
    config: ExperimentConfig,
    tmp_path: Path,
) -> dict[str, Path]:
    processed = config.root / "data" / "processed"
    medical_path = processed / "eval_medical_proxy.jsonl"
    general_path = processed / "eval_general_proxy.jsonl"
    write_jsonl(medical_path, [_eval_row(index) for index in range(100)])
    write_jsonl(
        general_path,
        [
            {**_eval_row(index), "id": f"general-val-{index}", "subject": "logic"}
            for index in range(100)
        ],
    )
    paths = {name: tmp_path / name for name in (
        "base_medical", "base_general", "teacher_medical", "teacher_general"
    )}
    for name, path in paths.items():
        path.mkdir()
        medical = "medical" in name
        teacher = name.startswith("teacher")
        atomic_write_json(
            path / "summary.json",
            {
                "status": "completed",
                "dataset": "medical_proxy" if medical else "general_proxy",
                "base_model": config.get("models", "teacher" if teacher else "student"),
                "model_path": None,
                "rag": False,
                "accuracy": 0.8 if teacher else 0.5,
                "count": 100,
                "dataset_sha256": sha256_file(medical_path if medical else general_path),
                "config_sha256": sha256_file(config.path),
                "tokenizer_audit_sha256": sha256_file(
                    processed / "tokenizer_compatibility.json"
                ),
                "protocol_id": config.get("evaluation", "thinking_protocol_id"),
                "prompt_contract": (
                    "Chinese choice prompt rendered once with the student tokenizer"
                ),
                "thinking": True,
                "max_tokens": config.get(
                    "evaluation",
                    "thinking_medical_max_tokens"
                    if medical
                    else "thinking_general_max_tokens",
                ),
                "temperature": config.get("evaluation", "thinking_temperature"),
                "top_p": config.get("evaluation", "thinking_top_p"),
                "seed_contract": "experiment.seed + frozen row index",
                "limit": 0,
                "scope": "proxy",
            },
        )
    return paths


@pytest.mark.parametrize(
    ("summary_name", "field", "value"),
    [
        ("base_medical", "base_model", "wrong/student"),
        ("teacher_medical", "base_model", "wrong/teacher"),
        ("teacher_general", "model_path", "trio://not-raw"),
        ("base_general", "rag", True),
    ],
)
def test_teacher_gate_rejects_wrong_or_non_raw_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_name: str,
    field: str,
    value: object,
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    paths = _write_gate_summaries(config, tmp_path)
    summary_path = paths[summary_name] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    atomic_write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="not the frozen raw model"):
        build_teacher_gate(
            config,
            paths["base_medical"],
            paths["base_general"],
            paths["teacher_medical"],
            paths["teacher_general"],
        )
    assert not (tmp_path / "reports" / "generated" / "teacher_gate.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("count", 99),
        ("dataset_sha256", "0" * 64),
        ("prompt_contract", "teacher tokenizer rendered this prompt"),
        ("protocol_id", "old-non-thinking"),
        ("thinking", False),
        ("max_tokens", 16),
        ("seed_contract", "unseeded"),
    ],
)
def test_teacher_gate_rejects_stale_or_noncomparable_evaluation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    config = _eval_config(tmp_path, monkeypatch)
    paths = _write_gate_summaries(config, tmp_path)
    summary_path = paths["teacher_medical"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    atomic_write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="frozen evaluation contract failed"):
        build_teacher_gate(
            config,
            paths["base_medical"],
            paths["base_general"],
            paths["teacher_medical"],
            paths["teacher_general"],
        )
