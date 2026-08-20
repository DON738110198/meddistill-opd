from __future__ import annotations

import asyncio
import builtins
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import medical_opd.config as config_module
import medical_opd.thinking_eval as thinking_module
from medical_opd.config import ExperimentConfig, load_config
from medical_opd.io_utils import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_jsonl,
)
from medical_opd.thinking_eval import (
    ThinkingEvalRequest,
    plan_thinking_evaluation,
    run_thinking_evaluation,
)


class _FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"A": 1, "B": 2, "medical": 3}

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        assert kwargs["enable_thinking"] is True
        return "\n".join(message["content"] for message in messages)

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return list(range(1, len(text.split()) + 2))


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
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


def _row(index: int) -> dict[str, Any]:
    return {
        "id": f"medical-dev-{index}",
        "question": f"medical question {index}",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "answer_idx": "A",
        "subject": "medical",
        "source_index": index,
        "source_split": "dev",
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


def test_thinking_evaluation_paid_guard_fires_before_local_or_remote_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_config()
    config = ExperimentConfig(tmp_path / "missing.toml", copy.deepcopy(loaded.raw))
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path / "missing-root")

    async def unexpected_remote(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("thinking evaluation reached paid async work")

    monkeypatch.setattr(thinking_module, "_evaluate_thinking_async", unexpected_remote)
    request = ThinkingEvalRequest(
        "M0",
        str(config.get("models", "student")),
        "medical_proxy",
        tmp_path / "eval",
    )

    with pytest.raises(ValueError, match="confirm-paid"):
        run_thinking_evaluation(config, request)


@pytest.mark.parametrize(
    ("dataset", "expected_cap"),
    [("medical_proxy", 1024), ("general_proxy", 8192)],
)
def test_thinking_plan_locks_protocol_and_never_imports_pytrio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    expected_cap: int,
) -> None:
    config = _config(tmp_path, monkeypatch)
    filename = thinking_module.DATASETS[dataset]
    write_jsonl(tmp_path / "data" / "processed" / filename, [_row(i) for i in range(100)])
    monkeypatch.setattr(thinking_module, "_load_student_tokenizer", lambda config: _FakeTokenizer())
    monkeypatch.setattr(thinking_module, "fetch_json", lambda url: _prices(config))
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "pytrio":
            raise AssertionError("thinking evaluation plan imported PyTRIO")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    request = ThinkingEvalRequest(
        "M0",
        str(config.get("models", "student")),
        dataset,
        tmp_path / "plan",
        limit=10,
    )

    plan = plan_thinking_evaluation(config, request)

    assert plan["protocol_id"] == "thinking-fixed-v1"
    assert plan["thinking"] is True
    assert plan["max_tokens"] == expected_cap
    assert plan["dataset_count"] == 10
    assert plan["seed_contract"] == "experiment.seed + frozen row index"
    assert plan["upper_bound_usage"]["student_sample_tokens"] == 10 * expected_cap
    assert plan["upper_bound_usage"]["teacher_sample_tokens"] == 0
    assert not request.output_dir.exists()


def test_thinking_request_rejects_negative_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ThinkingEvalRequest("M0", "Qwen/Qwen3.5-4B", "medical_proxy", tmp_path, limit=-1)


@pytest.mark.parametrize("concurrency", [0, 33])
def test_thinking_request_rejects_unsupported_concurrency(
    tmp_path: Path, concurrency: int
) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        ThinkingEvalRequest(
            "M0",
            "Qwen/Qwen3.5-4B",
            "medical_proxy",
            tmp_path,
            concurrency=concurrency,
        )


def test_thinking_evaluation_caches_completed_rows_before_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    write_jsonl(
        tmp_path / "data" / "processed" / "eval_medical_full.jsonl",
        [_row(0), _row(1)],
    )
    monkeypatch.setattr(thinking_module, "_load_student_tokenizer", lambda config: _FakeTokenizer())

    class _Sampler:
        calls = 0

        async def sample_async(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            call = self.calls
            await asyncio.sleep(0.01 if call == 1 else 0.05)
            if call == 2:
                raise RuntimeError("synthetic later failure")
            return SimpleNamespace(
                sequences=[SimpleNamespace(text="A", tokens=[1])],
                input_tokens=10,
                output_tokens=1,
            )

    class _Service:
        async def create_sampling_client_async(self, **kwargs: object) -> _Sampler:
            return _Sampler()

    fake_trio = SimpleNamespace(
        configure=lambda **kwargs: None,
        ServiceClient=_Service,
        ModelInput=SimpleNamespace(from_ints=lambda ids: ids),
        SamplingParams=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "pytrio", fake_trio)
    output = tmp_path / "eval-progress"
    request = ThinkingEvalRequest(
        "M0",
        str(config.get("models", "student")),
        "medical_full",
        output,
        confirm_paid=True,
    )

    with pytest.raises(RuntimeError, match="synthetic later failure"):
        asyncio.run(thinking_module._evaluate_thinking_async(config, request))

    cached = read_jsonl(output / "predictions.jsonl")
    assert len(cached) == 1
    assert cached[0]["id"] == "medical-dev-0"
