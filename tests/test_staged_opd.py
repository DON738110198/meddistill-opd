from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import medical_opd.config as config_module
from medical_opd.config import load_config
from medical_opd.io_utils import sha256_file
from medical_opd.medical_sft import load_medical_pipeline_config
from medical_opd.staged_opd import (
    StagedOPDRequest,
    _rollout_prompt,
    _tokenizer_contract_sha256,
    run_staged_opd,
    select_medical_teacher,
    staged_opd_contract,
)


class _ModelInput:
    @staticmethod
    def from_ints(values: list[int]) -> list[int]:
        return list(values)


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Trio:
    ModelInput = _ModelInput
    SamplingParams = _SamplingParams


class _Tokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, *_: object, **kwargs: Any) -> str:
        assert kwargs["enable_thinking"] is True
        return "prompt"

    def encode(self, text: str, **_: object) -> list[int]:
        assert text == "prompt"
        return [10, 11, 12]


class _Student:
    async def sample_async(self, **kwargs: Any) -> SimpleNamespace:
        assert kwargs["num_samples"] == 4
        return SimpleNamespace(
            input_tokens=3,
            output_tokens=8,
            sequences=[
                SimpleNamespace(tokens=[20 + index, 30 + index], logprobs=[-0.2, -0.3])
                for index in range(4)
            ],
        )


class _Teacher:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    async def sample_async(self, **kwargs: Any) -> SimpleNamespace:
        prompt = list(kwargs["prompt"])
        self.prompts.append(prompt)
        return SimpleNamespace(
            input_tokens=len(prompt),
            output_tokens=1,
            prompt_logprobs=[None, -0.1, -0.1, -0.4, -0.5],
        )


def test_group4_rollout_aligns_teacher_tokens_masks_and_usage() -> None:
    experiment = load_config()
    pipeline = load_medical_pipeline_config(experiment)
    section = dict(pipeline.section("medical_opd"))
    teacher = _Teacher()

    datums, reverse_kls, usage, lengths, removed = asyncio.run(
        _rollout_prompt(
            _Trio,
            _Student(),
            teacher,
            _Tokenizer(),
            {"question": "q"},
            system="medical",
            section=section,
            seed=7,
        )
    )

    assert len(datums) == len(teacher.prompts) == 4
    assert lengths == [2, 2, 2, 2]
    assert removed == 0
    assert len(reverse_kls) == 8
    assert all(datum.trainable_tokens == 2 for datum in datums)
    assert teacher.prompts == [
        [10, 11, 12, 20 + index, 30 + index] for index in range(4)
    ]
    assert usage.student_prefill_tokens == 3 + 4 * 5
    assert usage.student_sample_tokens == 8 + 4


def _write_eval(
    experiment: Any,
    path: Path,
    *,
    dataset: str,
    correct: int,
    model_path: str | None,
    format_valid: float = 1.0,
) -> None:
    path.mkdir(parents=True)
    data_file = (
        experiment.root
        / "data"
        / "processed"
        / ("eval_medical_proxy.jsonl" if dataset == "medical_proxy" else "eval_general_proxy.jsonl")
    )
    data_file.parent.mkdir(parents=True, exist_ok=True)
    if not data_file.exists():
        data_file.write_text('{"fixture": true}\n', encoding="utf-8")
    max_tokens = 1024 if dataset == "medical_proxy" else 8192
    summary = {
        "status": "completed",
        "dataset": dataset,
        "count": 100,
        "dataset_sha256": sha256_file(data_file),
        "protocol_id": "thinking-fixed-v1",
        "prompt_contract": (
            "Chinese choice prompt rendered once with the student tokenizer"
        ),
        "thinking": True,
        "max_tokens": max_tokens,
        "temperature": 0.01,
        "top_p": 0.9,
        "seed_contract": "experiment.seed + frozen row index",
        "limit": 0,
        "base_model": "Qwen/Qwen3.5-4B",
        "model_path": model_path,
        "accuracy": correct / 100,
        "correct": correct,
        "format_valid_rate": format_valid,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {
            "id": f"row-{index}",
            "gold": "A",
            "question_sha256": f"q-{index}",
            "prompt_token_sha256": f"p-{index}",
            "correct": index < correct,
        }
        for index in range(100)
    ]
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_teacher_gate_allows_medical_gain_and_records_general_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    experiment = load_config()
    pipeline = load_medical_pipeline_config(experiment)
    base_medical = tmp_path / "base-medical"
    base_general = tmp_path / "base-general"
    teacher_medical = tmp_path / "teacher-medical"
    teacher_general = tmp_path / "teacher-general"
    _write_eval(experiment, base_medical, dataset="medical_proxy", correct=74, model_path=None)
    _write_eval(experiment, base_general, dataset="general_proxy", correct=79, model_path=None)
    checkpoint = "trio://run/checkpoint"
    _write_eval(
        experiment,
        teacher_medical,
        dataset="medical_proxy",
        correct=80,
        model_path=checkpoint,
    )
    _write_eval(
        experiment,
        teacher_general,
        dataset="general_proxy",
        correct=69,
        model_path=checkpoint,
    )

    result = select_medical_teacher(
        experiment,
        pipeline,
        base_medical=base_medical,
        base_general=base_general,
        teacher_medical=teacher_medical,
        teacher_general=teacher_general,
        output_path=tmp_path / "gate.json",
    )

    assert result["status"] == "passed"
    assert result["medical"]["delta_percentage_points"] == pytest.approx(6.0)
    assert result["general_diagnostic"]["delta_percentage_points"] == pytest.approx(-10.0)
    assert result["teacher_model_path"] == checkpoint


def test_paid_staged_opd_guard_precedes_gate_or_network(tmp_path: Path) -> None:
    experiment = load_config()
    pipeline = load_medical_pipeline_config(experiment)
    request = StagedOPDRequest(
        stage="medical",
        target_steps=1,
        output_dir=tmp_path / "run",
        teacher_model_path="trio://run/checkpoint",
        teacher_gate=tmp_path / "missing-gate.json",
    )

    with pytest.raises(RuntimeError, match="--confirm-paid"):
        run_staged_opd(experiment, pipeline, request)


def test_medical_opd_accepts_registered_formal_target(tmp_path: Path) -> None:
    request = StagedOPDRequest(
        stage="medical",
        target_steps=300,
        output_dir=tmp_path / "run",
        teacher_model_path="trio://run/checkpoint",
        teacher_gate=tmp_path / "gate.json",
    )

    assert request.target_steps == 300


def test_medical_opd_rejects_unregistered_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target must be one of"):
        StagedOPDRequest(
            stage="medical",
            target_steps=26,
            output_dir=tmp_path / "run",
            teacher_model_path="trio://run/checkpoint",
            teacher_gate=tmp_path / "gate.json",
        )


def test_tokenizer_contract_ignores_preflight_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    experiment = load_config()
    path = experiment.root / "data" / "processed" / "tokenizer_compatibility.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"compatible": True, "checked_at": "2026-08-20T00:00:00Z"}),
        encoding="utf-8",
    )
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    expected = _tokenizer_contract_sha256(experiment)
    payload["checked_at"] = "2099-01-01T00:00:00Z"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        assert _tokenizer_contract_sha256(experiment) == expected
    finally:
        path.write_text(original, encoding="utf-8")


def test_staged_opd_contract_records_pytrio_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    experiment = load_config()
    pipeline = load_medical_pipeline_config(experiment)
    data_path = experiment.root / "data" / "pipeline" / "train_medical_clean.jsonl"
    data_path.parent.mkdir(parents=True)
    data_path.write_text('{"question": "fixture"}\n', encoding="utf-8")
    tokenizer_audit = experiment.root / "data" / "processed" / "tokenizer_compatibility.json"
    tokenizer_audit.parent.mkdir(parents=True)
    tokenizer_audit.write_text(json.dumps({"compatible": True}), encoding="utf-8")
    gate_path = tmp_path / "teacher-gate.json"
    gate = {
        "status": "passed",
        "decision": "allow_medical_opd",
        "teacher_model_path": "trio://run/checkpoint",
        "medical": {"delta_percentage_points": 8.0},
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    request = StagedOPDRequest(
        stage="medical",
        target_steps=300,
        output_dir=tmp_path / "run",
        teacher_model_path=gate["teacher_model_path"],
        teacher_gate=gate_path,
    )

    assert staged_opd_contract(experiment, pipeline, request)["pytrio_version"] == "0.2.8"
