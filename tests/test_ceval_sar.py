from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medical_opd.ceval_sar import (
    FORBIDDEN_TRAINING_KEYS,
    METHOD,
    CevalSARRequest,
    _batch_for_step,
    _build_presentations,
    _ceval_rollout_prompt,
    _tokenizer_contract_sha256,
    _validate_source,
    format_ceval_prompt,
    load_ceval_sar_config,
    run_ceval_sar,
)
from medical_opd.config import load_config
from medical_opd.medical_sft import load_medical_pipeline_config


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
    async def sample_async(self, **kwargs: Any) -> SimpleNamespace:
        prompt = list(kwargs["prompt"])
        return SimpleNamespace(
            input_tokens=len(prompt),
            output_tokens=1,
            prompt_logprobs=[None, -0.1, -0.1, -0.4, -0.5],
        )


def _source_state(path: Path, state_uri: str) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "method": "MED-OPD",
                "completed_steps": 25,
                "recoverable_step": 25,
                "uncheckpointed_steps": 0,
                "latest_optimizer_state": state_uri,
                "training_contract_sha256": "source-contract",
                "training_contract": {"student_base_model": "Qwen/Qwen3.5-4B"},
                "checkpoints": [
                    {
                        "step": 25,
                        "state": state_uri,
                        "sampler_weights": "trio://run/sampler_weights/step25",
                        "permanent": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_base_anchor_config_is_isolated_from_alpaca_sar() -> None:
    experiment = load_config()
    ceval = load_ceval_sar_config(experiment)
    historical = load_medical_pipeline_config(experiment)

    assert ceval.get("training", "method") == METHOD
    assert ceval.get("training", "screen_steps") == 50
    assert ceval.get("training", "group_size") == 4
    assert ceval.get("training", "learning_rate") == 5e-6
    assert ceval.get("protocol", "source_splits") == ["dev", "val"]
    assert historical.get("sar", "prompt_source") == "alpaca-zh-general-replay"
    assert historical.get("sar", "screen_steps") == 25


def test_prompt_and_schedule_strip_answers_and_are_deterministic() -> None:
    rows = [
        {
            "id": f"source-{index}",
            "question": f"Question {index}?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_idx": "A",
            "source_dataset": "ceval/ceval-exam",
            "source_split": "dev" if index == 0 else "val",
            "subject": "logic",
            "source_index": index,
        }
        for index in range(2)
    ]

    first = _build_presentations(rows, count=5, seed=7)
    second = _build_presentations(rows, count=5, seed=7)

    assert first == second
    assert len({row["id"] for row in first}) == 5
    assert {row["source_split"] for row in first} == {"dev", "val"}
    assert all(not (FORBIDDEN_TRAINING_KEYS & set(row)) for row in first)
    assert first[0]["question"].startswith("以下是中国考试中的单项选择题。")
    assert first[0]["question"].endswith("答案：")
    assert format_ceval_prompt(rows[0]).count("A. a") == 1


def test_tokenizer_contract_ignores_only_audit_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import medical_opd.config as config_module

    experiment = load_config()
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "data" / "processed" / "tokenizer_compatibility.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"checked_at": "first", "vocab_equal": True}), encoding="utf-8")
    first = _tokenizer_contract_sha256(experiment)
    path.write_text(json.dumps({"checked_at": "second", "vocab_equal": True}), encoding="utf-8")

    assert _tokenizer_contract_sha256(experiment) == first


def test_source_requires_permanent_completed_medical_opd(tmp_path: Path) -> None:
    state_uri = "trio://run/weights/medical-opd-step25"
    source_path = tmp_path / "state.json"
    _source_state(source_path, state_uri)
    request = CevalSARRequest(
        target_steps=1,
        output_dir=tmp_path / "run",
        initial_student_state=state_uri,
        initial_local_state=source_path,
    )

    source = _validate_source(request)
    assert source["training_contract_sha256"] == "source-contract"

    broken = json.loads(source_path.read_text(encoding="utf-8"))
    broken["checkpoints"][0]["permanent"] = False
    source_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RuntimeError, match="permanent, complete, fully checkpointed"):
        _validate_source(request)


def test_source_accepts_permanent_completed_medical_opd300(tmp_path: Path) -> None:
    state_uri = "trio://run/weights/medical-opd-step300"
    source_path = tmp_path / "state.json"
    _source_state(source_path, state_uri)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["completed_steps"] = 300
    source["recoverable_step"] = 300
    source["checkpoints"][0]["step"] = 300
    source_path.write_text(json.dumps(source), encoding="utf-8")
    request = CevalSARRequest(
        target_steps=1,
        output_dir=tmp_path / "run",
        initial_student_state=state_uri,
        initial_local_state=source_path,
    )

    assert _validate_source(request)["completed_steps"] == 300


def test_ceval_rollout_preserves_alignment_masks_and_role_usage() -> None:
    experiment = load_config()
    config = load_ceval_sar_config(experiment)

    datums, reverse_kls, usage, roles, lengths, removed = asyncio.run(
        _ceval_rollout_prompt(
            _Trio,
            _Student(),
            _Teacher(),
            _Tokenizer(),
            {"question": "formatted question"},
            system="system",
            section=config.section("training"),
            seed=7,
        )
    )

    assert len(datums) == 4
    assert all(datum.trainable_tokens == 2 for datum in datums)
    assert len(reverse_kls) == 8
    assert lengths == [2, 2, 2, 2]
    assert removed == 0
    assert roles.student_rollout_prefill_tokens == 3
    assert roles.student_rollout_sample_tokens == 8
    assert roles.base_teacher_scoring_prefill_tokens == 20
    assert roles.base_teacher_scoring_sample_tokens == 4
    assert usage.student_prefill_tokens == 23
    assert usage.student_sample_tokens == 12


def test_paid_guard_precedes_data_source_and_network(tmp_path: Path) -> None:
    experiment = load_config()
    config = load_ceval_sar_config(experiment)
    request = CevalSARRequest(
        target_steps=1,
        output_dir=tmp_path / "run",
        initial_student_state="trio://run/weights/missing",
        initial_local_state=tmp_path / "missing.json",
    )

    with pytest.raises(RuntimeError, match="--confirm-paid"):
        run_ceval_sar(experiment, config, request)


@pytest.mark.parametrize("target", [0, 2, 49, 51, 99, 301])
def test_request_rejects_non_stage_targets(tmp_path: Path, target: int) -> None:
    with pytest.raises(ValueError, match="target must be"):
        CevalSARRequest(
            target_steps=target,
            output_dir=tmp_path,
            initial_student_state="trio://run/weights/source",
            initial_local_state=tmp_path / "state.json",
        )


@pytest.mark.parametrize("target", [60, 100, 150, 200, 250, 300])
def test_request_accepts_authorized_continuation_targets(tmp_path: Path, target: int) -> None:
    request = CevalSARRequest(
        target_steps=target,
        output_dir=tmp_path,
        initial_student_state="trio://run/weights/source",
        initial_local_state=tmp_path / "state.json",
    )
    assert request.target_steps == target


def test_continuation_repeats_frozen_schedule_without_reshuffling() -> None:
    rows = [{"id": f"row-{index}"} for index in range(8)]

    assert [row["id"] for row in _batch_for_step(rows, 0, 4)] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
    assert [row["id"] for row in _batch_for_step(rows, 2, 4)] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
