from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import medical_opd.config as config_module
import medical_opd.teacher_sft as teacher_sft_module
from medical_opd.config import ExperimentConfig, load_config
from medical_opd.teacher_sft import (
    TeacherSFTRequest,
    _teacher_completion_text,
    build_teacher_sft_datum,
    plan_teacher_sft,
    run_teacher_sft,
)


class _Tokenizer:
    eos_token_id = 99

    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] = {}

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        self.template_kwargs = kwargs
        return "PROMPT"

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return [1, 2, 3] if text == "PROMPT" else [4, 5]


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    loaded = load_config()
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    rows = [
        {
            "id": f"medical-{index}",
            "question": f"question {index}",
            "reasoning": "reasoning",
            "response": "answer",
        }
        for index in range(4)
    ]
    (processed / "train_medical.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (processed / "manifest.json").write_text(
        json.dumps({"filters": {"medical": {"kept": 8}}}),
        encoding="utf-8",
    )
    return ExperimentConfig(loaded.path, copy.deepcopy(loaded.raw))


def _prices() -> dict[str, object]:
    return {
        "version": "test",
        "items": [
            {
                "display_name": "Qwen/Qwen3.5-4B",
                "prices": {
                    "prefill": {"unit_price": 149},
                    "sample": {"unit_price": 454},
                    "train": {"unit_price": 454},
                },
            },
            {
                "display_name": "Qwen/Qwen3.6-27B",
                "prices": {
                    "prefill": {"unit_price": 841},
                    "sample": {"unit_price": 2529},
                    "train": {"unit_price": 2529},
                },
            },
        ],
    }


def test_teacher_completion_closes_template_thinking_without_duplicate_open() -> None:
    text = _teacher_completion_text({"reasoning": "reason", "response": "answer"})

    assert text == "reason\n</think>\n\nanswer"
    assert "<think>" not in text


def test_teacher_sft_datum_uses_thinking_prompt_and_trains_eos() -> None:
    tokenizer = _Tokenizer()

    datum, removed, truncated = build_teacher_sft_datum(
        tokenizer,
        {"question": "q", "reasoning": "r", "response": "a"},
        system="medical",
        max_sequence=32,
    )

    assert tokenizer.template_kwargs["enable_thinking"] is True
    assert removed == 0
    assert truncated is False
    assert datum.input_tokens == (1, 2, 3, 4, 5)
    assert datum.target_tokens[-1] == 99
    assert datum.weights == (0.0, 0.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("steps", [0, 2, 26])
def test_teacher_sft_request_only_accepts_preregistered_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="1, 10, or 25"):
        TeacherSFTRequest(target_steps=steps, output_dir=Path("run"))


def test_teacher_sft_plan_counts_exact_27b_training_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(teacher_sft_module, "_load_student_tokenizer", lambda _: _Tokenizer())
    monkeypatch.setattr(teacher_sft_module, "fetch_json", lambda _: _prices())

    plan = plan_teacher_sft(
        config,
        TeacherSFTRequest(target_steps=1, output_dir=tmp_path / "run"),
    )

    assert plan["examples"] == 4
    assert plan["usage"]["teacher_train_tokens"] == 20
    assert plan["usage"]["optimizer_steps"] == 1
    assert plan["estimated_cny"] == 0.000506
    assert plan["completion_truncation_rows"] == 0
    assert plan["training_contract"]["base_model"] == "Qwen/Qwen3.6-27B"
    assert plan["full_epoch_extrapolation"]["examples"] == 4
    assert plan["cleaned_corpus_extrapolation"]["examples"] == 8
    assert (tmp_path / "reports" / "generated" / "teacher_sft_plan_step001.json").exists()


def test_teacher_sft_paid_guard_blocks_before_preflight_or_async_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(load_config().path, copy.deepcopy(load_config().raw))

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("paid guard reached preflight or async runner")

    monkeypatch.setattr(teacher_sft_module, "require_ready_preflight", unexpected)
    monkeypatch.setattr(teacher_sft_module.asyncio, "run", unexpected)

    with pytest.raises(RuntimeError, match="requires --confirm-paid"):
        run_teacher_sft(
            config,
            TeacherSFTRequest(target_steps=1, output_dir=tmp_path / "run"),
        )


def test_blocked_preflight_never_enters_teacher_sft_async_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(load_config().path, copy.deepcopy(load_config().raw))
    monkeypatch.setattr(
        teacher_sft_module,
        "require_ready_preflight",
        lambda _: (_ for _ in ()).throw(RuntimeError("preflight blocked")),
    )
    monkeypatch.setattr(
        teacher_sft_module.asyncio,
        "run",
        lambda _: (_ for _ in ()).throw(AssertionError("entered paid async runner")),
    )

    with pytest.raises(RuntimeError, match="preflight blocked"):
        run_teacher_sft(
            config,
            TeacherSFTRequest(
                target_steps=1,
                output_dir=tmp_path / "run",
                confirm_paid=True,
            ),
        )
