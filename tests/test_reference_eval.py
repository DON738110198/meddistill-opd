from __future__ import annotations

from pathlib import Path

import pytest

from medical_opd.config import load_config
from medical_opd.reference_eval import (
    REFERENCE_SYSTEM,
    ReferenceEvalRequest,
    build_reference_prompt,
    parse_reference_choice,
    render_reference_ids,
    run_reference_evaluation,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> str:
        self.kwargs = kwargs
        return "\n".join(message["content"] for message in messages)

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return list(range(len(text.split())))


def _row() -> dict[str, object]:
    return {
        "id": "eval-1",
        "question": "1+1=?",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer_idx": "B",
    }


def test_reference_prompt_and_template_match_frozen_thinking_contract() -> None:
    tokenizer = _Tokenizer()

    token_ids = render_reference_ids(tokenizer, _row())

    assert token_ids
    assert tokenizer.kwargs["enable_thinking"] is True
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert REFERENCE_SYSTEM.startswith("你是中文单项选择题作答助手")
    prompt = build_reference_prompt(_row())
    assert prompt.startswith("以下是中国考试中的单项选择题")
    assert prompt.endswith("答案：")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reasoning B</think>\nC", "C"),
        ("最终答案：D", "D"),
        ("Answer is A.", "A"),
        ("no valid choice", None),
    ],
)
def test_reference_parser_matches_frozen_final_answer_rules(
    text: str, expected: str | None
) -> None:
    assert parse_reference_choice(text) == expected


def test_reference_paid_guard_fires_before_manifest_or_network(tmp_path: Path) -> None:
    config = load_config()
    request = ReferenceEvalRequest("medical_600", tmp_path / "out")

    with pytest.raises(ValueError, match="confirm-paid"):
        run_reference_evaluation(config, request)


def test_reference_checkpoint_must_be_sampler_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampler weights"):
        ReferenceEvalRequest(
            "medical_600",
            tmp_path / "out",
            model_path="trio://run/weights/optimizer-state",
        )

    request = ReferenceEvalRequest(
        "medical_600",
        tmp_path / "out",
        model_path="trio://run/sampler_weights/sar-step50",
    )
    assert request.model_path.endswith("sar-step50")
