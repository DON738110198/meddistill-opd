from __future__ import annotations

import pytest

from medical_opd.mechanism_analysis import analyze_t27_records


def _row(
    row_id: str,
    *,
    correct: bool,
    valid: bool = True,
    truncated: bool = False,
) -> dict[str, object]:
    return {
        "id": row_id,
        "gold": "A",
        "question_sha256": f"question-{row_id}",
        "prompt_token_sha256": f"prompt-{row_id}",
        "correct": correct,
        "format_valid": valid,
        "truncated": truncated,
    }


def test_mechanism_analysis_separates_endpoint_and_valid_only_diagnostic() -> None:
    raw = [_row("a", correct=True), _row("b", correct=True), _row("c", correct=False)]
    sft = [
        _row("a", correct=False, valid=False, truncated=True),
        _row("b", correct=True),
        _row("c", correct=True),
    ]
    steps = [
        {
            "completion_truncation_rows": 1,
            "prompt_truncated_tokens": 3,
            "trainable_mask_tokens": 7,
            "trainer_metrics": {"loss_mean": 2.0, "token_count": 7},
        }
    ]
    result = analyze_t27_records(raw, sft, steps)

    assert result["end_to_end"]["raw_correct"] == 2
    assert result["end_to_end"]["sft_correct"] == 2
    assert result["format_valid_only_diagnostic"]["count"] == 2
    assert result["format_valid_only_diagnostic"]["raw_correct"] == 1
    assert result["format_valid_only_diagnostic"]["sft_correct"] == 2
    assert result["termination_diagnostic"]["sft_invalid_and_truncated"] == 1
    assert result["training_diagnostic"]["mask_token_count_mismatches"] == 0


def test_mechanism_analysis_rejects_prompt_contract_drift() -> None:
    raw = [_row("a", correct=True)]
    sft = [_row("a", correct=False)]
    sft[0]["prompt_token_sha256"] = "different"
    with pytest.raises(RuntimeError, match="prompt_token_sha256"):
        analyze_t27_records(raw, sft, [])
