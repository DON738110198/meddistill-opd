from __future__ import annotations

from medical_opd.ceval_sar_analysis import _paired, _training_ranges


def _prediction(index: int, *, correct: bool, truncated: bool = False) -> dict[str, object]:
    return {
        "id": f"row-{index}",
        "gold": "A",
        "prediction": "A" if correct else "B",
        "correct": correct,
        "truncated": truncated,
        "thinking_closed": not truncated,
        "question_sha256": f"q-{index}",
        "prompt_token_sha256": f"p-{index}",
        "usage": {"student_sample_tokens": 10 + index},
    }


def test_paired_analysis_preserves_contract_and_bad_case_direction() -> None:
    left = [_prediction(0, correct=True), _prediction(1, correct=False)]
    right = [
        _prediction(0, correct=False, truncated=True),
        _prediction(1, correct=True),
    ]

    test, cases = _paired(left, right)

    assert test["left_correct_right_wrong"] == 1
    assert test["left_wrong_right_correct"] == 1
    assert test["right_minus_left_percentage_points"] == 0.0
    assert len(cases) == 2
    assert cases[0]["right_truncated"] is True


def test_training_ranges_report_rollout_growth() -> None:
    rows = []
    for step in range(1, 51):
        completion = 16 * (100 if step <= 10 else 200 if step <= 25 else 300)
        rows.append(
            {
                "step": step,
                "completion_tokens": completion,
                "datums": 16,
                "trainer_metrics": {"loss_mean": 0.1, "token_count": completion},
                "reverse_kl_mean": 0.2,
                "prompt_truncated_tokens": 0,
                "trainable_mask_tokens": completion,
                "zero_advantage_completion_tokens": 0,
            }
        )

    ranges = _training_ranges(rows)

    assert [item["completion_tokens_per_rollout"] for item in ranges] == [100, 200, 300]
