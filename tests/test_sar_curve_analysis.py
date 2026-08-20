from __future__ import annotations

import pytest

from medical_opd.sar_curve_analysis import _training_interval


def test_training_interval_aggregates_rollout_and_usage_metrics() -> None:
    rows = [
        {
            "step": step,
            "completion_tokens": 8,
            "datums": 4,
            "reverse_kl_mean": 0.1 * step,
            "prompt_truncated_tokens": 0,
            "trainer_metrics": {"loss_mean": 0.2 * step},
            "usage": {
                "student_prefill_tokens": 10,
                "student_sample_tokens": 8,
                "student_train_tokens": 12,
            },
        }
        for step in (1, 2)
    ]

    result = _training_interval(rows, 1, 2)

    assert result["steps"] == 2
    assert result["completion_tokens_per_rollout"] == 2
    assert result["reverse_kl_mean"] == pytest.approx(0.15)
    assert result["usage"]["student_train_tokens"] == 24
