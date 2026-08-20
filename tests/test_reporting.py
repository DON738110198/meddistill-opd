from __future__ import annotations

import pytest

from medical_opd.reporting import exact_mcnemar


def _rows(values: list[bool]) -> list[dict[str, object]]:
    return [{"id": f"row-{index}", "correct": value} for index, value in enumerate(values)]


def test_exact_mcnemar_counts_discordant_pairs() -> None:
    result = exact_mcnemar(
        _rows([True, True, False, False]),
        _rows([True, False, True, False]),
    )

    assert result == {
        "count": 4,
        "left_correct_right_wrong": 1,
        "left_wrong_right_correct": 1,
        "discordant": 2,
        "exact_two_sided_p": 1.0,
    }


def test_exact_mcnemar_rejects_unpaired_ids() -> None:
    with pytest.raises(RuntimeError, match="row IDs"):
        exact_mcnemar(_rows([True]), [{"id": "other", "correct": True}])
