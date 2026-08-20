from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    distribution,
    fingerprint,
    nearest_rank_percentile,
    normalize_text,
    read_jsonl,
    redact_mapping,
    safe_slug,
    sha256_file,
    stable_hash,
    write_jsonl,
)


def test_fingerprints_are_deterministic_and_normalization_aware() -> None:
    variants = ["ＡＢＣ 医疗？", "abc医疗", "ABC---医疗"]

    assert {normalize_text(value) for value in variants} == {"abc医疗"}
    assert len({fingerprint(value) for value in variants}) == 1
    assert fingerprint("different") != fingerprint(variants[0])


def test_stable_hash_ignores_mapping_key_order_but_not_list_order() -> None:
    assert stable_hash({"b": 2, "a": [1, 2]}) == stable_hash({"a": [1, 2], "b": 2})
    assert stable_hash({"a": [1, 2]}) != stable_hash({"a": [2, 1]})


@pytest.mark.parametrize(
    ("values", "percentile", "expected"),
    [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9, 9), ([10, 1], 0.5, 1), ([7], 0.9, 7)],
)
def test_nearest_rank_p90_and_boundaries(
    values: list[int], percentile: float, expected: int
) -> None:
    assert nearest_rank_percentile(values, percentile) == expected


def test_percentile_and_distribution_empty_contract() -> None:
    assert nearest_rank_percentile([], 0.9) == 0
    assert distribution([]) == {
        "count": 0,
        "min": 0,
        "p50": 0,
        "p90": 0,
        "p95": 0,
        "max": 0,
        "mean": 0.0,
    }


@pytest.mark.parametrize("percentile", [-0.01, 1.01])
def test_percentile_rejects_out_of_range_value(percentile: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        nearest_rank_percentile([1], percentile)


def test_jsonl_round_trip_and_file_sha_are_deterministic(tmp_path: Path) -> None:
    rows = [{"id": "b", "text": "第二题"}, {"id": "a", "text": "first"}]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    assert write_jsonl(first, rows) == 2
    assert write_jsonl(second, rows) == 2
    assert read_jsonl(first) == rows
    assert sha256_file(first) == sha256_file(second)
    assert not first.with_suffix(".jsonl.tmp").exists()


def test_append_jsonl_adds_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl(path, {"step": 1})
    append_jsonl(path, {"step": 2})

    assert read_jsonl(path) == [{"step": 1}, {"step": 2}]


def test_read_jsonl_rejects_non_object_record(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\n[1, 2]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r":2 is not a JSON object"):
        read_jsonl(path)


def test_atomic_json_is_sorted_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"z": 1, "a": "医疗"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "医疗", "z": 1}
    assert not path.with_suffix(".json.tmp").exists()


def test_recursive_redaction_does_not_expose_credentials() -> None:
    redacted = redact_mapping(
        {
            "api_key": "secret-1",
            "nested": [{"Authorization": "Bearer secret-2", "safe": "visible"}],
            "access_token": "secret-3",
        }
    )

    rendered = json.dumps(redacted)
    assert "secret-1" not in rendered
    assert "secret-2" not in rendered
    assert "secret-3" not in rendered
    assert redacted["nested"][0]["safe"] == "visible"


@pytest.mark.parametrize("value", ["", "---", "???"])
def test_safe_slug_rejects_empty_sanitized_name(value: str) -> None:
    with pytest.raises(ValueError, match="slug is empty"):
        safe_slug(value)
