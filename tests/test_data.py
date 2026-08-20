from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import medical_opd.config as config_module
import medical_opd.data as data_module
from medical_opd.config import ExperimentConfig, load_config
from medical_opd.data import (
    NearDuplicateIndex,
    _dataset_revisions,
    _deduplicate_rows,
    _general_rows,
    _stratified_sample,
    prepare_data,
)
from medical_opd.io_utils import fingerprint, normalize_text, read_jsonl, sha256_file


def test_near_duplicate_index_detects_contained_eval_question() -> None:
    heldout = "患者持续发热咳嗽三天，最可能的诊断是什么"
    index = NearDuplicateIndex([heldout], threshold=0.85)

    assert index.match("患者持续发热咳嗽三天，最可能的诊断是什么？") is not None
    assert index.match("如何配置 Python 虚拟环境") is None


def test_medical_training_filter_removes_exact_near_and_intra_duplicates() -> None:
    heldout = "患者持续发热咳嗽三天最可能的诊断是什么"
    rows = [
        {"id": "exact", "question": heldout},
        {"id": "near", "question": f"请根据病史判断：{heldout}"},
        {"id": "keep", "question": "胰岛素的主要生理作用是什么"},
        {"id": "duplicate", "question": "胰岛素的主要生理作用是什么"},
    ]

    quarantine: list[dict[str, Any]] = []
    kept, counters = _deduplicate_rows(
        rows,
        text_key="question",
        heldout_texts=[heldout],
        heldout_ids=["medqa-dev-42"],
        threshold=0.85,
        quarantine=quarantine,
    )

    assert [row["id"] for row in kept] == ["keep"]
    assert counters == {
        "input": 4,
        "intra_duplicate": 1,
        "heldout_exact": 1,
        "heldout_near": 1,
        "kept": 1,
    }
    assert {row["reason"] for row in quarantine} == {
        "heldout_exact",
        "heldout_near",
        "intra_duplicate",
    }
    exact = next(row for row in quarantine if row["reason"] == "heldout_exact")
    near = next(row for row in quarantine if row["reason"] == "heldout_near")
    duplicate = next(row for row in quarantine if row["reason"] == "intra_duplicate")
    assert exact["matched_eval_id"] == "medqa-dev-42"
    assert exact["similarity"] == 1.0
    assert near["matched_eval_id"] == "medqa-dev-42"
    assert 0.85 <= near["similarity"] <= 1.0
    assert duplicate["matched_source_row_id"] == "keep"
    assert all("question" not in row and "text" not in row for row in quarantine)


def test_quarantine_heldout_ids_must_align_with_eval_questions() -> None:
    with pytest.raises(ValueError, match="heldout_ids must align"):
        _deduplicate_rows(
            [{"id": "row", "question": "medical question"}],
            text_key="question",
            heldout_texts=["heldout question"],
            heldout_ids=["eval-1", "eval-2"],
            threshold=0.85,
            quarantine=[],
        )


def test_general_replay_normalization_never_labels_rows_as_ceval() -> None:
    rows = _general_rows(
        [
            {"instruction": "解释快速排序", "input": "", "output": "分治排序。"},
            {"instruction": "求和", "input": "1 和 2", "output": "3"},
        ]
    )

    assert len(rows) == 2
    assert {row["source_dataset"] for row in rows} == {"shibing624/alpaca-zh"}
    assert {row["source_split"] for row in rows} == {"train"}
    assert all("ceval" not in row["source_dataset"].casefold() for row in rows)


def test_dataset_manifest_revisions_come_from_pinned_config_values() -> None:
    config = load_config()

    assert _dataset_revisions(config) == {
        "medical": config.get("data", "medical_revision"),
        "medqa": config.get("data", "medqa_revision"),
        "general_replay": config.get("data", "general_replay_revision"),
        "ceval": config.get("data", "ceval_labeled_revision"),
    }


def test_stratified_proxy_is_deterministic_across_subject_and_answer_buckets() -> None:
    rows = [
        {
            "id": f"{subject}-{answer}-{index}",
            "subject": subject,
            "answer_idx": answer,
        }
        for subject in ("math", "logic")
        for answer in ("A", "B")
        for index in range(4)
    ]

    first = _stratified_sample(copy.deepcopy(rows), size=8, seed=123)
    second = _stratified_sample(copy.deepcopy(rows), size=8, seed=123)

    assert first == second
    assert len(first) == 8
    bucket_counts = {
        (subject, answer): sum(
            row["subject"] == subject and row["answer_idx"] == answer for row in first
        )
        for subject in ("math", "logic")
        for answer in ("A", "B")
    }
    assert set(bucket_counts.values()) == {2}


def _training_row(prefix: str, index: int, *, completion: str | None = None) -> dict[str, Any]:
    question = f"{prefix} question {index} with enough unique text"
    return {
        "id": f"{prefix}-{index}",
        "question": question,
        "completion": completion or f"{prefix} completion {index}",
        "source_index": index,
        "source_split": "train",
        "source_dataset": "medical-source" if prefix == "medical" else "general-source",
    }


def _eval_row(
    prefix: str,
    index: int,
    *,
    answer_text: str = "unrelated answer",
    source_split: str,
) -> dict[str, Any]:
    return {
        "id": f"eval-{prefix}-{source_split}-{index}",
        "question": f"{prefix} heldout question {index} with distinct content",
        "options": {"A": answer_text, "B": "choice b", "C": "choice c", "D": "choice d"},
        "answer_idx": "A",
        "source_dataset": "medqa" if prefix == "medical" else "ceval",
        "source_split": source_split,
        "subject": prefix,
        "source_index": index,
    }


def _small_config() -> ExperimentConfig:
    loaded = load_config()
    raw = copy.deepcopy(loaded.raw)
    raw["data"]["max_optimizer_steps"] = 2
    raw["data"]["proxy_medical_size"] = 2
    raw["data"]["proxy_general_size"] = 2
    return ExperimentConfig(loaded.path, raw)


def _stub_remote_and_tokenizer(monkeypatch: pytest.MonkeyPatch, loaded: dict[str, Any]) -> None:
    monkeypatch.setattr(
        data_module,
        "_load_all",
        lambda config, dataset_cache, hub_cache: copy.deepcopy(loaded),
    )
    monkeypatch.setattr(
        data_module,
        "_dataset_revisions",
        lambda config: {"medical": "m1", "medqa": "q1", "general_replay": "g1", "ceval": "c1"},
    )
    monkeypatch.setattr(
        data_module,
        "_tokenizer_audit",
        lambda config, output, hub_cache: (
            object(),
            {"vocab_equal": True, "special_token_ids_equal": True},
        ),
    )
    monkeypatch.setattr(
        data_module,
        "_length_audit",
        lambda config, tokenizer, medical, general: {
            "sequence_tokens": {"p90": 128},
            "derived_max_sequence_tokens": 128,
        },
    )


def _fixture_rows(*, leaked_answer: str = "unrelated answer") -> dict[str, list[dict[str, Any]]]:
    medical = [_training_row("medical", index) for index in range(8)]
    medical.append(_training_row("medical", 99, completion=leaked_answer))
    return {
        "medical_train": medical,
        "general_replay": [_training_row("general", index) for index in range(2)],
        "medical_proxy_pool": [
            _eval_row("medical", 0, source_split="dev"),
            _eval_row("medical", 1, source_split="dev"),
            _eval_row("medical", 2, source_split="dev"),
        ],
        "medical_eval_full": [
            _eval_row("medical", 10, answer_text=leaked_answer, source_split="test"),
            _eval_row("medical", 11, source_split="test"),
            _eval_row("medical", 12, source_split="test"),
        ],
        "general_proxy_pool": [
            _eval_row("general", index, source_split="val") for index in range(3)
        ],
        "general_eval_full": [
            _eval_row("general", index + 10, source_split="test") for index in range(3)
        ],
    }


def test_prepare_data_freezes_deterministic_proxy_files_and_manifest_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config()
    loaded = _fixture_rows()
    _stub_remote_and_tokenizer(monkeypatch, loaded)
    roots = [tmp_path / "first", tmp_path / "second"]
    hashes: list[dict[str, str]] = []

    for root in roots:
        monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
        manifest = prepare_data(config, shared_cache=tmp_path / "cache")
        assert manifest["status"] == "frozen"
        assert manifest["frozen_counts"]["eval_medical_proxy"] == 2
        assert manifest["frozen_counts"]["eval_general_proxy"] == 2
        proxy_paths = {
            name: root / manifest["files"][name]["path"]
            for name in ("eval_medical_proxy", "eval_general_proxy")
        }
        assert all(path.exists() for path in proxy_paths.values())
        assert all(
            sha256_file(proxy_paths[name]) == manifest["files"][name]["sha256"]
            for name in proxy_paths
        )
        medical_proxy = read_jsonl(proxy_paths["eval_medical_proxy"])
        general_proxy = read_jsonl(proxy_paths["eval_general_proxy"])
        medical_full = read_jsonl(root / manifest["files"]["eval_medical_full"]["path"])
        general_full = read_jsonl(root / manifest["files"]["eval_general_full"]["path"])
        assert {row["source_split"] for row in medical_proxy} == {"dev"}
        assert {row["source_split"] for row in medical_full} == {"test"}
        assert {row["source_split"] for row in general_proxy} == {"val"}
        assert {row["source_split"] for row in general_full} == {"test"}
        assert {row["id"] for row in medical_proxy}.isdisjoint(
            row["id"] for row in medical_full
        )
        assert {row["id"] for row in general_proxy}.isdisjoint(
            row["id"] for row in general_full
        )
        hashes.append({name: sha256_file(path) for name, path in proxy_paths.items()})

    assert hashes[0] == hashes[1]


def test_prepare_data_rejects_underfilled_frozen_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config()
    config.raw["data"]["proxy_general_size"] = 4
    loaded = _fixture_rows()
    assert len(loaded["general_proxy_pool"]) == 3
    _stub_remote_and_tokenizer(monkeypatch, loaded)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="proxy"):
        prepare_data(config, shared_cache=tmp_path / "cache")


def test_prepare_data_filters_medical_train_against_dev_and_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config()
    loaded = _fixture_rows()
    for source_index, eval_key in enumerate(("medical_proxy_pool", "medical_eval_full"), 500):
        eval_question = str(loaded[eval_key][0]["question"])
        leaked = _training_row("medical", source_index)
        leaked["question"] = eval_question
        leaked["id"] = f"leaked-{eval_key}"
        loaded["medical_train"].append(leaked)
    _stub_remote_and_tokenizer(monkeypatch, loaded)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    manifest = prepare_data(config, shared_cache=tmp_path / "cache")

    frozen_train = read_jsonl(tmp_path / "data" / "processed" / "train_medical.jsonl")
    heldout_fingerprints = {
        fingerprint(str(row["question"]))
        for key in ("medical_proxy_pool", "medical_eval_full")
        for row in loaded[key]
    }
    assert manifest["filters"]["medical"]["heldout_exact"] >= 2
    assert heldout_fingerprints.isdisjoint(
        fingerprint(str(row["question"])) for row in frozen_train
    )


def test_rag_corpus_excludes_eval_answer_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaked_answer = "特异性极强的评测正确答案文本"
    config = _small_config()
    loaded = _fixture_rows(leaked_answer=leaked_answer)
    _stub_remote_and_tokenizer(monkeypatch, loaded)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    prepare_data(config, shared_cache=tmp_path / "cache")

    rag_rows = read_jsonl(tmp_path / "data" / "processed" / "rag_corpus.jsonl")
    normalized_answer = normalize_text(leaked_answer)
    assert all(normalized_answer not in normalize_text(str(row["text"])) for row in rag_rows)
