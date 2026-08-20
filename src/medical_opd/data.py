from __future__ import annotations

import json
import math
import random
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import (
    atomic_write_json,
    distribution,
    fingerprint,
    normalize_text,
    sha256_file,
    stable_hash,
    utc_now,
    write_jsonl,
)


def _first_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _char_ngrams(text: str, size: int = 5) -> set[str]:
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


class NearDuplicateIndex:
    def __init__(self, texts: Iterable[str], threshold: float) -> None:
        self.threshold = threshold
        self.grams: list[set[str]] = []
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for text in texts:
            grams = _char_ngrams(text)
            index = len(self.grams)
            self.grams.append(grams)
            for gram in grams:
                self.inverted[gram].add(index)

    def match(self, text: str) -> tuple[int, float] | None:
        grams = _char_ngrams(text)
        if not grams:
            return None
        intersections: dict[int, int] = defaultdict(int)
        for gram in grams:
            for index in self.inverted.get(gram, ()):
                intersections[index] += 1
        best: tuple[int, float] | None = None
        for index, intersection in intersections.items():
            union = len(grams) + len(self.grams[index]) - intersection
            jaccard = intersection / union if union else 1.0
            containment = intersection / min(len(grams), len(self.grams[index]))
            score = max(jaccard, containment)
            if score >= self.threshold and (best is None or score > best[1]):
                best = (index, score)
        return best


def _deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    text_key: str,
    heldout_texts: list[str],
    threshold: float,
    heldout_ids: list[str] | None = None,
    quarantine: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    heldout_ids = heldout_ids or [f"heldout-{index}" for index in range(len(heldout_texts))]
    if len(heldout_ids) != len(heldout_texts):
        raise ValueError("heldout_ids must align with heldout_texts")
    exact_index: dict[str, int] = {}
    for index, text in enumerate(heldout_texts):
        exact_index.setdefault(fingerprint(text), index)
    heldout_fingerprints = set(exact_index)
    near_index = NearDuplicateIndex(heldout_texts, threshold)
    seen: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    counters = {"input": len(rows), "intra_duplicate": 0, "heldout_exact": 0, "heldout_near": 0}
    for row in rows:
        text = str(row[text_key])
        value = fingerprint(text)
        if value in seen:
            counters["intra_duplicate"] += 1
            if quarantine is not None:
                quarantine.append(
                    {
                        "source_row_id": row.get("id"),
                        "source_index": row.get("source_index"),
                        "reason": "intra_duplicate",
                        "matched_source_row_id": seen[value],
                        "question_sha256": value,
                    }
                )
            continue
        seen[value] = str(row.get("id", ""))
        if value in heldout_fingerprints:
            counters["heldout_exact"] += 1
            if quarantine is not None:
                match_index = exact_index[value]
                quarantine.append(
                    {
                        "source_row_id": row.get("id"),
                        "source_index": row.get("source_index"),
                        "reason": "heldout_exact",
                        "matched_eval_id": heldout_ids[match_index],
                        "question_sha256": value,
                        "similarity": 1.0,
                    }
                )
            continue
        near_match = near_index.match(text)
        if near_match is not None:
            counters["heldout_near"] += 1
            if quarantine is not None:
                match_index, score = near_match
                quarantine.append(
                    {
                        "source_row_id": row.get("id"),
                        "source_index": row.get("source_index"),
                        "reason": "heldout_near",
                        "matched_eval_id": heldout_ids[match_index],
                        "question_sha256": value,
                        "similarity": round(score, 6),
                    }
                )
            continue
        kept.append(row)
    counters["kept"] = len(kept)
    return kept, counters


def _stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(rows):
        return list(rows)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category", row.get("subject", "")))
        answer = str(row.get("answer_idx", ""))
        buckets[(category, answer)].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    allocations = {
        key: int(size * len(values) / len(rows)) for key, values in buckets.items()
    }
    allocated = sum(allocations.values())
    ranked = sorted(
        buckets,
        key=lambda key: (
            size * len(buckets[key]) / len(rows) - allocations[key],
            stable_hash([seed, key]),
        ),
        reverse=True,
    )
    for key in ranked[: size - allocated]:
        allocations[key] += 1
    selected = [row for key, values in buckets.items() for row in values[: allocations[key]]]
    random.Random(seed + 1).shuffle(selected)
    return selected


def _medical_rows(dataset: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, raw in enumerate(dataset):
        row = dict(raw)
        question = _first_text(row, ("Question", "question", "instruction", "prompt"))
        reasoning = _first_text(row, ("Complex_CoT", "complex_cot", "cot", "reasoning"))
        response = _first_text(row, ("Response", "response", "answer", "output"))
        completion = "\n\n".join(part for part in (reasoning, response) if part)
        if not question or not completion:
            continue
        rows.append(
            {
                "id": f"medical-{fingerprint(question)[:20]}",
                "question": question,
                "completion": completion,
                "reasoning": reasoning,
                "response": response,
                "source_index": source_index,
                "source_split": "train",
                "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
            }
        )
    return rows


def _option_dict(row: dict[str, Any]) -> dict[str, str]:
    raw_options = row.get("options")
    if isinstance(raw_options, dict):
        options = {str(key).upper(): str(value).strip() for key, value in raw_options.items()}
    elif isinstance(raw_options, list):
        options = {
            chr(ord("A") + index): str(value).strip()
            for index, value in enumerate(raw_options)
        }
    else:
        options = {
            key: str(row.get(key, "")).strip()
            for key in ("A", "B", "C", "D")
            if str(row.get(key, "")).strip()
        }
    return {key: options[key] for key in ("A", "B", "C", "D") if key in options}


def _answer_index(row: dict[str, Any], options: dict[str, str]) -> str:
    for name in ("answer_idx", "answer_index", "answer", "Answer"):
        value = str(row.get(name, "")).strip().upper()
        if value in options:
            return value
    answer_text = str(row.get("answer", "")).strip()
    for key, text in options.items():
        if answer_text and text == answer_text:
            return key
    return ""


def _choice_row(
    row: dict[str, Any],
    *,
    source_dataset: str,
    source_split: str,
    subject: str,
    source_index: int,
) -> dict[str, Any] | None:
    question = _first_text(row, ("question", "Question"))
    options = _option_dict(row)
    answer_idx = _answer_index(row, options)
    if not question or len(options) != 4 or answer_idx not in options:
        return None
    identity = stable_hash([source_dataset, source_split, subject, source_index, question])[:20]
    category = _first_text(row, ("meta_info", "category", "subject")) or subject
    return {
        "id": f"eval-{identity}",
        "question": question,
        "options": options,
        "answer_idx": answer_idx,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "subject": subject,
        "category": category,
        "source_index": source_index,
    }


def _general_rows(dataset: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, raw in enumerate(dataset):
        row = dict(raw)
        instruction = _first_text(row, ("instruction", "prompt", "question"))
        context = _first_text(row, ("input", "context"))
        completion = _first_text(row, ("output", "response", "answer"))
        question = instruction if not context else f"{instruction}\n\nInput:\n{context}"
        if not question or not completion:
            continue
        rows.append(
            {
                "id": f"general-{fingerprint(question)[:20]}",
                "question": question,
                "completion": completion,
                "source_index": source_index,
                "source_split": "train",
                "source_dataset": "shibing624/alpaca-zh",
            }
        )
    return rows


def _load_medqa(
    config: ExperimentConfig, cache_dir: Path, split: str
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id=str(config.get("data", "medqa_repo")),
        filename="data_clean.zip",
        repo_type="dataset",
        cache_dir=str(cache_dir),
        revision=str(config.get("data", "medqa_revision")),
    )
    if split not in {"dev", "test"}:
        raise ValueError("MedQA evaluation split must be dev or test")
    suffix = f"data_clean/questions/Mainland/4_options/{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one MedQA member ending in {suffix}, found {matches}")
        with archive.open(matches[0]) as handle:
            for index, line in enumerate(handle):
                normalized = _choice_row(
                    json.loads(line.decode("utf-8")),
                    source_dataset=str(config.get("data", "medqa_repo")),
                    source_split=split,
                    subject="medical",
                    source_index=index,
                )
                if normalized is not None:
                    rows.append(normalized)
    return rows


def _dataset_revisions(config: ExperimentConfig) -> dict[str, str]:
    return {
        "medical": str(config.get("data", "medical_revision")),
        "medqa": str(config.get("data", "medqa_revision")),
        "general_replay": str(config.get("data", "general_replay_revision")),
        "ceval": str(config.get("data", "ceval_labeled_revision")),
    }


def _load_all(
    config: ExperimentConfig, dataset_cache: Path, hub_cache: Path
) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    medical_raw = load_dataset(
        str(config.get("data", "medical_repo")),
        str(config.get("data", "medical_config")),
        split="train",
        cache_dir=str(dataset_cache),
        revision=str(config.get("data", "medical_revision")),
    )
    general_raw = load_dataset(
        str(config.get("data", "general_replay_repo")),
        split="train",
        cache_dir=str(dataset_cache),
        revision=str(config.get("data", "general_replay_revision")),
    )
    ceval_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_key in ("ceval_proxy_split", "ceval_full_split"):
        ceval_split = str(config.get("data", split_key))
        ceval_rows: list[dict[str, Any]] = []
        for subject in config.get("data", "ceval_subjects"):
            dataset_rows = list(
                load_dataset(
                    str(config.get("data", "ceval_repo")),
                    str(subject),
                    split=ceval_split,
                    cache_dir=str(dataset_cache),
                    revision=str(config.get("data", "ceval_labeled_revision")),
                )
            )
            for index, raw in enumerate(dataset_rows):
                row = _choice_row(
                    dict(raw),
                    source_dataset=str(config.get("data", "ceval_repo")),
                    source_split=ceval_split,
                    subject=str(subject),
                    source_index=index,
                )
                if row is not None:
                    ceval_rows.append(row)
        ceval_by_split[split_key] = ceval_rows
    return {
        "medical_train": _medical_rows(medical_raw),
        "general_replay": _general_rows(general_raw),
        "medical_proxy_pool": _load_medqa(config, hub_cache, "dev"),
        "medical_eval_full": _load_medqa(config, hub_cache, "test"),
        "general_proxy_pool": ceval_by_split["ceval_proxy_split"],
        "general_eval_full": ceval_by_split["ceval_full_split"],
    }


def _prompt_ids(tokenizer: Any, question: str, system: str) -> list[int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _tokenizer_audit(
    config: ExperimentConfig, output_path: Path, hub_cache: Path
) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    student_name = str(config.get("models", "student"))
    teacher_name = str(config.get("models", "teacher"))
    student = AutoTokenizer.from_pretrained(student_name, cache_dir=str(hub_cache))
    teacher = AutoTokenizer.from_pretrained(teacher_name, cache_dir=str(hub_cache))
    student_vocab = student.get_vocab()
    teacher_vocab = teacher.get_vocab()
    audit = {
        "student": student_name,
        "teacher": teacher_name,
        "student_vocab_size": len(student_vocab),
        "teacher_vocab_size": len(teacher_vocab),
        "student_vocab_sha256": stable_hash(student_vocab),
        "teacher_vocab_sha256": stable_hash(teacher_vocab),
        "vocab_equal": student_vocab == teacher_vocab,
        "student_eos_token_id": student.eos_token_id,
        "teacher_eos_token_id": teacher.eos_token_id,
        "student_special_tokens": student.special_tokens_map,
        "teacher_special_tokens": teacher.special_tokens_map,
        "special_token_ids_equal": student.all_special_ids == teacher.all_special_ids,
        "student_chat_template_sha256": stable_hash(student.chat_template or ""),
        "teacher_chat_template_sha256": stable_hash(teacher.chat_template or ""),
        "prompt_contract": (
            "Render once with the student tokenizer; submit the same token IDs to both models."
        ),
        "checked_at": utc_now(),
    }
    atomic_write_json(output_path, audit)
    if not audit["vocab_equal"] or not audit["special_token_ids_equal"]:
        raise RuntimeError(
            "27B and 4B token ID spaces are incompatible; token-level OPD is blocked"
        )
    return student, audit


def _length_audit(
    config: ExperimentConfig,
    tokenizer: Any,
    medical_rows: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    eos = [] if tokenizer.eos_token_id is None else [int(tokenizer.eos_token_id)]
    for row, system in (
        *((row, str(config.get("training", "medical_system"))) for row in medical_rows),
        *((row, str(config.get("training", "general_system"))) for row in general_rows),
    ):
        prompt = _prompt_ids(tokenizer, str(row["question"]), system)
        completion = list(tokenizer.encode(str(row["completion"]), add_special_tokens=False)) + eos
        prompt_lengths.append(len(prompt))
        completion_lengths.append(len(completion))
        total_lengths.append(len(prompt) + len(completion))
    cap = int(config.get("training", "default_max_length"))
    floor = int(config.get("training", "minimum_max_length"))
    total_p90 = distribution(total_lengths)["p90"]
    completion_p90 = distribution(completion_lengths)["p90"]
    max_sequence = min(cap, max(floor, int(math.ceil(int(total_p90) / 16) * 16)))
    max_completion = min(cap, max(16, int(math.ceil(int(completion_p90) / 16) * 16)))
    return {
        "population": "frozen 300-step train order (medical plus general replay)",
        "prompt_tokens": distribution(prompt_lengths),
        "completion_tokens": distribution(completion_lengths),
        "sequence_tokens": distribution(total_lengths),
        "derived_max_sequence_tokens": max_sequence,
        "derived_max_completion_tokens": max_completion,
        "sequence_truncation_count": sum(value > max_sequence for value in total_lengths),
        "completion_truncation_count": sum(value > max_completion for value in completion_lengths),
    }


def prepare_data(config: ExperimentConfig, *, shared_cache: Path) -> dict[str, Any]:
    processed = config.root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    dataset_cache = shared_cache / "datasets"
    hub_cache = shared_cache / "hub"
    dataset_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    loaded = _load_all(config, dataset_cache, hub_cache)
    seed = int(config.get("experiment", "seed"))
    rng = random.Random(seed)
    for rows in loaded.values():
        rng.shuffle(rows)

    heldout_medical_rows = [
        row
        for key in ("medical_proxy_pool", "medical_eval_full")
        for row in loaded[key]
    ]
    heldout_general_rows = [
        row
        for key in ("general_proxy_pool", "general_eval_full")
        for row in loaded[key]
    ]
    heldout_medical = [str(row["question"]) for row in heldout_medical_rows]
    heldout_general = [str(row["question"]) for row in heldout_general_rows]
    threshold = float(config.get("data", "near_duplicate_threshold"))
    medical_quarantine: list[dict[str, Any]] = []
    general_quarantine: list[dict[str, Any]] = []
    medical, medical_filter = _deduplicate_rows(
        loaded["medical_train"],
        text_key="question",
        heldout_texts=heldout_medical,
        threshold=threshold,
        heldout_ids=[str(row["id"]) for row in heldout_medical_rows],
        quarantine=medical_quarantine,
    )
    general, general_filter = _deduplicate_rows(
        loaded["general_replay"],
        text_key="question",
        heldout_texts=heldout_general + heldout_medical,
        threshold=threshold,
        heldout_ids=[
            str(row["id"]) for row in heldout_general_rows + heldout_medical_rows
        ],
        quarantine=general_quarantine,
    )

    max_steps = int(config.get("data", "max_optimizer_steps"))
    medical_needed = max_steps * int(config.get("data", "medical_per_step"))
    general_needed = max_steps * int(config.get("data", "mixed_general_per_step"))
    if len(medical) < medical_needed or len(general) < general_needed:
        raise RuntimeError(
            f"insufficient post-filter data: medical={len(medical)}/{medical_needed}, "
            f"general={len(general)}/{general_needed}"
        )
    medical_order = medical[:medical_needed]
    general_order = general[:general_needed]
    medical_proxy_size = int(config.get("data", "proxy_medical_size"))
    general_proxy_size = int(config.get("data", "proxy_general_size"))
    if len(loaded["medical_proxy_pool"]) < medical_proxy_size:
        raise RuntimeError(
            "medical proxy pool is underfilled; verify that MedQA dev labels are available"
        )
    if len(loaded["general_proxy_pool"]) < general_proxy_size:
        raise RuntimeError(
            "general proxy pool is underfilled; verify that C-Eval val labels are available"
        )
    if not loaded["medical_eval_full"]:
        raise RuntimeError("MedQA test full evaluation is empty")
    if not loaded["general_eval_full"]:
        raise RuntimeError(
            "C-Eval test labels are unavailable or empty; full evaluation is blocked"
        )
    proxy_medical = _stratified_sample(
        loaded["medical_proxy_pool"], medical_proxy_size, seed + 101
    )
    proxy_general = _stratified_sample(
        loaded["general_proxy_pool"], general_proxy_size, seed + 202
    )

    outputs = {
        "train_medical": processed / "train_medical.jsonl",
        "train_general_replay": processed / "train_general_replay.jsonl",
        "eval_medical_proxy": processed / "eval_medical_proxy.jsonl",
        "eval_general_proxy": processed / "eval_general_proxy.jsonl",
        "eval_medical_full": processed / "eval_medical_full.jsonl",
        "eval_general_full": processed / "eval_general_full.jsonl",
        "rag_corpus": processed / "rag_corpus.jsonl",
        "quarantine_medical": processed / "quarantine_medical.jsonl",
        "quarantine_general": processed / "quarantine_general.jsonl",
    }
    write_jsonl(outputs["train_medical"], medical_order)
    write_jsonl(outputs["train_general_replay"], general_order)
    write_jsonl(outputs["eval_medical_proxy"], proxy_medical)
    write_jsonl(outputs["eval_general_proxy"], proxy_general)
    write_jsonl(outputs["eval_medical_full"], loaded["medical_eval_full"])
    write_jsonl(outputs["eval_general_full"], loaded["general_eval_full"])
    write_jsonl(outputs["quarantine_medical"], medical_quarantine)
    write_jsonl(outputs["quarantine_general"], general_quarantine)

    eval_index = NearDuplicateIndex(heldout_medical, threshold)
    protected_answers = sorted(
        {
            normalize_text(str(row["options"][row["answer_idx"]]))
            for key in ("medical_proxy_pool", "medical_eval_full")
            for row in loaded[key]
            if len(normalize_text(str(row["options"][row["answer_idx"]]))) >= 2
        },
        key=len,
        reverse=True,
    )
    rag_rows: list[dict[str, Any]] = []
    rag_rejections = {"question_near_match": 0, "answer_text_match": 0}
    for row in medical:
        completion = str(row["completion"])
        if eval_index.match(completion) is not None:
            rag_rejections["question_near_match"] += 1
            continue
        normalized_completion = normalize_text(completion)
        if any(value in normalized_completion for value in protected_answers):
            rag_rejections["answer_text_match"] += 1
            continue
        rag_rows.append(
            {
                "doc_id": f"rag-{row['id']}",
                "text": completion,
                "source_row_id": row["id"],
                "source_dataset": row["source_dataset"],
            }
        )
        if len(rag_rows) >= 5000:
            break
    minimum_rag_rows = min(100, medical_needed)
    if len(rag_rows) < minimum_rag_rows:
        raise RuntimeError(
            f"audited RAG corpus is underfilled: {len(rag_rows)}/{minimum_rag_rows}"
        )
    write_jsonl(outputs["rag_corpus"], rag_rows)

    tokenizer, tokenizer_audit = _tokenizer_audit(
        config, processed / "tokenizer_compatibility.json", hub_cache
    )
    lengths = _length_audit(config, tokenizer, medical_order, general_order)
    atomic_write_json(processed / "lengths.json", lengths)
    char_stats = {
        "medical_train_question_chars": distribution(
            [len(str(row["question"])) for row in medical]
        ),
        "medical_train_completion_chars": distribution(
            [len(str(row["completion"])) for row in medical]
        ),
        "general_replay_question_chars": distribution(
            [len(str(row["question"])) for row in general]
        ),
        "general_replay_completion_chars": distribution(
            [len(str(row["completion"])) for row in general]
        ),
    }
    manifest = {
        "status": "frozen",
        "created_at": utc_now(),
        "seed": seed,
        "source_revisions": _dataset_revisions(config),
        "source_counts": {name: len(rows) for name, rows in loaded.items()},
        "filters": {"medical": medical_filter, "general_replay": general_filter},
        "rag_filter": {
            "protected_answer_count": len(protected_answers),
            "rejections": rag_rejections,
            "kept": len(rag_rows),
            "minimum_required": minimum_rag_rows,
        },
        "frozen_counts": {
            "train_medical": len(medical_order),
            "train_general_replay": len(general_order),
            "eval_medical_proxy": len(proxy_medical),
            "eval_general_proxy": len(proxy_general),
            "eval_medical_full": len(loaded["medical_eval_full"]),
            "eval_general_full": len(loaded["general_eval_full"]),
            "rag_corpus": len(rag_rows),
        },
        "split_contract": {
            "medical_train": (
                "medical-o1 train after exact and 5-gram Jaccard/containment de-dup "
                "against all MedQA dev/test rows"
            ),
            "medical_proxy": (
                "MedQA Mainland 4-option dev, deterministic stratified 100; "
                "checkpoint selection only"
            ),
            "medical_eval": (
                "MedQA Mainland 4-option test, all official rows; final evaluation only"
            ),
            "general_replay": "Alpaca-ZH train; no C-Eval row is used for replay",
            "general_proxy": (
                "C-Eval val, deterministic stratified 100 across eight non-medical subjects; "
                "checkpoint selection only"
            ),
            "general_eval": (
                "C-Eval test, all labeled rows for the same subjects; final evaluation only"
            ),
            "rag": (
                "medical-o1 completions only; excludes eval rows, labels, question "
                "near-duplicates, and normalized correct-option text"
            ),
        },
        "character_length_stats": char_stats,
        "token_length_stats": lengths,
        "tokenizer_compatibility": tokenizer_audit,
        "files": {
            name: {"path": str(path.relative_to(config.root)), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
    }
    atomic_write_json(processed / "manifest.json", manifest)
    return manifest
