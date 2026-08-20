from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_opd.backend import (
    PRICES_URL,
    UsageLedger,
    estimate_cost,
    fetch_json,
    price_table,
    require_ready_preflight,
)
from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    fingerprint,
    normalize_text,
    read_jsonl,
    sha256_file,
    stable_hash,
    utc_now,
)

DATASETS = {
    "medical_proxy": "eval_medical_proxy.jsonl",
    "general_proxy": "eval_general_proxy.jsonl",
    "medical_full": "eval_medical_full.jsonl",
    "general_full": "eval_general_full.jsonl",
}


@dataclass(frozen=True)
class EvalRequest:
    model_label: str
    base_model: str
    dataset: str
    output_dir: Path
    model_path: str | None = None
    rag: bool = False
    confirm_paid: bool = False

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
        if self.rag and self.dataset != "medical_proxy":
            raise ValueError("M1 RAG is restricted to the 100-question medical proxy")
        if not self.model_label.strip() or not self.base_model.strip():
            raise ValueError("model_label and base_model are required")
        if self.dataset.endswith("full") and self.model_label not in {
            "M0",
            "best-simple",
            "M5@50",
            "M5-final",
        }:
            raise ValueError(
                "full evaluation is restricted to M0, best-simple, M5@50, and M5-final"
            )


def parse_choice(text: str) -> str | None:
    stripped = text.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    matches = re.findall(r"(?:答案|ANSWER|选项)\s*[:：]?\s*([ABCD])\b", stripped)
    if matches:
        return matches[-1]
    isolated = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", stripped)
    return isolated[-1] if len(set(isolated)) == 1 else None


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _lexemes(text: str) -> list[str]:
    normalized = normalize_text(text)
    latin = re.findall(r"[a-z0-9_]+", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
    bigrams = ["".join(chinese[index : index + 2]) for index in range(len(chinese) - 1)]
    return latin + chinese + bigrams


class BM25:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.documents = [Counter(_lexemes(str(row["text"]))) for row in rows]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(document.keys())
        count = len(self.documents)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = Counter(_lexemes(query))
        scored: list[tuple[float, int]] = []
        for index, document in enumerate(self.documents):
            score = 0.0
            length = self.lengths[index]
            for term, query_count in query_terms.items():
                frequency = document.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * 2.2
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / max(self.average_length, 1)
                )
                score += self.idf.get(term, 0.0) * numerator / denominator * query_count
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], str(self.rows[item[1]]["doc_id"])))
        return [self.rows[index] for _, index in scored[:top_k]]


def _prompt(config: ExperimentConfig, row: dict[str, Any], contexts: list[dict[str, Any]]) -> str:
    options = "\n".join(f"{key}. {row['options'][key]}" for key in ("A", "B", "C", "D"))
    context_text = ""
    if contexts:
        passages = "\n\n".join(
            f"[Document {index}]\n{document['text']}"
            for index, document in enumerate(contexts, start=1)
        )
        context_text = f"Reference documents:\n{passages}\n\n"
    return (
        f"{config.get('training', 'choice_system')}\n\n"
        f"{context_text}Question:\n{row['question']}\n{options}\n\nAnswer:"
    )


def _render_ids(tokenizer: Any, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Follow the requested answer format exactly."},
         {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _load_student_tokenizer(config: ExperimentConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.get("models", "student")),
        cache_dir=str(config.root.parent / ".cache" / "huggingface" / "hub"),
        local_files_only=True,
    )
    audit_path = config.root / "data" / "processed" / "tokenizer_compatibility.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if stable_hash(tokenizer.get_vocab()) != audit.get("student_vocab_sha256"):
            raise RuntimeError("local evaluation tokenizer differs from the frozen audit")
    return tokenizer


def _retriever(config: ExperimentConfig, enabled: bool) -> BM25 | None:
    if not enabled:
        return None
    corpus = read_jsonl(config.root / "data" / "processed" / "rag_corpus.jsonl")
    if len(corpus) < 100:
        raise RuntimeError("audited RAG corpus is too small; M1 fails closed")
    return BM25(corpus)


def plan_evaluation(config: ExperimentConfig, request: EvalRequest) -> dict[str, Any]:
    dataset_path = config.root / "data" / "processed" / DATASETS[request.dataset]
    rows = read_jsonl(dataset_path)
    if request.dataset.endswith("proxy") and len(rows) != 100:
        raise RuntimeError(f"frozen proxy must contain exactly 100 rows, found {len(rows)}")
    tokenizer = _load_student_tokenizer(config)
    retriever = _retriever(config, request.rag)
    prompt_lengths = []
    for row in rows:
        contexts = [] if retriever is None else retriever.search(
            str(row["question"]), int(config.get("evaluation", "rag_top_k"))
        )
        prompt_lengths.append(len(_render_ids(tokenizer, _prompt(config, row, contexts))))
    usage = UsageLedger()
    prefill = sum(prompt_lengths)
    sample = len(rows) * int(config.get("evaluation", "max_completion_tokens"))
    if request.base_model == str(config.get("models", "teacher")):
        usage.teacher_prefill_tokens = prefill
        usage.teacher_sample_tokens = sample
    else:
        usage.student_prefill_tokens = prefill
        usage.student_sample_tokens = sample
    prices_raw = fetch_json(PRICES_URL)
    tokenizer_audit = config.root / "data" / "processed" / "tokenizer_compatibility.json"
    rag_path = config.root / "data" / "processed" / "rag_corpus.jsonl"
    return {
        "mode": "remote_paid_evaluation",
        "model_label": request.model_label,
        "base_model": request.base_model,
        "model_path": request.model_path,
        "dataset": request.dataset,
        "dataset_count": len(rows),
        "dataset_sha256": sha256_file(dataset_path),
        "tokenizer_audit_sha256": sha256_file(tokenizer_audit),
        "rag_corpus_sha256": sha256_file(rag_path) if request.rag else None,
        "rag": request.rag,
        "prompt_token_total": prefill,
        "prompt_token_max": max(prompt_lengths, default=0),
        "max_output_tokens_per_row": int(
            config.get("evaluation", "max_completion_tokens")
        ),
        "upper_bound_usage": usage.to_dict(),
        "upper_bound_estimated_cny": estimate_cost(
            usage, price_table(prices_raw), config
        ),
        "price_version": prices_raw.get("version"),
        "process": "medical-opd evaluate",
        "output_dir": str(request.output_dir.resolve()),
        "success_criterion": (
            "all frozen rows have one cached prediction; parser and usage ledger complete"
        ),
    }


def _usage_value(response: Any, name: str, fallback: int) -> tuple[int, str]:
    value = getattr(response, name, None)
    if isinstance(value, int) and value >= 0:
        return value, "response"
    return fallback, "derived"


async def _evaluate_async(config: ExperimentConfig, request: EvalRequest) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    dataset_path = config.root / "data" / "processed" / DATASETS[request.dataset]
    rows = read_jsonl(dataset_path)
    if request.dataset.endswith("proxy") and len(rows) != 100:
        raise RuntimeError(f"frozen proxy must contain exactly 100 rows, found {len(rows)}")
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "predictions.jsonl"
    existing = read_jsonl(records_path) if records_path.exists() else []
    completed_ids = {str(record["id"]) for record in existing}
    usage = UsageLedger()
    for record in existing:
        usage.add(UsageLedger.from_dict(dict(record.get("usage", {}))))

    retriever = _retriever(config, request.rag)
    request_contract = {
        "model_label": request.model_label,
        "base_model": request.base_model,
        "model_path": request.model_path,
        "dataset": request.dataset,
        "dataset_sha256": sha256_file(dataset_path),
        "rag": request.rag,
        "config_sha256": sha256_file(config.path),
        "tokenizer_audit_sha256": sha256_file(
            config.root / "data" / "processed" / "tokenizer_compatibility.json"
        ),
        "rag_corpus_sha256": (
            sha256_file(config.root / "data" / "processed" / "rag_corpus.jsonl")
            if request.rag
            else None
        ),
    }
    request_path = output / "request.json"
    if request_path.exists():
        prior_contract = json.loads(request_path.read_text(encoding="utf-8"))
        if prior_contract != request_contract:
            raise RuntimeError("evaluation cache request does not match the current run")
    else:
        atomic_write_json(request_path, request_contract)
    tokenizer = _load_student_tokenizer(config)
    service = trio.ServiceClient()
    sampler = await service.create_sampling_client_async(
        base_model=request.base_model,
        model_path=request.model_path,
    )
    semaphore = asyncio.Semaphore(int(config.get("evaluation", "concurrency")))
    started = time.perf_counter()

    async def infer(index: int, row: dict[str, Any]) -> dict[str, Any]:
        contexts = [] if retriever is None else retriever.search(
            str(row["question"]), int(config.get("evaluation", "rag_top_k"))
        )
        prompt = _prompt(config, row, contexts)
        prompt_ids = _render_ids(tokenizer, prompt)
        async with semaphore:
            response = await sampler.sample_async(
                prompt=trio.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=trio.SamplingParams(
                    max_tokens=int(config.get("evaluation", "max_completion_tokens")),
                    temperature=float(config.get("evaluation", "temperature")),
                    top_p=float(config.get("evaluation", "top_p")),
                    seed=int(config.get("experiment", "seed")) + index,
                ),
                return_text=True,
            )
        if len(response.sequences) != 1:
            raise RuntimeError("evaluation did not return exactly one sequence")
        sequence = response.sequences[0]
        prediction = parse_choice(str(sequence.text or ""))
        input_tokens, input_source = _usage_value(response, "input_tokens", len(prompt_ids))
        output_tokens, output_source = _usage_value(
            response, "output_tokens", len(sequence.tokens)
        )
        sample_usage = UsageLedger()
        if request.base_model == str(config.get("models", "teacher")):
            sample_usage.teacher_prefill_tokens = input_tokens
            sample_usage.teacher_sample_tokens = output_tokens
        else:
            sample_usage.student_prefill_tokens = input_tokens
            sample_usage.student_sample_tokens = output_tokens
        return {
            "id": row["id"],
            "source_index": row.get("source_index"),
            "subject": row.get("subject"),
            "question_sha256": fingerprint(str(row["question"])),
            "prompt_token_sha256": stable_hash(prompt_ids),
            "gold": row["answer_idx"],
            "prediction": prediction,
            "correct": prediction == row["answer_idx"],
            "format_valid": prediction is not None,
            "retrieved_doc_ids": [document["doc_id"] for document in contexts],
            "usage": sample_usage.to_dict(),
            "usage_counter_source": {"input": input_source, "output": output_source},
        }

    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row["id"]) not in completed_ids
    ]
    for record in await asyncio.gather(*(infer(index, row) for index, row in pending)):
        append_jsonl(records_path, record)
        usage.add(UsageLedger.from_dict(dict(record["usage"])))

    records = read_jsonl(records_path)
    by_id = {str(record["id"]): record for record in records}
    records = [by_id[str(row["id"])] for row in rows]
    correct = sum(bool(record["correct"]) for record in records)
    valid = sum(bool(record["format_valid"]) for record in records)
    lower, upper = wilson_interval(correct, len(records))
    unique_records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for record in records:
        question_hash = str(record["question_sha256"])
        if question_hash not in seen_questions:
            unique_records.append(record)
            seen_questions.add(question_hash)
    unique_correct = sum(bool(record["correct"]) for record in unique_records)
    subject_scores: dict[str, dict[str, int | float]] = {}
    for subject in sorted({str(record.get("subject", "")) for record in records}):
        subset = [record for record in records if str(record.get("subject", "")) == subject]
        subject_scores[subject] = {
            "count": len(subset),
            "correct": sum(bool(record["correct"]) for record in subset),
            "accuracy": sum(bool(record["correct"]) for record in subset) / len(subset),
        }
    prices_raw = fetch_json(PRICES_URL)
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        "model_label": request.model_label,
        "base_model": request.base_model,
        "model_path": request.model_path,
        "dataset": request.dataset,
        "dataset_path": str(dataset_path.relative_to(config.root)),
        "dataset_sha256": sha256_file(dataset_path),
        "rag": request.rag,
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "wilson_95": [lower, upper],
        "format_valid_rate": valid / len(records),
        "normalized_unique_count": len(unique_records),
        "normalized_unique_accuracy": unique_correct / len(unique_records),
        "subject_scores": subject_scores,
        "subject_macro_accuracy": (
            sum(float(value["accuracy"]) for value in subject_scores.values())
            / max(len(subject_scores), 1)
        ),
        "usage": usage.to_dict(),
        "estimated_cny": estimate_cost(usage, price_table(prices_raw), config),
        "actual_billed_cny": None,
        "wall_seconds": time.perf_counter() - started,
        "prediction_cache_sha256": sha256_file(records_path),
        "scope": "proxy" if request.dataset.endswith("proxy") else "full",
        "prompt_contract": (
            "All models receive token IDs rendered once with the Qwen3.5-4B tokenizer."
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def run_evaluation(config: ExperimentConfig, request: EvalRequest) -> dict[str, Any]:
    if not request.confirm_paid:
        raise ValueError("refusing paid remote work without --confirm-paid")
    manifest = config.root / "data" / "processed" / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("frozen data manifest is required")
    require_ready_preflight(config)
    return asyncio.run(_evaluate_async(config, request))


def build_teacher_gate(
    config: ExperimentConfig,
    base_medical: Path,
    base_general: Path,
    teacher_medical: Path,
    teacher_general: Path,
) -> dict[str, Any]:
    summaries = {
        "base_medical": json.loads((base_medical / "summary.json").read_text(encoding="utf-8")),
        "base_general": json.loads((base_general / "summary.json").read_text(encoding="utf-8")),
        "teacher_medical": json.loads(
            (teacher_medical / "summary.json").read_text(encoding="utf-8")
        ),
        "teacher_general": json.loads(
            (teacher_general / "summary.json").read_text(encoding="utf-8")
        ),
    }
    for name, summary in summaries.items():
        expected = "medical_proxy" if "medical" in name else "general_proxy"
        if summary.get("status") != "completed" or summary.get("dataset") != expected:
            raise RuntimeError(f"invalid teacher-gate input: {name}")
        expected_dataset_path = (
            config.root / "data" / "processed" / DATASETS[expected]
        )
        expected_max_tokens = int(
            config.get(
                "evaluation",
                "thinking_medical_max_tokens"
                if expected == "medical_proxy"
                else "thinking_general_max_tokens",
            )
        )
        expected_tokenizer_audit = sha256_file(
            config.root / "data" / "processed" / "tokenizer_compatibility.json"
        )
        if (
            int(summary.get("count", -1)) != 100
            or summary.get("dataset_sha256") != sha256_file(expected_dataset_path)
            or summary.get("config_sha256") != sha256_file(config.path)
            or summary.get("tokenizer_audit_sha256") != expected_tokenizer_audit
            or summary.get("protocol_id")
            != str(config.get("evaluation", "thinking_protocol_id"))
            or summary.get("prompt_contract")
            != "Chinese choice prompt rendered once with the student tokenizer"
            or summary.get("thinking") is not True
            or int(summary.get("max_tokens", -1)) != expected_max_tokens
            or float(summary.get("temperature", -1))
            != float(config.get("evaluation", "thinking_temperature"))
            or float(summary.get("top_p", -1))
            != float(config.get("evaluation", "thinking_top_p"))
            or summary.get("seed_contract") != "experiment.seed + frozen row index"
            or int(summary.get("limit", -1)) != 0
            or summary.get("scope") != "proxy"
        ):
            raise RuntimeError(f"teacher-gate frozen evaluation contract failed: {name}")
        expected_model = str(
            config.get("models", "teacher" if name.startswith("teacher") else "student")
        )
        if (
            summary.get("base_model") != expected_model
            or summary.get("model_path") is not None
            or bool(summary.get("rag"))
        ):
            raise RuntimeError(f"teacher-gate input is not the frozen raw model: {name}")
        accuracy = float(summary.get("accuracy", -1))
        if not 0 <= accuracy <= 1:
            raise RuntimeError(f"teacher-gate accuracy is invalid: {name}")
    medical_margin = 100 * (
        float(summaries["teacher_medical"]["accuracy"])
        - float(summaries["base_medical"]["accuracy"])
    )
    base_mean = (
        float(summaries["base_medical"]["accuracy"])
        + float(summaries["base_general"]["accuracy"])
    ) / 2
    teacher_mean = (
        float(summaries["teacher_medical"]["accuracy"])
        + float(summaries["teacher_general"]["accuracy"])
    ) / 2
    overall_margin = 100 * (teacher_mean - base_mean)
    passed = (
        medical_margin >= float(config.get("gates", "teacher_medical_margin_pp"))
        and overall_margin >= float(config.get("gates", "teacher_overall_margin_pp"))
    )
    report = {
        "status": "passed" if passed else "failed",
        "created_at": utc_now(),
        "medical_margin_pp": medical_margin,
        "overall_mean_margin_pp": overall_margin,
        "thresholds": {
            "medical_margin_pp": config.get("gates", "teacher_medical_margin_pp"),
            "overall_mean_margin_pp": config.get("gates", "teacher_overall_margin_pp"),
        },
        "evaluation_contract": {
            "protocol_id": config.get("evaluation", "thinking_protocol_id"),
            "thinking": True,
            "medical_max_tokens": config.get(
                "evaluation", "thinking_medical_max_tokens"
            ),
            "general_max_tokens": config.get(
                "evaluation", "thinking_general_max_tokens"
            ),
            "seed_contract": "experiment.seed + frozen row index",
        },
        "inputs": {
            name: {
                "path": str(path),
                "summary_sha256": sha256_file(path / "summary.json"),
                "accuracy": summaries[name]["accuracy"],
                "format_valid_rate": summaries[name].get("format_valid_rate"),
                "thinking_closed_rate": summaries[name].get("thinking_closed_rate"),
                "truncation_rate": summaries[name].get("truncation_rate"),
                "estimated_cny": summaries[name].get("estimated_cny"),
            }
            for name, path in {
                "base_medical": base_medical,
                "base_general": base_general,
                "teacher_medical": teacher_medical,
                "teacher_general": teacher_general,
            }.items()
        },
        "decision": (
            "M4/M5 formal runs may proceed" if passed else "Stop M4/M5 formal runs"
        ),
    }
    atomic_write_json(config.root / "reports" / "generated" / "teacher_gate.json", report)
    return report
