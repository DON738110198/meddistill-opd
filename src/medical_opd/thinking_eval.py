from __future__ import annotations

import asyncio
import json
import time
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
from medical_opd.evaluation import DATASETS, _load_student_tokenizer, wilson_interval
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    fingerprint,
    read_jsonl,
    sha256_file,
    stable_hash,
    utc_now,
)
from medical_opd.reference_eval import parse_reference_choice, render_reference_ids


@dataclass(frozen=True)
class ThinkingEvalRequest:
    model_label: str
    base_model: str
    dataset: str
    output_dir: Path
    model_path: str | None = None
    limit: int = 0
    confirm_paid: bool = False
    concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
        if not self.model_label.strip() or not self.base_model.strip():
            raise ValueError("model_label and base_model are required")
        if self.limit < 0:
            raise ValueError("limit must be nonnegative")
        if self.concurrency is not None and not 1 <= self.concurrency <= 32:
            raise ValueError("concurrency must be between 1 and 32")


def _max_tokens(config: ExperimentConfig, dataset: str) -> int:
    domain = "medical" if dataset.startswith("medical_") else "general"
    return int(config.get("evaluation", f"thinking_{domain}_max_tokens"))


def _rows(
    config: ExperimentConfig,
    request: ThinkingEvalRequest,
) -> tuple[Path, list[dict[str, Any]]]:
    path = config.root / "data" / "processed" / DATASETS[request.dataset]
    rows = read_jsonl(path)
    if request.dataset.endswith("proxy") and len(rows) != 100:
        raise RuntimeError(f"frozen proxy must contain exactly 100 rows, found {len(rows)}")
    if request.limit:
        rows = rows[: request.limit]
    if not rows:
        raise RuntimeError("thinking evaluation dataset is empty")
    return path, rows


def _contract(
    config: ExperimentConfig,
    request: ThinkingEvalRequest,
    dataset_path: Path,
) -> dict[str, Any]:
    return {
        "protocol_id": str(config.get("evaluation", "thinking_protocol_id")),
        "model_label": request.model_label,
        "base_model": request.base_model,
        "model_path": request.model_path,
        "dataset": request.dataset,
        "limit": request.limit,
        "dataset_sha256": sha256_file(dataset_path),
        "config_sha256": sha256_file(config.path),
        "tokenizer_audit_sha256": sha256_file(
            config.root / "data" / "processed" / "tokenizer_compatibility.json"
        ),
        "prompt_contract": (
            "Chinese choice prompt rendered once with the student tokenizer"
        ),
        "thinking": True,
        "max_tokens": _max_tokens(config, request.dataset),
        "temperature": float(config.get("evaluation", "thinking_temperature")),
        "top_p": float(config.get("evaluation", "thinking_top_p")),
        "seed_contract": "experiment.seed + frozen row index",
    }


def _usage_value(response: Any, name: str, fallback: int) -> int:
    value = getattr(response, name, None)
    return value if isinstance(value, int) and value >= 0 else fallback


def plan_thinking_evaluation(
    config: ExperimentConfig, request: ThinkingEvalRequest
) -> dict[str, Any]:
    dataset_path, rows = _rows(config, request)
    tokenizer = _load_student_tokenizer(config)
    prompt_lengths = [len(render_reference_ids(tokenizer, row)) for row in rows]
    max_tokens = _max_tokens(config, request.dataset)
    usage = UsageLedger()
    role = (
        "teacher"
        if request.base_model == str(config.get("models", "teacher"))
        else "student"
    )
    setattr(usage, f"{role}_prefill_tokens", sum(prompt_lengths))
    setattr(usage, f"{role}_sample_tokens", len(rows) * max_tokens)
    prices = fetch_json(PRICES_URL)
    return {
        "mode": "remote_paid_thinking_evaluation",
        **_contract(config, request, dataset_path),
        "dataset_count": len(rows),
        "prompt_token_total": sum(prompt_lengths),
        "prompt_token_max": max(prompt_lengths),
        "upper_bound_usage": usage.to_dict(),
        "upper_bound_estimated_cny": estimate_cost(usage, price_table(prices), config),
        "price_version": prices.get("version"),
        "output_dir": str(request.output_dir.resolve()),
        "execution_concurrency": (
            request.concurrency
            if request.concurrency is not None
            else int(config.get("evaluation", "concurrency"))
        ),
        "success_criterion": (
            "one cached prediction per frozen row; finite usage; format, closure, and truncation "
            "rates reported"
        ),
    }


async def _evaluate_thinking_async(
    config: ExperimentConfig, request: ThinkingEvalRequest
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    dataset_path, rows = _rows(config, request)
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "predictions.jsonl"
    request_path = output / "request.json"
    contract = _contract(config, request, dataset_path)
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("thinking evaluation cache contract mismatch")
    else:
        atomic_write_json(request_path, contract)

    tokenizer = _load_student_tokenizer(config)
    service = trio.ServiceClient()
    sampler = await service.create_sampling_client_async(
        base_model=request.base_model,
        model_path=request.model_path,
    )
    existing = read_jsonl(records_path) if records_path.exists() else []
    completed_ids = {str(record["id"]) for record in existing}
    execution_concurrency = (
        request.concurrency
        if request.concurrency is not None
        else int(config.get("evaluation", "concurrency"))
    )
    semaphore = asyncio.Semaphore(execution_concurrency)
    started = time.perf_counter()
    max_tokens = _max_tokens(config, request.dataset)
    base_seed = int(config.get("experiment", "seed"))

    async def infer(index: int, row: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = render_reference_ids(tokenizer, row)
        async with semaphore:
            response = await sampler.sample_async(
                prompt=trio.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=trio.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=float(config.get("evaluation", "thinking_temperature")),
                    top_p=float(config.get("evaluation", "thinking_top_p")),
                    seed=base_seed + index,
                ),
                return_text=True,
            )
        if len(response.sequences) != 1:
            raise RuntimeError("thinking evaluation did not return exactly one sequence")
        sequence = response.sequences[0]
        text = str(sequence.text or "")
        prediction = parse_reference_choice(text)
        input_tokens = _usage_value(response, "input_tokens", len(prompt_ids))
        output_tokens = _usage_value(response, "output_tokens", len(sequence.tokens))
        usage = UsageLedger()
        role = (
            "teacher"
            if request.base_model == str(config.get("models", "teacher"))
            else "student"
        )
        setattr(usage, f"{role}_prefill_tokens", input_tokens)
        setattr(usage, f"{role}_sample_tokens", output_tokens)
        return {
            "id": row["id"],
            "source_index": row.get("source_index"),
            "source_split": row.get("source_split"),
            "subject": row.get("subject"),
            "question_sha256": fingerprint(str(row["question"])),
            "prompt_token_sha256": stable_hash(prompt_ids),
            "gold": row["answer_idx"],
            "prediction": prediction,
            "correct": prediction == row["answer_idx"],
            "format_valid": prediction is not None,
            "thinking_closed": "</think>" in text,
            "truncated": output_tokens >= max_tokens,
            "output_text": text,
            "usage": usage.to_dict(),
        }

    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row["id"]) not in completed_ids
    ]
    tasks = [asyncio.create_task(infer(index, row)) for index, row in pending]
    for completed in asyncio.as_completed(tasks):
        record = await completed
        append_jsonl(records_path, record)

    by_id = {str(record["id"]): record for record in read_jsonl(records_path)}
    records = [by_id[str(row["id"])] for row in rows]
    usage = UsageLedger()
    for record in records:
        usage.add(UsageLedger.from_dict(dict(record["usage"])))
    correct = sum(bool(record["correct"]) for record in records)
    valid = sum(bool(record["format_valid"]) for record in records)
    closed = sum(bool(record["thinking_closed"]) for record in records)
    truncated = sum(bool(record["truncated"]) for record in records)
    lower, upper = wilson_interval(correct, len(records))
    subject_scores: dict[str, dict[str, int | float]] = {}
    for subject in sorted({str(record.get("subject", "")) for record in records}):
        subset = [record for record in records if str(record.get("subject", "")) == subject]
        subset_correct = sum(bool(record["correct"]) for record in subset)
        subject_scores[subject] = {
            "count": len(subset),
            "correct": subset_correct,
            "accuracy": subset_correct / len(subset),
        }
    prices = fetch_json(PRICES_URL)
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        **contract,
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "wilson_95": [lower, upper],
        "format_valid_rate": valid / len(records),
        "thinking_closed_rate": closed / len(records),
        "truncation_rate": truncated / len(records),
        "output_tokens_mean": (
            usage.teacher_sample_tokens + usage.student_sample_tokens
        )
        / len(records),
        "subject_scores": subject_scores,
        "subject_macro_accuracy": sum(
            float(value["accuracy"]) for value in subject_scores.values()
        )
        / len(subject_scores),
        "usage": usage.to_dict(),
        "estimated_cny": estimate_cost(usage, price_table(prices), config),
        "actual_billed_cny": None,
        "prediction_cache_sha256": sha256_file(records_path),
        "wall_seconds": time.perf_counter() - started,
        "execution_concurrency": execution_concurrency,
        "scope": "proxy" if request.dataset.endswith("proxy") else "full",
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def run_thinking_evaluation(
    config: ExperimentConfig, request: ThinkingEvalRequest
) -> dict[str, Any]:
    if not request.confirm_paid:
        raise ValueError("refusing paid remote work without --confirm-paid")
    manifest = config.root / "data" / "processed" / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("frozen data manifest is required")
    require_ready_preflight(config)
    return asyncio.run(_evaluate_thinking_async(config, request))
