from __future__ import annotations

import asyncio
import json
import math
import random
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
from medical_opd.data import _choice_row, _load_medqa
from medical_opd.evaluation import _load_student_tokenizer
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    utc_now,
    write_jsonl,
)

REFERENCE_SYSTEM = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)
REFERENCE_SEED = 42
REFERENCE_PROTOCOL = {
    "medical_600": {
        "filename": "medqa_test_seed42_600.jsonl",
        "count": 600,
        "max_tokens": 1024,
        "reported_correct": 433,
    },
    "ceval_300": {
        "filename": "ceval_mixed_seed42_300.jsonl",
        "count": 300,
        "max_tokens": 8192,
        "reported_correct": 245,
    },
}


@dataclass(frozen=True)
class ReferenceEvalRequest:
    dataset: str
    output_dir: Path
    model_path: str | None = None
    limit: int = 0
    confirm_paid: bool = False

    def __post_init__(self) -> None:
        if self.dataset not in REFERENCE_PROTOCOL:
            raise ValueError(f"dataset must be one of {sorted(REFERENCE_PROTOCOL)}")
        if self.limit < 0:
            raise ValueError("limit must be nonnegative")
        if self.model_path is not None and "/sampler_weights/" not in self.model_path:
            raise ValueError("reference evaluation model_path must be sampler weights")


def _reference_dir(config: ExperimentConfig) -> Path:
    return config.root / "data" / "reference"


def _reference_path(config: ExperimentConfig, dataset: str) -> Path:
    return _reference_dir(config) / str(REFERENCE_PROTOCOL[dataset]["filename"])


def _ceval_reference_rows(config: ExperimentConfig, dataset_cache: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for subject in config.get("data", "ceval_subjects"):
        for split in ("dev", "val", "test"):
            source = load_dataset(
                str(config.get("data", "ceval_repo")),
                str(subject),
                split=split,
                cache_dir=str(dataset_cache),
                revision=str(config.get("data", "ceval_labeled_revision")),
            )
            for index, raw in enumerate(source):
                row = _choice_row(
                    dict(raw),
                    source_dataset=str(config.get("data", "ceval_repo")),
                    source_split=split,
                    subject=str(subject),
                    source_index=index,
                )
                if row is not None:
                    rows.append(row)
    random.Random(REFERENCE_SEED).shuffle(rows)
    heldout_size = math.ceil(len(rows) * 0.2)
    return rows[:heldout_size][: int(REFERENCE_PROTOCOL["ceval_300"]["count"])]


def prepare_reference_data(config: ExperimentConfig, shared_cache: Path) -> dict[str, Any]:
    reference_dir = _reference_dir(config)
    reference_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = shared_cache / "hub"
    dataset_cache = shared_cache / "datasets"

    medical = _load_medqa(config, hub_cache, "test")
    random.Random(REFERENCE_SEED).shuffle(medical)
    medical = medical[: int(REFERENCE_PROTOCOL["medical_600"]["count"])]
    ceval = _ceval_reference_rows(config, dataset_cache)
    if len(medical) != 600 or len(ceval) != 300:
        raise RuntimeError(
            f"reference data count mismatch: medical={len(medical)}, ceval={len(ceval)}"
        )

    paths = {
        "medical_600": _reference_path(config, "medical_600"),
        "ceval_300": _reference_path(config, "ceval_300"),
    }
    write_jsonl(paths["medical_600"], medical)
    write_jsonl(paths["ceval_300"], ceval)
    manifest = {
        "status": "frozen_reference_diagnostic_only",
        "created_at": utc_now(),
        "implementation_revision": "52c2f1c98fe",
        "seed": REFERENCE_SEED,
        "student_model": str(config.get("models", "student")),
        "source_revisions": {
            "medqa": str(config.get("data", "medqa_revision")),
            "ceval": str(config.get("data", "ceval_labeled_revision")),
        },
        "datasets": {
            name: {
                "path": str(path.relative_to(config.root)),
                "count": len(read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "ceval_split_counts": dict(Counter(str(row["source_split"]) for row in ceval)),
        "warning": (
            "This historical C-Eval diagnostic mixes dev/val/test before a random 80/20 split. "
            "It is retained only for protocol-level comparison and must never be used for "
            "training, checkpoint selection, or the leakage-safe main conclusion."
        ),
    }
    atomic_write_json(reference_dir / "manifest.json", manifest)
    return manifest


def build_reference_prompt(row: dict[str, Any]) -> str:
    options = row["options"]
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{row['question']}",
            f"A. {options['A']}",
            f"B. {options['B']}",
            f"C. {options['C']}",
            f"D. {options['D']}",
            "",
            "答案：",
        ]
    )


def render_reference_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": REFERENCE_SYSTEM},
            {"role": "user", "content": build_reference_prompt(row)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def parse_reference_choice(text: str) -> str | None:
    raw_text = text.strip()
    segments = [raw_text.rsplit("</think>", 1)[-1]] if "</think>" in raw_text else [raw_text]
    for segment in segments:
        normalized = segment.strip().upper()
        if not normalized:
            continue
        exact = re.fullmatch(r"[^A-Z]*([ABCD])[^A-Z]*", normalized)
        if exact:
            return exact.group(1)
        patterns = [
            r"(?:FINAL\s*(?:ANSWER|OUTPUT)|ANSWER|OUTPUT|最终答案|正确答案|答案|选项|选择|应选|故选)\s*(?:IS|是|为|:|：|->)?\s*[\(\[【]?\s*([ABCD])",
            r"[\(\[【]\s*([ABCD])\s*[\)\]】]",
        ]
        candidates = [match for pattern in patterns for match in re.findall(pattern, normalized)]
        if candidates:
            return candidates[-1]
        for line in reversed(normalized.splitlines()):
            line_match = re.fullmatch(r"\s*([ABCD])\s*[\.。:：、]?\s*", line)
            if line_match:
                return line_match.group(1)
    fallback = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", segments[0].upper()[-1000:])
    return fallback[-1] if fallback else None


def _selected_rows(config: ExperimentConfig, request: ReferenceEvalRequest) -> list[dict[str, Any]]:
    rows = read_jsonl(_reference_path(config, request.dataset))
    expected = int(REFERENCE_PROTOCOL[request.dataset]["count"])
    if len(rows) != expected:
        raise RuntimeError(f"reference dataset must contain {expected} rows, found {len(rows)}")
    return rows if request.limit == 0 else rows[: request.limit]


def plan_reference_evaluation(
    config: ExperimentConfig, request: ReferenceEvalRequest
) -> dict[str, Any]:
    rows = _selected_rows(config, request)
    tokenizer = _load_student_tokenizer(config)
    prompt_lengths = [len(render_reference_ids(tokenizer, row)) for row in rows]
    max_tokens = int(REFERENCE_PROTOCOL[request.dataset]["max_tokens"])
    usage = UsageLedger(
        student_prefill_tokens=sum(prompt_lengths),
        student_sample_tokens=len(rows) * max_tokens,
    )
    prices = fetch_json(PRICES_URL)
    return {
        "mode": "remote_paid_reference_protocol_evaluation",
        "model": str(config.get("models", "student")),
        "model_path": request.model_path,
        "dataset": request.dataset,
        "dataset_count": len(rows),
        "dataset_sha256": sha256_file(_reference_path(config, request.dataset)),
        "thinking": True,
        "temperature": 0.01,
        "top_p": 0.9,
        "seed": None,
        "max_output_tokens_per_row": max_tokens,
        "prompt_token_total": sum(prompt_lengths),
        "upper_bound_usage": usage.to_dict(),
        "upper_bound_estimated_cny": estimate_cost(usage, price_table(prices), config),
        "price_version": prices.get("version"),
        "output_dir": str(request.output_dir.resolve()),
        "success_criterion": (
            "all selected rows cached; thinking terminates; published-table delta reported"
        ),
    }


async def _evaluate_reference_async(
    config: ExperimentConfig, request: ReferenceEvalRequest
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    rows = _selected_rows(config, request)
    dataset_path = _reference_path(config, request.dataset)
    protocol = REFERENCE_PROTOCOL[request.dataset]
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "predictions.jsonl"
    existing = read_jsonl(records_path) if records_path.exists() else []
    completed_ids = {str(record["id"]) for record in existing}
    contract = {
        "model": str(config.get("models", "student")),
        "model_path": request.model_path,
        "dataset": request.dataset,
        "limit": request.limit,
        "dataset_sha256": sha256_file(dataset_path),
        "config_sha256": sha256_file(config.path),
        "thinking": True,
        "temperature": 0.01,
        "top_p": 0.9,
        "seed": None,
        "max_tokens": int(protocol["max_tokens"]),
        "prompt_contract": "Chinese choice prompt rendered with enable_thinking=True",
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("reference evaluation cache contract mismatch")
    else:
        atomic_write_json(request_path, contract)

    tokenizer = _load_student_tokenizer(config)
    service = trio.ServiceClient()
    sampler = await service.create_sampling_client_async(
        base_model=str(config.get("models", "student")), model_path=request.model_path
    )
    semaphore = asyncio.Semaphore(16)
    started = time.perf_counter()

    async def infer(row: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = render_reference_ids(tokenizer, row)
        async with semaphore:
            response = await sampler.sample_async(
                prompt=trio.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=trio.SamplingParams(
                    max_tokens=int(protocol["max_tokens"]),
                    temperature=0.01,
                    top_p=0.9,
                ),
                return_text=True,
            )
        sequence = response.sequences[0]
        text = str(sequence.text or "")
        prediction = parse_reference_choice(text)
        input_tokens = int(getattr(response, "input_tokens", len(prompt_ids)))
        output_tokens = int(getattr(response, "output_tokens", len(sequence.tokens)))
        return {
            "id": row["id"],
            "source_split": row.get("source_split"),
            "source_index": row.get("source_index"),
            "subject": row.get("subject"),
            "gold": row["answer_idx"],
            "prediction": prediction,
            "correct": prediction == row["answer_idx"],
            "format_valid": prediction is not None,
            "thinking_closed": "</think>" in text,
            "truncated": output_tokens >= int(protocol["max_tokens"]),
            "prompt_token_sha256": stable_hash(prompt_ids),
            "output_text": text,
            "usage": UsageLedger(
                student_prefill_tokens=input_tokens,
                student_sample_tokens=output_tokens,
            ).to_dict(),
        }

    pending = [row for row in rows if str(row["id"]) not in completed_ids]
    for record in await asyncio.gather(*(infer(row) for row in pending)):
        append_jsonl(records_path, record)

    by_id = {str(record["id"]): record for record in read_jsonl(records_path)}
    records = [by_id[str(row["id"])] for row in rows]
    usage = UsageLedger()
    for record in records:
        usage.add(UsageLedger.from_dict(record["usage"]))
    correct = sum(bool(record["correct"]) for record in records)
    reported_correct = int(protocol["reported_correct"])
    prices = fetch_json(PRICES_URL)
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        "model": str(config.get("models", "student")),
        "model_path": request.model_path,
        "dataset": request.dataset,
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "format_valid_rate": sum(bool(r["format_valid"]) for r in records) / len(records),
        "thinking_closed_rate": sum(bool(r["thinking_closed"]) for r in records) / len(records),
        "truncation_rate": sum(bool(r["truncated"]) for r in records) / len(records),
        "output_tokens_mean": usage.student_sample_tokens / len(records),
        "reported_reference_correct": reported_correct,
        "reported_reference_accuracy": reported_correct / int(protocol["count"]),
        "delta_vs_reported_pp": (
            correct / len(records) - reported_correct / int(protocol["count"])
        )
        * 100,
        "usage": usage.to_dict(),
        "estimated_cny": estimate_cost(usage, price_table(prices), config),
        "actual_billed_cny": None,
        "dataset_sha256": sha256_file(dataset_path),
        "prediction_cache_sha256": sha256_file(records_path),
        "wall_seconds": time.perf_counter() - started,
        "protocol": contract,
        "warning": (
            "Reference-protocol diagnostic only. Its mixed-split C-Eval set is not the "
            "leakage-safe main evaluation."
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def run_reference_evaluation(
    config: ExperimentConfig, request: ReferenceEvalRequest
) -> dict[str, Any]:
    if not request.confirm_paid:
        raise ValueError("refusing paid remote work without --confirm-paid")
    manifest = _reference_dir(config) / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("reference data manifest is required")
    require_ready_preflight(config)
    return asyncio.run(_evaluate_reference_async(config, request))
