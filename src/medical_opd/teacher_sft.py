from __future__ import annotations

import asyncio
import json
import math
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
from medical_opd.contracts import TokenizedDatum, build_ce_datum, to_pytrio_ce_datum
from medical_opd.evaluation import _load_student_tokenizer
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    read_jsonl,
    safe_slug,
    sha256_file,
    stable_hash,
    utc_now,
)
from medical_opd.training import _fit_prompt

METHOD = "T27-SFT"


@dataclass(frozen=True)
class TeacherSFTRequest:
    target_steps: int
    output_dir: Path
    confirm_paid: bool = False
    resume_state: str | None = None
    run_name: str = "t27-medical-sft"

    def __post_init__(self) -> None:
        if self.target_steps not in {1, 10, 25}:
            raise ValueError("teacher SFT target_steps must be 1, 10, or 25")
        safe_slug(self.run_name)


def _teacher_prompt_ids(tokenizer: Any, question: str, system: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    result = list(tokenizer.encode(text, add_special_tokens=False))
    if not result:
        raise ValueError("teacher SFT prompt is empty")
    return result


def _teacher_completion_text(row: dict[str, Any]) -> str:
    reasoning = str(row.get("reasoning", "")).strip()
    response = str(row.get("response", "")).strip()
    if not response:
        raise ValueError("teacher SFT row is missing the final response")
    # Thinking-enabled Qwen prompts already end with the opening <think> marker.
    if reasoning:
        return f"{reasoning}\n</think>\n\n{response}"
    return f"</think>\n\n{response}"


def build_teacher_sft_datum(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    system: str,
    max_sequence: int,
) -> tuple[TokenizedDatum, int, bool]:
    prompt = _teacher_prompt_ids(tokenizer, str(row["question"]), system)
    prompt, removed = _fit_prompt(
        prompt,
        max_sequence=max_sequence,
        reserved_completion=16,
    )
    completion = list(
        tokenizer.encode(_teacher_completion_text(row), add_special_tokens=False)
    )
    if tokenizer.eos_token_id is not None:
        completion.append(int(tokenizer.eos_token_id))
    truncated = len(prompt) + len(completion) > max_sequence
    return (
        build_ce_datum(prompt, completion, max_length=max_sequence),
        removed,
        truncated,
    )


def _medical_rows(config: ExperimentConfig) -> tuple[Path, list[dict[str, Any]]]:
    path = config.root / "data" / "processed" / "train_medical.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError("frozen medical training data is empty")
    return path, rows


def teacher_sft_contract(config: ExperimentConfig) -> dict[str, Any]:
    data_path, _ = _medical_rows(config)
    section = config.section("medical_teacher_sft")
    return {
        "protocol_id": section["protocol_id"],
        "method": METHOD,
        "base_model": str(config.get("models", "teacher")),
        "renderer_tokenizer": str(config.get("models", "student")),
        "data_sha256": sha256_file(data_path),
        "config_sha256": sha256_file(config.path),
        "seed": int(config.get("experiment", "seed")),
        "batch_size": int(section["batch_size"]),
        "lora_rank": int(section["lora_rank"]),
        "learning_rate": float(section["learning_rate"]),
        "beta1": float(section["beta1"]),
        "beta2": float(section["beta2"]),
        "max_sequence_tokens": int(section["max_sequence_tokens"]),
        "enable_thinking": True,
        "assistant_contract": "prompt opens <think>; completion closes </think> and includes EOS",
        "medical_system": str(section["medical_system"]),
    }


def _selected_rows(
    config: ExperimentConfig, target_steps: int
) -> tuple[Path, list[dict[str, Any]]]:
    path, rows = _medical_rows(config)
    count = target_steps * int(config.get("medical_teacher_sft", "batch_size"))
    if count > len(rows):
        raise RuntimeError("frozen medical order is too short for the requested teacher steps")
    return path, rows[:count]


def plan_teacher_sft(
    config: ExperimentConfig, request: TeacherSFTRequest
) -> dict[str, Any]:
    data_path, rows = _selected_rows(config, request.target_steps)
    tokenizer = _load_student_tokenizer(config)
    section = config.section("medical_teacher_sft")
    datums = [
        build_teacher_sft_datum(
            tokenizer,
            row,
            system=str(section["medical_system"]),
            max_sequence=int(section["max_sequence_tokens"]),
        )
        for row in rows
    ]
    usage = UsageLedger(
        teacher_train_tokens=sum(len(datum.input_tokens) for datum, _, _ in datums),
        optimizer_steps=request.target_steps,
    )
    prices_raw = fetch_json(PRICES_URL)
    mean_tokens = usage.teacher_train_tokens / len(rows)
    full_count = len(_medical_rows(config)[1])
    full_epoch_tokens = math.ceil(mean_tokens * full_count)
    full_epoch_usage = UsageLedger(teacher_train_tokens=full_epoch_tokens)
    manifest = json.loads(
        (config.root / "data" / "processed" / "manifest.json").read_text(encoding="utf-8")
    )
    cleaned_count = int(manifest["filters"]["medical"]["kept"])
    cleaned_tokens = math.ceil(mean_tokens * cleaned_count)
    cleaned_usage = UsageLedger(teacher_train_tokens=cleaned_tokens)
    plan = {
        "mode": "remote_paid_27b_medical_sft",
        "method": METHOD,
        "target_optimizer_steps": request.target_steps,
        "examples": len(rows),
        "batch_size": int(section["batch_size"]),
        "group_size": None,
        "base_model": str(config.get("models", "teacher")),
        "max_sequence_tokens": int(section["max_sequence_tokens"]),
        "output_dir": str(request.output_dir.resolve()),
        "training_contract": teacher_sft_contract(config),
        "training_contract_sha256": stable_hash(teacher_sft_contract(config)),
        "data_path": str(data_path.relative_to(config.root)),
        "usage": usage.to_dict(),
        "estimated_cny": estimate_cost(usage, price_table(prices_raw), config),
        "price_version": prices_raw.get("version"),
        "completion_truncation_rows": sum(truncated for _, _, truncated in datums),
        "prompt_truncated_tokens": sum(removed for _, removed, _ in datums),
        "full_epoch_extrapolation": {
            "scope": "one pass over the frozen 1,200-row project pool",
            "examples": full_count,
            "teacher_train_tokens": full_epoch_tokens,
            "estimated_cny": estimate_cost(
                full_epoch_usage,
                price_table(prices_raw),
                config,
            ),
            "warning": "Linear extrapolation from the frozen screen rows, not an invoice.",
        },
        "cleaned_corpus_extrapolation": {
            "scope": "one pass over every leakage-filtered medical-o1 row",
            "examples": cleaned_count,
            "teacher_train_tokens": cleaned_tokens,
            "estimated_cny": estimate_cost(
                cleaned_usage,
                price_table(prices_raw),
                config,
            ),
            "warning": "Linear extrapolation from the frozen screen rows, not an invoice.",
        },
        "success_criterion": (
            "finite CE loss; exact completion mask; optimizer state plus sampler weights saved; "
            "then 27B proxy evaluation must not regress before it can become an OPD teacher"
        ),
    }
    atomic_write_json(
        config.root
        / "reports"
        / "generated"
        / f"teacher_sft_plan_step{request.target_steps:03d}.json",
        plan,
    )
    return plan


def _local_state(
    config: ExperimentConfig,
    request: TeacherSFTRequest,
    *,
    status: str,
    completed_steps: int,
    usage: UsageLedger,
    checkpoints: list[dict[str, Any]],
    latest_optimizer_state: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    contract = teacher_sft_contract(config)
    result: dict[str, Any] = {
        "status": status,
        "method": METHOD,
        "target_steps": request.target_steps,
        "completed_steps": completed_steps,
        "next_step": completed_steps + 1,
        "sample_cursor": completed_steps
        * int(config.get("medical_teacher_sft", "batch_size")),
        "seed": int(config.get("experiment", "seed")),
        "base_model": str(config.get("models", "teacher")),
        "latest_optimizer_state": latest_optimizer_state,
        "checkpoints": checkpoints,
        "usage": usage.to_dict(),
        "training_contract": contract,
        "training_contract_sha256": stable_hash(contract),
        "updated_at": utc_now(),
    }
    if error is not None:
        result["error"] = error
    return result


def _validate_resume(
    config: ExperimentConfig,
    request: TeacherSFTRequest,
    prior: dict[str, Any],
) -> None:
    if prior.get("method") != METHOD:
        raise RuntimeError("teacher SFT resume method mismatch")
    completed = int(prior.get("completed_steps", -1))
    if completed < 0 or request.target_steps <= completed:
        raise RuntimeError("teacher SFT resume target must exceed the completed step")
    if prior.get("training_contract_sha256") != stable_hash(teacher_sft_contract(config)):
        raise RuntimeError("teacher SFT resume contract or data drift")
    if request.resume_state != prior.get("latest_optimizer_state"):
        raise RuntimeError("teacher SFT resume state does not match the latest saved optimizer")
    expected_cursor = completed * int(config.get("medical_teacher_sft", "batch_size"))
    if int(prior.get("sample_cursor", -1)) != expected_cursor:
        raise RuntimeError("teacher SFT resume cursor mismatch")


def _capability_map(capabilities: Any) -> dict[str, dict[str, Any]]:
    if hasattr(capabilities, "model_dump"):
        payload = capabilities.model_dump(mode="json")
    elif hasattr(capabilities, "dict"):
        payload = capabilities.dict()
    else:
        raise TypeError("unsupported PyTRIO capabilities response")
    return {str(item["model_name"]): item for item in payload.get("supported_models", [])}


async def _run_teacher_sft_async(
    config: ExperimentConfig,
    request: TeacherSFTRequest,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    output = request.output_dir.resolve()
    state_path = output / "state.json"
    steps_path = output / "steps.jsonl"
    section = config.section("medical_teacher_sft")
    _, rows = _medical_rows(config)
    tokenizer = _load_student_tokenizer(config)
    completed = 0 if prior is None else int(prior["completed_steps"])
    usage = UsageLedger() if prior is None else UsageLedger.from_dict(dict(prior["usage"]))
    checkpoints = [] if prior is None else list(prior.get("checkpoints", []))
    latest_state = None if prior is None else str(prior["latest_optimizer_state"])

    service = trio.ServiceClient()
    models = _capability_map(service.get_server_capabilities())
    teacher_name = str(config.get("models", "teacher"))
    teacher = models.get(teacher_name)
    if teacher is None:
        raise RuntimeError("configured 27B teacher is absent from workspace capabilities")
    if not bool(teacher["training"]["available"]):
        raise RuntimeError("configured 27B teacher training is unavailable")
    if not bool(teacher["sampling"]["available"]):
        raise RuntimeError(
            "configured 27B teacher sampling is unavailable; do not pay for an unusable SFT teacher"
        )

    if prior is None:
        training = await service.create_lora_training_client_async(
            base_model=teacher_name,
            rank=int(section["lora_rank"]),
            seed=int(config.get("experiment", "seed")),
        )
    else:
        training = await service.create_training_client_from_state_with_optimizer_async(
            request.resume_state
        )
    adam = trio.AdamParams(
        learning_rate=float(section["learning_rate"]),
        beta1=float(section["beta1"]),
        beta2=float(section["beta2"]),
    )
    batch_size = int(section["batch_size"])
    max_sequence = int(section["max_sequence_tokens"])
    save_steps = {int(value) for value in section["save_steps"]}
    try:
        for step in range(completed, request.target_steps):
            started = time.perf_counter()
            batch = rows[step * batch_size : (step + 1) * batch_size]
            if len(batch) != batch_size:
                raise RuntimeError("frozen medical order exhausted during teacher SFT")
            built = [
                build_teacher_sft_datum(
                    tokenizer,
                    row,
                    system=str(section["medical_system"]),
                    max_sequence=max_sequence,
                )
                for row in batch
            ]
            datums = [datum for datum, _, _ in built]
            submitted = [to_pytrio_ce_datum(trio, datum) for datum in datums]
            result_future = await training.forward_backward_async(
                submitted,
                loss_fn="cross_entropy",
            )
            optim_future = await training.optim_step_async(adam)
            result = await result_future
            await optim_future
            metrics = {
                str(key): float(value)
                for key, value in dict(getattr(result, "metrics", {})).items()
                if isinstance(value, (int, float))
            }
            if any(not math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("27B teacher trainer returned NaN or infinity")
            submitted_tokens = sum(len(datum.input_tokens) for datum in datums)
            trainable_tokens = sum(datum.trainable_tokens for datum in datums)
            step_usage = UsageLedger(
                teacher_train_tokens=submitted_tokens,
                optimizer_steps=1,
                wall_seconds=time.perf_counter() - started,
            )
            usage.add(step_usage)
            completed = step + 1
            append_jsonl(
                steps_path,
                {
                    "step": completed,
                    "method": METHOD,
                    "row_ids": [row["id"] for row in batch],
                    "submitted_sequence_tokens": submitted_tokens,
                    "trainable_mask_tokens": trainable_tokens,
                    "prompt_truncated_tokens": sum(removed for _, removed, _ in built),
                    "completion_truncation_rows": sum(
                        truncated for _, _, truncated in built
                    ),
                    "trainer_metrics": metrics,
                    "usage": step_usage.to_dict(),
                },
            )
            state_future = await training.save_state_async(
                name=f"{request.run_name}-step{completed:06d}-state"
            )
            latest_state = str((await state_future).path)
            if completed in save_steps or completed == request.target_steps:
                weights_future = await training.save_weights_for_sampler_async(
                    name=f"{request.run_name}-step{completed:06d}-weights"
                )
                weights = str((await weights_future).path)
                checkpoint = {
                    "step": completed,
                    "state": latest_state,
                    "sampler_weights": weights,
                    "permanent": completed == 25,
                }
                checkpoints.append(checkpoint)
                append_jsonl(
                    config.root / "reports" / "generated" / "checkpoint_index.jsonl",
                    {"run_dir": str(output), **checkpoint},
                )
            atomic_write_json(
                state_path,
                _local_state(
                    config,
                    request,
                    status="running",
                    completed_steps=completed,
                    usage=usage,
                    checkpoints=checkpoints,
                    latest_optimizer_state=latest_state,
                ),
            )
        prices_raw = fetch_json(PRICES_URL)
        summary = _local_state(
            config,
            request,
            status="completed",
            completed_steps=completed,
            usage=usage,
            checkpoints=checkpoints,
            latest_optimizer_state=latest_state,
        )
        summary["estimated_cny"] = estimate_cost(usage, price_table(prices_raw), config)
        summary["price_version"] = prices_raw.get("version")
        summary["actual_billed_cny"] = None
        summary["billing_note"] = (
            "No SDK billing field; reconcile against https://pytrio.cn/usage."
        )
        summary["result_scope"] = (
            "smoke_only"
            if completed == 1
            else "cost_calibration_only" if completed == 10 else "teacher_screen_training"
        )
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        atomic_write_json(
            state_path,
            _local_state(
                config,
                request,
                status="failed",
                completed_steps=completed,
                usage=usage,
                checkpoints=checkpoints,
                latest_optimizer_state=latest_state,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def run_teacher_sft(
    config: ExperimentConfig, request: TeacherSFTRequest
) -> dict[str, Any]:
    if not request.confirm_paid:
        raise RuntimeError("paid 27B teacher SFT requires --confirm-paid")
    require_ready_preflight(config)
    output = request.output_dir.resolve()
    state_path = output / "state.json"
    prior: dict[str, Any] | None = None
    if state_path.exists():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed" and int(prior["completed_steps"]) >= (
            request.target_steps
        ):
            return prior
        if request.resume_state is None:
            raise RuntimeError("existing teacher SFT run requires --resume-state")
        _validate_resume(config, request, prior)
    elif request.resume_state is not None:
        raise RuntimeError("teacher SFT resume requires an existing local state")
    output.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run_teacher_sft_async(config, request, prior))
