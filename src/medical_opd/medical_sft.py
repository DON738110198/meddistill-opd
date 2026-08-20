from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
import tomllib
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
from medical_opd.contracts import TokenizedDatum, to_pytrio_ce_datum
from medical_opd.data import _deduplicate_rows, _load_medqa, _medical_rows
from medical_opd.evaluation import _load_student_tokenizer
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    distribution,
    fingerprint,
    read_jsonl,
    safe_slug,
    sha256_file,
    stable_hash,
    utc_now,
    write_jsonl,
)
from medical_opd.teacher_sft import build_teacher_sft_datum
from medical_opd.training import TRAINING_PAUSE_FILENAME

DEFAULT_MEDICAL_PIPELINE_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "medical-pipeline.toml"
)
METHOD = "MED-SFT"


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: sha256_file(package / name)
        for name in (
            "contracts.py",
            "teacher_sft.py",
            "training.py",
            "medical_sft.py",
        )
    }


@dataclass(frozen=True)
class MedicalPipelineConfig:
    path: Path
    raw: dict[str, Any]
    root: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing pipeline config section: {name}")
        return value

    def get(self, section: str, key: str) -> Any:
        values = self.section(section)
        if key not in values:
            raise ValueError(f"missing pipeline config value: {section}.{key}")
        return values[key]

    def resolve(self, section: str, key: str) -> Path:
        path = (self.root / str(self.get(section, key))).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"pipeline path escapes project root: {section}.{key}") from exc
        return path


def load_medical_pipeline_config(
    experiment: ExperimentConfig, path: Path | None = None
) -> MedicalPipelineConfig:
    config_path = (path or DEFAULT_MEDICAL_PIPELINE_CONFIG).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = MedicalPipelineConfig(config_path, raw, experiment.root)
    _validate_medical_pipeline_config(experiment, config)
    return config


def _validate_medical_pipeline_config(
    experiment: ExperimentConfig, config: MedicalPipelineConfig
) -> None:
    for section in ("protocol", "medical_sft", "teacher_gate", "medical_opd", "sar"):
        config.section(section)
    if config.get("protocol", "id") != "medical-opd-sar-v1":
        raise ValueError("pipeline protocol id must remain medical-opd-sar-v1")
    implementation_revision = str(config.get("protocol", "implementation_revision"))
    if re.fullmatch(r"[0-9a-f]{40}", implementation_revision) is None:
        raise ValueError("pipeline implementation_revision must be a pinned 40-hex commit")
    if config.get("protocol", "base_model") != experiment.get("models", "student"):
        raise ValueError("pipeline must use the official configured 4B model")
    if int(config.get("protocol", "seed")) != int(experiment.get("experiment", "seed")):
        raise ValueError("pipeline seed must match the frozen experiment seed")
    if config.get("medical_sft", "method") != METHOD:
        raise ValueError("pipeline medical SFT method must remain MED-SFT")
    expected_sft = {
        "lora_rank": 32,
        "batch_size": 16,
        "epochs": 3,
        "max_sequence_tokens": 2048,
    }
    for key, expected in expected_sft.items():
        if int(config.get("medical_sft", key)) != expected:
            raise ValueError(f"pipeline medical_sft.{key} must remain {expected}")
    if float(config.get("medical_sft", "learning_rate")) != 1e-4:
        raise ValueError("pipeline Medical SFT learning rate must remain 1e-4")
    if int(config.get("medical_sft", "state_save_interval")) <= 0:
        raise ValueError("pipeline state_save_interval must be positive")
    if [int(value) for value in config.get("medical_sft", "checkpoint_steps")] != [
        1,
        10,
        25,
        50,
    ]:
        raise ValueError("pipeline Medical SFT must preserve checkpoints 1/10/25/50")
    for section, learning_rate in (("medical_opd", 4e-5), ("sar", 5e-6)):
        if int(config.get(section, "batch_size")) != 4:
            raise ValueError(f"pipeline {section} requires batch_size=4")
        if int(config.get(section, "group_size")) != 4:
            raise ValueError(f"pipeline {section} requires group_size=4")
        if config.get(section, "enable_thinking") is not True:
            raise ValueError(f"pipeline {section} requires thinking-enabled rollouts")
        if int(config.get(section, "max_completion_tokens")) != 2048:
            raise ValueError(f"pipeline {section} must preserve the 2048-token completion cap")
        if int(config.get(section, "max_sequence_tokens")) < 2048:
            raise ValueError(f"pipeline {section} max_sequence_tokens is too small")
        if float(config.get(section, "learning_rate")) != learning_rate:
            raise ValueError(f"pipeline {section} learning rate drift")
        if int(config.get(section, "state_save_interval")) <= 0:
            raise ValueError(f"pipeline {section} state_save_interval must be positive")
        if [int(value) for value in config.get(section, "checkpoint_steps")] != [1, 10, 25]:
            raise ValueError(f"pipeline {section} checkpoints must remain 1/10/25")
    if config.get("sar", "prompt_source") != "alpaca-zh-general-replay":
        raise ValueError("leakage-safe SAR must use Alpaca-ZH, never C-Eval")
    for section, key in (
        ("protocol", "medical_train_path"),
        ("protocol", "general_replay_path"),
    ):
        path = str(config.get(section, key)).casefold()
        if "ceval" in path or "c-eval" in path:
            raise ValueError("C-Eval rows must never be pipeline training prompts")
        config.resolve(section, key)


def _data_paths(config: MedicalPipelineConfig) -> dict[str, Path]:
    data_dir = config.resolve("protocol", "medical_train_path").parent
    return {
        "medical": config.resolve("protocol", "medical_train_path"),
        "quarantine": data_dir / "quarantine_medical.jsonl",
        "lengths": data_dir / "sft_lengths.jsonl",
        "manifest": data_dir / "manifest.json",
    }


def prepare_medical_sft_data(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    *,
    shared_cache: Path,
) -> dict[str, Any]:
    from datasets import load_dataset

    raw = load_dataset(
        str(experiment.get("data", "medical_repo")),
        str(experiment.get("data", "medical_config")),
        split="train",
        cache_dir=str((shared_cache / "datasets").resolve()),
        revision=str(experiment.get("data", "medical_revision")),
    )
    medical = _medical_rows(raw)
    heldout_dev = _load_medqa(experiment, (shared_cache / "hub").resolve(), "dev")
    heldout_test = read_jsonl(
        experiment.root / "data" / "processed" / "eval_medical_full.jsonl"
    )
    heldout = heldout_dev + heldout_test
    quarantine: list[dict[str, Any]] = []
    clean, counters = _deduplicate_rows(
        medical,
        text_key="question",
        heldout_texts=[str(row["question"]) for row in heldout],
        heldout_ids=[str(row["id"]) for row in heldout],
        threshold=float(experiment.get("data", "near_duplicate_threshold")),
        quarantine=quarantine,
    )
    if not clean:
        raise RuntimeError("pipeline leakage-cleaned medical corpus is empty")
    paths = _data_paths(config)
    write_jsonl(paths["medical"], clean)
    write_jsonl(paths["quarantine"], quarantine)

    tokenizer = _load_student_tokenizer(experiment)
    max_sequence = int(config.get("medical_sft", "max_sequence_tokens"))
    system = str(config.get("protocol", "medical_system"))
    length_rows: list[dict[str, Any]] = []
    for row in clean:
        datum, prompt_removed, completion_truncated = build_teacher_sft_datum(
            tokenizer,
            row,
            system=system,
            max_sequence=max_sequence,
        )
        length_rows.append(
            {
                "id": row["id"],
                "input_tokens": len(datum.input_tokens),
                "trainable_tokens": datum.trainable_tokens,
                "prompt_truncated_tokens": prompt_removed,
                "completion_truncated": completion_truncated,
            }
        )
    write_jsonl(paths["lengths"], length_rows)
    manifest = {
        "status": "frozen",
        "created_at": utc_now(),
        "protocol_id": config.get("protocol", "id"),
        "implementation_revision": config.get("protocol", "implementation_revision"),
        "source": {
            "repo": experiment.get("data", "medical_repo"),
            "config": experiment.get("data", "medical_config"),
            "revision": experiment.get("data", "medical_revision"),
            "input_rows": len(medical),
        },
        "leakage_filter": counters,
        "heldout": {
            "medqa_dev_count": len(heldout_dev),
            "medqa_dev_identity_sha256": stable_hash(
                [
                    [row["id"], fingerprint(str(row["question"]))]
                    for row in heldout_dev
                ]
            ),
            "medqa_proxy_sha256": sha256_file(
                experiment.root / "data" / "processed" / "eval_medical_proxy.jsonl"
            ),
            "medqa_test_count": len(heldout_test),
            "medqa_test_sha256": sha256_file(
                experiment.root / "data" / "processed" / "eval_medical_full.jsonl"
            ),
        },
        "files": {
            "medical": {"count": len(clean), "sha256": sha256_file(paths["medical"])},
            "quarantine": {
                "count": len(quarantine),
                "sha256": sha256_file(paths["quarantine"]),
            },
            "lengths": {
                "count": len(length_rows),
                "sha256": sha256_file(paths["lengths"]),
            },
        },
        "sft_length_audit": {
            "sequence_tokens": distribution(
                [int(row["input_tokens"]) + 1 for row in length_rows]
            ),
            "trainable_tokens": distribution(
                [int(row["trainable_tokens"]) for row in length_rows]
            ),
            "completion_truncation_rows": sum(
                bool(row["completion_truncated"]) for row in length_rows
            ),
            "prompt_truncated_tokens": sum(
                int(row["prompt_truncated_tokens"]) for row in length_rows
            ),
            "max_sequence_tokens": max_sequence,
        },
        "training_split_contract": (
            "Medical-O1 train after exact, intra-corpus, and near-duplicate filtering against "
            "all frozen MedQA dev/test rows; no C-Eval row is used."
        ),
    }
    atomic_write_json(paths["manifest"], manifest)
    return manifest


def _load_frozen_data(
    config: MedicalPipelineConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    paths = _data_paths(config)
    for path in paths.values():
        if not path.exists():
            raise RuntimeError(f"pipeline data artifact is missing: {path}")
    rows = read_jsonl(paths["medical"])
    lengths = {str(row["id"]): row for row in read_jsonl(paths["lengths"])}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if len(rows) != len(lengths) or any(str(row["id"]) not in lengths for row in rows):
        raise RuntimeError("pipeline data and length audit IDs do not align")
    if (
        manifest.get("status") != "frozen"
        or manifest["files"]["medical"]["sha256"] != sha256_file(paths["medical"])
        or manifest["files"]["quarantine"]["sha256"]
        != sha256_file(paths["quarantine"])
        or manifest["files"]["lengths"]["sha256"] != sha256_file(paths["lengths"])
    ):
        raise RuntimeError("pipeline frozen data manifest drift")
    return rows, lengths, manifest


def _steps_per_epoch(row_count: int, batch_size: int) -> int:
    return math.ceil(row_count / batch_size)


def _batch_indices(
    *, row_count: int, batch_size: int, seed: int, step: int
) -> tuple[int, list[int]]:
    steps_per_epoch = _steps_per_epoch(row_count, batch_size)
    epoch = step // steps_per_epoch
    batch_index = step % steps_per_epoch
    indices = list(range(row_count))
    random.Random(seed + epoch).shuffle(indices)
    start = batch_index * batch_size
    return epoch, indices[start : start + batch_size]


def medical_sft_contract(
    experiment: ExperimentConfig, config: MedicalPipelineConfig
) -> dict[str, Any]:
    paths = _data_paths(config)
    rows, _, _ = _load_frozen_data(config)
    section = config.section("medical_sft")
    return {
        "protocol_id": config.get("protocol", "id"),
        "implementation_revision": config.get("protocol", "implementation_revision"),
        "method": METHOD,
        "base_model": config.get("protocol", "base_model"),
        "renderer_tokenizer": experiment.get("models", "student"),
        "seed": config.get("protocol", "seed"),
        "data_count": len(rows),
        "data_sha256": sha256_file(paths["medical"]),
        "data_manifest_sha256": sha256_file(paths["manifest"]),
        "length_audit_sha256": sha256_file(paths["lengths"]),
        "implementation_sha256": _implementation_hashes(),
        "sft_config_sha256": stable_hash(
            {
                "protocol": config.section("protocol"),
                "medical_sft": config.section("medical_sft"),
                "teacher_gate": config.section("teacher_gate"),
            }
        ),
        "tokenizer_audit_sha256": sha256_file(
            experiment.root / "data" / "processed" / "tokenizer_compatibility.json"
        ),
        "lora_rank": section["lora_rank"],
        "batch_size": section["batch_size"],
        "epochs": section["epochs"],
        "max_sequence_tokens": section["max_sequence_tokens"],
        "learning_rate": section["learning_rate"],
        "beta1": section["beta1"],
        "beta2": section["beta2"],
        "enable_thinking": True,
        "assistant_contract": "prompt opens <think>; completion closes </think> and includes EOS",
        "medical_system": config.get("protocol", "medical_system"),
        "split_contract": "leakage-cleaned Medical-O1 only; MedQA and C-Eval are evaluation-only",
    }


@dataclass(frozen=True)
class MedicalSFTRequest:
    target_steps: int
    output_dir: Path
    confirm_paid: bool = False
    resume_state: str | None = None
    run_name: str = "medical-sft"

    def __post_init__(self) -> None:
        if self.target_steps <= 0:
            raise ValueError("pipeline SFT target_steps must be positive")
        safe_slug(self.run_name)


def plan_medical_sft(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: MedicalSFTRequest,
) -> dict[str, Any]:
    rows, lengths, manifest = _load_frozen_data(config)
    section = config.section("medical_sft")
    batch_size = int(section["batch_size"])
    epochs = int(section["epochs"])
    steps_per_epoch = _steps_per_epoch(len(rows), batch_size)
    total_steps = steps_per_epoch * epochs
    if request.target_steps > total_steps:
        raise ValueError(f"pipeline SFT target exceeds three epochs ({total_steps} steps)")
    selected_tokens = 0
    selected_trainable = 0
    selected_examples = 0
    selected_truncated = 0
    seed = int(config.get("protocol", "seed"))
    for step in range(request.target_steps):
        _, indices = _batch_indices(
            row_count=len(rows), batch_size=batch_size, seed=seed, step=step
        )
        for index in indices:
            audit = lengths[str(rows[index]["id"])]
            selected_tokens += int(audit["input_tokens"])
            selected_trainable += int(audit["trainable_tokens"])
            selected_truncated += int(bool(audit["completion_truncated"]))
            selected_examples += 1
    usage = UsageLedger(
        student_train_tokens=selected_tokens,
        optimizer_steps=request.target_steps,
    )
    prices_raw = fetch_json(PRICES_URL)
    plan = {
        "mode": "remote_paid_4b_medical_sft",
        "method": METHOD,
        "base_model": config.get("protocol", "base_model"),
        "target_optimizer_steps": request.target_steps,
        "epochs_completed_at_target": request.target_steps / steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "three_epoch_steps": total_steps,
        "examples_presented": selected_examples,
        "unique_training_rows": len(rows),
        "batch_size": batch_size,
        "group_size": None,
        "max_sequence_tokens": section["max_sequence_tokens"],
        "output_dir": str(request.output_dir.resolve()),
        "training_contract": medical_sft_contract(experiment, config),
        "training_contract_sha256": stable_hash(medical_sft_contract(experiment, config)),
        "usage": usage.to_dict(),
        "trainable_mask_tokens": selected_trainable,
        "completion_truncation_rows": selected_truncated,
        "estimated_cny": estimate_cost(usage, price_table(prices_raw), experiment),
        "price_version": prices_raw.get("version"),
        "data_manifest": manifest,
        "success_criterion": (
            "finite CE loss and exact completion mask; resumable optimizer plus sampler "
            "checkpoint; "
            "then improve the frozen MedQA proxy by at least 2pp with valid answer formatting"
        ),
    }
    atomic_write_json(
        experiment.root
        / "reports"
        / "generated"
        / f"pipeline_sft_plan_step{request.target_steps:04d}.json",
        plan,
    )
    return plan


def _capability_map(capabilities: Any) -> dict[str, dict[str, Any]]:
    if hasattr(capabilities, "model_dump"):
        payload = capabilities.model_dump(mode="json")
    elif hasattr(capabilities, "dict"):
        payload = capabilities.dict()
    else:
        raise TypeError("unsupported PyTRIO capabilities response")
    return {str(item["model_name"]): item for item in payload.get("supported_models", [])}


def _state_payload(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: MedicalSFTRequest,
    *,
    status: str,
    completed_steps: int,
    recoverable_step: int,
    latest_optimizer_state: str | None,
    usage: UsageLedger,
    checkpoints: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    rows, _, _ = _load_frozen_data(config)
    batch_size = int(config.get("medical_sft", "batch_size"))
    steps_per_epoch = _steps_per_epoch(len(rows), batch_size)
    contract = medical_sft_contract(experiment, config)
    payload: dict[str, Any] = {
        "status": status,
        "method": METHOD,
        "target_steps": request.target_steps,
        "completed_steps": completed_steps,
        "recoverable_step": recoverable_step,
        "uncheckpointed_steps": completed_steps - recoverable_step,
        "steps_per_epoch": steps_per_epoch,
        "epoch_progress": completed_steps / steps_per_epoch,
        "latest_optimizer_state": latest_optimizer_state,
        "usage": usage.to_dict(),
        "checkpoints": checkpoints,
        "training_contract": contract,
        "training_contract_sha256": stable_hash(contract),
        "updated_at": utc_now(),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _validate_resume(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: MedicalSFTRequest,
    prior: dict[str, Any],
) -> None:
    if prior.get("method") != METHOD:
        raise RuntimeError("pipeline SFT resume method mismatch")
    completed = int(prior.get("completed_steps", -1))
    recoverable = int(prior.get("recoverable_step", -2))
    if completed < 0 or completed != recoverable:
        raise RuntimeError(
            "pipeline SFT has an uncheckpointed tail; resume from the recorded recoverable "
            "checkpoint in a new audited run"
        )
    if request.target_steps <= completed:
        raise RuntimeError("pipeline SFT resume target must exceed the completed step")
    if prior.get("training_contract_sha256") != stable_hash(
        medical_sft_contract(experiment, config)
    ):
        raise RuntimeError("pipeline SFT resume contract or data drift")
    if request.resume_state != prior.get("latest_optimizer_state"):
        raise RuntimeError("pipeline SFT resume optimizer state mismatch")


def _save_due(config: MedicalPipelineConfig, completed: int, target: int, epoch_steps: int) -> bool:
    explicit = {int(value) for value in config.get("medical_sft", "checkpoint_steps")}
    interval = int(config.get("medical_sft", "state_save_interval"))
    return (
        completed == target
        or completed in explicit
        or completed % interval == 0
        or completed % epoch_steps == 0
    )


async def _run_medical_sft_async(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: MedicalSFTRequest,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    rows, _, _ = _load_frozen_data(config)
    output = request.output_dir.resolve()
    state_path = output / "state.json"
    steps_path = output / "steps.jsonl"
    section = config.section("medical_sft")
    batch_size = int(section["batch_size"])
    seed = int(config.get("protocol", "seed"))
    epoch_steps = _steps_per_epoch(len(rows), batch_size)
    total_steps = epoch_steps * int(section["epochs"])
    if request.target_steps > total_steps:
        raise ValueError(f"pipeline SFT target exceeds three epochs ({total_steps} steps)")
    completed = 0 if prior is None else int(prior["completed_steps"])
    recoverable = 0 if prior is None else int(prior["recoverable_step"])
    latest_state = None if prior is None else str(prior["latest_optimizer_state"])
    usage = UsageLedger() if prior is None else UsageLedger.from_dict(dict(prior["usage"]))
    checkpoints = [] if prior is None else list(prior.get("checkpoints", []))

    service = trio.ServiceClient()
    models = _capability_map(service.get_server_capabilities())
    model_name = str(config.get("protocol", "base_model"))
    model = models.get(model_name)
    if model is None:
        raise RuntimeError("official 4B model is absent from workspace capabilities")
    if not bool(model["training"]["available"]):
        raise RuntimeError("official 4B training is unavailable")
    if not bool(model["sampling"]["available"]):
        raise RuntimeError("official 4B sampling is unavailable for the future teacher")
    if prior is None:
        training = await service.create_lora_training_client_async(
            base_model=model_name,
            rank=int(section["lora_rank"]),
            seed=seed,
        )
    else:
        training = await service.create_training_client_from_state_with_optimizer_async(
            request.resume_state
        )
    tokenizer = training.get_tokenizer()
    adam = trio.AdamParams(
        learning_rate=float(section["learning_rate"]),
        beta1=float(section["beta1"]),
        beta2=float(section["beta2"]),
    )
    try:
        for step in range(completed, request.target_steps):
            started = time.perf_counter()
            epoch, indices = _batch_indices(
                row_count=len(rows), batch_size=batch_size, seed=seed, step=step
            )
            batch = [rows[index] for index in indices]
            built = [
                build_teacher_sft_datum(
                    tokenizer,
                    row,
                    system=str(config.get("protocol", "medical_system")),
                    max_sequence=int(section["max_sequence_tokens"]),
                )
                for row in batch
            ]
            datums: list[TokenizedDatum] = [datum for datum, _, _ in built]
            submitted = [to_pytrio_ce_datum(trio, datum) for datum in datums]
            result_future = await training.forward_backward_async(
                submitted, loss_fn="cross_entropy"
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
                raise FloatingPointError("pipeline SFT trainer returned NaN or infinity")
            submitted_tokens = sum(len(datum.input_tokens) for datum in datums)
            trainable_tokens = sum(datum.trainable_tokens for datum in datums)
            metric_tokens = metrics.get("token_count")
            if metric_tokens is not None and metric_tokens != float(trainable_tokens):
                raise RuntimeError("pipeline SFT completion mask/token_count mismatch")
            step_usage = UsageLedger(
                student_train_tokens=submitted_tokens,
                optimizer_steps=1,
                wall_seconds=time.perf_counter() - started,
            )
            usage.add(step_usage)
            completed = step + 1
            append_jsonl(
                steps_path,
                {
                    "step": completed,
                    "epoch": epoch + 1,
                    "method": METHOD,
                    "row_ids": [row["id"] for row in batch],
                    "submitted_sequence_tokens": submitted_tokens,
                    "trainable_mask_tokens": trainable_tokens,
                    "prompt_truncated_tokens": sum(value for _, value, _ in built),
                    "completion_truncation_rows": sum(value for _, _, value in built),
                    "trainer_metrics": metrics,
                    "usage": step_usage.to_dict(),
                },
            )
            if _save_due(config, completed, request.target_steps, epoch_steps):
                state_future = await training.save_state_async(
                    name=f"{request.run_name}-step{completed:06d}-state"
                )
                latest_state = str((await state_future).path)
                weights_future = await training.save_weights_for_sampler_async(
                    name=f"{request.run_name}-step{completed:06d}-weights"
                )
                weights = str((await weights_future).path)
                recoverable = completed
                checkpoint = {
                    "step": completed,
                    "epoch_progress": completed / epoch_steps,
                    "state": latest_state,
                    "sampler_weights": weights,
                    "permanent": completed == total_steps,
                }
                checkpoints.append(checkpoint)
                append_jsonl(
                    experiment.root / "reports" / "generated" / "checkpoint_index.jsonl",
                    {"run_dir": str(output), **checkpoint},
                )
            atomic_write_json(
                state_path,
                _state_payload(
                    experiment,
                    config,
                    request,
                    status="running",
                    completed_steps=completed,
                    recoverable_step=recoverable,
                    latest_optimizer_state=latest_state,
                    usage=usage,
                    checkpoints=checkpoints,
                ),
            )
        prices_raw = fetch_json(PRICES_URL)
        summary = _state_payload(
            experiment,
            config,
            request,
            status="completed",
            completed_steps=completed,
            recoverable_step=recoverable,
            latest_optimizer_state=latest_state,
            usage=usage,
            checkpoints=checkpoints,
        )
        summary["estimated_cny"] = estimate_cost(
            usage, price_table(prices_raw), experiment
        )
        summary["price_version"] = prices_raw.get("version")
        summary["actual_billed_cny"] = None
        summary["billing_note"] = (
            "No SDK billing field; reconcile against https://pytrio.cn/usage."
        )
        summary["result_scope"] = (
            "smoke_only"
            if completed == 1
            else "cost_calibration_only" if completed == 10 else "teacher_training"
        )
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        atomic_write_json(
            state_path,
            _state_payload(
                experiment,
                config,
                request,
                status="failed",
                completed_steps=completed,
                recoverable_step=recoverable,
                latest_optimizer_state=latest_state,
                usage=usage,
                checkpoints=checkpoints,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def run_medical_sft(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: MedicalSFTRequest,
) -> dict[str, Any]:
    if not request.confirm_paid:
        raise RuntimeError("paid pipeline Medical SFT requires --confirm-paid")
    pause_path = experiment.root / "reports" / "generated" / TRAINING_PAUSE_FILENAME
    if pause_path.exists():
        raise RuntimeError(f"paid training is paused by {pause_path}")
    require_ready_preflight(experiment)
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
            raise RuntimeError("existing pipeline SFT run requires --resume-state")
        _validate_resume(experiment, config, request, prior)
    elif request.resume_state is not None:
        raise RuntimeError("pipeline SFT resume requires an existing local state")
    output.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run_medical_sft_async(experiment, config, request, prior))
