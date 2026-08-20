from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medical_opd.backend import (
    PRICES_URL,
    UsageLedger,
    estimate_cost,
    fetch_json,
    price_table,
    require_ready_preflight,
)
from medical_opd.config import ExperimentConfig
from medical_opd.contracts import (
    TokenizedDatum,
    build_ce_datum,
    build_opd_datum,
    to_pytrio_ce_datum,
    to_pytrio_opd_datum,
)
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    read_jsonl,
    safe_slug,
    sha256_file,
    stable_hash,
    utc_now,
)

METHODS = {"M2", "M3", "M4", "M5"}
TRAINING_PAUSE_FILENAME = "PAUSE_PAID_TRAINING.json"
TRAINING_PROTOCOL_ID = "general-opd-compatible-v1"
TRAINING_PROMPT_ENABLE_THINKING = False
RESUME_MIGRATION_SCHEMA_VERSION = 1
RUNNER_CODE_FILES = (
    "backend.py",
    "config.py",
    "contracts.py",
    "io_utils.py",
    "training.py",
)


@dataclass(frozen=True)
class TrainRequest:
    method: str
    target_steps: int
    output_dir: Path
    confirm_paid: bool = False
    resume_state: str | None = None
    resume_local_state: Path | None = None
    resume_migration: Path | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        if self.target_steps <= 0:
            raise ValueError("target_steps must be positive")
        if self.run_name is not None:
            safe_slug(self.run_name)
        if (self.resume_local_state is None) != (self.resume_migration is None):
            raise ValueError(
                "resume_local_state and resume_migration must be provided together"
            )
        if self.resume_local_state is not None and self.resume_state is None:
            raise ValueError("a migrated resume requires resume_state")


def _load_trio() -> Any:
    import pytrio as trio

    trio.configure(timeout=600)
    return trio


def _prompt_ids(tokenizer: Any, question: str, system: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=TRAINING_PROMPT_ENABLE_THINKING,
    )
    result = list(tokenizer.encode(text, add_special_tokens=False))
    if not result:
        raise ValueError("rendered prompt is empty")
    return result


def _completion_ids(tokenizer: Any, completion: str) -> list[int]:
    result = list(tokenizer.encode(completion, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        result.append(int(tokenizer.eos_token_id))
    if not result:
        raise ValueError("completion is empty")
    return result


def _fit_prompt(
    prompt: list[int], *, max_sequence: int, reserved_completion: int = 16
) -> tuple[list[int], int]:
    """Keep the system prefix and generation suffix when a rare prompt exceeds the P90 cap."""
    max_prompt = max_sequence - reserved_completion
    if max_prompt < 2:
        raise ValueError("max_sequence leaves no usable prompt budget")
    if len(prompt) <= max_prompt:
        return prompt, 0
    prefix = min(64, max_prompt // 4)
    fitted = prompt[:prefix] + prompt[-(max_prompt - prefix) :]
    return fitted, len(prompt) - len(fitted)


def _model_input_tokens(response: Any, fallback: int) -> int:
    value = getattr(response, "input_tokens", None)
    if isinstance(value, int) and value >= 0:
        return value
    return fallback


def _model_output_tokens(response: Any) -> int:
    value = getattr(response, "output_tokens", None)
    if isinstance(value, int) and value >= 0:
        return value
    raise RuntimeError("PyTRIO sample response is missing output_tokens")


async def _teacher_score(
    trio: Any,
    teacher: Any,
    all_ids: list[int],
    completion_start: int,
    *,
    seed: int,
) -> tuple[list[float], UsageLedger]:
    response = await teacher.sample_async(
        prompt=trio.ModelInput.from_ints(all_ids),
        num_samples=1,
        sampling_params=trio.SamplingParams(max_tokens=1, temperature=0.0, seed=seed),
        include_prompt_logprobs=True,
        return_text=False,
    )
    prompt_logprobs = list(getattr(response, "prompt_logprobs", []))
    values = prompt_logprobs[completion_start:]
    if len(values) != len(all_ids) - completion_start or any(value is None for value in values):
        raise ValueError("teacher completion logprobs are missing or token-misaligned")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("teacher completion logprobs contain NaN or infinity")
    return result, UsageLedger(
        teacher_prefill_tokens=_model_input_tokens(response, len(all_ids)),
        teacher_sample_tokens=_model_output_tokens(response),
    )


def _batch(rows: list[dict[str, Any]], step: int, size: int) -> list[dict[str, Any]]:
    start = step * size
    if start + size > len(rows):
        raise RuntimeError(f"frozen train order exhausted at step {step + 1}")
    return rows[start : start + size]


def _mixed_batch(
    medical: list[dict[str, Any]],
    general: list[dict[str, Any]],
    step: int,
    medical_size: int,
    general_size: int,
) -> list[tuple[dict[str, Any], str]]:
    med = medical[step * medical_size : (step + 1) * medical_size]
    gen = general[step * general_size : (step + 1) * general_size]
    if len(med) != medical_size or len(gen) != general_size:
        raise RuntimeError(f"frozen mixed train order exhausted at step {step + 1}")
    return [(row, "medical") for row in med] + [(row, "general") for row in gen]


def _save_due(step: int, target: int, save_steps: set[int]) -> bool:
    return step == target or step in save_steps


def _artifact_paths(config: ExperimentConfig) -> dict[str, Path]:
    return {
        "config": config.path,
        "manifest": config.root / "data" / "processed" / "manifest.json",
        "medical_train": config.root / "data" / "processed" / "train_medical.jsonl",
        "general_train": (
            config.root / "data" / "processed" / "train_general_replay.jsonl"
        ),
    }


def _artifact_hashes(config: ExperimentConfig) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in _artifact_paths(config).items()
        if path.exists()
    }


def training_contract(config: ExperimentConfig, method: str) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    lengths_path = config.root / "data" / "processed" / "lengths.json"
    if not lengths_path.exists():
        raise RuntimeError("frozen token-length report is missing")
    lengths = json.loads(lengths_path.read_text(encoding="utf-8"))
    max_sequence = int(lengths["derived_max_sequence_tokens"])
    max_completion = min(
        int(lengths["derived_max_completion_tokens"]),
        max_sequence - int(config.get("training", "minimum_max_length")) // 2,
    )
    hashes = _artifact_hashes(config)
    return {
        "protocol_id": TRAINING_PROTOCOL_ID,
        "method": method,
        "models": {
            "student": str(config.get("models", "student")),
            "teacher": (
                str(config.get("models", "teacher")) if method in {"M4", "M5"} else None
            ),
            "lora_rank": int(config.get("models", "lora_rank")),
        },
        "seed": int(config.get("experiment", "seed")),
        "batch": {
            "batch_size": int(config.get("training", "batch_size")),
            "group_size": int(config.get("training", "group_size")),
            "mixed_medical_per_step": int(
                config.get("data", "mixed_medical_per_step")
            ),
            "mixed_general_per_step": int(
                config.get("data", "mixed_general_per_step")
            ),
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": float(config.get("training", "learning_rate")),
            "beta1": float(config.get("training", "beta1")),
            "beta2": float(config.get("training", "beta2")),
        },
        "prompt": {
            "renderer_tokenizer": "student training client tokenizer",
            "enable_thinking": TRAINING_PROMPT_ENABLE_THINKING,
            "medical_system": str(config.get("training", "medical_system")),
            "general_system": str(config.get("training", "general_system")),
        },
        "sampling": {
            "temperature": float(config.get("training", "temperature")),
            "top_p": float(config.get("training", "top_p")),
            "top_k": int(config.get("training", "top_k")),
            "sequence_kd_teacher_temperature": 0.0,
            "teacher_scoring_temperature": 0.0,
        },
        "loss": {
            "name": (
                str(config.get("training", "opd_loss")) if method == "M5" else "cross_entropy"
            ),
            "kl_penalty": (
                float(config.get("training", "kl_penalty")) if method == "M5" else None
            ),
            "advantage_clip": (
                float(config.get("training", "advantage_clip")) if method == "M5" else None
            ),
        },
        "lengths": {
            "max_sequence_tokens": max_sequence,
            "max_completion_tokens": max_completion,
            "lengths_sha256": sha256_file(lengths_path),
        },
        "data_sha256": {
            key: hashes[key]
            for key in ("manifest", "medical_train", "general_train")
            if key in hashes
        },
    }


def validate_resume_migration(
    source: dict[str, Any], request: TrainRequest, config: ExperimentConfig
) -> dict[str, Any]:
    if request.resume_local_state is None or request.resume_migration is None:
        raise RuntimeError(
            "resume to a new output directory requires --resume-local-state and "
            "--resume-migration"
        )
    source_path = request.resume_local_state.resolve()
    migration_path = request.resume_migration.resolve()
    if not source_path.is_file() or not migration_path.is_file():
        raise RuntimeError("resume migration source or approval record is missing")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    if migration.get("schema_version") != RESUME_MIGRATION_SCHEMA_VERSION:
        raise RuntimeError("resume migration schema is unsupported")
    if migration.get("status") != "approved":
        raise RuntimeError("resume migration is not approved")
    recorded_source = dict(migration.get("source", {}))
    expected_source_path = (config.root / str(recorded_source.get("local_state", ""))).resolve()
    if expected_source_path != source_path:
        raise RuntimeError("resume migration source path does not match the request")
    source_sha256 = sha256_file(source_path)
    if recorded_source.get("sha256") != source_sha256:
        raise RuntimeError("resume migration source SHA256 does not match")
    if source.get("method") != request.method or recorded_source.get("method") != request.method:
        raise RuntimeError("resume migration method does not match")
    completed = int(source.get("completed_steps", -1))
    if completed < 0 or request.target_steps <= completed:
        raise RuntimeError("migrated resume target must exceed the completed step")
    if int(recorded_source.get("completed_steps", -1)) != completed:
        raise RuntimeError("resume migration completed step does not match")
    if int(source.get("seed", -1)) != int(config.get("experiment", "seed")):
        raise RuntimeError("resume migration seed does not match")
    expected_cursor = completed * int(config.get("training", "batch_size"))
    if int(source.get("sample_cursor", -1)) != expected_cursor:
        raise RuntimeError("resume migration sample cursor is inconsistent")
    checkpoints = [
        checkpoint
        for checkpoint in source.get("checkpoints", [])
        if int(checkpoint.get("step", -1)) == completed
    ]
    if len(checkpoints) != 1:
        raise RuntimeError("resume migration requires exactly one checkpoint at the source step")
    checkpoint = checkpoints[0]
    if request.resume_state != checkpoint.get("state"):
        raise RuntimeError("resume optimizer URI does not match the source checkpoint")
    if recorded_source.get("optimizer_state") != request.resume_state:
        raise RuntimeError("resume migration optimizer URI does not match")
    if recorded_source.get("sampler_weights") != checkpoint.get("sampler_weights"):
        raise RuntimeError("resume migration sampler checkpoint does not match")
    if recorded_source.get("artifact_hashes") != source.get("hashes"):
        raise RuntimeError("resume migration source artifact hashes do not match")
    current_hashes = _artifact_hashes(config)
    for name in ("manifest", "medical_train", "general_train"):
        if source.get("hashes", {}).get(name) != current_hashes.get(name):
            raise RuntimeError(f"resume migration frozen {name} hash does not match")
    if recorded_source.get("code_hashes") != source.get("code_hashes"):
        raise RuntimeError("resume migration source code hashes do not match")
    identity = dict(migration.get("model_identity", {}))
    if identity.get("student") != config.get("models", "student"):
        raise RuntimeError("resume migration student model identity does not match")
    if request.method in {"M4", "M5"} and identity.get("teacher") != config.get(
        "models", "teacher"
    ):
        raise RuntimeError("resume migration teacher model identity does not match")
    if identity.get("verification_status") != "verified":
        raise RuntimeError("resume migration model identity is not verified")
    contract = training_contract(config, request.method)
    contract_hash = stable_hash(contract)
    if migration.get("approved_training_contract") != contract:
        raise RuntimeError("resume migration training contract does not match")
    if migration.get("approved_training_contract_sha256") != contract_hash:
        raise RuntimeError("resume migration training contract SHA256 does not match")
    return {
        "schema_version": RESUME_MIGRATION_SCHEMA_VERSION,
        "source_state": str(source_path),
        "source_state_sha256": source_sha256,
        "approval_record": str(migration_path),
        "approval_record_sha256": sha256_file(migration_path),
        "training_contract_sha256": contract_hash,
    }


def record_resume_migration(
    config: ExperimentConfig,
    *,
    method: str,
    source_state_path: Path,
    output_path: Path,
    identity_evidence: str,
    allowed_differences: list[str],
    confirm_audit: bool,
) -> dict[str, Any]:
    if not confirm_audit:
        raise ValueError("refusing to approve resume migration without --confirm-audit")
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    source_path = source_state_path.resolve()
    destination = output_path.resolve()
    if not source_path.is_file():
        raise RuntimeError("resume migration source state is missing")
    if destination.exists():
        raise RuntimeError("resume migration output already exists")
    if not identity_evidence.strip():
        raise ValueError("resume migration requires model identity evidence")
    if not allowed_differences or any(not value.strip() for value in allowed_differences):
        raise ValueError("resume migration requires explicit allowed differences")
    try:
        source_relative = source_path.relative_to(config.root)
    except ValueError as exc:
        raise RuntimeError("resume migration source must be inside the project root") from exc
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "completed" or source.get("method") != method:
        raise RuntimeError("resume migration source is not a completed matching method")
    completed = int(source.get("completed_steps", -1))
    checkpoints = [
        checkpoint
        for checkpoint in source.get("checkpoints", [])
        if int(checkpoint.get("step", -1)) == completed
    ]
    if completed <= 0 or len(checkpoints) != 1:
        raise RuntimeError("resume migration source lacks one final optimizer checkpoint")
    checkpoint = checkpoints[0]
    contract = training_contract(config, method)
    code_root = config.root / "src" / "medical_opd"
    current_code_hashes = {
        name: sha256_file(code_root / name)
        for name in RUNNER_CODE_FILES
        if (code_root / name).exists()
    }
    record = {
        "schema_version": RESUME_MIGRATION_SCHEMA_VERSION,
        "status": "approved",
        "created_at": utc_now(),
        "source": {
            "local_state": str(source_relative),
            "sha256": sha256_file(source_path),
            "method": method,
            "completed_steps": completed,
            "optimizer_state": checkpoint.get("state"),
            "sampler_weights": checkpoint.get("sampler_weights"),
            "artifact_hashes": source.get("hashes"),
            "code_hashes": source.get("code_hashes"),
        },
        "model_identity": {
            "student": str(config.get("models", "student")),
            "teacher": str(config.get("models", "teacher")),
            "verification_status": "verified",
            "evidence": identity_evidence.strip(),
        },
        "approved_training_contract": contract,
        "approved_training_contract_sha256": stable_hash(contract),
        "allowed_differences": allowed_differences,
        "approval_context": {
            "current_config_sha256": sha256_file(config.path),
            "current_code_hashes": current_code_hashes,
        },
    }
    atomic_write_json(destination, record)
    return record


def _local_state(
    request: TrainRequest,
    config: ExperimentConfig,
    *,
    status: str,
    completed_steps: int,
    usage: UsageLedger,
    checkpoints: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    hashes = _artifact_hashes(config)
    code_root = config.root / "src" / "medical_opd"
    code_hashes = {
        name: sha256_file(code_root / name)
        for name in RUNNER_CODE_FILES
        if (code_root / name).exists()
    }
    payload: dict[str, Any] = {
        "status": status,
        "updated_at": utc_now(),
        "method": request.method,
        "target_steps": request.target_steps,
        "completed_steps": completed_steps,
        "next_step": completed_steps + 1,
        "sample_cursor": completed_steps * int(config.get("training", "batch_size")),
        "seed": int(config.get("experiment", "seed")),
        "usage": usage.to_dict(),
        "checkpoints": checkpoints,
        "hashes": hashes,
        "code_hashes": code_hashes,
        "resume_source": request.resume_state,
        "student_base_model": str(config.get("models", "student")),
        "teacher_base_model": str(config.get("models", "teacher")),
    }
    contract = training_contract(config, request.method)
    payload["training_contract"] = contract
    payload["training_contract_sha256"] = stable_hash(contract)
    if request.resume_local_state is not None and request.resume_migration is not None:
        payload["resume_migration"] = {
            "source_state": str(request.resume_local_state.resolve()),
            "source_state_sha256": sha256_file(request.resume_local_state.resolve()),
            "approval_record": str(request.resume_migration.resolve()),
            "approval_record_sha256": sha256_file(request.resume_migration.resolve()),
        }
    if error is not None:
        payload["error"] = error
    return payload


def validate_resume_state(
    prior: dict[str, Any], request: TrainRequest, config: ExperimentConfig
) -> None:
    if prior.get("method") != request.method:
        raise RuntimeError("resume method does not match the existing run")
    completed = int(prior.get("completed_steps", -1))
    if completed < 0 or request.target_steps < completed:
        raise RuntimeError("resume target must be at least the completed step")
    if int(prior.get("seed", -1)) != int(config.get("experiment", "seed")):
        raise RuntimeError("resume seed does not match the current config")
    expected_hashes = _artifact_hashes(config)
    prior_hashes = dict(prior.get("hashes", {}))
    if prior_hashes != expected_hashes:
        raise RuntimeError("resume hashes do not match current config/data artifacts")
    prior_code_hashes = prior.get("code_hashes")
    if prior_code_hashes is not None:
        code_root = config.root / "src" / "medical_opd"
        expected_code_hashes = {
            name: sha256_file(code_root / name)
            for name in RUNNER_CODE_FILES
            if (code_root / name).exists()
        }
        if dict(prior_code_hashes) != expected_code_hashes:
            raise RuntimeError("resume runner code hashes do not match the frozen implementation")
    expected_cursor = completed * int(config.get("training", "batch_size"))
    if int(prior.get("sample_cursor", -1)) != expected_cursor:
        raise RuntimeError("resume sample cursor is inconsistent with completed steps")
    prior_contract = prior.get("training_contract")
    if prior_contract is not None:
        expected_contract = training_contract(config, request.method)
        if prior_contract != expected_contract:
            raise RuntimeError("resume training contract does not match")
        if prior.get("training_contract_sha256") != stable_hash(expected_contract):
            raise RuntimeError("resume training contract SHA256 does not match")


def _sequence_cache_path(
    config: ExperimentConfig, row: dict[str, Any], max_completion: int
) -> Path:
    key = stable_hash(
        {
            "teacher": config.get("models", "teacher"),
            "prompt_id": row["id"],
            "seed": config.get("experiment", "seed"),
            "max_completion": max_completion,
        }
    )
    return config.root / "cache" / "sequence_kd" / f"{key}.json"


async def _m4_completion(
    trio: Any,
    teacher: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    row: dict[str, Any],
    prompt: list[int],
    *,
    max_completion: int,
    seed: int,
) -> tuple[list[int], UsageLedger, bool]:
    cache_path = _sequence_cache_path(config, row, max_completion)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("prompt_sha256") != stable_hash(prompt):
            raise RuntimeError("Sequence KD cache prompt hash mismatch")
        return [int(value) for value in cached["completion_ids"]], UsageLedger(), True
    response = await teacher.sample_async(
        prompt=trio.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=trio.SamplingParams(
            max_tokens=max_completion,
            temperature=0.0,
            seed=seed,
            stop=[value for value in (tokenizer.eos_token, "<|im_end|>") if value],
        ),
        return_text=False,
    )
    sequences = [sequence for sequence in response.sequences if sequence.tokens]
    if len(sequences) != 1:
        raise RuntimeError("Sequence KD teacher did not return exactly one non-empty completion")
    completion = [int(value) for value in sequences[0].tokens]
    atomic_write_json(
        cache_path,
        {
            "teacher_model": config.get("models", "teacher"),
            "row_id": row["id"],
            "prompt_sha256": stable_hash(prompt),
            "completion_ids": completion,
            "input_tokens": _model_input_tokens(response, len(prompt)),
            "output_tokens": _model_output_tokens(response),
            "created_at": utc_now(),
        },
    )
    return completion, UsageLedger(
        teacher_prefill_tokens=_model_input_tokens(response, len(prompt)),
        teacher_sample_tokens=_model_output_tokens(response),
    ), False


async def _build_m4_datum(
    trio: Any,
    teacher: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    row: dict[str, Any],
    *,
    max_sequence: int,
    max_completion: int,
    seed: int,
) -> tuple[TokenizedDatum, UsageLedger, bool, int]:
    prompt = _prompt_ids(
        tokenizer, str(row["question"]), str(config.get("training", "medical_system"))
    )
    prompt, removed = _fit_prompt(prompt, max_sequence=max_sequence)
    completion, usage, cache_hit = await _m4_completion(
        trio,
        teacher,
        tokenizer,
        config,
        row,
        prompt,
        max_completion=min(max_completion, max_sequence - len(prompt)),
        seed=seed,
    )
    return (
        build_ce_datum(prompt, completion, max_length=max_sequence),
        usage,
        cache_hit,
        removed,
    )


async def _build_m5_datum(
    trio: Any,
    teacher: Any,
    student_sampler: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    row: dict[str, Any],
    *,
    max_sequence: int,
    max_completion: int,
    seed: int,
) -> tuple[TokenizedDatum, list[float], UsageLedger, int]:
    prompt = _prompt_ids(
        tokenizer, str(row["question"]), str(config.get("training", "medical_system"))
    )
    prompt, removed = _fit_prompt(prompt, max_sequence=max_sequence)
    response = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=trio.SamplingParams(
            max_tokens=min(max_completion, max_sequence - len(prompt)),
            temperature=float(config.get("training", "temperature")),
            top_p=float(config.get("training", "top_p")),
            top_k=int(config.get("training", "top_k")),
            seed=seed,
            stop=[value for value in (tokenizer.eos_token, "<|im_end|>") if value],
        ),
        return_text=False,
    )
    sequences = [sequence for sequence in response.sequences if sequence.tokens]
    if len(sequences) != 1:
        raise RuntimeError("student did not return exactly one non-empty rollout")
    completion = [int(value) for value in sequences[0].tokens]
    student_logprobs = [float(value) for value in sequences[0].logprobs]
    if len(completion) != len(student_logprobs):
        raise ValueError("student rollout tokens and logprobs are misaligned")
    usage = UsageLedger(
        student_prefill_tokens=_model_input_tokens(response, len(prompt)),
        student_sample_tokens=_model_output_tokens(response),
    )
    teacher_logprobs, score_usage = await _teacher_score(
        trio, teacher, prompt + completion, len(prompt), seed=seed
    )
    usage.add(score_usage)
    datum, reverse_kl = build_opd_datum(
        prompt,
        completion,
        student_logprobs,
        teacher_logprobs,
        coefficient=float(config.get("training", "kl_penalty")),
        clip=float(config.get("training", "advantage_clip")),
        max_length=max_sequence,
    )
    return datum, reverse_kl.tolist(), usage, removed


async def _train_async(config: ExperimentConfig, request: TrainRequest) -> dict[str, Any]:
    output = request.output_dir.resolve()
    state_path = output / "state.json"
    steps_path = output / "steps.jsonl"
    if state_path.exists():
        if request.resume_local_state is not None or request.resume_migration is not None:
            raise RuntimeError("resume migration is only valid for a new output directory")
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        legacy_resume_without_code_hash = prior.get("code_hashes") is None
        validate_resume_state(prior, request, config)
        completed = int(prior.get("completed_steps", 0))
        if prior.get("status") == "completed" and completed >= request.target_steps:
            return prior
        if not request.resume_state:
            raise RuntimeError("existing incomplete run requires --resume-state")
        usage = UsageLedger.from_dict(dict(prior.get("usage", {})))
        checkpoints = list(prior.get("checkpoints", []))
    else:
        if request.resume_state:
            if request.resume_local_state is None or request.resume_migration is None:
                raise RuntimeError(
                    "resume to a new output directory requires --resume-local-state and "
                    "--resume-migration"
                )
            prior = json.loads(request.resume_local_state.resolve().read_text(encoding="utf-8"))
            validate_resume_migration(prior, request, config)
            output.mkdir(parents=True, exist_ok=False)
            completed = int(prior.get("completed_steps", 0))
            usage = UsageLedger.from_dict(dict(prior.get("usage", {})))
            checkpoints = list(prior.get("checkpoints", []))
            legacy_resume_without_code_hash = prior.get("code_hashes") is None
        else:
            output.mkdir(parents=True, exist_ok=False)
            completed = 0
            usage = UsageLedger()
            checkpoints = []
            legacy_resume_without_code_hash = False
    atomic_write_json(
        state_path,
        _local_state(
            request,
            config,
            status="running",
            completed_steps=completed,
            usage=usage,
            checkpoints=checkpoints,
        ),
    )
    trio = _load_trio()
    service = trio.ServiceClient()
    if request.resume_state:
        training = await service.create_training_client_from_state_with_optimizer_async(
            request.resume_state
        )
    else:
        training = await service.create_lora_training_client_async(
            base_model=str(config.get("models", "student")),
            rank=int(config.get("models", "lora_rank")),
            seed=int(config.get("experiment", "seed")),
        )
    tokenizer = training.get_tokenizer()
    teacher = None
    if request.method in {"M4", "M5"}:
        teacher = await service.create_sampling_client_async(
            base_model=str(config.get("models", "teacher"))
        )

    medical = read_jsonl(config.root / "data" / "processed" / "train_medical.jsonl")
    general = read_jsonl(config.root / "data" / "processed" / "train_general_replay.jsonl")
    lengths = json.loads(
        (config.root / "data" / "processed" / "lengths.json").read_text(encoding="utf-8")
    )
    max_sequence = int(lengths["derived_max_sequence_tokens"])
    max_completion = min(
        int(lengths["derived_max_completion_tokens"]),
        max_sequence - int(config.get("training", "minimum_max_length")) // 2,
    )
    batch_size = int(config.get("training", "batch_size"))
    seed = int(config.get("experiment", "seed"))
    save_steps = {int(value) for value in config.get("training", "save_steps")}
    adam = trio.AdamParams(
        learning_rate=float(config.get("training", "learning_rate")),
        beta1=float(config.get("training", "beta1")),
        beta2=float(config.get("training", "beta2")),
    )
    run_name = request.run_name or f"medical-opd-{request.method.lower()}-{request.target_steps}"
    try:
        for step in range(completed, request.target_steps):
            started = time.perf_counter()
            step_usage = UsageLedger()
            cache_hits = 0
            prompt_truncated_tokens = 0
            reverse_kls: list[float] = []
            datums: list[TokenizedDatum] = []
            if request.method == "M3":
                rows = _mixed_batch(
                    medical,
                    general,
                    step,
                    int(config.get("data", "mixed_medical_per_step")),
                    int(config.get("data", "mixed_general_per_step")),
                )
            else:
                rows = [(row, "medical") for row in _batch(medical, step, batch_size)]

            if request.method in {"M2", "M3"}:
                for row, domain in rows:
                    system = str(
                        config.get(
                            "training",
                            "medical_system" if domain == "medical" else "general_system",
                        )
                    )
                    prompt = _prompt_ids(tokenizer, str(row["question"]), system)
                    prompt, removed = _fit_prompt(
                        prompt, max_sequence=max_sequence, reserved_completion=16
                    )
                    prompt_truncated_tokens += removed
                    completion = _completion_ids(tokenizer, str(row["completion"]))
                    datums.append(
                        build_ce_datum(prompt, completion, max_length=max_sequence)
                    )
                submitted = [to_pytrio_ce_datum(trio, datum) for datum in datums]
                result_future = await training.forward_backward_async(
                    submitted, loss_fn="cross_entropy"
                )
            elif request.method == "M4":
                assert teacher is not None
                m4_results = await asyncio.gather(
                    *(
                        _build_m4_datum(
                            trio,
                            teacher,
                            tokenizer,
                            config,
                            row,
                            max_sequence=max_sequence,
                            max_completion=max_completion,
                            seed=seed + step * batch_size + offset,
                        )
                        for offset, (row, _) in enumerate(rows)
                    )
                )
                for datum, generation_usage, cache_hit, removed in m4_results:
                    datums.append(datum)
                    step_usage.add(generation_usage)
                    cache_hits += int(cache_hit)
                    prompt_truncated_tokens += removed
                submitted = [to_pytrio_ce_datum(trio, datum) for datum in datums]
                result_future = await training.forward_backward_async(
                    submitted, loss_fn="cross_entropy"
                )
            else:
                assert teacher is not None
                student_sampler = await training.save_weights_and_get_sampling_client_async()
                m5_results = await asyncio.gather(
                    *(
                        _build_m5_datum(
                            trio,
                            teacher,
                            student_sampler,
                            tokenizer,
                            config,
                            row,
                            max_sequence=max_sequence,
                            max_completion=max_completion,
                            seed=seed + step * batch_size + offset,
                        )
                        for offset, (row, _) in enumerate(rows)
                    )
                )
                for datum, reverse_kl, row_usage, removed in m5_results:
                    datums.append(datum)
                    reverse_kls.extend(reverse_kl)
                    step_usage.add(row_usage)
                    prompt_truncated_tokens += removed
                submitted = [to_pytrio_opd_datum(trio, datum) for datum in datums]
                result_future = await training.forward_backward_async(
                    submitted,
                    loss_fn=str(config.get("training", "opd_loss")),
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
                raise FloatingPointError("remote trainer returned NaN or infinity")
            submitted_tokens = sum(datum.context_tokens for datum in datums)
            trainable_tokens = sum(datum.trainable_tokens for datum in datums)
            zero_advantage_tokens = 0
            if request.method == "M5":
                zero_advantage_tokens = sum(
                    value == 0.0
                    for datum in datums
                    for value, weight in zip(
                        datum.advantages or (), datum.weights, strict=True
                    )
                    if weight != 0.0
                )
            metric_token_count = metrics.get("token_count")
            expected_metric_counts = {float(trainable_tokens)}
            if request.method == "M5":
                expected_metric_counts.add(float(trainable_tokens - zero_advantage_tokens))
            if metric_token_count is not None and metric_token_count not in expected_metric_counts:
                raise RuntimeError(
                    "remote token_count is inconsistent with the explicit completion mask"
                )
            step_usage.student_train_tokens += submitted_tokens
            step_usage.optimizer_steps = 1
            step_usage.wall_seconds = time.perf_counter() - started
            usage.add(step_usage)
            completed = step + 1
            record = {
                "step": completed,
                "method": request.method,
                "row_ids": [row["id"] for row, _ in rows],
                "submitted_sequence_tokens": submitted_tokens,
                "trainable_mask_tokens": trainable_tokens,
                "zero_advantage_completion_tokens": zero_advantage_tokens,
                "remote_token_count_expected": sorted(expected_metric_counts),
                "legacy_resume_without_code_hash": legacy_resume_without_code_hash,
                "migrated_resume": request.resume_migration is not None,
                "trainer_metrics": metrics,
                "reverse_kl_mean": None if not reverse_kls else float(np.mean(reverse_kls)),
                "reverse_kl_std": None if not reverse_kls else float(np.std(reverse_kls)),
                "sequence_kd_cache_hits": cache_hits,
                "prompt_truncated_tokens": prompt_truncated_tokens,
                "usage": step_usage.to_dict(),
            }
            append_jsonl(steps_path, record)
            if _save_due(completed, request.target_steps, save_steps):
                state_future = await training.save_state_async(
                    name=f"{run_name}-step{completed:06d}-state"
                )
                saved_state = await state_future
                weights_future = await training.save_weights_for_sampler_async(
                    name=f"{run_name}-step{completed:06d}-weights"
                )
                saved_weights = await weights_future
                checkpoint = {
                    "step": completed,
                    "state": str(saved_state.path),
                    "sampler_weights": str(saved_weights.path),
                    "permanent": request.method == "M5" and completed == 50,
                }
                checkpoints.append(checkpoint)
                append_jsonl(config.root / "reports" / "generated" / "checkpoint_index.jsonl", {
                    "run_dir": str(output), **checkpoint
                })
            atomic_write_json(
                state_path,
                _local_state(
                    request,
                    config,
                    status="running",
                    completed_steps=completed,
                    usage=usage,
                    checkpoints=checkpoints,
                ),
            )
        prices_raw = fetch_json(PRICES_URL)
        summary = _local_state(
            request,
            config,
            status="completed",
            completed_steps=completed,
            usage=usage,
            checkpoints=checkpoints,
        )
        summary["estimated_cny"] = estimate_cost(usage, price_table(prices_raw), config)
        summary["price_version"] = prices_raw.get("version")
        summary["actual_billed_cny"] = None
        summary["billing_note"] = "No SDK billing field; reconcile against https://pytrio.cn/usage."
        if completed == 1:
            summary["result_scope"] = "smoke_only"
        elif completed == 10:
            summary["result_scope"] = "cost_calibration_only"
        else:
            summary["result_scope"] = "screening_training"
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        failed = _local_state(
            request,
            config,
            status="failed",
            completed_steps=completed,
            usage=usage,
            checkpoints=checkpoints,
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(state_path, failed)
        raise


def run_training(config: ExperimentConfig, request: TrainRequest) -> dict[str, Any]:
    if not request.confirm_paid:
        raise ValueError("refusing paid remote work without --confirm-paid")
    pause_path = config.root / "reports" / "generated" / TRAINING_PAUSE_FILENAME
    if pause_path.exists():
        raise RuntimeError(
            f"paid training is paused by {pause_path}; explicit user resume is required"
        )
    manifest = config.root / "data" / "processed" / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("frozen data manifest is missing; run prepare-data first")
    require_ready_preflight(config)
    if request.method in {"M4", "M5"}:
        gate_path = config.root / "reports" / "generated" / "teacher_gate.json"
        if not gate_path.exists():
            raise RuntimeError("teacher superiority gate is missing; M4/M5 are blocked")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "passed":
            raise RuntimeError("teacher superiority gate failed; M4/M5 are blocked")
    try:
        return asyncio.run(_train_async(config, request))
    except Exception as exc:
        state_path = request.output_dir.resolve() / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "failed"
            state["updated_at"] = utc_now()
            state["error"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(state_path, state)
        raise
