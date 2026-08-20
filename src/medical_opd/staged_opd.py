from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass
from importlib.metadata import version as package_version
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
from medical_opd.contracts import TokenizedDatum, build_opd_datum, to_pytrio_opd_datum
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
from medical_opd.medical_sft import MedicalPipelineConfig
from medical_opd.reporting import exact_mcnemar
from medical_opd.training import TRAINING_PAUSE_FILENAME, _fit_prompt

STAGES = {"medical": "MED-OPD", "sar": "BASE-SAR"}
OPD_TARGET_STEPS = frozenset({1, 10, 25, 50, 100, 150, 200, 250, 300})


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: sha256_file(package / name)
        for name in (
            "contracts.py",
            "training.py",
            "medical_sft.py",
            "staged_opd.py",
        )
    }


def _tokenizer_contract_sha256(experiment: ExperimentConfig) -> str:
    path = experiment.root / "data" / "processed" / "tokenizer_compatibility.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit.pop("checked_at", None)
    return stable_hash(audit)


def _summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"evaluation summary is missing: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _prediction_contract(
    left_path: Path, right_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = read_jsonl(left_path / "predictions.jsonl")
    right = read_jsonl(right_path / "predictions.jsonl")
    left_by_id = {str(row["id"]): row for row in left}
    right_by_id = {str(row["id"]): row for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise RuntimeError("teacher gate prediction IDs must be unique")
    if left_by_id.keys() != right_by_id.keys():
        raise RuntimeError("teacher gate predictions are not paired")
    ordered_left: list[dict[str, Any]] = []
    ordered_right: list[dict[str, Any]] = []
    for row_id in sorted(left_by_id):
        left_row = left_by_id[row_id]
        right_row = right_by_id[row_id]
        for field in ("gold", "question_sha256", "prompt_token_sha256"):
            if left_row.get(field) != right_row.get(field):
                raise RuntimeError(f"teacher gate paired contract differs at {field}")
        ordered_left.append(left_row)
        ordered_right.append(right_row)
    return ordered_left, ordered_right


def _validate_eval_summary(
    experiment: ExperimentConfig,
    summary: dict[str, Any],
    *,
    dataset: str,
    raw: bool,
) -> None:
    expected_path = (
        experiment.root
        / "data"
        / "processed"
        / ("eval_medical_proxy.jsonl" if dataset == "medical_proxy" else "eval_general_proxy.jsonl")
    )
    expected_max = int(
        experiment.get(
            "evaluation",
            "thinking_medical_max_tokens"
            if dataset == "medical_proxy"
            else "thinking_general_max_tokens",
        )
    )
    if (
        summary.get("status") != "completed"
        or summary.get("dataset") != dataset
        or int(summary.get("count", -1)) != 100
        or summary.get("dataset_sha256") != sha256_file(expected_path)
        or summary.get("protocol_id") != experiment.get("evaluation", "thinking_protocol_id")
        or summary.get("prompt_contract")
        != "Chinese choice prompt rendered once with the student tokenizer"
        or summary.get("thinking") is not True
        or int(summary.get("max_tokens", -1)) != expected_max
        or float(summary.get("temperature", -1))
        != float(experiment.get("evaluation", "thinking_temperature"))
        or float(summary.get("top_p", -1))
        != float(experiment.get("evaluation", "thinking_top_p"))
        or summary.get("seed_contract") != "experiment.seed + frozen row index"
        or int(summary.get("limit", -1)) != 0
        or summary.get("base_model") != experiment.get("models", "student")
    ):
        raise RuntimeError("pipeline teacher gate evaluation contract mismatch")
    if raw and summary.get("model_path") is not None:
        raise RuntimeError("pipeline raw gate input must be the untouched official 4B")
    if not raw and not str(summary.get("model_path") or "").startswith("trio://"):
        raise RuntimeError("pipeline SFT gate input lacks sampler weights")


def select_medical_teacher(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    *,
    base_medical: Path,
    base_general: Path,
    teacher_medical: Path,
    teacher_general: Path,
    output_path: Path,
) -> dict[str, Any]:
    summaries = {
        "base_medical": _summary(base_medical),
        "base_general": _summary(base_general),
        "teacher_medical": _summary(teacher_medical),
        "teacher_general": _summary(teacher_general),
    }
    for name, summary in summaries.items():
        _validate_eval_summary(
            experiment,
            summary,
            dataset="medical_proxy" if "medical" in name else "general_proxy",
            raw=name.startswith("base"),
        )
    base_rows, teacher_rows = _prediction_contract(base_medical, teacher_medical)
    _prediction_contract(base_general, teacher_general)
    medical_delta = 100 * (
        float(summaries["teacher_medical"]["accuracy"])
        - float(summaries["base_medical"]["accuracy"])
    )
    general_delta = 100 * (
        float(summaries["teacher_general"]["accuracy"])
        - float(summaries["base_general"]["accuracy"])
    )
    base_invalid = 100 * (1 - float(summaries["base_medical"]["format_valid_rate"]))
    teacher_invalid = 100 * (1 - float(summaries["teacher_medical"]["format_valid_rate"]))
    invalid_increase = teacher_invalid - base_invalid
    passed = (
        medical_delta >= float(config.get("teacher_gate", "medical_min_delta_pp"))
        and float(summaries["teacher_medical"]["format_valid_rate"])
        >= float(config.get("teacher_gate", "minimum_format_valid_rate"))
        and invalid_increase
        <= float(config.get("teacher_gate", "maximum_invalid_increase_pp"))
    )
    result = {
        "status": "passed" if passed else "failed",
        "decision": "allow_medical_opd" if passed else "stop_before_medical_opd",
        "created_at": utc_now(),
        "protocol_id": config.get("protocol", "id"),
        "base_model": experiment.get("models", "student"),
        "teacher_model_path": summaries["teacher_medical"]["model_path"],
        "medical": {
            "base_correct": summaries["base_medical"]["correct"],
            "teacher_correct": summaries["teacher_medical"]["correct"],
            "delta_percentage_points": medical_delta,
            "teacher_format_valid_rate": summaries["teacher_medical"]["format_valid_rate"],
            "invalid_increase_percentage_points": invalid_increase,
            "paired_exact_mcnemar": exact_mcnemar(base_rows, teacher_rows),
        },
        "general_diagnostic": {
            "base_correct": summaries["base_general"]["correct"],
            "teacher_correct": summaries["teacher_general"]["correct"],
            "delta_percentage_points": general_delta,
            "note": (
                "General loss does not block a medical-only teacher; SAR exists to test recovery."
            ),
        },
        "thresholds": config.section("teacher_gate"),
        "evidence": {
            "base_medical": sha256_file(base_medical / "summary.json"),
            "base_general": sha256_file(base_general / "summary.json"),
            "teacher_medical": sha256_file(teacher_medical / "summary.json"),
            "teacher_general": sha256_file(teacher_general / "summary.json"),
        },
    }
    atomic_write_json(output_path.resolve(), result)
    return result


@dataclass(frozen=True)
class StagedOPDRequest:
    stage: str
    target_steps: int
    output_dir: Path
    teacher_model_path: str | None = None
    teacher_gate: Path | None = None
    initial_student_state: str | None = None
    initial_local_state: Path | None = None
    resume_state: str | None = None
    confirm_paid: bool = False
    run_name: str = "staged-opd"

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"pipeline OPD stage must be one of {sorted(STAGES)}")
        if self.target_steps not in OPD_TARGET_STEPS:
            allowed = ", ".join(str(step) for step in sorted(OPD_TARGET_STEPS))
            raise ValueError(f"pipeline OPD target must be one of: {allowed}")
        safe_slug(self.run_name)
        if self.stage == "medical":
            if self.teacher_model_path is None or self.teacher_gate is None:
                raise ValueError("medical OPD requires a passed SFT teacher gate and model path")
            if self.initial_student_state is not None or self.initial_local_state is not None:
                raise ValueError("medical OPD must start from a fresh official 4B")
        else:
            if self.teacher_model_path is not None or self.teacher_gate is not None:
                raise ValueError("SAR uses the untouched official 4B teacher")
            if (self.initial_student_state is None) != (self.initial_local_state is None):
                raise ValueError("SAR initial optimizer and local state must be provided together")


def _stage_section(config: MedicalPipelineConfig, stage: str) -> dict[str, Any]:
    return config.section("medical_opd" if stage == "medical" else "sar")


def _stage_rows(config: MedicalPipelineConfig, stage: str) -> tuple[Path, list[dict[str, Any]]]:
    path = config.resolve(
        "protocol", "medical_train_path" if stage == "medical" else "general_replay_path"
    )
    rows = read_jsonl(path)
    random.Random(int(config.get("protocol", "seed"))).shuffle(rows)
    if not rows:
        raise RuntimeError(f"pipeline {stage} prompt data is empty")
    return path, rows


def _validate_teacher_gate(request: StagedOPDRequest) -> dict[str, Any]:
    if request.teacher_gate is None:
        raise RuntimeError("medical OPD teacher gate is missing")
    path = request.teacher_gate.resolve()
    if not path.exists():
        raise RuntimeError("medical OPD teacher gate artifact is missing")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "passed"
        or gate.get("decision") != "allow_medical_opd"
        or gate.get("teacher_model_path") != request.teacher_model_path
    ):
        raise RuntimeError("medical OPD teacher gate did not pass for this checkpoint")
    return gate


def _validate_sar_source(request: StagedOPDRequest) -> dict[str, Any]:
    if request.initial_student_state is None or request.initial_local_state is None:
        raise RuntimeError("new SAR stage requires the Medical OPD optimizer state")
    path = request.initial_local_state.resolve()
    if not path.exists():
        raise RuntimeError("SAR Medical OPD local state is missing")
    source = json.loads(path.read_text(encoding="utf-8"))
    matching = [
        checkpoint
        for checkpoint in source.get("checkpoints", [])
        if checkpoint.get("state") == request.initial_student_state
    ]
    if (
        source.get("status") != "completed"
        or source.get("method") != STAGES["medical"]
        or int(source.get("completed_steps", -1)) < 25
        or len(matching) != 1
    ):
        raise RuntimeError("SAR source is not a completed Medical OPD screen checkpoint")
    return source


def _prompt_ids(tokenizer: Any, question: str, system: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError("pipeline OPD prompt is empty")
    return ids


def staged_opd_contract(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: StagedOPDRequest,
) -> dict[str, Any]:
    section = _stage_section(config, request.stage)
    data_path, _ = _stage_rows(config, request.stage)
    provenance: dict[str, Any]
    if request.stage == "medical":
        gate = _validate_teacher_gate(request)
        provenance = {
            "teacher_gate_sha256": sha256_file(request.teacher_gate.resolve()),
            "teacher_gate_medical": gate["medical"],
        }
    else:
        source = _validate_sar_source(request)
        provenance = {
            "initial_student_state": request.initial_student_state,
            "initial_local_state_sha256": sha256_file(request.initial_local_state.resolve()),
            "source_training_contract_sha256": source["training_contract_sha256"],
        }
    return {
        "protocol_id": config.get("protocol", "id"),
        "implementation_revision": config.get("protocol", "implementation_revision"),
        "method": STAGES[request.stage],
        "stage": request.stage,
        "student_base_model": experiment.get("models", "student"),
        "teacher_base_model": experiment.get("models", "student"),
        "teacher_model_path": request.teacher_model_path,
        "pytrio_version": package_version("pytrio"),
        "data_sha256": sha256_file(data_path),
        "data_path": str(data_path.relative_to(experiment.root)),
        "seed": config.get("protocol", "seed"),
        "system": (
            config.get("protocol", "medical_system")
            if request.stage == "medical"
            else experiment.get("training", "general_system")
        ),
        "section": section,
        "pipeline_stage_config_sha256": stable_hash(section),
        "implementation_sha256": _implementation_hashes(),
        "tokenizer_audit_sha256": _tokenizer_contract_sha256(experiment),
        "c_eval_training_rows": 0,
        "provenance": provenance,
    }


def plan_staged_opd(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: StagedOPDRequest,
) -> dict[str, Any]:
    section = _stage_section(config, request.stage)
    _, rows = _stage_rows(config, request.stage)
    batch_size = int(section["batch_size"])
    required = request.target_steps * batch_size
    if required > len(rows):
        raise RuntimeError("pipeline OPD frozen prompt order is too short")
    tokenizer = _load_student_tokenizer(experiment)
    max_sequence = int(section["max_sequence_tokens"])
    max_completion = int(section["max_completion_tokens"])
    system = (
        str(config.get("protocol", "medical_system"))
        if request.stage == "medical"
        else str(experiment.get("training", "general_system"))
    )
    group = int(section["group_size"])
    prompt_lengths: list[int] = []
    for row in rows[:required]:
        prompt = _prompt_ids(tokenizer, str(row["question"]), system)
        prompt, _ = _fit_prompt(
            prompt, max_sequence=max_sequence, reserved_completion=max_completion
        )
        prompt_lengths.append(len(prompt))
    usage = UsageLedger(optimizer_steps=request.target_steps)
    for prompt_length in prompt_lengths:
        usage.student_prefill_tokens += prompt_length
        usage.student_sample_tokens += group * max_completion
        usage.student_prefill_tokens += group * (prompt_length + max_completion)
        usage.student_sample_tokens += group
        usage.student_train_tokens += group * (prompt_length + max_completion - 1)
    prices_raw = fetch_json(PRICES_URL)
    plan = {
        "mode": "remote_paid_4b_staged_opd",
        "method": STAGES[request.stage],
        "stage": request.stage,
        "student_model": experiment.get("models", "student"),
        "teacher_model": experiment.get("models", "student"),
        "teacher_model_path": request.teacher_model_path,
        "training_mode": "fresh student" if request.stage == "medical" else "resume Medical OPD",
        "target_optimizer_steps": request.target_steps,
        "prompts": required,
        "batch_size": batch_size,
        "group_size": group,
        "max_completion_tokens": max_completion,
        "max_sequence_tokens": max_sequence,
        "output_dir": str(request.output_dir.resolve()),
        "upper_bound_usage": usage.to_dict(),
        "upper_bound_estimated_cny": estimate_cost(
            usage, price_table(prices_raw), experiment
        ),
        "price_version": prices_raw.get("version"),
        "training_contract": staged_opd_contract(experiment, config, request),
        "training_contract_sha256": stable_hash(
            staged_opd_contract(experiment, config, request)
        ),
        "success_criterion": (
            "four aligned rollouts per prompt; finite teacher/student logprobs, KL and loss; "
            "exact completion mask; optimizer and sampler checkpoints saved"
        ),
    }
    atomic_write_json(
        experiment.root
        / "reports"
        / "generated"
        / f"pipeline_{request.stage}_opd_plan_step{request.target_steps:03d}.json",
        plan,
    )
    return plan


def _usage_value(response: Any, name: str, fallback: int) -> int:
    value = getattr(response, name, None)
    return value if isinstance(value, int) and value >= 0 else fallback


async def _score_4b_completion(
    trio: Any,
    teacher: Any,
    prompt: list[int],
    completion: list[int],
    *,
    seed: int,
) -> tuple[list[float], UsageLedger]:
    all_ids = prompt + completion
    response = await teacher.sample_async(
        prompt=trio.ModelInput.from_ints(all_ids),
        num_samples=1,
        sampling_params=trio.SamplingParams(max_tokens=1, temperature=0.0, seed=seed),
        include_prompt_logprobs=True,
        return_text=False,
    )
    all_logprobs = list(getattr(response, "prompt_logprobs", []))
    values = all_logprobs[len(prompt) :]
    if len(values) != len(completion) or any(value is None for value in values):
        raise ValueError("pipeline teacher logprobs are missing or token-misaligned")
    logprobs = [float(value) for value in values]
    if any(not math.isfinite(value) for value in logprobs):
        raise ValueError("pipeline teacher logprobs contain NaN or infinity")
    return logprobs, UsageLedger(
        student_prefill_tokens=_usage_value(response, "input_tokens", len(all_ids)),
        student_sample_tokens=_usage_value(response, "output_tokens", 1),
    )


async def _rollout_prompt(
    trio: Any,
    student_sampler: Any,
    teacher: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    system: str,
    section: dict[str, Any],
    seed: int,
) -> tuple[list[TokenizedDatum], list[float], UsageLedger, list[int], int]:
    max_sequence = int(section["max_sequence_tokens"])
    max_completion = int(section["max_completion_tokens"])
    prompt = _prompt_ids(tokenizer, str(row["question"]), system)
    prompt, removed = _fit_prompt(
        prompt, max_sequence=max_sequence, reserved_completion=max_completion
    )
    group_size = int(section["group_size"])
    response = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt),
        num_samples=group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=max_completion,
            temperature=float(section["temperature"]),
            top_p=float(section["top_p"]),
            top_k=int(section["top_k"]),
            seed=seed,
            stop=[value for value in (tokenizer.eos_token, "<|im_end|>") if value],
        ),
        return_text=False,
    )
    sequences = list(response.sequences)
    if len(sequences) != group_size or any(not sequence.tokens for sequence in sequences):
        raise RuntimeError("pipeline student did not return the requested rollout group")
    completion_lengths = [len(sequence.tokens) for sequence in sequences]
    usage = UsageLedger(
        student_prefill_tokens=_usage_value(response, "input_tokens", len(prompt)),
        student_sample_tokens=_usage_value(
            response, "output_tokens", sum(completion_lengths)
        ),
    )
    scored = await asyncio.gather(
        *(
            _score_4b_completion(
                trio,
                teacher,
                prompt,
                [int(value) for value in sequence.tokens],
                seed=seed + group_index,
            )
            for group_index, sequence in enumerate(sequences)
        )
    )
    datums: list[TokenizedDatum] = []
    reverse_kls: list[float] = []
    for sequence, (teacher_logprobs, score_usage) in zip(sequences, scored, strict=True):
        completion = [int(value) for value in sequence.tokens]
        student_logprobs = [float(value) for value in sequence.logprobs]
        if len(completion) != len(student_logprobs):
            raise ValueError("pipeline student tokens and logprobs are misaligned")
        datum, reverse_kl = build_opd_datum(
            prompt,
            completion,
            student_logprobs,
            teacher_logprobs,
            coefficient=float(section["kl_penalty"]),
            clip=float(section["advantage_clip"]),
            max_length=max_sequence,
        )
        datums.append(datum)
        reverse_kls.extend(reverse_kl.tolist())
        usage.add(score_usage)
    return datums, reverse_kls, usage, completion_lengths, removed


def _runner_state(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: StagedOPDRequest,
    *,
    status: str,
    completed_steps: int,
    recoverable_step: int,
    latest_optimizer_state: str | None,
    usage: UsageLedger,
    checkpoints: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    contract = staged_opd_contract(experiment, config, request)
    payload: dict[str, Any] = {
        "status": status,
        "method": STAGES[request.stage],
        "stage": request.stage,
        "target_steps": request.target_steps,
        "completed_steps": completed_steps,
        "recoverable_step": recoverable_step,
        "uncheckpointed_steps": completed_steps - recoverable_step,
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


def _validate_runner_resume(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: StagedOPDRequest,
    prior: dict[str, Any],
) -> None:
    if prior.get("method") != STAGES[request.stage]:
        raise RuntimeError("pipeline OPD resume method mismatch")
    completed = int(prior.get("completed_steps", -1))
    if completed != int(prior.get("recoverable_step", -2)):
        raise RuntimeError("pipeline OPD resume has an uncheckpointed tail")
    if request.target_steps <= completed:
        raise RuntimeError("pipeline OPD resume target must exceed completed steps")
    if prior.get("training_contract_sha256") != stable_hash(
        staged_opd_contract(experiment, config, request)
    ):
        raise RuntimeError("pipeline OPD resume contract or data drift")
    if request.resume_state != prior.get("latest_optimizer_state"):
        raise RuntimeError("pipeline OPD resume optimizer state mismatch")


def _capability_map(capabilities: Any) -> dict[str, dict[str, Any]]:
    if hasattr(capabilities, "model_dump"):
        payload = capabilities.model_dump(mode="json")
    elif hasattr(capabilities, "dict"):
        payload = capabilities.dict()
    else:
        raise TypeError("unsupported PyTRIO capabilities response")
    return {str(item["model_name"]): item for item in payload.get("supported_models", [])}


async def _run_staged_opd_async(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: StagedOPDRequest,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    output = request.output_dir.resolve()
    state_path = output / "state.json"
    steps_path = output / "steps.jsonl"
    section = _stage_section(config, request.stage)
    _, rows = _stage_rows(config, request.stage)
    batch_size = int(section["batch_size"])
    if request.target_steps * batch_size > len(rows):
        raise RuntimeError("pipeline OPD prompt order is too short")
    completed = 0 if prior is None else int(prior["completed_steps"])
    recoverable = 0 if prior is None else int(prior["recoverable_step"])
    latest_state = None if prior is None else str(prior["latest_optimizer_state"])
    usage = UsageLedger() if prior is None else UsageLedger.from_dict(dict(prior["usage"]))
    checkpoints = [] if prior is None else list(prior.get("checkpoints", []))
    service = trio.ServiceClient()
    model_name = str(experiment.get("models", "student"))
    model = _capability_map(service.get_server_capabilities()).get(model_name)
    if model is None or not bool(model["training"]["available"]):
        raise RuntimeError("official 4B training is unavailable for staged OPD")
    if not bool(model["sampling"]["available"]):
        raise RuntimeError("official 4B sampling is unavailable for staged OPD")
    if prior is not None:
        training = await service.create_training_client_from_state_with_optimizer_async(
            request.resume_state
        )
    elif request.stage == "sar":
        training = await service.create_training_client_from_state_with_optimizer_async(
            request.initial_student_state
        )
    else:
        training = await service.create_lora_training_client_async(
            base_model=model_name,
            rank=int(section["lora_rank"]),
            seed=int(config.get("protocol", "seed")),
        )
    teacher = await service.create_sampling_client_async(
        base_model=model_name,
        model_path=request.teacher_model_path,
    )
    tokenizer = training.get_tokenizer()
    adam = trio.AdamParams(
        learning_rate=float(section["learning_rate"]),
        beta1=float(section["beta1"]),
        beta2=float(section["beta2"]),
    )
    system = (
        str(config.get("protocol", "medical_system"))
        if request.stage == "medical"
        else str(experiment.get("training", "general_system"))
    )
    seed = int(config.get("protocol", "seed"))
    explicit = {int(value) for value in section["checkpoint_steps"]}
    interval = int(section["state_save_interval"])
    try:
        for step in range(completed, request.target_steps):
            started = time.perf_counter()
            batch = rows[step * batch_size : (step + 1) * batch_size]
            student_sampler = await training.save_weights_and_get_sampling_client_async()
            rollouts = await asyncio.gather(
                *(
                    _rollout_prompt(
                        trio,
                        student_sampler,
                        teacher,
                        tokenizer,
                        row,
                        system=system,
                        section=section,
                        seed=seed + step * batch_size + offset,
                    )
                    for offset, row in enumerate(batch)
                )
            )
            datums = [datum for result in rollouts for datum in result[0]]
            reverse_kls = [value for result in rollouts for value in result[1]]
            completion_lengths = [value for result in rollouts for value in result[3]]
            step_usage = UsageLedger()
            for result in rollouts:
                step_usage.add(result[2])
            submitted = [to_pytrio_opd_datum(trio, datum) for datum in datums]
            result_future = await training.forward_backward_async(
                submitted, loss_fn=str(section["loss_fn"])
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
                raise FloatingPointError("pipeline OPD trainer returned NaN or infinity")
            submitted_tokens = sum(datum.context_tokens for datum in datums)
            trainable_tokens = sum(datum.trainable_tokens for datum in datums)
            zero_advantage = sum(
                value == 0.0
                for datum in datums
                for value, weight in zip(datum.advantages or (), datum.weights, strict=True)
                if weight != 0.0
            )
            expected_counts = {float(trainable_tokens), float(trainable_tokens - zero_advantage)}
            metric_tokens = metrics.get("token_count")
            if metric_tokens is not None and metric_tokens not in expected_counts:
                raise RuntimeError("pipeline OPD completion mask/token_count mismatch")
            step_usage.student_train_tokens += submitted_tokens
            step_usage.optimizer_steps = 1
            step_usage.wall_seconds = time.perf_counter() - started
            usage.add(step_usage)
            completed = step + 1
            append_jsonl(
                steps_path,
                {
                    "step": completed,
                    "method": STAGES[request.stage],
                    "row_ids": [row["id"] for row in batch],
                    "datums": len(datums),
                    "group_size": int(section["group_size"]),
                    "completion_tokens": sum(completion_lengths),
                    "completion_tokens_mean": float(np.mean(completion_lengths)),
                    "submitted_sequence_tokens": submitted_tokens,
                    "trainable_mask_tokens": trainable_tokens,
                    "zero_advantage_completion_tokens": zero_advantage,
                    "reverse_kl_mean": float(np.mean(reverse_kls)),
                    "reverse_kl_std": float(np.std(reverse_kls)),
                    "prompt_truncated_tokens": sum(result[4] for result in rollouts),
                    "trainer_metrics": metrics,
                    "usage": step_usage.to_dict(),
                },
            )
            save_due = (
                completed == request.target_steps
                or completed in explicit
                or completed % interval == 0
            )
            if save_due:
                state_future = await training.save_state_async(
                    name=f"{request.run_name}-{request.stage}-step{completed:06d}-state"
                )
                latest_state = str((await state_future).path)
                weights_future = await training.save_weights_for_sampler_async(
                    name=f"{request.run_name}-{request.stage}-step{completed:06d}-weights"
                )
                weights = str((await weights_future).path)
                recoverable = completed
                checkpoint = {
                    "step": completed,
                    "state": latest_state,
                    "sampler_weights": weights,
                    "permanent": request.stage == "medical" and completed == 25,
                }
                checkpoints.append(checkpoint)
                append_jsonl(
                    experiment.root / "reports" / "generated" / "checkpoint_index.jsonl",
                    {"run_dir": str(output), **checkpoint},
                )
            atomic_write_json(
                state_path,
                _runner_state(
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
        summary = _runner_state(
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
            else "cost_calibration_only" if completed == 10 else "proxy_screen_training"
        )
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        atomic_write_json(
            state_path,
            _runner_state(
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


def run_staged_opd(
    experiment: ExperimentConfig,
    config: MedicalPipelineConfig,
    request: StagedOPDRequest,
) -> dict[str, Any]:
    if not request.confirm_paid:
        raise RuntimeError("paid pipeline staged OPD requires --confirm-paid")
    pause_path = experiment.root / "reports" / "generated" / TRAINING_PAUSE_FILENAME
    if pause_path.exists():
        raise RuntimeError(f"paid training is paused by {pause_path}")
    require_ready_preflight(experiment)
    if request.stage == "medical":
        _validate_teacher_gate(request)
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
            raise RuntimeError("existing pipeline OPD run requires --resume-state")
        _validate_runner_resume(experiment, config, request, prior)
    elif request.resume_state is not None:
        raise RuntimeError("pipeline OPD resume requires an existing local state")
    elif request.stage == "sar":
        _validate_sar_source(request)
    output.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run_staged_opd_async(experiment, config, request, prior))
