from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
import tomllib
from collections import Counter
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
from medical_opd.contracts import TokenizedDatum, build_opd_datum, to_pytrio_opd_datum
from medical_opd.data import _choice_row, _deduplicate_rows
from medical_opd.evaluation import _load_student_tokenizer
from medical_opd.io_utils import (
    append_jsonl,
    atomic_write_json,
    distribution,
    read_jsonl,
    safe_slug,
    sha256_file,
    stable_hash,
    utc_now,
    write_jsonl,
)
from medical_opd.staged_opd import (
    _capability_map,
    _prompt_ids,
    _score_4b_completion,
    _usage_value,
)
from medical_opd.training import TRAINING_PAUSE_FILENAME, _fit_prompt

DEFAULT_CEVAL_SAR_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "base-anchor-sar.toml"
)
METHOD = "BASE-SAR"
TARGET_STEPS = {1, 10, 25, 50, 60, 100, 150, 200, 250, 300}
PERMANENT_STEPS = {50, 100, 150, 200, 250, 300}
LEGACY_STEP50_IMPLEMENTATION_SHA256 = (
    "7c4e7491126cb50a59a4af462cb0627598dd902ddb51db7eaae65d117cfe1f0d"
)
CONTINUATION_CONTRACT = {
    "authorized_scope": (
        "frozen SAR checkpoints; continuation beyond step50 requires explicit "
        "authorization after the fixed evaluation gate"
    ),
    "maximum_target_steps": 300,
    "schedule_mode": "repeat the frozen 200-presentation schedule without reshuffling",
    "evaluation_steps": [50, 100, 150, 200, 250, 300],
}
SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)
FORBIDDEN_TRAINING_KEYS = {
    "answer",
    "answer_idx",
    "answer_index",
    "completion",
    "gold",
    "label",
    "options",
    "response",
}


@dataclass(frozen=True)
class CevalSARConfig:
    path: Path
    raw: dict[str, Any]
    root: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing C-Eval SAR config section: {name}")
        return value

    def get(self, section: str, key: str) -> Any:
        values = self.section(section)
        if key not in values:
            raise ValueError(f"missing C-Eval SAR config value: {section}.{key}")
        return values[key]

    def resolve(self, section: str, key: str) -> Path:
        path = (self.root / str(self.get(section, key))).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"C-Eval SAR path escapes project root: {section}.{key}") from exc
        return path


def load_ceval_sar_config(experiment: ExperimentConfig, path: Path | None = None) -> CevalSARConfig:
    config_path = (path or DEFAULT_CEVAL_SAR_CONFIG).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = CevalSARConfig(config_path, raw, experiment.root)
    _validate_ceval_sar_config(experiment, config)
    return config


def _validate_ceval_sar_config(experiment: ExperimentConfig, config: CevalSARConfig) -> None:
    config.section("protocol")
    config.section("training")
    if config.get("protocol", "id") != "base-anchor-sar-v1":
        raise ValueError("C-Eval SAR protocol id drift")
    implementation_revision = str(config.get("protocol", "implementation_revision"))
    if re.fullmatch(r"[0-9a-f]{40}", implementation_revision) is None:
        raise ValueError("C-Eval SAR implementation_revision must be a pinned 40-hex commit")
    if config.get("protocol", "base_model") != experiment.get("models", "student"):
        raise ValueError("C-Eval SAR must use the official configured 4B model")
    if int(config.get("protocol", "seed")) != int(experiment.get("experiment", "seed")):
        raise ValueError("C-Eval SAR seed must match the frozen experiment seed")
    if list(config.get("protocol", "source_splits")) != ["dev", "val"]:
        raise ValueError("C-Eval SAR may use only dev and non-proxy val rows")
    if str(config.get("protocol", "system_message")) != SYSTEM_MESSAGE:
        raise ValueError("C-Eval SAR system message drifted from the frozen prompt contract")
    training = config.section("training")
    expected: dict[str, Any] = {
        "method": METHOD,
        "screen_steps": 50,
        "batch_size": 4,
        "group_size": 4,
        "enable_thinking": True,
        "max_completion_tokens": 2048,
        "max_sequence_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "learning_rate": 5e-6,
        "beta1": 0.9,
        "beta2": 0.95,
        "loss_fn": "ppo",
        "kl_penalty": 1.0,
        "advantage_clip": 20.0,
        "state_save_interval": 5,
        "checkpoint_steps": [0, 1, 10, 25, 50],
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"C-Eval SAR training.{key} must remain {value!r}")
    presentations = int(config.get("protocol", "presentation_count"))
    if presentations != int(training["screen_steps"]) * int(training["batch_size"]):
        raise ValueError("C-Eval SAR presentation count must cover exactly 50 batches")
    for key in ("data_path", "manifest_path", "quarantine_path"):
        config.resolve("protocol", key)


def format_ceval_prompt(row: dict[str, Any]) -> str:
    options = row.get("options")
    if not isinstance(options, dict):
        raise ValueError("C-Eval row options must be an object")
    values = {key: str(options.get(key, "")).strip() for key in ("A", "B", "C", "D")}
    question = str(row.get("question", "")).strip()
    if not question or not all(values.values()):
        raise ValueError("C-Eval row is missing a question or A/B/C/D option")
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{question}",
            f"A. {values['A']}",
            f"B. {values['B']}",
            f"C. {values['C']}",
            f"D. {values['D']}",
            "",
            "答案：",
        ]
    )


def _load_ceval_candidates(
    experiment: ExperimentConfig, *, dataset_cache: Path
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for split in ("dev", "val"):
        for subject in experiment.get("data", "ceval_subjects"):
            raw_rows = load_dataset(
                str(experiment.get("data", "ceval_repo")),
                str(subject),
                split=split,
                cache_dir=str(dataset_cache),
                revision=str(experiment.get("data", "ceval_labeled_revision")),
            )
            for source_index, raw in enumerate(raw_rows):
                normalized = _choice_row(
                    dict(raw),
                    source_dataset=str(experiment.get("data", "ceval_repo")),
                    source_split=split,
                    subject=str(subject),
                    source_index=source_index,
                )
                if normalized is not None:
                    rows.append(normalized)
    return rows


def _build_presentations(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("leakage-safe C-Eval SAR source pool is empty")
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    presentations: list[dict[str, Any]] = []
    for index in range(count):
        source = ordered[index % len(ordered)]
        presentations.append(
            {
                "id": f"ceval-sar-presentation-{index:06d}",
                "source_row_id": source["id"],
                "source_dataset": source["source_dataset"],
                "source_split": source["source_split"],
                "subject": source["subject"],
                "source_index": source["source_index"],
                "cycle": index // len(ordered),
                "question": format_ceval_prompt(source),
            }
        )
    return presentations


def _data_paths(config: CevalSARConfig) -> dict[str, Path]:
    return {
        "data": config.resolve("protocol", "data_path"),
        "manifest": config.resolve("protocol", "manifest_path"),
        "quarantine": config.resolve("protocol", "quarantine_path"),
    }


def _tokenizer_contract_sha256(experiment: ExperimentConfig) -> str:
    path = experiment.root / "data" / "processed" / "tokenizer_compatibility.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit.pop("checked_at", None)
    return stable_hash(audit)


def prepare_ceval_sar_data(
    experiment: ExperimentConfig,
    config: CevalSARConfig,
    *,
    shared_cache: Path,
) -> dict[str, Any]:
    processed = experiment.root / "data" / "processed"
    proxy_path = processed / "eval_general_proxy.jsonl"
    full_path = processed / "eval_general_full.jsonl"
    if not proxy_path.exists() or not full_path.exists():
        raise RuntimeError("freeze the C-Eval proxy and full test before preparing SAR prompts")
    proxy = read_jsonl(proxy_path)
    full = read_jsonl(full_path)
    if len(proxy) != 100 or any(row.get("source_split") != "val" for row in proxy):
        raise RuntimeError("frozen C-Eval proxy must be exactly 100 val rows")
    if not full or any(row.get("source_split") != "test" for row in full):
        raise RuntimeError("frozen C-Eval full evaluation must contain only test rows")

    candidates = _load_ceval_candidates(
        experiment, dataset_cache=(shared_cache / "datasets").resolve()
    )
    heldout = proxy + full
    quarantine: list[dict[str, Any]] = []
    clean, counters = _deduplicate_rows(
        candidates,
        text_key="question",
        heldout_texts=[str(row["question"]) for row in heldout],
        heldout_ids=[str(row["id"]) for row in heldout],
        threshold=float(experiment.get("data", "near_duplicate_threshold")),
        quarantine=quarantine,
    )
    presentation_count = int(config.get("protocol", "presentation_count"))
    presentations = _build_presentations(
        clean,
        count=presentation_count,
        seed=int(config.get("protocol", "seed")),
    )
    heldout_ids = {str(row["id"]) for row in heldout}
    source_ids = {str(row["source_row_id"]) for row in presentations}
    if source_ids & heldout_ids:
        raise RuntimeError("C-Eval SAR source IDs overlap a frozen evaluation row")
    if any(FORBIDDEN_TRAINING_KEYS & set(row) for row in presentations):
        raise RuntimeError("C-Eval SAR prompt schedule contains an answer-bearing field")

    paths = _data_paths(config)
    write_jsonl(paths["data"], presentations)
    write_jsonl(paths["quarantine"], quarantine)
    tokenizer = _load_student_tokenizer(experiment)
    section = config.section("training")
    prompt_lengths: list[int] = []
    truncated_tokens = 0
    for row in presentations:
        prompt = _prompt_ids(
            tokenizer,
            str(row["question"]),
            str(config.get("protocol", "system_message")),
        )
        prompt, removed = _fit_prompt(
            prompt,
            max_sequence=int(section["max_sequence_tokens"]),
            reserved_completion=int(section["max_completion_tokens"]),
        )
        prompt_lengths.append(len(prompt))
        truncated_tokens += removed

    split_counts = Counter(str(row["source_split"]) for row in clean)
    subject_counts = Counter(str(row["subject"]) for row in clean)
    presentation_source_counts = Counter(str(row["source_row_id"]) for row in presentations)
    manifest = {
        "status": "frozen",
        "created_at": utc_now(),
        "protocol_id": config.get("protocol", "id"),
        "implementation_revision": config.get("protocol", "implementation_revision"),
        "source": {
            "repo": experiment.get("data", "ceval_repo"),
            "revision": experiment.get("data", "ceval_labeled_revision"),
            "splits_loaded": list(config.get("protocol", "source_splits")),
            "subjects": list(experiment.get("data", "ceval_subjects")),
            "input_rows": len(candidates),
            "clean_unique_rows": len(clean),
            "clean_split_counts": dict(sorted(split_counts.items())),
            "clean_subject_counts": dict(sorted(subject_counts.items())),
        },
        "leakage_filter": counters,
        "heldout": {
            "proxy_split": "val",
            "proxy_count": len(proxy),
            "proxy_sha256": sha256_file(proxy_path),
            "full_split": "test",
            "full_count": len(full),
            "full_sha256": sha256_file(full_path),
            "source_id_intersection": len(source_ids & heldout_ids),
        },
        "schedule": {
            "presentation_count": len(presentations),
            "unique_source_rows": len(presentation_source_counts),
            "repeated_presentations": len(presentations) - len(presentation_source_counts),
            "maximum_source_repetitions": max(presentation_source_counts.values()),
            "seed": config.get("protocol", "seed"),
            "answer_fields_stripped": True,
            "test_training_rows": 0,
            "proxy_training_rows": 0,
        },
        "prompt_length_audit": {
            "tokens": distribution(prompt_lengths),
            "truncated_tokens": truncated_tokens,
            "max_sequence_tokens": section["max_sequence_tokens"],
            "reserved_completion_tokens": section["max_completion_tokens"],
        },
        "files": {
            "data": {"count": len(presentations), "sha256": sha256_file(paths["data"])},
            "quarantine": {
                "count": len(quarantine),
                "sha256": sha256_file(paths["quarantine"]),
            },
        },
        "tokenizer_contract_sha256": _tokenizer_contract_sha256(experiment),
        "split_contract": (
            "Document-aligned C-Eval prompt-only Base-anchor SAR with a leakage fix: use "
            "C-Eval dev plus val rows outside the frozen val100 proxy; exclude all proxy and "
            "test rows by exact/near match; strip labels before freezing the 200-presentation "
            "schedule."
        ),
        "data_governance_note": (
            "This branch preserves the frozen token-level algorithm and prompt while keeping "
            "training rows disjoint from proxy and test rows."
        ),
    }
    atomic_write_json(paths["manifest"], manifest)
    _load_frozen_ceval_sar_data(experiment, config)
    return manifest


def _load_frozen_ceval_sar_data(
    experiment: ExperimentConfig, config: CevalSARConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _data_paths(config)
    for path in paths.values():
        if not path.exists():
            raise RuntimeError(f"C-Eval SAR frozen artifact is missing: {path}")
    rows = read_jsonl(paths["data"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    proxy_path = experiment.root / "data" / "processed" / "eval_general_proxy.jsonl"
    full_path = experiment.root / "data" / "processed" / "eval_general_full.jsonl"
    forbidden = [row["id"] for row in rows if FORBIDDEN_TRAINING_KEYS & set(row)]
    ids = [str(row.get("id", "")) for row in rows]
    source_splits = {str(row.get("source_split", "")) for row in rows}
    heldout_ids = {str(row["id"]) for row in read_jsonl(proxy_path) + read_jsonl(full_path)}
    source_ids = {str(row.get("source_row_id", "")) for row in rows}
    expected_count = int(config.get("protocol", "presentation_count"))
    if len(rows) != expected_count or len(set(ids)) != expected_count or "" in ids:
        raise RuntimeError("C-Eval SAR frozen schedule count or presentation IDs drift")
    if forbidden:
        raise RuntimeError(f"C-Eval SAR frozen schedule exposes answer fields: {forbidden[:3]}")
    if not source_splits or not source_splits <= {"dev", "val"}:
        raise RuntimeError("C-Eval SAR frozen schedule contains a forbidden source split")
    if source_ids & heldout_ids:
        raise RuntimeError("C-Eval SAR frozen schedule overlaps proxy/test IDs")
    schedule = manifest.get("schedule", {})
    heldout = manifest.get("heldout", {})
    source = manifest.get("source", {})
    files = manifest.get("files", {})
    if (
        manifest.get("status") != "frozen"
        or manifest.get("protocol_id") != config.get("protocol", "id")
        or source.get("revision") != experiment.get("data", "ceval_labeled_revision")
        or source.get("splits_loaded") != ["dev", "val"]
        or schedule.get("presentation_count") != expected_count
        or schedule.get("answer_fields_stripped") is not True
        or schedule.get("test_training_rows") != 0
        or schedule.get("proxy_training_rows") != 0
        or heldout.get("proxy_sha256") != sha256_file(proxy_path)
        or heldout.get("full_sha256") != sha256_file(full_path)
        or heldout.get("source_id_intersection") != 0
        or files.get("data", {}).get("sha256") != sha256_file(paths["data"])
        or files.get("data", {}).get("count") != len(rows)
        or files.get("quarantine", {}).get("sha256") != sha256_file(paths["quarantine"])
        or manifest.get("tokenizer_contract_sha256") != _tokenizer_contract_sha256(experiment)
    ):
        raise RuntimeError("C-Eval SAR frozen manifest or dependency drift")
    return rows, manifest


@dataclass
class RoleUsage:
    student_rollout_prefill_tokens: int = 0
    student_rollout_sample_tokens: int = 0
    base_teacher_scoring_prefill_tokens: int = 0
    base_teacher_scoring_sample_tokens: int = 0
    student_train_tokens: int = 0

    def add(self, other: RoleUsage) -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def to_dict(self) -> dict[str, int]:
        return {field: int(getattr(self, field)) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RoleUsage:
        return cls(**{field: int(raw.get(field, 0)) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CevalSARRequest:
    target_steps: int
    output_dir: Path
    initial_student_state: str
    initial_local_state: Path
    resume_state: str | None = None
    confirm_paid: bool = False
    run_name: str = "base-anchor-sar"

    def __post_init__(self) -> None:
        if self.target_steps not in TARGET_STEPS:
            raise ValueError("C-Eval SAR target must be 1, 10, 25, 50, 60, 100, 150, or 200")
        if "/sampler_weights/" in self.initial_student_state:
            raise ValueError(
                "C-Eval SAR initial student must be a Train state, not sampler weights"
            )
        safe_slug(self.run_name)


def _validate_source(request: CevalSARRequest) -> dict[str, Any]:
    path = request.initial_local_state.resolve()
    if not path.exists():
        raise RuntimeError("C-Eval SAR Medical OPD source state is missing")
    source = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = [
        checkpoint
        for checkpoint in source.get("checkpoints", [])
        if checkpoint.get("state") == request.initial_student_state
    ]
    contract = source.get("training_contract", {})
    source_step = int(source.get("completed_steps", -1))
    if (
        source.get("status") != "completed"
        or source.get("method") != "MED-OPD"
        or source_step <= 0
        or int(source.get("recoverable_step", -1)) != source_step
        or int(source.get("uncheckpointed_steps", -1)) != 0
        or source.get("latest_optimizer_state") != request.initial_student_state
        or len(checkpoints) != 1
        or int(checkpoints[0].get("step", -1)) != source_step
        or checkpoints[0].get("permanent") is not True
        or contract.get("student_base_model") != "Qwen/Qwen3.5-4B"
    ):
        raise RuntimeError(
            "C-Eval SAR source must be a permanent, complete, fully checkpointed "
            "Medical OPD final state"
        )
    return source


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: sha256_file(package / name)
        for name in ("ceval_sar.py", "contracts.py", "training.py", "staged_opd.py")
    }


def ceval_sar_contract(
    experiment: ExperimentConfig,
    config: CevalSARConfig,
    request: CevalSARRequest,
) -> dict[str, Any]:
    _, manifest = _load_frozen_ceval_sar_data(experiment, config)
    source = _validate_source(request)
    paths = _data_paths(config)
    return {
        "protocol_id": config.get("protocol", "id"),
        "implementation_revision": config.get("protocol", "implementation_revision"),
        "method": METHOD,
        "student_base_model": experiment.get("models", "student"),
        "teacher_base_model": experiment.get("models", "student"),
        "teacher_model_path": None,
        "system": config.get("protocol", "system_message"),
        "section": config.section("training"),
        "stage_config_sha256": stable_hash(config.raw),
        "data_sha256": sha256_file(paths["data"]),
        "data_manifest_sha256": sha256_file(paths["manifest"]),
        "quarantine_sha256": sha256_file(paths["quarantine"]),
        "data_manifest_identity": stable_hash(
            {
                "source": manifest["source"],
                "heldout": manifest["heldout"],
                "schedule": manifest["schedule"],
            }
        ),
        "tokenizer_contract_sha256": _tokenizer_contract_sha256(experiment),
        "implementation_sha256": _implementation_hashes(),
        "source_medical_opd_state": request.initial_student_state,
        "source_medical_opd_local_state_sha256": sha256_file(request.initial_local_state.resolve()),
        "source_medical_opd_contract_sha256": source["training_contract_sha256"],
        "source_medical_opd_step": int(source["completed_steps"]),
        "optimizer_inherited_from_medical_opd": False,
        "state_clone_contract": (
            "Load the Medical OPD training state weights-only, save a source copy, load that "
            "copy weights-only into the SAR client, then save step000000 before any update."
        ),
        "c_eval_training_splits": ["dev", "val-outside-proxy"],
        "c_eval_proxy_training_rows": 0,
        "c_eval_test_training_rows": 0,
        "continuation": CONTINUATION_CONTRACT,
    }


def _batch_for_step(rows: list[dict[str, Any]], step: int, batch_size: int) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("C-Eval SAR frozen presentation schedule is empty")
    start = step * batch_size
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def plan_ceval_sar(
    experiment: ExperimentConfig,
    config: CevalSARConfig,
    request: CevalSARRequest,
) -> dict[str, Any]:
    rows, manifest = _load_frozen_ceval_sar_data(experiment, config)
    _validate_source(request)
    section = config.section("training")
    batch_size = int(section["batch_size"])
    completed = 0
    state_path = request.output_dir.resolve() / "state.json"
    if state_path.exists():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("method") != METHOD:
            raise RuntimeError("C-Eval SAR plan found a different method in output state")
        completed = int(prior.get("completed_steps", -1))
        if completed < 0 or completed >= request.target_steps:
            raise RuntimeError("C-Eval SAR plan target must exceed the completed step")
    selected_rows = [
        row
        for step in range(completed, request.target_steps)
        for row in _batch_for_step(rows, step, batch_size)
    ]
    tokenizer = _load_student_tokenizer(experiment)
    lengths: list[int] = []
    for row in selected_rows:
        prompt = _prompt_ids(
            tokenizer,
            str(row["question"]),
            str(config.get("protocol", "system_message")),
        )
        prompt, _ = _fit_prompt(
            prompt,
            max_sequence=int(section["max_sequence_tokens"]),
            reserved_completion=int(section["max_completion_tokens"]),
        )
        lengths.append(len(prompt))
    group = int(section["group_size"])
    cap = int(section["max_completion_tokens"])
    usage = UsageLedger(optimizer_steps=request.target_steps - completed)
    roles = RoleUsage()
    for length in lengths:
        roles.student_rollout_prefill_tokens += length
        roles.student_rollout_sample_tokens += group * cap
        roles.base_teacher_scoring_prefill_tokens += group * (length + cap)
        roles.base_teacher_scoring_sample_tokens += group
        roles.student_train_tokens += group * (length + cap - 1)
    usage.student_prefill_tokens = (
        roles.student_rollout_prefill_tokens + roles.base_teacher_scoring_prefill_tokens
    )
    usage.student_sample_tokens = (
        roles.student_rollout_sample_tokens + roles.base_teacher_scoring_sample_tokens
    )
    usage.student_train_tokens = roles.student_train_tokens
    prices_raw = fetch_json(PRICES_URL)
    contract = ceval_sar_contract(experiment, config, request)
    plan = {
        "mode": "remote_paid_4b_ceval_base_anchor_sar",
        "method": METHOD,
        "student_model": experiment.get("models", "student"),
        "teacher_model": experiment.get("models", "student"),
        "teacher_model_path": None,
        "training_mode": (
            f"Medical OPD@{int(contract['source_medical_opd_step'])} weights clone; "
            "fresh SAR optimizer"
        ),
        "resume_from_step": completed,
        "target_optimizer_steps": request.target_steps,
        "incremental_optimizer_steps": request.target_steps - completed,
        "prompts": len(selected_rows),
        "frozen_presentations": len(rows),
        "schedule_mode": CONTINUATION_CONTRACT["schedule_mode"],
        "schedule_cycles_at_target": math.ceil(request.target_steps * batch_size / len(rows)),
        "unique_source_rows": manifest["schedule"]["unique_source_rows"],
        "batch_size": section["batch_size"],
        "group_size": section["group_size"],
        "max_completion_tokens": cap,
        "max_sequence_tokens": section["max_sequence_tokens"],
        "output_dir": str(request.output_dir.resolve()),
        "upper_bound_usage": usage.to_dict(),
        "upper_bound_role_usage": roles.to_dict(),
        "upper_bound_estimated_cny": estimate_cost(usage, price_table(prices_raw), experiment),
        "price_version": prices_raw.get("version"),
        "training_contract": contract,
        "training_contract_sha256": stable_hash(contract),
        "success_criterion": (
            "step000000 weight clone before updates; four aligned rollouts per prompt; "
            "finite teacher/student logprobs, reverse KL and loss; exact completion mask; "
            "resumable SAR optimizer plus sampler checkpoints"
        ),
        "effect_scope": (
            "1=smoke, 10=cost calibration, 50=fixed evaluation gate; "
            "60/100/150/200 require explicit post-gate authorization"
        ),
    }
    atomic_write_json(
        experiment.root
        / "reports"
        / "generated"
        / f"pipeline_ceval_sar_plan_step{request.target_steps:03d}.json",
        plan,
    )
    return plan


async def _ceval_rollout_prompt(
    trio: Any,
    student_sampler: Any,
    teacher: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    system: str,
    section: dict[str, Any],
    seed: int,
) -> tuple[list[TokenizedDatum], list[float], UsageLedger, RoleUsage, list[int], int]:
    max_sequence = int(section["max_sequence_tokens"])
    max_completion = int(section["max_completion_tokens"])
    prompt = _prompt_ids(tokenizer, str(row["question"]), system)
    prompt, removed = _fit_prompt(
        prompt, max_sequence=max_sequence, reserved_completion=max_completion
    )
    group = int(section["group_size"])
    response = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt),
        num_samples=group,
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
    if len(sequences) != group or any(not sequence.tokens for sequence in sequences):
        raise RuntimeError("C-Eval SAR student did not return the requested rollout group")
    lengths = [len(sequence.tokens) for sequence in sequences]
    roles = RoleUsage(
        student_rollout_prefill_tokens=_usage_value(response, "input_tokens", len(prompt)),
        student_rollout_sample_tokens=_usage_value(response, "output_tokens", sum(lengths)),
    )
    scored = await asyncio.gather(
        *(
            _score_4b_completion(
                trio,
                teacher,
                prompt,
                [int(value) for value in sequence.tokens],
                seed=seed + index,
            )
            for index, sequence in enumerate(sequences)
        )
    )
    datums: list[TokenizedDatum] = []
    reverse_kls: list[float] = []
    for sequence, (teacher_logprobs, score_usage) in zip(sequences, scored, strict=True):
        completion = [int(value) for value in sequence.tokens]
        student_logprobs = [float(value) for value in sequence.logprobs]
        if len(completion) != len(student_logprobs):
            raise ValueError("C-Eval SAR student token/logprob alignment failed")
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
        roles.base_teacher_scoring_prefill_tokens += score_usage.student_prefill_tokens
        roles.base_teacher_scoring_sample_tokens += score_usage.student_sample_tokens
    usage = UsageLedger(
        student_prefill_tokens=(
            roles.student_rollout_prefill_tokens + roles.base_teacher_scoring_prefill_tokens
        ),
        student_sample_tokens=(
            roles.student_rollout_sample_tokens + roles.base_teacher_scoring_sample_tokens
        ),
    )
    return datums, reverse_kls, usage, roles, lengths, removed


def _state_payload(
    experiment: ExperimentConfig,
    config: CevalSARConfig,
    request: CevalSARRequest,
    *,
    status: str,
    completed_steps: int,
    recoverable_step: int,
    latest_optimizer_state: str | None,
    source_copy_state: str | None,
    usage: UsageLedger,
    role_usage: RoleUsage,
    checkpoints: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    contract = ceval_sar_contract(experiment, config, request)
    payload: dict[str, Any] = {
        "status": status,
        "method": METHOD,
        "target_steps": request.target_steps,
        "completed_steps": completed_steps,
        "recoverable_step": recoverable_step,
        "uncheckpointed_steps": completed_steps - recoverable_step,
        "latest_optimizer_state": latest_optimizer_state,
        "source_copy_state": source_copy_state,
        "optimizer_inherited_from_medical_opd": False,
        "usage": usage.to_dict(),
        "role_usage": role_usage.to_dict(),
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
    config: CevalSARConfig,
    request: CevalSARRequest,
    prior: dict[str, Any],
) -> bool:
    completed = int(prior.get("completed_steps", -1))
    if prior.get("method") != METHOD:
        raise RuntimeError("C-Eval SAR resume method mismatch")
    if completed != int(prior.get("recoverable_step", -2)):
        raise RuntimeError("C-Eval SAR resume has an uncheckpointed tail")
    if request.target_steps <= completed:
        raise RuntimeError("C-Eval SAR resume target must exceed completed steps")
    current_contract = ceval_sar_contract(experiment, config, request)
    current_hash = stable_hash(current_contract)
    legacy_migration = False
    if prior.get("training_contract_sha256") != current_hash:
        prior_contract = dict(prior.get("training_contract", {}))
        prior_implementation = dict(prior_contract.pop("implementation_sha256", {}))
        current_core = dict(current_contract)
        current_implementation = dict(current_core.pop("implementation_sha256", {}))
        prior_contract.pop("continuation", None)
        current_core.pop("continuation", None)
        unchanged_implementations = all(
            prior_implementation.get(name) == current_implementation.get(name)
            for name in ("contracts.py", "training.py", "staged_opd.py")
        )
        legacy_migration = (
            completed == 50
            and request.target_steps > 50
            and int(prior.get("target_steps", -1)) == 50
            and prior.get("status") == "completed"
            and prior_contract == current_core
            and prior_implementation.get("ceval_sar.py") == LEGACY_STEP50_IMPLEMENTATION_SHA256
            and unchanged_implementations
        )
        if not legacy_migration:
            raise RuntimeError("C-Eval SAR resume contract, source, or data drift")
    if request.resume_state != prior.get("latest_optimizer_state"):
        raise RuntimeError("C-Eval SAR resume optimizer state mismatch")
    if prior.get("optimizer_inherited_from_medical_opd") is not False:
        raise RuntimeError("C-Eval SAR optimizer reset contract drift")
    return legacy_migration


async def _save_checkpoint(training: Any, *, name: str, step: int) -> dict[str, Any]:
    state_future = await training.save_state_async(name=f"{name}-step{step:06d}-state")
    state = str((await state_future).path)
    weights_future = await training.save_weights_for_sampler_async(
        name=f"{name}-step{step:06d}-weights"
    )
    weights = str((await weights_future).path)
    return {
        "step": step,
        "state": state,
        "sampler_weights": weights,
        "permanent": step in PERMANENT_STEPS,
    }


async def _run_ceval_sar_async(
    experiment: ExperimentConfig,
    config: CevalSARConfig,
    request: CevalSARRequest,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    rows, _ = _load_frozen_ceval_sar_data(experiment, config)
    section = config.section("training")
    output = request.output_dir.resolve()
    state_path = output / "state.json"
    steps_path = output / "steps.jsonl"
    completed = 0 if prior is None else int(prior["completed_steps"])
    recoverable = 0 if prior is None else int(prior["recoverable_step"])
    latest_state = None if prior is None else str(prior["latest_optimizer_state"])
    source_copy_state = None if prior is None else str(prior["source_copy_state"])
    usage = UsageLedger() if prior is None else UsageLedger.from_dict(dict(prior["usage"]))
    role_usage = RoleUsage() if prior is None else RoleUsage.from_dict(dict(prior["role_usage"]))
    checkpoints = [] if prior is None else list(prior.get("checkpoints", []))
    service = trio.ServiceClient()
    model_name = str(experiment.get("models", "student"))
    model = _capability_map(service.get_server_capabilities()).get(model_name)
    if model is None or not bool(model["training"]["available"]):
        raise RuntimeError("official 4B training is unavailable for C-Eval SAR")
    if not bool(model["sampling"]["available"]):
        raise RuntimeError("official 4B sampling is unavailable for C-Eval SAR")

    if prior is None:
        source_client = await service.create_training_client_from_state_async(
            request.initial_student_state
        )
        source_future = await source_client.save_state_async(name=f"{request.run_name}-source-copy")
        source_copy_state = str((await source_future).path)
        source_client = None
        training = await service.create_training_client_from_state_async(source_copy_state)
        initial = await _save_checkpoint(training, name=request.run_name, step=0)
        latest_state = str(initial["state"])
        checkpoints.append(initial)
        append_jsonl(
            experiment.root / "reports" / "generated" / "checkpoint_index.jsonl",
            {"run_dir": str(output), **initial},
        )
        atomic_write_json(
            state_path,
            _state_payload(
                experiment,
                config,
                request,
                status="running",
                completed_steps=0,
                recoverable_step=0,
                latest_optimizer_state=latest_state,
                source_copy_state=source_copy_state,
                usage=usage,
                role_usage=role_usage,
                checkpoints=checkpoints,
            ),
        )
    else:
        training = await service.create_training_client_from_state_with_optimizer_async(
            request.resume_state
        )
    teacher = await service.create_sampling_client_async(base_model=model_name)
    tokenizer = training.get_tokenizer()
    adam = trio.AdamParams(
        learning_rate=float(section["learning_rate"]),
        beta1=float(section["beta1"]),
        beta2=float(section["beta2"]),
    )
    batch_size = int(section["batch_size"])
    seed = int(config.get("protocol", "seed"))
    explicit = {int(value) for value in section["checkpoint_steps"]}
    interval = int(section["state_save_interval"])
    try:
        for step in range(completed, request.target_steps):
            started = time.perf_counter()
            batch = _batch_for_step(rows, step, batch_size)
            student_sampler = await training.save_weights_and_get_sampling_client_async()
            rollouts = await asyncio.gather(
                *(
                    _ceval_rollout_prompt(
                        trio,
                        student_sampler,
                        teacher,
                        tokenizer,
                        row,
                        system=str(config.get("protocol", "system_message")),
                        section=section,
                        seed=seed + step * batch_size + offset,
                    )
                    for offset, row in enumerate(batch)
                )
            )
            datums = [datum for result in rollouts for datum in result[0]]
            reverse_kls = [value for result in rollouts for value in result[1]]
            completion_lengths = [value for result in rollouts for value in result[4]]
            step_usage = UsageLedger()
            step_roles = RoleUsage()
            for result in rollouts:
                step_usage.add(result[2])
                step_roles.add(result[3])
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
                raise FloatingPointError("C-Eval SAR trainer returned NaN or infinity")
            submitted_tokens = sum(datum.context_tokens for datum in datums)
            trainable_tokens = sum(datum.trainable_tokens for datum in datums)
            zero_advantage = sum(
                value == 0.0
                for datum in datums
                for value, weight in zip(datum.advantages or (), datum.weights, strict=True)
                if weight != 0.0
            )
            metric_tokens = metrics.get("token_count")
            expected = {float(trainable_tokens), float(trainable_tokens - zero_advantage)}
            if metric_tokens is not None and metric_tokens not in expected:
                raise RuntimeError("C-Eval SAR completion mask/token_count mismatch")
            step_usage.student_train_tokens += submitted_tokens
            step_usage.optimizer_steps = 1
            step_usage.wall_seconds = time.perf_counter() - started
            step_roles.student_train_tokens += submitted_tokens
            usage.add(step_usage)
            role_usage.add(step_roles)
            completed = step + 1
            append_jsonl(
                steps_path,
                {
                    "step": completed,
                    "method": METHOD,
                    "row_ids": [row["id"] for row in batch],
                    "source_row_ids": [row["source_row_id"] for row in batch],
                    "source_splits": [row["source_split"] for row in batch],
                    "datums": len(datums),
                    "group_size": section["group_size"],
                    "completion_tokens": sum(completion_lengths),
                    "completion_tokens_mean": float(np.mean(completion_lengths)),
                    "submitted_sequence_tokens": submitted_tokens,
                    "trainable_mask_tokens": trainable_tokens,
                    "zero_advantage_completion_tokens": zero_advantage,
                    "reverse_kl_mean": float(np.mean(reverse_kls)),
                    "reverse_kl_std": float(np.std(reverse_kls)),
                    "prompt_truncated_tokens": sum(result[5] for result in rollouts),
                    "trainer_metrics": metrics,
                    "usage": step_usage.to_dict(),
                    "role_usage": step_roles.to_dict(),
                },
            )
            save_due = (
                completed == request.target_steps
                or completed in explicit
                or completed % interval == 0
            )
            if save_due:
                checkpoint = await _save_checkpoint(training, name=request.run_name, step=completed)
                latest_state = str(checkpoint["state"])
                recoverable = completed
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
                    source_copy_state=source_copy_state,
                    usage=usage,
                    role_usage=role_usage,
                    checkpoints=checkpoints,
                ),
            )
        summary = _state_payload(
            experiment,
            config,
            request,
            status="completed",
            completed_steps=completed,
            recoverable_step=recoverable,
            latest_optimizer_state=latest_state,
            source_copy_state=source_copy_state,
            usage=usage,
            role_usage=role_usage,
            checkpoints=checkpoints,
        )
        try:
            prices_raw = fetch_json(PRICES_URL)
            summary["estimated_cny"] = estimate_cost(usage, price_table(prices_raw), experiment)
            summary["price_version"] = prices_raw.get("version")
        except Exception as exc:
            summary["estimated_cny"] = None
            summary["price_version"] = None
            summary["price_error"] = f"{type(exc).__name__}: {exc}"
        summary["actual_billed_cny"] = None
        summary["billing_note"] = "No SDK billing field; reconcile against https://pytrio.cn/usage."
        summary["result_scope"] = (
            "smoke_only"
            if completed == 1
            else "cost_calibration_only"
            if completed == 10
            else "exploratory_training_checkpoint"
            if completed == 25
            else "user_authorized_post_gate_exploratory_continuation"
            if completed > 50
            else "proxy_screen_training"
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
                source_copy_state=source_copy_state,
                usage=usage,
                role_usage=role_usage,
                checkpoints=checkpoints,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def run_ceval_sar(
    experiment: ExperimentConfig,
    config: CevalSARConfig,
    request: CevalSARRequest,
) -> dict[str, Any]:
    if not request.confirm_paid:
        raise RuntimeError("paid C-Eval SAR requires --confirm-paid")
    pause_path = experiment.root / "reports" / "generated" / TRAINING_PAUSE_FILENAME
    if pause_path.exists():
        raise RuntimeError(f"paid training is paused by {pause_path}")
    require_ready_preflight(experiment)
    _load_frozen_ceval_sar_data(experiment, config)
    _validate_source(request)
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
            raise RuntimeError("existing C-Eval SAR run requires --resume-state")
        legacy_migration = _validate_resume(experiment, config, request, prior)
        if legacy_migration:
            snapshot_path = output / "state_step000050_legacy.json"
            if snapshot_path.exists():
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if snapshot != prior:
                    raise RuntimeError("C-Eval SAR legacy step50 snapshot drift")
            else:
                atomic_write_json(snapshot_path, prior)
            atomic_write_json(
                experiment.root
                / "reports"
                / "generated"
                / "u4_ceval_sar_step50_to200_migration.json",
                {
                    "status": "authorized_exploratory_continuation",
                    "recorded_at": utc_now(),
                    "source_local_state": str(state_path),
                    "source_local_state_sha256": sha256_file(snapshot_path),
                    "source_optimizer_state": prior["latest_optimizer_state"],
                    "source_completed_steps": prior["completed_steps"],
                    "source_contract_sha256": prior["training_contract_sha256"],
                    "new_contract_sha256": stable_hash(
                        ceval_sar_contract(experiment, config, request)
                    ),
                    "allowed_differences": [
                        "ceval_sar.py adds deterministic schedule cycling after step50",
                        "targets 60/100/150/200 and permanent milestone checkpoints",
                        "registered step50 failure remains preserved and is not reclassified",
                    ],
                    "schedule_contract": CONTINUATION_CONTRACT,
                },
            )
    elif request.resume_state is not None:
        raise RuntimeError("C-Eval SAR resume requires an existing local state")
    output.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run_ceval_sar_async(experiment, config, request, prior))
