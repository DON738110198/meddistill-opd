from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.toml"


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing config section: {name}")
        return value

    def get(self, section: str, key: str) -> Any:
        values = self.section(section)
        if key not in values:
            raise ValueError(f"missing config value: {section}.{key}")
        return values[key]


def load_config(path: Path | None = None) -> ExperimentConfig:
    config_path = (path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = ExperimentConfig(config_path, raw)
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    for section in (
        "experiment",
        "models",
        "data",
        "training",
        "medical_teacher_sft",
        "staged_opd",
        "gates",
        "evaluation",
        "pricing",
    ):
        config.section(section)

    batch_size = int(config.get("training", "batch_size"))
    group_size = int(config.get("training", "group_size"))
    mixed_total = int(config.get("data", "mixed_medical_per_step")) + int(
        config.get("data", "mixed_general_per_step")
    )
    if batch_size != 4:
        raise ValueError("the preregistered comparison requires training.batch_size=4")
    if group_size != 1:
        raise ValueError("the preregistered OPD comparison requires training.group_size=1")
    if mixed_total != batch_size:
        raise ValueError("the fixed mixed-SFT ratio must sum to batch_size")
    if int(config.get("training", "default_max_length")) > 1024:
        raise ValueError("default_max_length must not exceed 1024")
    if int(config.get("training", "minimum_max_length")) > int(
        config.get("training", "default_max_length")
    ):
        raise ValueError("minimum_max_length must not exceed default_max_length")
    if float(config.get("training", "kl_penalty")) < 0:
        raise ValueError("kl_penalty must be non-negative")
    if float(config.get("training", "advantage_clip")) <= 0:
        raise ValueError("advantage_clip must be positive")
    if str(config.get("models", "student")) != "Qwen/Qwen3.5-4B":
        raise ValueError("this experiment requires the official PyTRIO Qwen/Qwen3.5-4B")
    if str(config.get("evaluation", "thinking_protocol_id")) != "thinking-fixed-v1":
        raise ValueError("the corrected evaluation protocol id must remain thinking-fixed-v1")
    if config.get("evaluation", "thinking_enabled") is not True:
        raise ValueError("the corrected evaluation contract requires thinking_enabled=true")
    if int(config.get("evaluation", "thinking_medical_max_tokens")) != 1024:
        raise ValueError("thinking medical evaluation must use the frozen 1024-token budget")
    if int(config.get("evaluation", "thinking_general_max_tokens")) != 8192:
        raise ValueError("thinking general evaluation must use the frozen 8192-token budget")
    if float(config.get("evaluation", "thinking_temperature")) != 0.01:
        raise ValueError("thinking evaluation temperature must remain 0.01")
    if float(config.get("evaluation", "thinking_top_p")) != 0.9:
        raise ValueError("thinking evaluation top_p must remain 0.9")
    if config.get("evaluation", "thinking_seed_per_row") is not True:
        raise ValueError("thinking evaluation requires a deterministic seed per frozen row")

    if str(config.get("medical_teacher_sft", "protocol_id")) != "t27-medical-sft-v1":
        raise ValueError("medical teacher SFT protocol id must remain t27-medical-sft-v1")
    if int(config.get("medical_teacher_sft", "batch_size")) != 4:
        raise ValueError("medical teacher SFT requires batch_size=4 for the staged screen")
    if int(config.get("medical_teacher_sft", "lora_rank")) != 32:
        raise ValueError("medical teacher SFT requires the preregistered rank 32")
    if int(config.get("medical_teacher_sft", "max_sequence_tokens")) > 1024:
        raise ValueError("medical teacher SFT max_sequence_tokens must not exceed 1024")
    if [int(value) for value in config.get("medical_teacher_sft", "save_steps")] != [
        1,
        10,
        25,
    ]:
        raise ValueError("medical teacher SFT must save steps 1, 10, and 25")

    if str(config.get("staged_opd", "protocol_id")) != "paired-medical-teacher-sar-v1":
        raise ValueError("staged OPD protocol id must remain paired-medical-teacher-sar-v1")
    if int(config.get("staged_opd", "screen_steps")) != 25:
        raise ValueError("staged OPD screen must remain 25 steps")
    if int(config.get("staged_opd", "batch_size")) != 4:
        raise ValueError("staged OPD requires batch_size=4")
    if int(config.get("staged_opd", "group_size")) != 4:
        raise ValueError("staged OPD requires the frozen group_size=4 rollout contract")
    if config.get("staged_opd", "enable_thinking") is not True:
        raise ValueError("staged OPD requires thinking-enabled rollout prompts")
    staged_sequence = int(config.get("staged_opd", "max_sequence_tokens"))
    staged_completion = int(config.get("staged_opd", "max_completion_tokens"))
    if staged_sequence > 1024:
        raise ValueError("staged OPD max_sequence_tokens must not exceed 1024")
    if staged_completion <= 0 or staged_completion >= staged_sequence:
        raise ValueError("staged OPD completion cap must leave prompt room")
    if str(config.get("staged_opd", "sar_prompt_source")) != "alpaca-zh-general-replay":
        raise ValueError("SAR must use leakage-safe Alpaca-ZH replay, never C-Eval")
    save_steps = [int(value) for value in config.get("training", "save_steps")]
    if not {10, 25, 50}.issubset(save_steps):
        raise ValueError("save_steps must preserve 10, 25, and 50")
    for key in (
        "medical_revision",
        "medqa_revision",
        "general_replay_revision",
        "ceval_labeled_revision",
    ):
        revision = str(config.get("data", key))
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"data.{key} must be a pinned 40-hex commit")
