from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from medical_opd.config import DEFAULT_CONFIG_PATH, ExperimentConfig, load_config


@pytest.fixture
def valid_config() -> ExperimentConfig:
    loaded = load_config()
    return ExperimentConfig(loaded.path, copy.deepcopy(loaded.raw))


def write_config(path: Path, raw: dict[str, object]) -> None:
    # The fixture only needs the scalar/list shape used by the repository config.
    lines: list[str] = []
    for section, values in raw.items():
        lines.append(f"[{section}]")
        assert isinstance(values, dict)
        for key, value in values.items():
            if isinstance(value, str):
                rendered = repr(value).replace("'", '"')
            elif isinstance(value, bool):
                rendered = str(value).lower()
            elif isinstance(value, list):
                rendered = repr(value).replace("'", '"')
            else:
                rendered = str(value)
            lines.append(f"{key} = {rendered}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def mutate_and_load(
    tmp_path: Path,
    config: ExperimentConfig,
    section: str,
    key: str,
    value: object,
) -> ExperimentConfig:
    raw = copy.deepcopy(config.raw)
    raw[section][key] = value
    path = tmp_path / "experiment.toml"
    write_config(path, raw)
    return load_config(path)


def test_repository_config_satisfies_preregistered_invariants() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config.get("training", "batch_size") == 4
    assert config.get("training", "group_size") == 1
    assert config.get("data", "mixed_medical_per_step") == 3
    assert config.get("data", "mixed_general_per_step") == 1
    assert config.get("training", "default_max_length") <= 1024
    assert {10, 25, 50}.issubset(config.get("training", "save_steps"))
    assert config.get("data", "general_replay_repo") != config.get("data", "ceval_repo")
    assert config.get("models", "student") != config.get("models", "teacher")
    assert config.get("models", "student") == "Qwen/Qwen3.5-4B"
    assert config.get("gates", "max_general_drop_pp") == 2.0
    assert config.get("evaluation", "thinking_protocol_id") == "thinking-fixed-v1"
    assert config.get("evaluation", "thinking_enabled") is True
    assert config.get("evaluation", "thinking_medical_max_tokens") == 1024
    assert config.get("evaluation", "thinking_general_max_tokens") == 8192
    assert config.get("evaluation", "thinking_temperature") == 0.01
    assert config.get("evaluation", "thinking_top_p") == 0.9
    assert config.get("evaluation", "thinking_seed_per_row") is True
    assert config.get("medical_teacher_sft", "batch_size") == 4
    assert config.get("medical_teacher_sft", "lora_rank") == 32
    assert config.get("medical_teacher_sft", "save_steps") == [1, 10, 25]
    assert config.get("staged_opd", "screen_steps") == 25
    assert config.get("staged_opd", "group_size") == 4
    assert config.get("staged_opd", "enable_thinking") is True
    assert config.get("staged_opd", "sar_prompt_source") == "alpaca-zh-general-replay"
    for key in (
        "medical_revision",
        "medqa_revision",
        "general_replay_revision",
        "ceval_labeled_revision",
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", config.get("data", key))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("training", "batch_size", 8, "batch_size=4"),
        ("training", "group_size", 2, "group_size=1"),
        ("data", "mixed_general_per_step", 2, "sum to batch_size"),
        ("training", "default_max_length", 1025, "must not exceed 1024"),
        ("training", "minimum_max_length", 2048, "must not exceed default_max_length"),
        ("training", "kl_penalty", -0.1, "must be non-negative"),
        ("training", "advantage_clip", 0.0, "must be positive"),
        ("training", "save_steps", [10, 50], "preserve 10, 25, and 50"),
        ("models", "student", "Qwen/Qwen3.5-4B-Instruct", "official PyTRIO"),
        ("evaluation", "thinking_protocol_id", "drifted", "protocol id"),
        ("evaluation", "thinking_enabled", False, "thinking_enabled=true"),
        ("evaluation", "thinking_medical_max_tokens", 512, "1024-token"),
        ("evaluation", "thinking_general_max_tokens", 1024, "8192-token"),
        ("evaluation", "thinking_temperature", 0.1, "temperature"),
        ("evaluation", "thinking_top_p", 1.0, "top_p"),
        ("evaluation", "thinking_seed_per_row", False, "deterministic seed"),
        ("medical_teacher_sft", "batch_size", 8, "batch_size=4"),
        ("medical_teacher_sft", "lora_rank", 16, "rank 32"),
        ("medical_teacher_sft", "save_steps", [1, 25], "save steps 1, 10, and 25"),
        ("staged_opd", "screen_steps", 50, "remain 25 steps"),
        ("staged_opd", "group_size", 1, "group_size=4"),
        ("staged_opd", "enable_thinking", False, "thinking-enabled"),
        ("staged_opd", "max_sequence_tokens", 2048, "must not exceed 1024"),
        ("staged_opd", "sar_prompt_source", "ceval", "never C-Eval"),
    ],
)
def test_load_config_rejects_comparison_breaking_changes(
    tmp_path: Path,
    valid_config: ExperimentConfig,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mutate_and_load(tmp_path, valid_config, section, key, value)


def test_missing_section_is_rejected(tmp_path: Path, valid_config: ExperimentConfig) -> None:
    raw = copy.deepcopy(valid_config.raw)
    del raw["gates"]
    path = tmp_path / "experiment.toml"
    write_config(path, raw)

    with pytest.raises(ValueError, match="missing config section: gates"):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("medical_revision", "main"),
        ("medqa_revision", ""),
        ("general_replay_revision", "abc123"),
        ("ceval_labeled_revision", "Z" * 40),
    ],
)
def test_dataset_revisions_must_be_pinned_commit_shas(
    tmp_path: Path,
    valid_config: ExperimentConfig,
    key: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="revision"):
        mutate_and_load(tmp_path, valid_config, "data", key, value)
