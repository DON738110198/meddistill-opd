from __future__ import annotations

from pathlib import Path

import pytest

from medical_opd.config import load_config
from medical_opd.medical_sft import (
    DEFAULT_MEDICAL_PIPELINE_CONFIG,
    MedicalSFTRequest,
    _batch_indices,
    _steps_per_epoch,
    load_medical_pipeline_config,
    run_medical_sft,
)


def test_pipeline_config_freezes_group4_mechanism_without_ceval_replay() -> None:
    experiment = load_config()
    config = load_medical_pipeline_config(experiment)

    assert config.path == DEFAULT_MEDICAL_PIPELINE_CONFIG
    assert config.get("protocol", "base_model") == "Qwen/Qwen3.5-4B"
    assert config.get("medical_sft", "batch_size") == 16
    assert config.get("medical_sft", "epochs") == 3
    assert config.get("medical_sft", "max_sequence_tokens") == 2048
    assert config.get("medical_opd", "group_size") == 4
    assert config.get("sar", "prompt_source") == "alpaca-zh-general-replay"
    assert "ceval" not in str(config.get("protocol", "general_replay_path")).casefold()


def test_epoch_batch_order_is_deterministic_and_changes_between_epochs() -> None:
    row_count = 33
    batch_size = 16
    steps = _steps_per_epoch(row_count, batch_size)
    assert steps == 3

    epoch0, first = _batch_indices(
        row_count=row_count, batch_size=batch_size, seed=7, step=0
    )
    repeated_epoch0, repeated = _batch_indices(
        row_count=row_count, batch_size=batch_size, seed=7, step=0
    )
    epoch1, next_epoch = _batch_indices(
        row_count=row_count, batch_size=batch_size, seed=7, step=steps
    )
    _, final_short_batch = _batch_indices(
        row_count=row_count, batch_size=batch_size, seed=7, step=2
    )

    assert epoch0 == repeated_epoch0 == 0
    assert epoch1 == 1
    assert first == repeated
    assert first != next_epoch
    assert len(final_short_batch) == 1


def test_pipeline_sft_request_rejects_nonpositive_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        MedicalSFTRequest(target_steps=0, output_dir=tmp_path)


def test_paid_pipeline_sft_guard_precedes_preflight(tmp_path: Path) -> None:
    experiment = load_config()
    config = load_medical_pipeline_config(experiment)
    request = MedicalSFTRequest(target_steps=1, output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="--confirm-paid"):
        run_medical_sft(experiment, config, request)
