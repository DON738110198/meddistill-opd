from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import medical_opd.backend as backend_module
import medical_opd.config as config_module
from medical_opd.backend import (
    UsageLedger,
    estimate_cost,
    plan_run,
    price_table,
    require_ready_preflight,
)
from medical_opd.config import ExperimentConfig, load_config
from medical_opd.io_utils import atomic_write_json, sha256_file


def _config_at(tmp_path: Path) -> ExperimentConfig:
    loaded = load_config()
    return ExperimentConfig(loaded.path, copy.deepcopy(loaded.raw))


def _preflight_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ExperimentConfig:
    loaded = load_config()
    config_path = tmp_path / "experiment.toml"
    config_path.write_bytes(loaded.path.read_bytes())
    config = ExperimentConfig(config_path, copy.deepcopy(loaded.raw))
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "reports" / "generated").mkdir(parents=True)
    return config


def _ready_preflight(config: ExperimentConfig) -> dict[str, object]:
    return {
        "status": "ready",
        "config_sha256": sha256_file(config.path),
        "configured_models": {
            "student": config.get("models", "student"),
            "teacher": config.get("models", "teacher"),
        },
    }


def test_ready_preflight_requires_exact_base_student_teacher_and_config_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _preflight_config(tmp_path, monkeypatch)
    report = _ready_preflight(config)
    report["unrelated_audit_evidence"] = {"preserved": True}
    atomic_write_json(
        tmp_path / "reports" / "generated" / "preflight_latest.json",
        report,
    )

    loaded = require_ready_preflight(config)

    assert loaded == report
    assert loaded["configured_models"] == {
        "student": "Qwen/Qwen3.5-4B",
        "teacher": "Qwen/Qwen3.6-27B",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report, config: report.update(status="blocked"),
        lambda report, config: report.update(status="environment_ready_data_pending"),
        lambda report, config: report.update(config_sha256="0" * 64),
        lambda report, config: report["configured_models"].update(
            student="Qwen/Qwen3.5-4B-Base"
        ),
        lambda report, config: report["configured_models"].update(teacher="wrong/27B"),
        lambda report, config: report.update(
            configured_models={"student": config.get("models", "student")}
        ),
        lambda report, config: report["configured_models"].update(extra="unexpected"),
    ],
    ids=[
        "blocked-status",
        "data-pending-status",
        "stale-config-hash",
        "wrong-student-model",
        "wrong-teacher",
        "missing-teacher",
        "extra-model-key",
    ],
)
def test_require_ready_preflight_rejects_nonexact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
) -> None:
    config = _preflight_config(tmp_path, monkeypatch)
    report = _ready_preflight(config)
    mutate(report, config)  # type: ignore[operator]
    atomic_write_json(
        tmp_path / "reports" / "generated" / "preflight_latest.json",
        report,
    )

    with pytest.raises(RuntimeError, match="no paid work may start"):
        require_ready_preflight(config)


def test_ready_preflight_becomes_stale_after_config_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _preflight_config(tmp_path, monkeypatch)
    atomic_write_json(
        tmp_path / "reports" / "generated" / "preflight_latest.json",
        _ready_preflight(config),
    )
    config.path.write_bytes(config.path.read_bytes() + b"\n# changed after preflight\n")

    with pytest.raises(RuntimeError, match="blocked or stale"):
        require_ready_preflight(config)


def test_usage_ledger_round_trip_and_addition() -> None:
    usage = UsageLedger(student_prefill_tokens=10, student_train_tokens=20, optimizer_steps=1)
    usage.add(
        UsageLedger(
            student_sample_tokens=5,
            teacher_prefill_tokens=11,
            teacher_sample_tokens=7,
            optimizer_steps=2,
            wall_seconds=1.5,
        )
    )

    assert usage.student_tokens == 35
    assert usage.optimizer_steps == 3
    assert usage.wall_seconds == 1.5
    assert UsageLedger.from_dict(usage.to_dict()).to_dict() == usage.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"student_train_tokens": -1},
        {"teacher_sample_tokens": -1},
        {"optimizer_steps": -1},
        {"wall_seconds": -0.1},
        {"wall_seconds": float("nan")},
        {"wall_seconds": float("inf")},
    ],
)
def test_usage_ledger_rejects_invalid_counters(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        UsageLedger(**kwargs)


def test_official_price_payload_converts_fen_to_cny_per_million() -> None:
    table = price_table(
        {
            "items": [
                {
                    "display_name": "student",
                    "prices": {
                        "prefill": {"unit_price": 149},
                        "sample": {"unit_price": 454},
                        "train": {"unit_price": 454},
                    },
                }
            ]
        }
    )

    assert table == {"student": {"prefill": 1.49, "sample": 4.54, "train": 4.54}}


def test_estimated_cost_accounts_for_student_and_teacher_modes(tmp_path: Path) -> None:
    config = _config_at(tmp_path)
    prices = {
        str(config.get("models", "student")): {"prefill": 1.0, "sample": 2.0, "train": 3.0},
        str(config.get("models", "teacher")): {"prefill": 4.0, "sample": 5.0, "train": 6.0},
    }
    usage = UsageLedger(
        student_prefill_tokens=1_000_000,
        student_sample_tokens=1_000_000,
        student_train_tokens=1_000_000,
        teacher_prefill_tokens=1_000_000,
        teacher_sample_tokens=1_000_000,
        teacher_train_tokens=1_000_000,
    )

    assert estimate_cost(usage, prices, config) == 21.0


@pytest.fixture
def planned_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ExperimentConfig, dict[str, dict[str, float]]]:
    config = _config_at(tmp_path)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "lengths.json").write_text(
        json.dumps(
            {
                "prompt_tokens": {"p90": 100},
                "derived_max_completion_tokens": 50,
                "derived_max_sequence_tokens": 150,
            }
        ),
        encoding="utf-8",
    )
    student = str(config.get("models", "student"))
    teacher = str(config.get("models", "teacher"))
    prices = {
        student: {"prefill": 1.0, "sample": 2.0, "train": 3.0},
        teacher: {"prefill": 4.0, "sample": 5.0, "train": 6.0},
    }
    monkeypatch.setattr(
        backend_module,
        "fetch_json",
        lambda url: {
            "version": "test-price-v1",
            "items": [
                {
                    "display_name": name,
                    "prices": {
                        mode: {"unit_price": value * 100} for mode, value in modes.items()
                    },
                }
                for name, modes in prices.items()
            ],
        },
    )
    return config, prices


@pytest.mark.parametrize("method", ["M2", "M3", "M4", "M5"])
def test_plan_run_records_comparable_shape_and_budget_fields(
    planned_config: tuple[ExperimentConfig, dict[str, dict[str, float]]], method: str
) -> None:
    config, _ = planned_config
    plan = plan_run(config, method, target_steps=10)

    assert plan["mode"] == "remote_paid_training"
    assert plan["target_optimizer_steps"] == 10
    assert plan["batch_size"] == 4
    assert plan["max_sequence_tokens"] == 150
    assert plan["max_completion_tokens"] == 50
    assert plan["upper_bound_usage"]["optimizer_steps"] == 10
    assert plan["upper_bound_estimated_cny"] > 0
    assert plan["price_version"] == "test-price-v1"
    if method in {"M2", "M3"}:
        assert plan["teacher_model"] is None
    else:
        assert plan["teacher_model"] == config.get("models", "teacher")


def test_plan_run_distinguishes_sft_kd_and_opd_usage(
    planned_config: tuple[ExperimentConfig, dict[str, dict[str, float]]]
) -> None:
    config, _ = planned_config
    m2 = plan_run(config, "M2", target_steps=10)["upper_bound_usage"]
    m4 = plan_run(config, "M4", target_steps=10)["upper_bound_usage"]
    m5 = plan_run(config, "M5", target_steps=10)["upper_bound_usage"]

    assert m2["student_train_tokens"] == 10 * 4 * 150
    assert m2["teacher_prefill_tokens"] == 0
    assert m4["teacher_prefill_tokens"] == 10 * 4 * 100
    assert m4["teacher_sample_tokens"] == 10 * 4 * 50
    assert m5["student_prefill_tokens"] == 10 * 4 * 100
    assert m5["student_sample_tokens"] == 10 * 4 * 50
    assert m5["teacher_prefill_tokens"] == 10 * 4 * 150
    assert m5["student_train_tokens"] == 10 * 4 * 149


@pytest.mark.parametrize("target_steps", [0, -1])
def test_plan_run_rejects_nonpositive_steps(
    planned_config: tuple[ExperimentConfig, dict[str, dict[str, float]]], target_steps: int
) -> None:
    config, _ = planned_config
    with pytest.raises(ValueError, match="steps"):
        plan_run(config, "M2", target_steps=target_steps)


def test_plan_run_requires_prepared_length_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_at(tmp_path)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="length audit is missing"):
        plan_run(config, "M2", target_steps=1)


def test_plan_run_rejects_unknown_method(
    planned_config: tuple[ExperimentConfig, dict[str, dict[str, float]]]
) -> None:
    config, _ = planned_config
    with pytest.raises(ValueError, match="M2, M3, M4, or M5"):
        plan_run(config, "M0", target_steps=1)
