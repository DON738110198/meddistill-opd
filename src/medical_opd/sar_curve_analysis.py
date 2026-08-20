from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from medical_opd.ceval_sar_analysis import _load_evaluation, _paired
from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, read_jsonl, sha256_file, utc_now

CHECKPOINT_STEPS = (50, 100, 150, 200)


def _score(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    untruncated = [row for row in rows if not bool(row.get("truncated", False))]
    return {
        "correct": int(summary["correct"]),
        "count": int(summary["count"]),
        "accuracy": float(summary["accuracy"]),
        "format_valid_rate": float(summary["format_valid_rate"]),
        "thinking_closed_rate": float(summary["thinking_closed_rate"]),
        "truncation_rate": float(summary["truncation_rate"]),
        "output_tokens_mean": float(summary["output_tokens_mean"]),
        "estimated_cny": float(summary["estimated_cny"]),
        "untruncated_count": len(untruncated),
        "untruncated_correct": sum(bool(row["correct"]) for row in untruncated),
    }


def _training_interval(
    rows: list[dict[str, Any]], lower: int, upper: int
) -> dict[str, Any]:
    selected = [row for row in rows if lower <= int(row["step"]) <= upper]
    expected = upper - lower + 1
    if len(selected) != expected:
        raise RuntimeError(f"SAR training interval {lower}-{upper} is incomplete")
    completions = sum(int(row["completion_tokens"]) for row in selected)
    datums = sum(int(row["datums"]) for row in selected)
    usage_keys = (
        "student_prefill_tokens",
        "student_sample_tokens",
        "student_train_tokens",
    )
    usage = {
        key: sum(int(row["usage"][key]) for row in selected) for key in usage_keys
    }
    return {
        "range": f"{lower}-{upper}",
        "steps": len(selected),
        "completion_tokens_per_rollout": completions / datums,
        "reverse_kl_mean": sum(float(row["reverse_kl_mean"]) for row in selected)
        / len(selected),
        "loss_min": min(float(row["trainer_metrics"]["loss_mean"]) for row in selected),
        "loss_max": max(float(row["trainer_metrics"]["loss_mean"]) for row in selected),
        "prompt_truncated_tokens": sum(
            int(row["prompt_truncated_tokens"]) for row in selected
        ),
        "usage": usage,
    }


def _cost_from_usage(config: ExperimentConfig, usage: dict[str, int]) -> float:
    cny = (
        usage["student_prefill_tokens"]
        * float(config.get("pricing", "student_prefill_cny_per_million"))
        + usage["student_sample_tokens"]
        * float(config.get("pricing", "student_sample_cny_per_million"))
        + usage["student_train_tokens"]
        * float(config.get("pricing", "student_train_cny_per_million"))
    ) / 1_000_000
    return round(cny, 6)


def _both_untruncated_pair(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    right_by_id = {str(row["id"]): row for row in right}
    left_selected = [
        row
        for row in left
        if not bool(row.get("truncated", False))
        and not bool(right_by_id[str(row["id"])].get("truncated", False))
    ]
    selected_ids = {str(row["id"]) for row in left_selected}
    right_selected = [row for row in right if str(row["id"]) in selected_ids]
    test, _ = _paired(left_selected, right_selected)
    return test


def build_sar_curve_analysis(
    experiment: ExperimentConfig, *, output_path: Path
) -> dict[str, Any]:
    root = experiment.root
    paths = {
        "M0": root / "runs" / "reference-protocol" / "M0" / "medical600",
        **{
            f"SAR@{step}": root
            / "runs"
            / "pipeline"
            / "reference-eval"
            / f"sar-ceval-e2teacher-step{step:03d}"
            / "medical600"
            for step in CHECKPOINT_STEPS
        },
    }
    loaded = {name: _load_evaluation(path) for name, path in paths.items()}
    baseline_summary, baseline_rows = loaded["M0"]
    expected_ids = [str(row["id"]) for row in baseline_rows]
    if len(expected_ids) != 600 or len(set(expected_ids)) != 600:
        raise RuntimeError("reference medical600 baseline must contain 600 unique rows")
    dataset_sha = str(baseline_summary["dataset_sha256"])
    for name, (summary, rows) in loaded.items():
        if (
            summary.get("status") != "completed"
            or int(summary.get("count", -1)) != 600
            or str(summary.get("dataset_sha256")) != dataset_sha
            or [str(row["id"]) for row in rows] != expected_ids
        ):
            raise RuntimeError(f"SAR curve evaluation contract drift: {name}")

    comparisons: dict[str, Any] = {}
    bad_cases: dict[str, list[dict[str, Any]]] = {}
    previous_name = "M0"
    previous_rows = baseline_rows
    for step in CHECKPOINT_STEPS:
        name = f"SAR@{step}"
        rows = loaded[name][1]
        versus_base, versus_base_cases = _paired(baseline_rows, rows)
        versus_previous, versus_previous_cases = _paired(previous_rows, rows)
        comparisons[f"M0_to_{name}"] = {
            **versus_base,
            "both_untruncated": _both_untruncated_pair(baseline_rows, rows),
        }
        comparisons[f"{previous_name}_to_{name}"] = {
            **versus_previous,
            "both_untruncated": _both_untruncated_pair(previous_rows, rows),
        }
        bad_cases[f"M0_to_{name}"] = versus_base_cases
        bad_cases[f"{previous_name}_to_{name}"] = versus_previous_cases
        previous_name = name
        previous_rows = rows
    step50_rows = loaded["SAR@50"][1]
    step200_rows = loaded["SAR@200"][1]
    step50_to_200, step50_to_200_cases = _paired(step50_rows, step200_rows)
    comparisons["SAR@50_to_SAR@200"] = {
        **step50_to_200,
        "both_untruncated": _both_untruncated_pair(step50_rows, step200_rows),
    }
    bad_cases["SAR@50_to_SAR@200"] = step50_to_200_cases

    run_dir = root / "runs" / "pipeline" / "BASE-SAR-E2TEACHER"
    state_path = run_dir / "state.json"
    steps_path = run_dir / "steps.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = read_jsonl(steps_path)
    if (
        state.get("status") != "completed"
        or int(state.get("completed_steps", -1)) != 200
        or int(state.get("recoverable_step", -1)) != 200
        or [int(row["step"]) for row in steps] != list(range(1, 201))
    ):
        raise RuntimeError("SAR@200 training state or step ledger is incomplete")
    for row in steps:
        numeric = [
            float(row["reverse_kl_mean"]),
            float(row["reverse_kl_std"]),
            *(float(value) for value in row["trainer_metrics"].values()),
        ]
        if any(not math.isfinite(value) for value in numeric):
            raise RuntimeError("SAR training ledger contains NaN or infinity")
        trainable = int(row["trainable_mask_tokens"])
        zero = int(row["zero_advantage_completion_tokens"])
        token_count = float(row["trainer_metrics"]["token_count"])
        if token_count not in {float(trainable), float(trainable - zero)}:
            raise RuntimeError("SAR training mask/token_count identity failed")

    interval_bounds = ((1, 50), (51, 100), (101, 150), (151, 200))
    intervals = [_training_interval(steps, lower, upper) for lower, upper in interval_bounds]
    cumulative_cost: dict[str, float] = {}
    cumulative_usage = {
        "student_prefill_tokens": 0,
        "student_sample_tokens": 0,
        "student_train_tokens": 0,
    }
    for interval, checkpoint in zip(intervals, CHECKPOINT_STEPS, strict=True):
        for key, value in interval["usage"].items():
            cumulative_usage[key] += int(value)
        interval["estimated_cny"] = _cost_from_usage(
            experiment, {key: int(value) for key, value in interval["usage"].items()}
        )
        cumulative_cost[str(checkpoint)] = _cost_from_usage(experiment, cumulative_usage)
    if abs(cumulative_cost["200"] - float(state["estimated_cny"])) > 1e-6:
        raise RuntimeError("SAR step-ledger cost does not match final state")

    source_exposures = Counter(
        str(source_id) for row in steps for source_id in row["source_row_ids"]
    )
    evaluation_cost = sum(
        float(loaded[f"SAR@{step}"][0]["estimated_cny"]) for step in CHECKPOINT_STEPS
    )
    proxy_cost = sum(
        float(json.loads(path.read_text(encoding="utf-8"))["estimated_cny"])
        for path in (
            root
            / "runs"
            / "pipeline"
            / "eval"
            / "sar-ceval-e2teacher-step50"
            / "medical"
            / "summary.json",
            root
            / "runs"
            / "pipeline"
            / "eval"
            / "sar-ceval-e2teacher-step50"
            / "general"
            / "summary.json",
        )
    )
    checkpoint_by_step = {
        int(item["step"]): item
        for item in state["checkpoints"]
        if int(item["step"]) in CHECKPOINT_STEPS
    }
    if set(checkpoint_by_step) != set(CHECKPOINT_STEPS) or any(
        item.get("permanent") is not True for item in checkpoint_by_step.values()
    ):
        raise RuntimeError("SAR milestone checkpoint index is incomplete")

    result = {
        "status": "completed",
        "created_at": utc_now(),
        "scope": "fixed MedQA-zh seed42 medical600 protocol diagnostic",
        "method": "BASE-SAR",
        "scores": {
            name: _score(summary, rows) for name, (summary, rows) in loaded.items()
        },
        "paired_comparisons": comparisons,
        "hash_only_bad_cases": bad_cases,
        "training": {
            "completed_steps": 200,
            "recoverable_step": 200,
            "intervals": intervals,
            "cumulative_estimated_cny": cumulative_cost,
            "usage": state["usage"],
            "role_usage": state["role_usage"],
            "all_steps_finite": True,
            "all_mask_identities_valid": True,
            "schedule": {
                "presentations": len(steps) * 4,
                "unique_source_rows": len(source_exposures),
                "minimum_source_exposures": min(source_exposures.values()),
                "maximum_source_exposures": max(source_exposures.values()),
                "frozen_schedule_cycles": 4,
                "proxy_training_rows": 0,
                "test_training_rows": 0,
            },
        },
        "checkpoints": {str(step): checkpoint_by_step[step] for step in CHECKPOINT_STEPS},
        "cost": {
            "price_version": state["price_version"],
            "training_estimated_cny": float(state["estimated_cny"]),
            "medical600_evaluations_estimated_cny": evaluation_cost,
            "continuation_after_existing_step50_estimated_cny": round(
                float(state["estimated_cny"]) - cumulative_cost["50"] + evaluation_cost,
                6,
            ),
            "requested_curve_total_estimated_cny": float(state["estimated_cny"])
            + evaluation_cost,
            "prior_proxy_evaluations_estimated_cny": proxy_cost,
            "full_branch_total_estimated_cny": float(state["estimated_cny"])
            + evaluation_cost
            + proxy_cost,
            "actual_billed_cny": None,
            "unobserved_interrupted_request": (
                "At most one post-step50 optimizer update may have been submitted before the "
                "foreground process was terminated; it produced no checkpoint and is excluded "
                "from the local token ledger."
            ),
            "billing_note": "Reconcile estimates against https://pytrio.cn/usage.",
        },
        "decision": {
            "best_observed_checkpoint": "SAR@50",
            "best_selection_status": "post-hoc exploratory, not confirmatory",
            "stop_at_step": 200,
            "next_paid_action": "none",
            "general_retention_after_step50": "not measured by this medical600-only curve",
        },
        "claim": (
            "In this leakage-safe low-budget SAR continuation, medical600 accuracy peaked at "
            "step50 and then declined as reverse KL fell and answer lengths increased. Longer "
            "Base-anchor OPD did not monotonically improve medical capability."
        ),
        "artifacts": {
            "training_state_sha256": sha256_file(state_path),
            "training_steps_sha256": sha256_file(steps_path),
            **{
                f"{name}_summary_sha256": sha256_file(path / "summary.json")
                for name, path in paths.items()
            },
            **{
                f"{name}_predictions_sha256": sha256_file(path / "predictions.jsonl")
                for name, path in paths.items()
            },
        },
    }
    atomic_write_json(output_path.resolve(), result)
    return result
