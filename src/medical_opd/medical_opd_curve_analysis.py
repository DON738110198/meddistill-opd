from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from medical_opd.ceval_sar_analysis import _load_evaluation, _paired
from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, read_jsonl, sha256_file, utc_now
from medical_opd.sar_curve_analysis import (
    _both_untruncated_pair,
    _cost_from_usage,
    _score,
    _training_interval,
)

CHECKPOINT_STEPS = (50, 100, 150, 200, 250, 300)
DATASETS = {"medical": ("medical600", 600), "general": ("ceval300", 300)}


def _evaluation_paths(root: Path) -> dict[str, Path]:
    paths = {
        "M0_medical": root / "runs" / "reference-protocol" / "M0" / "medical600",
        "M0_general": root / "runs" / "reference-protocol" / "M0" / "ceval300",
    }
    for step in CHECKPOINT_STEPS:
        base = (
            root
            / "runs"
            / "pipeline"
            / "reference-eval"
            / f"medical-opd-e2teacher-step{step:03d}"
        )
        paths[f"OPD@{step}_medical"] = base / "medical600"
        paths[f"OPD@{step}_general"] = base / "ceval300"
    return paths


def _validate_evaluations(
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    for domain, (_, expected_count) in DATASETS.items():
        baseline_summary, baseline_rows = loaded[f"M0_{domain}"]
        expected_ids = [str(row["id"]) for row in baseline_rows]
        if len(expected_ids) != expected_count or len(set(expected_ids)) != expected_count:
            raise RuntimeError(f"reference {domain} baseline rows are not unique and complete")
        dataset_sha = str(baseline_summary["dataset_sha256"])
        for name in ("M0", *(f"OPD@{step}" for step in CHECKPOINT_STEPS)):
            summary, rows = loaded[f"{name}_{domain}"]
            if (
                summary.get("status") != "completed"
                or int(summary.get("count", -1)) != expected_count
                or str(summary.get("dataset_sha256")) != dataset_sha
                or [str(row["id"]) for row in rows] != expected_ids
            ):
                raise RuntimeError(f"Medical OPD curve evaluation contract drift: {name}_{domain}")
            _paired(baseline_rows, rows)


def _paired_curve(
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    comparisons: dict[str, Any] = {}
    bad_cases: dict[str, list[dict[str, Any]]] = {}
    for domain in DATASETS:
        baseline_rows = loaded[f"M0_{domain}"][1]
        previous_name = "M0"
        previous_rows = baseline_rows
        for step in CHECKPOINT_STEPS:
            name = f"OPD@{step}"
            rows = loaded[f"{name}_{domain}"][1]
            for label, left in (
                (f"M0_to_{name}", baseline_rows),
                (f"{previous_name}_to_{name}", previous_rows),
            ):
                test, cases = _paired(left, rows)
                rows_by_id = {str(row["id"]): row for row in rows}
                for case in cases:
                    case["prompt_token_sha256"] = rows_by_id[str(case["id"])]["prompt_token_sha256"]
                    if case.get("question_sha256") is None:
                        case.pop("question_sha256", None)
                key = f"{domain}:{label}"
                comparisons[key] = {
                    **test,
                    "both_untruncated": _both_untruncated_pair(left, rows),
                }
                bad_cases[key] = cases
            previous_name = name
            previous_rows = rows
        anchor_name = "OPD@100"
        anchor_rows = loaded[f"{anchor_name}_{domain}"][1]
        for step in (200, 250, 300):
            name = f"OPD@{step}"
            rows = loaded[f"{name}_{domain}"][1]
            test, cases = _paired(anchor_rows, rows)
            rows_by_id = {str(row["id"]): row for row in rows}
            for case in cases:
                case["prompt_token_sha256"] = rows_by_id[str(case["id"])]["prompt_token_sha256"]
                if case.get("question_sha256") is None:
                    case.pop("question_sha256", None)
            key = f"{domain}:{anchor_name}_to_{name}"
            comparisons[key] = {
                **test,
                "both_untruncated": _both_untruncated_pair(anchor_rows, rows),
            }
            bad_cases[key] = cases
    return comparisons, bad_cases


def _validate_training(root: Path) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    run_dir = root / "runs" / "pipeline" / "MED-OPD-E2TEACHER"
    state_path = run_dir / "state.json"
    steps_path = run_dir / "steps.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = read_jsonl(steps_path)
    if (
        state.get("status") != "completed"
        or int(state.get("completed_steps", -1)) != 300
        or int(state.get("recoverable_step", -1)) != 300
        or [int(row["step"]) for row in steps] != list(range(1, 301))
    ):
        raise RuntimeError("Medical OPD@300 training state or step ledger is incomplete")
    for row in steps:
        numeric = [
            float(row["reverse_kl_mean"]),
            float(row["reverse_kl_std"]),
            *(float(value) for value in row["trainer_metrics"].values()),
        ]
        if any(not math.isfinite(value) for value in numeric):
            raise RuntimeError("Medical OPD training ledger contains NaN or infinity")
        trainable = int(row["trainable_mask_tokens"])
        zero = int(row["zero_advantage_completion_tokens"])
        token_count = float(row["trainer_metrics"]["token_count"])
        if token_count not in {float(trainable), float(trainable - zero)}:
            raise RuntimeError("Medical OPD mask/token_count identity failed")
    return state_path, steps_path, state, steps


def build_medical_opd_curve_analysis(
    experiment: ExperimentConfig, *, output_path: Path
) -> dict[str, Any]:
    root = experiment.root
    paths = _evaluation_paths(root)
    loaded = {name: _load_evaluation(path) for name, path in paths.items()}
    _validate_evaluations(loaded)
    comparisons, bad_cases = _paired_curve(loaded)
    state_path, steps_path, state, steps = _validate_training(root)

    interval_bounds = (
        (1, 50),
        (51, 100),
        (101, 150),
        (151, 200),
        (201, 250),
        (251, 300),
    )
    intervals = [_training_interval(steps, lower, upper) for lower, upper in interval_bounds]
    cumulative_usage = {
        "student_prefill_tokens": 0,
        "student_sample_tokens": 0,
        "student_train_tokens": 0,
    }
    cumulative_cost: dict[str, float] = {}
    for interval, step in zip(intervals, CHECKPOINT_STEPS, strict=True):
        for key, value in interval["usage"].items():
            cumulative_usage[key] += int(value)
        interval["estimated_cny"] = _cost_from_usage(experiment, interval["usage"])
        cumulative_cost[str(step)] = _cost_from_usage(experiment, cumulative_usage)
    if abs(cumulative_cost["300"] - float(state["estimated_cny"])) > 1e-6:
        raise RuntimeError("Medical OPD step-ledger cost does not match final state")

    checkpoints = {
        int(item["step"]): item
        for item in state["checkpoints"]
        if int(item["step"]) in CHECKPOINT_STEPS
    }
    if set(checkpoints) != set(CHECKPOINT_STEPS) or any(
        item.get("permanent") is not True for item in checkpoints.values()
    ):
        raise RuntimeError("Medical OPD milestone checkpoint index is incomplete")

    scores = {name: _score(summary, rows) for name, (summary, rows) in loaded.items()}
    evaluation_cost = sum(
        float(loaded[f"OPD@{step}_{domain}"][0]["estimated_cny"])
        for step in CHECKPOINT_STEPS
        for domain in DATASETS
    )
    proxy_paths = (
        root
        / "runs"
        / "pipeline"
        / "eval"
        / "medical-opd-e2teacher-step50"
        / domain
        / "summary.json"
        for domain in ("medical100", "general100")
    )
    proxy_cost = sum(
        float(json.loads(path.read_text(encoding="utf-8"))["estimated_cny"]) for path in proxy_paths
    )
    result = {
        "status": "completed",
        "created_at": utc_now(),
        "scope": "fixed MedQA-zh600 and C-Eval non-med300 protocol diagnostics",
        "method": "MED-OPD-E2TEACHER",
        "scores": scores,
        "paired_comparisons": comparisons,
        "hash_only_bad_cases": bad_cases,
        "training": {
            "completed_steps": 300,
            "recoverable_step": 300,
            "intervals": intervals,
            "cumulative_estimated_cny": cumulative_cost,
            "usage": state["usage"],
            "all_steps_finite": True,
            "all_mask_identities_valid": True,
            "prompt_truncated_tokens": sum(int(row["prompt_truncated_tokens"]) for row in steps),
        },
        "checkpoints": {str(step): checkpoints[step] for step in CHECKPOINT_STEPS},
        "cost": {
            "price_version": state["price_version"],
            "training_estimated_cny": float(state["estimated_cny"]),
            "six_medical600_and_ceval300_evaluations_estimated_cny": evaluation_cost,
            "proxy100_step50_estimated_cny": proxy_cost,
            "new_branch_total_estimated_cny": round(
                float(state["estimated_cny"]) + evaluation_cost + proxy_cost, 6
            ),
            "actual_billed_cny": None,
            "cache_note": (
                "PyTRIO 0.2.8 and the 2026-08-15 price table support cached prefill, but "
                "SampleResponse exposes no cached-token counter. Local estimates conservatively "
                "price every input token as ordinary prefill; reconcile against Usage."
            ),
        },
        "decision": {
            "best_observed_medical_opd_checkpoint": "OPD@100",
            "best_selection_status": "post-hoc exploratory",
            "pause_at_step": 300,
            "next_paid_action": "none_until_curve_review_then_SAR_from_OPD300",
            "protocol_alignment_note": (
                "The frozen Medical OPD stage is now complete. A later SAR stage must "
                "start from this Medical OPD@300 optimizer state, not a post-hoc earlier peak."
            ),
        },
        "claim": (
            "Epoch-2-teacher Medical OPD produced a large end-to-end medical gain and a large "
            "general-capability loss by step50. Step100 was the best observed checkpoint through "
            "step300; additional Medical OPD updates did not improve either fixed diagnostic. "
            "Much of the medical gain is associated with reliable answer termination rather "
            "than a demonstrated uniform gain on mutually untruncated rows."
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
