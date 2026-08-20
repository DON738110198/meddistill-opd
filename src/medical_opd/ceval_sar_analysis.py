from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, read_jsonl, sha256_file, utc_now
from medical_opd.reporting import exact_mcnemar


def _load_evaluation(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = path / "summary.json"
    predictions_path = path / "predictions.jsonl"
    if not summary_path.exists() or not predictions_path.exists():
        raise RuntimeError(f"evaluation artifact is incomplete: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(predictions_path)
    if summary.get("status") != "completed" or int(summary.get("count", -1)) != len(rows):
        raise RuntimeError(f"evaluation summary/count drift: {path}")
    return summary, rows


def _paired(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left_by_id = {str(row["id"]): row for row in left}
    right_by_id = {str(row["id"]): row for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise RuntimeError("paired evaluation IDs must be unique")
    if left_by_id.keys() != right_by_id.keys():
        raise RuntimeError("paired evaluation ID sets differ")
    ordered_left: list[dict[str, Any]] = []
    ordered_right: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for row_id in sorted(left_by_id):
        left_row = left_by_id[row_id]
        right_row = right_by_id[row_id]
        for field in ("gold", "question_sha256", "prompt_token_sha256"):
            if left_row.get(field) != right_row.get(field):
                raise RuntimeError(f"paired evaluation differs at {field}: {row_id}")
        ordered_left.append(left_row)
        ordered_right.append(right_row)
        if bool(left_row["correct"]) != bool(right_row["correct"]):
            cases.append(
                {
                    "id": row_id,
                    "subject": right_row.get("subject"),
                    "gold": right_row.get("gold"),
                    "left_prediction": left_row.get("prediction"),
                    "right_prediction": right_row.get("prediction"),
                    "left_correct": bool(left_row["correct"]),
                    "right_correct": bool(right_row["correct"]),
                    "left_truncated": bool(left_row.get("truncated", False)),
                    "right_truncated": bool(right_row.get("truncated", False)),
                    "left_thinking_closed": bool(left_row.get("thinking_closed", False)),
                    "right_thinking_closed": bool(right_row.get("thinking_closed", False)),
                    "left_output_tokens": int(
                        left_row.get("usage", {}).get("student_sample_tokens", 0)
                    ),
                    "right_output_tokens": int(
                        right_row.get("usage", {}).get("student_sample_tokens", 0)
                    ),
                    "question_sha256": right_row.get("question_sha256"),
                }
            )
    test = exact_mcnemar(ordered_left, ordered_right)
    test["left_correct"] = sum(bool(row["correct"]) for row in ordered_left)
    test["right_correct"] = sum(bool(row["correct"]) for row in ordered_right)
    test["right_minus_left_percentage_points"] = 100 * (
        test["right_correct"] - test["left_correct"]
    ) / len(ordered_left)
    return test, cases


def _training_ranges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges = (("1-10", 1, 10), ("11-25", 11, 25), ("26-50", 26, 50))
    result: list[dict[str, Any]] = []
    for label, lower, upper in ranges:
        selected = [row for row in rows if lower <= int(row["step"]) <= upper]
        completions = sum(int(row["completion_tokens"]) for row in selected)
        datums = sum(int(row["datums"]) for row in selected)
        losses = [float(row["trainer_metrics"]["loss_mean"]) for row in selected]
        kls = [float(row["reverse_kl_mean"]) for row in selected]
        result.append(
            {
                "range": label,
                "steps": len(selected),
                "completion_tokens": completions,
                "completion_tokens_per_rollout": completions / datums,
                "loss_min": min(losses),
                "loss_max": max(losses),
                "reverse_kl_mean": sum(kls) / len(kls),
                "reverse_kl_max": max(kls),
                "prompt_truncated_tokens": sum(
                    int(row["prompt_truncated_tokens"]) for row in selected
                ),
                "trainable_mask_tokens": sum(
                    int(row["trainable_mask_tokens"]) for row in selected
                ),
                "zero_advantage_tokens": sum(
                    int(row["zero_advantage_completion_tokens"]) for row in selected
                ),
            }
        )
    return result


def build_ceval_sar_analysis(
    experiment: ExperimentConfig, *, output_path: Path
) -> dict[str, Any]:
    root = experiment.root
    eval_root = root / "runs" / "pipeline" / "eval"
    paths = {
        "m0_medical": root / "runs" / "proxy-thinking-v1" / "screening" / "M0" / "medical100",
        "m0_general": root / "runs" / "proxy-thinking-v1" / "screening" / "M0" / "general100",
        "opd_medical": eval_root / "medical-opd-e2teacher-step25" / "medical100",
        "opd_general": eval_root / "medical-opd-e2teacher-step25" / "general100",
        "alpaca_sar_medical": eval_root / "sar-e2teacher-step25" / "medical100",
        "alpaca_sar_general": eval_root / "sar-e2teacher-step25" / "general100",
        "ceval_sar_medical": eval_root / "sar-ceval-e2teacher-step50" / "medical",
        "ceval_sar_general": eval_root / "sar-ceval-e2teacher-step50" / "general",
    }
    loaded = {name: _load_evaluation(path) for name, path in paths.items()}
    comparisons: dict[str, Any] = {}
    bad_cases: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("medical", "general"):
        for left_name in ("m0", "opd", "alpaca_sar"):
            key = f"{left_name}_to_ceval_sar_{dataset}"
            test, cases = _paired(
                loaded[f"{left_name}_{dataset}"][1],
                loaded[f"ceval_sar_{dataset}"][1],
            )
            comparisons[key] = test
            bad_cases[key] = cases

    run_dir = root / "runs" / "pipeline" / "BASE-SAR-E2TEACHER"
    state_path = run_dir / "state.json"
    steps_path = run_dir / "steps.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = read_jsonl(steps_path)
    if (
        state.get("status") != "completed"
        or int(state.get("completed_steps", -1)) != 50
        or [int(row["step"]) for row in steps] != list(range(1, 51))
    ):
        raise RuntimeError("C-Eval SAR@50 training artifact is incomplete")
    for row in steps:
        numeric = [
            float(row["reverse_kl_mean"]),
            float(row["reverse_kl_std"]),
            *(float(value) for value in row["trainer_metrics"].values()),
        ]
        if any(not math.isfinite(value) for value in numeric):
            raise RuntimeError("C-Eval SAR training contains NaN or infinity")
        trainable = int(row["trainable_mask_tokens"])
        zero = int(row["zero_advantage_completion_tokens"])
        token_count = float(row["trainer_metrics"]["token_count"])
        if token_count not in {float(trainable), float(trainable - zero)}:
            raise RuntimeError("C-Eval SAR training mask identity failed")

    training_cost = float(state["estimated_cny"])
    medical_cost = float(loaded["ceval_sar_medical"][0]["estimated_cny"])
    general_cost = float(loaded["ceval_sar_general"][0]["estimated_cny"])
    general_correct = int(loaded["ceval_sar_general"][0]["correct"])
    retention_floor = 77
    allow_extension = general_correct >= retention_floor
    result = {
        "status": "completed",
        "created_at": utc_now(),
        "scope": "frozen proxy100 only; no 600/300 or full evaluation",
        "method": "BASE-SAR",
        "checkpoint": state["checkpoints"][-1],
        "scores": {
            name: {
                "correct": int(summary["correct"]),
                "count": int(summary["count"]),
                "accuracy": float(summary["accuracy"]),
                "format_valid_rate": float(summary["format_valid_rate"]),
                "thinking_closed_rate": float(summary["thinking_closed_rate"]),
                "truncation_rate": float(summary["truncation_rate"]),
                "output_tokens_mean": float(summary["output_tokens_mean"]),
            }
            for name, (summary, _) in loaded.items()
        },
        "paired_comparisons": comparisons,
        "bad_cases": bad_cases,
        "training_diagnostics": {
            "ranges": _training_ranges(steps),
            "usage": state["usage"],
            "role_usage": state["role_usage"],
            "all_steps_finite": True,
            "all_mask_identities_valid": True,
            "prompt_truncated_tokens": sum(
                int(row["prompt_truncated_tokens"]) for row in steps
            ),
        },
        "cost": {
            "price_version": state["price_version"],
            "training_estimated_cny": training_cost,
            "medical_proxy_estimated_cny": medical_cost,
            "general_proxy_estimated_cny": general_cost,
            "branch_total_estimated_cny": training_cost + medical_cost + general_cost,
            "actual_billed_cny": None,
            "billing_note": "Reconcile token ledger against https://pytrio.cn/usage.",
        },
        "decision": {
            "general_retention_floor_correct_of_100": retention_floor,
            "observed_general_correct_of_100": general_correct,
            "allow_extension": allow_extension,
            "action": "stop_no_more_paid_steps_or_full_evaluation",
            "reasons": [
                "general proxy is below the preregistered M0-minus-2pp retention floor",
                "medical proxy fell 2pp from the Medical OPD@25 source checkpoint",
                "rollout length and proxy truncation increased sharply despite falling KL",
            ],
        },
        "claim": (
            "Leakage-safe C-Eval Base-anchor SAR@50 recovered only 1pp "
            "general proxy accuracy over Medical OPD@25 while losing 2pp medical accuracy; "
            "it failed the retention gate and increased answer-termination failures."
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
