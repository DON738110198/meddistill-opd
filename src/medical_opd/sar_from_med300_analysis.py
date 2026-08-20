from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from medical_opd.ceval_sar_analysis import _load_evaluation, _paired
from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, read_jsonl, sha256_file, utc_now
from medical_opd.sar_curve_analysis import _both_untruncated_pair, _score, _training_interval


def _paths(root: Path) -> dict[str, Path]:
    return {
        "m0_medical": root / "runs/reference-protocol/M0/medical600",
        "opd300_medical": root
        / "runs/pipeline/reference-eval/medical-opd-e2teacher-step300/medical600",
        "sar50_medical": root
        / "runs/pipeline/reference-eval/sar-from-med300-step050/medical600",
        "m0_general": root / "runs/proxy-thinking-v1/screening/M0/general100",
        "opd300_general": root
        / "runs/pipeline/eval/medical-opd-e2teacher-step300/general100",
        "sar50_general": root
        / "runs/pipeline/eval/sar-from-med300-step050/general100",
        "m0_ceval300": root / "runs/reference-protocol/M0/ceval300",
        "opd300_ceval300": root
        / "runs/pipeline/reference-eval/medical-opd-e2teacher-step300/ceval300",
        "sar50_ceval300": root
        / "runs/pipeline/reference-eval/sar-from-med300-step050/ceval300",
    }


def _validate_group(
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    names: tuple[str, ...],
    expected_count: int,
) -> None:
    baseline_summary, baseline_rows = loaded[names[0]]
    baseline_ids = [str(row["id"]) for row in baseline_rows]
    if len(baseline_ids) != expected_count or len(set(baseline_ids)) != expected_count:
        raise RuntimeError(f"{names[0]} is not a complete unique evaluation set")
    for name in names:
        summary, rows = loaded[name]
        if (
            summary.get("status") != "completed"
            or int(summary.get("count", -1)) != expected_count
            or str(summary.get("dataset_sha256"))
            != str(baseline_summary.get("dataset_sha256"))
            or {str(row["id"]) for row in rows} != set(baseline_ids)
        ):
            raise RuntimeError(f"evaluation contract drift: {name}")
        _paired(baseline_rows, rows)


def _hash_only_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in cases:
        clean = dict(case)
        if clean.get("question_sha256") is None:
            clean.pop("question_sha256", None)
        result.append(clean)
    return result


def _comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test, cases = _paired(left, right)
    return {
        **test,
        "both_untruncated": _both_untruncated_pair(left, right),
    }, _hash_only_cases(cases)


def _subset_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    untruncated = [row for row in rows if not bool(row.get("truncated", False))]
    count = len(rows)
    return {
        "correct": sum(bool(row["correct"]) for row in rows),
        "count": count,
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "format_valid_rate": sum(bool(row.get("format_valid")) for row in rows) / count,
        "thinking_closed_rate": sum(bool(row.get("thinking_closed")) for row in rows) / count,
        "truncation_rate": sum(bool(row.get("truncated")) for row in rows) / count,
        "untruncated_count": len(untruncated),
        "untruncated_correct": sum(bool(row["correct"]) for row in untruncated),
    }


def _validate_training(root: Path) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    run_dir = root / "runs/pipeline/BASE-SAR-FROM-MED300"
    state_path = run_dir / "state.json"
    steps_path = run_dir / "steps.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = read_jsonl(steps_path)
    if (
        state.get("status") != "completed"
        or int(state.get("completed_steps", -1)) != 50
        or int(state.get("recoverable_step", -1)) != 50
        or int(state.get("uncheckpointed_steps", -1)) != 0
        or [int(row["step"]) for row in steps] != list(range(1, 51))
    ):
        raise RuntimeError("SAR-from-Medical-OPD@300 training state is incomplete")
    final = [item for item in state.get("checkpoints", []) if int(item["step"]) == 50]
    if len(final) != 1 or final[0].get("permanent") is not True:
        raise RuntimeError("SAR@50 permanent checkpoint is missing")
    for row in steps:
        numeric = [
            float(row["reverse_kl_mean"]),
            float(row["reverse_kl_std"]),
            *(float(value) for value in row["trainer_metrics"].values()),
        ]
        if any(not math.isfinite(value) for value in numeric):
            raise RuntimeError("SAR training contains NaN or infinity")
        trainable = int(row["trainable_mask_tokens"])
        zero = int(row["zero_advantage_completion_tokens"])
        token_count = float(row["trainer_metrics"]["token_count"])
        if token_count not in {float(trainable), float(trainable - zero)}:
            raise RuntimeError("SAR training mask/token_count identity failed")
    return state_path, steps_path, state, steps


def build_sar_from_med300_analysis(
    experiment: ExperimentConfig, *, output_path: Path
) -> dict[str, Any]:
    root = experiment.root
    paths = _paths(root)
    loaded = {name: _load_evaluation(path) for name, path in paths.items()}
    _validate_group(
        loaded, ("m0_medical", "opd300_medical", "sar50_medical"), 600
    )
    _validate_group(
        loaded, ("m0_general", "opd300_general", "sar50_general"), 100
    )
    _validate_group(
        loaded, ("m0_ceval300", "opd300_ceval300", "sar50_ceval300"), 300
    )

    audit_path = root / "reports/generated/sar_from_med300_eval_overlap_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schedule_path = root / str(audit["sar_source"]["path"])
    if (
        audit.get("status") != "frozen_before_paid_evaluation"
        or sha256_file(schedule_path) != audit["sar_source"]["sha256"]
        or int(audit["reference_ceval_300"]["near_only_overlap_at_threshold_0_85"]) != 0
    ):
        raise RuntimeError("SAR/C-Eval overlap audit is missing or stale")
    training_ids = {str(row["source_row_id"]) for row in read_jsonl(schedule_path)}
    overlap_ids = set(audit["reference_ceval_300"]["exact_overlap_ids"])
    ceval_ids = {str(row["id"]) for row in loaded["sar50_ceval300"][1]}
    if training_ids & ceval_ids != overlap_ids or len(overlap_ids) != 19:
        raise RuntimeError("frozen C-Eval overlap list does not match the evaluated rows")

    comparisons: dict[str, Any] = {}
    bad_cases: dict[str, list[dict[str, Any]]] = {}
    for domain in ("medical", "general", "ceval300"):
        for left_name in ("m0", "opd300"):
            key = f"{left_name}_to_sar50_{domain}"
            comparisons[key], bad_cases[key] = _comparison(
                loaded[f"{left_name}_{domain}"][1], loaded[f"sar50_{domain}"][1]
            )

    clean_rows: dict[str, list[dict[str, Any]]] = {
        name: [row for row in loaded[f"{name}_ceval300"][1] if str(row["id"]) not in overlap_ids]
        for name in ("m0", "opd300", "sar50")
    }
    if any(len(rows) != 281 for rows in clean_rows.values()):
        raise RuntimeError("C-Eval overlap-free sensitivity set must contain 281 rows")
    for left_name in ("m0", "opd300"):
        key = f"{left_name}_to_sar50_ceval281_overlap_free"
        comparisons[key], bad_cases[key] = _comparison(
            clean_rows[left_name], clean_rows["sar50"]
        )

    state_path, steps_path, state, steps = _validate_training(root)
    scores = {
        name: _score(summary, rows) for name, (summary, rows) in loaded.items()
    }
    scores.update(
        {f"{name}_ceval281_overlap_free": _subset_score(rows) for name, rows in clean_rows.items()}
    )
    medical_correct = int(loaded["sar50_medical"][0]["correct"])
    general_correct = int(loaded["sar50_general"][0]["correct"])
    medical_floor = int(audit["registered_gate"]["medical"]["minimum_correct"])
    general_floor = int(audit["registered_gate"]["general"]["minimum_correct"])
    allow_extension = medical_correct >= medical_floor and general_correct >= general_floor
    sar_eval_cost = sum(
        float(loaded[name][0]["estimated_cny"])
        for name in ("sar50_medical", "sar50_general", "sar50_ceval300")
    )
    source_eval_cost = float(loaded["opd300_general"][0]["estimated_cny"])
    result = {
        "status": "completed",
        "created_at": utc_now(),
        "scope": (
            "Medical OPD@300 -> frozen C-Eval Base-anchor SAR@50; "
            "MedQA600 and frozen general100 are gates, mixed C-Eval300 is diagnostic only"
        ),
        "method": "BASE-SAR-FROM-MED300",
        "checkpoint": [
            item for item in state["checkpoints"] if int(item["step"]) == 50
        ][0],
        "scores": scores,
        "paired_comparisons": comparisons,
        "hash_only_bad_cases": bad_cases,
        "training": {
            "source_medical_opd_step": state["training_contract"]["source_medical_opd_step"],
            "completed_steps": 50,
            "recoverable_step": 50,
            "all_steps_finite": True,
            "all_mask_identities_valid": True,
            "prompt_truncated_tokens": sum(
                int(row["prompt_truncated_tokens"]) for row in steps
            ),
            "intervals": [
                _training_interval(steps, 1, 10),
                _training_interval(steps, 11, 25),
                _training_interval(steps, 26, 50),
            ],
            "usage": state["usage"],
            "role_usage": state["role_usage"],
        },
        "leakage_boundary": {
            "general_gate_dataset": "frozen general_proxy100 with zero exact/near overlap",
            "ceval300_exact_training_overlap": 19,
            "ceval300_near_only_overlap": 0,
            "ceval300_role": "descriptive only",
            "ceval281_role": "overlap-free sensitivity only",
        },
        "cost": {
            "price_version": state["price_version"],
            "training_estimated_cny": float(state["estimated_cny"]),
            "sar_evaluations_estimated_cny": sar_eval_cost,
            "sar_branch_total_estimated_cny": float(state["estimated_cny"])
            + sar_eval_cost,
            "source_opd300_general_gate_estimated_cny": source_eval_cost,
            "decision_package_total_estimated_cny": float(state["estimated_cny"])
            + sar_eval_cost
            + source_eval_cost,
            "actual_billed_cny": None,
            "billing_note": "Reconcile token estimates against https://pytrio.cn/usage.",
        },
        "decision": {
            "medical_gate": {
                "minimum_correct": medical_floor,
                "observed_correct": medical_correct,
                "passed": medical_correct >= medical_floor,
            },
            "general_gate": {
                "minimum_correct": general_floor,
                "observed_correct": general_correct,
                "passed": general_correct >= general_floor,
            },
            "allow_extension_beyond_step50": allow_extension,
            "action": "continue_to_step100" if allow_extension else "stop_at_step50",
            "reason": (
                "both registered medical and general gates passed"
                if allow_extension
                else "the frozen general-retention gate failed"
            ),
        },
        "claim": (
            "C-Eval Base-anchor SAR@50 improved Medical OPD@300 on MedQA600 and partially "
            "recovered general accuracy, but it remained below M0 on both clean general views, "
            "increased truncation, and failed the preregistered retention gate."
        ),
        "artifacts": {
            "training_state_sha256": sha256_file(state_path),
            "training_steps_sha256": sha256_file(steps_path),
            "overlap_audit_sha256": sha256_file(audit_path),
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
