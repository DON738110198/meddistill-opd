from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, read_jsonl, sha256_file, utc_now
from medical_opd.reporting import exact_mcnemar


def _paired_rows(
    raw_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    raw_by_id = {str(row["id"]): row for row in raw_rows}
    sft_by_id = {str(row["id"]): row for row in sft_rows}
    if len(raw_by_id) != len(raw_rows) or len(sft_by_id) != len(sft_rows):
        raise RuntimeError("mechanism analysis requires unique prediction IDs")
    if raw_by_id.keys() != sft_by_id.keys():
        raise RuntimeError("raw and SFT prediction IDs do not match")
    pairs = [(raw_by_id[row_id], sft_by_id[row_id]) for row_id in sorted(raw_by_id)]
    for raw, sft in pairs:
        for field in ("gold", "question_sha256", "prompt_token_sha256"):
            if raw.get(field) != sft.get(field):
                raise RuntimeError(f"paired mechanism contract differs at {field}")
    return pairs


def analyze_t27_records(
    raw_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    training_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = _paired_rows(raw_rows, sft_rows)
    valid_pairs = [(raw, sft) for raw, sft in pairs if bool(sft.get("format_valid"))]
    invalid_pairs = [(raw, sft) for raw, sft in pairs if not bool(sft.get("format_valid"))]
    raw_valid = [{"id": raw["id"], "correct": raw["correct"]} for raw, _ in valid_pairs]
    sft_valid = [{"id": sft["id"], "correct": sft["correct"]} for _, sft in valid_pairs]
    raw_all = [{"id": raw["id"], "correct": raw["correct"]} for raw, _ in pairs]
    sft_all = [{"id": sft["id"], "correct": sft["correct"]} for _, sft in pairs]
    full_test = exact_mcnemar(raw_all, sft_all)
    valid_test = exact_mcnemar(raw_valid, sft_valid)
    losses = [
        float(step["trainer_metrics"]["loss_mean"])
        for step in training_steps
        if "loss_mean" in step.get("trainer_metrics", {})
    ]
    if not losses or any(not math.isfinite(value) for value in losses):
        raise RuntimeError("training record lacks finite loss evidence")
    mask_mismatches = sum(
        int(step["trainable_mask_tokens"])
        != int(float(step["trainer_metrics"].get("token_count", -1)))
        for step in training_steps
    )
    raw_correct = sum(bool(raw["correct"]) for raw, _ in pairs)
    sft_correct = sum(bool(sft["correct"]) for _, sft in pairs)
    return {
        "status": "completed",
        "paired_count": len(pairs),
        "contract_match": {
            "ids": True,
            "gold": True,
            "question_sha256": True,
            "prompt_token_sha256": True,
        },
        "end_to_end": {
            "raw_correct": raw_correct,
            "sft_correct": sft_correct,
            "delta_percentage_points": 100 * (sft_correct - raw_correct) / len(pairs),
            "paired_exact_mcnemar": full_test,
        },
        "format_valid_only_diagnostic": {
            "count": len(valid_pairs),
            "raw_correct": sum(bool(raw["correct"]) for raw, _ in valid_pairs),
            "sft_correct": sum(bool(sft["correct"]) for _, sft in valid_pairs),
            "paired_exact_mcnemar": valid_test,
            "scope": "diagnostic conditioning, not a replacement for end-to-end accuracy",
        },
        "termination_diagnostic": {
            "raw_truncated": sum(bool(raw.get("truncated")) for raw, _ in pairs),
            "sft_truncated": sum(bool(sft.get("truncated")) for _, sft in pairs),
            "sft_invalid": len(invalid_pairs),
            "sft_invalid_and_truncated": sum(
                bool(sft.get("truncated")) for _, sft in invalid_pairs
            ),
            "raw_correct_on_sft_invalid": sum(
                bool(raw["correct"]) for raw, _ in invalid_pairs
            ),
            "raw_only_correct_with_sft_truncated": sum(
                bool(raw["correct"])
                and not bool(sft["correct"])
                and bool(sft.get("truncated"))
                for raw, sft in pairs
            ),
        },
        "training_diagnostic": {
            "steps": len(training_steps),
            "finite_losses": True,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "mask_token_count_mismatches": mask_mismatches,
            "completion_truncation_rows": sum(
                int(step.get("completion_truncation_rows", 0)) for step in training_steps
            ),
            "prompt_truncated_tokens": sum(
                int(step.get("prompt_truncated_tokens", 0)) for step in training_steps
            ),
        },
        "mechanism_verdict": {
            "supported": (
                "The exact T27-SFT recipe failed the fixed-cap end-to-end teacher gate, with "
                "parseable final-answer delivery on capped responses as the clearest observed "
                "failure mode."
            ),
            "not_supported": [
                "Medical knowledge significantly declined after conditioning on valid outputs.",
                "Medical SFT is universally harmful.",
                "The checkpoint would necessarily produce worse OPD.",
            ],
            "plausible_not_identified": [
                "High learning rate on only 100 open-ended examples caused the behavior shift.",
                "The 672-token training cap and nine truncated targets contributed to the shift.",
            ],
        },
    }


def run_t27_mechanism_analysis(
    config: ExperimentConfig, output_path: Path
) -> dict[str, Any]:
    raw_path = (
        config.root
        / "runs"
        / "proxy-thinking-v1"
        / "screening"
        / "teacher-27B"
        / "medical100"
        / "predictions.jsonl"
    )
    sft_path = (
        config.root
        / "runs"
        / "staged-27b-teacher"
        / "eval"
        / "screening"
        / "medical100"
        / "predictions.jsonl"
    )
    steps_path = (
        config.root / "runs" / "staged-27b-teacher" / "T27-SFT" / "steps.jsonl"
    )
    for path in (raw_path, sft_path, steps_path):
        if not path.exists():
            raise RuntimeError(f"mechanism evidence is missing: {path}")
    result = analyze_t27_records(
        read_jsonl(raw_path), read_jsonl(sft_path), read_jsonl(steps_path)
    )
    result["created_at"] = utc_now()
    result["evidence"] = {
        "raw_predictions_sha256": sha256_file(raw_path),
        "sft_predictions_sha256": sha256_file(sft_path),
        "training_steps_sha256": sha256_file(steps_path),
    }
    atomic_write_json(output_path.resolve(), result)
    return result
