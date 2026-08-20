from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from medical_opd.config import ExperimentConfig
from medical_opd.evaluation import DATASETS
from medical_opd.io_utils import atomic_write_json, read_jsonl, sha256_file, utc_now

SCREENING_RUNS = {
    "M0": ("M0", None),
    "teacher-27B": ("teacher-27B", None),
    "M2@50": ("M2-50", "runs/smoke-calibration/M2/state.json"),
    "M3@50": ("M3-50", "runs/smoke-calibration/M3/state.json"),
    "M4@50": ("M4-50", "runs/smoke-calibration/M4/state.json"),
    "M4@100": ("M4-100", "runs/matched-budget/M4-100/summary.json"),
    "M5@50": ("M5-50", None),
    "M5@100": ("M5-100", "runs/smoke-calibration/M5/state.json"),
    "M5@200": ("M5-200", "runs/extended/M5-200/summary.json"),
}

FULL_SELECTION = ("M0", "M4@50", "M5@50", "M5@200")
FULL_RUNS = {
    "M0": "M0",
    "M4@50": "M4-50",
    "M5@50": "M5-50",
    "M5@200": "M5-200",
}
FULL_DOMAINS = {
    "medical": ("medical_full", "medical3426", 3426),
    "general": ("general_full", "general1925", 1925),
}


def _summary_path(config: ExperimentConfig, run_dir: str, domain: str) -> Path:
    return (
        config.root
        / "runs"
        / "proxy-thinking-v1"
        / "screening"
        / run_dir
        / f"{domain}100"
        / "summary.json"
    )


def _predictions_path(config: ExperimentConfig, run_dir: str, domain: str) -> Path:
    return _summary_path(config, run_dir, domain).with_name("predictions.jsonl")


def exact_mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {str(row["id"]): bool(row["correct"]) for row in left}
    right_by_id = {str(row["id"]): bool(row["correct"]) for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise RuntimeError("paired comparison contains duplicate row IDs")
    if left_by_id.keys() != right_by_id.keys():
        raise RuntimeError("paired comparison row IDs do not match")
    left_only = sum(left_by_id[key] and not right_by_id[key] for key in left_by_id)
    right_only = sum(right_by_id[key] and not left_by_id[key] for key in left_by_id)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "count": len(left_by_id),
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def _usage_cost(config: ExperimentConfig, usage: dict[str, Any]) -> float:
    values = {
        "student_prefill_tokens": float(
            config.get("pricing", "student_prefill_cny_per_million")
        ),
        "student_sample_tokens": float(
            config.get("pricing", "student_sample_cny_per_million")
        ),
        "student_train_tokens": float(
            config.get("pricing", "student_train_cny_per_million")
        ),
        "teacher_prefill_tokens": float(
            config.get("pricing", "teacher_prefill_cny_per_million")
        ),
        "teacher_sample_tokens": float(
            config.get("pricing", "teacher_sample_cny_per_million")
        ),
        "teacher_train_tokens": float(
            config.get("pricing", "teacher_train_cny_per_million")
        ),
    }
    return sum(float(usage.get(key, 0)) * price for key, price in values.items()) / 1_000_000


def _sum_step_usage(path: Path, maximum_step: int) -> dict[str, float]:
    totals = {
        "optimizer_steps": 0.0,
        "student_prefill_tokens": 0.0,
        "student_sample_tokens": 0.0,
        "student_train_tokens": 0.0,
        "teacher_prefill_tokens": 0.0,
        "teacher_sample_tokens": 0.0,
        "teacher_train_tokens": 0.0,
        "wall_seconds": 0.0,
    }
    for row in read_jsonl(path):
        if int(row["step"]) > maximum_step:
            continue
        for key in totals:
            totals[key] += float(row["usage"].get(key, 0))
    return totals


def _training_record(config: ExperimentConfig, label: str, state_rel: str | None) -> Any:
    if label in {"M0", "teacher-27B"}:
        return None
    if label == "M5@50":
        usage = _sum_step_usage(
            config.root / "runs" / "smoke-calibration" / "M5" / "steps.jsonl", 50
        )
        return {
            "optimizer_steps": 50,
            "usage": usage,
            "estimated_cny": _usage_cost(config, usage),
            "checkpoint_permanent": True,
        }
    assert state_rel is not None
    state_path = config.root / state_rel
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "optimizer_steps": int(state["completed_steps"]),
        "usage": state["usage"],
        "estimated_cny": float(state.get("estimated_cny", _usage_cost(config, state["usage"]))),
        "state_path": state_rel,
        "state_sha256": sha256_file(state_path),
    }


def _full_forecast(
    config: ExperimentConfig, label: str, summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts = {"medical": 3426, "general": 1925}
    datasets: dict[str, Any] = {}
    for domain, full_count in counts.items():
        summary = summaries[domain]
        ratio = full_count / int(summary["count"])
        prefill = math.ceil(float(summary["usage"]["student_prefill_tokens"]) * ratio)
        sample = math.ceil(float(summary["usage"]["student_sample_tokens"]) * ratio)
        wall = float(summary["wall_seconds"]) * ratio
        usage = {"student_prefill_tokens": prefill, "student_sample_tokens": sample}
        datasets[domain] = {
            "proxy_source": str(_summary_path(config, SCREENING_RUNS[label][0], domain)),
            "full_count": full_count,
            "extrapolated_usage": usage,
            "estimated_cny": _usage_cost(config, usage),
            "estimated_wall_seconds": wall,
        }
    return {
        "datasets": datasets,
        "estimated_cny": sum(value["estimated_cny"] for value in datasets.values()),
        "estimated_wall_seconds": sum(
            value["estimated_wall_seconds"] for value in datasets.values()
        ),
    }


def build_screening_report(config: ExperimentConfig, output_path: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label, (run_dir, state_rel) in SCREENING_RUNS.items():
        summaries: dict[str, dict[str, Any]] = {}
        for domain in ("medical", "general"):
            path = _summary_path(config, run_dir, domain)
            if not path.exists():
                raise RuntimeError(f"missing screening summary: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("protocol_id") != config.get(
                "evaluation", "thinking_protocol_id"
            ):
                raise RuntimeError(f"stale screening protocol: {path}")
            summaries[domain] = summary
            predictions[(label, domain)] = read_jsonl(
                _predictions_path(config, run_dir, domain)
            )
        records[label] = {
            "medical_accuracy": summaries["medical"]["accuracy"],
            "general_accuracy": summaries["general"]["accuracy"],
            "equal_weight_mean_accuracy": (
                float(summaries["medical"]["accuracy"])
                + float(summaries["general"]["accuracy"])
            )
            / 2,
            "evaluation_estimated_cny": sum(
                float(value["estimated_cny"]) for value in summaries.values()
            ),
            "evaluation": {
                domain: {
                    key: summaries[domain][key]
                    for key in (
                        "accuracy",
                        "correct",
                        "count",
                        "format_valid_rate",
                        "thinking_closed_rate",
                        "truncation_rate",
                        "estimated_cny",
                        "wilson_95",
                    )
                }
                for domain in summaries
            },
            "training": _training_record(config, label, state_rel),
        }
    base = records["M0"]
    for record in records.values():
        record["medical_delta_pp_vs_m0"] = 100 * (
            float(record["medical_accuracy"]) - float(base["medical_accuracy"])
        )
        record["general_delta_pp_vs_m0"] = 100 * (
            float(record["general_accuracy"]) - float(base["general_accuracy"])
        )

    comparisons: dict[str, Any] = {}
    for left, right in (
        ("M5@50", "M4@50"),
        ("M5@100", "M4@100"),
        ("M5@50", "M5@100"),
        ("M5@100", "M5@200"),
        ("M5@50", "M5@200"),
    ):
        comparisons[f"{left}_vs_{right}"] = {
            domain: exact_mcnemar(
                predictions[(left, domain)], predictions[(right, domain)]
            )
            for domain in ("medical", "general")
        }

    simple_labels = ("M2@50", "M3@50", "M4@50", "M4@100")
    best_simple = max(
        simple_labels, key=lambda value: records[value]["equal_weight_mean_accuracy"]
    )
    full_forecasts = {
        label: _full_forecast(
            config,
            label,
            {
                domain: json.loads(
                    _summary_path(config, SCREENING_RUNS[label][0], domain).read_text(
                        encoding="utf-8"
                    )
                )
                for domain in ("medical", "general")
            },
        )
        for label in FULL_SELECTION
    }
    report = {
        "status": "screening_complete_m5_300_stopped",
        "created_at": utc_now(),
        "protocol_id": config.get("evaluation", "thinking_protocol_id"),
        "student_model": config.get("models", "student"),
        "teacher_model": config.get("models", "teacher"),
        "records": records,
        "paired_exact_mcnemar": comparisons,
        "selection": {
            "best_simple_baseline": best_simple,
            "permanent_checkpoint": "M5@50",
            "m5_final": "M5@200",
            "full_evaluation": list(FULL_SELECTION),
        },
        "promotion_decision": {
            "M5@100_to_200": "ran because M5@100 was +4pp medical and -1pp general vs M0",
            "M5@200_to_300": "stopped because M5@200 medical was -5pp vs M0",
            "general_retention_at_200": "+2pp vs M0",
        },
        "full_evaluation_forecast": {
            "models": full_forecasts,
            "total_estimated_cny": sum(
                value["estimated_cny"] for value in full_forecasts.values()
            ),
            "total_estimated_wall_seconds_serial": sum(
                value["estimated_wall_seconds"] for value in full_forecasts.values()
            ),
            "warning": (
                "Linear extrapolation from each model's fixed 100-row proxy usage; "
                "not an invoice."
            ),
        },
    }
    atomic_write_json(output_path.resolve(), report)
    return report


def _full_path(config: ExperimentConfig, label: str, domain: str, name: str) -> Path:
    run_dir = FULL_RUNS[label]
    dataset_dir = FULL_DOMAINS[domain][1]
    return config.root / "runs" / "full-thinking-v1" / run_dir / dataset_dir / name


def _load_full_predictions(
    config: ExperimentConfig, label: str, domain: str
) -> list[dict[str, Any]]:
    return read_jsonl(_full_path(config, label, domain, "predictions.jsonl"))


def _paired_bad_cases(
    dataset: list[dict[str, Any]],
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> dict[str, Any]:
    dataset_by_id = {str(row["id"]): row for row in dataset}
    left_by_id = {str(row["id"]): row for row in left}
    right_by_id = {str(row["id"]): row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise RuntimeError("bad-case comparison IDs do not match")

    def compact(row_id: str) -> dict[str, Any]:
        source = dataset_by_id[row_id]
        left_row = left_by_id[row_id]
        right_row = right_by_id[row_id]
        return {
            "id": row_id,
            "source_index": source.get("source_index"),
            "source_split": source.get("source_split"),
            "subject": source.get("subject"),
            "category": source.get("category"),
            "question_sha256": left_row["question_sha256"],
            "gold": left_row["gold"],
            "left_prediction": left_row["prediction"],
            "right_prediction": right_row["prediction"],
            "left_truncated": left_row["truncated"],
            "right_truncated": right_row["truncated"],
            "left_thinking_closed": left_row["thinking_closed"],
            "right_thinking_closed": right_row["thinking_closed"],
        }

    regressions = sorted(
        row_id
        for row_id in left_by_id
        if left_by_id[row_id]["correct"] and not right_by_id[row_id]["correct"]
    )
    improvements = sorted(
        row_id
        for row_id in left_by_id
        if not left_by_id[row_id]["correct"] and right_by_id[row_id]["correct"]
    )
    category_net: dict[str, dict[str, int]] = {}
    for row_id in regressions + improvements:
        category = str(
            dataset_by_id[row_id].get("category")
            or dataset_by_id[row_id].get("subject")
            or "unknown"
        )
        bucket = category_net.setdefault(category, {"regressions": 0, "improvements": 0})
        if row_id in regressions:
            bucket["regressions"] += 1
        else:
            bucket["improvements"] += 1
    for bucket in category_net.values():
        bucket["net_right_minus_left"] = (
            bucket["improvements"] - bucket["regressions"]
        )
    return {
        "left_correct_right_wrong": len(regressions),
        "left_wrong_right_correct": len(improvements),
        "category_net": category_net,
        "regression_examples": [compact(row_id) for row_id in regressions[:limit]],
        "improvement_examples": [compact(row_id) for row_id in improvements[:limit]],
        "privacy_note": "No raw benchmark question or model output text is copied here.",
    }


def _sum_summary_costs(root: Path) -> tuple[float, int]:
    cost = 0.0
    count = 0
    for path in root.rglob("summary.json") if root.exists() else []:
        summary = json.loads(path.read_text(encoding="utf-8"))
        value = summary.get("estimated_cny")
        if isinstance(value, (int, float)):
            cost += float(value)
            count += 1
    return cost, count


def _current_cost_report(config: ExperimentConfig) -> dict[str, Any]:
    training_paths = {
        "M2@50": config.root / "runs" / "smoke-calibration" / "M2" / "state.json",
        "M3@50": config.root / "runs" / "smoke-calibration" / "M3" / "state.json",
        "M4@100": config.root / "runs" / "matched-budget" / "M4-100" / "summary.json",
        "M5@200": config.root / "runs" / "extended" / "M5-200" / "summary.json",
    }
    training = {
        label: float(json.loads(path.read_text(encoding="utf-8"))["estimated_cny"])
        for label, path in training_paths.items()
    }
    evaluation_roots = {
        "reference_protocol": config.root / "runs" / "reference-protocol",
        "superseded_non_thinking_proxy": config.root / "runs" / "proxy",
        "superseded_non_thinking_full": config.root / "runs" / "full",
        "thinking_proxy": config.root / "runs" / "proxy-thinking-v1",
        "thinking_full_completed": config.root / "runs" / "full-thinking-v1",
    }
    evaluations: dict[str, Any] = {}
    for label, root in evaluation_roots.items():
        cost, summaries = _sum_summary_costs(root)
        evaluations[label] = {"estimated_cny": cost, "completed_summaries": summaries}
    partial_path = _full_path(config, "M5@200", "general", "predictions.jsonl")
    partial = read_jsonl(partial_path)
    partial_usage = {
        "student_prefill_tokens": sum(
            int(row["usage"]["student_prefill_tokens"]) for row in partial
        ),
        "student_sample_tokens": sum(
            int(row["usage"]["student_sample_tokens"]) for row in partial
        ),
    }
    partial_summary_exists = _full_path(
        config, "M5@200", "general", "summary.json"
    ).exists()
    partial_cost = 0.0 if partial_summary_exists else _usage_cost(config, partial_usage)
    probe = json.loads(
        (config.root / "reports" / "generated" / "ppo_mask_probe.json").read_text(
            encoding="utf-8"
        )
    )
    total = (
        sum(training.values())
        + sum(value["estimated_cny"] for value in evaluations.values())
        + partial_cost
        + float(probe["estimated_cny"])
    )
    return {
        "training_chains_no_double_count": training,
        "training_estimated_cny": sum(training.values()),
        "evaluation": evaluations,
        "incomplete_M5@200_general": {
            "cached_rows": 0 if partial_summary_exists else len(partial),
            "usage": partial_usage,
            "estimated_cny": partial_cost,
            "not_an_effect_result": not partial_summary_exists,
        },
        "ppo_mask_probe_estimated_cny": float(probe["estimated_cny"]),
        "total_estimated_cny_to_current_blocker": total,
        "actual_billed_cny": None,
        "billing_note": (
            "SDK exposes tokens but no invoice. The final retry was rejected with "
            "billing_insufficient_balance; reconcile against PyTRIO Usage after credits "
            "are restored."
        ),
    }


def build_full_report(
    config: ExperimentConfig, output_path: Path, bad_cases_path: Path
) -> dict[str, Any]:
    datasets = {
        domain: read_jsonl(
            config.root / "data" / "processed" / DATASETS[dataset_name]
        )
        for domain, (dataset_name, _, _) in FULL_DOMAINS.items()
    }
    records: dict[str, Any] = {}
    complete_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label in FULL_RUNS:
        domains: dict[str, Any] = {}
        for domain, (dataset_name, _, expected_count) in FULL_DOMAINS.items():
            summary_path = _full_path(config, label, domain, "summary.json")
            predictions_path = _full_path(config, label, domain, "predictions.jsonl")
            if not summary_path.exists():
                partial = read_jsonl(predictions_path) if predictions_path.exists() else []
                domains[domain] = {
                    "status": "incomplete",
                    "cached_unique_rows": len({str(row["id"]) for row in partial}),
                    "expected_rows": expected_count,
                    "remaining_rows": expected_count - len({str(row["id"]) for row in partial}),
                    "accuracy": None,
                    "not_an_effect_result": True,
                }
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "completed":
                raise RuntimeError(f"full summary is not completed: {summary_path}")
            if summary.get("protocol_id") != config.get(
                "evaluation", "thinking_protocol_id"
            ):
                raise RuntimeError(f"stale full evaluation protocol: {summary_path}")
            dataset_path = config.root / "data" / "processed" / DATASETS[dataset_name]
            if summary.get("dataset_sha256") != sha256_file(dataset_path):
                raise RuntimeError(f"full evaluation dataset hash mismatch: {summary_path}")
            if int(summary.get("count", -1)) != expected_count:
                raise RuntimeError(f"full evaluation row count mismatch: {summary_path}")
            predictions = read_jsonl(predictions_path)
            if len({str(row["id"]) for row in predictions}) != expected_count:
                raise RuntimeError(f"full prediction cache is incomplete: {predictions_path}")
            complete_predictions[(label, domain)] = predictions
            domains[domain] = {
                key: summary[key]
                for key in (
                    "status",
                    "count",
                    "correct",
                    "accuracy",
                    "wilson_95",
                    "subject_macro_accuracy",
                    "subject_scores",
                    "format_valid_rate",
                    "thinking_closed_rate",
                    "truncation_rate",
                    "estimated_cny",
                    "wall_seconds",
                    "usage",
                )
            }
            domains[domain]["summary_path"] = str(summary_path.relative_to(config.root))
            domains[domain]["summary_sha256"] = sha256_file(summary_path)
        records[label] = domains

    base = records["M0"]
    for label, domains in records.items():
        for domain, record in domains.items():
            if record["status"] == "completed":
                record["delta_pp_vs_M0"] = (
                    0.0
                    if label == "M0"
                    else 100
                    * (float(record["accuracy"]) - float(base[domain]["accuracy"]))
                )

    comparisons: dict[str, Any] = {}
    bad_cases: dict[str, Any] = {
        "created_at": utc_now(),
        "protocol_id": config.get("evaluation", "thinking_protocol_id"),
    }
    requested_pairs = (
        ("M0", "M4@50"),
        ("M0", "M5@50"),
        ("M0", "M5@200"),
        ("M4@50", "M5@50"),
        ("M5@50", "M5@200"),
    )
    for left_label, right_label in requested_pairs:
        pair_name = f"{left_label}_vs_{right_label}"
        comparisons[pair_name] = {}
        bad_cases[pair_name] = {}
        for domain in FULL_DOMAINS:
            left = complete_predictions.get((left_label, domain))
            right = complete_predictions.get((right_label, domain))
            if left is None or right is None:
                comparisons[pair_name][domain] = {"status": "incomplete"}
                bad_cases[pair_name][domain] = {"status": "incomplete"}
                continue
            comparisons[pair_name][domain] = {
                "status": "completed",
                **exact_mcnemar(left, right),
            }
            bad_cases[pair_name][domain] = _paired_bad_cases(
                datasets[domain], left, right
            )

    m5_general_complete = records["M5@200"]["general"]["status"] == "completed"
    m5_general_decision = (
        "overtrained: MedQA -3.42pp vs M0; M5@300 stopped; C-Eval-8 result "
        f"{records['M5@200']['general']['delta_pp_vs_M0']:+.2f}pp vs M0"
        if m5_general_complete
        else (
            "overtrained: MedQA -3.42pp vs M0; M5@300 stopped; final general "
            "evaluation blocked by insufficient balance"
        )
    )
    report = {
        "status": (
            "full_evaluation_complete"
            if m5_general_complete
            else "blocked_only_M5@200_general_full_incomplete"
        ),
        "created_at": utc_now(),
        "protocol_id": config.get("evaluation", "thinking_protocol_id"),
        "student_model": config.get("models", "student"),
        "records": records,
        "paired_exact_mcnemar": comparisons,
        "decision": {
            "M4@50": (
                "negative: MedQA -0.53pp and C-Eval-8 micro -2.60pp vs M0"
            ),
            "M5@50": (
                "bounded positive direction: MedQA +0.29pp and C-Eval-8 micro -0.99pp; "
                "paired significance is reported separately"
            ),
            "M5@200": m5_general_decision,
            "overall": (
                "Current complete evidence does not establish that OPD materially beats the "
                "untouched official 4B. It is more retention-friendly than Sequence KD at "
                "50 steps, but the medical gain is small and longer OPD regresses."
            ),
        },
        "cost": _current_cost_report(config),
        "blocker": (
            None
            if m5_general_complete
            else "reports/generated/BLOCKED_INSUFFICIENT_BALANCE.json"
        ),
    }
    atomic_write_json(bad_cases_path.resolve(), bad_cases)
    report["bad_cases_path"] = str(bad_cases_path.resolve().relative_to(config.root))
    report["bad_cases_sha256"] = sha256_file(bad_cases_path.resolve())
    atomic_write_json(output_path.resolve(), report)
    return report
