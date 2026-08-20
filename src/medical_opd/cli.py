from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medical_opd.backend import plan_run, run_preflight
from medical_opd.ceval_sar import (
    DEFAULT_CEVAL_SAR_CONFIG,
    CevalSARRequest,
    load_ceval_sar_config,
    plan_ceval_sar,
    prepare_ceval_sar_data,
    run_ceval_sar,
)
from medical_opd.ceval_sar_analysis import build_ceval_sar_analysis
from medical_opd.config import DEFAULT_CONFIG_PATH, ExperimentConfig, load_config
from medical_opd.data import prepare_data
from medical_opd.diagnostics import run_ppo_mask_probe
from medical_opd.evaluation import (
    EvalRequest,
    build_teacher_gate,
    plan_evaluation,
    run_evaluation,
)
from medical_opd.mechanism_analysis import run_t27_mechanism_analysis
from medical_opd.medical_opd_curve_analysis import build_medical_opd_curve_analysis
from medical_opd.medical_sft import (
    DEFAULT_MEDICAL_PIPELINE_CONFIG,
    MedicalSFTRequest,
    load_medical_pipeline_config,
    plan_medical_sft,
    prepare_medical_sft_data,
    run_medical_sft,
)
from medical_opd.reference_eval import (
    ReferenceEvalRequest,
    plan_reference_evaluation,
    prepare_reference_data,
    run_reference_evaluation,
)
from medical_opd.reporting import build_full_report, build_screening_report
from medical_opd.sar_curve_analysis import build_sar_curve_analysis
from medical_opd.sar_from_med300_analysis import build_sar_from_med300_analysis
from medical_opd.staged_opd import (
    OPD_TARGET_STEPS,
    StagedOPDRequest,
    plan_staged_opd,
    run_staged_opd,
    select_medical_teacher,
)
from medical_opd.teacher_sft import (
    TeacherSFTRequest,
    plan_teacher_sft,
    run_teacher_sft,
)
from medical_opd.thinking_eval import (
    ThinkingEvalRequest,
    plan_thinking_evaluation,
    run_thinking_evaluation,
)
from medical_opd.training import TrainRequest, record_resume_migration, run_training


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _config(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable medical OPD and capability restoration")
    parser.add_argument("--config", type=_path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="non-billable auth/model/price/tokenizer gate")

    prepare = subparsers.add_parser(
        "prepare-data", help="download once, de-duplicate, and freeze data"
    )
    prepare.add_argument(
        "--shared-cache",
        type=_path,
        default=DEFAULT_CONFIG_PATH.parents[2] / ".cache" / "huggingface",
    )

    prepare_reference = subparsers.add_parser(
        "prepare-diagnostic-data",
        help="freeze the 600/300 diagnostic sets without using them for training",
    )
    prepare_reference.set_defaults(command="prepare-diagnostic-data")
    prepare_reference.add_argument(
        "--shared-cache",
        type=_path,
        default=DEFAULT_CONFIG_PATH.parents[2] / ".cache" / "huggingface",
    )

    prepare_u4 = subparsers.add_parser(
        "prepare-medical-sft-data",
        help="freeze the leakage-cleaned Medical-O1 corpus for the staged 4B pipeline",
    )
    prepare_u4.set_defaults(command="prepare-medical-sft-data")
    prepare_u4.add_argument(
        "--pipeline-config", type=_path, default=DEFAULT_MEDICAL_PIPELINE_CONFIG
    )
    prepare_u4.add_argument(
        "--shared-cache",
        type=_path,
        default=DEFAULT_CONFIG_PATH.parents[2] / ".cache" / "huggingface",
    )

    prepare_ceval_sar = subparsers.add_parser(
        "prepare-base-anchor-data",
        help="freeze leakage-safe C-Eval prompt-only Base-anchor SAR data",
    )
    prepare_ceval_sar.set_defaults(command="prepare-base-anchor-data")
    prepare_ceval_sar.add_argument(
        "--sar-config", type=_path, default=DEFAULT_CEVAL_SAR_CONFIG
    )
    prepare_ceval_sar.add_argument(
        "--shared-cache",
        type=_path,
        default=DEFAULT_CONFIG_PATH.parents[2] / ".cache" / "huggingface",
    )

    plan = subparsers.add_parser("plan", help="non-billable upper-bound cost plan")
    plan.add_argument("--method", choices=["M2", "M3", "M4", "M5"], required=True)
    plan.add_argument("--steps", type=int, required=True)

    plan_teacher = subparsers.add_parser(
        "plan-teacher-sft",
        help="plan the bounded 27B medical-SFT teacher screen",
    )
    plan_teacher.add_argument("--steps", type=int, choices=[1, 10, 25], required=True)
    plan_teacher.add_argument("--output-dir", type=_path, required=True)

    teacher_sft = subparsers.add_parser(
        "teacher-sft",
        help="run or resume the paid 27B medical-SFT teacher screen",
    )
    teacher_sft.add_argument("--steps", type=int, choices=[1, 10, 25], required=True)
    teacher_sft.add_argument("--output-dir", type=_path, required=True)
    teacher_sft.add_argument("--run-name", default="t27-medical-sft")
    teacher_sft.add_argument("--resume-state", help="latest trio:// optimizer state")
    teacher_sft.add_argument("--confirm-paid", action="store_true")

    plan_u4_sft = subparsers.add_parser(
        "plan-medical-sft",
        help="plan the leakage-safe 4B Medical SFT teacher",
    )
    plan_u4_sft.set_defaults(command="plan-medical-sft")
    plan_u4_sft.add_argument(
        "--pipeline-config", type=_path, default=DEFAULT_MEDICAL_PIPELINE_CONFIG
    )
    plan_u4_sft.add_argument("--steps", type=int, required=True)
    plan_u4_sft.add_argument("--output-dir", type=_path, required=True)

    u4_sft = subparsers.add_parser(
        "medical-sft",
        help="run or resume the paid 4B Medical SFT teacher",
    )
    u4_sft.set_defaults(command="medical-sft")
    u4_sft.add_argument("--pipeline-config", type=_path, default=DEFAULT_MEDICAL_PIPELINE_CONFIG)
    u4_sft.add_argument("--steps", type=int, required=True)
    u4_sft.add_argument("--output-dir", type=_path, required=True)
    u4_sft.add_argument("--run-name", default="medical-sft")
    u4_sft.add_argument("--resume-state", help="latest trio:// optimizer state")
    u4_sft.add_argument("--confirm-paid", action="store_true")

    u4_gate = subparsers.add_parser(
        "select-medical-teacher",
        help="freeze the paired proxy gate for a 4B Medical SFT teacher",
    )
    u4_gate.set_defaults(command="select-medical-teacher")
    u4_gate.add_argument("--pipeline-config", type=_path, default=DEFAULT_MEDICAL_PIPELINE_CONFIG)
    u4_gate.add_argument("--base-medical", type=_path, required=True)
    u4_gate.add_argument("--base-general", type=_path, required=True)
    u4_gate.add_argument("--teacher-medical", type=_path, required=True)
    u4_gate.add_argument("--teacher-general", type=_path, required=True)
    u4_gate.add_argument("--output", type=_path, required=True)

    plan_u4_opd = subparsers.add_parser(
        "plan-staged-opd",
        help="plan Medical OPD or leakage-safe Base-anchor SAR",
    )
    plan_u4_opd.set_defaults(command="plan-staged-opd")
    plan_u4_opd.add_argument(
        "--pipeline-config", type=_path, default=DEFAULT_MEDICAL_PIPELINE_CONFIG
    )
    plan_u4_opd.add_argument("--stage", choices=["medical", "sar"], required=True)
    plan_u4_opd.add_argument(
        "--steps",
        type=int,
        choices=sorted(OPD_TARGET_STEPS),
        required=True,
    )
    plan_u4_opd.add_argument("--output-dir", type=_path, required=True)
    plan_u4_opd.add_argument("--teacher-model-path")
    plan_u4_opd.add_argument("--teacher-gate", type=_path)
    plan_u4_opd.add_argument("--initial-student-state")
    plan_u4_opd.add_argument("--initial-local-state", type=_path)

    u4_opd = subparsers.add_parser(
        "staged-opd",
        help="run or resume Medical OPD or leakage-safe Base-anchor SAR",
    )
    u4_opd.set_defaults(command="staged-opd")
    u4_opd.add_argument("--pipeline-config", type=_path, default=DEFAULT_MEDICAL_PIPELINE_CONFIG)
    u4_opd.add_argument("--stage", choices=["medical", "sar"], required=True)
    u4_opd.add_argument(
        "--steps",
        type=int,
        choices=sorted(OPD_TARGET_STEPS),
        required=True,
    )
    u4_opd.add_argument("--output-dir", type=_path, required=True)
    u4_opd.add_argument("--teacher-model-path")
    u4_opd.add_argument("--teacher-gate", type=_path)
    u4_opd.add_argument("--initial-student-state")
    u4_opd.add_argument("--initial-local-state", type=_path)
    u4_opd.add_argument("--resume-state")
    u4_opd.add_argument("--run-name", default="staged-opd")
    u4_opd.add_argument("--confirm-paid", action="store_true")

    plan_ceval_sar_parser = subparsers.add_parser(
        "plan-base-anchor-sar",
        help="plan the leakage-safe C-Eval Base-anchor SAR stage",
    )
    plan_ceval_sar_parser.set_defaults(command="plan-base-anchor-sar")
    plan_ceval_sar_parser.add_argument(
        "--sar-config", type=_path, default=DEFAULT_CEVAL_SAR_CONFIG
    )
    plan_ceval_sar_parser.add_argument(
        "--steps",
        type=int,
        choices=[1, 10, 25, 50, 60, 100, 150, 200, 250, 300],
        required=True,
    )
    plan_ceval_sar_parser.add_argument("--output-dir", type=_path, required=True)
    plan_ceval_sar_parser.add_argument("--initial-student-state", required=True)
    plan_ceval_sar_parser.add_argument("--initial-local-state", type=_path, required=True)

    ceval_sar_parser = subparsers.add_parser(
        "base-anchor-sar",
        help="run or resume the paid C-Eval prompt-only Base-anchor SAR stage",
    )
    ceval_sar_parser.set_defaults(command="base-anchor-sar")
    ceval_sar_parser.add_argument(
        "--sar-config", type=_path, default=DEFAULT_CEVAL_SAR_CONFIG
    )
    ceval_sar_parser.add_argument(
        "--steps",
        type=int,
        choices=[1, 10, 25, 50, 60, 100, 150, 200, 250, 300],
        required=True,
    )
    ceval_sar_parser.add_argument("--output-dir", type=_path, required=True)
    ceval_sar_parser.add_argument("--initial-student-state", required=True)
    ceval_sar_parser.add_argument("--initial-local-state", type=_path, required=True)
    ceval_sar_parser.add_argument("--resume-state")
    ceval_sar_parser.add_argument("--run-name", default="base-anchor-sar")
    ceval_sar_parser.add_argument("--confirm-paid", action="store_true")

    plan_eval = subparsers.add_parser(
        "plan-eval", help="non-billable exact-token evaluation cost plan"
    )
    plan_eval.add_argument("--model-label", required=True)
    plan_eval.add_argument("--base-model", required=True)
    plan_eval.add_argument("--model-path")
    plan_eval.add_argument("--dataset", choices=sorted({
        "medical_proxy", "general_proxy", "medical_full", "general_full"
    }), required=True)
    plan_eval.add_argument("--output-dir", type=_path, required=True)
    plan_eval.add_argument("--rag", action="store_true")

    plan_reference = subparsers.add_parser(
        "plan-diagnostic",
        help="plan the thinking-enabled 600/300 diagnostic",
    )
    plan_reference.set_defaults(command="plan-diagnostic")
    plan_reference.add_argument(
        "--dataset", choices=["medical_600", "ceval_300"], required=True
    )
    plan_reference.add_argument("--output-dir", type=_path, required=True)
    plan_reference.add_argument(
        "--model-path", help="optional trio:// sampler_weights checkpoint"
    )
    plan_reference.add_argument("--limit", type=int, default=0)

    plan_thinking = subparsers.add_parser(
        "plan-thinking-eval", help="plan the frozen thinking-enabled experiment evaluation"
    )
    plan_thinking.add_argument("--model-label", required=True)
    plan_thinking.add_argument("--base-model", required=True)
    plan_thinking.add_argument("--model-path")
    plan_thinking.add_argument(
        "--dataset",
        choices=["medical_proxy", "general_proxy", "medical_full", "general_full"],
        required=True,
    )
    plan_thinking.add_argument("--output-dir", type=_path, required=True)
    plan_thinking.add_argument("--limit", type=int, default=0)
    plan_thinking.add_argument("--concurrency", type=int)

    train = subparsers.add_parser("train", help="run or resume a paid training method")
    train.add_argument("--method", choices=["M2", "M3", "M4", "M5"], required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--output-dir", type=_path, required=True)
    train.add_argument("--run-name")
    train.add_argument("--resume-state", help="trio:// train state with optimizer")
    train.add_argument(
        "--resume-local-state",
        type=_path,
        help="immutable prior local state.json used by an approved new-directory resume",
    )
    train.add_argument(
        "--resume-migration",
        type=_path,
        help="audited migration approval JSON for a new-directory resume",
    )
    train.add_argument("--confirm-paid", action="store_true")

    migrate = subparsers.add_parser(
        "record-resume-migration",
        help="record an audited legacy checkpoint migration without contacting PyTRIO",
    )
    migrate.add_argument("--method", choices=["M2", "M3", "M4", "M5"], required=True)
    migrate.add_argument("--source-local-state", type=_path, required=True)
    migrate.add_argument("--output", type=_path, required=True)
    migrate.add_argument("--identity-evidence", required=True)
    migrate.add_argument("--allowed-difference", action="append", required=True)
    migrate.add_argument("--confirm-audit", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run a cached paid proxy/full evaluation")
    evaluate.add_argument("--model-label", required=True)
    evaluate.add_argument("--base-model", required=True)
    evaluate.add_argument("--model-path", help="optional trio:// sampler_weights checkpoint")
    evaluate.add_argument("--dataset", choices=sorted({
        "medical_proxy", "general_proxy", "medical_full", "general_full"
    }), required=True)
    evaluate.add_argument("--output-dir", type=_path, required=True)
    evaluate.add_argument("--rag", action="store_true")
    evaluate.add_argument("--confirm-paid", action="store_true")

    reference_eval = subparsers.add_parser(
        "diagnostic-evaluate",
        help="run the thinking-enabled 600/300 diagnostic",
    )
    reference_eval.set_defaults(command="diagnostic-evaluate")
    reference_eval.add_argument(
        "--dataset", choices=["medical_600", "ceval_300"], required=True
    )
    reference_eval.add_argument("--output-dir", type=_path, required=True)
    reference_eval.add_argument(
        "--model-path", help="optional trio:// sampler_weights checkpoint"
    )
    reference_eval.add_argument("--limit", type=int, default=0)
    reference_eval.add_argument("--confirm-paid", action="store_true")

    thinking_eval = subparsers.add_parser(
        "thinking-evaluate", help="run the frozen thinking-enabled experiment evaluation"
    )
    thinking_eval.add_argument("--model-label", required=True)
    thinking_eval.add_argument("--base-model", required=True)
    thinking_eval.add_argument("--model-path")
    thinking_eval.add_argument(
        "--dataset",
        choices=["medical_proxy", "general_proxy", "medical_full", "general_full"],
        required=True,
    )
    thinking_eval.add_argument("--output-dir", type=_path, required=True)
    thinking_eval.add_argument("--limit", type=int, default=0)
    thinking_eval.add_argument("--concurrency", type=int)
    thinking_eval.add_argument("--confirm-paid", action="store_true")

    gate = subparsers.add_parser("teacher-gate", help="freeze the 27B>4B proxy gate decision")
    gate.add_argument("--base-medical", type=_path, required=True)
    gate.add_argument("--base-general", type=_path, required=True)
    gate.add_argument("--teacher-medical", type=_path, required=True)
    gate.add_argument("--teacher-general", type=_path, required=True)

    report = subparsers.add_parser(
        "build-screening-report",
        help="aggregate frozen proxy results, paired tests, and full-eval forecast",
    )
    report.add_argument("--output", type=_path, required=True)

    full_report = subparsers.add_parser(
        "build-full-report",
        help="aggregate complete full tests, paired bad cases, costs, and blockers",
    )
    full_report.add_argument("--output", type=_path, required=True)
    full_report.add_argument("--bad-cases", type=_path, required=True)

    probe = subparsers.add_parser(
        "probe-ppo-mask", help="paid no-optimizer probe of PyTRIO PPO token_count semantics"
    )
    probe.add_argument("--resume-state", required=True)
    probe.add_argument("--output", type=_path, required=True)
    probe.add_argument("--confirm-paid", action="store_true")

    mechanism = subparsers.add_parser(
        "analyze-t27-mechanism",
        help="reproduce the local paired diagnosis for the failed 27B SFT teacher gate",
    )
    mechanism.add_argument("--output", type=_path, required=True)

    ceval_sar_analysis = subparsers.add_parser(
        "analyze-base-anchor-sar",
        help="build the paired proxy and cost analysis for C-Eval Base-anchor SAR@50",
    )
    ceval_sar_analysis.add_argument("--output", type=_path, required=True)
    sar_curve = subparsers.add_parser(
        "analyze-sar-curve",
        help="analyze the fixed medical600 SAR checkpoints through step200",
    )
    sar_curve.add_argument("--output", type=_path, required=True)
    medical_opd_curve = subparsers.add_parser(
        "analyze-medical-opd-curve",
        help="analyze the fixed 600/300 Medical OPD checkpoints through step300",
    )
    medical_opd_curve.add_argument("--output", type=_path, required=True)
    sar_from_med300 = subparsers.add_parser(
        "analyze-sar-from-med300",
        help="analyze the gated Medical OPD@300 to C-Eval SAR@50 branch",
    )
    sar_from_med300.add_argument("--output", type=_path, required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    if args.command == "preflight":
        return run_preflight(config)
    if args.command == "prepare-data":
        return prepare_data(config, shared_cache=args.shared_cache.resolve())
    if args.command == "prepare-diagnostic-data":
        return prepare_reference_data(config, shared_cache=args.shared_cache.resolve())
    if args.command == "prepare-medical-sft-data":
        pipeline = load_medical_pipeline_config(config, args.pipeline_config)
        return prepare_medical_sft_data(
            config, pipeline, shared_cache=args.shared_cache.resolve()
        )
    if args.command == "prepare-base-anchor-data":
        sar_config = load_ceval_sar_config(config, args.sar_config)
        return prepare_ceval_sar_data(
            config, sar_config, shared_cache=args.shared_cache.resolve()
        )
    if args.command == "plan":
        return plan_run(config, args.method, args.steps)
    if args.command == "plan-teacher-sft":
        return plan_teacher_sft(
            config,
            TeacherSFTRequest(
                target_steps=args.steps,
                output_dir=args.output_dir,
            ),
        )
    if args.command == "teacher-sft":
        return run_teacher_sft(
            config,
            TeacherSFTRequest(
                target_steps=args.steps,
                output_dir=args.output_dir,
                confirm_paid=args.confirm_paid,
                resume_state=args.resume_state,
                run_name=args.run_name,
            ),
        )
    if args.command == "plan-medical-sft":
        pipeline = load_medical_pipeline_config(config, args.pipeline_config)
        return plan_medical_sft(
            config,
            pipeline,
            MedicalSFTRequest(
                target_steps=args.steps,
                output_dir=args.output_dir,
            ),
        )
    if args.command == "medical-sft":
        pipeline = load_medical_pipeline_config(config, args.pipeline_config)
        return run_medical_sft(
            config,
            pipeline,
            MedicalSFTRequest(
                target_steps=args.steps,
                output_dir=args.output_dir,
                confirm_paid=args.confirm_paid,
                resume_state=args.resume_state,
                run_name=args.run_name,
            ),
        )
    if args.command == "select-medical-teacher":
        pipeline = load_medical_pipeline_config(config, args.pipeline_config)
        return select_medical_teacher(
            config,
            pipeline,
            base_medical=args.base_medical,
            base_general=args.base_general,
            teacher_medical=args.teacher_medical,
            teacher_general=args.teacher_general,
            output_path=args.output,
        )
    if args.command in {"plan-staged-opd", "staged-opd"}:
        pipeline = load_medical_pipeline_config(config, args.pipeline_config)
        request = StagedOPDRequest(
            stage=args.stage,
            target_steps=args.steps,
            output_dir=args.output_dir,
            teacher_model_path=args.teacher_model_path,
            teacher_gate=args.teacher_gate,
            initial_student_state=args.initial_student_state,
            initial_local_state=args.initial_local_state,
            resume_state=getattr(args, "resume_state", None),
            confirm_paid=bool(getattr(args, "confirm_paid", False)),
            run_name=getattr(args, "run_name", "staged-opd"),
        )
        if args.command == "plan-staged-opd":
            return plan_staged_opd(config, pipeline, request)
        return run_staged_opd(config, pipeline, request)
    if args.command in {"plan-base-anchor-sar", "base-anchor-sar"}:
        sar_config = load_ceval_sar_config(config, args.sar_config)
        request = CevalSARRequest(
            target_steps=args.steps,
            output_dir=args.output_dir,
            initial_student_state=args.initial_student_state,
            initial_local_state=args.initial_local_state,
            resume_state=getattr(args, "resume_state", None),
            confirm_paid=bool(getattr(args, "confirm_paid", False)),
            run_name=getattr(args, "run_name", "base-anchor-sar"),
        )
        if args.command == "plan-base-anchor-sar":
            return plan_ceval_sar(config, sar_config, request)
        return run_ceval_sar(config, sar_config, request)
    if args.command == "plan-eval":
        return plan_evaluation(
            config,
            EvalRequest(
                model_label=args.model_label,
                base_model=args.base_model,
                model_path=args.model_path,
                dataset=args.dataset,
                output_dir=args.output_dir,
                rag=args.rag,
            ),
        )
    if args.command == "plan-diagnostic":
        return plan_reference_evaluation(
            config,
            ReferenceEvalRequest(
                dataset=args.dataset,
                output_dir=args.output_dir,
                model_path=args.model_path,
                limit=args.limit,
            ),
        )
    if args.command == "plan-thinking-eval":
        return plan_thinking_evaluation(
            config,
            ThinkingEvalRequest(
                model_label=args.model_label,
                base_model=args.base_model,
                model_path=args.model_path,
                dataset=args.dataset,
                output_dir=args.output_dir,
                limit=args.limit,
                concurrency=args.concurrency,
            ),
        )
    if args.command == "train":
        return run_training(
            config,
            TrainRequest(
                method=args.method,
                target_steps=args.steps,
                output_dir=args.output_dir,
                confirm_paid=args.confirm_paid,
                resume_state=args.resume_state,
                resume_local_state=args.resume_local_state,
                resume_migration=args.resume_migration,
                run_name=args.run_name,
            ),
        )
    if args.command == "record-resume-migration":
        return record_resume_migration(
            config,
            method=args.method,
            source_state_path=args.source_local_state,
            output_path=args.output,
            identity_evidence=args.identity_evidence,
            allowed_differences=args.allowed_difference,
            confirm_audit=args.confirm_audit,
        )
    if args.command == "evaluate":
        return run_evaluation(
            config,
            EvalRequest(
                model_label=args.model_label,
                base_model=args.base_model,
                model_path=args.model_path,
                dataset=args.dataset,
                output_dir=args.output_dir,
                rag=args.rag,
                confirm_paid=args.confirm_paid,
            ),
        )
    if args.command == "diagnostic-evaluate":
        return run_reference_evaluation(
            config,
            ReferenceEvalRequest(
                dataset=args.dataset,
                output_dir=args.output_dir,
                model_path=args.model_path,
                limit=args.limit,
                confirm_paid=args.confirm_paid,
            ),
        )
    if args.command == "thinking-evaluate":
        return run_thinking_evaluation(
            config,
            ThinkingEvalRequest(
                model_label=args.model_label,
                base_model=args.base_model,
                model_path=args.model_path,
                dataset=args.dataset,
                output_dir=args.output_dir,
                limit=args.limit,
                confirm_paid=args.confirm_paid,
                concurrency=args.concurrency,
            ),
        )
    if args.command == "teacher-gate":
        return build_teacher_gate(
            config,
            args.base_medical,
            args.base_general,
            args.teacher_medical,
            args.teacher_general,
        )
    if args.command == "build-screening-report":
        return build_screening_report(config, args.output)
    if args.command == "build-full-report":
        return build_full_report(config, args.output, args.bad_cases)
    if args.command == "probe-ppo-mask":
        return run_ppo_mask_probe(
            config,
            resume_state=args.resume_state,
            output_path=args.output,
            confirm_paid=args.confirm_paid,
        )
    if args.command == "analyze-t27-mechanism":
        return run_t27_mechanism_analysis(config, args.output)
    if args.command == "analyze-base-anchor-sar":
        return build_ceval_sar_analysis(config, output_path=args.output)
    if args.command == "analyze-sar-curve":
        return build_sar_curve_analysis(config, output_path=args.output)
    if args.command == "analyze-medical-opd-curve":
        return build_medical_opd_curve_analysis(config, output_path=args.output)
    if args.command == "analyze-sar-from-med300":
        return build_sar_from_med300_analysis(config, output_path=args.output)
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = _run(args)
    except Exception as exc:
        error = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(error, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
