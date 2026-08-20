from __future__ import annotations

import importlib.metadata
import json
import math
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, sha256_file, utc_now

PYTRIO_VERSION = "0.2.8"
PRICES_URL = "https://pytrio.cn/api/model-prices"
MODELS_URL = "https://pytrio.cn/api/models?limit=100&offset=0"


@dataclass
class UsageLedger:
    student_prefill_tokens: int = 0
    student_sample_tokens: int = 0
    student_train_tokens: int = 0
    teacher_prefill_tokens: int = 0
    teacher_sample_tokens: int = 0
    teacher_train_tokens: int = 0
    optimizer_steps: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "student_prefill_tokens",
            "student_sample_tokens",
            "student_train_tokens",
            "teacher_prefill_tokens",
            "teacher_sample_tokens",
            "teacher_train_tokens",
            "optimizer_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.wall_seconds < 0 or not math.isfinite(self.wall_seconds):
            raise ValueError("wall_seconds must be finite and non-negative")

    def add(self, other: UsageLedger) -> None:
        self.student_prefill_tokens += other.student_prefill_tokens
        self.student_sample_tokens += other.student_sample_tokens
        self.student_train_tokens += other.student_train_tokens
        self.teacher_prefill_tokens += other.teacher_prefill_tokens
        self.teacher_sample_tokens += other.teacher_sample_tokens
        self.teacher_train_tokens += other.teacher_train_tokens
        self.optimizer_steps += other.optimizer_steps
        self.wall_seconds += other.wall_seconds

    @property
    def student_tokens(self) -> int:
        return (
            self.student_prefill_tokens
            + self.student_sample_tokens
            + self.student_train_tokens
        )

    def to_dict(self) -> dict[str, int | float]:
        value = asdict(self)
        value["student_tokens"] = self.student_tokens
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UsageLedger:
        names = cls.__dataclass_fields__.keys()
        return cls(**{name: value.get(name, 0) for name in names})


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "medical-opd-audit/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return payload


def price_table(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for item in raw.get("items", []):
        name = str(item.get("display_name", ""))
        prices = item.get("prices", {})
        if not name or not isinstance(prices, dict):
            continue
        # The endpoint stores fen per million tokens: 149 means CNY 1.49.
        table[name] = {
            mode: float(prices[mode]["unit_price"]) / 100.0
            for mode in ("prefill", "sample", "train")
            if mode in prices
        }
    return table


def estimate_cost(
    usage: UsageLedger,
    prices: dict[str, dict[str, float]],
    config: ExperimentConfig,
) -> float:
    student = prices[str(config.get("models", "student"))]
    teacher = prices[str(config.get("models", "teacher"))]
    cny = (
        usage.student_prefill_tokens * student["prefill"]
        + usage.student_sample_tokens * student["sample"]
        + usage.student_train_tokens * student["train"]
        + usage.teacher_prefill_tokens * teacher["prefill"]
        + usage.teacher_sample_tokens * teacher["sample"]
        + usage.teacher_train_tokens * teacher["train"]
    ) / 1_000_000
    return round(cny, 6)


def _capability_payload(capabilities: Any) -> dict[str, Any]:
    if hasattr(capabilities, "model_dump"):
        return dict(capabilities.model_dump(mode="json"))
    if hasattr(capabilities, "dict"):
        return dict(capabilities.dict())
    raise TypeError("unsupported PyTRIO capabilities response")


def require_ready_preflight(config: ExperimentConfig) -> dict[str, Any]:
    path = config.root / "reports" / "generated" / "preflight_latest.json"
    if not path.exists():
        raise RuntimeError("current environment preflight is missing; run preflight first")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_models = {
        "student": str(config.get("models", "student")),
        "teacher": str(config.get("models", "teacher")),
    }
    if (
        report.get("status") != "ready"
        or report.get("configured_models") != expected_models
        or report.get("config_sha256") != sha256_file(config.path)
    ):
        raise RuntimeError(
            "preflight is blocked or stale for the configured models; no paid work may start"
        )
    return report


def run_preflight(config: ExperimentConfig) -> dict[str, Any]:
    import pytrio as trio

    installed = importlib.metadata.version("pytrio")
    if installed != PYTRIO_VERSION:
        raise RuntimeError(f"expected pytrio {PYTRIO_VERSION}, found {installed}")
    price_raw = fetch_json(PRICES_URL)
    public_models = fetch_json(MODELS_URL)
    prices = price_table(price_raw)
    student = str(config.get("models", "student"))
    teacher = str(config.get("models", "teacher"))
    if student not in prices or teacher not in prices:
        raise RuntimeError("current official price table does not cover the configured models")

    client = trio.ServiceClient()
    workspace_capabilities = _capability_payload(client.get_server_capabilities())
    models = {
        str(item["model_name"]): item
        for item in workspace_capabilities.get("supported_models", [])
    }
    missing = [name for name in (student, teacher) if name not in models]
    if missing:
        report = {
            "status": "blocked",
            "checked_at": utc_now(),
            "config_sha256": sha256_file(config.path),
            "authentication": {
                "status": "logged_in",
                "credential_source": "saved ~/.pytrio profile",
            },
            "configured_models": {"student": student, "teacher": teacher},
            "blocking_items": [
                {
                    "kind": "workspace_model_unavailable",
                    "model": name,
                    "fix": (
                        "Ask PyTRIO to enable this exact model for sampling and training, then "
                        "rerun preflight. Do not substitute a differently named checkpoint."
                    ),
                }
                for name in missing
            ],
            "workspace_capabilities": workspace_capabilities,
            "public_model_catalog": public_models,
            "official_price_snapshot": price_raw,
            "note": "A model appearing in the price table does not prove workspace availability.",
        }
        generated = config.root / "reports" / "generated"
        atomic_write_json(generated / "price_snapshot_latest.json", price_raw)
        atomic_write_json(generated / "preflight_latest.json", report)
        return report
    capability_blocks: list[dict[str, Any]] = []
    for model_name, mode in (
        (student, "training"),
        (student, "sampling"),
        (teacher, "training"),
        (teacher, "sampling"),
    ):
        if not bool(models[model_name][mode]["available"]):
            capability_blocks.append(
                {
                    "kind": "workspace_capability_unavailable",
                    "model": model_name,
                    "mode": mode,
                    "fix": (
                        "Ask PyTRIO to restore this exact workspace capability, then rerun "
                        "preflight. Do not start a paid stage that cannot be evaluated."
                    ),
                }
            )
    if capability_blocks:
        report = {
            "status": "blocked",
            "checked_at": utc_now(),
            "config_sha256": sha256_file(config.path),
            "authentication": {
                "status": "logged_in",
                "credential_source": "saved ~/.pytrio profile",
            },
            "configured_models": {"student": student, "teacher": teacher},
            "blocking_items": capability_blocks,
            "workspace_capabilities": workspace_capabilities,
            "public_model_catalog": public_models,
            "official_price_snapshot": price_raw,
            "note": (
                "Availability is checked per workspace and per mode. Public catalog presence "
                "alone is insufficient."
            ),
        }
        generated = config.root / "reports" / "generated"
        atomic_write_json(generated / "price_snapshot_latest.json", price_raw)
        atomic_write_json(generated / "preflight_latest.json", report)
        return report

    processed = config.root / "data" / "processed"
    tokenizer_path = processed / "tokenizer_compatibility.json"
    data_manifest_path = processed / "manifest.json"
    manifest = (
        json.loads(data_manifest_path.read_text(encoding="utf-8"))
        if data_manifest_path.exists()
        else None
    )
    from medical_opd.data import _tokenizer_audit

    tokenizer_cache = config.root.parent / ".cache" / "huggingface" / "hub"
    _, tokenizer = _tokenizer_audit(config, tokenizer_path, tokenizer_cache)

    report = {
        "status": "ready" if manifest is not None else "environment_ready_data_pending",
        "checked_at": utc_now(),
        "config_sha256": sha256_file(config.path),
        "python_dependency": {"pytrio_required": PYTRIO_VERSION, "pytrio_installed": installed},
        "authentication": {"status": "logged_in", "credential_source": "saved ~/.pytrio profile"},
        "configured_models": {"student": student, "teacher": teacher},
        "workspace_capabilities": workspace_capabilities,
        "public_model_catalog": public_models,
        "official_price_snapshot": price_raw,
        "price_table_cny_per_million_tokens": prices,
        "tokenizer_gate": tokenizer,
        "data_manifest_status": None if manifest is None else manifest.get("status"),
        "billing_observability": {
            "sdk_money_field": False,
            "local_counters": True,
            "estimate_formula": "sum(tokens * official CNY-per-million price) / 1e6",
            "rounding_scope": "official endpoint says ceil; scope is undocumented",
            "actual_bill_source": "https://pytrio.cn/usage (manual; SDK has no balance/cost API)",
        },
    }
    generated = config.root / "reports" / "generated"
    atomic_write_json(generated / "price_snapshot_latest.json", price_raw)
    atomic_write_json(generated / "preflight_latest.json", report)
    return report


def plan_run(config: ExperimentConfig, method: str, target_steps: int) -> dict[str, Any]:
    if method not in {"M2", "M3", "M4", "M5"}:
        raise ValueError("method must be M2, M3, M4, or M5")
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    lengths_path = config.root / "data" / "processed" / "lengths.json"
    if not lengths_path.exists():
        raise RuntimeError("data length audit is missing; run prepare-data first")
    lengths = json.loads(lengths_path.read_text(encoding="utf-8"))
    prompt = int(lengths["prompt_tokens"]["p90"])
    completion = int(lengths["derived_max_completion_tokens"])
    sequence = int(lengths["derived_max_sequence_tokens"])
    batch = int(config.get("training", "batch_size"))
    usage = UsageLedger(optimizer_steps=target_steps)
    if method in {"M2", "M3"}:
        usage.student_train_tokens = target_steps * batch * sequence
    elif method == "M4":
        usage.student_train_tokens = target_steps * batch * sequence
        usage.teacher_prefill_tokens = target_steps * batch * prompt
        usage.teacher_sample_tokens = target_steps * batch * completion
    else:
        usage.student_prefill_tokens = target_steps * batch * prompt
        usage.student_sample_tokens = target_steps * batch * completion
        usage.teacher_prefill_tokens = target_steps * batch * (prompt + completion)
        usage.teacher_sample_tokens = target_steps * batch
        usage.student_train_tokens = target_steps * batch * (prompt + completion - 1)
    raw_prices = fetch_json(PRICES_URL)
    prices = price_table(raw_prices)
    return {
        "mode": "remote_paid_training",
        "method": method,
        "student_model": config.get("models", "student"),
        "teacher_model": config.get("models", "teacher") if method in {"M4", "M5"} else None,
        "target_optimizer_steps": target_steps,
        "batch_size": batch,
        "group_size": int(config.get("training", "group_size")) if method == "M5" else 1,
        "max_sequence_tokens": sequence,
        "max_completion_tokens": completion,
        "p90_prompt_tokens": prompt,
        "upper_bound_usage": usage.to_dict(),
        "upper_bound_estimated_cny": estimate_cost(usage, prices, config),
        "price_version": raw_prices.get("version"),
        "estimate_warning": (
            "Upper bound, not an invoice. Reconcile observed counters with PyTRIO Usage UI."
        ),
        "success_criterion": (
            "target step completed; finite loss; local ledger and remote optimizer-state "
            "checkpoint saved"
        ),
    }
