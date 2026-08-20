from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from medical_opd.backend import PRICES_URL, UsageLedger, estimate_cost, fetch_json, price_table
from medical_opd.config import ExperimentConfig
from medical_opd.io_utils import atomic_write_json, utc_now


async def _probe_async(
    config: ExperimentConfig, resume_state: str, output_path: Path
) -> dict[str, Any]:
    import pytrio as trio

    trio.configure(timeout=600)
    service = trio.ServiceClient()
    training = await service.create_training_client_from_state_with_optimizer_async(
        resume_state
    )
    tokenizer = training.get_tokenizer()
    tokens = list(tokenizer.encode("PPO explicit mask diagnostic", add_special_tokens=False))
    if len(tokens) < 4:
        raise RuntimeError("mask diagnostic tokenizer produced too few tokens")
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    advantages = np.zeros(len(input_tokens), dtype=np.float32)
    advantages[1] = 1.0
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.zeros(len(input_tokens), dtype=np.float32),
            "advantages": advantages,
        },
    )
    future = await training.forward_async([datum], loss_fn="ppo")
    result = await future
    metrics = {str(key): float(value) for key, value in result.metrics.items()}
    usage = UsageLedger(student_train_tokens=len(input_tokens))
    prices = fetch_json(PRICES_URL)
    report = {
        "status": "passed" if metrics.get("token_count") == 1.0 else "failed",
        "created_at": utc_now(),
        "probe": "PPO token_count with one nonzero and remaining zero advantages",
        "resume_state": resume_state,
        "submitted_tokens": len(input_tokens),
        "nonzero_advantage_tokens": 1,
        "metrics": metrics,
        "usage": usage.to_dict(),
        "estimated_cny": estimate_cost(usage, price_table(prices), config),
        "actual_billed_cny": None,
        "optimizer_step_performed": False,
        "conclusion": (
            "PyTRIO token_count counts nonzero-advantage PPO tokens; the local completion "
            "mask remains the student-token accounting source."
        ),
    }
    atomic_write_json(output_path, report)
    if report["status"] != "passed":
        raise RuntimeError("PPO mask diagnostic did not match the preregistered contract")
    return report


def run_ppo_mask_probe(
    config: ExperimentConfig,
    *,
    resume_state: str,
    output_path: Path,
    confirm_paid: bool,
) -> dict[str, Any]:
    if not confirm_paid:
        raise ValueError("refusing paid remote work without --confirm-paid")
    return asyncio.run(_probe_async(config, resume_state, output_path.resolve()))
