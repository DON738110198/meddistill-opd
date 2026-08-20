from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokenizedDatum:
    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    weights: tuple[float, ...]
    old_logprobs: tuple[float, ...] | None = None
    advantages: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        size = len(self.input_tokens)
        if size == 0:
            raise ValueError("datum must not be empty")
        if len(self.target_tokens) != size or len(self.weights) != size:
            raise ValueError("input, target, and weights must align")
        if self.old_logprobs is not None and len(self.old_logprobs) != size:
            raise ValueError("old logprobs must align")
        if self.advantages is not None and len(self.advantages) != size:
            raise ValueError("advantages must align")
        numeric = list(self.weights)
        if self.old_logprobs is not None:
            numeric.extend(self.old_logprobs)
        if self.advantages is not None:
            numeric.extend(self.advantages)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("datum contains NaN or infinity")

    @property
    def context_tokens(self) -> int:
        return len(self.input_tokens)

    @property
    def trainable_tokens(self) -> int:
        return sum(value != 0.0 for value in self.weights)


def build_ce_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    max_length: int,
) -> TokenizedDatum:
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must not be empty")
    full = (prompt_ids + completion_ids)[:max_length]
    if len(full) < 2:
        raise ValueError("sequence is too short")
    prompt_len = min(len(prompt_ids), len(full))
    token_weights = [0.0] * prompt_len + [1.0] * (len(full) - prompt_len)
    datum = TokenizedDatum(
        input_tokens=tuple(full[:-1]),
        target_tokens=tuple(full[1:]),
        weights=tuple(token_weights[1:]),
    )
    if datum.trainable_tokens == 0:
        raise ValueError("truncation removed all completion tokens")
    return datum


def sampled_reverse_kl(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    *,
    coefficient: float,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    if clip <= 0 or not math.isfinite(clip):
        raise ValueError("clip must be finite and positive")
    if len(student_logprobs) != len(teacher_logprobs) or not student_logprobs:
        raise ValueError("student and teacher logprobs must be non-empty and aligned")
    student = np.asarray(student_logprobs, dtype=np.float64)
    teacher = np.asarray(teacher_logprobs, dtype=np.float64)
    if not np.isfinite(student).all() or not np.isfinite(teacher).all():
        raise ValueError("student or teacher logprobs contain NaN or infinity")
    reverse_kl_sample = student - teacher
    advantages = np.clip(-coefficient * reverse_kl_sample, -clip, clip)
    if not np.isfinite(advantages).all():
        raise ValueError("OPD advantages are numerically unstable")
    return reverse_kl_sample, advantages


def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    *,
    coefficient: float,
    clip: float,
    max_length: int,
) -> tuple[TokenizedDatum, np.ndarray]:
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must not be empty")
    available = max_length - len(prompt_ids)
    if available <= 0:
        raise ValueError("prompt alone reaches max_length")
    completion_ids = completion_ids[:available]
    student_logprobs = student_logprobs[:available]
    teacher_logprobs = teacher_logprobs[:available]
    reverse_kl, completion_advantages = sampled_reverse_kl(
        student_logprobs,
        teacher_logprobs,
        coefficient=coefficient,
        clip=clip,
    )
    prompt_mask_size = len(prompt_ids) - 1
    input_tokens = prompt_ids + completion_ids[:-1]
    target_tokens = [0] * prompt_mask_size + completion_ids
    old_logprobs = [0.0] * prompt_mask_size + student_logprobs
    advantages = [0.0] * prompt_mask_size + completion_advantages.tolist()
    weights = [0.0] * prompt_mask_size + [1.0] * len(completion_ids)
    datum = TokenizedDatum(
        input_tokens=tuple(input_tokens),
        target_tokens=tuple(target_tokens),
        weights=tuple(weights),
        old_logprobs=tuple(old_logprobs),
        advantages=tuple(advantages),
    )
    if datum.trainable_tokens != len(completion_ids):
        raise ValueError("effective token mask does not match the rollout completion")
    return datum, reverse_kl


def to_pytrio_ce_datum(trio: Any, datum: TokenizedDatum) -> Any:
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(list(datum.input_tokens)),
        loss_fn_inputs={
            "target_tokens": np.asarray(datum.target_tokens, dtype=np.int64),
            "weights": np.asarray(datum.weights, dtype=np.float32),
        },
    )


def to_pytrio_opd_datum(trio: Any, datum: TokenizedDatum) -> Any:
    if datum.old_logprobs is None or datum.advantages is None:
        raise ValueError("OPD datum is missing old logprobs or advantages")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(list(datum.input_tokens)),
        loss_fn_inputs={
            "target_tokens": np.asarray(datum.target_tokens, dtype=np.int64),
            "logprobs": np.asarray(datum.old_logprobs, dtype=np.float32),
            "advantages": np.asarray(datum.advantages, dtype=np.float32),
        },
    )
