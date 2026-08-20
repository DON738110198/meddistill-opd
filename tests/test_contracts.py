from __future__ import annotations

import numpy as np
import pytest

from medical_opd.contracts import (
    TokenizedDatum,
    build_ce_datum,
    build_opd_datum,
    sampled_reverse_kl,
    to_pytrio_ce_datum,
    to_pytrio_opd_datum,
)


class _FakeModelInput:
    @staticmethod
    def from_ints(values: list[int]) -> tuple[int, ...]:
        return tuple(values)


class _FakeDatum:
    def __init__(self, *, model_input: tuple[int, ...], loss_fn_inputs: dict[str, np.ndarray]):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class _FakeTrio:
    ModelInput = _FakeModelInput
    Datum = _FakeDatum


def test_ce_uses_next_token_shift_and_masks_every_prompt_target() -> None:
    datum = build_ce_datum([10, 11, 12], [20, 21], max_length=8)

    assert datum.input_tokens == (10, 11, 12, 20)
    assert datum.target_tokens == (11, 12, 20, 21)
    assert datum.weights == (0.0, 0.0, 1.0, 1.0)
    assert datum.context_tokens == 4
    assert datum.trainable_tokens == 2


def test_ce_truncation_keeps_only_trainable_completion_targets() -> None:
    datum = build_ce_datum([10, 11, 12], [20, 21], max_length=4)

    assert datum.input_tokens == (10, 11, 12)
    assert datum.target_tokens == (11, 12, 20)
    assert datum.weights == (0.0, 0.0, 1.0)
    assert datum.trainable_tokens == 1


@pytest.mark.parametrize(
    ("prompt", "completion", "max_length"),
    [([], [2], 8), ([1], [], 8), ([1, 2], [3], 2)],
)
def test_ce_rejects_empty_or_fully_truncated_completion(
    prompt: list[int], completion: list[int], max_length: int
) -> None:
    with pytest.raises(ValueError):
        build_ce_datum(prompt, completion, max_length=max_length)


def test_opd_targets_are_aligned_to_student_rollout_tokens() -> None:
    datum, reverse_kl = build_opd_datum(
        [10, 11, 12],
        [20, 21],
        [-0.4, -0.7],
        [-0.5, -0.6],
        coefficient=2.0,
        clip=20.0,
        max_length=8,
    )

    assert datum.input_tokens == (10, 11, 12, 20)
    assert datum.target_tokens == (0, 0, 20, 21)
    assert datum.weights == (0.0, 0.0, 1.0, 1.0)
    assert datum.old_logprobs == pytest.approx((0.0, 0.0, -0.4, -0.7))
    assert datum.advantages == pytest.approx((0.0, 0.0, -0.2, 0.2))
    np.testing.assert_allclose(reverse_kl, [0.1, -0.1])


@pytest.mark.parametrize(
    ("student", "teacher"),
    [([-0.1], [-0.1, -0.2]), ([], []), ([-0.1, -0.2], [-0.1])],
)
def test_opd_rejects_token_logprob_length_mismatch(
    student: list[float], teacher: list[float]
) -> None:
    with pytest.raises(ValueError, match="aligned"):
        build_opd_datum(
            [10, 11],
            [20, 21],
            student,
            teacher,
            coefficient=1.0,
            clip=20.0,
            max_length=8,
        )


def test_opd_truncates_tokens_and_both_logprob_streams_together() -> None:
    datum, reverse_kl = build_opd_datum(
        [10, 11, 12],
        [20, 21, 22],
        [-0.4, -0.7, -0.8],
        [-0.5, -0.6, -1.0],
        coefficient=1.0,
        clip=20.0,
        max_length=5,
    )

    assert datum.target_tokens == (0, 0, 20, 21)
    assert datum.old_logprobs == pytest.approx((0.0, 0.0, -0.4, -0.7))
    assert datum.trainable_tokens == 2
    np.testing.assert_allclose(reverse_kl, [0.1, -0.1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_token_contract_rejects_non_finite_training_values(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        TokenizedDatum((1,), (2,), (bad,))
    with pytest.raises(ValueError, match="NaN or infinity"):
        TokenizedDatum((1,), (2,), (1.0,), old_logprobs=(bad,))
    with pytest.raises(ValueError, match="NaN or infinity"):
        TokenizedDatum((1,), (2,), (1.0,), advantages=(bad,))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_reverse_kl_rejects_non_finite_logprobs(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        sampled_reverse_kl([-0.1, bad], [-0.2, -0.3], coefficient=1.0, clip=20.0)
    with pytest.raises(ValueError, match="NaN or infinity"):
        sampled_reverse_kl([-0.1, -0.2], [-0.3, bad], coefficient=1.0, clip=20.0)


def test_reverse_kl_advantages_are_clipped_symmetrically() -> None:
    reverse_kl, advantages = sampled_reverse_kl(
        [100.0, -100.0, -0.25],
        [0.0, 0.0, -0.5],
        coefficient=2.0,
        clip=3.0,
    )

    np.testing.assert_allclose(reverse_kl, [100.0, -100.0, 0.25])
    np.testing.assert_allclose(advantages, [-3.0, 3.0, -0.5])


@pytest.mark.parametrize(
    ("coefficient", "clip"),
    [(-1.0, 20.0), (1.0, 0.0), (1.0, -1.0), (float("nan"), 20.0), (1.0, float("nan"))],
)
def test_reverse_kl_rejects_invalid_scale_or_clip(coefficient: float, clip: float) -> None:
    with pytest.raises(ValueError):
        sampled_reverse_kl([-0.1], [-0.2], coefficient=coefficient, clip=clip)


def test_pytrio_ce_conversion_preserves_shapes_and_dtypes() -> None:
    converted = to_pytrio_ce_datum(
        _FakeTrio,
        build_ce_datum([10, 11], [20, 21], max_length=8),
    )

    assert converted.model_input == (10, 11, 20)
    assert converted.loss_fn_inputs["target_tokens"].shape == (3,)
    assert converted.loss_fn_inputs["target_tokens"].dtype == np.int64
    assert converted.loss_fn_inputs["weights"].shape == (3,)
    assert converted.loss_fn_inputs["weights"].dtype == np.float32


def test_pytrio_opd_conversion_preserves_shapes_and_dtypes() -> None:
    datum, _ = build_opd_datum(
        [10, 11],
        [20, 21],
        [-0.2, -0.3],
        [-0.3, -0.2],
        coefficient=1.0,
        clip=20.0,
        max_length=8,
    )
    converted = to_pytrio_opd_datum(_FakeTrio, datum)

    assert converted.model_input == (10, 11, 20)
    assert set(converted.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    assert all(value.shape == (3,) for value in converted.loss_fn_inputs.values())
    assert converted.loss_fn_inputs["target_tokens"].dtype == np.int64
    assert converted.loss_fn_inputs["logprobs"].dtype == np.float32
    assert converted.loss_fn_inputs["advantages"].dtype == np.float32


def test_pytrio_opd_conversion_rejects_ce_datum() -> None:
    with pytest.raises(ValueError, match="missing old logprobs or advantages"):
        to_pytrio_opd_datum(_FakeTrio, build_ce_datum([10], [20], max_length=8))
