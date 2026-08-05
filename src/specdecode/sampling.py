"""Numerically defensive categorical and residual sampling utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


class DistributionError(ValueError):
    """Raised when a model returns an invalid probability distribution."""


def normalize_probabilities(
    probabilities: Sequence[float], *, expected_size: int | None = None
) -> tuple[float, ...]:
    """Validate and normalize a categorical probability vector."""

    if expected_size is not None and len(probabilities) != expected_size:
        raise DistributionError(
            f"expected {expected_size} probabilities, received {len(probabilities)}"
        )
    if not probabilities:
        raise DistributionError("probability vector cannot be empty")

    normalized: list[float] = []
    for value in probabilities:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise DistributionError("probabilities must be finite and non-negative")
        normalized.append(number)

    total = math.fsum(normalized)
    if total <= 0.0:
        raise DistributionError("probability vector must have positive mass")
    return tuple(value / total for value in normalized)


def sample_categorical_from_uniform(
    probabilities: Sequence[float], uniform: float
) -> int:
    """Sample from normalized probabilities using an explicit uniform draw."""

    if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise DistributionError("uniform must be finite and in [0, 1)")
    if not probabilities:
        raise DistributionError("probability vector cannot be empty")

    threshold = uniform
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold < cumulative:
            return index
    # Floating-point addition can leave cumulative infinitesimally below one.
    return len(probabilities) - 1


def sample_categorical(probabilities: Sequence[float], rng: random.Random) -> int:
    """Sample an index from an already-normalized probability vector."""

    return sample_categorical_from_uniform(probabilities, rng.random())


def residual_distribution(
    target_probabilities: Sequence[float], draft_probabilities: Sequence[float]
) -> tuple[float, ...]:
    """Return normalized `max(0, target - draft)` rejection probabilities.

    If round-off removes all residual mass, falling back to the target is safe
    and avoids an undefined distribution. In exact arithmetic that branch can
    only be reached on a zero-probability rejection event.
    """

    if len(target_probabilities) != len(draft_probabilities):
        raise DistributionError("target and draft vocabularies must have equal size")
    residual = tuple(max(0.0, p - q) for p, q in zip(target_probabilities, draft_probabilities))
    if math.fsum(residual) <= 1e-15:
        return tuple(target_probabilities)
    return normalize_probabilities(residual)
