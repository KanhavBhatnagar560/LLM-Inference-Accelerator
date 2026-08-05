"""Sampling backend protocol and the dependency-free Python oracle."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

from .sampling import DistributionError, normalize_probabilities, sample_categorical_from_uniform


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of checking explicit uniforms against acceptance probabilities."""

    accepted_count: int
    rejection_index: int | None


@runtime_checkable
class SamplingBackend(Protocol):
    """Deterministic numerical operations used by the decoder.

    The decoder owns randomness and passes each uniform draw explicitly.
    """

    name: str

    def categorical(self, probabilities: Sequence[float], uniform: float) -> int: ...

    def acceptance_probabilities(
        self,
        target_rows: Sequence[Sequence[float]],
        draft_rows: Sequence[Sequence[float]],
        token_ids: Sequence[int],
    ) -> tuple[float, ...]: ...

    def residual_weights(
        self,
        target_probabilities: Sequence[float],
        draft_probabilities: Sequence[float],
    ) -> tuple[float, ...]: ...

    def first_rejection(
        self,
        acceptance_probabilities: Sequence[float],
        uniforms: Sequence[float],
    ) -> VerificationResult: ...


class PythonSamplingBackend:
    """Auditable Python implementation used as the native parity oracle."""

    name = "python"

    def categorical(self, probabilities: Sequence[float], uniform: float) -> int:
        normalized = normalize_probabilities(probabilities)
        return sample_categorical_from_uniform(normalized, uniform)

    def acceptance_probabilities(
        self,
        target_rows: Sequence[Sequence[float]],
        draft_rows: Sequence[Sequence[float]],
        token_ids: Sequence[int],
    ) -> tuple[float, ...]:
        if not (len(target_rows) == len(draft_rows) == len(token_ids)):
            raise DistributionError("acceptance inputs must have equal row counts")
        if not token_ids:
            raise DistributionError("at least one proposal is required")

        probabilities: list[float] = []
        for target, draft, token in zip(target_rows, draft_rows, token_ids):
            p = normalize_probabilities(target)
            q = normalize_probabilities(draft, expected_size=len(p))
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < len(p):
                raise DistributionError("proposed token is outside the vocabulary")
            probabilities.append(1.0 if q[token] == 0.0 else min(1.0, p[token] / q[token]))
        return tuple(probabilities)

    def residual_weights(
        self,
        target_probabilities: Sequence[float],
        draft_probabilities: Sequence[float],
    ) -> tuple[float, ...]:
        target = normalize_probabilities(target_probabilities)
        draft = normalize_probabilities(draft_probabilities, expected_size=len(target))
        residual = tuple(max(0.0, p - q) for p, q in zip(target, draft))
        if math.fsum(residual) <= 1.0e-15:
            return target
        return residual

    def first_rejection(
        self,
        acceptance_probabilities: Sequence[float],
        uniforms: Sequence[float],
    ) -> VerificationResult:
        if len(acceptance_probabilities) != len(uniforms):
            raise DistributionError("acceptance probabilities and uniforms must have equal size")
        for index, (probability, uniform) in enumerate(
            zip(acceptance_probabilities, uniforms)
        ):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise DistributionError("acceptance probability must be finite and in [0, 1]")
            if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
                raise DistributionError("uniform must be finite and in [0, 1)")
            if not uniform < probability:
                return VerificationResult(index, index)
        return VerificationResult(len(acceptance_probabilities), None)


BackendMode = Literal["auto", "python", "native"]


def load_sampling_backend(
    mode: BackendMode = "auto",
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> SamplingBackend:
    """Select a sampling backend without making native code mandatory.

    ``auto`` falls back only when no native library can be found. An explicitly
    configured path, ABI mismatch, or native execution error is never hidden.
    """

    environment_mode = os.environ.get("SPECDECODE_SAMPLING_BACKEND")
    if mode == "auto" and library_path is None and environment_mode:
        if environment_mode not in ("auto", "python", "native"):
            raise ValueError(
                "SPECDECODE_SAMPLING_BACKEND must be auto, python, or native"
            )
        mode = cast(BackendMode, environment_mode)
    if mode == "python":
        return PythonSamplingBackend()
    if mode not in ("auto", "native"):
        raise ValueError(f"unknown sampling backend mode: {mode}")

    from .native import NativeLibraryNotFound, NativeSamplingBackend

    explicit_library = library_path is not None or bool(
        os.environ.get("SPECDECODE_NATIVE_LIBRARY")
    )
    try:
        return NativeSamplingBackend.load(library_path)
    except NativeLibraryNotFound:
        if mode == "native" or explicit_library:
            raise
        return PythonSamplingBackend()
