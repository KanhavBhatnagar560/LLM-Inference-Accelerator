"""Lazy ctypes bindings for the optional C++ sampling library."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .backends import VerificationResult
from .kv_cache import QuantizedVector

ABI_VERSION = 0x00010001


class NativeLibraryNotFound(OSError):
    """Raised when native execution is required but no library can be loaded."""


class NativeBackendError(RuntimeError):
    """Raised when the native ABI rejects an operation or is incompatible."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _platform_library_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libspecdecode_native.dylib", "libspecdecode_native.1.dylib")
    if os.name == "nt":
        return ("specdecode_native.dll", "libspecdecode_native.dll")
    return ("libspecdecode_native.so", "libspecdecode_native.so.1")


def _development_candidates() -> tuple[Path, ...]:
    repository = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    packaged = tuple(
        candidate
        for candidate in package.glob("libspecdecode_native*")
        if candidate.suffix.lower() in (".dll", ".dylib", ".pyd", ".so")
    )
    directories = (
        repository / "work" / "native-build" / "native",
        repository / "build" / "native",
    )
    return packaged + tuple(
        directory / name
        for directory in directories
        for name in _platform_library_names()
    )


def _resolve_library_path(path: str | os.PathLike[str] | None) -> str:
    explicit = path or os.environ.get("SPECDECODE_NATIVE_LIBRARY")
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise NativeLibraryNotFound(f"native library does not exist: {candidate}")
        return str(candidate)

    for candidate in _development_candidates():
        if candidate.is_file():
            return str(candidate)
    discovered = ctypes.util.find_library("specdecode_native")
    if discovered:
        return discovered
    raise NativeLibraryNotFound(
        "specdecode native library was not found; build it with CMake or set "
        "SPECDECODE_NATIVE_LIBRARY"
    )


class _NativeLibrary:
    def __init__(self, path: str) -> None:
        try:
            self.handle = ctypes.CDLL(path)
        except OSError as error:
            raise NativeLibraryNotFound(
                f"could not load native library {path}: {error}"
            ) from error
        self.path = path
        self._configure()
        version = int(self.handle.sd_abi_version())
        if version != ABI_VERSION:
            raise NativeBackendError(
                f"native ABI mismatch: expected 0x{ABI_VERSION:08x}, received 0x{version:08x}"
            )

    def _configure(self) -> None:
        double_pointer = ctypes.POINTER(ctypes.c_double)
        int8_pointer = ctypes.POINTER(ctypes.c_int8)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        size_pointer = ctypes.POINTER(ctypes.c_size_t)

        self.handle.sd_abi_version.argtypes = []
        self.handle.sd_abi_version.restype = ctypes.c_uint32
        self.handle.sd_status_string.argtypes = [ctypes.c_int]
        self.handle.sd_status_string.restype = ctypes.c_char_p
        self.handle.sd_normalize_f64.argtypes = [
            double_pointer,
            ctypes.c_size_t,
            double_pointer,
        ]
        self.handle.sd_normalize_f64.restype = ctypes.c_int
        self.handle.sd_sample_categorical_f64.argtypes = [
            double_pointer,
            ctypes.c_size_t,
            ctypes.c_double,
            uint64_pointer,
        ]
        self.handle.sd_sample_categorical_f64.restype = ctypes.c_int
        self.handle.sd_residual_weights_f64.argtypes = [
            double_pointer,
            double_pointer,
            ctypes.c_size_t,
            double_pointer,
        ]
        self.handle.sd_residual_weights_f64.restype = ctypes.c_int
        self.handle.sd_acceptance_probabilities_f64.argtypes = [
            double_pointer,
            double_pointer,
            uint64_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_pointer,
        ]
        self.handle.sd_acceptance_probabilities_f64.restype = ctypes.c_int
        self.handle.sd_first_rejection_f64.argtypes = [
            double_pointer,
            double_pointer,
            ctypes.c_size_t,
            size_pointer,
            size_pointer,
        ]
        self.handle.sd_first_rejection_f64.restype = ctypes.c_int
        self.handle.sd_quantize_symmetric_int8_f64.argtypes = [
            double_pointer,
            ctypes.c_size_t,
            int8_pointer,
            double_pointer,
        ]
        self.handle.sd_quantize_symmetric_int8_f64.restype = ctypes.c_int
        self.handle.sd_dequantize_symmetric_int8_f64.argtypes = [
            int8_pointer,
            ctypes.c_size_t,
            ctypes.c_double,
            double_pointer,
        ]
        self.handle.sd_dequantize_symmetric_int8_f64.restype = ctypes.c_int

    def check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        raw_message = self.handle.sd_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else f"status {status}"
        raise NativeBackendError(f"native {operation} failed: {message}", status=status)


def _double_array(values: Sequence[float]) -> ctypes.Array:
    return (ctypes.c_double * len(values))(*(float(value) for value in values))


def _uint64_array(values: Sequence[int]) -> ctypes.Array:
    return (ctypes.c_uint64 * len(values))(*(int(value) for value in values))


def _int8_array(values: Sequence[int]) -> ctypes.Array:
    return (ctypes.c_int8 * len(values))(*(int(value) for value in values))


class NativeKVQuantizer:
    """Per-vector symmetric INT8 quantization using the native C ABI."""

    name = "native"

    def __init__(self, library: _NativeLibrary) -> None:
        self._library = library

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "NativeKVQuantizer":
        return cls(_NativeLibrary(_resolve_library_path(path)))

    @property
    def library_path(self) -> str:
        return self._library.path

    def quantize(self, values: Sequence[float]) -> QuantizedVector:
        source = tuple(values)
        if any(isinstance(value, bool) for value in source):
            raise TypeError("KV values must be real numbers, not booleans")
        inputs = _double_array(source)
        output = (ctypes.c_int8 * len(source))()
        scale = ctypes.c_double()
        status = self._library.handle.sd_quantize_symmetric_int8_f64(
            inputs,
            len(source),
            output,
            ctypes.byref(scale),
        )
        self._library.check(status, "INT8 quantization")
        return QuantizedVector(tuple(int(value) for value in output), scale.value)

    def dequantize(self, vector: QuantizedVector) -> tuple[float, ...]:
        if not isinstance(vector, QuantizedVector):
            raise TypeError("dequantization requires a QuantizedVector")
        inputs = _int8_array(vector.values)
        output = (ctypes.c_double * len(vector.values))()
        status = self._library.handle.sd_dequantize_symmetric_int8_f64(
            inputs,
            len(vector.values),
            vector.scale,
            output,
        )
        self._library.check(status, "INT8 dequantization")
        return tuple(output)


class NativeSamplingBackend:
    """Sampling backend backed by the versioned C++ C ABI."""

    name = "native"

    def __init__(self, library: _NativeLibrary) -> None:
        self._library = library

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] | None = None
    ) -> "NativeSamplingBackend":
        return cls(_NativeLibrary(_resolve_library_path(path)))

    @property
    def library_path(self) -> str:
        return self._library.path

    def normalize(self, weights: Sequence[float]) -> tuple[float, ...]:
        inputs = _double_array(weights)
        output = (ctypes.c_double * len(weights))()
        status = self._library.handle.sd_normalize_f64(inputs, len(weights), output)
        self._library.check(status, "normalization")
        return tuple(output)

    def categorical(self, probabilities: Sequence[float], uniform: float) -> int:
        inputs = _double_array(probabilities)
        output = ctypes.c_uint64()
        status = self._library.handle.sd_sample_categorical_f64(
            inputs,
            len(probabilities),
            uniform,
            ctypes.byref(output),
        )
        self._library.check(status, "categorical sampling")
        return int(output.value)

    def residual_weights(
        self,
        target_probabilities: Sequence[float],
        draft_probabilities: Sequence[float],
    ) -> tuple[float, ...]:
        if len(target_probabilities) != len(draft_probabilities):
            raise NativeBackendError("native residual inputs must have equal sizes")
        target = _double_array(target_probabilities)
        draft = _double_array(draft_probabilities)
        output = (ctypes.c_double * len(target_probabilities))()
        status = self._library.handle.sd_residual_weights_f64(
            target,
            draft,
            len(target_probabilities),
            output,
        )
        self._library.check(status, "residual construction")
        return tuple(output)

    def acceptance_probabilities(
        self,
        target_rows: Sequence[Sequence[float]],
        draft_rows: Sequence[Sequence[float]],
        token_ids: Sequence[int],
    ) -> tuple[float, ...]:
        if not (len(target_rows) == len(draft_rows) == len(token_ids)):
            raise NativeBackendError("native acceptance inputs must have equal row counts")
        if not target_rows:
            # Let native validation own malformed non-empty inputs while
            # avoiding an impossible vocabulary lookup here.
            raise NativeBackendError("native acceptance requires at least one proposal")
        vocabulary_size = len(target_rows[0])
        if vocabulary_size == 0:
            raise NativeBackendError("native acceptance vocabulary cannot be empty")
        if any(len(row) != vocabulary_size for row in target_rows):
            raise NativeBackendError("native target rows must have equal vocabulary sizes")
        if any(len(row) != vocabulary_size for row in draft_rows):
            raise NativeBackendError("native draft rows must have equal vocabulary sizes")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise NativeBackendError("native proposed token IDs must be non-negative integers")
        target_flat = _double_array(tuple(value for row in target_rows for value in row))
        draft_flat = _double_array(tuple(value for row in draft_rows for value in row))
        tokens = _uint64_array(token_ids)
        output = (ctypes.c_double * len(token_ids))()
        status = self._library.handle.sd_acceptance_probabilities_f64(
            target_flat,
            draft_flat,
            tokens,
            len(token_ids),
            vocabulary_size,
            output,
        )
        self._library.check(status, "acceptance verification")
        return tuple(output)

    def first_rejection(
        self,
        acceptance_probabilities: Sequence[float],
        uniforms: Sequence[float],
    ) -> VerificationResult:
        if len(acceptance_probabilities) != len(uniforms):
            raise NativeBackendError(
                "native acceptance probabilities and uniforms must have equal sizes"
            )
        acceptance = _double_array(acceptance_probabilities)
        uniform_values = _double_array(uniforms)
        accepted_count = ctypes.c_size_t()
        rejection_index = ctypes.c_size_t()
        status = self._library.handle.sd_first_rejection_f64(
            acceptance,
            uniform_values,
            len(acceptance_probabilities),
            ctypes.byref(accepted_count),
            ctypes.byref(rejection_index),
        )
        self._library.check(status, "first-rejection verification")
        rejected = int(rejection_index.value)
        if rejected == len(acceptance_probabilities):
            return VerificationResult(int(accepted_count.value), None)
        return VerificationResult(int(accepted_count.value), rejected)
