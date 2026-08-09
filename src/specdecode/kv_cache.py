"""Dependency-free paged INT8 KV-cache correctness engine."""

from __future__ import annotations

import heapq
import math
import struct
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class KVCacheError(RuntimeError):
    """Base error for paged KV-cache operations."""


class KVCacheCapacityError(KVCacheError):
    """Raised when an append cannot be satisfied by the physical block pool."""


@dataclass(frozen=True, slots=True)
class KVCacheConfig:
    """Physical cache shape and block-pool capacity."""

    num_layers: int
    num_heads: int
    head_dim: int
    block_size: int = 16
    num_blocks: int = 256

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "num_heads",
            "head_dim",
            "block_size",
            "num_blocks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def capacity_tokens(self) -> int:
        return self.block_size * self.num_blocks

    @property
    def fp32_bytes_per_token(self) -> int:
        # K and V, each with one float32 value per layer/head/dimension.
        return 2 * self.num_layers * self.num_heads * self.head_dim * 4

    @property
    def quantized_bytes_per_token(self) -> int:
        # One signed byte per value plus one float32 scale per K/V head.
        values = 2 * self.num_layers * self.num_heads * self.head_dim
        scales = 2 * self.num_layers * self.num_heads * 4
        return values + scales

    @property
    def theoretical_savings_ratio(self) -> float:
        return 1.0 - (self.quantized_bytes_per_token / self.fp32_bytes_per_token)


@dataclass(frozen=True, slots=True)
class QuantizedVector:
    """One symmetric INT8 vector and its float32-compatible scale."""

    values: tuple[int, ...]
    scale: float

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("a quantized vector cannot be empty")
        if not math.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("quantization scale must be finite and non-negative")
        try:
            float32_scale = struct.unpack("=f", struct.pack("=f", self.scale))[0]
        except OverflowError as error:
            raise ValueError("quantization scale must fit in float32") from error
        if float32_scale != self.scale:
            raise ValueError("quantization scale must be exactly representable as float32")
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("quantized values must be integers")
            if not -127 <= value <= 127:
                raise ValueError("quantized values must be inside [-127, 127]")
        if self.scale == 0.0 and any(self.values):
            raise ValueError("a zero quantization scale requires all-zero values")


@runtime_checkable
class KVQuantizer(Protocol):
    """Quantization operations used by :class:`PagedKVCache`."""

    name: str

    def quantize(self, values: Sequence[float]) -> QuantizedVector: ...

    def dequantize(self, vector: QuantizedVector) -> tuple[float, ...]: ...


def _finite_values(values: Sequence[float]) -> tuple[float, ...]:
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("KV values must be real numbers, not booleans")
        try:
            converted_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("KV values must be real numbers") from error
        if not math.isfinite(converted_value):
            raise ValueError("KV values must be finite")
        converted.append(converted_value)
    if not converted:
        raise ValueError("a KV head vector cannot be empty")
    return tuple(converted)


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


class PythonKVQuantizer:
    """Per-vector symmetric INT8 quantization reference implementation."""

    name = "python"

    def quantize(self, values: Sequence[float]) -> QuantizedVector:
        source = _finite_values(values)
        maximum = max(abs(value) for value in source)
        if maximum == 0.0:
            return QuantizedVector((0,) * len(source), 0.0)

        try:
            scale = struct.unpack("=f", struct.pack("=f", maximum / 127.0))[0]
        except OverflowError as error:
            raise ValueError("quantization scale is outside the float32 range") from error
        if not math.isfinite(scale) or scale == 0.0:
            raise ValueError("quantization scale is outside the float32 range")
        quantized = tuple(
            max(-127, min(127, _round_half_away_from_zero(value / scale)))
            for value in source
        )
        return QuantizedVector(quantized, scale)

    def dequantize(self, vector: QuantizedVector) -> tuple[float, ...]:
        if not isinstance(vector, QuantizedVector):
            raise TypeError("dequantization requires a QuantizedVector")
        return tuple(value * vector.scale for value in vector.values)


@dataclass(frozen=True, slots=True)
class QuantizedKVToken:
    """Quantized K and V heads for one token across every model layer."""

    keys: tuple[tuple[QuantizedVector, ...], ...]
    values: tuple[tuple[QuantizedVector, ...], ...]


@dataclass(frozen=True, slots=True)
class DequantizedKVToken:
    """Dequantized K and V tensors for one token."""

    keys: tuple[tuple[tuple[float, ...], ...], ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class KVCacheCheckpoint:
    """A sequence length that can be used to discard a speculative suffix."""

    sequence_id: Hashable
    token_count: int


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    """Logical utilization and theoretical storage-format accounting."""

    sequences: int
    used_tokens: int
    capacity_tokens: int
    allocated_blocks: int
    free_blocks: int
    used_quantized_bytes: int
    allocated_quantized_bytes: int
    capacity_quantized_bytes: int
    fp32_equivalent_bytes: int
    block_table_bytes: int
    theoretical_savings_ratio: float


KVHead = Sequence[float]
KVLayer = Sequence[KVHead]
KVTensor = Sequence[KVLayer]
KVEntry = tuple[KVTensor, KVTensor]


class PagedKVCache:
    """Fixed-pool paged cache with deterministic allocation and rollback.

    Physical blocks are never shared between sequences in this correctness
    implementation. That keeps reclamation auditable while still exercising
    the logical-to-physical mapping required by a later PagedAttention kernel.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        *,
        quantizer: KVQuantizer | None = None,
    ) -> None:
        self.config = config
        self.quantizer = quantizer if quantizer is not None else PythonKVQuantizer()
        self._blocks: list[list[QuantizedKVToken | None]] = [
            [None] * config.block_size for _ in range(config.num_blocks)
        ]
        self._free_blocks = list(range(config.num_blocks))
        heapq.heapify(self._free_blocks)
        self._block_tables: dict[Hashable, list[int]] = {}
        self._lengths: dict[Hashable, int] = {}

    def _require_sequence(self, sequence_id: Hashable) -> None:
        try:
            exists = sequence_id in self._lengths
        except TypeError as error:
            raise TypeError("sequence_id must be hashable") from error
        if not exists:
            raise KeyError(f"unknown sequence: {sequence_id!r}")

    def create_sequence(self, sequence_id: Hashable) -> None:
        try:
            if sequence_id in self._lengths:
                raise ValueError(f"sequence already exists: {sequence_id!r}")
            hash(sequence_id)
        except TypeError as error:
            raise TypeError("sequence_id must be hashable") from error
        self._block_tables[sequence_id] = []
        self._lengths[sequence_id] = 0

    def sequence_length(self, sequence_id: Hashable) -> int:
        self._require_sequence(sequence_id)
        return self._lengths[sequence_id]

    def block_table(self, sequence_id: Hashable) -> tuple[int, ...]:
        self._require_sequence(sequence_id)
        return tuple(self._block_tables[sequence_id])

    def physical_slot(self, sequence_id: Hashable, token_index: int) -> tuple[int, int]:
        self._require_sequence(sequence_id)
        length = self._lengths[sequence_id]
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise TypeError("token_index must be an integer")
        if not 0 <= token_index < length:
            raise IndexError("token_index is outside the cached sequence")
        logical_block, offset = divmod(token_index, self.config.block_size)
        return self._block_tables[sequence_id][logical_block], offset

    def _quantize_tensor(
        self,
        tensor: KVTensor,
        *,
        label: str,
    ) -> tuple[tuple[QuantizedVector, ...], ...]:
        if len(tensor) != self.config.num_layers:
            raise ValueError(f"{label} must contain one tensor per model layer")
        layers: list[tuple[QuantizedVector, ...]] = []
        for layer in tensor:
            if len(layer) != self.config.num_heads:
                raise ValueError(f"each {label} layer must contain num_heads vectors")
            heads: list[QuantizedVector] = []
            for head in layer:
                if len(head) != self.config.head_dim:
                    raise ValueError(f"each {label} head must contain head_dim values")
                quantized = self.quantizer.quantize(head)
                if len(quantized.values) != self.config.head_dim:
                    raise KVCacheError("quantizer returned the wrong vector size")
                heads.append(quantized)
            layers.append(tuple(heads))
        return tuple(layers)

    def _quantize_token(self, keys: KVTensor, values: KVTensor) -> QuantizedKVToken:
        return QuantizedKVToken(
            keys=self._quantize_tensor(keys, label="key"),
            values=self._quantize_tensor(values, label="value"),
        )

    def append(
        self,
        sequence_id: Hashable,
        keys: KVTensor,
        values: KVTensor,
    ) -> int:
        return self.append_many(sequence_id, ((keys, values),))[0]

    def append_many(
        self,
        sequence_id: Hashable,
        entries: Sequence[KVEntry],
    ) -> tuple[int, ...]:
        """Atomically quantize and append one or more token entries."""

        self._require_sequence(sequence_id)
        pending = tuple(entries)
        if not pending:
            return ()

        start = self._lengths[sequence_id]
        current_blocks = len(self._block_tables[sequence_id])
        final_length = start + len(pending)
        required_blocks = (final_length + self.config.block_size - 1) // self.config.block_size
        additional_blocks = required_blocks - current_blocks
        if additional_blocks > len(self._free_blocks):
            raise KVCacheCapacityError(
                f"append requires {additional_blocks} new blocks but only "
                f"{len(self._free_blocks)} are free"
            )

        # Quantize everything before mutating block tables so malformed input
        # cannot leave a partially appended speculative suffix.
        quantized = tuple(self._quantize_token(keys, values) for keys, values in pending)

        table = self._block_tables[sequence_id]
        for _ in range(additional_blocks):
            table.append(heapq.heappop(self._free_blocks))

        for token_index, token in enumerate(quantized, start=start):
            logical_block, offset = divmod(token_index, self.config.block_size)
            physical_block = table[logical_block]
            if self._blocks[physical_block][offset] is not None:
                raise AssertionError("cache slot was unexpectedly occupied")
            self._blocks[physical_block][offset] = token
        self._lengths[sequence_id] = final_length
        return tuple(range(start, final_length))

    def read_quantized_token(
        self,
        sequence_id: Hashable,
        token_index: int,
    ) -> QuantizedKVToken:
        physical_block, offset = self.physical_slot(sequence_id, token_index)
        token = self._blocks[physical_block][offset]
        if token is None:
            raise AssertionError("mapped cache slot is empty")
        return token

    def _dequantize_tensor(
        self,
        tensor: tuple[tuple[QuantizedVector, ...], ...],
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        layers: list[tuple[tuple[float, ...], ...]] = []
        for layer in tensor:
            heads: list[tuple[float, ...]] = []
            for head in layer:
                values = self.quantizer.dequantize(head)
                if len(values) != self.config.head_dim:
                    raise KVCacheError("quantizer returned the wrong dequantized size")
                heads.append(tuple(values))
            layers.append(tuple(heads))
        return tuple(layers)

    def read_token(
        self,
        sequence_id: Hashable,
        token_index: int,
    ) -> DequantizedKVToken:
        token = self.read_quantized_token(sequence_id, token_index)
        return DequantizedKVToken(
            keys=self._dequantize_tensor(token.keys),
            values=self._dequantize_tensor(token.values),
        )

    def read_sequence(self, sequence_id: Hashable) -> tuple[DequantizedKVToken, ...]:
        self._require_sequence(sequence_id)
        return tuple(
            self.read_token(sequence_id, index)
            for index in range(self._lengths[sequence_id])
        )

    def checkpoint(self, sequence_id: Hashable) -> KVCacheCheckpoint:
        self._require_sequence(sequence_id)
        return KVCacheCheckpoint(sequence_id, self._lengths[sequence_id])

    def rollback(
        self,
        sequence_id: Hashable,
        checkpoint: KVCacheCheckpoint,
    ) -> int:
        if checkpoint.sequence_id != sequence_id:
            raise ValueError("checkpoint belongs to a different sequence")
        return self.truncate(sequence_id, checkpoint.token_count)

    def truncate(self, sequence_id: Hashable, token_count: int) -> int:
        """Discard a suffix and return the number of physical blocks released."""

        self._require_sequence(sequence_id)
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise TypeError("token_count must be an integer")
        current_length = self._lengths[sequence_id]
        if not 0 <= token_count <= current_length:
            raise ValueError("token_count must be between zero and the sequence length")
        if token_count == current_length:
            return 0

        table = self._block_tables[sequence_id]
        for token_index in range(token_count, current_length):
            logical_block, offset = divmod(token_index, self.config.block_size)
            self._blocks[table[logical_block]][offset] = None

        kept_blocks = (token_count + self.config.block_size - 1) // self.config.block_size
        released = table[kept_blocks:]
        del table[kept_blocks:]
        for physical_block in released:
            if any(slot is not None for slot in self._blocks[physical_block]):
                raise AssertionError("released cache block still contains live entries")
            heapq.heappush(self._free_blocks, physical_block)
        self._lengths[sequence_id] = token_count
        return len(released)

    def remove_sequence(self, sequence_id: Hashable) -> int:
        self._require_sequence(sequence_id)
        released = self.truncate(sequence_id, 0)
        del self._block_tables[sequence_id]
        del self._lengths[sequence_id]
        return released

    def stats(self) -> KVCacheStats:
        used_tokens = sum(self._lengths.values())
        allocated_blocks = sum(len(table) for table in self._block_tables.values())
        bytes_per_token = self.config.quantized_bytes_per_token
        return KVCacheStats(
            sequences=len(self._lengths),
            used_tokens=used_tokens,
            capacity_tokens=self.config.capacity_tokens,
            allocated_blocks=allocated_blocks,
            free_blocks=len(self._free_blocks),
            used_quantized_bytes=used_tokens * bytes_per_token,
            allocated_quantized_bytes=(
                allocated_blocks * self.config.block_size * bytes_per_token
            ),
            capacity_quantized_bytes=self.config.capacity_tokens * bytes_per_token,
            fp32_equivalent_bytes=used_tokens * self.config.fp32_bytes_per_token,
            block_table_bytes=allocated_blocks * 4,
            theoretical_savings_ratio=self.config.theoretical_savings_ratio,
        )

    def validate_invariants(self) -> None:
        """Raise ``AssertionError`` when allocator or block-table state is corrupt."""

        allocated = [block for table in self._block_tables.values() for block in table]
        if len(allocated) != len(set(allocated)):
            raise AssertionError("a physical block is owned by multiple logical blocks")
        if set(allocated) & set(self._free_blocks):
            raise AssertionError("a physical block is both allocated and free")
        if set(allocated) | set(self._free_blocks) != set(range(self.config.num_blocks)):
            raise AssertionError("physical block ownership is incomplete")

        for sequence_id, length in self._lengths.items():
            table = self._block_tables[sequence_id]
            expected_blocks = (length + self.config.block_size - 1) // self.config.block_size
            if len(table) != expected_blocks:
                raise AssertionError("logical block-table length is inconsistent")
            for logical_block, physical_block in enumerate(table):
                for offset, slot in enumerate(self._blocks[physical_block]):
                    token_index = logical_block * self.config.block_size + offset
                    if (token_index < length) != (slot is not None):
                        raise AssertionError("physical block occupancy is inconsistent")

        for physical_block in self._free_blocks:
            if any(slot is not None for slot in self._blocks[physical_block]):
                raise AssertionError("a free physical block contains cached entries")
