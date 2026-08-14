"""Device-resident packed INT8 KV-cache storage for future CUDA attention."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from .cuda import CudaExecutionRuntime, CudaTask
from .kv_cache import KVCacheConfig, PagedKVCache, QuantizedKVToken


class CudaPagedKVCacheError(RuntimeError):
    """Raised when a reference cache cannot be synchronized to CUDA storage."""


@dataclass(frozen=True, slots=True)
class CudaPagedKVCacheView:
    """Kernel-facing tensors and logical metadata for one cached sequence.

    Key and value tensors use
    ``[physical_block, block_offset, layer, head, head_dim]`` layout. Scale
    tensors omit the final dimension. ``block_table`` maps logical blocks to
    physical blocks and contains only the active sequence's entries.
    """

    keys: Any
    values: Any
    key_scales: Any
    value_scales: Any
    block_table: Any
    token_count: int
    block_size: int


@dataclass(frozen=True, slots=True)
class CudaPagedKVCacheStats:
    """Cumulative synchronization and compact device-storage accounting."""

    synchronizations: int
    uploaded_tokens: int
    cleared_tokens: int
    active_tokens: int
    active_blocks: int
    device_storage_bytes: int


class CudaPagedKVCacheStorage:
    """Persistent CUDA buffers mirroring one :class:`PagedKVCache` sequence.

    This class defines the packed device layout and transfer lifecycle. It does
    not implement attention. Synchronization uses the runtime's transfer stream,
    uploads only the changed suffix, clears released physical slots, and returns
    a task whose result is safe for a later kernel to consume.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        runtime: CudaExecutionRuntime,
        *,
        name: str = "paged_kv",
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("CUDA paged-cache storage name cannot be empty")
        self.config = config
        self.runtime = runtime
        self.name = name
        self._torch = runtime.torch_module
        for dtype_name in ("int8", "float32", "long"):
            if not hasattr(self._torch, dtype_name):
                raise CudaPagedKVCacheError(
                    f"PyTorch module does not expose {dtype_name}"
                )

        capacity = config.capacity_tokens
        value_shape = (
            capacity,
            config.num_layers,
            config.num_heads,
            config.head_dim,
        )
        scale_shape = (capacity, config.num_layers, config.num_heads)
        self._keys = runtime.acquire_workspace(
            f"{name}.keys",
            value_shape,
            dtype=self._torch.int8,
        )
        self._values = runtime.acquire_workspace(
            f"{name}.values",
            value_shape,
            dtype=self._torch.int8,
        )
        self._key_scales = runtime.acquire_workspace(
            f"{name}.key_scales",
            scale_shape,
            dtype=self._torch.float32,
        )
        self._value_scales = runtime.acquire_workspace(
            f"{name}.value_scales",
            scale_shape,
            dtype=self._torch.float32,
        )
        self._device_block_table = self._empty_block_table()
        self._sequence_id: Hashable | None = None
        self._tokens: tuple[QuantizedKVToken, ...] = ()
        self._block_table: tuple[int, ...] = ()
        self._synchronizations = 0
        self._uploaded_tokens = 0
        self._cleared_tokens = 0

    def _empty_block_table(self) -> Any:
        return self._torch.empty(
            (0,),
            dtype=self._torch.long,
            device=self.runtime.device,
        )

    def _validate_reference(self, cache: PagedKVCache) -> None:
        if not isinstance(cache, PagedKVCache):
            raise TypeError("cache must be a PagedKVCache")
        if cache.config != self.config:
            raise CudaPagedKVCacheError(
                "reference and CUDA paged-cache configurations differ"
            )

    @staticmethod
    def _token_payload(
        token: QuantizedKVToken,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        keys = [
            [list(head.values) for head in layer]
            for layer in token.keys
        ]
        values = [
            [list(head.values) for head in layer]
            for layer in token.values
        ]
        key_scales = [
            [head.scale for head in layer]
            for layer in token.keys
        ]
        value_scales = [
            [head.scale for head in layer]
            for layer in token.values
        ]
        return keys, values, key_scales, value_scales

    def _cpu_tensor(self, values: Any, *, dtype: Any) -> Any:
        return self._torch.tensor(values, dtype=dtype, device="cpu")

    def _physical_slots(
        self,
        table: tuple[int, ...],
        token_count: int,
    ) -> tuple[int, ...]:
        return tuple(
            table[index // self.config.block_size] * self.config.block_size
            + index % self.config.block_size
            for index in range(token_count)
        )

    @staticmethod
    def _stable_prefix(
        old_tokens: tuple[QuantizedKVToken, ...],
        old_slots: tuple[int, ...],
        new_tokens: tuple[QuantizedKVToken, ...],
        new_slots: tuple[int, ...],
    ) -> int:
        stable = 0
        for old_token, old_slot, new_token, new_slot in zip(
            old_tokens,
            old_slots,
            new_tokens,
            new_slots,
        ):
            if old_slot != new_slot or old_token != new_token:
                break
            stable += 1
        return stable

    def _view(self, *, token_count: int | None = None) -> CudaPagedKVCacheView:
        config = self.config
        return CudaPagedKVCacheView(
            keys=self._keys.view(
                (
                    config.num_blocks,
                    config.block_size,
                    config.num_layers,
                    config.num_heads,
                    config.head_dim,
                )
            ),
            values=self._values.view(
                (
                    config.num_blocks,
                    config.block_size,
                    config.num_layers,
                    config.num_heads,
                    config.head_dim,
                )
            ),
            key_scales=self._key_scales.view(
                (
                    config.num_blocks,
                    config.block_size,
                    config.num_layers,
                    config.num_heads,
                )
            ),
            value_scales=self._value_scales.view(
                (
                    config.num_blocks,
                    config.block_size,
                    config.num_layers,
                    config.num_heads,
                )
            ),
            block_table=self._device_block_table,
            token_count=len(self._tokens) if token_count is None else token_count,
            block_size=config.block_size,
        )

    @property
    def view(self) -> CudaPagedKVCacheView:
        """Return the most recently synchronized kernel-facing view."""

        return self._view()

    @property
    def stats(self) -> CudaPagedKVCacheStats:
        config = self.config
        return CudaPagedKVCacheStats(
            synchronizations=self._synchronizations,
            uploaded_tokens=self._uploaded_tokens,
            cleared_tokens=self._cleared_tokens,
            active_tokens=len(self._tokens),
            active_blocks=len(self._block_table),
            device_storage_bytes=(
                config.capacity_tokens * config.quantized_bytes_per_token
            ),
        )

    def synchronize(
        self,
        cache: PagedKVCache,
        sequence_id: Hashable,
    ) -> CudaTask[CudaPagedKVCacheView]:
        """Synchronize one logical sequence and return its transfer-stream task."""

        self._validate_reference(cache)
        length = cache.sequence_length(sequence_id)
        table = cache.block_table(sequence_id)
        tokens = tuple(
            cache.read_quantized_token(sequence_id, index)
            for index in range(length)
        )
        if self._sequence_id is not None and sequence_id != self._sequence_id:
            raise CudaPagedKVCacheError(
                "CUDA paged-cache storage is already bound to another sequence"
            )

        old_slots = self._physical_slots(self._block_table, len(self._tokens))
        new_slots = self._physical_slots(table, length)
        stable = self._stable_prefix(
            self._tokens,
            old_slots,
            tokens,
            new_slots,
        )
        stale_slots = tuple(sorted(set(old_slots[stable:]) - set(new_slots[stable:])))
        upload_slots = new_slots[stable:]
        upload_tokens = tokens[stable:]

        dependencies: list[CudaTask[Any]] = []
        upload_tasks: tuple[CudaTask[Any], ...] = ()
        if upload_tokens:
            payloads = tuple(self._token_payload(token) for token in upload_tokens)
            source_tensors = (
                self._cpu_tensor(
                    [payload[0] for payload in payloads],
                    dtype=self._torch.int8,
                ),
                self._cpu_tensor(
                    [payload[1] for payload in payloads],
                    dtype=self._torch.int8,
                ),
                self._cpu_tensor(
                    [payload[2] for payload in payloads],
                    dtype=self._torch.float32,
                ),
                self._cpu_tensor(
                    [payload[3] for payload in payloads],
                    dtype=self._torch.float32,
                ),
            )
            slot_source = self._cpu_tensor([list(upload_slots)], dtype=self._torch.long)
            upload_tasks = (
                self.runtime.copy_to_device(f"{self.name}.upload.keys", source_tensors[0]),
                self.runtime.copy_to_device(f"{self.name}.upload.values", source_tensors[1]),
                self.runtime.copy_to_device(
                    f"{self.name}.upload.key_scales", source_tensors[2]
                ),
                self.runtime.copy_to_device(
                    f"{self.name}.upload.value_scales", source_tensors[3]
                ),
                self.runtime.copy_to_device(f"{self.name}.upload.slots", slot_source),
            )
            dependencies.extend(upload_tasks)

        stale_task: CudaTask[Any] | None = None
        if stale_slots:
            stale_source = self._cpu_tensor([list(stale_slots)], dtype=self._torch.long)
            stale_task = self.runtime.copy_to_device(
                f"{self.name}.stale_slots",
                stale_source,
            )
            dependencies.append(stale_task)

        table_task: CudaTask[Any] | None = None
        if table:
            table_source = self._cpu_tensor([list(table)], dtype=self._torch.long)
            table_task = self.runtime.copy_to_device(
                f"{self.name}.block_table",
                table_source,
            )
            dependencies.append(table_task)

        def apply_updates() -> CudaPagedKVCacheView:
            if stale_task is not None:
                stale = stale_task.result.view((len(stale_slots),))
                for storage in (
                    self._keys,
                    self._values,
                    self._key_scales,
                    self._value_scales,
                ):
                    storage.index_fill_(0, stale, 0)
            if upload_tasks:
                slots = upload_tasks[4].result.view((len(upload_slots),))
                self._keys.index_copy_(0, slots, upload_tasks[0].result)
                self._values.index_copy_(0, slots, upload_tasks[1].result)
                self._key_scales.index_copy_(0, slots, upload_tasks[2].result)
                self._value_scales.index_copy_(0, slots, upload_tasks[3].result)
            self._device_block_table = (
                table_task.result.view((len(table),))
                if table_task is not None
                else self._empty_block_table()
            )
            return self._view(token_count=length)

        task = self.runtime.submit_transfer(
            apply_updates,
            wait_for=tuple(dependencies),
            label=f"specdecode.transfer.{self.name}.synchronize",
        )
        self._sequence_id = sequence_id
        self._tokens = tokens
        self._block_table = table
        self._synchronizations += 1
        self._uploaded_tokens += len(upload_tokens)
        self._cleared_tokens += len(stale_slots)
        return task
