"""Optional PyTorch CUDA scheduling, transfer, and profiling utilities."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import Any, Generic, TypeVar


class CudaDependencyError(ImportError):
    """Raised when the optional PyTorch CUDA dependency is unavailable."""


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA execution is requested without a usable device."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise CudaDependencyError(
            "CUDA execution requires PyTorch. Install the transformers extra with: "
            "pip install -e '.[transformers]'"
        ) from error
    return torch


@dataclass(frozen=True, slots=True)
class CudaRuntimeConfig:
    """Configuration for one CUDA device's reusable execution resources."""

    device: str = "cuda:0"
    enable_timing: bool = True
    enable_nvtx: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.startswith("cuda"):
            raise ValueError("CUDA runtime device must be a CUDA device string")


@dataclass(frozen=True, slots=True)
class CudaRuntimeStats:
    """Allocation/reuse counters for persistent CUDA and pinned buffers."""

    workspace_allocations: int
    workspace_reuses: int
    pinned_allocations: int
    pinned_reuses: int
    submitted_draft_tasks: int
    submitted_target_tasks: int
    submitted_transfer_tasks: int


@dataclass(frozen=True, slots=True)
class CudaMemorySnapshot:
    """Current and peak PyTorch allocator bytes for the configured device."""

    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class CudaTask(Generic[ResultT]):
    """One asynchronously submitted CUDA operation and its completion event."""

    result: ResultT
    completion_event: Any
    start_event: Any | None
    _timing_enabled: bool

    def wait(self) -> ResultT:
        self.completion_event.synchronize()
        return self.result

    def elapsed_ms(self) -> float | None:
        if not self._timing_enabled or self.start_event is None:
            return None
        self.completion_event.synchronize()
        return float(self.start_event.elapsed_time(self.completion_event))


class CudaExecutionRuntime:
    """Own reusable streams, events, device workspaces, and pinned buffers.

    Operations are submitted eagerly using PyTorch's CUDA API. The returned
    :class:`CudaTask` carries an event that can be used as an explicit
    cross-stream dependency without synchronizing the entire device.
    """

    _ROLES = ("draft", "target", "transfer")

    def __init__(
        self,
        config: CudaRuntimeConfig | None = None,
        *,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config or CudaRuntimeConfig()
        self._torch = torch_module if torch_module is not None else _require_torch()
        self._cuda = self._torch.cuda
        if not bool(self._cuda.is_available()):
            raise CudaUnavailableError("PyTorch reports that CUDA is unavailable")

        self.device = self._torch.device(self.config.device)
        if getattr(self.device, "type", "cuda") != "cuda":
            raise ValueError("CUDA runtime resolved a non-CUDA device")
        self._streams = {
            role: self._cuda.Stream(device=self.device) for role in self._ROLES
        }
        self._workspaces: dict[tuple[Any, ...], Any] = {}
        self._pinned_buffers: dict[tuple[Any, ...], Any] = {}
        self._workspace_allocations = 0
        self._workspace_reuses = 0
        self._pinned_allocations = 0
        self._pinned_reuses = 0
        self._submitted = {role: 0 for role in self._ROLES}

    @property
    def streams(self) -> Mapping[str, Any]:
        return dict(self._streams)

    @property
    def torch_module(self) -> Any:
        """Return the lazily supplied PyTorch module for CUDA integrations."""

        return self._torch

    @staticmethod
    def _shape(shape: Sequence[int]) -> tuple[int, ...]:
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, Integral)
            for dimension in shape
        ):
            raise ValueError("buffer dimensions must be integers")
        converted = tuple(int(dimension) for dimension in shape)
        if not converted or any(dimension < 1 for dimension in converted):
            raise ValueError("buffer shape must contain positive dimensions")
        return converted

    def acquire_workspace(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: Any,
    ) -> Any:
        normalized_shape = self._shape(shape)
        required = prod(normalized_shape)
        key = (name, str(dtype))
        if key in self._workspaces:
            workspace = self._workspaces[key]
            if int(workspace.numel()) >= required:
                self._workspace_reuses += 1
                return workspace[:required].view(normalized_shape)
        capacity = 1 << (required - 1).bit_length()
        workspace = self._torch.empty(
            (capacity,),
            dtype=dtype,
            device=self.device,
        )
        self._workspaces[key] = workspace
        self._workspace_allocations += 1
        return workspace[:required].view(normalized_shape)

    def acquire_pinned_buffer(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: Any,
    ) -> Any:
        normalized_shape = self._shape(shape)
        required = prod(normalized_shape)
        key = (name, str(dtype))
        if key in self._pinned_buffers:
            buffer = self._pinned_buffers[key]
            if int(buffer.numel()) >= required:
                self._pinned_reuses += 1
                return buffer[:required].view(normalized_shape)
        capacity = 1 << (required - 1).bit_length()
        buffer = self._torch.empty(
            (capacity,),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self._pinned_buffers[key] = buffer
        self._pinned_allocations += 1
        return buffer[:required].view(normalized_shape)

    def _range_push(self, label: str) -> bool:
        if not self.config.enable_nvtx:
            return False
        nvtx = getattr(self._cuda, "nvtx", None)
        push = getattr(nvtx, "range_push", None)
        if push is None:
            return False
        push(label)
        return True

    def _range_pop(self) -> None:
        nvtx = getattr(self._cuda, "nvtx", None)
        pop = getattr(nvtx, "range_pop", None)
        if pop is not None:
            pop()

    def submit(
        self,
        role: str,
        operation: Callable[[], ResultT],
        *,
        wait_for: Sequence[CudaTask[Any]] = (),
        label: str | None = None,
    ) -> CudaTask[ResultT]:
        if role not in self._streams:
            raise ValueError(f"unknown CUDA stream role: {role}")
        stream = self._streams[role]
        start_event = (
            self._cuda.Event(enable_timing=True) if self.config.enable_timing else None
        )
        completion_event = self._cuda.Event(enable_timing=self.config.enable_timing)

        with self._cuda.stream(stream):
            for dependency in wait_for:
                stream.wait_event(dependency.completion_event)
            if start_event is not None:
                start_event.record(stream)
            range_active = self._range_push(label or f"specdecode.{role}")
            try:
                result = operation()
            finally:
                if range_active:
                    self._range_pop()
            completion_event.record(stream)
        self._submitted[role] += 1
        return CudaTask(
            result=result,
            completion_event=completion_event,
            start_event=start_event,
            _timing_enabled=self.config.enable_timing,
        )

    def submit_draft(
        self,
        operation: Callable[[], ResultT],
        *,
        wait_for: Sequence[CudaTask[Any]] = (),
        label: str = "specdecode.draft",
    ) -> CudaTask[ResultT]:
        return self.submit("draft", operation, wait_for=wait_for, label=label)

    def submit_target(
        self,
        operation: Callable[[], ResultT],
        *,
        wait_for: Sequence[CudaTask[Any]] = (),
        label: str = "specdecode.target",
    ) -> CudaTask[ResultT]:
        return self.submit("target", operation, wait_for=wait_for, label=label)

    def submit_transfer(
        self,
        operation: Callable[[], ResultT],
        *,
        wait_for: Sequence[CudaTask[Any]] = (),
        label: str = "specdecode.transfer",
    ) -> CudaTask[ResultT]:
        return self.submit("transfer", operation, wait_for=wait_for, label=label)

    def copy_to_device(
        self,
        name: str,
        source: Any,
        *,
        wait_for: Sequence[CudaTask[Any]] = (),
    ) -> CudaTask[Any]:
        shape = tuple(int(dimension) for dimension in source.shape)
        dtype = source.dtype
        pinned = self.acquire_pinned_buffer(name, shape, dtype=dtype)
        pinned.copy_(source, non_blocking=False)
        workspace = self.acquire_workspace(name, shape, dtype=dtype)
        return self.submit_transfer(
            lambda: workspace.copy_(pinned, non_blocking=True),
            wait_for=wait_for,
            label=f"specdecode.transfer.{name}",
        )

    def stage_token_ids(
        self,
        name: str,
        token_ids: Sequence[int],
    ) -> CudaTask[Any]:
        if not token_ids:
            raise ValueError("token_ids cannot be empty")
        source = self._torch.tensor(
            [list(token_ids)],
            dtype=self._torch.long,
            device="cpu",
        )
        return self.copy_to_device(name, source)

    def run_speculative_round(
        self,
        draft_operation: Callable[[], ResultT],
        target_operation: Callable[[ResultT], Any],
    ) -> tuple[CudaTask[ResultT], CudaTask[Any]]:
        draft_task = self.submit_draft(draft_operation)
        target_task = self.submit_target(
            lambda: target_operation(draft_task.result),
            wait_for=(draft_task,),
        )
        return draft_task, target_task

    def synchronize(self) -> None:
        self._cuda.synchronize(self.device)

    def reset_peak_memory(self) -> None:
        self._cuda.reset_peak_memory_stats(self.device)

    def memory_snapshot(self) -> CudaMemorySnapshot:
        return CudaMemorySnapshot(
            allocated_bytes=int(self._cuda.memory_allocated(self.device)),
            reserved_bytes=int(self._cuda.memory_reserved(self.device)),
            peak_allocated_bytes=int(self._cuda.max_memory_allocated(self.device)),
            peak_reserved_bytes=int(self._cuda.max_memory_reserved(self.device)),
        )

    def stats(self) -> CudaRuntimeStats:
        return CudaRuntimeStats(
            workspace_allocations=self._workspace_allocations,
            workspace_reuses=self._workspace_reuses,
            pinned_allocations=self._pinned_allocations,
            pinned_reuses=self._pinned_reuses,
            submitted_draft_tasks=self._submitted["draft"],
            submitted_target_tasks=self._submitted["target"],
            submitted_transfer_tasks=self._submitted["transfer"],
        )

    def environment(self) -> dict[str, Any]:
        capability = self._cuda.get_device_capability(self.device)
        driver_version = None
        driver = getattr(self._cuda, "driver_version", None)
        if callable(driver):
            driver_version = driver()
        torch_version = getattr(self._torch, "__version__", "unknown")
        torch_cuda = getattr(getattr(self._torch, "version", None), "cuda", None)
        return {
            "python": sys.version.split()[0],
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "cxx": os.environ.get("CXX"),
            "torch": str(torch_version),
            "torch_cuda": torch_cuda,
            "cuda_driver": driver_version,
            "device": str(self.device),
            "device_name": self._cuda.get_device_name(self.device),
            "compute_capability": list(capability),
        }
