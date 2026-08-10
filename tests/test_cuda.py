import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from specdecode.cuda import (
    CudaExecutionRuntime,
    CudaRuntimeConfig,
    CudaUnavailableError,
)


class FakeDevice:
    def __init__(self, value):
        self.value = value
        self.type = value.split(":", 1)[0]

    def __str__(self):
        return self.value


class FakeTensor:
    def __init__(self, shape, dtype, device, data=None, pin_memory=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.pin_memory = pin_memory
        size = 1
        for dimension in self.shape:
            size *= dimension
        self.data = list(data) if data is not None else [0] * size
        self.copy_calls = []

    def numel(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("fake tensor only supports slicing")
        data = self.data[key]
        return FakeTensor((len(data),), self.dtype, self.device, data, self.pin_memory)

    def view(self, shape):
        return FakeTensor(shape, self.dtype, self.device, self.data, self.pin_memory)

    def copy_(self, source, non_blocking=False):
        self.data = list(source.data)
        self.copy_calls.append(non_blocking)
        return self


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class FakeEvent:
    def __init__(self, cuda, enable_timing):
        self.cuda = cuda
        self.enable_timing = enable_timing
        self.stream = None
        self.tick = None
        self.synchronized = False

    def record(self, stream):
        self.stream = stream
        self.tick = self.cuda.event_tick
        self.cuda.event_tick += 1

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, other):
        return float(other.tick - self.tick)


class FakeNvtx:
    def __init__(self):
        self.operations = []

    def range_push(self, label):
        self.operations.append(("push", label))

    def range_pop(self):
        self.operations.append(("pop", None))


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.streams = []
        self.active_streams = []
        self.event_tick = 0
        self.nvtx = FakeNvtx()
        self.synchronize_calls = []
        self.reset_calls = []

    def is_available(self):
        return self.available

    def Stream(self, device):
        stream = FakeStream(f"stream-{len(self.streams)}")
        self.streams.append((stream, device))
        return stream

    def Event(self, enable_timing):
        return FakeEvent(self, enable_timing)

    @contextmanager
    def stream(self, stream):
        self.active_streams.append(stream)
        yield

    def synchronize(self, device):
        self.synchronize_calls.append(device)

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)

    def memory_allocated(self, device):
        return 10

    def memory_reserved(self, device):
        return 20

    def max_memory_allocated(self, device):
        return 30

    def max_memory_reserved(self, device):
        return 40

    def get_device_capability(self, device):
        return (9, 0)

    def get_device_name(self, device):
        return "Fake GPU"

    def driver_version(self):
        return 12040


class FakeTorch:
    __version__ = "2.fake"
    long = "int64"
    version = SimpleNamespace(cuda="12.4")

    def __init__(self, *, available=True):
        self.cuda = FakeCuda(available)
        self.empty_calls = []

    def device(self, value):
        return FakeDevice(value)

    def empty(self, shape, *, dtype, device, pin_memory=False):
        tensor = FakeTensor(shape, dtype, device, pin_memory=pin_memory)
        self.empty_calls.append(tensor)
        return tensor

    def tensor(self, values, *, dtype, device):
        flat = [value for row in values for value in row]
        return FakeTensor((len(values), len(values[0])), dtype, device, flat)


class CudaRuntimeTests(unittest.TestCase):
    def test_unavailable_cuda_is_rejected(self) -> None:
        with self.assertRaises(CudaUnavailableError):
            CudaExecutionRuntime(torch_module=FakeTorch(available=False))

    def test_config_requires_cuda_device(self) -> None:
        with self.assertRaises(ValueError):
            CudaRuntimeConfig(device="cpu")

    def test_buffer_shapes_require_integer_dimensions(self) -> None:
        runtime = CudaExecutionRuntime(torch_module=FakeTorch())
        with self.assertRaises(ValueError):
            runtime.acquire_workspace("bad", (1.5,), dtype="float32")

    def test_stream_dependencies_events_timing_and_nvtx(self) -> None:
        torch = FakeTorch()
        runtime = CudaExecutionRuntime(torch_module=torch)

        draft = runtime.submit_draft(lambda: "proposal")
        target = runtime.submit_target(
            lambda: "verified",
            wait_for=(draft,),
            label="verify-proposal",
        )

        self.assertEqual(target.wait(), "verified")
        self.assertEqual(draft.elapsed_ms(), 1.0)
        self.assertEqual(runtime.streams["target"].waited_events, [draft.completion_event])
        self.assertEqual(
            torch.cuda.nvtx.operations,
            (
                [("push", "specdecode.draft"), ("pop", None)]
                + [("push", "verify-proposal"), ("pop", None)]
            ),
        )
        stats = runtime.stats()
        self.assertEqual(stats.submitted_draft_tasks, 1)
        self.assertEqual(stats.submitted_target_tasks, 1)

    def test_geometric_workspaces_and_pinned_buffers_are_reused(self) -> None:
        runtime = CudaExecutionRuntime(torch_module=FakeTorch())

        first_workspace = runtime.acquire_workspace("ids", (3,), dtype="int64")
        second_workspace = runtime.acquire_workspace("ids", (2,), dtype="int64")
        grown_workspace = runtime.acquire_workspace("ids", (5,), dtype="int64")
        first_pinned = runtime.acquire_pinned_buffer("ids", (3,), dtype="int64")
        second_pinned = runtime.acquire_pinned_buffer("ids", (2,), dtype="int64")

        self.assertEqual(first_workspace.shape, (3,))
        self.assertEqual(second_workspace.shape, (2,))
        self.assertEqual(grown_workspace.shape, (5,))
        self.assertTrue(first_pinned.pin_memory)
        self.assertTrue(second_pinned.pin_memory)
        stats = runtime.stats()
        self.assertEqual(stats.workspace_allocations, 2)
        self.assertEqual(stats.workspace_reuses, 1)
        self.assertEqual(stats.pinned_allocations, 1)
        self.assertEqual(stats.pinned_reuses, 1)

    def test_copy_to_device_uses_pinned_memory_and_nonblocking_copy(self) -> None:
        torch = FakeTorch()
        runtime = CudaExecutionRuntime(torch_module=torch)
        source = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cpu")

        transfer = runtime.copy_to_device("tokens", source)
        result = transfer.wait()

        self.assertEqual(result.data, [1, 2, 3])
        self.assertEqual(result.copy_calls, [True])
        pinned = [tensor for tensor in torch.empty_calls if tensor.pin_memory]
        self.assertEqual(len(pinned), 1)
        self.assertEqual(runtime.stats().submitted_transfer_tasks, 1)

    def test_round_helper_connects_draft_and_target_streams(self) -> None:
        runtime = CudaExecutionRuntime(torch_module=FakeTorch())

        draft, target = runtime.run_speculative_round(
            lambda: (1, 2),
            lambda proposal: proposal + (3,),
        )

        self.assertEqual(draft.result, (1, 2))
        self.assertEqual(target.wait(), (1, 2, 3))
        self.assertEqual(runtime.streams["target"].waited_events, [draft.completion_event])

    def test_memory_and_environment_metadata_are_reported(self) -> None:
        runtime = CudaExecutionRuntime(torch_module=FakeTorch())

        runtime.reset_peak_memory()
        snapshot = runtime.memory_snapshot()
        environment = runtime.environment()

        self.assertEqual(snapshot.allocated_bytes, 10)
        self.assertEqual(snapshot.peak_reserved_bytes, 40)
        self.assertEqual(environment["torch"], "2.fake")
        self.assertEqual(environment["torch_cuda"], "12.4")
        self.assertEqual(environment["cuda_driver"], 12040)
        self.assertEqual(environment["device_name"], "Fake GPU")
        self.assertEqual(environment["compute_capability"], [9, 0])

    def test_stage_token_ids_rejects_empty_input(self) -> None:
        runtime = CudaExecutionRuntime(torch_module=FakeTorch())
        with self.assertRaises(ValueError):
            runtime.stage_token_ids("tokens", ())


if __name__ == "__main__":
    unittest.main()
