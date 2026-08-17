import unittest
from contextlib import contextmanager
from itertools import product
from math import exp, sqrt
from types import SimpleNamespace

from specdecode.cuda import (
    CudaExecutionRuntime,
    CudaRuntimeConfig,
    CudaUnavailableError,
)
from specdecode.cuda_kv_cache import (
    CudaPagedAttentionError,
    CudaPagedKVCacheError,
    CudaPagedKVCacheStorage,
    submit_paged_attention,
)
from specdecode.kv_cache import KVCacheConfig, PagedKVCache


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

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def _offset(shape, coordinates):
        offset = 0
        for dimension, coordinate in zip(shape, coordinates):
            offset = offset * dimension + coordinate
        return offset

    @staticmethod
    def _coordinates(shape):
        return product(*(range(dimension) for dimension in shape))

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("fake tensor only supports slicing")
        start, stop, step = key.indices(self.shape[0])
        if step != 1:
            raise ValueError("fake tensor only supports contiguous slicing")
        row_size = 1
        for size in self.shape[1:]:
            row_size *= size
        data = self.data[start * row_size : stop * row_size]
        shape = (max(0, stop - start),) + self.shape[1:]
        return FakeTensor(shape, self.dtype, self.device, data, self.pin_memory)

    def view(self, shape):
        return FakeTensor(shape, self.dtype, self.device, self.data, self.pin_memory)

    def flatten(self, start_dim, end_dim):
        merged = 1
        for dimension in self.shape[start_dim : end_dim + 1]:
            merged *= dimension
        shape = self.shape[:start_dim] + (merged,) + self.shape[end_dim + 1 :]
        return self.view(shape)

    def index_select(self, dimension, indices):
        if dimension != 0:
            raise ValueError("fake tensor only supports dimension-zero index_select")
        row_size = 1
        for size in self.shape[1:]:
            row_size *= size
        data = []
        for row in indices.data:
            start = row * row_size
            data.extend(self.data[start : start + row_size])
        return FakeTensor(
            (len(indices.data),) + self.shape[1:],
            self.dtype,
            self.device,
            data,
            self.pin_memory,
        )

    def select(self, dimension, index):
        if not 0 <= index < self.shape[dimension]:
            raise IndexError("fake tensor selection is out of range")
        shape = self.shape[:dimension] + self.shape[dimension + 1 :]
        data = []
        for coordinates in self._coordinates(shape):
            source = coordinates[:dimension] + (index,) + coordinates[dimension:]
            data.append(self.data[self._offset(self.shape, source)])
        return FakeTensor(shape, self.dtype, self.device, data, self.pin_memory)

    def unsqueeze(self, dimension):
        if dimension < 0:
            dimension += len(self.shape) + 1
        shape = self.shape[:dimension] + (1,) + self.shape[dimension:]
        return self.view(shape)

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        data = []
        for coordinates in self._coordinates(tuple(shape)):
            source = list(coordinates)
            source[first], source[second] = source[second], source[first]
            data.append(self.data[self._offset(self.shape, source)])
        return FakeTensor(tuple(shape), self.dtype, self.device, data, self.pin_memory)

    def repeat_interleave(self, repeats, *, dim):
        shape = list(self.shape)
        shape[dim] *= repeats
        data = []
        for coordinates in self._coordinates(tuple(shape)):
            source = list(coordinates)
            source[dim] //= repeats
            data.append(self.data[self._offset(self.shape, source)])
        return FakeTensor(tuple(shape), self.dtype, self.device, data, self.pin_memory)

    def to(self, *, dtype):
        return FakeTensor(self.shape, dtype, self.device, self.data, self.pin_memory)

    def _binary(self, other, operation, *, dtype=None):
        output_shape = []
        left = (1,) * (max(self.ndim, other.ndim) - self.ndim) + self.shape
        right = (1,) * (max(self.ndim, other.ndim) - other.ndim) + other.shape
        for left_size, right_size in zip(left, right):
            if left_size != right_size and left_size != 1 and right_size != 1:
                raise ValueError("fake tensor shapes are not broadcastable")
            output_shape.append(max(left_size, right_size))
        data = []
        for coordinates in self._coordinates(tuple(output_shape)):
            left_coordinates = tuple(
                0 if size == 1 else coordinate
                for size, coordinate in zip(left, coordinates)
            )
            right_coordinates = tuple(
                0 if size == 1 else coordinate
                for size, coordinate in zip(right, coordinates)
            )
            left_value = self.data[self._offset(left, left_coordinates)]
            right_value = other.data[self._offset(right, right_coordinates)]
            data.append(operation(left_value, right_value))
        return FakeTensor(
            tuple(output_shape),
            self.dtype if dtype is None else dtype,
            self.device,
            data,
        )

    def __mul__(self, other):
        return self._binary(other, lambda left, right: left * right)

    def __ge__(self, other):
        return self._binary(other, lambda left, right: left >= right, dtype="bool")

    def copy_(self, source, non_blocking=False):
        self.data = list(source.data)
        self.copy_calls.append(non_blocking)
        return self

    def index_copy_(self, dimension, indices, source):
        if dimension != 0:
            raise ValueError("fake tensor only supports dimension-zero index_copy_")
        row_size = 1
        for size in self.shape[1:]:
            row_size *= size
        for source_row, target_row in enumerate(indices.data):
            source_start = source_row * row_size
            target_start = target_row * row_size
            self.data[target_start : target_start + row_size] = source.data[
                source_start : source_start + row_size
            ]
        return self

    def index_fill_(self, dimension, indices, value):
        if dimension != 0:
            raise ValueError("fake tensor only supports dimension-zero index_fill_")
        row_size = 1
        for size in self.shape[1:]:
            row_size *= size
        for target_row in indices.data:
            target_start = target_row * row_size
            self.data[target_start : target_start + row_size] = [value] * row_size
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


class FakeFunctional:
    def __init__(self):
        self.last_attention_mask = None

    def scaled_dot_product_attention(
        self,
        query,
        keys,
        values,
        *,
        attn_mask,
        dropout_p,
    ):
        if dropout_p != 0.0:
            raise ValueError("fake attention does not support dropout")
        self.last_attention_mask = attn_mask
        batch, heads, query_tokens, head_dim = query.shape
        key_tokens = keys.shape[2]
        output = []
        for batch_index in range(batch):
            for head_index in range(heads):
                for query_index in range(query_tokens):
                    scores = []
                    for key_index in range(key_tokens):
                        if not attn_mask.data[query_index * key_tokens + key_index]:
                            scores.append(float("-inf"))
                            continue
                        score = 0.0
                        for dimension in range(head_dim):
                            query_offset = FakeTensor._offset(
                                query.shape,
                                (batch_index, head_index, query_index, dimension),
                            )
                            key_offset = FakeTensor._offset(
                                keys.shape,
                                (batch_index, head_index, key_index, dimension),
                            )
                            score += query.data[query_offset] * keys.data[key_offset]
                        scores.append(score / sqrt(head_dim))
                    maximum = max(score for score in scores if score != float("-inf"))
                    weights = [
                        0.0 if score == float("-inf") else exp(score - maximum)
                        for score in scores
                    ]
                    total = sum(weights)
                    for dimension in range(head_dim):
                        value = 0.0
                        for key_index, weight in enumerate(weights):
                            value_offset = FakeTensor._offset(
                                values.shape,
                                (batch_index, head_index, key_index, dimension),
                            )
                            value += weight * values.data[value_offset]
                        output.append(value / total)
        return FakeTensor(query.shape, query.dtype, query.device, output)


class FakeTorch:
    __version__ = "2.fake"
    float32 = "float32"
    int8 = "int8"
    long = "int64"
    version = SimpleNamespace(cuda="12.4")

    def __init__(self, *, available=True):
        self.cuda = FakeCuda(available)
        self.empty_calls = []
        self.nn = SimpleNamespace(functional=FakeFunctional())

    def device(self, value):
        return FakeDevice(value)

    def empty(self, shape, *, dtype, device, pin_memory=False):
        tensor = FakeTensor(shape, dtype, device, pin_memory=pin_memory)
        self.empty_calls.append(tensor)
        return tensor

    def tensor(self, values, *, dtype, device):
        def shape_of(value):
            shape = []
            while isinstance(value, (list, tuple)):
                shape.append(len(value))
                if not value:
                    break
                value = value[0]
            return tuple(shape)

        def flatten(value):
            if not isinstance(value, (list, tuple)):
                return [value]
            result = []
            for item in value:
                result.extend(flatten(item))
            return result

        return FakeTensor(shape_of(values), dtype, device, flatten(values))

    def arange(self, start, stop=None, *, device):
        if stop is None:
            start, stop = 0, start
        return FakeTensor((stop - start,), self.long, device, range(start, stop))


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


class CudaPagedKVCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = KVCacheConfig(
            num_layers=1,
            num_heads=1,
            head_dim=2,
            block_size=2,
            num_blocks=3,
        )
        self.cache = PagedKVCache(self.config)
        self.cache.create_sequence("model")
        self.runtime = CudaExecutionRuntime(torch_module=FakeTorch())
        self.storage = CudaPagedKVCacheStorage(self.config, self.runtime)

    @staticmethod
    def entry(seed):
        keys = (((float(seed), float(-seed)),),)
        values = (((float(seed + 1), float(seed + 2)),),)
        return keys, values

    def append(self, *seeds):
        self.cache.append_many(
            "model",
            tuple(self.entry(seed) for seed in seeds),
        )

    def test_packs_physical_layout_and_block_table(self) -> None:
        self.append(1, 2, 3)

        view = self.storage.synchronize(self.cache, "model").wait()

        self.assertEqual(view.keys.shape, (3, 2, 1, 1, 2))
        self.assertEqual(view.key_scales.shape, (3, 2, 1, 1))
        self.assertEqual(view.block_table.data, [0, 1])
        self.assertEqual(view.token_count, 3)
        expected_keys = []
        for index in range(3):
            expected_keys.extend(
                self.cache.read_quantized_token("model", index).keys[0][0].values
            )
        self.assertEqual(view.keys.data[:6], expected_keys)
        stats = self.storage.stats
        self.assertEqual(stats.synchronizations, 1)
        self.assertEqual(stats.uploaded_tokens, 3)
        self.assertEqual(stats.active_tokens, 3)
        self.assertEqual(stats.active_blocks, 2)
        self.assertEqual(
            stats.device_storage_bytes,
            self.config.capacity_tokens * self.config.quantized_bytes_per_token,
        )

    def test_incremental_sync_uploads_only_changed_suffix(self) -> None:
        self.append(1, 2)
        self.storage.synchronize(self.cache, "model").wait()
        self.append(3)

        view = self.storage.synchronize(self.cache, "model").wait()

        self.assertEqual(view.token_count, 3)
        self.assertEqual(self.storage.stats.synchronizations, 2)
        self.assertEqual(self.storage.stats.uploaded_tokens, 3)

    def test_rollback_clears_released_physical_slots(self) -> None:
        self.append(1, 2, 3, 4)
        self.storage.synchronize(self.cache, "model").wait()
        self.cache.truncate("model", 1)

        view = self.storage.synchronize(self.cache, "model").wait()

        self.assertEqual(view.token_count, 1)
        self.assertEqual(view.block_table.data, [0])
        self.assertEqual(view.keys.data[2:8], [0] * 6)
        self.assertEqual(self.storage.stats.cleared_tokens, 3)

    def test_rewrites_reused_slots_after_unsynchronized_rollback(self) -> None:
        self.append(1, 2, 3)
        self.storage.synchronize(self.cache, "model").wait()
        self.cache.truncate("model", 1)
        self.append(8, 9)

        view = self.storage.synchronize(self.cache, "model").wait()

        expected_keys = []
        for index in range(3):
            expected_keys.extend(
                self.cache.read_quantized_token("model", index).keys[0][0].values
            )
        self.assertEqual(view.keys.data[:6], expected_keys)
        self.assertEqual(self.storage.stats.uploaded_tokens, 5)

    def test_rejects_config_mismatch_and_second_sequence(self) -> None:
        incompatible = PagedKVCache(
            KVCacheConfig(num_layers=1, num_heads=1, head_dim=3)
        )
        incompatible.create_sequence("model")
        with self.assertRaises(CudaPagedKVCacheError):
            self.storage.synchronize(incompatible, "model")

        self.storage.synchronize(self.cache, "model").wait()
        self.cache.create_sequence("other")
        with self.assertRaises(CudaPagedKVCacheError):
            self.storage.synchronize(self.cache, "other")

    def test_paged_attention_matches_reference_and_waits_for_sync(self) -> None:
        self.cache.create_sequence("occupied")
        self.cache.append_many("occupied", (self.entry(99), self.entry(100)))
        self.append(1, 2, 3)
        synchronized = self.storage.synchronize(self.cache, "model")
        self.assertEqual(synchronized.result.block_table.data, [1, 2])
        torch = self.runtime.torch_module
        query_values = [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ]
        query = torch.tensor(
            query_values,
            dtype=torch.float32,
            device=self.runtime.device,
        )

        task = submit_paged_attention(
            self.runtime,
            query,
            synchronized.result,
            0,
            wait_for=(synchronized,),
        )
        output = task.wait()

        tokens = self.cache.read_sequence("model")
        expected = []
        for _head in range(2):
            for query_index, query_row in enumerate(query_values[0][0]):
                absolute_position = len(tokens) - 2 + query_index
                scores = [
                    sum(
                        query_row[dimension] * token.keys[0][0][dimension]
                        for dimension in range(2)
                    )
                    / sqrt(2)
                    for token in tokens[: absolute_position + 1]
                ]
                maximum = max(scores)
                weights = [exp(score - maximum) for score in scores]
                total = sum(weights)
                for dimension in range(2):
                    expected.append(
                        sum(
                            weight * token.values[0][0][dimension]
                            for weight, token in zip(
                                weights,
                                tokens[: absolute_position + 1],
                            )
                        )
                        / total
                    )
        self.assertEqual(output.shape, (1, 2, 2, 2))
        for actual, reference in zip(output.data, expected):
            self.assertAlmostEqual(actual, reference, places=6)
        self.assertEqual(
            torch.nn.functional.last_attention_mask.data,
            [True, True, False, True, True, True],
        )
        self.assertEqual(
            self.runtime.streams["target"].waited_events,
            [synchronized.completion_event],
        )

    def test_paged_attention_rejects_invalid_query_geometry(self) -> None:
        self.append(1)
        view = self.storage.synchronize(self.cache, "model").wait()
        torch = self.runtime.torch_module
        too_long = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]]],
            dtype=torch.float32,
            device=self.runtime.device,
        )

        with self.assertRaises(CudaPagedAttentionError):
            submit_paged_attention(self.runtime, too_long, view, 0)
        with self.assertRaises(CudaPagedAttentionError):
            submit_paged_attention(self.runtime, too_long, view, 1)


if __name__ == "__main__":
    unittest.main()
