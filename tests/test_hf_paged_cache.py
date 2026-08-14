import unittest
from types import SimpleNamespace

from specdecode.hf_paged_cache import (
    HuggingFacePagedCacheConfig,
    HuggingFacePagedCacheError,
    HuggingFacePagedCacheMirror,
)


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class FakeKVTensor:
    def __init__(self, heads, sequence_length, head_dim, base, *, batch=1):
        self.shape = (batch, heads, sequence_length, head_dim)
        self.device = "cuda:0"
        self.dtype = "bfloat16"
        self.rows = [
            [
                base + head * 0.2 + token * 0.05 + dimension * 0.01
                for dimension in range(head_dim)
            ]
            for head in range(heads)
            for token in range(sequence_length)
        ]
        self.heads = heads
        self.sequence_length = sequence_length

    def __getitem__(self, key):
        batch, heads, token, dimensions = key
        assert batch == 0
        assert heads == slice(None)
        assert dimensions == slice(None)
        rows = [
            self.rows[head * self.sequence_length + token]
            for head in range(self.heads)
        ]
        return FakeRows(rows)


class FakeModernCache:
    def __init__(self, legacy):
        self.legacy = legacy

    def to_legacy_cache(self):
        return self.legacy

    @classmethod
    def from_legacy_cache(cls, legacy):
        return cls(legacy)


class FakeMaterializedTensor:
    def __init__(self, values, shape, *, device=None, dtype=None):
        self.values = values
        self.shape = shape
        self.device = device
        self.dtype = dtype


class FakeTorch:
    def tensor(self, values, *, device=None, dtype=None):
        shape = (
            len(values),
            len(values[0]),
            len(values[0][0]),
            len(values[0][0][0]),
        )
        return FakeMaterializedTensor(
            values,
            shape,
            device=device,
            dtype=dtype,
        )

    def empty(self, shape, *, device=None, dtype=None):
        return FakeMaterializedTensor(
            None,
            tuple(shape),
            device=device,
            dtype=dtype,
        )


def make_cache(sequence_length, *, heads=2, head_dim=4, batch=1):
    return tuple(
        (
            FakeKVTensor(
                heads,
                sequence_length,
                head_dim,
                1.0 + layer,
                batch=batch,
            ),
            FakeKVTensor(
                heads,
                sequence_length,
                head_dim,
                -1.0 - layer,
                batch=batch,
            ),
        )
        for layer in range(2)
    )


class HuggingFacePagedCacheMirrorTests(unittest.TestCase):
    def setUp(self):
        model_config = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=2,
            num_attention_heads=2,
            hidden_size=8,
            max_position_embeddings=8,
        )
        self.mirror = HuggingFacePagedCacheMirror.from_model_config(
            model_config,
            HuggingFacePagedCacheConfig(block_size=2),
        )

    def test_model_geometry_and_capacity_are_derived(self) -> None:
        config = self.mirror.cache.config

        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_heads, 2)
        self.assertEqual(config.head_dim, 4)
        self.assertEqual(config.block_size, 2)
        self.assertEqual(config.num_blocks, 4)

    def test_sync_quantizes_new_tokens_without_duplicating_prefix(self) -> None:
        appended = self.mirror.synchronize(FakeModernCache(make_cache(3)))
        second_append = self.mirror.synchronize(make_cache(4))

        self.assertEqual(appended, 3)
        self.assertEqual(second_append, 1)
        self.assertEqual(self.mirror.token_count, 4)
        self.assertEqual(self.mirror.block_table, (0, 1))
        quantized = self.mirror.cache.read_quantized_token("model", 2)
        restored = self.mirror.cache.read_token("model", 2)
        expected = 1.0 + 0.0 + 2 * 0.05 + 0.0
        self.assertLessEqual(
            abs(restored.keys[0][0][0] - expected),
            quantized.keys[0][0].scale / 2.0 + 1e-12,
        )
        self.assertEqual(self.mirror.stats.synchronized_tokens, 4)

    def test_truncate_and_reset_follow_speculative_lifecycle(self) -> None:
        self.mirror.synchronize(make_cache(4))

        removed = self.mirror.truncate(1)
        self.mirror.reset()

        self.assertEqual(removed, 3)
        self.assertEqual(self.mirror.token_count, 0)
        self.assertEqual(self.mirror.stats.rollback_tokens, 3)
        self.assertEqual(self.mirror.stats.resets, 1)
        self.assertEqual(self.mirror.stats.paged_cache.allocated_blocks, 0)

    def test_materialization_restores_hugging_face_tensor_layout(self) -> None:
        source = make_cache(3)
        self.mirror.synchronize(source)

        materialized = self.mirror.materialize_legacy_cache(
            FakeTorch(),
            template=source,
        )

        self.assertEqual(len(materialized), 2)
        key_tensor, value_tensor = materialized[0]
        self.assertEqual(key_tensor.shape, (1, 2, 3, 4))
        self.assertEqual(value_tensor.shape, (1, 2, 3, 4))
        self.assertEqual(key_tensor.device, "cuda:0")
        self.assertEqual(key_tensor.dtype, "bfloat16")
        quantized = self.mirror.cache.read_quantized_token("model", 2)
        expected = 1.0 + 0.0 + 2 * 0.05 + 3 * 0.01
        actual = key_tensor.values[0][0][2][3]
        self.assertLessEqual(
            abs(actual - expected),
            quantized.keys[0][0].scale / 2.0 + 1e-12,
        )
        self.assertEqual(self.mirror.stats.materializations, 1)

    def test_materialize_like_preserves_modern_cache_container(self) -> None:
        template = FakeModernCache(make_cache(2))
        self.mirror.synchronize(template)

        materialized = self.mirror.materialize_like(FakeTorch(), template)

        self.assertIsInstance(materialized, FakeModernCache)
        self.assertEqual(materialized.legacy[1][1].shape, (1, 2, 2, 4))

    def test_empty_cache_materializes_geometry_without_values(self) -> None:
        materialized = self.mirror.materialize_legacy_cache(FakeTorch())

        self.assertEqual(materialized[0][0].shape, (1, 2, 0, 4))
        self.assertEqual(materialized[0][1].shape, (1, 2, 0, 4))

    def test_invalid_model_cache_is_rejected_atomically(self) -> None:
        with self.assertRaises(HuggingFacePagedCacheError):
            self.mirror.synchronize(make_cache(2, heads=1))
        self.assertEqual(self.mirror.token_count, 0)

        with self.assertRaises(HuggingFacePagedCacheError):
            self.mirror.synchronize(make_cache(2, batch=2))
        self.assertEqual(self.mirror.token_count, 0)

    def test_shorter_model_cache_requires_explicit_truncation(self) -> None:
        self.mirror.synchronize(make_cache(3))

        with self.assertRaises(HuggingFacePagedCacheError):
            self.mirror.synchronize(make_cache(2))

        self.assertEqual(self.mirror.token_count, 3)

    def test_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            HuggingFacePagedCacheConfig(block_size=0)
        with self.assertRaises(HuggingFacePagedCacheError):
            HuggingFacePagedCacheMirror.from_model_config(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
