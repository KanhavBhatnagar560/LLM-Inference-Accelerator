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
