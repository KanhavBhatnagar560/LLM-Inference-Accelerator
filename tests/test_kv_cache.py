import math
import unittest

from specdecode.kv_cache import (
    KVCacheCapacityError,
    KVCacheConfig,
    PagedKVCache,
    PythonKVQuantizer,
    QuantizedVector,
)


def make_tensor(config, base):
    return tuple(
        tuple(
            tuple(
                base + layer * 0.7 + head * 0.2 + dimension * 0.05
                for dimension in range(config.head_dim)
            )
            for head in range(config.num_heads)
        )
        for layer in range(config.num_layers)
    )


def make_entry(config, base):
    return make_tensor(config, base), make_tensor(config, -base - 0.25)


class QuantizationTests(unittest.TestCase):
    def test_symmetric_int8_quantization_and_error_bound(self) -> None:
        quantizer = PythonKVQuantizer()
        source = (-1.0, -0.5, 0.0, 0.5, 1.0)

        quantized = quantizer.quantize(source)
        restored = quantizer.dequantize(quantized)

        self.assertEqual(quantized.values, (-127, -64, 0, 64, 127))
        self.assertAlmostEqual(quantized.scale, 1.0 / 127.0)
        for expected, actual in zip(source, restored):
            self.assertLessEqual(abs(expected - actual), quantized.scale / 2.0 + 1e-15)

    def test_zero_vector_uses_zero_scale(self) -> None:
        quantizer = PythonKVQuantizer()

        quantized = quantizer.quantize((0.0, 0.0, 0.0))

        self.assertEqual(quantized, QuantizedVector((0, 0, 0), 0.0))
        self.assertEqual(quantizer.dequantize(quantized), (0.0, 0.0, 0.0))

    def test_invalid_quantization_inputs_are_rejected(self) -> None:
        quantizer = PythonKVQuantizer()
        for values in ((), (math.nan,), (math.inf,), (True,)):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    quantizer.quantize(values)
        with self.assertRaises(ValueError):
            QuantizedVector((1,), 0.0)


class PagedKVCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = KVCacheConfig(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            block_size=2,
            num_blocks=4,
        )
        self.cache = PagedKVCache(self.config)
        self.cache.create_sequence("request-a")

    def test_config_validates_shape_and_accounts_for_format(self) -> None:
        self.assertEqual(self.config.fp32_bytes_per_token, 128)
        self.assertEqual(self.config.quantized_bytes_per_token, 64)
        self.assertEqual(self.config.capacity_tokens, 8)
        self.assertEqual(self.config.theoretical_savings_ratio, 0.5)
        with self.assertRaises(ValueError):
            KVCacheConfig(num_layers=1, num_heads=0, head_dim=4)

    def test_append_builds_logical_to_physical_block_table(self) -> None:
        indices = self.cache.append_many(
            "request-a",
            tuple(make_entry(self.config, float(index + 1)) for index in range(3)),
        )

        self.assertEqual(indices, (0, 1, 2))
        self.assertEqual(self.cache.block_table("request-a"), (0, 1))
        self.assertEqual(self.cache.physical_slot("request-a", 0), (0, 0))
        self.assertEqual(self.cache.physical_slot("request-a", 1), (0, 1))
        self.assertEqual(self.cache.physical_slot("request-a", 2), (1, 0))
        self.cache.validate_invariants()

    def test_dequantized_values_stay_within_per_head_error_bound(self) -> None:
        keys, values = make_entry(self.config, 1.0)
        self.cache.append("request-a", keys, values)

        quantized = self.cache.read_quantized_token("request-a", 0)
        restored = self.cache.read_token("request-a", 0)

        for source_tensor, quantized_tensor, restored_tensor in (
            (keys, quantized.keys, restored.keys),
            (values, quantized.values, restored.values),
        ):
            for source_layer, quantized_layer, restored_layer in zip(
                source_tensor,
                quantized_tensor,
                restored_tensor,
            ):
                for source_head, quantized_head, restored_head in zip(
                    source_layer,
                    quantized_layer,
                    restored_layer,
                ):
                    for expected, actual in zip(source_head, restored_head):
                        self.assertLessEqual(
                            abs(expected - actual),
                            quantized_head.scale / 2.0 + 1e-14,
                        )

    def test_append_many_is_atomic_on_shape_failure(self) -> None:
        good = make_entry(self.config, 1.0)
        malformed_keys = (((1.0, 2.0, 3.0),),)

        with self.assertRaises(ValueError):
            self.cache.append_many(
                "request-a",
                (good, (malformed_keys, good[1])),
            )

        self.assertEqual(self.cache.sequence_length("request-a"), 0)
        self.assertEqual(self.cache.block_table("request-a"), ())
        self.assertEqual(self.cache.stats().free_blocks, self.config.num_blocks)

    def test_capacity_failure_does_not_allocate_partial_blocks(self) -> None:
        small_config = KVCacheConfig(1, 1, 2, block_size=2, num_blocks=1)
        cache = PagedKVCache(small_config)
        cache.create_sequence("limited")

        with self.assertRaises(KVCacheCapacityError):
            cache.append_many(
                "limited",
                tuple(make_entry(small_config, float(index)) for index in range(3)),
            )

        self.assertEqual(cache.sequence_length("limited"), 0)
        self.assertEqual(cache.stats().free_blocks, 1)
        cache.validate_invariants()

    def test_checkpoint_rollback_releases_suffix_blocks(self) -> None:
        self.cache.append_many(
            "request-a",
            tuple(make_entry(self.config, float(index)) for index in range(1, 4)),
        )
        checkpoint = self.cache.checkpoint("request-a")
        self.cache.append_many(
            "request-a",
            tuple(make_entry(self.config, float(index)) for index in range(4, 7)),
        )

        released = self.cache.rollback("request-a", checkpoint)

        self.assertEqual(released, 1)
        self.assertEqual(self.cache.sequence_length("request-a"), 3)
        self.assertEqual(self.cache.block_table("request-a"), (0, 1))
        self.cache.append("request-a", *make_entry(self.config, 9.0))
        self.assertEqual(self.cache.physical_slot("request-a", 3), (1, 1))
        self.cache.validate_invariants()

    def test_checkpoint_cannot_be_applied_to_another_sequence(self) -> None:
        checkpoint = self.cache.checkpoint("request-a")
        self.cache.create_sequence("request-b")

        with self.assertRaises(ValueError):
            self.cache.rollback("request-b", checkpoint)

    def test_sequences_are_isolated_and_removal_reclaims_blocks(self) -> None:
        self.cache.create_sequence("request-b")
        self.cache.append("request-a", *make_entry(self.config, 1.0))
        self.cache.append("request-b", *make_entry(self.config, 2.0))

        self.assertEqual(self.cache.block_table("request-a"), (0,))
        self.assertEqual(self.cache.block_table("request-b"), (1,))
        released = self.cache.remove_sequence("request-a")

        self.assertEqual(released, 1)
        self.cache.create_sequence("request-c")
        self.cache.append("request-c", *make_entry(self.config, 3.0))
        self.assertEqual(self.cache.block_table("request-c"), (0,))
        self.cache.validate_invariants()

    def test_memory_stats_distinguish_used_and_allocated_storage(self) -> None:
        self.cache.append_many(
            "request-a",
            tuple(make_entry(self.config, float(index)) for index in range(3)),
        )

        stats = self.cache.stats()

        self.assertEqual(stats.used_tokens, 3)
        self.assertEqual(stats.allocated_blocks, 2)
        self.assertEqual(stats.free_blocks, 2)
        self.assertEqual(stats.used_quantized_bytes, 3 * 64)
        self.assertEqual(stats.allocated_quantized_bytes, 4 * 64)
        self.assertEqual(stats.capacity_quantized_bytes, 8 * 64)
        self.assertEqual(stats.fp32_equivalent_bytes, 3 * 128)
        self.assertEqual(stats.block_table_bytes, 8)
        self.assertEqual(stats.theoretical_savings_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
