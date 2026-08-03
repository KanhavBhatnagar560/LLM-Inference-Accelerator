import random
import unittest

from specdecode import DecodeConfig, SpeculativeDecoder, TableModel


class BrokenDraft:
    vocab_size = 2

    def next_token_probs(self, token_ids):
        raise RuntimeError("simulated draft failure")


class DecoderTests(unittest.TestCase):
    def test_identical_models_accept_and_stop_at_eos(self) -> None:
        model = TableModel(
            {(): (0.0, 1.0, 0.0), (1,): (0.0, 0.0, 1.0)},
            default=(0.0, 0.0, 1.0),
        )
        decoder = SpeculativeDecoder(
            model,
            model,
            DecodeConfig(max_new_tokens=8, initial_draft_tokens=2, eos_token_id=2),
            rng=random.Random(1),
        )

        result = decoder.generate([])

        self.assertEqual(result.generated_tokens, (1, 2))
        self.assertEqual(result.stats.accepted_tokens, 2)
        self.assertEqual(result.stats.rejected_tokens, 0)

    def test_rejection_uses_residual_correction(self) -> None:
        draft = TableModel({}, default=(1.0, 0.0))
        target = TableModel({}, default=(0.0, 1.0))
        decoder = SpeculativeDecoder(
            draft,
            target,
            DecodeConfig(max_new_tokens=1, initial_draft_tokens=1),
            rng=random.Random(3),
        )

        result = decoder.generate([])

        self.assertEqual(result.generated_tokens, (1,))
        self.assertEqual(result.stats.rejected_tokens, 1)
        self.assertEqual(result.stats.target_tokens, 1)

    def test_broken_draft_falls_back_to_target(self) -> None:
        target = TableModel({}, default=(0.0, 1.0))
        decoder = SpeculativeDecoder(
            BrokenDraft(),
            target,
            DecodeConfig(max_new_tokens=3),
            rng=random.Random(2),
        )

        result = decoder.generate([])

        self.assertEqual(result.generated_tokens, (1, 1, 1))
        self.assertEqual(result.stats.fallback_steps, 3)
        self.assertEqual(result.stats.drafted_tokens, 0)

    def test_never_exceeds_max_new_tokens(self) -> None:
        model = TableModel({}, default=(0.5, 0.5))
        decoder = SpeculativeDecoder(
            model,
            model,
            DecodeConfig(max_new_tokens=5, initial_draft_tokens=4),
            rng=random.Random(4),
        )
        self.assertEqual(len(decoder.generate([1]).generated_tokens), 5)

    def test_empirical_output_matches_target_distribution(self) -> None:
        draft = TableModel({}, default=(0.85, 0.15))
        target = TableModel({}, default=(0.30, 0.70))
        decoder = SpeculativeDecoder(
            draft,
            target,
            DecodeConfig(max_new_tokens=1, initial_draft_tokens=1),
            rng=random.Random(2026),
        )

        samples = 20_000
        token_one = sum(decoder.generate([]).generated_tokens[0] == 1 for _ in range(samples))
        observed = token_one / samples

        self.assertAlmostEqual(observed, 0.70, delta=0.015)


if __name__ == "__main__":
    unittest.main()

