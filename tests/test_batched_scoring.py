import random
import unittest

from specdecode import CausalLMProbabilityAdapter, DecodeConfig, SpeculativeDecoder, TableModel
from specdecode.sampling import DistributionError


class RecordingCausalAdapter(CausalLMProbabilityAdapter):
    def __init__(self) -> None:
        super().__init__(vocab_size=2)
        self.calls = []

    def _probabilities_at_positions(self, token_ids, positions):
        self.calls.append((tuple(token_ids), tuple(positions)))
        return [(1.0, 0.0) for _ in positions]


class MalformedScorer:
    vocab_size = 2

    def next_token_probs(self, token_ids):
        return (1.0, 0.0)

    def score_proposal(self, prefix, proposal):
        return [(1.0, 0.0)]


class BatchedScoringTests(unittest.TestCase):
    def test_causal_logit_positions_are_aligned(self) -> None:
        adapter = RecordingCausalAdapter()

        rows = adapter.score_proposal([7, 8, 9], [0, 1])

        self.assertEqual(len(rows), 3)
        self.assertEqual(adapter.calls, [((7, 8, 9, 0, 1), (2, 3, 4))])

    def test_decoder_uses_one_batched_target_call(self) -> None:
        draft = TableModel({}, default=(1.0, 0.0))
        target = RecordingCausalAdapter()
        decoder = SpeculativeDecoder(
            draft,
            target,
            DecodeConfig(
                max_new_tokens=2,
                initial_draft_tokens=1,
                min_draft_tokens=1,
                max_draft_tokens=1,
                dynamic_draft=False,
            ),
            rng=random.Random(4),
        )

        result = decoder.generate([1])

        self.assertEqual(result.generated_tokens, (0, 0))
        self.assertEqual(target.calls, [((1, 0), (0, 1))])
        self.assertEqual(result.stats.verification_rounds, 1)

    def test_malformed_batched_result_is_rejected(self) -> None:
        decoder = SpeculativeDecoder(
            TableModel({}, default=(1.0, 0.0)),
            MalformedScorer(),
            DecodeConfig(max_new_tokens=2, initial_draft_tokens=1),
            rng=random.Random(1),
        )

        with self.assertRaises(DistributionError):
            decoder.generate([0])


if __name__ == "__main__":
    unittest.main()
