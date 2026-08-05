import random
import unittest

from specdecode import DecodeConfig, SpeculativeDecoder, TableModel


class BrokenDraft:
    vocab_size = 2

    def next_token_probs(self, token_ids):
        raise RuntimeError("simulated draft failure")


class StreamingTests(unittest.TestCase):
    def test_events_cover_accepted_and_bonus_tokens(self) -> None:
        model = TableModel({}, default=(1.0, 0.0))
        events = []
        result = SpeculativeDecoder(
            model,
            model,
            DecodeConfig(max_new_tokens=3, initial_draft_tokens=2),
            rng=random.Random(2),
        ).generate([0], on_token=events.append)

        self.assertEqual(tuple(event.token_id for event in events), result.generated_tokens)
        self.assertEqual([event.index for event in events], [0, 1, 2])
        self.assertEqual(
            [event.source for event in events],
            ["accepted_draft", "accepted_draft", "target_bonus"],
        )

    def test_rejection_emits_one_correction_event(self) -> None:
        events = []
        SpeculativeDecoder(
            TableModel({}, default=(1.0, 0.0)),
            TableModel({}, default=(0.0, 1.0)),
            DecodeConfig(max_new_tokens=1, initial_draft_tokens=1),
            rng=random.Random(3),
        ).generate([0], on_token=events.append)

        self.assertEqual([(event.token_id, event.source) for event in events], [(1, "target_correction")])

    def test_draft_failure_emits_fallback_events(self) -> None:
        events = []
        SpeculativeDecoder(
            BrokenDraft(),
            TableModel({}, default=(0.0, 1.0)),
            DecodeConfig(max_new_tokens=2),
            rng=random.Random(3),
        ).generate([0], on_token=events.append)

        self.assertEqual([event.source for event in events], ["target_fallback", "target_fallback"])

    def test_prompt_and_eos_ids_are_validated(self) -> None:
        model = TableModel({}, default=(0.5, 0.5))
        decoder = SpeculativeDecoder(model, model)
        with self.assertRaises(ValueError):
            decoder.generate([2])
        with self.assertRaises(TypeError):
            decoder.generate([True])
        with self.assertRaises(ValueError):
            SpeculativeDecoder(model, model, DecodeConfig(eos_token_id=2))


if __name__ == "__main__":
    unittest.main()
