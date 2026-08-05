import unittest

from specdecode.tokenizers import (
    IncrementalTextDecoder,
    TokenizerCompatibilityError,
    encode_prompt,
    validate_tokenizer_compatibility,
)


class FakeTokenizer:
    def __init__(self, vocab=None, *, eos_token_id=2, encoded=None) -> None:
        self._vocab = vocab or {"<bos>": 0, "hello": 1, "<eos>": 2}
        self._encoded = [0, 1] if encoded is None else encoded
        self.bos_token_id = 0
        self.eos_token_id = eos_token_id
        self.pad_token_id = None
        self.unk_token_id = None
        self.additional_special_tokens_ids = []
        self.chat_calls = 0

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return {"<bos>": 0, "<eos>": 2}

    def encode(self, prompt, add_special_tokens=True):
        return list(self._encoded)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_calls += 1
        return [0, 1]

    def decode(self, token_ids, **kwargs):
        pieces = {0: "", 1: "hello", 2: " world"}
        return "".join(pieces[token] for token in token_ids)


class RevisingTokenizer(FakeTokenizer):
    def decode(self, token_ids, **kwargs):
        states = {
            (1,): "\N{REPLACEMENT CHARACTER}",
            (1, 2): "é",
            (1, 2, 3): "é ",
        }
        return states[tuple(token_ids)]


class TokenizerTests(unittest.TestCase):
    def test_identical_tokenizers_are_compatible(self) -> None:
        validate_tokenizer_compatibility(FakeTokenizer(), FakeTokenizer())

    def test_different_token_ids_are_rejected(self) -> None:
        with self.assertRaises(TokenizerCompatibilityError):
            validate_tokenizer_compatibility(
                FakeTokenizer(),
                FakeTokenizer({"<bos>": 0, "hello": 2, "<eos>": 1}),
            )

    def test_different_special_tokens_are_rejected(self) -> None:
        with self.assertRaises(TokenizerCompatibilityError):
            validate_tokenizer_compatibility(FakeTokenizer(), FakeTokenizer(eos_token_id=1))

    def test_prompt_encoding_supports_plain_and_chat_input(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(encode_prompt(tokenizer, "hello"), (0, 1))
        self.assertEqual(encode_prompt(tokenizer, "hello", chat=True), (0, 1))
        self.assertEqual(tokenizer.chat_calls, 1)

    def test_empty_prompt_uses_bos(self) -> None:
        self.assertEqual(encode_prompt(FakeTokenizer(encoded=[]), ""), (0,))

    def test_incremental_decoder_returns_new_suffix(self) -> None:
        decoder = IncrementalTextDecoder(FakeTokenizer())
        self.assertEqual(decoder.push(1), "")
        self.assertEqual(decoder.push(2), "hello ")
        self.assertEqual(decoder.flush(), "world")

    def test_incremental_decoder_does_not_duplicate_revised_unicode(self) -> None:
        decoder = IncrementalTextDecoder(RevisingTokenizer())
        self.assertEqual(decoder.push(1), "")
        self.assertEqual(decoder.push(2), "")
        self.assertEqual(decoder.push(3), "é ")
        self.assertEqual(decoder.flush(), "")


if __name__ == "__main__":
    unittest.main()
