"""Dependency-free tokenizer validation and incremental text decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class TokenizerCompatibilityError(ValueError):
    """Raised when draft and target token IDs do not have identical meaning."""


_SPECIAL_TOKEN_ID_FIELDS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "additional_special_tokens_ids",
)


def _vocabulary(tokenizer: Any) -> dict[str, int]:
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise TokenizerCompatibilityError("tokenizer must provide get_vocab()")
    vocabulary = dict(getter())
    if not vocabulary:
        raise TokenizerCompatibilityError("tokenizer vocabulary cannot be empty")
    return vocabulary


def validate_tokenizer_compatibility(draft_tokenizer: Any, target_tokenizer: Any) -> None:
    """Require identical token-to-ID mappings and generation special tokens."""

    draft_vocab = _vocabulary(draft_tokenizer)
    target_vocab = _vocabulary(target_tokenizer)
    if draft_vocab != target_vocab:
        raise TokenizerCompatibilityError(
            "draft and target tokenizers have different token-to-ID mappings"
        )
    if len(draft_tokenizer) != len(target_tokenizer):
        raise TokenizerCompatibilityError("draft and target tokenizer lengths differ")

    for field in _SPECIAL_TOKEN_ID_FIELDS:
        draft_value = getattr(draft_tokenizer, field, None)
        target_value = getattr(target_tokenizer, field, None)
        if draft_value != target_value:
            raise TokenizerCompatibilityError(f"tokenizer {field} values differ")

    for tokenizer in (draft_tokenizer, target_tokenizer):
        added_getter = getattr(tokenizer, "get_added_vocab", None)
        if callable(added_getter) and not set(added_getter()).issubset(draft_vocab):
            raise TokenizerCompatibilityError("added tokenizer vocabulary is inconsistent")

    draft_added = getattr(draft_tokenizer, "get_added_vocab", lambda: {})()
    target_added = getattr(target_tokenizer, "get_added_vocab", lambda: {})()
    if dict(draft_added) != dict(target_added):
        raise TokenizerCompatibilityError("tokenizer added-token mappings differ")


def encode_prompt(tokenizer: Any, prompt: str, *, chat: bool = False) -> tuple[int, ...]:
    """Encode one prompt, using the tokenizer's chat template when requested."""

    if chat:
        template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(template):
            raise ValueError("this tokenizer does not provide a chat template")
        encoded = template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)

    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], Sequence):
        encoded = encoded[0]

    token_ids = tuple(int(token) for token in encoded)
    if not token_ids:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError("encoded prompt is empty and the tokenizer has no BOS token")
        token_ids = (int(bos_token_id),)
    return token_ids


class IncrementalTextDecoder:
    """Decode token IDs while buffering text that a later token may revise.

    Subword tokenizers can temporarily decode an incomplete byte sequence as a
    replacement character and repair it after another token arrives. Text is
    therefore emitted only through a whitespace boundary; ``flush()`` releases
    the final buffered word after generation ends.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted_text = ""

    def _decode(self) -> str:
        return self._tokenizer.decode(
            self._token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def push(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        decoded = self._decode()
        if not decoded.startswith(self._emitted_text):
            # Already-emitted text cannot be retracted from a terminal. Keeping
            # the existing prefix avoids duplicating a revised decode.
            return ""

        stable_end = len(self._emitted_text)
        pending_text = decoded[len(self._emitted_text) :]
        for index, character in enumerate(pending_text, len(self._emitted_text)):
            if character.isspace():
                stable_end = index + 1
        suffix = decoded[len(self._emitted_text) : stable_end]
        self._emitted_text = decoded[:stable_end]
        return suffix

    def flush(self) -> str:
        """Release text buffered after the final stable boundary."""

        decoded = self._decode()
        if not decoded.startswith(self._emitted_text):
            return ""
        suffix = decoded[len(self._emitted_text) :]
        self._emitted_text = decoded
        return suffix
