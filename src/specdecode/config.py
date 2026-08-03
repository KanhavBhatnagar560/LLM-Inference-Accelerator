"""Configuration for speculative decoding."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    """Controls one generation run.

    The adaptive window affects efficiency only; it does not alter the exact
    sampling distribution.
    """

    max_new_tokens: int = 32
    initial_draft_tokens: int = 4
    min_draft_tokens: int = 1
    max_draft_tokens: int = 8
    dynamic_draft: bool = True
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.min_draft_tokens < 1:
            raise ValueError("min_draft_tokens must be at least 1")
        if self.max_draft_tokens < self.min_draft_tokens:
            raise ValueError("max_draft_tokens must be >= min_draft_tokens")
        if not self.min_draft_tokens <= self.initial_draft_tokens <= self.max_draft_tokens:
            raise ValueError("initial_draft_tokens must be inside the configured range")

