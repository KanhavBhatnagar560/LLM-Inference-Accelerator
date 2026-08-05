"""Streaming events emitted after tokens are committed to the output."""

from dataclasses import dataclass
from typing import Literal


TokenSource = Literal[
    "accepted_draft",
    "target_correction",
    "target_bonus",
    "target_fallback",
]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """Describes one generated token and why it was emitted."""

    token_id: int
    index: int
    source: TokenSource
