"""Small model protocol and deterministic models used by tests and examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProbabilityModel(Protocol):
    """Minimal interface needed by the correctness reference.

    Stage 2 adapters will implement this protocol for real causal language
    models and override sequence scoring with batched tensor execution.
    """

    @property
    def vocab_size(self) -> int: ...

    def next_token_probs(self, token_ids: Sequence[int]) -> Sequence[float]: ...


class TableModel:
    """A context-to-distribution table with a configurable default.

    Contexts are matched exactly first, then by their final token. This makes
    the model convenient for deterministic examples without external packages.
    """

    def __init__(
        self,
        table: Mapping[tuple[int, ...], Sequence[float]],
        default: Sequence[float],
    ) -> None:
        self._table = {tuple(context): tuple(probs) for context, probs in table.items()}
        self._default = tuple(default)
        self._vocab_size = len(self._default)
        if self._vocab_size == 0:
            raise ValueError("default distribution cannot be empty")
        if any(len(probs) != self._vocab_size for probs in self._table.values()):
            raise ValueError("every table distribution must use the same vocabulary size")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def next_token_probs(self, token_ids: Sequence[int]) -> Sequence[float]:
        context = tuple(token_ids)
        if context in self._table:
            return self._table[context]
        if context and (context[-1],) in self._table:
            return self._table[(context[-1],)]
        return self._default

