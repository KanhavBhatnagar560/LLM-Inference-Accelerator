"""Small model protocol and deterministic models used by tests and examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from abc import ABC, abstractmethod
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


@runtime_checkable
class ProposalScoringModel(ProbabilityModel, Protocol):
    """Optional fast path for scoring a complete proposal in one call.

    The result contains ``len(proposal) + 1`` probability rows. Row ``i``
    predicts the token after ``prefix + proposal[:i]``; the final row is the
    target bonus-token distribution.
    """

    def score_proposal(
        self,
        prefix: Sequence[int],
        proposal: Sequence[int],
    ) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class CacheAwareProbabilityModel(ProbabilityModel, Protocol):
    """Optional stateful model whose cache is reset at generation boundaries."""

    def reset_cache(self) -> None: ...


class CausalLMProbabilityAdapter(ABC):
    """Turns causal-LM logit positions into the probability model protocol.

    Backends implement one method that evaluates a token sequence and returns
    probabilities only for requested logit positions. Keeping position
    selection here makes the off-by-one-sensitive proposal alignment testable
    without importing a tensor framework.
    """

    def __init__(self, vocab_size: int) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @abstractmethod
    def _probabilities_at_positions(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
    ) -> Sequence[Sequence[float]]:
        """Run the backend once and return rows for zero-based logit positions."""

    def next_token_probs(self, token_ids: Sequence[int]) -> Sequence[float]:
        if not token_ids:
            raise ValueError("a causal language model requires a non-empty context")
        rows = self._probabilities_at_positions(token_ids, (len(token_ids) - 1,))
        if len(rows) != 1:
            raise ValueError("backend must return one probability row for one position")
        return rows[0]

    def score_proposal(
        self,
        prefix: Sequence[int],
        proposal: Sequence[int],
    ) -> Sequence[Sequence[float]]:
        if not prefix:
            raise ValueError("a causal language model requires a non-empty prefix")
        combined = tuple(prefix) + tuple(proposal)
        # A causal LM's logits at position n predict the token at position n+1.
        # Therefore L-1 verifies the first proposal token and L+K-1 predicts
        # the bonus after K proposals.
        positions = tuple(range(len(prefix) - 1, len(combined)))
        rows = self._probabilities_at_positions(combined, positions)
        if len(rows) != len(proposal) + 1:
            raise ValueError("backend returned the wrong number of proposal rows")
        return rows


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
