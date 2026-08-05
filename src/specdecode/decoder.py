"""Exact speculative sampling reference implementation."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

from .backends import PythonSamplingBackend, SamplingBackend
from .config import DecodeConfig
from .events import TokenEvent, TokenSource
from .models import ProbabilityModel
from .sampling import (
    DistributionError,
    normalize_probabilities,
)


@dataclass(slots=True)
class DecodeStats:
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    target_tokens: int = 0
    verification_rounds: int = 0
    fallback_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens


@dataclass(frozen=True, slots=True)
class DecodeResult:
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    stats: DecodeStats

    @property
    def all_tokens(self) -> tuple[int, ...]:
        return self.prompt_tokens + self.generated_tokens


class _DraftWindow:
    def __init__(self, config: DecodeConfig) -> None:
        self._config = config
        self.size = config.initial_draft_tokens

    def observe(self, accepted: int, proposed: int) -> None:
        if not self._config.dynamic_draft or proposed == 0:
            return
        if accepted == proposed:
            self.size = min(self._config.max_draft_tokens, self.size + 1)
        elif accepted * 2 < proposed:
            self.size = max(self._config.min_draft_tokens, self.size // 2)
        else:
            self.size = max(self._config.min_draft_tokens, self.size - 1)


class SpeculativeDecoder:
    """Generate tokens with exact speculative sampling.

    Target proposal scoring and numerical sampling both have optional optimized
    backends while retaining dependency-free Python reference paths.
    """

    def __init__(
        self,
        draft_model: ProbabilityModel,
        target_model: ProbabilityModel,
        config: DecodeConfig | None = None,
        *,
        rng: random.Random | None = None,
        sampling_backend: SamplingBackend | None = None,
    ) -> None:
        if draft_model.vocab_size != target_model.vocab_size:
            raise ValueError("draft and target models must share a vocabulary")
        if draft_model.vocab_size < 1:
            raise ValueError("model vocabulary must be positive")
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or DecodeConfig()
        if self.config.eos_token_id is not None and not (
            0 <= self.config.eos_token_id < target_model.vocab_size
        ):
            raise ValueError("eos_token_id must be inside the shared vocabulary")
        self.rng = rng or random.Random()
        self.sampling_backend = (
            sampling_backend
            if sampling_backend is not None
            else PythonSamplingBackend()
        )

    def _probabilities(self, model: ProbabilityModel, context: list[int]) -> tuple[float, ...]:
        return normalize_probabilities(
            model.next_token_probs(context), expected_size=model.vocab_size
        )

    def _target_token(self, context: list[int]) -> int:
        probabilities = self._probabilities(self.target_model, context)
        return self._sample(probabilities)

    def _sample(self, probabilities: Sequence[float]) -> int:
        return self.sampling_backend.categorical(probabilities, self.rng.random())

    def _target_distributions(
        self,
        prefix: list[int],
        proposed: list[int],
    ) -> list[tuple[float, ...]]:
        scorer = getattr(self.target_model, "score_proposal", None)
        if callable(scorer):
            rows = scorer(prefix, proposed)
        else:
            rows = [
                self.target_model.next_token_probs(prefix + proposed[:index])
                for index in range(len(proposed) + 1)
            ]
        if len(rows) != len(proposed) + 1:
            raise DistributionError(
                "proposal scorer must return one row per proposal plus one bonus row"
            )
        return [
            normalize_probabilities(row, expected_size=self.target_model.vocab_size)
            for row in rows
        ]

    def _validate_prompt(self, prompt_tokens: Sequence[int]) -> tuple[int, ...]:
        prompt: list[int] = []
        for token in prompt_tokens:
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise TypeError("prompt token IDs must be integers")
            token_id = int(token)
            if not 0 <= token_id < self.target_model.vocab_size:
                raise ValueError("prompt token ID is outside the shared vocabulary")
            prompt.append(token_id)
        return tuple(prompt)

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        on_token: Callable[[TokenEvent], None] | None = None,
    ) -> DecodeResult:
        config = self.config
        prompt = self._validate_prompt(prompt_tokens)
        generated: list[int] = []
        stats = DecodeStats()
        window = _DraftWindow(config)

        def emit(token: int, source: TokenSource) -> None:
            generated.append(token)
            if on_token is not None:
                on_token(TokenEvent(token_id=token, index=len(generated) - 1, source=source))

        if config.max_new_tokens == 0:
            return DecodeResult(prompt, (), stats)

        while len(generated) < config.max_new_tokens:
            prefix = list(prompt) + generated
            remaining = config.max_new_tokens - len(generated)
            proposal_limit = min(window.size, remaining)
            proposed: list[int] = []
            draft_distributions: list[tuple[float, ...]] = []

            # A draft failure is recoverable. Already-created proposals remain
            # valid and can still be verified; a failure before the first
            # proposal becomes one ordinary target-only decoding step.
            for _ in range(proposal_limit):
                try:
                    q = self._probabilities(self.draft_model, prefix + proposed)
                except (DistributionError, RuntimeError, ValueError):
                    break
                token = self._sample(q)
                draft_distributions.append(q)
                proposed.append(token)
                stats.drafted_tokens += 1
                if token == config.eos_token_id:
                    break

            if not proposed:
                token = self._target_token(prefix)
                emit(token, "target_fallback")
                stats.target_tokens += 1
                stats.fallback_steps += 1
                if token == config.eos_token_id:
                    break
                continue

            target_distributions = self._target_distributions(prefix, proposed)
            stats.verification_rounds += 1

            acceptance_probabilities = self.sampling_backend.acceptance_probabilities(
                target_distributions[: len(proposed)],
                draft_distributions,
                proposed,
            )
            if len(acceptance_probabilities) != len(proposed):
                raise DistributionError(
                    "sampling backend returned the wrong number of acceptance probabilities"
                )

            accepted_this_round = 0
            rejected = False
            for index, token in enumerate(proposed):
                p = target_distributions[index]
                q = draft_distributions[index]

                if self.rng.random() < acceptance_probabilities[index]:
                    emit(token, "accepted_draft")
                    stats.accepted_tokens += 1
                    accepted_this_round += 1
                    if token == config.eos_token_id or len(generated) == config.max_new_tokens:
                        break
                    continue

                correction_weights = self.sampling_backend.residual_weights(p, q)
                correction = normalize_probabilities(
                    correction_weights,
                    expected_size=self.target_model.vocab_size,
                )
                replacement = self._sample(correction)
                emit(replacement, "target_correction")
                stats.rejected_tokens += 1
                stats.target_tokens += 1
                rejected = True
                break

            window.observe(accepted_this_round, len(proposed))

            if generated and generated[-1] == config.eos_token_id:
                break
            if len(generated) == config.max_new_tokens:
                break
            if rejected:
                continue

            # Every draft token was accepted, so emit one target bonus token.
            bonus = self._sample(target_distributions[-1])
            emit(bonus, "target_bonus")
            stats.target_tokens += 1
            if bonus == config.eos_token_id:
                break

        return DecodeResult(prompt, tuple(generated), stats)
