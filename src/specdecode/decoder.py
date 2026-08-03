"""Exact speculative sampling reference implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import DecodeConfig
from .models import ProbabilityModel
from .sampling import (
    DistributionError,
    normalize_probabilities,
    residual_distribution,
    sample_categorical,
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

    This implementation prioritizes clarity and correctness. Target scoring is
    sequential here; Stage 2 will replace it with a single batched model pass.
    """

    def __init__(
        self,
        draft_model: ProbabilityModel,
        target_model: ProbabilityModel,
        config: DecodeConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        if draft_model.vocab_size != target_model.vocab_size:
            raise ValueError("draft and target models must share a vocabulary")
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or DecodeConfig()
        self.rng = rng or random.Random()

    def _probabilities(self, model: ProbabilityModel, context: list[int]) -> tuple[float, ...]:
        return normalize_probabilities(
            model.next_token_probs(context), expected_size=model.vocab_size
        )

    def _target_token(self, context: list[int]) -> int:
        probabilities = self._probabilities(self.target_model, context)
        return sample_categorical(probabilities, self.rng)

    def generate(self, prompt_tokens: list[int] | tuple[int, ...]) -> DecodeResult:
        config = self.config
        prompt = tuple(prompt_tokens)
        generated: list[int] = []
        stats = DecodeStats()
        window = _DraftWindow(config)

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
                token = sample_categorical(q, self.rng)
                draft_distributions.append(q)
                proposed.append(token)
                stats.drafted_tokens += 1
                if token == config.eos_token_id:
                    break

            if not proposed:
                token = self._target_token(prefix)
                generated.append(token)
                stats.target_tokens += 1
                stats.fallback_steps += 1
                if token == config.eos_token_id:
                    break
                continue

            # The reference evaluates each prefix independently. A tensor model
            # adapter can batch these contexts without changing this contract.
            target_distributions = [
                self._probabilities(self.target_model, prefix + proposed[:index])
                for index in range(len(proposed) + 1)
            ]
            stats.verification_rounds += 1

            accepted_this_round = 0
            rejected = False
            for index, token in enumerate(proposed):
                p = target_distributions[index]
                q = draft_distributions[index]
                q_token = q[token]
                acceptance = 1.0 if q_token == 0.0 else min(1.0, p[token] / q_token)

                if self.rng.random() <= acceptance:
                    generated.append(token)
                    stats.accepted_tokens += 1
                    accepted_this_round += 1
                    if token == config.eos_token_id or len(generated) == config.max_new_tokens:
                        break
                    continue

                correction = residual_distribution(p, q)
                replacement = sample_categorical(correction, self.rng)
                generated.append(replacement)
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
            bonus = sample_categorical(target_distributions[-1], self.rng)
            generated.append(bonus)
            stats.target_tokens += 1
            if bonus == config.eos_token_id:
                break

        return DecodeResult(prompt, tuple(generated), stats)

