"""A tiny, dependency-free executable demonstration."""

from __future__ import annotations

import random

from .config import DecodeConfig
from .decoder import SpeculativeDecoder
from .models import TableModel


def main() -> None:
    # Token 3 is EOS. The draft is deliberately imperfect so both acceptance
    # and correction behavior can appear with different seeds.
    draft = TableModel(
        {(0,): (0.05, 0.70, 0.20, 0.05), (1,): (0.05, 0.20, 0.65, 0.10)},
        default=(0.05, 0.10, 0.35, 0.50),
    )
    target = TableModel(
        {(0,): (0.05, 0.55, 0.30, 0.10), (1,): (0.05, 0.15, 0.60, 0.20)},
        default=(0.05, 0.10, 0.25, 0.60),
    )
    decoder = SpeculativeDecoder(
        draft,
        target,
        DecodeConfig(max_new_tokens=12, eos_token_id=3),
        rng=random.Random(7),
    )
    result = decoder.generate([0])
    print("generated token ids:", list(result.generated_tokens))
    print(
        "draft acceptance:",
        f"{result.stats.accepted_tokens}/{result.stats.drafted_tokens}",
        f"({result.stats.acceptance_rate:.1%})",
    )
    print("target verification rounds:", result.stats.verification_rounds)


if __name__ == "__main__":
    main()

