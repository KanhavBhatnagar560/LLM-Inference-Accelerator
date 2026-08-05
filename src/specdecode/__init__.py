"""Public API for the speculative-decoding reference engine."""

from .config import DecodeConfig
from .decoder import DecodeResult, DecodeStats, SpeculativeDecoder
from .events import TokenEvent, TokenSource
from .models import (
    CausalLMProbabilityAdapter,
    ProbabilityModel,
    ProposalScoringModel,
    TableModel,
)

__all__ = [
    "DecodeConfig",
    "DecodeResult",
    "DecodeStats",
    "CausalLMProbabilityAdapter",
    "ProbabilityModel",
    "ProposalScoringModel",
    "SpeculativeDecoder",
    "TableModel",
    "TokenEvent",
    "TokenSource",
]
