"""Public API for the speculative-decoding reference engine."""

from .backends import (
    PythonSamplingBackend,
    SamplingBackend,
    VerificationResult,
    load_sampling_backend,
)
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
    "PythonSamplingBackend",
    "ProbabilityModel",
    "ProposalScoringModel",
    "SpeculativeDecoder",
    "SamplingBackend",
    "TableModel",
    "TokenEvent",
    "TokenSource",
    "VerificationResult",
    "load_sampling_backend",
]
