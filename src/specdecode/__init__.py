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
from .kv_cache import (
    DequantizedKVToken,
    KVCacheCapacityError,
    KVCacheCheckpoint,
    KVCacheConfig,
    KVCacheError,
    KVCacheStats,
    KVQuantizer,
    PagedKVCache,
    PythonKVQuantizer,
    QuantizedKVToken,
    QuantizedVector,
)
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
    "DequantizedKVToken",
    "CausalLMProbabilityAdapter",
    "KVCacheCapacityError",
    "KVCacheCheckpoint",
    "KVCacheConfig",
    "KVCacheError",
    "KVCacheStats",
    "KVQuantizer",
    "PagedKVCache",
    "PythonSamplingBackend",
    "PythonKVQuantizer",
    "QuantizedKVToken",
    "QuantizedVector",
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
