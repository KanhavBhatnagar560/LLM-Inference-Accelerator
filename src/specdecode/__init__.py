"""Public API for the speculative-decoding reference engine."""

from .backends import (
    PythonSamplingBackend,
    SamplingBackend,
    VerificationResult,
    load_sampling_backend,
)
from .benchmark import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkReport,
    BenchmarkSample,
    DecoderBenchmarkRunner,
    GenerationRunner,
    run_comparison_benchmark,
)
from .config import DecodeConfig
from .cuda import (
    CudaDependencyError,
    CudaExecutionRuntime,
    CudaMemorySnapshot,
    CudaRuntimeConfig,
    CudaRuntimeStats,
    CudaTask,
    CudaUnavailableError,
)
from .decoder import DecodeResult, DecodeStats, SpeculativeDecoder, TargetOnlyDecoder
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
    "BenchmarkConfig",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "BenchmarkSample",
    "CudaDependencyError",
    "CudaExecutionRuntime",
    "CudaMemorySnapshot",
    "CudaRuntimeConfig",
    "CudaRuntimeStats",
    "CudaTask",
    "CudaUnavailableError",
    "DecodeConfig",
    "DecodeResult",
    "DecodeStats",
    "DecoderBenchmarkRunner",
    "DequantizedKVToken",
    "CausalLMProbabilityAdapter",
    "KVCacheCapacityError",
    "KVCacheCheckpoint",
    "KVCacheConfig",
    "KVCacheError",
    "KVCacheStats",
    "KVQuantizer",
    "GenerationRunner",
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
    "TargetOnlyDecoder",
    "TokenEvent",
    "TokenSource",
    "VerificationResult",
    "load_sampling_backend",
    "run_comparison_benchmark",
]
