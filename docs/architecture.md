# Architecture

## Correctness contract

Let `q(x | prefix)` be the draft distribution and `p(x | prefix)` be the target
distribution. A drafted token `x` is accepted with probability:

```text
min(1, p(x | prefix) / q(x | prefix))
```

If rejected, the replacement is sampled from the normalized residual:

```text
max(0, p(x | prefix) - q(x | prefix))
```

When every proposal is accepted, one bonus token is sampled from the target.
This construction produces the same output distribution as sampling directly
from the target model. Draft-window adaptation can change performance but not
the probability law, because it only changes how many proposals are attempted.

## Layered design

```text
CLI / benchmark harness
        |
Python orchestration and model adapters
        |
C++ verification, sampling, and cache allocator
        |
CUDA attention, quantization, and stream pipeline
```

The Python implementation in `src/specdecode` is the behavioral oracle. Native
implementations must pass the same seeded scenario tests and statistical tests.

## Stage boundaries

### Stage 1: Reference engine

- Model protocol expressed as next-token probability distributions.
- Exact accept/reject/correction loop.
- Dynamic proposal window and fallback path.
- Standard-library-only correctness tests.

### Stage 2: Real models

- A target adapter that verifies a proposal in one batched forward pass.
- A draft adapter with independent cache state.
- Token streaming and CLI configuration.

### Stage 3: Native verification

- Contiguous probability buffers shared with Python.
- Vectorized acceptance and residual sampling.
- Seeded parity tests against Stage 1.

### Stage 4: KV memory engine

- Fixed-size logical/physical block table.
- Per-head or per-channel INT8 scale metadata.
- Reference dequantization and attention parity thresholds.

### Stage 5: CUDA and measurement

- Separate draft, target, and transfer streams with explicit events.
- Pinned host memory and reusable device workspaces.
- Warmup, throughput, latency-percentile, memory, and accuracy harnesses.

