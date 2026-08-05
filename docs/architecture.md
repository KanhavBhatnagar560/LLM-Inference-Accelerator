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

## Model integration contract

Every model implements `next_token_probs(context)`. Targets may also implement
`score_proposal(prefix, proposal)`, which returns exactly one probability row per
proposal position plus one row for the target bonus token. The decoder selects
this batched path when available and otherwise retains the sequential Stage 1
oracle.

For a prefix of length `L` and `K` proposed tokens, a causal language model is
run on `prefix + proposal`. Logit positions `L-1` through `L+K-1` produce the
`K+1` required distributions. The Hugging Face adapter slices those positions,
converts logits to float32 probabilities, and performs no gradient tracking.

Draft and target tokenizers must have identical token-to-ID and added-token
mappings plus matching BOS, EOS, PAD, and UNK IDs. Equal vocabulary sizes alone
are not sufficient.

Each committed token can emit a `TokenEvent` whose source is accepted draft,
target correction, target bonus, or target fallback. Events are fired only after
the token becomes part of the output.

## Native backend contract

`SpeculativeDecoder` delegates deterministic numerical work through a
`SamplingBackend`. The Python backend is the oracle. The optional C++ backend is
loaded through a versioned C ABI and `ctypes`; it provides categorical sampling,
residual weights, vectorized acceptance probabilities, and first-rejection
verification.

Randomness always stays in Python. The decoder passes explicit uniform draws to
the selected backend and stops drawing after the first rejected proposal. This
preserves seeded draw order across Python and native execution. Backend fallback
is decided before generation and never hides native execution failures.

## Stage boundaries

### Stage 1: Reference engine — complete

- Model protocol expressed as next-token probability distributions.
- Exact accept/reject/correction loop.
- Dynamic proposal window and fallback path.
- Standard-library-only correctness tests.

### Stage 2: Real models — complete

- A target adapter that verifies a proposal in one batched forward pass.
- Lazy PyTorch/Transformers loading with independent device placement.
- Token streaming and CLI configuration.
- Stateless full-context execution; cache reuse remains intentionally deferred.

### Stage 3: Native verification — complete

- Versioned C ABI and contiguous buffers shared through `ctypes`.
- Vectorized acceptance plus native residual and categorical operations.
- C/C++ build tests and seeded parity against the Python oracle.
- Automatic pure-Python fallback when no native library is present.

### Stage 4: KV memory engine

- Fixed-size logical/physical block table.
- Per-head or per-channel INT8 scale metadata.
- Reference dequantization and attention parity thresholds.

### Stage 5: CUDA and measurement

- Separate draft, target, and transfer streams with explicit events.
- Pinned host memory and reusable device workspaces.
- Warmup, throughput, latency-percentile, memory, and accuracy harnesses.
