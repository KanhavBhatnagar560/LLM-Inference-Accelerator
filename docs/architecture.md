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
Python orchestration, model adapters, and paged cache oracle
        |
C++ verification, sampling, and INT8 quantization
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

With cache reuse enabled, the first call prefills the prompt and retains the
returned `past_key_values`. Later calls find the longest shared prefix, crop a
speculatively cached suffix when necessary, and forward only uncached tokens.
The next-token distribution at the cache boundary is retained so target proposal
verification can reuse it without replaying the final prefix token. Decoders
reset cache-aware models before every generation request.

Draft and target tokenizers must have identical token-to-ID and added-token
mappings plus matching BOS, EOS, PAD, and UNK IDs. Equal vocabulary sizes alone
are not sufficient.

Each committed token can emit a `TokenEvent` whose source is accepted draft,
target correction, target bonus, target fallback, or direct target-only output.
Events are fired only after the token becomes part of the output.

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

## KV memory contract

`PagedKVCache` owns a fixed physical block pool and one logical block table per
sequence. Multi-token append is atomic, checkpoints record a committed length,
and rollback clears rejected suffix entries before reclaiming unused blocks.

Each key and value head is quantized independently with a symmetric INT8 scale.
`PythonKVQuantizer` is the numerical oracle and `NativeKVQuantizer` exposes the
matching C++ CPU kernels. The dequantization error bound and format-level byte
formulas are documented in `docs/kv-cache.md`. Hugging Face model state can be
mirrored into this layout. A slow reference path can reconstruct standard cache
tensors from it for attention, but CUDA attention does not consume it directly.

## CUDA execution contract

`CudaExecutionRuntime` is optional and imports PyTorch lazily. One runtime owns
separate draft, target, and transfer streams for one CUDA device. Submitted work
returns a `CudaTask` containing a completion event; downstream streams wait on
that event instead of synchronizing the whole device.

Pinned host buffers and device workspaces grow geometrically and are reused by
logical name and dtype. Input copies use the transfer stream and model forwards
wait on their transfer events. Timing events and NVTX ranges expose operation
boundaries to profilers. Full-device synchronization is reserved for explicit
benchmark boundaries.

`CudaPagedKVCacheStorage` owns persistent INT8 K/V and float32 scale workspaces
in physical-page order plus an active device block table. Synchronization reads
the dependency-free cache oracle, finds the stable logical prefix, uploads only
the changed suffix on the transfer stream, and zeroes physical slots released
by rollback. The resulting `CudaPagedKVCacheView` is a kernel-facing contract;
no model attention kernel consumes it yet.

The Hugging Face integration reuses standard `past_key_values` and sends only
uncached suffix tokens after prefill. Probability rows still return to the CPU
for exact sampling. `--no-kv-cache` retains stateless full-context execution.

With `--paged-cache-mirror`, each new Hugging Face K/V tensor suffix is also
converted from `[batch, heads, sequence, head_dim]` into per-token entries in
`PagedKVCache`. Only unseen tokens are quantized and appended. Cache cropping,
full refills, and request resets apply the matching truncate/reset operation to
the mirror. The exact Hugging Face cache remains the attention source, ensuring
the lossy INT8 shadow cannot change logits or sampling behavior.

With both `--paged-cache-mirror` and
`--paged-cache-reference-attention`, the adapter instead dequantizes the retained
paged prefix into `[1, heads, sequence, head_dim]` tensors before each
incremental forward. It preserves compatible modern cache container types and
then mirrors the model's returned native cache for the next step. This opt-in
reference validates the consumption contract but does not remove the native
output cache, bypass Python materialization, or preserve exact logits after INT8
quantization.

## Benchmark contract

`TargetOnlyDecoder` provides a direct baseline using the same target model,
sampling backend, output limit, EOS rule, and seeded RNG contract. The comparison
harness warms both paths, gives them identical prompt/seed pairs, alternates run
order, synchronizes around timed samples, and emits a versioned JSON report.

Reports include environment and model settings, throughput, p50/p95/p99 token
latency, allocator peaks, and hashed input/output token sequences. They exclude
raw prompt content. Shared-process allocator peaks are useful for steady-state
comparison but are not isolated total model-footprint measurements.

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
- Stateless full-context reference path remains available.

### Stage 3: Native verification — complete

- Versioned C ABI and contiguous buffers shared through `ctypes`.
- Vectorized acceptance plus native residual and categorical operations.
- C/C++ build tests and seeded parity against the Python oracle.
- Automatic pure-Python fallback when no native library is present.

### Stage 4: KV memory engine — complete

- Fixed-size logical/physical block tables with deterministic reclamation.
- Atomic speculative appends, checkpoints, suffix rollback, and invariants.
- Per-head symmetric INT8 scale metadata and Python/native CPU kernels.
- Reference dequantization, error-bound tests, and memory accounting.

### Stage 5: CUDA and measurement — in progress

- Implemented: separate draft, target, and transfer streams with explicit events.
- Implemented: pinned host memory and reusable device workspaces.
- Implemented: timing events, NVTX ranges, allocator snapshots, and environment
  metadata.
- Implemented: fair target-only/speculative warmup, throughput, latency, memory,
  and JSON reporting harness.
- Implemented: Hugging Face `past_key_values` prefill/reuse, speculative suffix
  cropping, correction replay, and request-boundary resets.
- Implemented: opt-in model-state conversion into the Stage 4 paged INT8 layout,
  including incremental append and speculative rollback synchronization.
- Implemented: opt-in reference attention consumption by reconstructing standard
  Hugging Face cache tensors from paged INT8 state.
- Implemented: persistent packed CUDA K/V and scale storage, device block-table
  transfer, changed-suffix upload, and rollback-safe slot clearing.
- Pending: direct paged INT8 attention consumption and device-resident
  sampling/verification without Python materialization.
- Pending: custom CUDA PagedAttention and INT8 cache kernels.
- Pending: benchmark and quality evidence from documented NVIDIA hardware.
