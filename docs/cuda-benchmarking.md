# CUDA execution and benchmarking

Stage 5 begins with an optional PyTorch CUDA execution runtime and a reproducible
target-only versus speculative benchmark harness. The dependency-free package
still imports and runs without PyTorch, while CUDA behavior is covered by fake
backend tests on non-NVIDIA development machines.

This milestone does not include a custom CUDA PagedAttention kernel, CUDA INT8
KV-cache kernel, or measured NVIDIA results. The Hugging Face adapter reuses its
standard `past_key_values`; it does not consume the custom paged INT8 cache.

## CUDA runtime

`CudaExecutionRuntime` owns one device's reusable resources:

- separate draft, target, and host-to-device transfer streams;
- explicit CUDA events for cross-stream dependencies;
- optional timing events;
- geometrically grown reusable device workspaces;
- geometrically grown pinned host buffers;
- nonblocking pinned-to-device copies;
- NVTX ranges for draft, target, and transfer operations;
- current and peak PyTorch allocator memory snapshots;
- GPU, driver, CUDA, PyTorch, Python, and platform metadata.

`CudaTask` contains an eagerly launched operation's result and completion event.
A task can be passed through `wait_for` to make another stream wait on that event
without synchronizing the complete device. `wait()` synchronizes only the task's
completion event; `CudaExecutionRuntime.synchronize()` is reserved for benchmark
boundaries.

Buffers are keyed by logical name and dtype. Capacity grows to the next power of
two, allowing increasing token sequences to reuse allocations between growth
points. Callers must not reuse one logical buffer name concurrently before its
dependent operation finishes. The Hugging Face adapter satisfies this rule by
waiting for each forward before beginning the next decoding step.

When CUDA execution is enabled for a `HuggingFaceModelPair`, draft and target
models must use the same CUDA device. Input IDs and attention masks are copied
through pinned buffers on the transfer stream. Model forwards run on their named
draft or target streams after waiting for both transfer events.

## Benchmark command

Install the real-model dependencies and build the optional native library, then
run on an NVIDIA CUDA machine:

```bash
specdecode benchmark \
  --draft-model meta-llama/Llama-3.2-1B \
  --draft-revision DRAFT_COMMIT \
  --target-model meta-llama/Llama-3.1-8B \
  --target-revision TARGET_COMMIT \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt "Explain speculative decoding." \
  --prompt "Write a short CUDA optimization checklist." \
  --warmup-runs 2 \
  --measured-runs 10 \
  --output outputs/benchmark.json
```

The command tokenizes and loads models before timing. For every measured prompt,
the target-only and speculative paths receive the same seed and decode settings.
Their execution order alternates to reduce systematic first/second-path bias.
Every sample synchronizes CUDA immediately before and after measurement and
resets peak allocator statistics at its start.

The target-only baseline uses `TargetOnlyDecoder`, avoiding the exception-driven
draft-fallback path. Both paths use the same target adapter, sampling backend,
temperature, maximum output length, prompts, device, and dtype.

## Report contents

The versioned JSON report contains:

- warmup count, measured repetitions, and base seed;
- model IDs and revisions, dtype, device, batch size, and draft-window settings;
- whether model loading and tokenization are excluded;
- the selected Hugging Face KV-cache mode;
- Python, platform, PyTorch, Transformers, CUDA, driver, GPU, and compute
  capability metadata where available;
- total generated tokens, elapsed time, throughput, and throughput ratio;
- mean, p50, p95, and p99 token-commit latency;
- peak allocated and reserved PyTorch CUDA memory;
- per-sample input/output lengths, elapsed time, latency values, and SHA-256
  hashes of token IDs.

Raw prompt and generated token IDs are intentionally excluded from the report.
Hashes make run identity auditable without copying prompt contents into benchmark
artifacts.

## Measurement limitations

The two paths share already-loaded draft and target model objects and persistent
runtime buffers. Peak memory therefore represents steady-state execution within
that shared process, not isolated total model footprint. Model loading,
tokenization, and Hugging Face weight download time are excluded.

The current adapter copies probability rows back to the CPU for exact Python
sampling, so this is not yet a fully device-resident decoder. Custom CUDA
PagedAttention, INT8 cache consumption, batched serving, and isolated
process-level memory comparisons remain future work. Use `--no-kv-cache` when a
model exposes a cache representation that cannot be cropped safely.

Do not report the project's target throughput, latency, memory, or batch-size
numbers as achieved until a checked-in JSON report from documented NVIDIA
hardware reproduces them.
