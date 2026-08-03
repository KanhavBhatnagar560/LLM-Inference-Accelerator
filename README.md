# LLM Inference Accelerator

An engineered C++/Python speculative-decoding pipeline designed to pair a small
draft model with a larger target model while preserving the target model's exact
output distribution.

> **Project status:** Stage 1 implements and tests the exact reference algorithm
> in dependency-free Python. C++/CUDA kernels, real Hugging Face model adapters,
> INT8 KV-cache quantization, PagedAttention, and GPU benchmarks are planned
> stages—not measured claims yet.

## Why start with a reference engine?

Speculative decoding is only useful if acceleration does not change model output
quality. The reference implementation is deliberately small and auditable. It is
the correctness oracle that every later native or CUDA optimization must match.

Implemented now:

- exact rejection sampling using the `max(0, p - q)` correction distribution;
- adaptive draft-window sizing based on observed acceptance;
- safe target-only fallback when the draft model fails;
- EOS and maximum-length handling;
- deterministic unit tests and an empirical distribution-equivalence test;
- zero required third-party Python dependencies.

## Run the Stage 1 demo

```bash
PYTHONPATH=src python3 -m specdecode
```

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Planned stages

1. **Exact Python reference** — complete in this commit.
2. **Model integration** — Hugging Face adapters, Llama-family support, streaming CLI.
3. **Native core** — C++ batched verification and sampling with Python bindings.
4. **Memory engine** — paged KV blocks and per-head INT8 quantization.
5. **CUDA pipeline** — asynchronous streams, pinned transfers, kernels, benchmarks.

The target numbers (82 tok/s, 2.4x throughput, 42% lower KV memory, and 42 ms p99
latency) must be reproduced on a documented GPU and model pair before being
reported as achieved results.

See [docs/architecture.md](docs/architecture.md) for the evolving design.

