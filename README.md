# LLM Inference Accelerator

An engineered C++/Python speculative-decoding pipeline designed to pair a small
draft model with a larger target model while preserving the target model's exact
output distribution.

> **Project status:** Stages 1 and 2 are implemented. The exact Python reference
> engine now supports optional Hugging Face causal models, one-pass proposal
> verification, tokenizer validation, and streaming output. C++/CUDA kernels,
> INT8 KV-cache quantization, PagedAttention, and GPU benchmarks remain planned.

## Why start with a reference engine?

Speculative decoding is only useful if acceleration does not change model output
quality. The reference implementation is deliberately small and auditable. It is
the correctness oracle that every later native or CUDA optimization must match.

Implemented now:

- exact rejection sampling using the `max(0, p - q)` correction distribution;
- adaptive draft-window sizing based on observed acceptance;
- safe target-only fallback when the draft model fails;
- EOS and maximum-length handling;
- one-forward-pass target scoring through an optional proposal interface;
- lazy PyTorch/Transformers model loading and exact tokenizer compatibility checks;
- token-level streaming events and a real-model command-line interface;
- deterministic unit tests and an empirical distribution-equivalence test;
- zero required third-party Python dependencies.

## Run the dependency-free demo

```bash
PYTHONPATH=src python3 -m specdecode demo
```

## Run real Hugging Face models

Install the optional backend:

```bash
python3 -m pip install -e '.[transformers]'
```

Then provide a draft and target model that use the exact same tokenizer:

```bash
specdecode generate \
  --draft-model meta-llama/Llama-3.2-1B \
  --target-model meta-llama/Llama-3.1-8B \
  --prompt "Explain speculative decoding in one paragraph." \
  --dtype bfloat16
```

The Meta Llama repositories are gated and require accepted access plus Hugging
Face authentication. Model loading fails early if token-to-ID mappings, special
tokens, model vocabulary sizes, or embedding ranges are incompatible. Use
`--chat` for instruction-tuned checkpoints with a chat template.

Stage 2 intentionally performs stateless full-context forwards. It proves real
model integration and batches target proposal verification, but KV-cache reuse
is deferred to the memory-engine stages.

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Planned stages

1. **Exact Python reference** — complete.
2. **Model integration** — complete; Hugging Face adapters and streaming CLI.
3. **Native core** — C++ batched verification and sampling with Python bindings.
4. **Memory engine** — paged KV blocks and per-head INT8 quantization.
5. **CUDA pipeline** — asynchronous streams, pinned transfers, kernels, benchmarks.

The target numbers (82 tok/s, 2.4x throughput, 42% lower KV memory, and 42 ms p99
latency) must be reproduced on a documented GPU and model pair before being
reported as achieved results.

See [docs/architecture.md](docs/architecture.md) for the evolving design.
