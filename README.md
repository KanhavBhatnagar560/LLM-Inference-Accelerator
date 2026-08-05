# LLM Inference Accelerator

An engineered C++/Python speculative-decoding pipeline designed to pair a small
draft model with a larger target model while preserving the target model's exact
output distribution.

> **Project status:** Stages 1 through 3 are implemented. The exact Python
> reference engine supports optional Hugging Face models and can route sampling,
> residual construction, and vectorized draft verification through a compiled
> C++17 library. CUDA kernels, INT8 KV-cache quantization, PagedAttention, and GPU
> benchmarks remain planned.

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
- a versioned native C ABI with dependency-free `ctypes` bindings;
- automatic native selection with a pure-Python fallback;
- seeded native/Python parity for tokens, events, statistics, and RNG state;
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

## Build the Stage 3 native core

The pure-Python package still works without a compiler. To enable the C++
backend in a source checkout:

```bash
cmake -S . -B work/native-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build work/native-build --parallel
ctest --test-dir work/native-build --output-on-failure
```

The development loader discovers this build automatically. Backend selection is
also explicit:

```bash
PYTHONPATH=src python3 -m specdecode demo --sampling-backend python
PYTHONPATH=src python3 -m specdecode demo --sampling-backend native
```

For another build location, pass `--native-library /path/to/library` or set
`SPECDECODE_NATIVE_LIBRARY`. `auto` uses native code when available and otherwise
selects Python before generation begins. Native operation errors never trigger a
silent mid-generation fallback. Set `SPECDECODE_SAMPLING_BACKEND=python` to force
the reference path for a complete process or test run.

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Planned stages

1. **Exact Python reference** — complete.
2. **Model integration** — complete; Hugging Face adapters and streaming CLI.
3. **Native core** — complete; C++ verification/sampling with Python bindings.
4. **Memory engine** — paged KV blocks and per-head INT8 quantization.
5. **CUDA pipeline** — asynchronous streams, pinned transfers, kernels, benchmarks.

The target numbers (82 tok/s, 2.4x throughput, 42% lower KV memory, and 42 ms p99
latency) must be reproduced on a documented GPU and model pair before being
reported as achieved results.

See [docs/architecture.md](docs/architecture.md) for the evolving design.
See [docs/native.md](docs/native.md) for the C ABI and backend contract.
