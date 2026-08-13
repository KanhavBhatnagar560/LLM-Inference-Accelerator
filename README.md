# LLM Inference Accelerator

An engineered C++/Python speculative-decoding pipeline designed to pair a small
draft model with a larger target model while preserving the target model's exact
output distribution.

> **Project status:** Stages 1 through 4 are complete. Stage 5 now has an
> optional PyTorch CUDA execution layer, Hugging Face `past_key_values` reuse,
> and an opt-in bridge that mirrors real model K/V tensors into the custom paged
> INT8 cache. The model still uses its exact native cache for attention.

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
- request-local Hugging Face KV-cache reuse with speculative suffix cropping;
- a versioned native C ABI with dependency-free `ctypes` bindings;
- automatic native selection with a pure-Python fallback;
- seeded native/Python parity for tokens, events, statistics, and RNG state;
- deterministic logical-to-physical KV block allocation and reclamation;
- atomic speculative cache appends with checkpoint rollback;
- per-head symmetric INT8 KV quantization with Python/C++ parity;
- dequantization error-bound tests;
- separate CUDA draft, target, and transfer streams with event dependencies;
- reusable device workspaces, pinned host buffers, and nonblocking transfers;
- optional NVTX profiling ranges;
- deterministic unit tests and distribution-equivalence coverage;
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

Real-model generation uses Hugging Face `past_key_values` by default. The first
forward prefills the prompt; later calls score only uncached suffix tokens. After
a speculative rejection, the adapter crops to the shared prefix and replays the
correction. Each generation resets request-local state. Pass `--no-kv-cache` to
retain the stateless full-context reference path.

## Build the Stage 3/4 native core

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

## Stage 4 cache boundary

`PagedKVCache` provides fixed-pool physical blocks, per-sequence block tables,
atomic multi-token appends, suffix rollback, and INT8 K/V storage.
`PythonKVQuantizer` is the dependency-free oracle;
`NativeKVQuantizer` runs the matching C++ CPU kernels.

Real models can mirror their native cache into `PagedKVCache` by adding
`--paged-cache-mirror`. The bridge converts Hugging Face tensors from
`[batch, heads, sequence, head_dim]` into per-token paged entries, applies INT8
quantization, and follows speculative rollback. It intentionally remains a
shadow cache so quantization cannot alter logits. CUDA PagedAttention will later
consume the same layout directly.

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Planned stages

1. **Exact Python reference** — complete.
2. **Model integration** — complete; Hugging Face adapters and streaming CLI.
3. **Native core** — complete; C++ verification/sampling with Python bindings.
4. **Memory engine** — complete; paged KV blocks and per-head INT8 quantization.
5. **CUDA pipeline** — in progress; execution/profiling, Hugging Face cache reuse,
   and paged INT8 shadow-cache integration are implemented.

See [docs/architecture.md](docs/architecture.md) for the evolving design.
See [docs/native.md](docs/native.md) for the C ABI and backend contract.
See [docs/kv-cache.md](docs/kv-cache.md) for the Stage 4 memory contract.
