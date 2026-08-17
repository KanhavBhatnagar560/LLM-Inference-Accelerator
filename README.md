# LLM Inference Accelerator

An engineered C++/Python speculative-decoding pipeline designed to pair a small
draft model with a larger target model while preserving the target model's exact
output distribution.

> **Project status:** Stages 1 through 4 are complete. Stage 5 now has an
> optional PyTorch CUDA execution layer, Hugging Face `past_key_values` reuse,
> and an opt-in bridge that mirrors real model K/V tensors into the custom paged
> INT8 cache. An additional reference mode dequantizes that paged state back
> into Hugging Face tensors and consumes it during attention. A persistent
> packed CUDA storage boundary now exposes INT8 pages, scales, and block tables
> to an unfused torch-native attention path. Fused custom CUDA PagedAttention
> and end-to-end model integration remain pending.

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
- Hugging Face-compatible reconstruction of paged INT8 state;
- opt-in attention forwards that consume the reconstructed reference cache;
- separate CUDA draft, target, and transfer streams with event dependencies;
- reusable device workspaces, pinned host buffers, and nonblocking transfers;
- persistent packed CUDA INT8 pages with incremental synchronization and
  rollback-safe slot clearing;
- direct physical-page gathering, on-device dequantization, grouped-query head
  expansion, causal masking, and CUDA SDPA dispatch;
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
quantization, and follows speculative rollback. By default it remains a shadow
cache so quantization cannot alter logits.

For correctness and integration testing, the dequantized reference cache can be
fed back into Hugging Face attention:

```bash
specdecode generate \
  --draft-model meta-llama/Llama-3.2-1B \
  --target-model meta-llama/Llama-3.1-8B \
  --prompt "Explain paged attention." \
  --paged-cache-mirror \
  --paged-cache-reference-attention
```

This mode reconstructs standard `[1, heads, sequence, head_dim]` tensors in
Python before an incremental forward. It proves that attention can consume the
paged cache's logical state, but it is intentionally slow, retains the standard
Hugging Face output cache, and may perturb logits because the K/V state is INT8.
It is not the custom CUDA PagedAttention implementation or a performance path.

`CudaPagedKVCacheStorage` provides the next kernel-facing boundary. It packs one
logical sequence into persistent CUDA tensors with
`[physical_block, block_offset, layer, head, head_dim]` key/value layout, keeps
the active logical-to-physical block table on device, uploads only a changed
suffix, and clears released slots after rollback. The current implementation
quantizes through the CPU reference cache and is not yet wired into model
attention; it establishes storage and synchronization semantics for the custom
kernel milestone.

`submit_paged_attention()` now consumes that view without reconstructing Python
or Hugging Face cache objects. It gathers logical pages through the device block
table, dequantizes one layer on-device, applies an offset-aware causal mask, and
submits PyTorch scaled-dot-product attention on the target stream. This is an
unfused direct-consumption path: it materializes gathered K/V tensors and is not
yet installed inside Hugging Face model layers.

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
   paged INT8 mirroring, reference attention consumption, and a packed
   device-storage boundary with unfused CUDA SDPA consumption are implemented.

See [docs/architecture.md](docs/architecture.md) for the evolving design.
See [docs/native.md](docs/native.md) for the C ABI and backend contract.
See [docs/kv-cache.md](docs/kv-cache.md) for the Stage 4 memory contract.
