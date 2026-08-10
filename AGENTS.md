# AGENTS.md — LLM Inference Accelerator

This file is the operating guide for contributors and coding agents working in
this repository. It documents the current implementation, the correctness
contract, the commands used to verify changes, and the staged project plan.

## Project goal

Build a production-style C++/Python speculative-decoding pipeline that pairs a
small draft language model (approximately 1B parameters) with a Llama-family 8B
target model. The completed system is intended to include:

- exact speculative sampling with dynamic proposal lengths;
- safe fallback to ordinary target-model decoding;
- native C++ verification and sampling;
- PagedAttention-style KV-cache block management;
- INT8 KV-cache quantization;
- asynchronous CUDA execution and optimized memory transfers;
- reproducible throughput, latency, memory, and quality benchmarks.

Correctness comes before optimization. Every optimized implementation must match
the Python reference engine's behavior and output distribution.

## Current repository state

Stages 1 through 4 are complete. The Stage 5 execution and benchmark foundation
is implemented, while custom CUDA attention/quantization kernels and hardware
measurements remain pending. The dependency-free Python engine remains the
oracle, the Hugging Face adapter connects real causal models, a paged cache
reference manages quantized K/V entries, and an optional C++17 shared library
handles sampling, verification, and INT8 quantization.

Implemented:

- categorical probability validation and normalization;
- exact draft-token acceptance testing;
- residual-distribution sampling after rejection;
- a target-model bonus token when all proposals are accepted;
- dynamic draft-window growth and shrinkage;
- fallback target decoding when the draft model fails;
- EOS and maximum-new-token handling;
- seeded deterministic tests;
- a 20,000-sample empirical distribution-equivalence test;
- a small executable toy-model demonstration;
- an optional one-forward-pass target proposal-scoring interface;
- a lazy `AutoModelForCausalLM` adapter using inference mode and float32 softmax;
- exact draft/target tokenizer and model-vocabulary validation;
- prompt and optional chat-template encoding;
- token callbacks identifying accepted, corrected, bonus, and fallback output;
- a configurable real-model CLI with independent model devices and revisions;
- a CMake-based native library with a stable, versioned C ABI;
- native normalization, categorical sampling, residual construction, vectorized
  acceptance probabilities, and first-rejection detection;
- lazy `ctypes` bindings with `auto`, `python`, and `native` selection modes;
- exact RNG ownership in Python and seeded Python/native parity tests;
- dependency-free C++ tests and a plain-C public-header test;
- fixed-pool logical-to-physical KV block tables;
- deterministic block allocation, reclamation, and allocator invariants;
- atomic multi-token cache appends and speculative-suffix rollback;
- per-head symmetric INT8 quantization with Python and C++ implementations;
- reference dequantization with documented error bounds;
- used, allocated, capacity, FP32-equivalent, and block-table memory accounting;
- a direct target-only decoder for benchmark baselines;
- lazy optional PyTorch CUDA execution without changing base dependencies;
- dedicated draft, target, and transfer streams with explicit event dependencies;
- geometric device-workspace and pinned-host-buffer reuse;
- nonblocking host-to-device copies, timing events, and NVTX profiling ranges;
- CUDA allocator snapshots and reproducible environment metadata;
- identical-seed target/speculative warmup and measured comparisons;
- versioned JSON throughput, latency-percentile, memory, and token-hash reports.

Not implemented yet:

- an HTTP or production serving layer;
- Hugging Face `past_key_values` reuse and cache integration;
- custom CUDA PagedAttention and INT8 quantization kernels;
- device-resident verification and sampling;
- shared-prefix cache blocks and copy-on-write serving optimizations;
- platform-specific native wheel production;
- isolated process-level model-memory and maximum-batch measurements;
- validated performance numbers on NVIDIA hardware.

Do not describe a planned feature or target metric as completed.

## Repository map

```text
.
├── AGENTS.md                    contributor and agent operating guide
├── CMakeLists.txt               root native build entry point
├── MANIFEST.in                  native sources included in source distributions
├── README.md                    user-facing overview and status
├── docs/
│   ├── architecture.md          design layers and stage boundaries
│   ├── cuda-benchmarking.md     Stage 5 runtime and measurement contract
│   ├── kv-cache.md              paging, quantization, and memory contract
│   └── native.md                C ABI, loader, RNG, and native-kernel contract
├── native/
│   ├── include/specdecode/      stable public C ABI
│   ├── src/                     C++17 implementation
│   └── tests/                   dependency-free C and C++ tests
├── pyproject.toml               Python package metadata and optional dependencies
├── src/specdecode/
│   ├── __init__.py              public Python API
│   ├── __main__.py              dependency-free demo
│   ├── backends.py              backend protocol and Python oracle
│   ├── benchmark.py             fair comparison harness and JSON reports
│   ├── cli.py                   toy and real-model command-line interface
│   ├── config.py                immutable decoding configuration
│   ├── cuda.py                  optional streams, events, buffers, profiling
│   ├── decoder.py               exact speculative-decoding loop
│   ├── events.py                typed streaming token events
│   ├── huggingface.py           optional PyTorch/Transformers adapter
│   ├── kv_cache.py              paged INT8 KV-cache correctness engine
│   ├── models.py                sequential and batched model protocols
│   ├── native.py                lazy ctypes loader and native backend
│   ├── sampling.py              probability and residual-sampling utilities
│   └── tokenizers.py            compatibility, prompt, and streaming helpers
└── tests/
    ├── test_batched_scoring.py  causal-logit alignment and dispatch tests
    ├── test_benchmark.py        fairness, metrics, and report tests
    ├── test_cuda.py             dependency-free fake-CUDA runtime tests
    ├── test_decoder.py          decoder correctness and distribution tests
    ├── test_huggingface.py      dependency-free adapter component test
    ├── test_kv_cache.py         paging, rollback, quantization, accounting tests
    ├── test_streaming.py        event and prompt validation tests
    └── ...                      CLI, tokenizer, and sampling tests
```

`work/` and `outputs/` are local workspace folders and are intentionally ignored
by Git. Generated model files, build products, and benchmark results must also
remain untracked unless a later stage explicitly adds a small fixture.

## Correctness contract

For a prefix `c`, let:

- `q(x | c)` be the draft model probability for token `x`;
- `p(x | c)` be the target model probability for token `x`.

A token sampled from `q` is accepted with probability:

```text
min(1, p(x | c) / q(x | c))
```

On rejection, its replacement must be sampled from:

```text
normalize(max(0, p(. | c) - q(. | c)))
```

If all proposed tokens are accepted and generation has room, one bonus token is
sampled from the target model at the final proposed prefix. This accept/reject
and correction procedure is what preserves the target model's output
distribution.

The following are non-negotiable invariants:

1. Draft and target models use the same vocabulary and token IDs.
2. Every probability vector is finite, non-negative, non-empty, and normalized
   before sampling.
3. A rejected draft token is never emitted; the residual replacement is emitted
   at that position.
4. Generation never emits more than `max_new_tokens`.
5. Generation stops after emitting the configured EOS token.
6. Draft failure may reduce performance but must not prevent target-only output.
7. Dynamic draft sizing may affect work performed, never the output probability
   law.
8. Native and CUDA paths must retain a reference fallback and parity tests.

## Current decoding flow

`SpeculativeDecoder.generate()` performs these steps:

1. Build a prefix from the prompt and tokens already generated.
2. Sample up to the current draft-window size from the draft model.
3. Evaluate target distributions for each proposed position and the bonus
   position.
4. Accept proposals in order using the probability ratio above.
5. At the first rejection, sample one residual correction and start a new round.
6. If all proposals pass, sample one target bonus token.
7. Adjust the next draft-window size using the round's acceptance result.
8. Stop on EOS or the maximum generation length.

The sequential Stage 1 path remains available. A target implementing
`score_proposal()` evaluates all proposal positions and the bonus position in one
call. Hugging Face execution is currently stateless and uses the full context.
When explicitly enabled, the adapter stages inputs through reusable pinned and
device buffers, then runs forwards on named CUDA streams after transfer events.
The standalone Stage 4 cache can roll back rejected proposal suffixes, but model
cache integration and custom CUDA kernels remain pending Stage 5 work.

Numerical sampling routes through `SamplingBackend`. Python owns every random
draw and passes explicit uniforms into either implementation. Native acceptance
probabilities can be computed for an entire proposal without consuming RNG. A
backend is selected before generation; native runtime errors must propagate. The
library API defaults to the stable Python oracle, while the CLI's `auto` mode
opts into native discovery.

## Public Python API

The root package exports:

- `DecodeConfig` — generation limit, draft-window bounds, dynamic-window flag,
  and optional EOS token ID;
- `SpeculativeDecoder` — exact reference generator;
- `TargetOnlyDecoder` — direct target baseline using the same sampling contract;
- `DecodeResult` — prompt tokens, generated tokens, and statistics;
- `DecodeStats` — drafted, accepted, rejected, target-sampled, verification, and
  fallback counters;
- `ProbabilityModel` — minimal next-token probability protocol;
- `ProposalScoringModel` — optional one-call target proposal protocol;
- `CausalLMProbabilityAdapter` — framework-neutral causal-logit alignment base;
- `TableModel` — deterministic context table used by tests and the demo;
- `TokenEvent` — one committed token's ID, output index, and source;
- `SamplingBackend` — deterministic sampling and acceptance operation protocol;
- `PythonSamplingBackend` — dependency-free correctness oracle;
- `load_sampling_backend` — explicit or automatic backend selection;
- `KVCacheConfig` — model shape and physical block-pool configuration;
- `PagedKVCache` — logical block tables, quantized storage, and rollback;
- `KVCacheCheckpoint` and `KVCacheStats` — rollback and accounting records;
- `KVQuantizer` and `PythonKVQuantizer` — injectable quantization contract and
  dependency-free oracle;
- `QuantizedVector`, `QuantizedKVToken`, and `DequantizedKVToken` — immutable
  cache data records;
- `CudaRuntimeConfig`, `CudaExecutionRuntime`, `CudaTask`, `CudaRuntimeStats`, and
  `CudaMemorySnapshot` — lazy optional CUDA scheduling and profiling interfaces;
- `BenchmarkConfig`, `BenchmarkSample`, `BenchmarkMetrics`, and `BenchmarkReport`
  — reproducible comparison configuration and results;
- `DecoderBenchmarkRunner` and `run_comparison_benchmark` — adapters and the fair
  target-only/speculative measurement harness.

The `ProbabilityModel` contract currently requires:

```python
vocab_size: int
next_token_probs(token_ids: Sequence[int]) -> Sequence[float]
```

Targets may additionally implement:

```python
score_proposal(prefix, proposal) -> Sequence[Sequence[float]]
```

It returns `len(proposal) + 1` rows and must preserve the simple reference path.

## Development commands

The current stage requires Python 3.10 or newer and no third-party runtime
packages.

Run the demo:

```bash
PYTHONPATH=src python3 -m specdecode demo
```

Install and run real-model support:

```bash
python3 -m pip install -e '.[transformers]'
specdecode generate --draft-model DRAFT --target-model TARGET --prompt "Hello"
```

Run an instrumented CUDA comparison on NVIDIA hardware:

```bash
specdecode benchmark \
  --draft-model DRAFT \
  --target-model TARGET \
  --device cuda:0 \
  --prompt "Hello" \
  --output outputs/benchmark.json
```

Build and test the native core:

```bash
cmake -S . -B work/native-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build work/native-build --parallel
ctest --test-dir work/native-build --output-on-failure
```

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Check that all Python files compile:

```bash
python3 -m compileall -q src tests
```

Check patch formatting before committing:

```bash
git diff --check
```

When optional development dependencies are installed, `pytest` and `ruff` may
also be used, but standard-library tests must continue to work.

## Test coverage through the Stage 5 foundation

The 74 current Python tests plus two CTest executables cover:

- valid normalization;
- rejection of empty, negative, zero-mass, and NaN distributions;
- residual correction probabilities;
- seeded categorical sampling;
- full acceptance and EOS termination for identical models;
- forced rejection and residual replacement;
- fallback after a simulated draft-model failure;
- strict maximum generation length;
- empirical agreement with a target distribution over 20,000 samples;
- causal-logit position alignment and single-call proposal scoring;
- malformed batched target output rejection;
- streaming sources for acceptance, correction, bonus, and fallback;
- prompt and EOS vocabulary validation;
- exact tokenizer compatibility, chat encoding, and incremental text decoding;
- lazy CLI parsing and adapter execution with dependency-free fake components;
- automatic fallback and explicit native-loader failures;
- deterministic RNG draw order owned by the decoder;
- native/Python categorical, residual, acceptance, event, statistics, and RNG
  state parity;
- ABI validation errors, strict acceptance comparison, and public C linkage;
- logical-to-physical KV block mapping and deterministic block reuse;
- atomic capacity/shape failures and speculative checkpoint rollback;
- per-head INT8 scale metadata and dequantization error bounds;
- Python/native quantization, dequantization, and paged-cache parity;
- used-versus-allocated compact-format memory accounting;
- a direct target-only baseline with streaming and EOS behavior;
- dependency-free fake-CUDA stream, event, timing, NVTX, and buffer-pool tests;
- pinned nonblocking transfers and Hugging Face stream integration;
- fair benchmark seeds, alternating order, latency/throughput aggregation,
  allocator peaks, validation, and JSON report contents.

Any change to sampling, verification, scheduling, or fallback behavior must add
or update tests. Prefer deterministic unit cases for branches and a bounded
statistical test for distribution-level behavior.

## Coding rules

- Support Python 3.10+ unless the project explicitly raises the minimum version.
- Keep the Python reference understandable rather than micro-optimized.
- Use type hints on public interfaces and dataclasses for structured results.
- Validate tensor/probability dimensions at component boundaries.
- Make random-number generators explicit and seedable in tests.
- Do not silently replace target-model failures with invented output.
- Avoid mandatory heavy dependencies in the reference package.
- Keep model-specific code in adapters rather than the core sampling algorithm.
- Keep generated artifacts, downloaded weights, and benchmark output out of Git.
- Update `README.md`, `docs/architecture.md`, and this file when a stage changes
  project status or developer commands.

## Staged implementation plan

### Stage 1 — exact Python reference: complete

The current repository provides the behavioral oracle and test suite.

### Stage 2 — real model integration: complete

Optional PyTorch/Transformers loading, tokenizer validation, batched target
verification, streaming events, and the configurable CLI are implemented. Real
GPU measurements are not part of this stage.

### Stage 3 — C++ native core: complete

The CMake C++17 library, versioned C ABI, `ctypes` bindings, vectorized
acceptance, native sampling/residual operations, compiled tests, seeded parity,
and pure-Python fallback are implemented. Native wheel production is deferred.

### Stage 4 — paged and quantized KV cache: complete

The dependency-free paged cache implements logical-to-physical block tables,
deterministic allocation/reclamation, atomic append and suffix rollback, per-head
INT8 metadata, Python and native CPU quantization kernels, reference
dequantization, format-level memory accounting, and numerical parity tests.
Hugging Face model cache integration and custom CUDA kernels remain Stage 5 work.

### Stage 5 — CUDA execution and benchmarking: in progress

Implemented: reusable device workspaces, pinned host buffers, separate CUDA
streams with explicit events, timing/NVTX profiling, allocator/environment
reporting, a direct target-only baseline, and a fair versioned JSON benchmark.

Pending: Hugging Face `past_key_values` reuse, integration with the Stage 4 cache,
device-resident verification/sampling, custom CUDA PagedAttention and INT8
quantization kernels, isolated memory/batch testing, and measurements on
documented NVIDIA hardware.

## Benchmark reporting rules

The numbers in the project idea are goals, not current results:

- 2.4x generation throughput;
- 82 generated tokens per second;
- 42% lower GPU memory use;
- 4x larger supported batches;
- p99 per-token latency reduced from 115 ms to 42 ms;
- no measurable accuracy or distributional regression across 1,000+ prompts.

Before reporting any number as achieved, record:

- GPU model, driver, CUDA, compiler, PyTorch, and Transformers versions;
- exact draft and target model revisions and quantization settings;
- prompt dataset, input/output lengths, batch size, and sampling parameters;
- warmup and measured iteration counts;
- whether tokenization and model loading are excluded from timing;
- baseline and optimized measurements from the same environment;
- peak allocated and reserved GPU memory;
- median, p95, and p99 latency plus total throughput;
- the correctness or quality comparison method.

Never copy target metrics into a resume or README as measured results until the
benchmark artifacts reproduce them.

## Commit and review policy

Work in small, independently verifiable stage commits. Before each commit:

1. run the relevant tests;
2. run Python compilation or the native build as applicable;
3. run `git diff --check`;
4. inspect the final diff for generated files, model weights, or benchmark noise;
5. update documentation when behavior, commands, or status changed.

Use descriptive commit subjects such as:

```text
feat: add transformers model adapter
feat: add native verification kernel
test: validate int8 kv-cache error bounds
docs: document stage 1 repository contract
```

Do not combine an unrelated refactor with a milestone implementation. Do not
rewrite or discard contributor changes outside the active task.

## Definition of done for the complete project

The project is complete only when:

- a documented 1B/8B model pair runs end to end;
- speculative output passes distribution and quality checks against target-only
  decoding;
- C++/CUDA acceleration has tested reference fallbacks;
- paged INT8 KV-cache memory savings are measured rather than estimated;
- streaming output and fallback behavior work under failure scenarios;
- benchmark scripts reproduce throughput, memory, batch-size, and latency claims;
- setup and benchmark instructions work from a clean checkout;
- all tests, builds, and documentation checks pass.
