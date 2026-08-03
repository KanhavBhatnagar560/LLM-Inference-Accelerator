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

The current baseline is Stage 1, introduced by commit `fed5890`:

```text
feat: add exact speculative decoding reference
```

Stage 1 is a dependency-free Python correctness implementation. It does not yet
load real language models and does not contain C++ or CUDA kernels.

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
- a small executable toy-model demonstration.

Not implemented yet:

- Hugging Face or PyTorch model adapters;
- tokenizer or text input/output support;
- Llama-8B or a 1B draft-model integration;
- streaming generation callbacks or an HTTP server;
- C++, Python bindings, CUDA, PagedAttention, or INT8 KV-cache code;
- GPU benchmark and profiling infrastructure;
- validated performance numbers on NVIDIA hardware.

Do not describe a planned feature or target metric as completed.

## Repository map

```text
.
├── AGENTS.md                    contributor and agent operating guide
├── README.md                    user-facing overview and status
├── docs/
│   └── architecture.md          design layers and stage boundaries
├── pyproject.toml               Python package metadata and optional dependencies
├── src/specdecode/
│   ├── __init__.py              public Python API
│   ├── __main__.py              dependency-free demo
│   ├── config.py                immutable decoding configuration
│   ├── decoder.py               exact speculative-decoding loop
│   ├── models.py                model protocol and table-backed test model
│   └── sampling.py              probability and residual-sampling utilities
└── tests/
    ├── test_decoder.py          decoder correctness and distribution tests
    └── test_sampling.py         sampling utility tests
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

The Stage 1 target evaluations are sequential for readability. A real tensor
adapter should compute the proposal-position logits in one batched target-model
forward pass without changing the decoder's mathematical contract.

## Public Python API

The root package exports:

- `DecodeConfig` — generation limit, draft-window bounds, dynamic-window flag,
  and optional EOS token ID;
- `SpeculativeDecoder` — exact reference generator;
- `DecodeResult` — prompt tokens, generated tokens, and statistics;
- `DecodeStats` — drafted, accepted, rejected, target-sampled, verification, and
  fallback counters;
- `ProbabilityModel` — minimal next-token probability protocol;
- `TableModel` — deterministic context table used by tests and the demo.

The `ProbabilityModel` contract currently requires:

```python
vocab_size: int
next_token_probs(token_ids: Sequence[int]) -> Sequence[float]
```

Future adapters may add a batched sequence-scoring method, but they must preserve
this simple reference path for testing.

## Development commands

The current stage requires Python 3.10 or newer and no third-party runtime
packages.

Run the demo:

```bash
PYTHONPATH=src python3 -m specdecode
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

## Test coverage in Stage 1

The nine current tests cover:

- valid normalization;
- rejection of empty, negative, zero-mass, and NaN distributions;
- residual correction probabilities;
- seeded categorical sampling;
- full acceptance and EOS termination for identical models;
- forced rejection and residual replacement;
- fallback after a simulated draft-model failure;
- strict maximum generation length;
- empirical agreement with a target distribution over 20,000 samples.

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

### Stage 2 — real model integration

Add optional PyTorch/Transformers dependencies, Hugging Face causal-LM adapters,
shared-tokenizer validation, batched target verification, streaming token events,
and a configurable CLI. Keep toy-model tests runnable without these dependencies.

### Stage 3 — C++ native core

Add CMake-based C++ code and Python bindings for vectorized acceptance, residual
construction, and sampling. Add seeded parity tests against the Python oracle and
retain a pure-Python fallback when the extension is unavailable.

### Stage 4 — paged and quantized KV cache

Implement logical-to-physical KV block tables, block allocation/reclamation,
INT8 quantization metadata and kernels, reference dequantization, memory accounting,
and numerical parity tests. Quantization scales and error tolerances must be
documented.

### Stage 5 — CUDA execution and benchmarking

Introduce reusable device workspaces, pinned host buffers, asynchronous CUDA
streams/events, and profiling ranges. Benchmark target-only and speculative paths
under identical sampling settings, prompts, warmup, batch sizes, and hardware.

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

