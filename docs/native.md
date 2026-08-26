# Native numerical core

Stages 3 and 4 provide a dependency-free C++17 shared library and lazy Python
`ctypes` bindings. It accelerates small numerical operations around speculative
decoding and supplies CPU INT8 quantization kernels; it does not replace model
execution or establish an end-to-end speedup claim.

## Build and test

```bash
cmake -S . -B work/native-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build work/native-build --parallel
ctest --test-dir work/native-build --output-on-failure
```

The CMake suite builds:

- `specdecode_native`, the shared library;
- a dependency-free C++ numerical test executable;
- a C executable proving the public header and symbols are C-compatible.

Build outputs remain ignored by Git. The Python loader checks an explicit path,
`SPECDECODE_NATIVE_LIBRARY`, the documented development build directory, a
packaged library location, and finally the operating system library search path.
`SPECDECODE_SAMPLING_BACKEND` can set the process-wide mode.

## Backend selection

`load_sampling_backend()` supports three modes:

- `python` always uses the correctness oracle;
- `auto` uses native code when a library is discoverable and otherwise selects
  Python before decoding starts;
- `native` requires a compatible library and returns an actionable error when it
  cannot be loaded.

Fallback only covers an absent library during selection. An explicit path, ABI
mismatch, invalid native input, or execution error is propagated rather than
silently changing backends during a generation.

## Stable C ABI

The public header is `native/include/specdecode/native.h`. ABI revision 1.1
exposes:

- probability normalization;
- categorical sampling from an explicit uniform draw;
- residual-weight construction;
- vectorized acceptance probabilities over proposal rows;
- first-rejection detection from explicit uniforms;
- symmetric INT8 quantization and dequantization;
- stable status codes and status messages.

The functions accept caller-owned buffers and never return allocated memory. All
inputs are validated before outputs are written, C++ exceptions are contained at
the ABI boundary, and compiler fast-math transformations are not enabled.

## Randomness and exactness

The C++ library has no random-number generator. `SpeculativeDecoder` draws every
uniform value from its Python RNG and passes that value to the selected backend.
The draw order remains:

1. one categorical draw per drafted token;
2. one acceptance draw for each proposal actually reached;
3. one correction draw only after rejection;
4. one bonus draw only after full acceptance;
5. one fallback draw only when drafting produced no proposal.

Acceptance probabilities may be calculated for the full proposal in one native
call because this deterministic operation consumes no randomness. This design
allows seeded Python/native tests to compare generated tokens, stream events,
statistics, and final RNG state exactly.

Floating-point vector outputs use tight tolerance comparisons. Draws used for
decision parity intentionally avoid numerically ambiguous boundaries, except for
explicit tests such as zero acceptance with a zero uniform.

## Platform wheels

The setuptools build compiles `native/src/native.cpp` as an optional C++17
extension and places the shared library inside the `specdecode` package. The
native loader discovers CPython-tagged `.so`/`.pyd` files there before checking
development build directories or the operating-system search path.

The extension is marked optional: installations without a compatible compiler
still succeed and use the Python backend. Release wheels must be built and tested
on each supported target platform; one platform wheel must never be relabeled for
another operating system or architecture.

See [kv-cache.md](kv-cache.md) for the quantization formula, error bound, paged
allocator, and memory-accounting contract.
