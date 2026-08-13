# Paged INT8 KV cache

Stage 4 adds a dependency-free correctness implementation of the memory layout
that a later CUDA PagedAttention path will consume. It includes deterministic
physical-block allocation, logical block tables, speculative-suffix rollback,
per-head symmetric INT8 quantization, native CPU quantization kernels, and
format-level memory accounting.

Real-model decoding reuses the standard Hugging Face `past_key_values`
representation. An opt-in Stage 5 mirror now converts those exact model tensors
into this custom paged INT8 layout and follows cache rollback. The model does not
consume the quantized shadow during attention yet. Stage 5 still owns CUDA
PagedAttention kernels and measured performance.

## Hugging Face shadow integration

`HuggingFacePagedCacheMirror` derives layer, KV-head, head-dimension, and capacity
geometry from a decoder-only model configuration. After each cached forward it:

1. accepts legacy cache tuples or objects exposing `to_legacy_cache()`;
2. validates batch-one `[batch, heads, sequence, head_dim]` tensors;
3. extracts only token positions not already present in the mirror;
4. atomically quantizes and appends those positions to `PagedKVCache`;
5. validates allocator invariants after every mutation.

When speculative verification discards a suffix, the Hugging Face adapter crops
its native cache and truncates the mirror to the same logical token count. A new
request resets both. The mirror is deliberately opt-in with
`--paged-cache-mirror` because its Python tensor conversion is a correctness
bridge, not an optimized inference path.

## Tensor and page layout

One cache token contains keys and values in this logical shape:

```text
[key_or_value][layer][attention_head][head_dimension]
```

`KVCacheConfig` fixes the model shape, tokens per block, and number of physical
blocks. Each sequence owns a logical block table whose entries identify physical
blocks in the fixed pool. Token index `t` maps to:

```text
logical_block = t // block_size
block_offset  = t % block_size
physical      = block_table[logical_block]
```

The reference allocator uses the lowest available physical block ID, making
allocation and reclamation deterministic in tests. Physical blocks are not
shared between sequences in Stage 4; prefix sharing and copy-on-write are future
serving optimizations.

## Speculative append and rollback

`append_many()` validates capacity and quantizes every pending token before it
changes allocator state. A malformed tensor or insufficient block pool therefore
leaves the sequence unchanged.

Before appending a speculative suffix, callers can record `checkpoint()`. If a
target-model verification rejects part of that suffix, `rollback()` truncates to
the checkpoint, clears discarded slots, and returns every now-unused physical
block to the pool. `validate_invariants()` audits block ownership, occupancy, and
free-list completeness.

## Symmetric INT8 quantization

Every key head and value head receives an independent scale. For a finite vector
`x`, the Python oracle and C++ kernel calculate:

```text
scale = float32(max(abs(x)) / 127)
q_i   = clamp(round_half_away_from_zero(x_i / scale), -127, 127)
x'_i  = q_i * scale
```

An all-zero vector uses `scale = 0` and all-zero quantized values. The value
`-128` is intentionally unused, keeping the representation symmetric. For each
finite input value, round-to-nearest quantization has the documented absolute
error bound:

```text
abs(x_i - x'_i) <= scale / 2
```

Tests apply a small floating-point allowance when checking this bound. NaN,
infinity, empty vectors, negative scales, and nonzero data paired with a zero
scale are rejected. The native ABI validates all inputs before writing outputs.

## Memory accounting

The accounting describes the compact storage format, not Python object overhead,
CUDA allocator behavior, or measured GPU memory. Per token:

```text
FP32 bytes = 2 * layers * heads * head_dim * 4
INT8 bytes = 2 * layers * heads * head_dim
scale bytes = 2 * layers * heads * 4
compact bytes = INT8 bytes + scale bytes
```

`KVCacheStats` reports used tokens, allocated and free blocks, used/allocated/
capacity bytes, FP32-equivalent bytes, block-table bytes, and the theoretical
format savings ratio. Allocated bytes include unused slots in partially occupied
blocks, while used bytes count committed token entries only.

For a typical head dimension of 128, the representation itself is about 74%
smaller than FP32 before allocator effects. That is a mathematical layout ratio,
not the project's planned 42% measured end-to-end GPU reduction.

## Native ABI

ABI revision 1.1 adds:

- `sd_quantize_symmetric_int8_f64`;
- `sd_dequantize_symmetric_int8_f64`;
- `SD_STATUS_INVALID_SCALE`.

`NativeKVQuantizer` exposes these CPU kernels to `PagedKVCache`. The injectable
`KVQuantizer` protocol keeps `PythonKVQuantizer` as the default oracle and allows
the cache allocator tests to run without a compiler or third-party dependency.
