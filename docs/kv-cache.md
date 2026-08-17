# Paged INT8 KV cache

Stage 4 adds a dependency-free correctness implementation of the memory layout
that a later CUDA PagedAttention path will consume. It includes deterministic
physical-block allocation, logical block tables, speculative-suffix rollback,
per-head symmetric INT8 quantization, native CPU quantization kernels, and
format-level memory accounting.

Real-model decoding reuses the standard Hugging Face `past_key_values`
representation. An opt-in Stage 5 mirror now converts those exact model tensors
into this custom paged INT8 layout and follows cache rollback. A separate opt-in
reference mode reconstructs standard tensors from the paged representation and
feeds them into Hugging Face attention. Stage 5 still owns direct CUDA
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

## Reference attention consumption

`materialize_legacy_cache()` reads the logical sequence through its block table,
dequantizes every per-head vector, and returns one key/value tensor pair per
layer in `[1, heads, sequence, head_dim]` layout. When given an existing cache as
a template, it preserves each tensor's device and dtype. `materialize_like()`
also reconstructs modern cache containers that expose `from_legacy_cache()`.

Generation and benchmarking can enable this path with both
`--paged-cache-mirror` and `--paged-cache-reference-attention`. Before an
incremental forward, the adapter crops the native and paged states together,
materializes the retained paged prefix, and supplies that reconstructed cache to
the model. The returned native cache is mirrored again for the next step.

This establishes an auditable attention-consumption reference without changing
the default exact-cache path. It does not reduce live model memory, it performs
dequantization and tensor construction in Python, and INT8 reconstruction can
change logits. Custom CUDA kernels must be evaluated against this path's layout
and against the exact Hugging Face baseline.

## Packed CUDA storage boundary

`CudaPagedKVCacheStorage` mirrors one `PagedKVCache` sequence into persistent
CUDA tensors. Keys and values use:

```text
[physical_block, block_offset, layer, head, head_dim]
```

The float32 scale tensors omit `head_dim`, and the active device block table maps
logical blocks to the physical-block dimension. Each synchronization compares
the physical slot and quantized value at every retained position, preserves the
longest stable prefix, uploads only the changed suffix through pinned transfer
buffers, and clears slots that stopped being live. Comparing values as well as
slot IDs handles rollback followed by reuse before the next synchronization.

One storage object binds to one logical sequence so a later kernel cannot
silently consume another request's pages. This boundary does not quantize on the
GPU and is not connected to Hugging Face model layers yet; those are separate
Stage 5 milestones.

`submit_paged_attention()` accepts a query in
`[1, query_heads, query_tokens, head_dim]` layout for the final cached positions.
It follows the device block table instead of assuming contiguous physical pages,
selects one model layer, dequantizes INT8 K/V using the per-head scales, repeats
K/V heads for grouped-query attention, and dispatches PyTorch SDPA with an
explicit causal mask. The operation waits on cache synchronization through a
CUDA event and runs on the target stream.

This path consumes packed cache storage directly, but it is deliberately
unfused: page gathering and dequantization materialize temporary tensors before
SDPA. A fused kernel should replace those operations only after NVIDIA profiling
shows the interface is correct and the materialization cost matters.

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
