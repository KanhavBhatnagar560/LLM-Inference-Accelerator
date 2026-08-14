"""Hugging Face KV-state mirror backed by the paged INT8 reference cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kv_cache import KVCacheConfig, KVCacheStats, KVQuantizer, PagedKVCache


class HuggingFacePagedCacheError(RuntimeError):
    """Raised when model cache tensors cannot be mirrored safely."""


@dataclass(frozen=True, slots=True)
class HuggingFacePagedCacheConfig:
    """Paged-cache geometry derived from a decoder-only model configuration."""

    block_size: int = 16
    num_blocks: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        if self.num_blocks is not None and (
            isinstance(self.num_blocks, bool)
            or not isinstance(self.num_blocks, int)
            or self.num_blocks < 1
        ):
            raise ValueError("num_blocks must be a positive integer when provided")


@dataclass(frozen=True, slots=True)
class HuggingFacePagedCacheMirrorStats:
    """Mirror activity plus the underlying paged-cache accounting."""

    synchronized_tokens: int
    rollback_tokens: int
    resets: int
    materializations: int
    paged_cache: KVCacheStats


def _positive_config_int(model_config: Any, *names: str) -> int:
    for name in names:
        value = getattr(model_config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    joined = " or ".join(names)
    raise HuggingFacePagedCacheError(f"model config must define a positive {joined}")


class HuggingFacePagedCacheMirror:
    """Quantize Hugging Face K/V tensors into :class:`PagedKVCache`.

    This is a correctness and layout bridge. The model's original cache remains
    the source for attention, so mirroring cannot perturb logits. A future CUDA
    PagedAttention implementation can consume the same logical representation.
    """

    _SEQUENCE_ID = "model"

    def __init__(
        self,
        cache: PagedKVCache,
    ) -> None:
        self.cache = cache
        self.cache.create_sequence(self._SEQUENCE_ID)
        self._synchronized_tokens = 0
        self._rollback_tokens = 0
        self._resets = 0
        self._materializations = 0

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        config: HuggingFacePagedCacheConfig | None = None,
        *,
        quantizer: KVQuantizer | None = None,
    ) -> "HuggingFacePagedCacheMirror":
        mirror_config = config or HuggingFacePagedCacheConfig()
        num_layers = _positive_config_int(
            model_config,
            "num_hidden_layers",
            "n_layer",
        )
        num_heads = _positive_config_int(
            model_config,
            "num_key_value_heads",
            "num_attention_heads",
            "n_head",
        )
        head_dim = getattr(model_config, "head_dim", None)
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1:
            hidden_size = _positive_config_int(model_config, "hidden_size", "n_embd")
            attention_heads = _positive_config_int(
                model_config,
                "num_attention_heads",
                "n_head",
            )
            if hidden_size % attention_heads:
                raise HuggingFacePagedCacheError(
                    "model hidden size must be divisible by its attention-head count"
                )
            head_dim = hidden_size // attention_heads

        num_blocks = mirror_config.num_blocks
        if num_blocks is None:
            maximum_tokens = _positive_config_int(
                model_config,
                "max_position_embeddings",
                "n_positions",
            )
            num_blocks = (
                maximum_tokens + mirror_config.block_size - 1
            ) // mirror_config.block_size
        cache_config = KVCacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=mirror_config.block_size,
            num_blocks=num_blocks,
        )
        return cls(PagedKVCache(cache_config, quantizer=quantizer))

    @property
    def token_count(self) -> int:
        return self.cache.sequence_length(self._SEQUENCE_ID)

    @property
    def block_table(self) -> tuple[int, ...]:
        return self.cache.block_table(self._SEQUENCE_ID)

    @property
    def stats(self) -> HuggingFacePagedCacheMirrorStats:
        return HuggingFacePagedCacheMirrorStats(
            synchronized_tokens=self._synchronized_tokens,
            rollback_tokens=self._rollback_tokens,
            resets=self._resets,
            materializations=self._materializations,
            paged_cache=self.cache.stats(),
        )

    @staticmethod
    def _legacy_layers(past_key_values: Any) -> tuple[Any, ...]:
        convert = getattr(past_key_values, "to_legacy_cache", None)
        if callable(convert):
            past_key_values = convert()
        if not isinstance(past_key_values, (tuple, list)):
            raise HuggingFacePagedCacheError(
                "past_key_values must be a legacy sequence or support to_legacy_cache()"
            )
        layers = tuple(past_key_values)
        if not layers:
            raise HuggingFacePagedCacheError("past_key_values cannot be empty")
        return layers

    def _validate_layers(self, layers: tuple[Any, ...]) -> int:
        config = self.cache.config
        if len(layers) != config.num_layers:
            raise HuggingFacePagedCacheError(
                "past_key_values layer count does not match the paged cache"
            )
        sequence_length: int | None = None
        for layer in layers:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise HuggingFacePagedCacheError(
                    "each past_key_values layer must contain key and value tensors"
                )
            for tensor in layer[:2]:
                shape = tuple(int(dimension) for dimension in tensor.shape)
                if len(shape) != 4:
                    raise HuggingFacePagedCacheError(
                        "KV tensors must use [batch, heads, sequence, head_dim] layout"
                    )
                if shape[0] != 1:
                    raise HuggingFacePagedCacheError(
                        "the paged cache mirror currently supports batch size one"
                    )
                if shape[1] != config.num_heads or shape[3] != config.head_dim:
                    raise HuggingFacePagedCacheError(
                        "KV tensor head geometry does not match the paged cache"
                    )
                if sequence_length is None:
                    sequence_length = shape[2]
                elif sequence_length != shape[2]:
                    raise HuggingFacePagedCacheError(
                        "every KV tensor must have the same sequence length"
                    )
        if sequence_length is None:
            raise HuggingFacePagedCacheError("past_key_values has no KV tensors")
        return sequence_length

    @staticmethod
    def _head_rows(tensor: Any, token_index: int) -> tuple[tuple[float, ...], ...]:
        selected = tensor[0, :, token_index, :].detach().float().cpu().tolist()
        return tuple(tuple(float(value) for value in head) for head in selected)

    def synchronize(self, past_key_values: Any) -> int:
        """Append newly produced model-cache tokens and return their count."""

        layers = self._legacy_layers(past_key_values)
        sequence_length = self._validate_layers(layers)
        current = self.token_count
        if sequence_length < current:
            raise HuggingFacePagedCacheError(
                "model cache is shorter than its paged mirror; truncate first"
            )
        if sequence_length == current:
            return 0

        entries = []
        for token_index in range(current, sequence_length):
            keys = tuple(
                self._head_rows(layer[0], token_index)
                for layer in layers
            )
            values = tuple(
                self._head_rows(layer[1], token_index)
                for layer in layers
            )
            entries.append((keys, values))
        self.cache.append_many(self._SEQUENCE_ID, entries)
        appended = sequence_length - current
        self._synchronized_tokens += appended
        self.cache.validate_invariants()
        return appended

    @staticmethod
    def _tensor_kwargs(template: Any | None) -> dict[str, Any]:
        if template is None:
            return {}
        kwargs: dict[str, Any] = {}
        device = getattr(template, "device", None)
        dtype = getattr(template, "dtype", None)
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return kwargs

    def materialize_legacy_cache(
        self,
        torch_module: Any,
        *,
        template: Any | None = None,
    ) -> tuple[tuple[Any, Any], ...]:
        """Dequantize the paged state into legacy Hugging Face KV tensors.

        The result uses ``[batch, heads, sequence, head_dim]`` layout with a
        batch size of one. When a template cache is supplied, each reconstructed
        tensor inherits the corresponding device and dtype. This deliberately
        slow reference path defines the tensor contract for future CUDA kernels.
        """

        tensor = getattr(torch_module, "tensor", None)
        empty = getattr(torch_module, "empty", None)
        if not callable(tensor) or not callable(empty):
            raise HuggingFacePagedCacheError(
                "torch_module must provide callable tensor() and empty() factories"
            )

        template_layers: tuple[Any, ...] | None = None
        if template is not None:
            template_layers = self._legacy_layers(template)
            self._validate_layers(template_layers)

        config = self.cache.config
        tokens = self.cache.read_sequence(self._SEQUENCE_ID)
        layers: list[tuple[Any, Any]] = []
        for layer_index in range(config.num_layers):
            key_template = None
            value_template = None
            if template_layers is not None:
                key_template = template_layers[layer_index][0]
                value_template = template_layers[layer_index][1]

            if tokens:
                keys = [
                    [
                        list(token.keys[layer_index][head_index])
                        for token in tokens
                    ]
                    for head_index in range(config.num_heads)
                ]
                values = [
                    [
                        list(token.values[layer_index][head_index])
                        for token in tokens
                    ]
                    for head_index in range(config.num_heads)
                ]
                key_tensor = tensor(
                    [keys],
                    **self._tensor_kwargs(key_template),
                )
                value_tensor = tensor(
                    [values],
                    **self._tensor_kwargs(value_template),
                )
            else:
                shape = (1, config.num_heads, 0, config.head_dim)
                key_tensor = empty(shape, **self._tensor_kwargs(key_template))
                value_tensor = empty(shape, **self._tensor_kwargs(value_template))
            layers.append((key_tensor, value_tensor))

        self._materializations += 1
        return tuple(layers)

    def materialize_like(self, torch_module: Any, template: Any) -> Any:
        """Materialize paged state using the template cache's container type."""

        legacy = self.materialize_legacy_cache(torch_module, template=template)
        if isinstance(template, list):
            return list(legacy)
        if isinstance(template, tuple):
            return legacy

        factory = getattr(type(template), "from_legacy_cache", None)
        if not callable(factory):
            raise HuggingFacePagedCacheError(
                "cache type cannot be reconstructed from legacy KV tensors"
            )
        try:
            return factory(legacy)
        except Exception as error:
            raise HuggingFacePagedCacheError(
                "cache type rejected reconstructed legacy KV tensors"
            ) from error

    def truncate(self, token_count: int) -> int:
        """Mirror speculative rollback and return the number of removed tokens."""

        current = self.token_count
        self.cache.truncate(self._SEQUENCE_ID, token_count)
        removed = current - token_count
        self._rollback_tokens += removed
        self.cache.validate_invariants()
        return removed

    def reset(self) -> None:
        if self.token_count:
            self.cache.truncate(self._SEQUENCE_ID, 0)
        self._resets += 1
        self.cache.validate_invariants()
