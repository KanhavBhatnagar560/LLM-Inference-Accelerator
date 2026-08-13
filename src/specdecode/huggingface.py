"""Optional Hugging Face/PyTorch model integration.

Heavy dependencies are imported only when a real model is loaded, so the Python
reference engine and its tests remain dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .hf_paged_cache import HuggingFacePagedCacheConfig, HuggingFacePagedCacheMirror
from .models import CausalLMProbabilityAdapter
from .tokenizers import validate_tokenizer_compatibility


class MissingOptionalDependencyError(ImportError):
    """Raised when real-model support is requested without its extra packages."""


@dataclass(frozen=True, slots=True)
class HuggingFaceCacheStats:
    """Cumulative cache reuse and reconciliation counters for one adapter."""

    full_prefills: int
    incremental_forwards: int
    exact_cache_hits: int
    cropped_tokens: int


def _require_backends() -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    except ImportError as error:
        raise MissingOptionalDependencyError(
            "Real-model support requires PyTorch, Transformers, and Accelerate. "
            "Install them with: pip install -e '.[transformers]'"
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer, __version__


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return choices[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {dtype}") from error


def _hub_kwargs(
    *,
    revision: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    return kwargs


class HuggingFaceCausalLM(CausalLMProbabilityAdapter):
    """Adapt one ``AutoModelForCausalLM`` to the probability interfaces."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        *,
        use_kv_cache: bool = False,
        paged_cache_mirror: HuggingFacePagedCacheMirror | None = None,
    ) -> None:
        vocab_size = int(model.config.vocab_size)
        super().__init__(vocab_size)
        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch_module
        self.use_kv_cache = use_kv_cache
        self.paged_cache_mirror = paged_cache_mirror
        self.cuda_runtime: Any | None = None
        self.cuda_stream_role = "target"
        self._cache_tokens: tuple[int, ...] = ()
        self._past_key_values: Any | None = None
        self._cached_next_probabilities: tuple[float, ...] | None = None
        self._full_prefills = 0
        self._incremental_forwards = 0
        self._exact_cache_hits = 0
        self._cropped_tokens = 0
        self.model.eval()

        vocabulary = dict(tokenizer.get_vocab())
        if vocabulary and max(vocabulary.values()) >= vocab_size:
            raise ValueError("tokenizer contains IDs outside the model output vocabulary")
        embeddings = model.get_input_embeddings()
        embedding_count = getattr(embeddings, "num_embeddings", vocab_size)
        if vocabulary and max(vocabulary.values()) >= int(embedding_count):
            raise ValueError("tokenizer contains IDs outside the model input embeddings")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        tokenizer: Any | None = None,
        use_kv_cache: bool = True,
        mirror_paged_kv_cache: bool = False,
        paged_cache_block_size: int = 16,
        paged_cache_num_blocks: int | None = None,
    ) -> "HuggingFaceCausalLM":
        torch, model_class, tokenizer_class, transformers_version = _require_backends()
        common_kwargs = _hub_kwargs(
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        if tokenizer is None:
            tokenizer = tokenizer_class.from_pretrained(model_id, use_fast=True, **common_kwargs)
        model_kwargs = dict(common_kwargs)
        model_kwargs["device_map"] = device if device == "auto" else {"": device}
        dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
        model_kwargs[dtype_key] = _resolve_dtype(torch, dtype)
        model = model_class.from_pretrained(model_id, **model_kwargs)
        mirror = None
        if mirror_paged_kv_cache:
            if not use_kv_cache:
                raise ValueError("paged cache mirroring requires Hugging Face KV caching")
            mirror = HuggingFacePagedCacheMirror.from_model_config(
                model.config,
                HuggingFacePagedCacheConfig(
                    block_size=paged_cache_block_size,
                    num_blocks=paged_cache_num_blocks,
                ),
            )
        return cls(
            model,
            tokenizer,
            torch,
            use_kv_cache=use_kv_cache,
            paged_cache_mirror=mirror,
        )

    @property
    def input_device(self) -> Any:
        return self.model.get_input_embeddings().weight.device

    def configure_cuda_runtime(self, runtime: Any, *, stream_role: str) -> None:
        if stream_role not in ("draft", "target"):
            raise ValueError("model CUDA stream role must be draft or target")
        self.cuda_runtime = runtime
        self.cuda_stream_role = stream_role

    @property
    def cached_token_count(self) -> int:
        return len(self._cache_tokens)

    @property
    def cache_stats(self) -> HuggingFaceCacheStats:
        return HuggingFaceCacheStats(
            full_prefills=self._full_prefills,
            incremental_forwards=self._incremental_forwards,
            exact_cache_hits=self._exact_cache_hits,
            cropped_tokens=self._cropped_tokens,
        )

    def reset_cache(self) -> None:
        """Discard request-local model state without resetting cumulative stats."""

        self._cache_tokens = ()
        self._past_key_values = None
        self._cached_next_probabilities = None
        if self.paged_cache_mirror is not None:
            self.paged_cache_mirror.reset()

    @staticmethod
    def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
        length = 0
        for left_token, right_token in zip(left, right):
            if left_token != right_token:
                break
            length += 1
        return length

    @staticmethod
    def _crop_legacy_cache(value: Any, length: int) -> Any:
        if isinstance(value, tuple):
            return tuple(
                HuggingFaceCausalLM._crop_legacy_cache(item, length)
                for item in value
            )
        if isinstance(value, list):
            return [
                HuggingFaceCausalLM._crop_legacy_cache(item, length)
                for item in value
            ]
        ndim = getattr(value, "ndim", 0)
        if ndim >= 3:
            return value[..., :length, :]
        return value

    def _crop_cache(self, length: int) -> tuple[Any | None, int]:
        old_length = len(self._cache_tokens)
        if not 0 <= length <= old_length:
            raise ValueError("cache crop length is outside the cached prefix")
        if length == old_length:
            return self._past_key_values, length
        if length == 0:
            self._cropped_tokens += old_length
            self.reset_cache()
            return None, 0

        past_key_values = self._past_key_values
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            physical_length = int(get_seq_length())
            retained_start = old_length - physical_length
            if length < retained_start:
                self._cropped_tokens += old_length
                self.reset_cache()
                return None, 0
        crop = getattr(past_key_values, "crop", None)
        if callable(crop):
            cropped = crop(length - old_length)
            past_key_values = past_key_values if cropped is None else cropped
        elif isinstance(past_key_values, (tuple, list)):
            past_key_values = self._crop_legacy_cache(past_key_values, length)
        else:
            self.reset_cache()
            raise RuntimeError(
                "past_key_values does not support prefix cropping; disable KV-cache "
                "reuse with --no-kv-cache"
            )
        self._cropped_tokens += old_length - length
        self._cache_tokens = self._cache_tokens[:length]
        self._past_key_values = past_key_values
        self._cached_next_probabilities = None
        if self.paged_cache_mirror is not None:
            self.paged_cache_mirror.truncate(length)
        return past_key_values, length

    def _forward(
        self,
        input_token_ids: Sequence[int],
        *,
        total_context_length: int,
        past_key_values: Any | None,
        use_cache: bool,
    ) -> Any:
        if not input_token_ids:
            raise ValueError("model forward requires at least one input token")
        input_device = "cpu" if self.cuda_runtime is not None else self.input_device
        input_ids = self._torch.tensor(
            [list(input_token_ids)],
            dtype=self._torch.long,
            device=input_device,
        )
        attention_mask = self._torch.ones(
            (1, total_context_length),
            dtype=self._torch.long,
            device=input_device,
        )

        dependencies = ()
        if self.cuda_runtime is not None:
            input_task = self.cuda_runtime.copy_to_device(
                f"{self.cuda_stream_role}.input_ids",
                input_ids,
            )
            mask_task = self.cuda_runtime.copy_to_device(
                f"{self.cuda_stream_role}.attention_mask",
                attention_mask,
            )
            input_ids = input_task.result
            attention_mask = mask_task.result
            dependencies = (input_task, mask_task)

        def forward() -> Any:
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            with self._torch.inference_mode():
                return self.model(**kwargs)

        if self.cuda_runtime is None:
            return forward()
        submit = (
            self.cuda_runtime.submit_draft
            if self.cuda_stream_role == "draft"
            else self.cuda_runtime.submit_target
        )
        return submit(
            forward,
            wait_for=dependencies,
            label=f"specdecode.{self.cuda_stream_role}.forward",
        ).wait()

    def _rows_from_logits(
        self,
        logits: Any,
        positions: Sequence[int],
    ) -> list[list[float]]:
        if logits.ndim != 3 or int(logits.shape[-1]) != self.vocab_size:
            raise ValueError("causal LM returned logits with an unexpected shape")
        selected = logits[0, list(positions), :].float()
        probabilities = self._torch.softmax(selected, dim=-1)
        return probabilities.detach().cpu().tolist()

    def _cached_probability_rows(
        self,
        token_ids: Sequence[int],
        prefix_lengths: Sequence[int],
    ) -> list[list[float]]:
        tokens = tuple(int(token) for token in token_ids)
        desired = tuple(int(length) for length in prefix_lengths)
        if not tokens or not desired:
            raise ValueError("cached scoring requires tokens and output positions")
        if any(length < 1 or length > len(tokens) for length in desired):
            raise ValueError("cached scoring position is outside the token sequence")
        if tuple(sorted(desired)) != desired or len(set(desired)) != len(desired):
            raise ValueError("cached scoring positions must be strictly increasing")

        cached_length = len(self._cache_tokens)
        if (
            self._cached_next_probabilities is not None
            and desired[0] == cached_length
            and tokens[:cached_length] == self._cache_tokens
        ):
            first = self._cached_next_probabilities
            self._exact_cache_hits += 1
            if len(desired) == 1:
                return [list(first)]
            reuse_length = cached_length
            cached_rows = [list(first)]
            output_lengths = desired[1:]
        else:
            common = self._common_prefix_length(self._cache_tokens, tokens)
            reuse_length = min(common, desired[0] - 1)
            cached_rows = []
            output_lengths = desired

        try:
            past_key_values, reuse_length = self._crop_cache(reuse_length)
        except Exception:
            self.reset_cache()
            raise
        input_tokens = tokens[reuse_length:]
        try:
            outputs = self._forward(
                input_tokens,
                total_context_length=len(tokens),
                past_key_values=past_key_values,
                use_cache=True,
            )
            output_cache = getattr(outputs, "past_key_values", None)
            if output_cache is None:
                raise RuntimeError("causal LM did not return past_key_values")
            if self.paged_cache_mirror is not None:
                self.paged_cache_mirror.synchronize(output_cache)
            indices = tuple(length - reuse_length - 1 for length in output_lengths)
            rows = self._rows_from_logits(outputs.logits, indices)
            final_row = self._rows_from_logits(
                outputs.logits,
                (len(input_tokens) - 1,),
            )[0]
        except Exception:
            self.reset_cache()
            raise

        if reuse_length == 0:
            self._full_prefills += 1
        else:
            self._incremental_forwards += 1
        self._cache_tokens = tokens
        self._past_key_values = output_cache
        self._cached_next_probabilities = tuple(final_row)
        return cached_rows + rows

    def _probabilities_at_positions(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
    ) -> Sequence[Sequence[float]]:
        if not token_ids:
            raise ValueError("a causal language model requires a non-empty context")
        outputs = self._forward(
            token_ids,
            total_context_length=len(token_ids),
            past_key_values=None,
            use_cache=False,
        )
        return self._rows_from_logits(outputs.logits, positions)

    def next_token_probs(self, token_ids: Sequence[int]) -> Sequence[float]:
        if not self.use_kv_cache:
            return super().next_token_probs(token_ids)
        return self._cached_probability_rows(token_ids, (len(token_ids),))[0]

    def score_proposal(
        self,
        prefix: Sequence[int],
        proposal: Sequence[int],
    ) -> Sequence[Sequence[float]]:
        if not self.use_kv_cache:
            return super().score_proposal(prefix, proposal)
        if not prefix:
            raise ValueError("a causal language model requires a non-empty prefix")
        combined = tuple(prefix) + tuple(proposal)
        lengths = tuple(range(len(prefix), len(combined) + 1))
        return self._cached_probability_rows(combined, lengths)


@dataclass(frozen=True, slots=True)
class HuggingFaceModelPair:
    """A validated draft/target pair sharing one token-ID space."""

    draft: HuggingFaceCausalLM
    target: HuggingFaceCausalLM
    tokenizer: Any
    cuda_runtime: Any | None = None
    transformers_version: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        draft_model_id: str,
        target_model_id: str,
        *,
        draft_revision: str | None = None,
        target_revision: str | None = None,
        draft_device: str = "auto",
        target_device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        enable_cuda_runtime: bool = False,
        cuda_device: str | None = None,
        use_kv_cache: bool = True,
        mirror_paged_kv_cache: bool = False,
        paged_cache_block_size: int = 16,
        paged_cache_num_blocks: int | None = None,
    ) -> "HuggingFaceModelPair":
        _, _, tokenizer_class, transformers_version = _require_backends()
        draft_tokenizer = tokenizer_class.from_pretrained(
            draft_model_id,
            use_fast=True,
            **_hub_kwargs(
                revision=draft_revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            ),
        )
        target_tokenizer = tokenizer_class.from_pretrained(
            target_model_id,
            use_fast=True,
            **_hub_kwargs(
                revision=target_revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            ),
        )
        # Validate the lightweight tokenizer artifacts before loading either
        # model's large weight files.
        validate_tokenizer_compatibility(draft_tokenizer, target_tokenizer)

        draft = HuggingFaceCausalLM.from_pretrained(
            draft_model_id,
            revision=draft_revision,
            device=draft_device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            tokenizer=draft_tokenizer,
            use_kv_cache=use_kv_cache,
            mirror_paged_kv_cache=mirror_paged_kv_cache,
            paged_cache_block_size=paged_cache_block_size,
            paged_cache_num_blocks=paged_cache_num_blocks,
        )
        target = HuggingFaceCausalLM.from_pretrained(
            target_model_id,
            revision=target_revision,
            device=target_device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            tokenizer=target_tokenizer,
            use_kv_cache=use_kv_cache,
            mirror_paged_kv_cache=mirror_paged_kv_cache,
            paged_cache_block_size=paged_cache_block_size,
            paged_cache_num_blocks=paged_cache_num_blocks,
        )
        if draft.vocab_size != target.vocab_size:
            raise ValueError("draft and target model output vocabularies differ")
        runtime = None
        if enable_cuda_runtime:
            from .cuda import CudaExecutionRuntime, CudaRuntimeConfig

            draft_device = str(draft.input_device)
            target_device = str(target.input_device)
            if draft_device != target_device:
                raise ValueError(
                    "one shared CUDA runtime requires draft and target on the same device"
                )
            selected_device = cuda_device or target_device
            if str(target._torch.device(selected_device)) != str(
                target._torch.device(target_device)
            ):
                raise ValueError("CUDA runtime device must match the loaded model device")
            runtime = CudaExecutionRuntime(
                CudaRuntimeConfig(device=selected_device),
                torch_module=target._torch,
            )
            draft.configure_cuda_runtime(runtime, stream_role="draft")
            target.configure_cuda_runtime(runtime, stream_role="target")
        return cls(
            draft=draft,
            target=target,
            tokenizer=target.tokenizer,
            cuda_runtime=runtime,
            transformers_version=transformers_version,
        )
