"""Optional Hugging Face/PyTorch model integration.

Heavy dependencies are imported only when a real model is loaded, so the Python
reference engine and its tests remain dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .models import CausalLMProbabilityAdapter
from .tokenizers import validate_tokenizer_compatibility


class MissingOptionalDependencyError(ImportError):
    """Raised when real-model support is requested without its extra packages."""


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

    def __init__(self, model: Any, tokenizer: Any, torch_module: Any) -> None:
        vocab_size = int(model.config.vocab_size)
        super().__init__(vocab_size)
        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch_module
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
        return cls(model, tokenizer, torch)

    @property
    def input_device(self) -> Any:
        return self.model.get_input_embeddings().weight.device

    def _probabilities_at_positions(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
    ) -> Sequence[Sequence[float]]:
        if not token_ids:
            raise ValueError("a causal language model requires a non-empty context")
        input_ids = self._torch.tensor(
            [list(token_ids)],
            dtype=self._torch.long,
            device=self.input_device,
        )
        attention_mask = self._torch.ones_like(input_ids)
        with self._torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits = outputs.logits
        if logits.ndim != 3 or int(logits.shape[-1]) != self.vocab_size:
            raise ValueError("causal LM returned logits with an unexpected shape")
        selected = logits[0, list(positions), :].float()
        probabilities = self._torch.softmax(selected, dim=-1)
        return probabilities.detach().cpu().tolist()


@dataclass(frozen=True, slots=True)
class HuggingFaceModelPair:
    """A validated draft/target pair sharing one token-ID space."""

    draft: HuggingFaceCausalLM
    target: HuggingFaceCausalLM
    tokenizer: Any

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
    ) -> "HuggingFaceModelPair":
        _, _, tokenizer_class, _ = _require_backends()
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
        )
        target = HuggingFaceCausalLM.from_pretrained(
            target_model_id,
            revision=target_revision,
            device=target_device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            tokenizer=target_tokenizer,
        )
        if draft.vocab_size != target.vocab_size:
            raise ValueError("draft and target model output vocabularies differ")
        return cls(draft=draft, target=target, tokenizer=target.tokenizer)
