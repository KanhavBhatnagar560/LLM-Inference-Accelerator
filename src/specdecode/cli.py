"""Command-line interface for toy and Hugging Face generation."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence

from .backends import load_sampling_backend
from .config import DecodeConfig
from .decoder import SpeculativeDecoder
from .events import TokenEvent
from .models import TableModel
from .tokenizers import IncrementalTextDecoder, encode_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact speculative-decoding reference engine")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="run the dependency-free toy model")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument(
        "--sampling-backend", choices=("auto", "python", "native"), default="auto"
    )
    demo.add_argument("--native-library")

    generate = subparsers.add_parser("generate", help="generate with Hugging Face models")
    generate.add_argument("--draft-model", required=True)
    generate.add_argument("--target-model", required=True)
    generate.add_argument("--draft-revision")
    generate.add_argument("--target-revision")
    generate.add_argument("--draft-device", default="auto")
    generate.add_argument("--target-device", default="auto")
    generate.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--chat", action="store_true", help="apply the target chat template")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--initial-draft-tokens", type=int, default=4)
    generate.add_argument("--min-draft-tokens", type=int, default=1)
    generate.add_argument("--max-draft-tokens", type=int, default=8)
    generate.add_argument("--no-dynamic-draft", action="store_true")
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument(
        "--sampling-backend", choices=("auto", "python", "native"), default="auto"
    )
    generate.add_argument("--native-library")
    generate.add_argument("--local-files-only", action="store_true")
    generate.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow model repositories to execute custom code",
    )
    return parser


def _toy_decoder(
    seed: int,
    *,
    backend_mode: str = "auto",
    native_library: str | None = None,
) -> SpeculativeDecoder:
    draft = TableModel(
        {(0,): (0.05, 0.70, 0.20, 0.05), (1,): (0.05, 0.20, 0.65, 0.10)},
        default=(0.05, 0.10, 0.35, 0.50),
    )
    target = TableModel(
        {(0,): (0.05, 0.55, 0.30, 0.10), (1,): (0.05, 0.15, 0.60, 0.20)},
        default=(0.05, 0.10, 0.25, 0.60),
    )
    backend = load_sampling_backend(backend_mode, library_path=native_library)
    return SpeculativeDecoder(
        draft,
        target,
        DecodeConfig(max_new_tokens=12, eos_token_id=3),
        rng=random.Random(seed),
        sampling_backend=backend,
    )


def run_demo(
    seed: int,
    *,
    backend_mode: str = "auto",
    native_library: str | None = None,
) -> int:
    decoder = _toy_decoder(
        seed,
        backend_mode=backend_mode,
        native_library=native_library,
    )
    result = decoder.generate([0])
    print("generated token ids:", list(result.generated_tokens))
    print(
        "draft acceptance:",
        f"{result.stats.accepted_tokens}/{result.stats.drafted_tokens}",
        f"({result.stats.acceptance_rate:.1%})",
    )
    print("target verification rounds:", result.stats.verification_rounds)
    print("sampling backend:", decoder.sampling_backend.name)
    return 0


def run_generate(args: argparse.Namespace) -> int:
    from .huggingface import HuggingFaceModelPair

    pair = HuggingFaceModelPair.from_pretrained(
        args.draft_model,
        args.target_model,
        draft_revision=args.draft_revision,
        target_revision=args.target_revision,
        draft_device=args.draft_device,
        target_device=args.target_device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    prompt_tokens = encode_prompt(pair.tokenizer, args.prompt, chat=args.chat)
    config = DecodeConfig(
        max_new_tokens=args.max_new_tokens,
        initial_draft_tokens=args.initial_draft_tokens,
        min_draft_tokens=args.min_draft_tokens,
        max_draft_tokens=args.max_draft_tokens,
        dynamic_draft=not args.no_dynamic_draft,
        eos_token_id=pair.tokenizer.eos_token_id,
    )
    backend = load_sampling_backend(
        args.sampling_backend,
        library_path=args.native_library,
    )
    decoder = SpeculativeDecoder(
        pair.draft,
        pair.target,
        config,
        rng=random.Random(args.seed),
        sampling_backend=backend,
    )
    text_decoder = IncrementalTextDecoder(pair.tokenizer)

    def stream(event: TokenEvent) -> None:
        text = text_decoder.push(event.token_id)
        if text:
            print(text, end="", flush=True)

    result = decoder.generate(prompt_tokens, on_token=stream)
    remaining_text = text_decoder.flush()
    if remaining_text:
        print(remaining_text, end="", flush=True)
    print()
    print(
        f"accepted {result.stats.accepted_tokens}/{result.stats.drafted_tokens} draft tokens "
        f"({result.stats.acceptance_rate:.1%}) in "
        f"{result.stats.verification_rounds} verification rounds using "
        f"the {decoder.sampling_backend.name} sampling backend",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "demo"):
            return run_demo(
                getattr(args, "seed", 7),
                backend_mode=getattr(args, "sampling_backend", "auto"),
                native_library=getattr(args, "native_library", None),
            )
        if args.command == "generate":
            return run_generate(args)
    except (ImportError, OSError, RuntimeError) as error:
        parser.exit(2, f"specdecode: error: {error}\n")
    parser.error(f"unknown command: {args.command}")
    return 2
