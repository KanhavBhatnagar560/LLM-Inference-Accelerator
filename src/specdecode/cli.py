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
    generate.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="disable Hugging Face past_key_values reuse",
    )
    generate.add_argument("--local-files-only", action="store_true")
    generate.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow model repositories to execute custom code",
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="benchmark target-only and speculative decoding on one CUDA device",
    )
    benchmark.add_argument("--draft-model", required=True)
    benchmark.add_argument("--target-model", required=True)
    benchmark.add_argument("--draft-revision")
    benchmark.add_argument("--target-revision")
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    benchmark.add_argument("--prompt", action="append", required=True)
    benchmark.add_argument("--chat", action="store_true")
    benchmark.add_argument("--max-new-tokens", type=int, default=64)
    benchmark.add_argument("--initial-draft-tokens", type=int, default=4)
    benchmark.add_argument("--min-draft-tokens", type=int, default=1)
    benchmark.add_argument("--max-draft-tokens", type=int, default=8)
    benchmark.add_argument("--no-dynamic-draft", action="store_true")
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.add_argument("--warmup-runs", type=int, default=2)
    benchmark.add_argument("--measured-runs", type=int, default=10)
    benchmark.add_argument("--output", default="outputs/benchmark.json")
    benchmark.add_argument(
        "--sampling-backend",
        choices=("auto", "python", "native"),
        default="auto",
    )
    benchmark.add_argument("--native-library")
    benchmark.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="benchmark stateless full-context model forwards",
    )
    benchmark.add_argument("--local-files-only", action="store_true")
    benchmark.add_argument("--trust-remote-code", action="store_true")
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
        use_kv_cache=not args.no_kv_cache,
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


def run_benchmark_command(args: argparse.Namespace) -> int:
    from .benchmark import (
        BenchmarkConfig,
        DecoderBenchmarkRunner,
        run_comparison_benchmark,
    )
    from .decoder import TargetOnlyDecoder
    from .huggingface import HuggingFaceModelPair

    pair = HuggingFaceModelPair.from_pretrained(
        args.draft_model,
        args.target_model,
        draft_revision=args.draft_revision,
        target_revision=args.target_revision,
        draft_device=args.device,
        target_device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        enable_cuda_runtime=True,
        cuda_device=args.device,
        use_kv_cache=not args.no_kv_cache,
    )
    if pair.cuda_runtime is None:
        raise RuntimeError("benchmark requested CUDA but no CUDA runtime was configured")
    prompts = tuple(
        encode_prompt(pair.tokenizer, prompt, chat=args.chat) for prompt in args.prompt
    )
    decode_config = DecodeConfig(
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
    target_runner = DecoderBenchmarkRunner(
        "target_only",
        lambda seed: TargetOnlyDecoder(
            pair.target,
            decode_config,
            rng=random.Random(seed),
            sampling_backend=backend,
        ),
    )
    speculative_runner = DecoderBenchmarkRunner(
        "speculative",
        lambda seed: SpeculativeDecoder(
            pair.draft,
            pair.target,
            decode_config,
            rng=random.Random(seed),
            sampling_backend=backend,
        ),
    )
    environment = pair.cuda_runtime.environment()
    environment["transformers"] = pair.transformers_version
    report = run_comparison_benchmark(
        target_runner,
        speculative_runner,
        prompts,
        BenchmarkConfig(
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            seed=args.seed,
        ),
        settings={
            "draft_model": args.draft_model,
            "draft_revision": args.draft_revision,
            "target_model": args.target_model,
            "target_revision": args.target_revision,
            "dtype": args.dtype,
            "device": args.device,
            "batch_size": 1,
            "prompt_count": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "initial_draft_tokens": args.initial_draft_tokens,
            "min_draft_tokens": args.min_draft_tokens,
            "max_draft_tokens": args.max_draft_tokens,
            "dynamic_draft": not args.no_dynamic_draft,
            "sampling_backend": backend.name,
            "temperature": 1.0,
            "model_loading_excluded": True,
            "tokenization_excluded": True,
            "kv_cache_mode": (
                "huggingface_past_key_values"
                if not args.no_kv_cache
                else "stateless_full_context"
            ),
        },
        environment=environment,
        synchronize=pair.cuda_runtime.synchronize,
        memory_probe=pair.cuda_runtime,
    )
    destination = report.write_json(args.output)
    speedup = report.throughput_speedup
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "unavailable"
    print("benchmark report:", destination)
    print("target-only throughput:", f"{report.target_only.tokens_per_second:.3f} tok/s")
    print("speculative throughput:", f"{report.speculative.tokens_per_second:.3f} tok/s")
    print("throughput speedup:", speedup_text)
    print("speculative p99 token latency:", f"{report.speculative.p99_token_latency_ms:.3f} ms")
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
        if args.command == "benchmark":
            return run_benchmark_command(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"specdecode: error: {error}\n")
    parser.error(f"unknown command: {args.command}")
    return 2
