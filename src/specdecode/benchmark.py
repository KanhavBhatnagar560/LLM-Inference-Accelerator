"""Reproducible target-only versus speculative benchmark harness."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Warmup, repetition, and seed controls shared by both benchmark paths."""

    warmup_runs: int = 2
    measured_runs: int = 10
    seed: int = 7

    def __post_init__(self) -> None:
        if (
            isinstance(self.warmup_runs, bool)
            or not isinstance(self.warmup_runs, int)
            or self.warmup_runs < 0
        ):
            raise ValueError("warmup_runs must be a non-negative integer")
        if (
            isinstance(self.measured_runs, bool)
            or not isinstance(self.measured_runs, int)
            or self.measured_runs < 1
        ):
            raise ValueError("measured_runs must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@runtime_checkable
class GenerationRunner(Protocol):
    """One benchmarkable generation path."""

    name: str

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        seed: int,
        on_token: Callable[[int], None],
    ) -> Sequence[int]: ...


@runtime_checkable
class MemoryProbe(Protocol):
    """Optional peak-memory probe, implemented by the CUDA runtime."""

    def reset_peak_memory(self) -> None: ...

    def memory_snapshot(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    path: str
    prompt_index: int
    repetition: int
    seed: int
    prompt_sha256: str
    output_sha256: str
    input_tokens: int
    output_tokens: int
    elapsed_ms: float
    token_latencies_ms: tuple[float, ...]
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    prompt_runs: int
    generated_tokens: int
    total_seconds: float
    tokens_per_second: float
    mean_token_latency_ms: float
    p50_token_latency_ms: float
    p95_token_latency_ms: float
    p99_token_latency_ms: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    schema_version: int
    created_at_utc: str
    config: BenchmarkConfig
    settings: Mapping[str, Any]
    environment: Mapping[str, Any]
    target_only: BenchmarkMetrics
    speculative: BenchmarkMetrics
    throughput_speedup: float | None
    samples: tuple[BenchmarkSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


class DecoderBenchmarkRunner:
    """Adapt a seed-specific decoder factory to :class:`GenerationRunner`."""

    def __init__(self, name: str, decoder_factory: Callable[[int], Any]) -> None:
        self.name = name
        self._decoder_factory = decoder_factory

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        seed: int,
        on_token: Callable[[int], None],
    ) -> Sequence[int]:
        decoder = self._decoder_factory(seed)
        result = decoder.generate(
            prompt_tokens,
            on_token=lambda event: on_token(event.token_id),
        )
        return result.generated_tokens


def system_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _token_hash(tokens: Sequence[int]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _snapshot_peaks(memory_probe: MemoryProbe | None) -> tuple[int | None, int | None]:
    if memory_probe is None:
        return None, None
    snapshot = memory_probe.memory_snapshot()
    return int(snapshot.peak_allocated_bytes), int(snapshot.peak_reserved_bytes)


def _run_sample(
    path: str,
    runner: GenerationRunner,
    prompt: tuple[int, ...],
    *,
    prompt_index: int,
    repetition: int,
    seed: int,
    synchronize: Callable[[], None],
    memory_probe: MemoryProbe | None,
    clock: Callable[[], float],
) -> BenchmarkSample:
    synchronize()
    if memory_probe is not None:
        memory_probe.reset_peak_memory()
    start = clock()
    token_times: list[float] = []
    generated = tuple(
        int(token)
        for token in runner.generate(
            prompt,
            seed=seed,
            on_token=lambda _token: token_times.append(clock()),
        )
    )
    synchronize()
    end = clock()
    if len(token_times) != len(generated):
        raise RuntimeError(
            f"{path} emitted {len(token_times)} callbacks for {len(generated)} tokens"
        )

    previous = start
    latencies: list[float] = []
    for timestamp in token_times:
        if timestamp < previous:
            raise RuntimeError("benchmark clock moved backwards")
        latencies.append((timestamp - previous) * 1000.0)
        previous = timestamp
    peak_allocated, peak_reserved = _snapshot_peaks(memory_probe)
    return BenchmarkSample(
        path=path,
        prompt_index=prompt_index,
        repetition=repetition,
        seed=seed,
        prompt_sha256=_token_hash(prompt),
        output_sha256=_token_hash(generated),
        input_tokens=len(prompt),
        output_tokens=len(generated),
        elapsed_ms=(end - start) * 1000.0,
        token_latencies_ms=tuple(latencies),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def _aggregate(samples: Sequence[BenchmarkSample]) -> BenchmarkMetrics:
    token_latencies = [
        latency
        for sample in samples
        for latency in sample.token_latencies_ms
    ]
    total_seconds = sum(sample.elapsed_ms for sample in samples) / 1000.0
    generated_tokens = sum(sample.output_tokens for sample in samples)
    allocated = [
        sample.peak_allocated_bytes
        for sample in samples
        if sample.peak_allocated_bytes is not None
    ]
    reserved = [
        sample.peak_reserved_bytes
        for sample in samples
        if sample.peak_reserved_bytes is not None
    ]
    return BenchmarkMetrics(
        prompt_runs=len(samples),
        generated_tokens=generated_tokens,
        total_seconds=total_seconds,
        tokens_per_second=(
            generated_tokens / total_seconds if total_seconds > 0.0 else 0.0
        ),
        mean_token_latency_ms=(
            statistics.fmean(token_latencies) if token_latencies else 0.0
        ),
        p50_token_latency_ms=_percentile(token_latencies, 0.50),
        p95_token_latency_ms=_percentile(token_latencies, 0.95),
        p99_token_latency_ms=_percentile(token_latencies, 0.99),
        peak_allocated_bytes=max(allocated) if allocated else None,
        peak_reserved_bytes=max(reserved) if reserved else None,
    )


def run_comparison_benchmark(
    target_only_runner: GenerationRunner,
    speculative_runner: GenerationRunner,
    prompts: Sequence[Sequence[int]],
    config: BenchmarkConfig | None = None,
    *,
    settings: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    synchronize: Callable[[], None] | None = None,
    memory_probe: MemoryProbe | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkReport:
    """Measure both paths with identical prompts, seeds, and run counts."""

    benchmark_config = config or BenchmarkConfig()
    prepared: list[tuple[int, ...]] = []
    for prompt in prompts:
        converted: list[int] = []
        for token in prompt:
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise TypeError("benchmark prompt token IDs must be integers")
            converted.append(int(token))
        prepared.append(tuple(converted))
    prepared_prompts = tuple(prepared)
    if not prepared_prompts:
        raise ValueError("benchmark requires at least one prompt")
    if any(not prompt for prompt in prepared_prompts):
        raise ValueError("benchmark prompts cannot be empty")
    sync = synchronize or (lambda: None)

    runners = (
        ("target_only", target_only_runner),
        ("speculative", speculative_runner),
    )
    for warmup in range(benchmark_config.warmup_runs):
        for prompt_index, prompt in enumerate(prepared_prompts):
            seed = benchmark_config.seed + warmup * len(prepared_prompts) + prompt_index
            for _, runner in runners:
                sync()
                runner.generate(prompt, seed=seed, on_token=lambda _token: None)
                sync()

    samples: list[BenchmarkSample] = []
    for repetition in range(benchmark_config.measured_runs):
        for prompt_index, prompt in enumerate(prepared_prompts):
            seed = (
                benchmark_config.seed
                + benchmark_config.warmup_runs * len(prepared_prompts)
                + repetition * len(prepared_prompts)
                + prompt_index
            )
            ordered_runners = (
                runners if (repetition + prompt_index) % 2 == 0 else tuple(reversed(runners))
            )
            for path, runner in ordered_runners:
                samples.append(
                    _run_sample(
                        path,
                        runner,
                        prompt,
                        prompt_index=prompt_index,
                        repetition=repetition,
                        seed=seed,
                        synchronize=sync,
                        memory_probe=memory_probe,
                        clock=clock,
                    )
                )

    target_samples = tuple(sample for sample in samples if sample.path == "target_only")
    speculative_samples = tuple(
        sample for sample in samples if sample.path == "speculative"
    )
    target_metrics = _aggregate(target_samples)
    speculative_metrics = _aggregate(speculative_samples)
    speedup = (
        speculative_metrics.tokens_per_second / target_metrics.tokens_per_second
        if target_metrics.tokens_per_second > 0.0
        else None
    )
    return BenchmarkReport(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=benchmark_config,
        settings=dict(settings or {}),
        environment=dict(environment or system_environment()),
        target_only=target_metrics,
        speculative=speculative_metrics,
        throughput_speedup=speedup,
        samples=tuple(samples),
    )
