import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from specdecode import DecodeConfig, TableModel, TargetOnlyDecoder
from specdecode.benchmark import (
    BenchmarkConfig,
    DecoderBenchmarkRunner,
    run_comparison_benchmark,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class TimedRunner:
    def __init__(self, name, clock, token_seconds, output=(1, 2)):
        self.name = name
        self.clock = clock
        self.token_seconds = token_seconds
        self.output = output
        self.calls = []

    def generate(self, prompt_tokens, *, seed, on_token):
        self.calls.append((tuple(prompt_tokens), seed))
        for token in self.output:
            self.clock.advance(self.token_seconds)
            on_token(token)
        return self.output


class FixedMemoryProbe:
    def __init__(self):
        self.resets = 0

    def reset_peak_memory(self):
        self.resets += 1

    def memory_snapshot(self):
        return SimpleNamespace(peak_allocated_bytes=100, peak_reserved_bytes=200)


class BrokenCallbackRunner(TimedRunner):
    def generate(self, prompt_tokens, *, seed, on_token):
        return self.output


class BenchmarkTests(unittest.TestCase):
    def test_comparison_uses_identical_seeds_and_computes_metrics(self) -> None:
        clock = FakeClock()
        target = TimedRunner("target", clock, 0.1)
        speculative = TimedRunner("speculative", clock, 0.05)
        memory = FixedMemoryProbe()

        report = run_comparison_benchmark(
            target,
            speculative,
            ((1, 2), (3, 4, 5)),
            BenchmarkConfig(warmup_runs=1, measured_runs=2, seed=10),
            settings={"temperature": 1.0},
            environment={"device": "fake-cuda"},
            synchronize=lambda: None,
            memory_probe=memory,
            clock=clock,
        )

        self.assertEqual(report.target_only.prompt_runs, 4)
        self.assertEqual(report.speculative.prompt_runs, 4)
        self.assertEqual(report.target_only.generated_tokens, 8)
        self.assertAlmostEqual(report.target_only.tokens_per_second, 10.0)
        self.assertAlmostEqual(report.speculative.tokens_per_second, 20.0)
        self.assertAlmostEqual(report.throughput_speedup, 2.0)
        self.assertAlmostEqual(report.target_only.p99_token_latency_ms, 100.0)
        self.assertAlmostEqual(report.speculative.p99_token_latency_ms, 50.0)
        self.assertEqual(report.target_only.peak_allocated_bytes, 100)
        self.assertEqual(report.speculative.peak_reserved_bytes, 200)
        self.assertEqual(memory.resets, 8)

        measured_target = target.calls[2:]
        measured_speculative = speculative.calls[2:]
        self.assertEqual(sorted(measured_target), sorted(measured_speculative))
        self.assertEqual(report.settings, {"temperature": 1.0})
        self.assertEqual(report.environment, {"device": "fake-cuda"})

    def test_order_alternates_to_reduce_path_order_bias(self) -> None:
        clock = FakeClock()
        target = TimedRunner("target", clock, 0.01, output=(1,))
        speculative = TimedRunner("speculative", clock, 0.01, output=(1,))

        report = run_comparison_benchmark(
            target,
            speculative,
            ((0,), (1,)),
            BenchmarkConfig(warmup_runs=0, measured_runs=2),
            clock=clock,
        )

        self.assertEqual(
            tuple(sample.path for sample in report.samples),
            (
                "target_only",
                "speculative",
                "speculative",
                "target_only",
                "speculative",
                "target_only",
                "target_only",
                "speculative",
            ),
        )

    def test_report_writes_reproducible_json_without_raw_tokens(self) -> None:
        clock = FakeClock()
        target = TimedRunner("target", clock, 0.01, output=(2,))
        speculative = TimedRunner("speculative", clock, 0.01, output=(2,))
        report = run_comparison_benchmark(
            target,
            speculative,
            ((0, 1),),
            BenchmarkConfig(warmup_runs=0, measured_runs=1),
            clock=clock,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = report.write_json(Path(directory) / "report.json")
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["config"]["measured_runs"], 1)
        self.assertEqual(len(payload["samples"]), 2)
        self.assertIn("prompt_sha256", payload["samples"][0])
        self.assertIn("output_sha256", payload["samples"][0])
        self.assertNotIn("prompt_tokens", payload["samples"][0])

    def test_callback_count_must_match_generated_tokens(self) -> None:
        clock = FakeClock()
        broken = BrokenCallbackRunner("broken", clock, 0.01)
        valid = TimedRunner("valid", clock, 0.01)

        with self.assertRaises(RuntimeError):
            run_comparison_benchmark(
                broken,
                valid,
                ((0,),),
                BenchmarkConfig(warmup_runs=0, measured_runs=1),
                clock=clock,
            )

    def test_config_and_prompts_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(measured_runs=0)
        with self.assertRaises(ValueError):
            BenchmarkConfig(warmup_runs="1")
        clock = FakeClock()
        runner = TimedRunner("runner", clock, 0.01)
        with self.assertRaises(ValueError):
            run_comparison_benchmark(runner, runner, (), clock=clock)
        with self.assertRaises(ValueError):
            run_comparison_benchmark(runner, runner, ((),), clock=clock)
        with self.assertRaises(TypeError):
            run_comparison_benchmark(runner, runner, ((True,),), clock=clock)

    def test_decoder_runner_wraps_target_only_baseline(self) -> None:
        model = TableModel({}, default=(0.0, 1.0))
        config = DecodeConfig(max_new_tokens=3, eos_token_id=1)
        runner = DecoderBenchmarkRunner(
            "target_only",
            lambda seed: TargetOnlyDecoder(
                model,
                config,
                rng=random.Random(seed),
            ),
        )
        emitted = []

        generated = runner.generate((0,), seed=7, on_token=emitted.append)

        self.assertEqual(generated, (1,))
        self.assertEqual(emitted, [1])


if __name__ == "__main__":
    unittest.main()
