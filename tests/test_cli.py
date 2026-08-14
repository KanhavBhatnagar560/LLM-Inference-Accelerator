import contextlib
import io
import unittest
from unittest import mock

from specdecode.cli import build_parser, main
from specdecode.huggingface import MissingOptionalDependencyError


class CliTests(unittest.TestCase):
    def test_generate_arguments_parse_without_loading_optional_dependencies(self) -> None:
        args = build_parser().parse_args(
            [
                "generate",
                "--draft-model",
                "draft/model",
                "--target-model",
                "target/model",
                "--prompt",
                "hello",
            ]
        )
        self.assertEqual(args.draft_model, "draft/model")
        self.assertEqual(args.target_model, "target/model")
        self.assertFalse(args.no_kv_cache)
        self.assertFalse(args.paged_cache_mirror)
        self.assertFalse(args.paged_cache_reference_attention)

    def test_benchmark_arguments_parse_without_loading_optional_dependencies(self) -> None:
        args = build_parser().parse_args(
            [
                "benchmark",
                "--draft-model",
                "draft/model",
                "--target-model",
                "target/model",
                "--prompt",
                "hello",
                "--prompt",
                "world",
            ]
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.prompt, ["hello", "world"])
        self.assertEqual(args.warmup_runs, 2)
        self.assertEqual(args.measured_runs, 10)
        self.assertFalse(args.no_kv_cache)
        self.assertFalse(args.paged_cache_mirror)
        self.assertFalse(args.paged_cache_reference_attention)

    def test_conflicting_cache_modes_fail_before_model_loading(self) -> None:
        error_output = io.StringIO()
        arguments = [
            "generate",
            "--draft-model",
            "draft/model",
            "--target-model",
            "target/model",
            "--prompt",
            "hello",
            "--no-kv-cache",
            "--paged-cache-mirror",
        ]
        with contextlib.redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("cannot be combined", error_output.getvalue())

    def test_paged_reference_attention_requires_mirroring(self) -> None:
        error_output = io.StringIO()
        arguments = [
            "generate",
            "--draft-model",
            "draft/model",
            "--target-model",
            "target/model",
            "--prompt",
            "hello",
            "--paged-cache-reference-attention",
        ]
        with contextlib.redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --paged-cache-mirror", error_output.getvalue())

    def test_demo_runs_without_optional_dependencies(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["demo", "--seed", "7", "--sampling-backend", "python"])
        self.assertEqual(exit_code, 0)
        self.assertIn("generated token ids:", output.getvalue())
        self.assertIn("sampling backend: python", output.getvalue())

    def test_missing_optional_backend_has_actionable_cli_error(self) -> None:
        error_output = io.StringIO()
        arguments = [
            "generate",
            "--draft-model",
            "draft/model",
            "--target-model",
            "target/model",
            "--prompt",
            "hello",
        ]
        with mock.patch(
            "specdecode.cli.run_generate",
            side_effect=MissingOptionalDependencyError("install the transformers extra"),
        ):
            with contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit) as raised:
                    main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("install the transformers extra", error_output.getvalue())

    def test_required_native_backend_reports_missing_library(self) -> None:
        error_output = io.StringIO()
        with contextlib.redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "demo",
                        "--sampling-backend",
                        "native",
                        "--native-library",
                        "/definitely/missing/specdecode.so",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("native library does not exist", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
