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

    def test_demo_runs_without_optional_dependencies(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["demo", "--seed", "7"])
        self.assertEqual(exit_code, 0)
        self.assertIn("generated token ids:", output.getvalue())

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


if __name__ == "__main__":
    unittest.main()
