import os
import random
import unittest
from unittest import mock

from specdecode import DecodeConfig, SpeculativeDecoder, TableModel
from specdecode.backends import PythonSamplingBackend, load_sampling_backend
from specdecode.native import NativeLibraryNotFound


class RecordingBackend(PythonSamplingBackend):
    def __init__(self) -> None:
        self.categorical_uniforms = []
        self.acceptance_calls = 0

    def categorical(self, probabilities, uniform):
        self.categorical_uniforms.append(uniform)
        return super().categorical(probabilities, uniform)

    def acceptance_probabilities(self, target_rows, draft_rows, token_ids):
        self.acceptance_calls += 1
        return super().acceptance_probabilities(target_rows, draft_rows, token_ids)


class SequenceRandom(random.Random):
    def __init__(self, values):
        super().__init__(0)
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


class BackendTests(unittest.TestCase):
    def test_decoder_defaults_to_stable_python_backend(self) -> None:
        model = TableModel({}, default=(1.0, 0.0))

        decoder = SpeculativeDecoder(model, model)

        self.assertEqual(decoder.sampling_backend.name, "python")

    def test_python_backend_primitives(self) -> None:
        backend = PythonSamplingBackend()
        self.assertEqual(backend.categorical((2.0, 3.0), 0.0), 0)
        self.assertEqual(backend.categorical((2.0, 3.0), 0.4), 1)
        self.assertEqual(
            backend.residual_weights((0.25, 0.75), (0.75, 0.25)),
            (0.0, 0.5),
        )
        self.assertEqual(
            backend.acceptance_probabilities(
                ((0.2, 0.8), (0.7, 0.3)),
                ((0.4, 0.6), (0.2, 0.8)),
                (0, 1),
            ),
            (0.5, 0.37499999999999994),
        )
        self.assertEqual(backend.first_rejection((0.5, 1.0), (0.5, 0.1)).rejection_index, 0)

    def test_auto_mode_falls_back_only_when_library_is_absent(self) -> None:
        with mock.patch.dict(os.environ, {"SPECDECODE_NATIVE_LIBRARY": ""}):
            with mock.patch(
                "specdecode.native.NativeSamplingBackend.load",
                side_effect=NativeLibraryNotFound("missing"),
            ):
                backend = load_sampling_backend("auto")
        self.assertEqual(backend.name, "python")

    def test_explicit_missing_library_does_not_silently_fallback(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SPECDECODE_SAMPLING_BACKEND": "python"},
        ):
            with self.assertRaises(NativeLibraryNotFound):
                load_sampling_backend(
                    "auto",
                    library_path="/definitely/missing/specdecode.so",
                )

    def test_environment_can_force_python_backend(self) -> None:
        with mock.patch.dict(os.environ, {"SPECDECODE_SAMPLING_BACKEND": "python"}):
            self.assertEqual(load_sampling_backend().name, "python")

    def test_decoder_keeps_rng_ownership_and_draw_order(self) -> None:
        model = TableModel({}, default=(1.0, 0.0))
        backend = RecordingBackend()
        rng = SequenceRandom((0.1, 0.2, 0.3))
        decoder = SpeculativeDecoder(
            model,
            model,
            DecodeConfig(
                max_new_tokens=2,
                initial_draft_tokens=1,
                min_draft_tokens=1,
                max_draft_tokens=1,
                dynamic_draft=False,
            ),
            rng=rng,
            sampling_backend=backend,
        )

        result = decoder.generate([0])

        self.assertEqual(result.generated_tokens, (0, 0))
        self.assertEqual(rng.calls, 3)
        self.assertEqual(backend.categorical_uniforms, [0.1, 0.3])
        self.assertEqual(backend.acceptance_calls, 1)


if __name__ == "__main__":
    unittest.main()
