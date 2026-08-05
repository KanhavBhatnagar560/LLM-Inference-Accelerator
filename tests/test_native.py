import math
import random
import unittest

from specdecode import DecodeConfig, SpeculativeDecoder, TableModel
from specdecode.backends import PythonSamplingBackend
from specdecode.native import (
    NativeBackendError,
    NativeLibraryNotFound,
    NativeSamplingBackend,
)


def load_native():
    try:
        return NativeSamplingBackend.load()
    except NativeLibraryNotFound:
        return None


NATIVE = load_native()


@unittest.skipUnless(NATIVE is not None, "native library has not been built")
class NativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.native = NATIVE
        cls.python = PythonSamplingBackend()

    def assertVectorClose(self, left, right, tolerance=1.0e-12):
        self.assertEqual(len(left), len(right))
        for left_value, right_value in zip(left, right):
            self.assertAlmostEqual(left_value, right_value, delta=tolerance)

    def test_normalization_and_categorical_parity(self) -> None:
        weights = (0.0, 2.0, 3.0, 5.0, 0.0)
        normalized = self.native.normalize(weights)
        self.assertVectorClose(normalized, (0.0, 0.2, 0.3, 0.5, 0.0))
        for uniform in (0.0, 0.19, 0.2, 0.499, 0.5, math.nextafter(1.0, 0.0)):
            with self.subTest(uniform=uniform):
                self.assertEqual(
                    self.native.categorical(weights, uniform),
                    self.python.categorical(weights, uniform),
                )

    def test_residual_and_acceptance_parity(self) -> None:
        target_rows = (
            (0.2, 0.5, 0.3),
            (0.2, 0.6, 0.2),
            (0.0, 0.0, 1.0),
        )
        draft_rows = (
            (0.4, 0.4, 0.2),
            (0.6, 0.2, 0.2),
            (0.0, 1.0, 0.0),
        )
        tokens = (0, 1, 2)
        self.assertVectorClose(
            self.native.acceptance_probabilities(target_rows, draft_rows, tokens),
            self.python.acceptance_probabilities(target_rows, draft_rows, tokens),
        )
        self.assertVectorClose(
            self.native.residual_weights(target_rows[0], draft_rows[0]),
            self.python.residual_weights(target_rows[0], draft_rows[0]),
        )
        self.assertVectorClose(
            self.native.residual_weights(target_rows[2], target_rows[2]),
            self.python.residual_weights(target_rows[2], target_rows[2]),
        )

    def test_first_rejection_uses_strict_comparison(self) -> None:
        probabilities = (0.0, 0.5, 1.0)
        uniforms = (0.0, 0.1, 0.1)
        self.assertEqual(
            self.native.first_rejection(probabilities, uniforms),
            self.python.first_rejection(probabilities, uniforms),
        )

    def test_native_validation_errors_are_not_hidden(self) -> None:
        with self.assertRaises(NativeBackendError):
            self.native.categorical((0.5, -0.5), 0.1)
        with self.assertRaises(NativeBackendError):
            self.native.residual_weights((0.5, 0.5), (1.0,))
        with self.assertRaises(NativeBackendError):
            self.native.acceptance_probabilities(((1.0, 0.0),), (), (0,))

    def test_decoder_tokens_events_stats_and_rng_state_match(self) -> None:
        draft = TableModel(
            {(0,): (0.05, 0.70, 0.20, 0.05), (1,): (0.05, 0.20, 0.65, 0.10)},
            default=(0.05, 0.10, 0.35, 0.50),
        )
        target = TableModel(
            {(0,): (0.05, 0.55, 0.30, 0.10), (1,): (0.05, 0.15, 0.60, 0.20)},
            default=(0.05, 0.10, 0.25, 0.60),
        )
        config = DecodeConfig(max_new_tokens=12, eos_token_id=3)
        python_rng = random.Random(7)
        native_rng = random.Random(7)
        python_events = []
        native_events = []

        python_result = SpeculativeDecoder(
            draft,
            target,
            config,
            rng=python_rng,
            sampling_backend=self.python,
        ).generate([0], on_token=python_events.append)
        native_result = SpeculativeDecoder(
            draft,
            target,
            config,
            rng=native_rng,
            sampling_backend=self.native,
        ).generate([0], on_token=native_events.append)

        self.assertEqual(native_result, python_result)
        self.assertEqual(native_events, python_events)
        self.assertEqual(native_rng.getstate(), python_rng.getstate())


if __name__ == "__main__":
    unittest.main()
