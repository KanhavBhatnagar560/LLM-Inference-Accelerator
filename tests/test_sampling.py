import random
import unittest

from specdecode.sampling import (
    DistributionError,
    normalize_probabilities,
    residual_distribution,
    sample_categorical,
)


class SamplingTests(unittest.TestCase):
    def test_normalize_probabilities(self) -> None:
        self.assertEqual(normalize_probabilities([2, 3]), (0.4, 0.6))

    def test_invalid_probabilities_are_rejected(self) -> None:
        for probabilities in ([], [0, 0], [0.5, -0.5], [float("nan"), 1.0]):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(DistributionError):
                    normalize_probabilities(probabilities)

    def test_residual_distribution(self) -> None:
        residual = residual_distribution((0.25, 0.75), (0.75, 0.25))
        self.assertEqual(residual, (0.0, 1.0))

    def test_categorical_sampling_is_seeded(self) -> None:
        left = [sample_categorical((0.2, 0.8), random.Random(9)) for _ in range(5)]
        right = [sample_categorical((0.2, 0.8), random.Random(9)) for _ in range(5)]
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()

