import unittest

import numpy as np

from hdr_sc import CallableDetectorAdapter, HDRSCConfig, HDRSCSelector


class CallableAdapterTests(unittest.TestCase):
    def test_external_functions_can_drive_selection(self) -> None:
        def prepare(paths):
            return np.arange(len(paths), dtype=np.float64)

        def fit(pool, support):
            return float(pool[list(support)].mean())

        def score(pool, center, evaluation):
            return np.abs(pool[list(evaluation)] - center)

        def map_normal(pool, center, index):
            return np.full((10, 10), abs(pool[index] - center))

        adapter = CallableDetectorAdapter("callable", prepare, fit, score, map_normal)
        result = HDRSCSelector(
            HDRSCConfig(support_size=2, max_candidates=10, seed=1)
        ).select([f"normal-{index}" for index in range(7)], adapter)

        self.assertEqual(result.detector, "callable")
        self.assertEqual(len(result.ranking), 10)
        self.assertIsNotNone(adapter.detector_state)


if __name__ == "__main__":
    unittest.main()

