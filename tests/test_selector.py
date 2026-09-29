from __future__ import annotations

import unittest
from typing import Sequence

import numpy as np

from hdr_sc import HDRSCConfig, HDRSCSelector


class RecordingAdapter:
    name = "recording"

    def __init__(self) -> None:
        self.values: np.ndarray | None = None
        self.support: tuple[int, ...] | None = None
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def prepare(self, pool_paths: Sequence[str]) -> None:
        self.values = np.arange(len(pool_paths), dtype=np.float64)

    def fit(self, support_indices: tuple[int, ...]) -> None:
        self.support = support_indices

    def score(self, evaluation_indices: tuple[int, ...]) -> np.ndarray:
        assert self.values is not None
        assert self.support is not None
        assert not set(self.support).intersection(evaluation_indices)
        self.calls.append((self.support, evaluation_indices))
        center = self.values[list(self.support)].mean()
        return np.abs(self.values[list(evaluation_indices)] - center)

    def anomaly_map(self, evaluation_index: int) -> np.ndarray:
        assert evaluation_index not in self.support
        return np.ones((10, 10))


class SelectorTests(unittest.TestCase):
    def test_selector_is_deterministic_and_refits_winner(self) -> None:
        paths = [f"normal-{index}" for index in range(8)]
        first_adapter = RecordingAdapter()
        second_adapter = RecordingAdapter()
        config = HDRSCConfig(support_size=2, max_candidates=12, seed=3)

        first = HDRSCSelector(config).select(paths, first_adapter)
        second = HDRSCSelector(config).select(paths, second_adapter)

        self.assertEqual(
            first.selected.support_indices, second.selected.support_indices
        )
        self.assertEqual(first_adapter.support, first.selected.support_indices)
        self.assertEqual(len(first.ranking), 12)
        self.assertEqual(
            [row.rank for row in first.ranking], list(range(1, 13))
        )
        self.assertTrue(
            all(
                not set(support).intersection(evaluation)
                for support, evaluation in first_adapter.calls
            )
        )


if __name__ == "__main__":
    unittest.main()

