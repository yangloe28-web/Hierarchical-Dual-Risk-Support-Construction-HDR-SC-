import csv
import json
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

from hdr_sc import HDRSCConfig, HDRSCSelector, spatial_dispersion
from hdr_sc.io import save_selection


class ControlledAdapter:
    name = "controlled-synthetic"

    def __init__(self, q):
        self.q = q
        self.map_calls = []
        self.last_map = None

    def prepare(self, paths):
        pass

    def fit(self, support):
        self.support = support

    def score(self, indices):
        assert self.support[0] not in indices
        return [self.q[self.support[0]] / 2] * len(indices)

    def anomaly_map(self, index):
        assert index not in self.support
        assert self.last_map is None or self.last_map() is None
        self.map_calls.append((self.support[0], index))
        value = np.ones((10, 10))
        if self.support == (1,):
            value[:] = 0
            value[0, 0] = 1
        self.last_map = weakref.ref(value)
        return value


class HDRSCTests(unittest.TestCase):
    def select(self, q):
        adapter = ControlledAdapter(q)
        result = HDRSCSelector(HDRSCConfig(1)).select(
            [str(i) for i in range(len(q))], adapter
        )
        return result, adapter

    def test_gate_and_spatial_winner(self):
        result, adapter = self.select([1, 1.1, 2, 3])
        self.assertAlmostEqual(result.median_q, 1.55)
        self.assertAlmostEqual(result.mad_q, .5)
        self.assertAlmostEqual(result.epsilon, .25)
        self.assertEqual(result.omega_size, 2)
        self.assertEqual(result.selected.support_indices, (1,))
        self.assertEqual(result.selected.rank, 2)
        self.assertAlmostEqual(result.selected.spatial_risk, .02)
        self.assertEqual(adapter.support, (1,))
        self.assertEqual({i for i, _ in adapter.map_calls}, {0, 1})
        self.assertEqual(len(adapter.map_calls), 6)
        self.assertTrue(all(r.spatial_risk is None for r in result.ranking[2:]))
        self.assertIsNone(adapter.last_map())

    def test_singleton_never_requests_maps(self):
        result, adapter = self.select([0, 10, 11, 12])
        self.assertEqual(result.selected.support_indices, (0,))
        self.assertEqual(result.omega_size, 1)
        self.assertIsNone(result.selected.spatial_risk)
        self.assertEqual(adapter.map_calls, [])

    def test_zero_mad_retains_tied_minima(self):
        result, _ = self.select([1, 1, 1, 1])
        self.assertEqual(result.epsilon, 0)
        self.assertEqual(result.omega_size, 4)
        self.assertEqual(result.selected.support_indices, (1,))

    def test_spatial_tie_uses_candidate_id(self):
        result, _ = self.select([1, 10, 1, 1])
        self.assertEqual(result.selected.support_indices, (0,))

    def test_gate_includes_boundary(self):
        result, _ = self.select([0, .5, 2, 3])
        self.assertAlmostEqual(result.epsilon, .5)
        self.assertEqual(result.omega_size, 2)

    def test_export_contains_selection_audit(self):
        result, _ = self.select([1, 1.1, 2, 3])
        with tempfile.TemporaryDirectory() as directory:
            save_selection(result, directory)
            payload = json.loads((Path(directory) / 'selected_support.json').read_text())
            self.assertEqual(payload['method'], 'HDR-SC')
            self.assertEqual(payload['selected_Q_img_rank'], 2)
            self.assertEqual(payload['omega_size'], 2)
            with (Path(directory) / 'candidate_ranking.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(r['selected'] == 'True' for r in rows), 1)
            self.assertEqual(rows[2]['R_spatial'], '')

    def test_spatial_map_cases(self):
        self.assertEqual(spatial_dispersion(np.zeros((2, 2))), 0)
        self.assertEqual(spatial_dispersion(np.ones((2, 2))), 1)
        sparse = np.zeros((10, 10))
        sparse[0, 0] = 100
        self.assertAlmostEqual(spatial_dispersion(sparse), .01)
        self.assertAlmostEqual(spatial_dispersion(sparse * 3), .01)
        values = np.arange(101).reshape(1, 101)
        self.assertAlmostEqual(spatial_dispersion(values), 50 / 99.5)

    def test_invalid_maps_and_configuration(self):
        for value in [np.ones(3), np.zeros((0, 0)), np.array([[np.nan]]), np.array([[-1]])]:
            with self.assertRaises(ValueError):
                spatial_dispersion(value)
        for name in ['epsilon_weight', 'beta']:
            for value in [np.nan, np.inf, -1]:
                with self.assertRaises(ValueError):
                    HDRSCConfig(1, **{name: value})
        with self.assertRaises(ValueError):
            self.select([1])


if __name__ == '__main__':
    unittest.main()
