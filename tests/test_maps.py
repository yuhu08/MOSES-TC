"""Check plotted quantities against small, hand-calculated examples."""

import unittest

import numpy as np

from moses.data_loading import TrackRecord
from moses.grid_utils import GridSpec
from moses.map_products import intensity_fields, track_density_fields
from test_core import toy_inputs


class MapTests(unittest.TestCase):
    def test_conditional_means(self):
        _, observed, synthetic = toy_inputs()
        active = np.flatnonzero(np.diff(observed.point_ptrs))
        selected = np.zeros(synthetic.n_tracks, dtype=bool)
        selected[[3, 4, 6]] = True  # Winds 70, 90, 120 kt; observations are 65, 95, 125 kt.
        weights = selected.astype(float)
        means = intensity_fields(observed, synthetic, weights, selected, active)
        np.testing.assert_allclose(means["observed"][active], 95.0)
        np.testing.assert_allclose(means["original"][active], 83.75)
        np.testing.assert_allclose(means["subset"][active], 280.0 / 3)
        np.testing.assert_allclose(means["weighted"][active], 280.0 / 3)
        # Conditional quantities are invariant to scaling all physical weights.
        scaled_mean = intensity_fields(observed, synthetic, weights * 17, selected, active)
        np.testing.assert_allclose(scaled_mean["weighted"], means["weighted"], equal_nan=True)

    def test_no_impacts_produce_undefined_mean(self):
        _, observed, synthetic = toy_inputs()
        active = np.flatnonzero(np.diff(observed.point_ptrs))
        means = intensity_fields(
            observed, synthetic, np.ones(synthetic.n_tracks), np.zeros(synthetic.n_tracks, dtype=bool), active
        )
        self.assertTrue(np.isnan(means["subset"][active]).all())

    def test_density_counts_one_track_once_per_box_and_scales_counts(self):
        grid = GridSpec(100.0, 105.0, 10.0, 15.0, 2.5)
        observed = [TrackRecord("o", np.array([10.0, 10.1]), np.array([100.0, 100.1]), np.array([50.0, 70.0]))]
        arrays = {
            "lat": np.array([[10.0, 10.0], [10.1, 12.5], [10.2, 12.5]]),
            "lon": np.array([[100.0, 100.0], [100.1, 102.5], [100.2, 102.5]]),
        }
        fields = track_density_fields(observed, arrays, grid, np.array([0.25, 0.75]), np.array([True, False]), 2)
        # Both synthetic tracks visit box zero; only the second visits box four, twice.
        for name in fields:
            self.assertAlmostEqual(fields[name][0], 0.5)
        self.assertAlmostEqual(fields["original"][4], 0.25)
        self.assertAlmostEqual(fields["weighted"][4], 0.375)
        self.assertAlmostEqual(fields["subset"][4], 0.0)
        self.assertAlmostEqual(fields["observed"].sum(), 0.5)
        self.assertAlmostEqual(fields["original"].sum(), 0.75)


if __name__ == "__main__":
    unittest.main()
