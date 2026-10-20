"""Small artificial-data checks; these do not rerun the scientific case study."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from moses.balanced_sampling import balanced_sample
from moses.config import Config
from moses.data_loading import TrackRecord
from moses.grid_utils import GridSpec, SphericalNeighborLookup
from moses.intensity_index import (
    build_point_samples_from_tracks,
    build_point_track_matrix_from_arrays,
)
from moses.ot_utils import HistogramConfig, build_cdf_system, build_histogram_data
from moses.util import scale_to_sum_with_cap
from moses.workflow import run_demo, select_tracks
from plot_results import plot_results


def toy_inputs():
    config = replace(
        Config(), lon_min=100.0, lon_max=102.0, lat_min=10.0, lat_max=12.0,
        resolution=1.0, subset_size=3, min_observed=1, min_synthetic=1,
        w_max=1.0, rounding_points=1, max_iter=80,
    )
    grid = config.grid()
    lookup = SphericalNeighborLookup(grid)
    tracks = [
        TrackRecord(str(i), np.array([10.0]), np.array([100.0]), np.array([wind]))
        for i, wind in enumerate((65.0, 95.0, 125.0))
    ]
    observed = build_point_samples_from_tracks(grid, tracks, lookup, config.radius_km, track_count=3)
    winds = np.array([[40.0, 50.0, 60.0, 70.0, 90.0, 100.0, 120.0, 140.0]], dtype=np.float32)
    synthetic = build_point_track_matrix_from_arrays(
        grid, np.full_like(winds, 10.0), np.full_like(winds, 100.0), winds, lookup, config.radius_km,
    )
    return config, observed, synthetic


class CoreTests(unittest.TestCase):
    def test_each_track_contributes_its_maximum_once(self):
        grid = GridSpec(100.0, 101.0, 10.0, 11.0, 1.0)
        lookup = SphericalNeighborLookup(grid)
        record = TrackRecord("test", np.array([10.0, 10.0]), np.array([100.0, 100.0]), np.array([50.0, 90.0]))
        observed = build_point_samples_from_tracks(grid, [record], lookup, 250.0, track_count=1)
        self.assertEqual(observed.total_tracks, 1)
        np.testing.assert_array_equal(np.diff(observed.point_ptrs), np.ones(4, dtype=int))
        np.testing.assert_array_equal(observed.intensities, np.full(4, 90.0))

    def test_cdf_forward_and_adjoint_agree(self):
        config, observed, synthetic = toy_inputs()
        histogram = build_histogram_data(observed, synthetic, HistogramConfig(config.histogram_bins))
        active = np.flatnonzero(np.diff(observed.point_ptrs))
        cdf = build_cdf_system(histogram, active, np.diff(observed.point_ptrs)[active], synthetic.n_tracks)
        rng = np.random.default_rng(11)
        weights = rng.random(synthetic.n_tracks)
        dual = rng.standard_normal(cdf.n_rows)
        self.assertAlmostEqual(float(cdf.apply_A(weights) @ dual), float(weights @ cdf.adjoint(dual)), places=10)
        # Equal count in each target bin gives zero CDF error.
        weights = np.zeros(synthetic.n_tracks)
        weights[[3, 4, 6]] = 1.0
        np.testing.assert_allclose(histogram.wasserstein_distances(weights, active), 0.0)

    def test_inclusion_probabilities_preserve_total_with_saturation(self):
        probabilities = scale_to_sum_with_cap(np.array([0.9, 0.05, 0.03, 0.02, 0.0]), 3)
        self.assertAlmostEqual(float(probabilities.sum()), 3.0, places=6)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertEqual(probabilities[-1], 0.0)
        self.assertEqual(probabilities[0], 1.0)

    def test_balanced_sampling_is_exact_and_repeatable(self):
        probabilities = np.full(40, 0.25)
        balance = np.array([np.linspace(0.0, 1.0, 40)])
        first = balanced_sample(probabilities, balance, random_state=7)
        second = balanced_sample(probabilities, balance, random_state=7)
        self.assertEqual(int(first.sum()), 10)
        np.testing.assert_array_equal(first, second)

    def test_calibration_and_selection_on_artificial_tracks(self):
        config, observed, synthetic = toy_inputs()
        arrays, errors, summary = select_tracks(observed, synthetic, config)
        self.assertEqual(len(np.unique(arrays["selected_track_ids"])), config.subset_size)
        self.assertAlmostEqual(float(arrays["weights"].sum()), 1.0, places=5)
        self.assertTrue(np.all(np.isfinite(arrays["weights_phys"])))
        self.assertLessEqual(float(arrays["weights_phys"].max()), config.w_max + 1e-6)
        self.assertLess(summary["cdf_error_weighted_mean_kt"]["weighted"], summary["cdf_error_weighted_mean_kt"]["original"])
        self.assertEqual(errors["point_ids"].size, summary["active_grid_points"])

    def test_infeasible_subset_is_rejected(self):
        config, observed, synthetic = toy_inputs()
        with self.assertRaisesRegex(ValueError, "Subset size"):
            select_tracks(observed, synthetic, replace(config, subset_size=9))

    def test_file_workflow_and_cache_in_an_independent_directory(self):
        config, _, _ = toy_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for name, winds in (
                ("ibtracs_tracks.nc", [65.0, 95.0, 125.0]),
                ("synthetic_ACCESS-CM2.nc", [40.0, 50.0, 60.0, 70.0, 90.0, 100.0, 120.0, 140.0]),
            ):
                wind = np.asarray([winds], dtype=np.float32)
                tracks = xr.Dataset(
                    {
                        "lat": (("time", "track"), np.full_like(wind, 10.0)),
                        "lon": (("time", "track"), np.full_like(wind, 100.0)),
                        "max_sustained_wind": (("time", "track"), wind),
                        "track_year": ("track", np.full(len(winds), 1990)),
                    },
                    coords={"time": [0], "track": [f"1990_track_{i}" for i in range(len(winds))]},
                    attrs={"max_sustained_wind_unit": "kn"},
                )
                tracks.to_netcdf(data / name)
            source = Path(__file__).resolve().parents[1]
            shutil.copyfile(source / "data/coastline_110m.npz", data / "coastline_110m.npz")
            (root / "moses").mkdir()
            for name in ("data_loading.py", "grid_utils.py", "intensity_index.py"):
                shutil.copyfile(source / "moses" / name, root / "moses" / name)
            output = root / "results"
            with patch("moses.workflow.ROOT", root), patch("plot_results.ROOT", root):
                run_demo(config, output)
                figures = plot_results(output)
                cache = sorted((output / "cache").rglob("*.npz"))
                timestamps = [path.stat().st_mtime_ns for path in cache]
                selected = np.load(output / "selected_track_ids.npy")
                run_demo(config, output)
            self.assertEqual(len(cache), 2)
            self.assertEqual(timestamps, [path.stat().st_mtime_ns for path in cache])
            np.testing.assert_array_equal(selected, np.load(output / "selected_track_ids.npy"))
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(summary["selected_tracks"], config.subset_size)
            self.assertTrue((output / "cdf_errors.npz").is_file())
            self.assertEqual({path.name for path in figures}, {"track_density.png", "mean_tc_intensity.png"})
            self.assertTrue(all(path.is_file() for path in figures))


if __name__ == "__main__":
    unittest.main()
