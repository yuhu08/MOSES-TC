"""Historical calibration and subset selection for the bundled ACCESS-CM2 data."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .balance import point_bin_balance
from .balanced_sampling import balanced_sample
from .config import Config, ROOT
from .data_loading import iter_ibtracs_tracks, load_prepared_synthetic_tracks
from .grid_utils import SphericalNeighborLookup
from .intensity_index import (
    PointSampleMatrix,
    PointTrackMatrix,
    build_point_samples_from_tracks,
    build_point_track_matrix_from_arrays,
)
from .optimization import _jax_pdhg_solve
from .optimization_relu import (
    TailFeatureSystem,
    compute_lambda_tail_autoscale,
    solve_pdhg_relu,
)
from .ot_utils import (
    HistogramConfig,
    build_cdf_system,
    build_histogram_data,
    compute_global_bin_balance,
)
from .tail_features import build_tail_system_from_active_points
from .util import scale_to_sum_with_cap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inputs(data_dir: Path = ROOT / "data") -> dict:
    """Check bundled input variables and historical coverage without fitting."""
    summary = {}
    for name in ("ibtracs_tracks.nc", "synthetic_ACCESS-CM2.nc"):
        path = data_dir / name
        with xr.open_dataset(path) as ds:
            required = ("lat", "lon", "max_sustained_wind", "track_year")
            if any(key not in ds for key in required):
                raise ValueError(f"Missing required track variables: {name}")
            years = np.asarray(ds["track_year"].values)
            if ds.sizes.get("track", 0) <= 0 or not np.all((years >= 1979) & (years <= 2014)):
                raise ValueError(f"Unexpected historical coverage: {name}")
            summary[name] = {"tracks": int(ds.sizes["track"]), "years": [int(years.min()), int(years.max())]}
    return summary


def build_matrices(config: Config, data_dir: Path, cache_dir: Path, workers: int):
    """Cache only the track-to-grid mappings, keyed by inputs and mapping code."""
    mapping_settings = {
        key: value for key, value in asdict(config).items()
        if key in {"years", "lon_min", "lon_max", "lat_min", "lat_max", "resolution", "radius_km"}
    }
    mapping_settings["inputs"] = {
        name: sha256(data_dir / name) for name in ("ibtracs_tracks.nc", "synthetic_ACCESS-CM2.nc")
    }
    mapping_settings["code"] = {
        name: sha256(Path(__file__).parent / name)
        for name in ("data_loading.py", "grid_utils.py", "intensity_index.py")
    }
    key = hashlib.sha256(json.dumps(mapping_settings, sort_keys=True).encode()).hexdigest()[:16]
    cache = cache_dir / key
    cache.mkdir(parents=True, exist_ok=True)
    observed_path, synthetic_path = cache / "observed.npz", cache / "synthetic.npz"
    grid = config.grid()
    lookup = SphericalNeighborLookup(grid)
    if observed_path.exists():
        observed = PointSampleMatrix.load(observed_path)
    else:
        print("Mapping IBTrACS tracks to the grid...", flush=True)
        tracks = list(iter_ibtracs_tracks(data_dir / "ibtracs_tracks.nc", years=config.years))
        observed = build_point_samples_from_tracks(
            grid, tracks, lookup, config.radius_km, track_count=len(tracks)
        )
        observed.save(observed_path)
    if synthetic_path.exists():
        synthetic = PointTrackMatrix.load(synthetic_path)
    else:
        print("Mapping ACCESS-CM2 tracks to the grid...", flush=True)
        arrays = load_prepared_synthetic_tracks(data_dir / "synthetic_ACCESS-CM2.nc")
        synthetic = build_point_track_matrix_from_arrays(
            grid, arrays["lat"], arrays["lon"], arrays["vmax"], lookup,
            config.radius_km, n_workers=workers,
        )
        synthetic.save(synthetic_path)
    (cache / "metadata.json").write_text(json.dumps(mapping_settings, indent=2) + "\n")
    return observed, synthetic


def select_tracks(observed, synthetic, config: Config):
    """Apply the original raw-grid historical calibration and rounding steps."""
    count = int(observed.total_tracks or 0)
    n_tracks = int(synthetic.n_tracks)
    if count <= 0 or count > n_tracks * config.w_max:
        raise ValueError("Observed track count is infeasible for the weight cap.")
    if not 0 < config.subset_size <= n_tracks:
        raise ValueError("Subset size must be between 1 and the synthetic track count.")
    observed_counts = np.diff(observed.point_ptrs)
    synthetic_counts = np.diff(synthetic.point_ptrs)
    active = np.flatnonzero(
        (observed_counts >= config.min_observed) & (synthetic_counts >= config.min_synthetic)
    )
    if active.size == 0:
        raise ValueError("No grid points meet the event support thresholds.")
    point_weights = observed_counts[active].astype(np.float64)
    histogram = build_histogram_data(
        observed, synthetic, HistogramConfig(bin_edges=config.histogram_bins), synthetic_scale=1.0
    )
    cdf = build_cdf_system(histogram, active, point_weights, n_tracks)

    print(f"Calibrating weights at {active.size} grid points...", flush=True)
    baseline, baseline_history, _, _ = _jax_pdhg_solve(
        cdf, config.lambda_reg, float(count), config.w_max, config.max_iter, config.tolerance
    )
    meta, row_ids, track_ids, values, tau = build_tail_system_from_active_points(
        observed=observed, synthetic=synthetic, active_points=active,
        thresholds_kt=config.tail_thresholds, synthetic_scale=1.0, min_tau=1,
    )
    tail = TailFeatureSystem(n_tracks, int(tau.size), row_ids, track_ids, tau, values, meta)
    lambda_tail = compute_lambda_tail_autoscale(
        cdf_system=cdf, weights_phys=baseline.astype(np.float64), tail_system=tail, match="ot_l1"
    )
    weights_phys, tail_history, _, _ = solve_pdhg_relu(
        cdf_system=cdf, lambda_reg=config.lambda_reg, lambda_tail=lambda_tail,
        target_sum=float(count), w_max=config.w_max, max_iter=config.max_iter,
        tol=config.tolerance, tail_system=tail if tail.n_rows else None, w_init=baseline,
    )
    weights_phys = weights_phys.astype(np.float64)
    if (
        not np.all(np.isfinite(weights_phys))
        or np.any(weights_phys < -1e-6)
        or np.any(weights_phys > config.w_max + 1e-6)
        or not np.isclose(weights_phys.sum(), count, rtol=1e-5)
    ):
        raise RuntimeError("Calibrated weights failed the finite, cap or total-mass check.")
    weights = weights_phys / float(count)

    print("Selecting the balanced subset...", flush=True)
    balance = compute_global_bin_balance(synthetic, config.histogram_bins)
    order = np.argsort(observed_counts[active])[::-1]
    points = active[order[:config.rounding_points]]
    local_rows = point_bin_balance(synthetic, points, config.histogram_bins)
    balance = np.vstack([balance, local_rows])
    probabilities = scale_to_sum_with_cap(weights, config.subset_size)
    if probabilities.sum() < config.subset_size - 1e-3:
        probabilities = scale_to_sum_with_cap(weights + 1e-12, config.subset_size)
    selected = balanced_sample(probabilities, balance, config.random_seed)
    indices = np.flatnonzero(selected)
    if indices.size != config.subset_size:
        raise RuntimeError("Balanced sampling did not return the requested subset size.")

    errors = {}
    evaluations = {
        "original": np.full(n_tracks, float(count) / n_tracks, dtype=np.float32),
        "weighted": weights_phys.astype(np.float32),
        "subset": selected.astype(np.float32) * (float(count) / config.subset_size),
    }
    for name, evaluation in evaluations.items():
        errors[name] = histogram.wasserstein_distances(evaluation, active)
    summary = {
        "observed_tracks": count,
        "synthetic_tracks": n_tracks,
        "selected_tracks": int(indices.size),
        "active_grid_points": int(active.size),
        "weights_sum": float(weights.sum()),
        "weights_phys_sum": float(weights_phys.sum()),
        "physical_weight_cap": float(config.w_max),
        "inclusion_probabilities_sum": float(probabilities.sum()),
        "lambda_tail": float(lambda_tail),
        "tail_rows": tail.n_rows,
        "baseline_objective": baseline_history[-1],
        "tail_objective": tail_history[-1],
        "cdf_error_weighted_mean_kt": {
            name: float(np.average(error, weights=point_weights)) for name, error in errors.items()
        },
    }
    return {
        "weights": weights.astype(np.float32),
        "weights_phys": weights_phys.astype(np.float32),
        "inclusion_probabilities": probabilities,
        "selected_track_ids": indices,
    }, {"point_ids": active, **errors}, summary


def run_demo(config: Config, output_dir: Path, workers: int = 1) -> dict:
    data_dir = ROOT / "data"
    print("Checking the bundled historical inputs...", flush=True)
    inputs = check_inputs(data_dir)
    observed, synthetic = build_matrices(config, data_dir, output_dir / "cache", workers)
    arrays, errors, summary = select_tracks(observed, synthetic, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)
    np.savez_compressed(output_dir / "cdf_errors.npz", **errors)
    summary = {"gcm": "ACCESS-CM2", "period": "historical", "config": asdict(config), "inputs": inputs, **summary}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {summary['selected_tracks']} selected track indices to {output_dir}", flush=True)
    return summary
