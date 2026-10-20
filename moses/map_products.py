"""Track density and conditional mean intensity maps."""

from __future__ import annotations

import numpy as np

from .grid_utils import GridSpec


def visited_cells(lat, lon, grid: GridSpec) -> np.ndarray:
    """Return distinct grid boxes entered by one track."""
    valid = np.isfinite(lat) & np.isfinite(lon)
    lon_edges = np.r_[grid.lon - grid.resolution / 2, grid.lon[-1] + grid.resolution / 2]
    lat_edges = np.r_[grid.lat - grid.resolution / 2, grid.lat[-1] + grid.resolution / 2]
    ix = np.digitize(np.mod(lon[valid], 360.0), lon_edges) - 1
    iy = np.digitize(lat[valid], lat_edges) - 1
    inside = (ix >= 0) & (ix < grid.lon.size) & (iy >= 0) & (iy < grid.lat.size)
    return np.unique(iy[inside] * grid.lon.size + ix[inside])


def track_density_fields(observed_tracks, synthetic_arrays, grid, weights_phys, selected, years):
    """Annual unique-track density with the same physical count in each sample."""
    n_observed = len(observed_tracks)
    n_synthetic = synthetic_arrays["lat"].shape[1]
    weights = np.asarray(weights_phys, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if weights.size != n_synthetic or selected.size != n_synthetic or not selected.any():
        raise ValueError("Track weights and selection must match the synthetic track dimension.")
    if years <= 0 or weights.sum() <= 0:
        raise ValueError("Years and total physical weight must be positive.")
    weights = weights * (n_observed / weights.sum())
    original_scale = n_observed / n_synthetic
    subset_scale = n_observed / int(selected.sum())
    fields = {name: np.zeros(grid.n_points) for name in ("observed", "original", "weighted", "subset")}
    for track in observed_tracks:
        fields["observed"][visited_cells(track.lat, track.lon, grid)] += 1
    for j in range(n_synthetic):
        cells = visited_cells(synthetic_arrays["lat"][:, j], synthetic_arrays["lon"][:, j], grid)
        fields["original"][cells] += original_scale
        fields["weighted"][cells] += weights[j]
        if selected[j]:
            fields["subset"][cells] += subset_scale
    return {name: values / years for name, values in fields.items()}


def intensity_fields(observed, synthetic, weights_phys, selected, active_points):
    """Conditional means of per-track maximum impact intensities.

    Each mean is normalized by its own impact mass. Undefined means remain NaN
    when a sample has no positively weighted impacts.
    """
    names = ("original", "weighted", "subset")
    means = {name: np.full(observed.n_points, np.nan) for name in ("observed", *names)}
    weights_phys = np.asarray(weights_phys, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if weights_phys.size != synthetic.n_tracks or selected.size != synthetic.n_tracks:
        raise ValueError("Track weights and selection must match the synthetic matrix.")
    for point in active_points:
        target = observed.slice_for_point(int(point))
        target = target[np.isfinite(target)]
        if not target.size:
            continue
        means["observed"][point] = target.mean()
        ids, winds = synthetic.slice_for_point(int(point))
        local_weights = {
            "original": np.ones(ids.size),
            "weighted": weights_phys[ids],
            "subset": selected[ids].astype(float),
        }
        for name, weights in local_weights.items():
            keep = np.isfinite(winds) & np.isfinite(weights) & (weights > 0)
            if not keep.any():
                continue
            values, mass = winds[keep].astype(float), weights[keep]
            means[name][point] = np.average(values, weights=mass)
    return means
