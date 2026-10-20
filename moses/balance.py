"""Grid intensity-bin features for balanced subset selection."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .intensity_index import PointTrackMatrix


def point_bin_balance(
    synthetic: PointTrackMatrix,
    points: np.ndarray,
    bin_edges: Sequence[float],
) -> np.ndarray:
    """Build per-track per-bin features for selected grid points.

    Each row corresponds to a (grid_point, bin) pair. Values are counts of tracks
    whose max intensity at that grid point falls into the bin. Rows are normalized
    to sum to 1 so they behave like other balance features.
    """
    pts = np.asarray(points, dtype=np.int32)
    if pts.size == 0:
        return np.empty((0, synthetic.n_tracks), dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = max(0, edges.size - 1)
    if n_bins <= 0:
        return np.empty((0, synthetic.n_tracks), dtype=np.float64)

    feats = np.zeros((pts.size * n_bins, synthetic.n_tracks), dtype=np.float64)
    for i, p in enumerate(pts):
        if p < 0 or p >= synthetic.n_points:
            continue
        track_ids, vals = synthetic.slice_for_point(int(p))
        if vals.size == 0:
            continue
        mask = np.isfinite(vals)
        if not np.any(mask):
            continue
        ids = track_ids[mask]
        b_ids = np.digitize(vals[mask], edges, right=False) - 1
        b_ids = np.clip(b_ids, 0, n_bins - 1)
        offset = i * n_bins
        np.add.at(feats, (offset + b_ids, ids), 1.0)

    row_sums = feats.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = np.where(row_sums > 0, feats / row_sums, feats)
    return feats
