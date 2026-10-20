"""Sparse track exceedance features for the historical tail penalty."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .intensity_index import PointSampleMatrix, PointTrackMatrix


def _resolve_point_scale(
    synthetic_scale: float | np.ndarray,
    *,
    point_idx: int,
    n_points: int,
) -> float:
    if np.isscalar(synthetic_scale):
        return float(synthetic_scale)
    arr = np.asarray(synthetic_scale, dtype=np.float64).reshape(-1)
    if arr.size != int(n_points):
        raise ValueError(f"synthetic_scale vector must have length {n_points}, got {arr.size}.")
    return float(arr[int(point_idx)])


def build_tail_system_from_active_points(
    *,
    observed: PointSampleMatrix,
    synthetic: PointTrackMatrix,
    active_points: np.ndarray,
    thresholds_kt: Sequence[float],
    synthetic_scale: float | np.ndarray = 1.0,
    min_tau: int = 1,
) -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_meta, edge_row_ids, edge_track_ids, edge_values, tau)."""
    pts = np.asarray(active_points, dtype=np.int32).reshape(-1)
    if pts.size == 0:
        return [], np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float64)
    if int(observed.n_points) != int(synthetic.n_points):
        raise ValueError("Observed and synthetic matrices must share the same n_points for tail features.")

    thresholds = sorted({float(x) for x in thresholds_kt})
    row_meta: List[dict] = []
    row_track_lists: List[np.ndarray] = []
    row_value_lists: List[np.ndarray] = []
    tau_list: List[float] = []

    for p in pts:
        p_int = int(p)
        if p_int < 0 or p_int >= int(observed.n_points):
            continue
        obs_vals = observed.slice_for_point(p_int)
        if obs_vals.size == 0:
            continue
        obs_vals = np.asarray(obs_vals, dtype=np.float64)
        obs_mask = np.isfinite(obs_vals)
        if not np.any(obs_mask):
            continue
        obs_vals = obs_vals[obs_mask]

        track_ids, syn_vals = synthetic.slice_for_point(p_int)
        if syn_vals.size == 0:
            continue
        track_ids = np.asarray(track_ids, dtype=np.int32)
        syn_vals = np.asarray(syn_vals, dtype=np.float64)
        syn_mask = np.isfinite(syn_vals)
        if not np.any(syn_mask):
            continue
        track_ids = track_ids[syn_mask]
        syn_vals = syn_vals[syn_mask]

        scale = _resolve_point_scale(synthetic_scale, point_idx=p_int, n_points=int(synthetic.n_points))

        for thr in thresholds:
            tau_count = int(np.count_nonzero(obs_vals >= thr))
            if tau_count < int(min_tau):
                continue
            exceed = track_ids[syn_vals >= thr]
            if exceed.size == 0:
                # infeasible row (no synthetic support); skip to avoid distorting autoscaling
                continue
            # Track ids should already be unique per point, but keep it safe.
            if exceed.size > 1:
                exceed = np.unique(exceed)
            row_idx = len(row_meta)
            row_meta.append(
                {
                    "row": int(row_idx),
                    "point": int(p_int),
                    "threshold_kt": float(thr),
                    "tau": int(tau_count),
                    "support": int(exceed.size),
                    "scale": float(scale),
                    "observed_samples": int(obs_vals.size),
                }
            )
            row_track_lists.append(exceed.astype(np.int32, copy=False))
            row_value_lists.append(np.full(exceed.size, float(scale), dtype=np.float32))
            tau_list.append(float(tau_count))

    if not row_meta:
        return [], np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float64)

    edge_row_ids = np.concatenate([np.full(arr.size, i, dtype=np.int32) for i, arr in enumerate(row_track_lists)])
    edge_track_ids = np.concatenate(row_track_lists).astype(np.int32, copy=False)
    edge_values = np.concatenate(row_value_lists).astype(np.float32, copy=False)
    tau = np.asarray(tau_list, dtype=np.float64)
    return row_meta, edge_row_ids, edge_track_ids, edge_values, tau
