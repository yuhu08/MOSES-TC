"""Histogram and sparse CDF operators for marginal distribution matching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .config import BIN_EDGES
from .intensity_index import PointSampleMatrix, PointTrackMatrix

@dataclass
class HistogramConfig:
    """Intensity bins and optional CDF cost multipliers."""

    bin_edges: Sequence[float] = BIN_EDGES

    # Optional per-bin multipliers applied to the Wasserstein-1/CDF mismatch cost.
    # Use this to emphasize specific intensity ranges (e.g. high-intensity tail).
    # Must have length n_bins = len(bin_edges) - 1 when provided.
    bin_cost_weights: Sequence[float] | None = None

    @property
    def n_bins(self) -> int:
        return max(0, len(self.bin_edges) - 1)

    @property
    def cost_weights(self) -> np.ndarray:
        """Return per-bin cost multipliers (defaults to ones)."""
        n_bins = self.n_bins
        if n_bins <= 0:
            return np.zeros(0, dtype=np.float64)
        if self.bin_cost_weights is None:
            return np.ones(n_bins, dtype=np.float64)
        arr = np.asarray(self.bin_cost_weights, dtype=np.float64)
        if arr.size != n_bins:
            raise ValueError(
                f"bin_cost_weights has length {arr.size}, expected {n_bins} (= len(bin_edges)-1)."
            )
        if np.any(arr < 0):
            raise ValueError("bin_cost_weights must be non-negative.")
        return arr

    @property
    def bin_centers(self) -> np.ndarray:
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def bin_widths(self) -> np.ndarray:
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        return np.diff(edges)


@dataclass
class CdfGraphSystem:
    """
    Graph-based linear system for per-point CDF matching.

    Rows correspond to (point, bin) pairs; columns correspond to tracks.
    Coefficients are stored per edge (point-track pair) and per bin, and
    applied via scatter/gather instead of dense matrix multiplication.
    """

    n_tracks: int
    n_bins: int
    point_ids: np.ndarray  # shape (n_points_active,)
    point_edge_ptrs: np.ndarray  # shape (n_points_active + 1,)
    edge_track_ids: np.ndarray  # shape (n_edges,)
    edge_cdf_values: np.ndarray  # shape (n_edges, n_bins)
    row_weights: np.ndarray  # shape (n_rows,)
    b: np.ndarray  # shape (n_rows,)
    metadata: Optional[dict] = None
    _edge_points: Optional[np.ndarray] = None

    @property
    def n_rows(self) -> int:
        return int(self.point_ids.size * self.n_bins)

    @property
    def edge_point_ids(self) -> np.ndarray:
        """Edge-to-point mapping for scatter operations."""
        if self._edge_points is None:
            counts = np.diff(self.point_edge_ptrs)
            self._edge_points = np.repeat(np.arange(self.point_ids.size, dtype=np.int32), counts)
        return self._edge_points

    def apply_A(self, weights: np.ndarray) -> np.ndarray:
        """Compute A @ weights via scatter-add (no b subtraction)."""
        if self.n_rows == 0:
            return np.zeros(0, dtype=np.float64)
        if self.edge_track_ids.size == 0:
            return np.zeros(self.n_rows, dtype=np.float64)
        edge_vals = (
            weights[self.edge_track_ids].astype(np.float64, copy=False)[:, None] * self.edge_cdf_values
        )
        accum = np.zeros((self.point_ids.size, self.n_bins), dtype=np.float64)
        np.add.at(accum, self.edge_point_ids, edge_vals)
        return accum.reshape(-1)

    def forward(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute A @ weights - b using scatter-add on the bipartite graph.
        """
        return self.apply_A(weights) - self.b

    def adjoint(self, dual: np.ndarray) -> np.ndarray:
        """
        Compute A.T @ dual using edge-wise scatter-add.
        """
        if dual.size != self.n_rows:
            raise ValueError(f"dual has size {dual.size}, expected {self.n_rows}")
        if self.edge_track_ids.size == 0:
            return np.zeros(self.n_tracks, dtype=np.float64)
        dual_mat = dual.reshape(self.point_ids.size, self.n_bins)
        edge_dual = dual_mat[self.edge_point_ids]
        edge_grad = np.sum(edge_dual * self.edge_cdf_values, axis=1)
        grad = np.zeros(self.n_tracks, dtype=np.float64)
        np.add.at(grad, self.edge_track_ids, edge_grad)
        return grad


@dataclass
class PointHistogramData:
    """Per-point histogram representations and synthetic contributions."""

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_widths: np.ndarray
    bin_cost_weights: np.ndarray
    observed_hist: np.ndarray  # shape (n_points, n_bins)
    observed_cdf: np.ndarray  # normalized CDFs, shape (n_points, n_bins)
    observed_mass: np.ndarray  # total observed counts per point
    observed_total_tracks: int | None
    point_track_ids: List[np.ndarray]
    point_bin_contribs: List[np.ndarray]  # list of (n_tracks_local, n_bins)
    point_edge_ptrs: np.ndarray  # shape (n_points + 1,)
    edge_track_ids: np.ndarray  # shape (n_edges,)
    edge_bin_contribs: np.ndarray  # shape (n_edges, n_bins)
    edge_point_ids: np.ndarray  # shape (n_edges,)

    def synthetic_histogram_for_point(
        self, weights: np.ndarray, point_idx: int
    ) -> np.ndarray:
        start = int(self.point_edge_ptrs[point_idx])
        end = int(self.point_edge_ptrs[point_idx + 1])
        if start == end:
            return np.zeros(self.bin_centers.size, dtype=np.float64)
        track_ids = self.edge_track_ids[start:end]
        contribs = self.edge_bin_contribs[start:end]
        local_weights = weights[track_ids].astype(np.float64, copy=False)
        # Each row holds bin counts contributed by a single track.
        return np.dot(local_weights, contribs)

    def _scatter_histograms(self, weights: np.ndarray) -> np.ndarray:
        """Compute synthetic histograms for all points via edge scatter-add."""
        n_points = self.observed_hist.shape[0]
        hists = np.zeros((n_points, self.bin_centers.size), dtype=np.float64)
        if self.edge_track_ids.size == 0:
            return hists
        edge_weights = weights[self.edge_track_ids].astype(np.float64, copy=False)[:, None]
        contrib = edge_weights * self.edge_bin_contribs
        np.add.at(hists, self.edge_point_ids, contrib)
        return hists

    def synthetic_histograms(
        self, weights: np.ndarray, points: Iterable[int]
    ) -> np.ndarray:
        points = np.asarray(points, dtype=int)
        hists = np.zeros((points.size, self.bin_centers.size), dtype=np.float64)
        if points.size == 0:
            return hists
        total_weight = float(np.sum(weights))
        # For larger batches, a single scatter over edges is faster and memory-friendly.
        if points.size > 32:
            full = self._scatter_histograms(weights)
            hists = full[points]
        else:
            for idx, point_idx in enumerate(points):
                hists[idx] = self.synthetic_histogram_for_point(weights, int(point_idx))
        return hists

    def synthetic_cdfs(
        self, weights: np.ndarray, points: Iterable[int], target_masses: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return normalized cumulative histograms for selected points."""
        points = np.asarray(points, dtype=int)
        hists = self.synthetic_histograms(weights, points)
        if target_masses is not None:
            masses = target_masses.reshape(-1, 1)
            masses = np.where(masses <= 0, 1.0, masses)
        else:
            masses = hists.sum(axis=1, keepdims=True)
            masses = np.where(masses <= 0, 1.0, masses)
        return np.cumsum(hists, axis=1) / masses

    def wasserstein_distances(
        self,
        weights: np.ndarray,
        points: Iterable[int],
    ) -> np.ndarray:
        """Compute W1 distances for selected points using CDF formulation."""
        points = np.asarray(points, dtype=int)
        n_bins = self.bin_centers.size
        if n_bins == 0 or points.size == 0:
            return np.zeros(points.size, dtype=np.float64)
        bin_costs = self.bin_widths * self.bin_cost_weights
        target_masses = self.observed_mass[points]
        synthetic_cdf = self.synthetic_cdfs(weights, points, target_masses=target_masses)
        observed_cdf = self.observed_cdf[points]
        dists = np.zeros(points.size, dtype=np.float64)
        for idx in range(points.size):
            dists[idx] = wasserstein_1d_from_cdf(
                synthetic_cdf[idx], observed_cdf[idx], bin_costs
            )
        return dists


def build_cdf_system(
    histogram_data: PointHistogramData,
    active_points: np.ndarray,
    point_weights: np.ndarray,
    n_tracks: int,
) -> CdfGraphSystem:
    """Build the sparse CDF system for the active grid points."""
    bin_widths = histogram_data.bin_widths.copy()
    bin_cost_weights = np.asarray(histogram_data.bin_cost_weights, dtype=np.float64)
    if bin_cost_weights.size != bin_widths.size:
        raise ValueError(
            f"bin_cost_weights has length {bin_cost_weights.size}, expected {bin_widths.size}."
        )
    bin_costs = bin_widths * bin_cost_weights
    n_bins = bin_widths.size

    active_points = np.asarray(active_points, dtype=int)
    if active_points.size == 0:
        raise ValueError("active_points cannot be empty.")
    weight_sum = float(point_weights.sum())
    norm_weights = point_weights / max(weight_sum, 1e-8)

    point_ids: List[int] = []
    point_edge_ptrs: List[int] = [0]
    edge_tracks: List[np.ndarray] = []
    edge_cdf_blocks: List[np.ndarray] = []
    b_vals: List[float] = []
    row_weights_arr: List[float] = []
    for local_idx, point_idx in enumerate(active_points):
        obs_mass = float(histogram_data.observed_mass[point_idx])
        if obs_mass <= 0:
            continue
        start = int(histogram_data.point_edge_ptrs[point_idx])
        end = int(histogram_data.point_edge_ptrs[point_idx + 1])
        if start == end:
            continue
        tracks = histogram_data.edge_track_ids[start:end]
        contribs = histogram_data.edge_bin_contribs[start:end]
        if tracks.size == 0 or contribs.size == 0:
            continue
        cum = np.cumsum(contribs, axis=1) / obs_mass  # shape (n_tracks_local, n_bins)

        point_ids.append(int(point_idx))
        edge_tracks.append(tracks)
        edge_cdf_blocks.append(cum.astype(np.float32, copy=False))
        point_edge_ptrs.append(point_edge_ptrs[-1] + tracks.size)

        # Row weights for CDF bins
        row_weights_arr.extend((norm_weights[local_idx] * bin_costs).tolist())
        b_vals.extend(histogram_data.observed_cdf[point_idx].tolist())

    point_ids_arr = np.asarray(point_ids, dtype=np.int32)
    ptrs_arr = np.asarray(point_edge_ptrs, dtype=np.int64)
    if edge_tracks:
        edge_track_ids = np.concatenate(edge_tracks).astype(np.int32, copy=False)
        edge_cdf_values = np.vstack(edge_cdf_blocks)
    else:
        edge_track_ids = np.empty(0, dtype=np.int32)
        edge_cdf_values = np.empty((0, n_bins), dtype=np.float32)
    row_weights = np.asarray(row_weights_arr, dtype=np.float32)
    b_vec = np.asarray(b_vals, dtype=np.float32)
    return CdfGraphSystem(
        n_tracks=n_tracks,
        n_bins=n_bins,
        point_ids=point_ids_arr,
        point_edge_ptrs=ptrs_arr,
        edge_track_ids=edge_track_ids,
        edge_cdf_values=edge_cdf_values,
        row_weights=row_weights,
        b=b_vec,
        metadata=None,
    )


def build_histogram_data(
    observed: PointSampleMatrix,
    synthetic: PointTrackMatrix,
    config: HistogramConfig,
    synthetic_scale: float | np.ndarray = 1.0,
) -> PointHistogramData:
    """Precompute observed histograms and synthetic bin contributions."""
    bin_edges = np.asarray(config.bin_edges, dtype=np.float64)
    n_bins = bin_edges.size - 1
    if n_bins <= 0:
        raise ValueError("Histogram configuration must include at least two bin edges.")
    bin_cost_weights = config.cost_weights

    # Observed histograms (counts) per point.
    obs_hist = np.zeros((observed.n_points, n_bins), dtype=np.float64)
    obs_ptrs = observed.point_ptrs
    obs_vals = observed.intensities
    obs_cdf = np.zeros((observed.n_points, n_bins), dtype=np.float64)
    obs_mass = np.zeros(observed.n_points, dtype=np.float64)
    obs_total_tracks = observed.total_tracks if observed.total_tracks and observed.total_tracks > 0 else None
    for point_idx in range(observed.n_points):
        start = int(obs_ptrs[point_idx])
        end = int(obs_ptrs[point_idx + 1])
        if start == end:
            continue
        values = obs_vals[start:end]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        hist, _ = np.histogram(values, bins=bin_edges)
        obs_hist[point_idx] = hist.astype(np.float64)
        mass = float(hist.sum())
        obs_mass[point_idx] = mass
        if mass > 0:
            obs_cdf[point_idx] = np.cumsum(obs_hist[point_idx]) / mass

    # Synthetic bin contributions per point/track.
    scale_arr: np.ndarray | None
    if isinstance(synthetic_scale, np.ndarray):
        scale_arr = np.asarray(synthetic_scale, dtype=np.float64)
        if scale_arr.size != synthetic.n_points:
            raise ValueError(
                f"synthetic_scale vector must have length {synthetic.n_points}, got {scale_arr.size}."
            )
    else:
        scale_arr = None
    bin_ids = np.digitize(synthetic.intensities, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1, out=bin_ids)
    ptrs = synthetic.point_ptrs
    track_ids = synthetic.track_ids
    point_track_ids: List[np.ndarray] = []
    point_contribs: List[np.ndarray] = []
    edge_counts: List[int] = []
    for point_idx in range(synthetic.n_points):
        start = int(ptrs[point_idx])
        end = int(ptrs[point_idx + 1])
        if start == end:
            point_track_ids.append(np.empty(0, dtype=np.int64))
            point_contribs.append(np.empty((0, n_bins), dtype=np.float64))
            edge_counts.append(0)
            continue
        segment_mask = np.isfinite(synthetic.intensities[start:end])
        if not np.any(segment_mask):
            point_track_ids.append(np.empty(0, dtype=np.int64))
            point_contribs.append(np.empty((0, n_bins), dtype=np.float64))
            edge_counts.append(0)
            continue
        ids = track_ids[start:end][segment_mask]
        bins = bin_ids[start:end][segment_mask]
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        contrib = np.zeros((unique_ids.size, n_bins), dtype=np.float64)
        np.add.at(contrib, (inverse, bins), 1.0)
        if scale_arr is not None:
            s = float(scale_arr[point_idx])
            if s != 1.0:
                contrib *= s
        else:
            if synthetic_scale != 1.0:
                contrib *= float(synthetic_scale)
        point_track_ids.append(unique_ids.astype(np.int32))
        point_contribs.append(contrib)
        edge_counts.append(unique_ids.size)

    point_edge_ptrs = np.zeros(observed.n_points + 1, dtype=np.int64)
    np.cumsum(np.asarray(edge_counts, dtype=np.int64), out=point_edge_ptrs[1:])
    total_edges = int(point_edge_ptrs[-1])
    if total_edges > 0:
        edge_track_ids = np.concatenate(point_track_ids).astype(np.int32, copy=False)
        edge_bin_contribs = np.vstack(point_contribs)
        edge_point_ids = np.repeat(
            np.arange(observed.n_points, dtype=np.int32), np.diff(point_edge_ptrs)
        )
    else:
        edge_track_ids = np.empty(0, dtype=np.int32)
        edge_bin_contribs = np.empty((0, n_bins), dtype=np.float64)
        edge_point_ids = np.empty(0, dtype=np.int32)

    return PointHistogramData(
        bin_edges=bin_edges,
        bin_centers=config.bin_centers,
        bin_widths=config.bin_widths,
        bin_cost_weights=bin_cost_weights,
        observed_hist=obs_hist,
        observed_cdf=obs_cdf,
        observed_mass=obs_mass,
        observed_total_tracks=obs_total_tracks,
        point_track_ids=point_track_ids,
        point_bin_contribs=point_contribs,
        point_edge_ptrs=point_edge_ptrs,
        edge_track_ids=edge_track_ids,
        edge_bin_contribs=edge_bin_contribs,
        edge_point_ids=edge_point_ids,
    )


def wasserstein_1d_from_cdf(cdf_a: np.ndarray, cdf_b: np.ndarray, bin_costs: np.ndarray) -> float:
    """Compute 1D W1 given normalized CDFs and per-bin costs."""
    if cdf_a.size == 0:
        return 0.0
    return float(np.sum(np.abs(cdf_a - cdf_b) * bin_costs))


def compute_global_bin_balance(
    synthetic: PointTrackMatrix,
    bin_edges: Sequence[float],
) -> np.ndarray:
    """
    Compute per-track global histogram contributions for balanced sampling.

    Returns
    -------
    np.ndarray of shape (n_bins, n_tracks)
    """
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = bin_edges.size - 1
    if n_bins <= 0:
        raise ValueError("Histogram configuration must include at least two bin edges.")
    intensities = synthetic.intensities
    mask = np.isfinite(intensities)
    bin_ids = np.digitize(intensities[mask], bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1, out=bin_ids)
    balance = np.zeros((n_bins, synthetic.n_tracks), dtype=np.float64)
    np.add.at(balance, (bin_ids, synthetic.track_ids[mask]), 1.0)
    # Normalize per-bin to reduce scaling issues during rounding.
    bin_totals = balance.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        balance = np.where(bin_totals > 0, balance / bin_totals, balance)
    return balance
