"""
Sparse representations for per-point track intensities and builders.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from .grid_utils import GridSpec, SphericalNeighborLookup


@dataclass
class PointTrackMatrix:
    """
    Sparse representation of I_{i, p} for synthetic tracks.
    """

    n_points: int
    n_tracks: int
    point_ptrs: np.ndarray
    track_ids: np.ndarray
    intensities: np.ndarray

    def slice_for_point(self, point_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = int(self.point_ptrs[point_idx])
        end = int(self.point_ptrs[point_idx + 1])
        return self.track_ids[start:end], self.intensities[start:end]

    def nnz(self) -> int:
        return int(self.intensities.size)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            n_points=self.n_points,
            n_tracks=self.n_tracks,
            point_ptrs=self.point_ptrs,
            track_ids=self.track_ids,
            intensities=self.intensities,
        )

    @staticmethod
    def load(path: Path) -> "PointTrackMatrix":
        data = np.load(path, allow_pickle=False)
        return PointTrackMatrix(
            n_points=int(data["n_points"]),
            n_tracks=int(data["n_tracks"]),
            point_ptrs=data["point_ptrs"],
            track_ids=data["track_ids"],
            intensities=data["intensities"],
        )


@dataclass
class PointSampleMatrix:
    """Observed intensities per grid point."""

    n_points: int
    point_ptrs: np.ndarray
    intensities: np.ndarray
    total_tracks: int | None = None

    def slice_for_point(self, point_idx: int) -> np.ndarray:
        start = int(self.point_ptrs[point_idx])
        end = int(self.point_ptrs[point_idx + 1])
        return self.intensities[start:end]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            n_points=self.n_points,
            point_ptrs=self.point_ptrs,
            intensities=self.intensities,
            total_tracks=np.array([self.total_tracks if self.total_tracks is not None else -1], dtype=np.int32),
        )

    @staticmethod
    def load(path: Path) -> "PointSampleMatrix":
        data = np.load(path, allow_pickle=False)
        total_tracks_arr = data.get("total_tracks")
        total_tracks = None
        if total_tracks_arr is not None and total_tracks_arr.size:
            total_tracks_val = int(total_tracks_arr.flat[0])
            total_tracks = None if total_tracks_val < 0 else total_tracks_val
        return PointSampleMatrix(
            n_points=int(data["n_points"]),
            point_ptrs=data["point_ptrs"],
            intensities=data["intensities"],
            total_tracks=total_tracks,
        )


def _compute_track_mapping(
    lat: np.ndarray,
    lon: np.ndarray,
    vmax: np.ndarray,
    neighbor_lookup: SphericalNeighborLookup,
    radius_km: float,
) -> Dict[int, float]:
    """
    Map grid-point indices to a single intensity sample per storm.

    Each TC contributes at most one intensity value to any grid point: the
    maximum vmax encountered while the storm center stays within radius_km of
    that grid point.
    """

    valid = ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(vmax)
    if not np.any(valid):
        return {}
    lat_valid = lat[valid]
    lon_valid = lon[valid]
    vmax_valid = vmax[valid]
    mapping: Dict[int, float] = {}

    for lt, ln, intensity in zip(lat_valid, lon_valid, vmax_valid):
        if not np.isfinite(intensity):
            continue
        neighbor_ids = neighbor_lookup.query(float(lt), float(ln), radius_km)
        if neighbor_ids.size == 0:
            continue
        unique_ids = np.unique(neighbor_ids)
        for idx in unique_ids:
            idx = int(idx)
            previous = mapping.get(idx)
            if previous is None or intensity > previous:
                mapping[idx] = float(intensity)
    return mapping


def _process_chunk(
    start: int,
    stop: int,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    vmax_array: np.ndarray,
    neighbor_lookup: SphericalNeighborLookup,
    radius_km: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_blocks: List[np.ndarray] = []
    col_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    lat_chunk = lat_array[:, start:stop]
    lon_chunk = lon_array[:, start:stop]
    vmax_chunk = vmax_array[:, start:stop]

    for offset in range(lat_chunk.shape[1]):
        track_id = start + offset
        mapping = _compute_track_mapping(
            lat_chunk[:, offset],
            lon_chunk[:, offset],
            vmax_chunk[:, offset],
            neighbor_lookup,
            radius_km,
        )
        if not mapping:
            continue
        points = np.fromiter(mapping.keys(), dtype=np.int32, count=len(mapping))
        values = np.fromiter(mapping.values(), dtype=np.float32, count=len(mapping))
        rows = points
        cols = np.full(points.shape, track_id, dtype=np.int32)
        row_blocks.append(rows)
        col_blocks.append(cols)
        val_blocks.append(values)

    if not row_blocks:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    vals = np.concatenate(val_blocks)
    return rows, cols, vals


def _process_chunk_with_grid_params(
    start: int,
    stop: int,
    lat_chunk: np.ndarray,
    lon_chunk: np.ndarray,
    vmax_chunk: np.ndarray,
    grid_params: Tuple[float, float, float, float, float],
    radius_km: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a chunk of tracks, rebuilding the KD-tree lookup in each worker.
    This function receives already-sliced arrays and grid parameters to minimize pickling issues.
    """
    lon_min, lon_max, lat_min, lat_max, resolution = grid_params
    # Rebuild grid and lookup in each worker (avoids pickling cKDTree and numpy arrays)
    grid = GridSpec(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, resolution=resolution)
    neighbor_lookup = SphericalNeighborLookup(grid)

    row_blocks: List[np.ndarray] = []
    col_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []

    for offset in range(lat_chunk.shape[1]):
        track_id = start + offset
        mapping = _compute_track_mapping(
            lat_chunk[:, offset],
            lon_chunk[:, offset],
            vmax_chunk[:, offset],
            neighbor_lookup,
            radius_km,
        )
        if not mapping:
            continue
        points = np.fromiter(mapping.keys(), dtype=np.int32, count=len(mapping))
        values = np.fromiter(mapping.values(), dtype=np.float32, count=len(mapping))
        rows = points
        cols = np.full(points.shape, track_id, dtype=np.int32)
        row_blocks.append(rows)
        col_blocks.append(cols)
        val_blocks.append(values)

    if not row_blocks:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    vals = np.concatenate(val_blocks)
    return rows, cols, vals


def build_point_track_matrix_from_arrays(
    grid: GridSpec,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    vmax_array: np.ndarray,
    neighbor_lookup: SphericalNeighborLookup,
    radius_km: float,
    n_workers: int = 1,
    chunk_size: int = 256,
) -> PointTrackMatrix:
    """
    Build the PointTrackMatrix for synthetic tracks using optional parallelism.

    Uses joblib with loky backend (process-based) for efficient parallel processing.
    Each worker process rebuilds its own KD-tree to avoid pickling issues with cKDTree.
    When using multiple workers, the data is split into n_workers large chunks so each
    worker only builds the KD-tree once.
    """
    if lat_array.shape != lon_array.shape or lat_array.shape != vmax_array.shape:
        raise ValueError("Synthetic arrays must share the same shape.")
    n_steps, n_tracks = lat_array.shape

    if n_workers == 1:
        # Single worker: use the provided lookup directly
        chunks = list(range(0, n_tracks, chunk_size))
        tasks = [(start, min(start + chunk_size, n_tracks)) for start in chunks]
        chunk_outputs = [
            _process_chunk(start, stop, lat_array, lon_array, vmax_array, neighbor_lookup, radius_km)
            for start, stop in tasks
        ]
    else:
        # Multiple workers: split into n_workers large chunks
        # Each worker builds its own KD-tree once and processes its chunk
        tracks_per_worker = (n_tracks + n_workers - 1) // n_workers
        tasks = []
        for i in range(n_workers):
            start = i * tracks_per_worker
            stop = min((i + 1) * tracks_per_worker, n_tracks)
            if start < n_tracks:
                tasks.append((start, stop))

        # Extract grid parameters as simple floats to avoid pickling numpy arrays
        grid_params = (grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max, grid.resolution)

        # Use loky backend (process-based) for true parallelism
        chunk_outputs = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_process_chunk_with_grid_params)(
                start,
                stop,
                lat_array[:, start:stop].copy(),  # Copy slice for each worker
                lon_array[:, start:stop].copy(),
                vmax_array[:, start:stop].copy(),
                grid_params,
                radius_km,
            )
            for start, stop in tasks
        )

    row_blocks = [out[0] for out in chunk_outputs if out[0].size > 0]
    col_blocks = [out[1] for out in chunk_outputs if out[1].size > 0]
    val_blocks = [out[2] for out in chunk_outputs if out[2].size > 0]
    if not row_blocks:
        raise RuntimeError("No synthetic tracks intersected the grid.")

    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    vals = np.concatenate(val_blocks)
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    counts = np.bincount(rows, minlength=grid.n_points)
    point_ptrs = np.zeros(grid.n_points + 1, dtype=np.int64)
    np.cumsum(counts, out=point_ptrs[1:])

    return PointTrackMatrix(
        n_points=grid.n_points,
        n_tracks=n_tracks,
        point_ptrs=point_ptrs,
        track_ids=cols,
        intensities=vals,
    )


def build_point_samples_from_tracks(
    grid: GridSpec,
    tracks: Iterable,
    neighbor_lookup: SphericalNeighborLookup,
    radius_km: float,
    track_count: int | None = None,
) -> PointSampleMatrix:
    """
    Build a PointSampleMatrix for observed intensities.
    """

    point_store: Dict[int, List[float]] = defaultdict(list)
    for record in tracks:
        mapping = _compute_track_mapping(
            record.lat,
            record.lon,
            record.vmax,
            neighbor_lookup,
            radius_km,
        )
        for idx, intensity in mapping.items():
            point_store[idx].append(float(intensity))

    ptrs = np.zeros(grid.n_points + 1, dtype=np.int64)
    value_arrays: List[np.ndarray] = []
    for point_idx in range(grid.n_points):
        values = point_store.get(point_idx)
        if values:
            arr = np.asarray(values, dtype=np.float32)
        else:
            arr = np.empty(0, dtype=np.float32)
        value_arrays.append(arr)
        ptrs[point_idx + 1] = ptrs[point_idx] + arr.size
    if value_arrays:
        intensities = np.concatenate(value_arrays)
    else:
        intensities = np.empty(0, dtype=np.float32)
    if track_count is None:
        try:
            track_count = len(tracks)
        except Exception:
            track_count = None
    return PointSampleMatrix(
        n_points=grid.n_points,
        point_ptrs=ptrs,
        intensities=intensities,
        total_tracks=track_count,
    )
