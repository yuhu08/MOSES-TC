"""
Grid construction utilities and spherical neighbor lookup for TC sampling.

This module keeps the logic lightweight so it can be imported by both the
precomputation and optimization steps without creating cyclic dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0


def _to_cartesian(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon in degrees to 3-D Cartesian unit vectors."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))


@dataclass
class GridSpec:
    """Simple rectangular grid definition."""

    lon_min: float = 100.0
    lon_max: float = 180.0
    lat_min: float = 0.0
    lat_max: float = 60.0
    resolution: float = 0.5

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max or self.lat_min >= self.lat_max:
            raise ValueError("Invalid grid bounds.")
        if self.resolution <= 0:
            raise ValueError("Grid resolution must be positive.")
        self.lon = np.arange(
            self.lon_min,
            self.lon_max + 0.5 * self.resolution,
            self.resolution,
            dtype=np.float32,
        )
        self.lat = np.arange(
            self.lat_min,
            self.lat_max + 0.5 * self.resolution,
            self.resolution,
            dtype=np.float32,
        )
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        self.lon2d = lon2d.astype(np.float32)
        self.lat2d = lat2d.astype(np.float32)
        self.n_points = self.lon2d.size

    @property
    def shape(self) -> Tuple[int, int]:
        return self.lat.size, self.lon.size

    def flatten(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return flattened (lat, lon) arrays."""
        return self.lat2d.ravel(), self.lon2d.ravel()

    def point_coordinates(self, idx: int) -> Tuple[float, float]:
        """Return latitude and longitude for a flattened index."""
        if idx < 0 or idx >= self.n_points:
            raise IndexError("Grid point index out of range.")
        n_lon = self.lon.size
        lat_idx = idx // n_lon
        lon_idx = idx % n_lon
        return float(self.lat[lat_idx]), float(self.lon[lon_idx])


class SphericalNeighborLookup:
    """
    Lightweight neighbor queries on the sphere using a KD-tree.

    The input grid is assumed to be modest in size (<= ~200k points) so that
    the KD-tree fits comfortably in memory.
    """

    def __init__(self, grid: GridSpec):
        self.grid = grid
        lat_flat, lon_flat = grid.flatten()
        self._coords = _to_cartesian(lat_flat, lon_flat)
        self._tree = cKDTree(self._coords)
        # Pre-compute loose geographic bounds to skip obviously distant points.
        self.lat_min = float(grid.lat_min - 1.0)
        self.lat_max = float(grid.lat_max + 1.0)
        self.lon_min = float(grid.lon_min - 1.0)
        self.lon_max = float(grid.lon_max + 1.0)

    def query(self, lat: float, lon: float, radius_km: float) -> np.ndarray:
        """
        Return indices of grid points that are within radius_km of (lat, lon).
        """
        if lat < self.lat_min or lat > self.lat_max:
            return np.empty(0, dtype=np.int32)
        lon = (lon + 360.0) % 360.0
        lon_min = (self.lon_min + 360.0) % 360.0
        lon_max = (self.lon_max + 360.0) % 360.0
        if lon_min < lon_max:
            if lon < lon_min - 1.0 or lon > lon_max + 1.0:
                return np.empty(0, dtype=np.int32)
        # Convert query point to Cartesian coordinates.
        point_vec = _to_cartesian(np.array([lat]), np.array([lon]))[0]
        theta = radius_km / EARTH_RADIUS_KM
        chord = 2.0 * np.sin(0.5 * theta)
        neighbor_ids = self._tree.query_ball_point(point_vec, r=chord)
        if not neighbor_ids:
            return np.empty(0, dtype=np.int32)
        return np.asarray(neighbor_ids, dtype=np.int32)
