"""Historical ACCESS-CM2 demo settings."""

from dataclasses import dataclass
from pathlib import Path

from .grid_utils import GridSpec

ROOT = Path(__file__).resolve().parents[1]
BIN_EDGES = (0.0, 34.0, 50.0, 64.0, 80.0, 96.0, 120.0, 150.0, 200.0, 250.0)


@dataclass(frozen=True)
class Config:
    years: tuple[int, int] = (1979, 2014)
    lon_min: float = 100.0
    lon_max: float = 180.0
    lat_min: float = 0.0
    lat_max: float = 60.0
    resolution: float = 0.5
    radius_km: float = 250.0
    subset_size: int = 3600
    min_observed: int = 5
    min_synthetic: int = 20
    histogram_bins: tuple[float, ...] = BIN_EDGES
    lambda_reg: float = 0.0015
    w_max: float = 1.5  # Physical TC count units; sum(weights_phys) equals the observed track count.
    max_iter: int = 300
    tolerance: float = 1e-4
    tail_thresholds: tuple[float, ...] = (96.0, 113.0, 137.0)
    rounding_points: int = 20
    random_seed: int = 7

    def grid(self) -> GridSpec:
        return GridSpec(self.lon_min, self.lon_max, self.lat_min, self.lat_max, self.resolution)
