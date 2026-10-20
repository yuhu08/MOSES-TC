"""Read the bundled, prepared historical NetCDF tracks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
import xarray as xr

KNOTS_TO_MS = 0.514444
MS_TO_KNOTS = 1.0 / KNOTS_TO_MS
MIN_VMAX_KT = 34.0

def normalize_longitudes(lon: np.ndarray) -> np.ndarray:
    """Convert longitude array to [0, 360) and ensure float32 dtype."""
    lon = np.asarray(lon, dtype=np.float32)
    lon = np.mod(lon, 360.0)
    lon[lon < 0.0] += 360.0
    return lon


@dataclass
class TrackRecord:
    """Container for a single TC track."""

    track_id: str
    lat: np.ndarray
    lon: np.ndarray
    vmax: np.ndarray  # knots
    time_hours: Optional[np.ndarray] = None
    track_year: Optional[int] = None
    track_month: Optional[int] = None


def iter_ibtracs_tracks(
    data_path: Path,
    years: Sequence[int],
    basin_filters: Optional[Sequence[str]] = None,
    max_tracks: Optional[int] = None,
) -> Iterator[TrackRecord]:
    """Iterate through the prepared historical IBTrACS NetCDF tracks."""

    def _iter_from_integrated(nc_path: Path) -> Iterator[TrackRecord]:
        start_year, end_year = years
        with xr.open_dataset(nc_path) as ds:
            lat_da = ds["lat"].transpose("time", "track")
            lon_da = ds["lon"].transpose("time", "track")
            vmax_da = ds["max_sustained_wind"].transpose("time", "track")
            time_da = ds.get("time_hours")
            basin_da = ds.get("basin")
            track_years = ds.get("track_year")
            track_months = ds.get("track_month")
            unit = ds.attrs.get("max_sustained_wind_unit", "kn").lower()
            if unit in {"m/s", "ms", "meter per second"}:
                vmax_da = vmax_da * MS_TO_KNOTS

            n_tracks = vmax_da.shape[1]
            n_yielded = 0
            for idx in range(n_tracks):
                track_id = str(ds["track"].values[idx]) if "track" in ds.coords else str(idx)
                try:
                    track_year = int(track_years.values[idx]) if track_years is not None else int(track_id[:4])
                except Exception:
                    track_year = -1
                if track_year < start_year or track_year > end_year:
                    continue
                track_month = None
                if track_months is not None:
                    try:
                        track_month = int(track_months.values[idx])
                    except Exception:
                        track_month = None

                if basin_filters is not None and basin_da is not None:
                    basin_val = str(np.asarray(basin_da.values)[idx]).strip()
                    if basin_val and basin_filters and basin_val not in basin_filters:
                        continue

                lat = lat_da.isel(track=idx).values.astype(np.float32)
                lon = lon_da.isel(track=idx).values.astype(np.float32)
                vmax = vmax_da.isel(track=idx).values.astype(np.float32)
                time_hours = (
                    time_da.isel(track=idx).values.astype(np.float32) if time_da is not None else None
                )

                valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(vmax) & (vmax >= MIN_VMAX_KT)
                if not np.any(valid):
                    continue

                yield TrackRecord(
                    track_id=track_id,
                    lat=lat[valid],
                    lon=normalize_longitudes(lon[valid]),
                    vmax=vmax[valid],
                    time_hours=time_hours[valid] if time_hours is not None else None,
                    track_year=track_year if track_year >= 0 else None,
                    track_month=track_month,
                )
                n_yielded += 1
                if max_tracks is not None and n_yielded >= max_tracks:
                    return

    if data_path.is_dir():
        candidate = data_path / "ibtracs_tracks.nc"
        if candidate.exists():
            data_path = candidate
        else:
            raise FileNotFoundError(
                f"Directory {data_path} does not contain ibtracs_tracks.nc."
                " Use the bundled prepared NetCDF file."
            )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"IBTrACS path {data_path} is not a file. Expected the prepared NetCDF output."
        )
    yield from _iter_from_integrated(data_path)


def load_prepared_synthetic_tracks(
    nc_path: Path,
    *,
    include_track_length: bool = False,
    include_meta: bool = False,
) -> Dict[str, np.ndarray]:
    """Load processed synthetic tracks from a NetCDF supplied with the research code."""
    if nc_path.suffix.lower() != ".nc":
        raise ValueError(
            f"Prepared synthetic tracks must be NetCDF (.nc). Got: {nc_path}. "
            "Use the bundled prepared NetCDF file."
        )
    with xr.open_dataset(nc_path) as ds:
        lat = ds["lat"].transpose("time", "track").values.astype(np.float32)
        lon = normalize_longitudes(ds["lon"].transpose("time", "track").values)
        vmax = ds["max_sustained_wind"].transpose("time", "track").values.astype(np.float32)
        unit = ds.attrs.get("max_sustained_wind_unit", "kn").lower()
        if unit in {"m/s", "ms", "meter per second"}:
            vmax = vmax * MS_TO_KNOTS
        out = {"lat": lat, "lon": lon, "vmax": vmax}
        if include_track_length and "track_length" in ds:
            out["track_length"] = ds["track_length"].values.astype(np.int32)
        if include_meta:
            for key in ("track_year", "track_month"):
                if key in ds:
                    out[key] = ds[key].values.astype(np.int32, copy=False)
            if "seed_per_month" in ds:
                out["seed_per_month"] = ds["seed_per_month"].values
                if "seed_year" in ds.coords:
                    out["seed_year"] = ds["seed_year"].values.astype(np.int32, copy=False)
                if "seed_month" in ds.coords:
                    out["seed_month"] = ds["seed_month"].values.astype(np.int32, copy=False)
        return out
