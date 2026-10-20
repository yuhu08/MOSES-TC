"""Draw track density and mean intensity maps with coastlines from saved results."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from moses.config import Config, ROOT
from moses.data_loading import iter_ibtracs_tracks, load_prepared_synthetic_tracks
from moses.grid_utils import GridSpec
from moses.intensity_index import PointSampleMatrix, PointTrackMatrix
from moses.map_products import intensity_fields, track_density_fields

LABELS = {
    "observed": "IBTrACS",
    "original": "Original synthetic",
    "weighted": "Weighted synthetic",
    "subset": "Selected subset",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_matrices(output, config):
    expected = {
        key: list(value) if key == "years" else value
        for key, value in asdict(config).items()
        if key in {"years", "lon_min", "lon_max", "lat_min", "lat_max", "resolution", "radius_km"}
    }
    expected["inputs"] = {
        name: _sha256(ROOT / "data" / name)
        for name in ("ibtracs_tracks.nc", "synthetic_ACCESS-CM2.nc")
    }
    expected["code"] = {
        name: _sha256(ROOT / "moses" / name)
        for name in ("data_loading.py", "grid_utils.py", "intensity_index.py")
    }
    matches = []
    for path in (output / "cache").glob("*/metadata.json"):
        metadata = json.loads(path.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()):
            matches.append(path.parent)
    if len(matches) != 1:
        raise FileNotFoundError("Expected one matching track mapping cache. Run run_demo.py first.")
    cache = matches[0]
    return PointSampleMatrix.load(cache / "observed.npz"), PointTrackMatrix.load(cache / "synthetic.npz")


def _coastlines():
    path = ROOT / "data/coastline_110m.npz"
    with np.load(path, allow_pickle=False) as saved:
        vertices, offsets = saved["lon_lat"], saved["offsets"]
    return [vertices[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def _draw_maps(grid, fields, coastlines, path, title, colorbar_label, norm, cmap, columns=2):
    rows = (len(fields) + columns - 1) // columns
    size = (12.0, 8.4) if rows == 2 else (15.0, 4.8)
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    shown = []
    for axis, (name, values) in zip(axes.flat, fields.items()):
        image = axis.pcolormesh(
            grid.lon, grid.lat, np.asarray(values).reshape(grid.shape),
            shading="nearest", cmap=cmap, norm=norm, rasterized=True,
        )
        axis.add_collection(LineCollection(coastlines, colors="#333333", linewidths=0.65, zorder=3))
        axis.set(xlim=(grid.lon_min, grid.lon_max), ylim=(grid.lat_min, grid.lat_max), title=LABELS[name])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks(np.arange(grid.lon_min, grid.lon_max + 0.01, 20))
        axis.set_yticks(np.arange(grid.lat_min, grid.lat_max + 0.01, 20))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}°E"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}°N"))
        axis.tick_params(labelsize=9)
        shown.append(axis)
    for axis in list(axes.flat)[len(fields):]:
        axis.set_visible(False)
    fig.colorbar(image, ax=shown, label=colorbar_label, shrink=0.88, pad=0.02)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def plot_results(output: Path) -> list[Path]:
    summary = json.loads((output / "run_summary.json").read_text())
    config = Config(**summary["config"])
    coastlines = _coastlines()
    observed, synthetic = _load_matrices(output, config)
    weights = np.load(output / "weights_phys.npy")
    ids = np.load(output / "selected_track_ids.npy")
    if len(np.unique(ids)) != summary["selected_tracks"] or ids.min() < 0 or ids.max() >= synthetic.n_tracks:
        raise ValueError("Selected track indices do not match the saved run.")
    selected = np.zeros(synthetic.n_tracks, dtype=bool)
    selected[ids] = True
    with np.load(output / "cdf_errors.npz") as saved:
        points = saved["point_ids"]
    print("Computing conditional mean intensity...", flush=True)
    means = intensity_fields(observed, synthetic, weights, selected, points)

    density_grid = GridSpec(config.lon_min, config.lon_max, config.lat_min, config.lat_max, 2.5)
    observed_tracks = list(iter_ibtracs_tracks(ROOT / "data/ibtracs_tracks.nc", config.years))
    arrays = load_prepared_synthetic_tracks(ROOT / "data/synthetic_ACCESS-CM2.nc")
    print("Computing physically normalized 2.5-degree track density...", flush=True)
    density = track_density_fields(observed_tracks, arrays, density_grid, weights, selected, config.years[1] - config.years[0] + 1)
    grid = config.grid()
    context = f"Historical ACCESS-CM2 ({config.years[0]}–{config.years[1]})"
    paths = [output / name for name in ("track_density.png", "mean_tc_intensity.png")]
    _draw_maps(
        density_grid, density, coastlines, paths[0], "Track density comparison\n" + context,
        "TCs / 2.5° box / year", PowerNorm(0.6, vmin=0, vmax=max(v.max() for v in density.values())), "YlGnBu",
    )
    _draw_maps(
        grid, means, coastlines, paths[1], "Mean TC impact intensity comparison\n" + context,
        "Mean TC impact intensity (kt)", Normalize(34, max(np.nanmax(v) for v in means.values())), "YlOrRd",
    )
    data = {
        "intensity_lon": grid.lon, "intensity_lat": grid.lat, "active_point_ids": points,
        "density_lon": density_grid.lon, "density_lat": density_grid.lat,
        **{"density_" + name: values for name, values in density.items()},
        **{"mean_intensity_" + name: values for name, values in means.items()},
    }
    np.savez_compressed(output / "map_data.npz", **data)
    common = np.logical_and.reduce([np.isfinite(v) for v in means.values()])
    metrics = {
        "physical_weight_cap": config.w_max,
        "comparison_grid_points": int(common.sum()),
        "density_grid_resolution_degrees": 2.5,
        "intensity_grid_resolution_degrees": config.resolution,
        "coastline_file": "data/coastline_110m.npz",
        "mean_intensity_definition": "Conditional mean of the maximum retained wind within 250 km per track.",
        "density_definition": "Unique tracks per centered 2.5-degree box per year, normalized to the observed basin count.",
        "schemes": {},
    }
    for name in ("original", "weighted", "subset"):
        metrics["schemes"][name] = {
            "track_density_rmse": float(np.sqrt(np.mean((density[name] - density["observed"]) ** 2))),
            "mean_intensity_rmse_kt": float(np.sqrt(np.mean((means[name][common] - means["observed"][common]) ** 2))),
            "active_points_without_mean": int(np.sum(~np.isfinite(means[name][points]))),
        }
    (output / "map_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="Directory containing the saved demo outputs and mapping cache.")
    args = parser.parse_args()
    plot_results(args.output.resolve())
