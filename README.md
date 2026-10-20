# MOSES-TC

**Marginal-Optimized Spatial Events Sampling (MOSES)** selects a smaller set of synthetic tropical cyclones to match target marginal distributions at multiple grid points while retaining the selected tracks.

Code accompanying the manuscript:

> **Tropical cyclone landfall risk assessment at consistent warming levels using MOSES-selected synthetic storms**
>
> Yu Hu, Dazhi Xi, Yurong Gao, and Peiyan Chen

This repository provides one historical western North Pacific example (1979–2014): 947 prepared IBTrACS tracks and 36,000 synthetic ACCESS-CM2 tracks, with a target subset of 3,600 tracks. Both inputs are included; no external data paths or tropical cyclone generator are required to run the example.

## Setup and run

Use Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
```

The full example builds track-to-grid mappings, calibrates all 36,000 synthetic tracks, and writes two comparison figures with coastlines. It can require substantial memory and runtime. Mappings are cached under `outputs/cache/`; subsequent runs reuse them while repeating calibration, selection, and plotting. Use `--workers 4` to map synthetic tracks with four processes; the default is one process.

To check the included data without running calibration:

```bash
python run_demo.py --check-inputs
```

Paths are resolved relative to the code, so the repository can be moved independently. Use `--output /path/to/results` to choose an output directory. A repeated run replaces result files in that directory.

## Core workflow

1. Map each track to its maximum wind speed within 250 km of each 0.5° grid point. Use grid points with at least 5 observed and 20 synthetic events.
2. Fit nonnegative track weights with the original sparse CDF/Wasserstein-1 objective, L2 regularization, and a squared ReLU tail penalty.
3. Convert the weights to inclusion probabilities and select exactly 3,600 tracks using the original approximate cube method, including its landing and sample-size adjustment steps.

Settings are centralized in [`moses/config.py`](moses/config.py).

Calibration uses JAX PDHG: 300 iterations for the initial solve and 300 for the tail-penalized solve, with tail thresholds of 96, 113, and 137 kt. The squared ReLU penalty applies when weighted synthetic exceedance counts fall below observed counts. Rounding balances global intensity bins and bins at 20 grid points with the highest observed counts. The random seed is 7. The fixed iteration budget does not certify convergence, and rounding need not preserve every balance feature exactly.

## Outputs

Results are written to `outputs/`:

| File | Contents |
| --- | --- |
| `selected_track_ids.npy` | Zero-based positions in the input synthetic `track` dimension |
| `weights.npy` | Calibrated weights, summing approximately to 1 |
| `weights_phys.npy` | Calibrated count weights, summing approximately to 947 |
| `inclusion_probabilities.npy` | Probabilities used for rounding, summing to 3,600 |
| `cdf_errors.npz` | Active grid indices and count-normalized CDF errors for original, weighted, and selected tracks |
| `run_summary.json` | Configuration, track counts, objective values, and mean CDF errors |
| `track_density.png` | IBTrACS, original, weighted, and selected track density on a 2.5° grid |
| `mean_tc_intensity.png` | IBTrACS, original, weighted, and selected mean TC impact intensity |
| `map_data.npz`, `map_summary.json` | Plotted arrays, metric definitions, and comparison scores |

Track density counts each TC once per visited 2.5° box per year. Synthetic samples are normalized to the observed basin count. Mean TC impact intensity is the conditional mean of each track's maximum wind within 250 km of a grid point. The intensity map uses the active 0.5° calibration grid; white cells indicate absent impacts or points outside that grid. The optimization's count-normalized CDF discrepancies remain in `cdf_errors.npz`.

Coastlines are bundled with the repository and drawn without an external map-data download. To redraw the two figures from saved results and mapping caches:

```bash
python plot_results.py
```

Retrieve the selected tracks without changing their values:

```python
import numpy as np
import xarray as xr

with xr.open_dataset("data/synthetic_ACCESS-CM2.nc") as tracks:
    subset = tracks.isel(track=np.load("outputs/selected_track_ids.npy")).load()
```

The reported errors compare against the calibration targets; they are not an independent holdout test. This is a core historical example, not a reproduction of all manuscript analyses or figures.

## Files and checks

- `run_demo.py`: historical calibration and subset selection.
- `plot_results.py`: plot saved outputs without running calibration again.
- `moses/`: grid mapping, CDF construction, calibration, and balanced sampling.
- `data/`: prepared inputs, coastlines, and [data notes](data/README.md).
- `tests/`: small artificial-data checks, independent of the scientific case study.

```bash
python -m unittest discover -s tests -v
```

Please cite the manuscript above when using this implementation. Data and synthetic model attribution are listed in the data notes.
