# Historical demo data

Both NetCDF files cover 1979–2014 and are prepared inputs supplied with the research code.

| File | Tracks | Source |
| --- | ---: | --- |
| `ibtracs_tracks.nc` | 947 | IBTrACS, JTWC (`usa`) wind estimates |
| `synthetic_ACCESS-CM2.nc` | 36,000 | Synthetic tropical cyclones driven by ACCESS-CM2 historical environmental fields |

The preparation retains valid track records with maximum sustained wind speeds of at least 34 kt. The IBTrACS preparation uses bounds 100–180°E and 0–60°N. The files contain `lat`, `lon`, and `max_sustained_wind` arrays with dimensions `(time, track)`, padded with NaNs; wind speeds are in knots. `track` contains the original string identifiers. `track_year`, `track_month`, and `track_length` retain the supplied metadata. The synthetic file also retains the original seed counts, which are not used by this historical demo.

The bundled files use lossless NetCDF4 compression. All variable values, coordinates, dimensions, data types, and attributes match the original prepared inputs. The original raw-source version and CMIP6 member are not specified in these prepared files.

Attribution, as given in the manuscript:

- IBTrACS: Knapp et al. (2010) and Gahtan et al. (2024); DOI: `10.25921/82ty-9e16`. [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).
- Synthetic TC model: Lin (2024); DOI: `10.5281/zenodo.11193652`. [Model source](https://github.com/linjonathan/tropical_cyclone_risk).

The prepared ACCESS-CM2 tracks are an input to MOSES, not a storm generator distributed by this repository.

`coastline_110m.npz` contains [Natural Earth 1:110m coastline](https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-coastline/) line coordinates clipped to 98–182°E and 2°S–62°N. The arrays `lon_lat` and `offsets` store the line vertices and segment boundaries. The coastline is [public domain](https://www.naturalearthdata.com/about/terms-of-use/). Plotting uses this local file directly.
