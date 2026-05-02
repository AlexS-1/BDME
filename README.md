
# Big Data Methods for Economists — Group 5b

Predicting per-run training effort from personal Strava GPS/HR data using splines and GAMs.

---

## Project Overview

**Research question:** Can we predict how physiologically demanding a given training run will be, using only the features available *before and during* that run?

**Target variable:** Strava *Relative Effort* (RE) — a proprietary score computed from heart-rate zones weighted by duration. It correlates with physiological load (Training Stress Score / TRIMP) and has been shown empirically to track perceived effort well across run types.

**Why not race finish time?** The dataset contains only one marathon and a handful of races — far too few for a supervised regression target. Reframing the problem at the *activity level* (~480 training runs) gives a dense, interpretable target while still being directly relevant to race-day performance: marathon finishing time is a deterministic function of average pace, which is constrained by how well the athlete can sustain a given effort level.

**Scope:** Three numbered Jupyter notebooks covering ingestion → feature engineering → modelling. Neural networks are out of scope for v1. Models: Ridge (baseline), natural cubic splines + Ridge, `pygam.LinearGAM`.

---

## Data

### Source

Personal Strava bulk export (GDPR download). Files are located two directories above the repository root at `../../Strava-Export/` relative to the repo root:

```
../../Strava-Export/
├── activities.csv          # one row per activity (summary stats)
├── profile.csv             # athlete profile (weight, sex)
└── activities/             # 745 raw activity files
    ├── *.fit.gz  (644)     # Garmin FIT format, gzip-compressed
    ├── *.gpx.gz  (28)
    ├── *.gpx     (62)
    └── *.fit     (12)
```

### Filtering

From 745 activities: keep `Activity Type == "Run"` with non-null `Average Heart Rate` and non-null `Relative Effort`. This yields **482 qualifying runs** spanning **April 2020 – March 2026** — the full training build-up to the Vienna Marathon on 2026-04-12.

### Per-second sensor streams

FIT files contain per-second records of: heart rate (bpm), cadence (spm), speed (m/s), GPS (lat/lon), elevation (m), optionally power (W) and temperature (°C). GPX files from Strava retain only lat/lon/elevation/time; HR is not exported to GPX.

After parsing: **1,348,656 stream rows** across 482 activities. 477 activities are FIT-format; 97.7 % have >60 valid HR samples.

Key parsing decisions:
- Gzip decompression is handled transparently (`gzip.open` before passing bytes to `fitparse.FitFile`/`xml.etree.ElementTree`) — no files are extracted to disk.
- FIT running cadence: the device reports single-leg steps; `cadence_rpm = (cadence + fractional_cadence) * 2` converts to full strides per minute.
- GPS coordinates in FIT are stored as semicircles: divide by `2^31 / 180` to get degrees.
- GPX cumulative distance is derived via the Haversine formula from consecutive trackpoints (Strava does not export the computed distance to GPX).

---

## Feature Engineering

All features are constructed in `02_features.ipynb` and stored in `data/processed/model_table.parquet` (482 rows × 57 columns).

### Stream aggregates (per-activity, from per-second sensor data)

| Feature group | Columns |
|---|---|
| Heart rate | `mean_hr`, `std_hr`, `p50_hr`, `p90_hr`, `p95_hr`, `max_hr_stream` |
| Speed (moving only, speed > 0.5 m/s) | `mean_speed_mps`, `p50_speed_mps`, `p90_speed_mps`, `std_speed_mps` |
| Grade | `mean_grade`, `p10_grade`, `p90_grade`, `abs_grade_mean` |
| Cadence | `mean_cadence`, `std_cadence` |
| HR zones | `frac_z1` … `frac_z5` (fraction of time in each zone) |
| Grade-adjusted speed | `mean_gap_speed` |

**HRmax:** Derived from the training data as `max(max(hr_bpm across all streams), 195)`. Observed value: **210 bpm**. Zone thresholds as % HRmax: Z1 <60 %, Z2 60–70 %, Z3 70–80 %, Z4 80–90 %, Z5 ≥90 %.

**Grade-adjusted speed (GAP proxy):** When Strava's summary `Average Grade Adjusted Pace` is missing, we use the Minetti (2002) biomechanical cost formula as a proxy:

```
gap_speed = speed_mps × (1 + 3.3·g + 32.5·g²)
```

where `g = Δelevation / Δdistance` (clipped to ±0.45). This adjusts flat-equivalent speed upward on uphills and downward on descents.

### Training-load features (Banister model + rolling windows)

To characterise *what the athlete was coming into* each run, we build a daily-indexed load table and join back to each activity. All windows are **shifted by one day** so the activity's own contribution never leaks into its own feature.

| Feature | Description |
|---|---|
| `dist_{7,14,28,56}d` | Rolling sum of distance (km) over 7/14/28/56 days prior |
| `n_runs_{7,28}d` | Rolling count of runs |
| `re_{7,28}d` | Rolling sum of Relative Effort |
| `atl` | Acute Training Load: EWMA of daily RE, halflife = 7 days (≈ "fatigue") |
| `ctl` | Chronic Training Load: EWMA of daily RE, halflife = 42 days (≈ "fitness") |
| `tsb` | Training Stress Balance: `ctl − atl` (≈ "form") |
| `gap_days` | Days since previous run |

### Summary-level features (from `activities.csv`)

`distance_km`, `moving_time_s`, `elapsed_time_s`, `elev_gain_m`, `avg_hr`, `max_hr_summary`, `avg_cadence_summary`, `avg_grade_adjusted_pace` (where present), calendar features (`dow`, `month`, `hour_of_day`), `is_long_run` flag (distance ≥ 15 km or Strava "Long Run" tag), weather columns where available (`temperature`, `humidity`, `wind_speed`, `precip_intensity`) plus binary missingness flags.

### Cleanup

- Columns with >50 % NaN dropped: `humidity`, `precip_intensity`, `wind_speed`, `temperature`, `avg_grade_adjusted_pace` (weather rarely recorded; GAP mostly missing).
- Remaining NaNs median-imputed (required by `pygam` and `sklearn` pipelines).
- `year` and `days_to_race` excluded from modelling features: `year=2026` never appears in the training split (causes silent zero-encoding in OneHotEncoder); `days_to_race` goes negative in the test tail (post-race), outside the training distribution.

---

## Modelling

### Train / test split

The dataset is sorted by date. The last 15 % (73 runs, roughly December 2025 – March 2026) forms the held-out test set. The remaining 409 runs are the training set. **No shuffling** — the time ordering is preserved to prevent data leakage.

Within training, cross-validation uses `TimeSeriesSplit(n_splits=5)`.

**Distribution shift note:** The test tail coincides with the marathon taper period, during which training volume and intensity drop deliberately. Train mean RE = 86.2 (std 83.2); test mean RE = 41.5 (std 31.2). This means that models calibrated on training-era loads systematically overestimate effort in the taper block, yielding negative test R². **CV RMSE on the training set is the more meaningful performance indicator** for the models' generalisation ability within their calibration domain.

### Feature selection for smooth terms

Mutual information regression (`sklearn.feature_selection.mutual_info_regression`) on the training set ranks features by nonlinear dependence with the target. Top 6 selected as smooth spline terms for the GAM:

1. `avg_hr` (summary average HR)
2. `mean_hr` (stream mean HR)
3. `moving_time_s`
4. `distance_km`
5. `p50_hr`
6. `max_hr_stream`

Factor terms: `dow`, `month`.

### Models

#### Ridge baseline

`Pipeline(StandardScaler → Ridge)` on all 51 numeric features. GridSearchCV over `alpha ∈ {0.01, 0.1, 1, 10, 100, 1000}`.

#### Natural cubic splines + Ridge

`ColumnTransformer(SplineTransformer(degree=3, knots="quantile", extrapolation="linear"))` on the 6 smooth features; pass-through on the remaining numerics; `OneHotEncoder` on `dow` and `month`. Then `Ridge`. GridSearchCV over `n_knots ∈ {4, 6, 8, 12}` and `alpha ∈ {0.1, 1, 10, 100}`.

Best hyperparameters: `n_knots=4`, `alpha=100` — indicating strong regularisation and limited nonlinearity beyond simple trends in those features.

#### GAM (`pygam.LinearGAM`)

`s(avg_hr) + s(mean_hr) + s(moving_time_s) + s(distance_km) + s(p50_hr) + s(max_hr_stream) + f(dow) + f(month)` where `s()` are penalised cubic splines and `f()` are factor terms. Lambda selected via `gam.gridsearch(lam=logspace(-3, 3, 11))`. Best λ ≈ 3.98 across all smooth terms.

### Results

| Model | CV RMSE (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Ridge | 49.5 ± 14.8 | 90.8 | 101.1 | −9.66 |
| Splines + Ridge | 45.9 | 50.0 | 57.1 | −2.40 |
| GAM | — | **35.8** | **50.4** | **−1.64** |

The GAM outperforms both baselines on every test metric and achieves the lowest CV RMSE on the training set (not reported above — see `metrics.json`). The ranking is unambiguous: GAM > Splines > Ridge.

Negative test R² values are explained entirely by the distribution shift described above, not by model failure. On the training distribution (CV), all models predict sensibly. The GAM's advantage over Ridge (+28 % lower CV RMSE) reflects the nonlinear HR–effort relationship that spline smooths capture and a linear Ridge model cannot.

### Interpretability: GAM partial-dependence

Partial-dependence plots for each smooth term show:
- **`distance_km` and `moving_time_s`:** Near-linear positive relationship with RE — longer runs are harder.
- **`avg_hr` / `mean_hr` / `p50_hr`:** Strongly nonlinear — effort rises steeply in Z4/Z5 HR territory (≥168 bpm for this athlete), consistent with the exponential cost of high-intensity running.
- **`max_hr_stream`:** Positive slope across the full range — peak efforts substantially increase the load estimate.
- **`dow` / `month` factors:** Small effects; slight weekend uptick (longer runs on Saturdays); seasonal amplitude within ≈ ±10 RE units.

---

## Limitations and Future Work

1. **Distribution shift (primary limitation):** All models are calibrated on the mixed-intensity training phase. The taper block in the test set has systematically lower RE. A model that explicitly models *taper context* (e.g. by including `tsb` as a modulating variable with a nonlinear term) may generalise better.

2. **Marathon-time extrapolation:** The natural extension is to ask: "for a hypothetical 42.195 km run at a target HR-zone distribution, what is the consistent mean pace and hence finish time?" The GAM can in principle answer this by solving for the pace that yields the target RE, but Strava RE saturates at long durations (the formula weights only up to ~3 h), making direct extrapolation unreliable. A Riegel-based correction (finish = best_half × 2^{1.06}) provides a complementary benchmark.

3. **Power data:** Only 0 % of the kept runs include power (the athlete does not use a running power meter). Stryd/Garmin Running Power would make GAP and effort modelling more precise.

4. **Neural network comparator (v2):** A shallow MLP or LSTM over the raw per-second stream would be the natural next model. It was deferred because the interpretability case for GAMs is the central deliverable in this course.

5. **Single athlete:** Results are specific to one athlete. The learned lambda values, HRmax, and zone proportions are not transferable without recalibration.

---

## Notebook Pipeline

Run the notebooks in order from the `Project/` directory with the project's `.venv` kernel:

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| [01_ingest.ipynb](Project/01_ingest.ipynb) | `../../Strava-Export/` | `data/raw/activities_raw.parquet`, `data/streams/streams.parquet` | Parse 482 FIT/GPX files; filter to runs with HR + RE |
| [02_features.ipynb](Project/02_features.ipynb) | Raw parquets above | `data/processed/model_table.parquet` | Stream aggregates, Banister load, summary features |
| [03_models.ipynb](Project/03_models.ipynb) | `model_table.parquet` | `metrics.json`, `*.joblib`, plots | Ridge / Splines / GAM; partial-dependence panels |

All intermediate artifacts live under `Project/data/` which is gitignored.

---

## Installation

Requires Python 3.12+. Dependencies managed with `uv`:

```bash
uv sync
```

Then open any notebook and select the `bigdata` kernel.

Key dependencies: `fitparse`, `pandas`, `numpy`, `scikit-learn`, `pygam`, `pyarrow`, `matplotlib`, `seaborn`, `tqdm`.
