
# Big Data Methods for Economists — Group 5b

**Authors:** Alexander Schranner, Nicolas Stocker

Predicting **running race finish times at 100 % effort** from personal Strava GPS / HR data.

---

## Research Question

*Given only an athlete's open Strava export, how accurately can a transparent statistical or machine-learning model predict race finish times?*

"100 % effort" is operationalised as the maximum sustainable HR over the target race distance — the condition under which all six known race-day activities were produced (2023-12-03, 2024-04-14, 2024-07-03, 2024-10-19, 2025-06-29, 2026-04-12).

**Key result:** On the held-out 2026 Milano Marathon (42.2 km), Support Vector Regression predicted a finish time of 2:58:49 — within 33 seconds of the actual 2:58:16 (+0.3 %), matching Garmin's proprietary race predictor and outperforming Strava's pre-race forecast.

**Scope:** Four numbered Jupyter notebooks: ingest → descriptive statistics → models → evaluation.

---

## Data

### Source

Personal Strava bulk export. Located one directory above the repository root at `../Strava-Export/` relative to the `Project/` working directory (available via UZH OneDrive for UZH members [here](https://uzh-my.sharepoint.com/:f:/g/personal/alexander_schranner_uzh_ch/IgCl8kt08dnDRKZhKc9aGbzhAaTwbyys3vjuOy9T4CeOiWk?e=5g3dA7)):

```
../Strava-Export/
├── activities.csv                  # one row per activity
└── activities/                     # raw activity files (mixed)
    ├── *.fit.gz                    # FIT compressed (most files)
    ├── *.fit                       # FIT uncompressed
    ├── *.gpx.gz                    # GPX compressed
    └── *.gpx                       # GPX uncompressed (Strava-stripped)
```

### Filtering

Keep `Activity Type ∈ {Run, Ride, Hike}`. Run is the modelling target; Ride and Hike enter only via the Banister load context (ATL/CTL/TSB) so cross-training contributes to the race-day fitness state.

### Per-Second Sensor Streams

FIT files contain per-second records of: heart rate (bpm), cadence (spm), speed (m/s), GPS (lat/lon), and elevation (m). Strava-stripped GPX retains only lat/lon/ele/time.

### Extended-GPX Intermediates

`01_ingest.ipynb` decompresses each source file in memory and rewrites it once as a Garmin-style extended GPX 1.1 (with HR / cadence inside the standard `gpxtpx:TrackPointExtension` block), stored in:

```
Project/data/activities/
├── running/<activity_id>.gpx
├── cycling/<activity_id>.gpx
└── hiking/<activity_id>.gpx
```

After this single pass over the raw export, no downstream notebook touches the raw export again — every other artefact is produced from these GPX files.

### Per-Second Grade-Adjusted Pace (GAP)

Race predictions assume a flat course so elevation must be neutralised in the per-second streams. For every GPS row we compute `gap_speed_mps`:

| Discipline | Formula | Source |
|---|---|---|
| running | `gap = v · (1 + 3.3·g + 32.5·g²)` | Minetti et al., *J. Appl. Physiol.* 2002 |
| cycling | `gap = v · (1 + 4·g)` (clipped) | Linearised approximation |
| hiking | Minetti running, used as a coarse proxy | Documented limitation |

`g = Δele / Δdist`, clipped to ±0.45 to suppress GPS noise.

---

## Notebook Pipeline

Run notebooks in order from `Project/` with the project's `.venv` kernel:

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| [01_ingest.ipynb](Project/01_ingest.ipynb) | `../Strava-Export/` | `data/activities/{running,cycling,hiking}/*.gpx`, `data/raw/activities_raw.parquet`, `data/streams/streams.parquet` | Decompress + emit extended GPX; per-second GAP |
| [02_descriptive.ipynb](Project/02_descriptive.ipynb) | The two parquets | inline plots + verification table | Frame the modelling problem; visualise HR zone structure and training load |
| [03_models.ipynb](Project/03_models.ipynb) | `model_table` derived inline | `data/processed/loro_results.parquet`, `data/processed/feature_importance.parquet` | GAM + Trees + SVR + 1D-CNN; leave-one-race-out CV; race finish-time inference |
| [04_evaluation.ipynb](Project/04_evaluation.ipynb) | `loro_results.parquet`, `feature_importance.parquet`, `data/courses/*.gpx` (tracked) | metric tables, comparison plots, race-time predictor | Model ranking by MAPE/MAE; GBR PI coverage; course-elevation adjustment; daily-cadence Garmin/Strava-style predictor over configurable distances and models |

All intermediate artefacts live under `Project/data/` (gitignored), **except** the reference race-course GPX files in `Project/data/courses/*.gpx`, which **are tracked** in git.

---

## Modelling — `03_models.ipynb`

### Race Labels (Filled in Manually)

`RACE_LABELS` at the top of the notebook is a `dict[date_str, (distance_km, finish_time_seconds)]`. The notebook will warn loudly if any are missing; supervised models cannot run without them.

**Data Exception:** The 2024-04-14 Bonn Half Marathon was recorded without an HR strap (the FIT file contains no heart-rate samples). Its HR features (`mean_hr`, `p50_hr`, `p90_hr`, `frac_z1–z5`, `hr_load`) are imputed from the 28-day pre-race median of other running activities. This is the only manual data exception in the pipeline. The `effort` field for all race rows is set to `1.00` by construction.

### Confidence Weights

Not all training runs are equally reliable as pseudo-labels. Each observation receives a weight

> *w = q_GPS · q_ele · q_HR · w_dur*

where each *q* factor is the fraction of non-missing sensor coverage and *w_dur* is a sigmoidal duration weight that up-weights activities lasting more than 20 minutes. Race rows are always assigned *w = 1.0*.

### Pseudo-Labels and Inversion Steps

With only six race labels we can't fit a model directly. Instead, every training run is treated as a pseudo-label — a pace observed at a known fractional effort (`mean_hr / HRMAX`). Each tabular model learns the surface

> *pace = f(distance, effort, fitness state, calendar)*

We then **invert** this surface at `effort = 1.00`, `distance = race_distance`, with race-day fitness state, to predict finish time.

### Validation

Six race labels → Leave-One-Race-Out (LORO) is the only honest validation. Each fold trains only on activities recorded **before** the held-out race date — no future data leakage.

### Models

- **GAM** (`pygam.LinearGAM`). Smooth terms on `distance_km`, `mean_hr`, `mean_gap_speed_mps`, `tsb`; factor terms on `dow`, `month`. Lambda picked by `gridsearch` (optimal λ = 10).
- **Random Forest** (`RandomForestRegressor(400)`). Built-in feature importance via mean decrease in impurity.
- **Gradient Boosting** (`GradientBoostingRegressor(300)`), plus quantile-loss boosters at α = 0.05 / 0.95 for a 90 % prediction interval.
- **SVR**. RBF kernel inside `Pipeline(StandardScaler → SVR)`, grid over `(C, γ, ε)`. **Best aggregate MAPE: 18.0 %; +0.3 % on the marathon.**
- **1D-CNN** (PyTorch). Ingests per-second tensors `(gap_speed, hr, cadence, grade)` of shape `4 × 7200`. Three Conv-BN-ReLU-Pool blocks → adaptive avg-pool → MLP head → log-pace. AdamW + cosine schedule, dropout 0.3, weight-decay 1e-4. Set `RUN_NN = True` to train. Note: despite abundant per-second training data, the CNN is poorly suited to this regime because supervised signal is still anchored to only six race labels — it tends to overfit idiosyncratic per-second patterns that tabular models smooth away.
- **Riegel baseline**. Power-law formula `T₂ = T₁ · (D₂/D₁)^1.06`. Reference race is selected as the closest-distance prior race available before the held-out fold. Undefined for fold 1 (no prior race at all). Achieves 8.0 % MAPE over the four folds where a reference is available.

---

## Plain-Language Interpretation

Each model answers the same question — *how fast can this athlete run a race of this distance at maximum effort?* — with different approaches:

- **GAM.** A separate smooth curve for each input, added together. Interpretable: readable partial-dependence plots per feature.
- **Random Forest / Gradient Boosting.** Many decision rules averaged. Strong on tabular data; provides prediction intervals via quantile loss.
- **SVR.** Penalises large errors heavily. Less interpretable but most accurate on this small dataset.
- **1D-CNN.** Looks at the raw per-second shape of each run. Complementary to tabular models on trail races; unreliable when per-second patterns are out-of-distribution (missing HR, triathlon fatigue).

### Limitations

- **Six race labels.** LORO is correct, but six folds cannot establish absolute calibration.
- **Single athlete.** HRmax, lactate threshold, biomechanics are personal; re-fitting required for any other athlete.
- **Distribution shift.** Race-day Banister state and short trail races / triathlon run legs lie near the edge of the training distribution.
- **GAP for cycling/hiking is approximate.** Cycling load is really power, not speed; hiking has its own metabolic cost curve.
- **No course/weather conditioning.** Wind, heat, humidity, and the target race elevation profile are not model inputs.

### Future Work

- **Multi-athlete corpus** with a hierarchical model.
- **Power data** (Stryd / Garmin Running Power) replacing GAP.
- **Pacing-strategy model** — predict per-km splits, aggregate to finish time. Natural fit for an LSTM.
- **Attention over streams** — Transformer encoder on per-second tokens.
- **External shocks** — sleep, illness, work stress (Apple Watch / Oura integration).

---

## Installation

Requires Python 3.12+. Dependencies managed with `uv`:

```bash
uv sync
```

Then open any notebook and select the `bigdata` kernel.

Key dependencies: `fitparse`, `gpxpy`, `pandas`, `numpy`, `scikit-learn`, `pygam`, `xgboost`, `torch`, `pyarrow`, `matplotlib`, `seaborn`, `tqdm`.
