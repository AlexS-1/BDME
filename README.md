
# Big Data Methods for Economists — Group 5b

Predicting **running race finish times at 100 % effort** from personal Strava GPS / HR data.

---

## Project Overview

**Research question:** Given an athlete's history of GPS-bearing training runs, what finish time can they expect to achieve at 100 % effort? "100 % effort" is operationalised as the maximum sustainable HR over the target race distance — the same condition under which all six known race-day activities were produced (2023-12-03, 2024-04-14, 2024-07-03, 2024-10-19, 2025-06-29, 2026-04-12 *Vienna Marathon*).

**Why not Strava Relative Effort?** Earlier drafts modelled per-activity Relative Effort. RE is a tidy regression target but predicting it just predicts Strava's formula, not race performance. The current pipeline keeps RE and `Perceived Exertion` as **diagnostic guides** (used in 02 to confirm load consistency and to flag race-equivalent training efforts) but the regression target is now race finish time.

**Scope:** Three numbered Jupyter notebooks: ingest → descriptive statistics → models. The course requires at least three of the eight ISLR/ESL topics — we exercise four:

| Topic | Chapter | Implementation |
|---|---|---|
| Non-linear models | ISLR §3.5 + ch. 7 | Natural cubic splines + `pygam.LinearGAM` |
| Regression trees | ISLR ch. 8 | `RandomForestRegressor`, `GradientBoostingRegressor` |
| Support Vector Machines | ISLR ch. 9 | `SVR(kernel="rbf")` |
| Neural Networks | ESL ch. 11 | PyTorch 1D-CNN over per-second streams |

---

## Data

### Source

Personal Strava bulk export (GDPR download). Located **one directory above the repository root** at `../Strava-Export/` relative to the `Project/` working directory:

```
../Strava-Export/
├── activities.csv                  # one row per activity
├── profile.csv                     # athlete profile
└── activities/                     # raw activity files (mixed)
    ├── *.fit.gz                    # FIT compressed (most files)
    ├── *.fit                       # FIT uncompressed
    ├── *.gpx.gz                    # GPX compressed
    └── *.gpx                       # GPX uncompressed (Strava-stripped)
```

Raw files include duplicates with `" 2."` suffixes from iCloud sync; these are **not referenced from `activities.csv`** and are dropped automatically by filtering on the `Filename` column.

### Filtering

From all activities: keep `Activity Type ∈ {Run, Ride, Hike}` with `activity_dt ≤ 2026-04-12 23:59:59` (cuts off post-final-race activities). Run is the modelling target; Ride and Hike enter only via the load context (ATL/CTL/TSB) so cross-training contributes to the race-day fitness state.

### Per-second sensor streams

FIT files contain per-second records of: heart rate (bpm), cadence (spm), speed (m/s), GPS (lat/lon), elevation (m), optionally power (W) and temperature (°C). Strava-stripped GPX retains only lat/lon/ele/time.

### Canonical extended-GPX intermediates

`01_ingest.ipynb` decompresses each source file in memory and **rewrites it once** as a Garmin-style **extended GPX 1.1** (with HR / cadence / power / temperature inside the standard `gpxtpx:TrackPointExtension` block), stored in:

```
Project/data/activities/
├── running/<activity_id>.gpx
├── cycling/<activity_id>.gpx
└── hiking/<activity_id>.gpx
```

After this single pass over the raw export, **no downstream notebook touches the raw export again** — every other artefact is produced from these canonical GPX files.

### Per-second grade-adjusted speed

Race predictions assume a **flat course**, so elevation must be neutralised in the per-second streams (not just at aggregate level). For every GPS row we compute `gap_speed_mps`:

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
| [01_ingest.ipynb](Project/01_ingest.ipynb) | `../Strava-Export/` | `data/activities/{running,cycling,hiking}/*.gpx`, `data/raw/activities_raw.parquet`, `data/streams/streams.parquet` | Decompress + emit canonical extended GPX; per-second GAP |
| [02_descriptive.ipynb](Project/02_descriptive.ipynb) | The two parquets | inline plots + provenance table | Frame the modelling problem; confirm GAP collapses elevation variance; visualise taper before each race |
| [03_models.ipynb](Project/03_models.ipynb) | `model_table` derived inline | `metrics.json`, model joblibs, plots | GAM + Trees + SVR + 1D-CNN; leave-one-race-out CV; race finish-time inference |

All intermediate artefacts live under `Project/data/` (gitignored).

---

## Modelling — `03_models.ipynb`

### Race labels (must be filled in before running 03)

`RACE_LABELS` at the top of the notebook is a `dict[date_str, (distance_km, finish_time_seconds)]`. The notebook will warn loudly if any are missing; supervised models cannot run without them.

**Data exception:** The 2024-04-14 Bonn Half Marathon was recorded without an HR strap (the FIT file contains no heart-rate samples). To keep this race label in the regression, its HR features (`mean_hr`, `p50_hr`, `p90_hr`, `frac_z1–z5`, `hr_load`) are imputed from the 28-day pre-race median of other running activities — the same window used in race-day inference. This is the only manual data exception in the pipeline. The `effort` field for all race rows is set to `1.00` by construction (races are 100%-effort observations by definition).

### Pseudo-labels and the inversion trick

With only six race labels we can't fit a model directly. Instead, every training run is treated as a **pseudo-label**: a pace observed at a known fractional effort (`mean_hr / HRMAX`). Each tabular model learns the surface

> *pace = f(distance, effort, fitness state, calendar)*

We then **invert** this surface at `effort = 1.00`, `distance = 42.195 km`, with race-day fitness state, to predict marathon pace → finish time.

In plain language:

> *We're not predicting any single race directly. We're learning the curve "**how fast does this athlete go at effort X?**" from hundreds of runs, then asking the curve "**and at effort X = 1.00, over the target race distance, in their fitness state on race day?**"*

### Validation

Six race labels → **leave-one-race-out** is the only honest validation. Pseudo-labels (training runs) are shared across folds with the held-out race row excluded.

**Temporal honesty:** Each LORO fold trains *only* on activities recorded **before** the held-out race date. This matches what other tools (Riegel-style calculators, Strava's race predictor) have access to at race time — no future data. The per-race error reported in the LORO summary is therefore interpretable as *"how well could this predictor have done on race day, given only the history available then?"*. Concretely, the earliest race (2023-12-03) trains on a much smaller history than the marathon (2026-04-12); the notebook reports `train_n` per fold to make this asymmetry visible.

### Models

- **Splines + GAM** (`pygam.LinearGAM`). Smooth terms on `distance_km`, `mean_hr`, `mean_gap_speed_mps`, `tsb`; factor terms on `dow`, `month`. Lambda picked by `gridsearch`.
- **Trees / Boosting**. `RandomForestRegressor(400)` and `GradientBoostingRegressor(300)`, plus quantile-loss boosters at α = 0.05 / 0.95 for a 90 % prediction interval.
- **SVR**. RBF kernel inside `Pipeline(StandardScaler → SVR)`, grid over `(C, γ, ε)`.
- **Neural Network**. PyTorch 1D-CNN over per-second tensors `(channels=[gap_speed, hr, cadence, grade], T=7200)`. Three Conv-BN-ReLU-Pool blocks → adaptive avg-pool → MLP head → log-pace. Loss: MSE on log-pace. Optimiser: AdamW + cosine schedule. Regularisation: dropout 0.3, weight-decay 1e-4, early stopping on the LORO fold. The training recipe (loaders, augmentations, what to monitor, how to swap CNN → LSTM) is documented in §3.8 of the notebook. Set `RUN_NN = True` to actually train (requires `torch`).

### Race finish-time inference

For each tabular model: build a one-row inference frame for race day (distance = `RACE_DISTANCE_KM`, effort = 1.00, fitness state from race-day Banister load, all other columns = median over the last 4 weeks of running). Predict pace, multiply by `RACE_DISTANCE_KM`. The notebook also reports a **Riegel baseline** `time = best_HM × 2^1.06` as a sanity check.

---

## Plain-language interpretation (excerpted from 03)

For a non-CS reader: each model below answers the same question — *how fast can this athlete run a race of this distance at maximum effort?* — using the same data, but with very different ways of "drawing the curve".

- **GAM.** A separate smooth line for each input (distance, HR, freshness), added together. Easy to read off "*if my heart rate is X, expected pace shifts by Y*".
- **Random Forest / Gradient Boosting.** Many small decision rules averaged together. Cuts the input space into rectangles. Strong on tabular data.
- **SVR.** A smooth surface that ignores small errors but heavily penalises large ones. Less interpretable, often very accurate on small-to-medium tabular datasets.
- **1D-CNN.** Looks at the *shape* of each run (when HR climbed, when cadence dropped, how grade-adjusted speed evolved). Summary statistics throw this away; the CNN does not.

### Limitations

- **Six race labels.** LORO is correct, but six points cannot establish absolute calibration.
- **Single athlete.** Curves are personal — HRmax, lactate threshold, biomechanics. None of this generalises without re-fitting.
- **Distribution shift around taper.** Race-day Banister state lies near the *edge* of the training distribution.
- **GAP for cycling/hiking is approximate.** Cycling load is really power, not speed. Hiking has its own metabolic cost curve.
- **No course/weather conditioning.** Wind, heat, humidity, course profile of the target race are not inputs. The flat-course assumption is enforced via per-second GAP.

### Future work

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
