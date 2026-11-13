# Follow-up paper: status

Status as of 13 November 2026. This document hands the follow-up work over to a
new conversation: what was done, every result with its source, and what is
still open.

## 0. Where things live

| What | Where |
|---|---|
| Code (private) | this repository, `FernandoFS96/sensor-agnostic-localization-dev` |
| AOR paper code (public) | `FernandoFS96/sensor-agnostic-localization`, tag `v1.0` (= commit `9592356` here) |
| AOR manuscript | `paper_elsevier/main.tex` (not tracked; text removed during the reframe is in `paper_elsevier/reserved_for_followup.tex`) |
| Training runs | `output/hydra/training/<date>/<run name>/seed_<k>/` (`best.pt`, `train_log.csv`, `.hydra/overrides.yaml`) |
| Evaluation results | `output/eval_paper/<group>/results.csv` |
| Launch scripts | `scratch/acoustic/*.sh` (follow-up), `scratch/rescore/`, `scratch/ab_traj/`, `scratch/depth/` (AOR) |
| Python | `/home/fer/miniconda3/envs/torch/bin/python` |

Machine: desktop `nemo` (RTX 5080 16 GB, 64 cores, 125 GB RAM), reached over SSH.
Launch long jobs with `setsid nohup ... &` so they survive the session (a sweep
died once when a session ended). Two training runs saturate the GPU; a
development-set run holds about 55 GB of RAM.

## 1. Starting point and question

The AOR paper (control protocol for shortcut use; benchmarks B1 and B2; PFN
with the legacy spatial front end) ends with models around **20 m** MAE:
calibration-trained PFN 19.7 m on B1 and 19.2 m on B2 with calibration context.
The follow-up asks why the models stop there and how far a better model goes.
The AOR paper deliberately leaves out the new front end, the chirp signal model,
B3 and the MFP baselines; they are reserved for this paper.

## 2. Datasets and evaluation protocol

| Dataset | Path | Deployments | Trajectories each | Signal | Notes |
|---|---|---|---|---|---|
| Development set, shared | `data/random_task/hermite_shared/processed/geometry_split` | 200 (160/20/20) | 400, shared | tone | split `splits/dev.json` |
| Development set, distinct | `data/random_task/hermite_distinct/processed_common/geometry_split` | 200 | 400, own | tone | same split |
| B1 (unseen layouts) | `data/pfn10kd/hermite_distinct/processed/geometry_split` | 10 000 (8000/1000/1000) | 3 | tone, theta 0.1/0.2 | `splits/b1.json` |
| B2 (unseen environments) | `data/pathD/hermite_distinct/processed/geometry_split` | 4000 (3200/400/400) | 6 | tone, theta 0.1 | pinned array; `splits/b2.json` |
| B2 chirp (pilot) | `data/pathD_chirp/hermite_distinct/processed_delay/geometry_split` | 4000 | 6 | chirp | same layout, environments, trajectories and split as B2; raw 55 GB kept |
| B3 (unknown positions) | `data/calib_10k/` (`cells/g<deploy>_t<traj>`, `hermite_distinct/processed_delay`) | up to 8000 | 10 or 50 | chirp | raw 1.12 TB kept; config `config/data/calib_bb.yaml` still points to the old path `data/random_task_bb` |

Signals:
- **Tone** (B1, B2, development set): a single tone over 10–20 kHz at 50 Hz, 201 samples per sensor, SNR 10 dB. Each record is referenced to its own sensor's direct arrival, so arrival-time differences between sensors are absent; the record has effective rank about 2.
- **Chirp** (B3, B2 pilot): `signal_type=chirp absolute_delay=true matched_filter=true`, B = 300 Hz, df = 0.5 Hz (601 taps, about 5 m per tap), `signal_n=2048`; processed into 4 delay features per sensor (`process_data.py --delay-features --delay-df 0.5`): first-arrival range, strongest-arrival range, log peak amplitude, delay spread.

B2 environment prior (uniform): water depth 30–120 m, sound speed 1470–1540 m/s,
bottom speed 0.70–0.95 times that, spreading 1.4–2.0, receiver height 1–5 m,
source at the surface.

Evaluation conventions:
- **Calibration context** (`scripts/eval/eval_calibration.py`): the context is the first 10 points of *another* transit of the same deployment and theta; all 50 points of the target transit are scored. This is the default for the new PFN results below.
- **Prefix context** (`scripts/eval/eval_suite.py`, AOR tables): the first 10 points of the same transit are the context; points 11–50 are scored.
- MFP and the regressor are scored on points 11–50, all held-out transits.
- The held-out pool is the validation and test halves together (the validation half also selects the checkpoint). In B1 and the development set, the validation half holds the arrays farthest from the training ring; scored on the test half alone, no B2 error moves by more than 0.3 m.

## 3. Diagnosis: the front end was the bottleneck

### 3.1 Matched-field processing (`scripts/eval/mfp/`, results in `output/eval_paper/mfp/`)

The MFP uses the same noisy records with the exact forward model.

| Setting | MAE (m) |
|---|---|
| B1, true environment, 2000 held-out deployments | **3.80** (interp 2.87, extrap 4.74; median point 1.0) |
| B1 subset of 500: true environment / theta unknown | 4.06 / 4.24 |
| B1 subset: level only / level demeaned | 5.74 / 13.13 |
| B1 subset: sensor positions off by 2 m | 6.34 |
| B1 subset: c + 15 m/s / depth + 2.5 m | 16.5 / 19.3 |
| B1 subset: all three errors combined | 36.5 |
| B1, 3 / 5 / 7 sensors lost (true / combined errors) | 5.4 / 8.2 / 34.2 and 41.6 / 50.8 / 96.9 |
| **B2, true environment** | **1.11** |
| B2, environment fitted from 10 calibration points (focalization) | **13.51** (median point 2.23; 68% within 5 m; deep 7.88, shallow 19.45) |
| B2, nominal environment (center of the prior) | 181.1 |

Focalization fits (h0, hr0, c, c2/c, k) with random search plus Nelder-Mead;
median fit errors are 1.4 m depth and 11.6 m/s sound speed. A stronger search
would lower its error. The nominal-environment and combined-error rows deny MFP
information the networks have, so they are not fair head-to-heads.

### 3.2 No-context regressor (`scripts/eval/acoustic_regressor.py`)

A sensor-set transformer maps one point's 10 sensor records (plus centered
sensor positions) to the source position, with no context. Results in
`output/eval_paper/baselines/{b1,b2}/results_acoustic_<tag>.csv`.

| Tag | Setting | MAE (m), points 11–50 |
|---|---|---|
| `raw` | B1, raw records | **3.01** (interp 2.35, extrap 3.67; 86% within 5 m) |
| `raw_s1` | second seed | 3.02 |
| `tone` | B1, tone features | 3.05 |
| `tone` (b2) | B2, environment not given | **4.53** (all 50 points: 3.88) |

Raw records and tone features give the same result, so the input format is not
the problem. The regressor also beats MFP with focalization on B2 by about 3.5x.

### 3.3 Front-end factorial (B1 regressor, points 11–50)

| Tag | Front end | MAE (m) |
|---|---|---|
| `raw_npenc` | legacy `SpatialEncoder` + MLP head | 23.16 |
| `raw_npenc_cent` | legacy + centroid frame | 13.28 |
| `raw_stack5` | new encoder with the five old design choices, old position handling | 12.75 |
| `raw_stack5_pos` | same, new position handling | 7.78 |
| `tone_npos` | new encoder, old position handling (Fourier code of absolute positions) | 3.92 |
| `raw` | new encoder, new position handling | **3.01** |

Leave-one-out from the 3.01 m baseline: no feed-forward 3.84 (1.77 M parameters),
3 layers 3.32, width 128 3.23 (1.25 M), attention pooling 3.02, linear tokenizer
3.02. There is no single main factor; the effects interact. The legacy token
projection is a single linear layer with no activation.

Lighter regressors (B1, all / extrap): width 256 × 6 layers (4.92 M) 3.01 / 3.67;
128 × 6 (1.25 M) 3.23 / 3.89; 128 × 3 (0.65 M) 3.77 / 4.46; 96 × 4 (0.49 M)
3.73 / 4.43; 64 × 6 (0.32 M) 3.69 / 4.37.

## 4. The new model

Code: `SensorSetEncoder` in `src/models/anp.py` (`model.spatial.encoder=set`),
`CentroidFrame` in `src/models/frame.py` (`model.frame=centroid`), both wired
through `src/models/registry.py`. Configs: `config/model/spatial_pfn_set.yaml`
and `spatial_cnp_set.yaml`. The legacy encoder stays the default, and legacy
checkpoints load unchanged.

Design: per-sensor MLP tokens of (record, centered position); 6 pre-norm
transformer layers across the sensors, width 256, 8 heads, with feed-forward
blocks; a readout token; positions relative to the array centroid
(`set_pos_scale` 100 m); labels in the centroid frame; bfloat16 autocast.
6.1 M parameters in the PFN.

**Training recipe ("fixdrop")**, used for every result below unless stated:
```bash
python scripts/train/train_np_geometry.py model=spatial_pfn_set model.readout=joint \
  data.data_dir=<dataset> data.max_train_geometries=null data.max_train_trajectories=null \
  data.max_val_trajectories=null data.rot_aug=true data.context.per_sample=true \
  data.context.calibration=true data.sensor_drop.mode=bernoulli \
  training.epochs=200 training.lr=1e-4 seed=<k> exp_name=<run name>
```
`spatial_pfn_set.yaml` sets `var_floor_phys: 0.01` (0.1 m sd). The earlier
floor (1e-3 in normalized units, a 6 m sd at the 193 m label scale) was binding
and made the model overconfident. Sensor dropout defaults: p 0.3, at least 3
sensors kept. Launch scripts: `scratch/acoustic/run_pfnset_b1.sh`,
`run_pfnset_b2.sh`, `run_pfnset_b2_fixdrop.sh`, `run_pfnset_dev.sh`.

Evaluation:
```bash
python scripts/eval/eval_calibration.py --data-dir <dataset> --models pfn=<run>/best.pt \
  --out-dir output/eval_paper/<group> [--drop k] [--pos-err m] [--n-ctx n]
```

## 5. Results of the new PFN (calibration context, all 50 points)

| Dataset | Run (seeds) | MAE (m) | Old model | Group |
|---|---|---|---|---|
| Development set, distinct | `AB_dist_pfnset` (s0) | **1.99**; noise ratio 173.9; one context fix suffices; 1.77–1.98 under calibration context | S-AttnCNP 20.31 (prefix); 22–24 calibration | `ab_distinct_pfnset`, `dev_distinct_calibctx` |
| B1 | `CL_B_pfnset_d256_l6` (s0) | **3.72**; 3 / 5 / 7 sensors lost 5.3 / 9.7 / 49 | PFN 19.70; lost 29.0 / 47.6 / 110.4 | `b1_pfn_size_sweep_drop{0,3,5,7}` |
| B1, before the floor fix, no dropout | `CL_B_pfnset` (s0) | 3.73 (best checkpoint; last 3.48) | | `b1_calibctx_set` |
| B2 | `PD_B_pfnset_fixdrop` (s0, s1) | **4.14 / 3.90**; lost 7.19 / 18.05 / 87.16; 90% coverage 0.96 / 0.94 / 0.93 | PFN 19.23; lost 38 / 100 / 217, overconfident | `b2_calibctx_set_drop{0,3,5,7}` |
| B2, before the floor fix, no dropout | `PD_B_pfnset` (s0) | 4.80 | | `b2_calibctx_set` |

- **Context size** (B2, before the floor fix): 2 / 10 / 40 context points give 6.14 / 4.80 / 4.63 m. On B1 the context adds nothing (`b1_calibctx_set_n2`, `_n40`).
- **The AOR paper's deep-water penalty disappears** with the new front end.

### Size sweeps (fixdrop recipe; all 50 points)

| Width × layers | Parameters | B2 run → MAE | B1 run → MAE |
|---|---|---|---|
| 128 × 6 | 2.47 M | `PD_B_pfnset_d128` → 6.71 | `CL_B_pfnset_d128_l6` → 4.41 |
| 256 × 3 | 3.74 M | `PD_B_pfnset_d256_l3` → 6.20 | `CL_B_pfnset_d256_l3` → 4.40 |
| 192 × 6 | 3.99 M | `PD_B_pfnset_d192_l6` → 4.83 | `CL_B_pfnset_d192_l6` → 3.95 |
| **256 × 6** | **6.11 M** | `PD_B_pfnset_fixdrop` → **4.14 / 3.90** | `CL_B_pfnset_d256_l6` → **3.72** |
| 384 × 6 | 12.15 M | `PD_B_pfnset_d384_l6` → 4.18 | not run |

Keep 256 × 6; depth matters more than width. The PFN is far more sensitive to
width than the regressor (+2.6 m against +0.2 m at width 128). Seed noise on B2
is about 0.25 m. Groups: `{b1,b2}_pfn_size_sweep_drop{0,3,5,7}`.

## 6. Robustness and the value of the context

- **The regressor with the PFN's recipe** (B2, all 50 points; tags `tone_nll`, `tone_nll_drop`): NLL loss 3.75 m; NLL plus sensor dropout **3.62 m**, with 3 / 5 / 7 sensors lost 5.81 / 12.45 / 68.78 m, calibrated. A no-context regressor matches or beats the PFN.
- **Sensor-position error** (`PD_B_pfnset_poserr`, trained with `data.sensor_pos_err_m=6`; group `b2_poserr`; regressor tag `tone_nll_drop_poserr`):

  | Position error at test | 0 m | 2 m | 5 m | 8 m |
  |---|---|---|---|---|
  | PFN trained with position error | 4.19 | 4.84 | 7.21 | 10.26 |
  | PFN, fixdrop recipe | 4.14 | 5.07 | 8.17 | 11.93 |
  | Regressor trained with position error | 4.12 | 5.00 | 7.51 | — |

  Context size 1 / 10 / 40 at 0 m error: 4.93 / 4.19 / 4.18; at 5 m: 7.79 / 7.21 / 7.13. The context gain (about 0.7 m) is the same with and without the error, so **the context does not correct position errors**. Training with position error fixes calibration at 5 m (coverage 50/90/95%: 0.57 / 0.91 / 0.95, against 0.34 / 0.71 / 0.79).
- **Context-value check** (B2 regressor): the per-deployment mean residual is only 4% of the squared error (per transit 14%). Subtracting it using 10 fixes of another transit makes things slightly worse (3.62 → 3.68 m; affine correction 3.90).

## 7. Signal model: chirp pilot on B2

Run `PD_B_pfnset_chirp` (s0, fixdrop recipe), launched by
`scratch/acoustic/run_queue_pilot.sh`; groups `b2_chirp_pilot_drop{0,3,5,7}`.

| | Chirp | Tone |
|---|---|---|
| New PFN | **2.97 m** (points 11–50: 3.02) | 4.14 / 3.90 |
| 3 / 5 / 7 sensors lost | **3.5 / 4.4 / 26.5** | 7.2 / 18.1 / 87 |
| 90% coverage | 0.95 | 0.96 |

Classical linear multilateration on first-arrival ranges
(`scratch/acoustic/pilot_chirp_classical.py`, group `b2_chirp_pilot_classical`,
points 11–50): true sound speed 10.45 m, nominal 13.37, calibrated 10.05. The
solver is basic and the ranges are quantized to about 5 m.

Cost of moving every result to chirp: about 1 day of data generation and
1.5–2 weeks of GPU (85–100 runs). Disk: 968 GB free after the cleanup.

## 8. B3: unknown sensor positions

Old models (legacy front end) plateau around **100 m**; groups
`b3_g<deploy>_t<traj>`.

| Method | Result |
|---|---|
| Plain MLP given the true positions | 27 m (19 m in the centroid frame) |
| Classical geometry from calibration points (`scripts/eval/b3_classical.py`, group `b3_classical`): 40 scattered / 40 contiguous / 10 contiguous | 9.9 / 11.4 / 606 m (median sensor-position error) |
| Classical source localization with the true positions / with calibrated positions (40 points) | 21.1 / 61.4 m |

Diagnosis: the legacy Fourier code memorizes layouts (60 m with raw positions
against 194 m with the Fourier code in a probe); a single linear layer per
sensor underfits (66 m training error); the B3 runs used no translation
augmentation. Old run costs: position-aware about 10 h per seed,
position-free about 3 h per seed.

Two designs, neither run yet:
- **(A) Two stages:** estimate the sensor positions from the calibration
  transit, then a PFN trained with position noise.
- **(B) Joint attention over points and sensors** (axial), in the centroid frame
  of the calibration sources.

The first step either way: the new PFN on B3 data given the true geometry, as
an upper bound.

## 9. Related results carried over from the AOR work

- **Trajectory-only network** (`scripts/eval/trajectory_baseline.py`; 10 context
  positions, no acoustics): B1 24.18, B2 24.06, development set 24.26 (distinct)
  and 25.55 m (shared); with 5 context points 21.5 m. The Hermite trajectories
  are exact cubics in the point index (knots at points 17 and 34), so the
  context determines the first segment and the next knot. Not in the AOR paper.
- **Old models under calibration context**: PFN 19.70 / 19.23 (B1 / B2),
  S-AttnCNP 28.39 / 24.38, S-RecCNP 45.21 / 49.26. Under calibration context,
  the AOR paper's "recurrent model better in deep water" result does not hold
  (B2 deep: PFN 47.5 against S-RecCNP 52.8).
- **MaNGO** (Dahlinger et al., NeurIPS 2025) also subtracts the mean node
  position: the same idea as the centroid frame. Cite and credit it.

## 10. The open problem: what does the context buy?

A no-context regressor with the same front end and recipe matches or beats the
PFN; the context adds little on B2 and nothing on B1, and it does not correct
position errors. As it stands, the honest headline is "a much better front end
(and signal model)", not "the PFN beats regressors".

For the PFN to be the point of the paper, it needs a setting where the context
is essential. Candidates:
1. **B3 (unknown sensor positions)**: the geometry must be inferred from the
   calibration transit. The natural candidate.
2. Per-sensor position or gain errors built into the training prior.
3. Environments outside the training range (train shallow, test deep).
4. Clock or sound-speed errors.
5. A simulation-to-real gap.

## 11. How solid each result is

- **Solid**: the diagnosis (MFP over thousands of deployments, regressor with
  two seeds at 0.01 m apart, clean factorial ablation); the size choice; the B2
  tone result (two seeds).
- **One seed only**: the new PFN on B1 and on the development set; the chirp
  pilot; every sensor-loss and position-error result; the size-sweep points
  other than 256 × 6 on B2.

## 12. Next steps and costs

1. **Decide the paper's claim** ("better front end + signal model" or "the PFN
   where context is essential", i.e. B3). Everything else follows from it.
2. **Core retrain**: new PFN and S-AttnCNP on B1 and B2, 3 seeds each, about
   33 GPU-hours; S-RecCNP about one more day.
3. **Chirp for all benchmarks**: about 1 day of generation and 1.5–2 weeks of GPU.
4. **B3**: the upper-bound run, then design A or B.
5. **Stronger classical baselines**: better environment search for focalization,
   a better multilateration solver for chirp.
6. **Figures and tables**: the AOR scripts in `paper/scripts/` read
   `output/eval_paper/`; the removed MFP and B3 table code is in the history of
   `paper/scripts/make_tables.py` before the AOR release (and in
   `paper_elsevier/reserved_for_followup.tex` for the text).
