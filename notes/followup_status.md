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

**Answered in sections 13.7 and 13.8 (15–16 November 2026):** the context models were undertrained. With the regressor's
budget the PFN reaches 1.8 m and the attention variant 1.7 m on B2 against 4.2 m without context; on B1 the context has
nothing to infer. The text below is the earlier state.

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

## 13. Context diagnostics (13 November 2026)

Question of section 10: why does the context not pay off, and can it? Scripts in
`scratch/context_diag/` (logs beside them), outputs in
`output/eval_paper/context_diag/`. Nothing in `src/` or `scripts/` was changed.
All numbers are MAE in meters over all 50 points of the held-out transits,
calibration context, unless stated.

### 13.1 Upper bound: the environment is worth 2.3 m on B2

`regressor_env.py`: the no-context regressor (tone features, NLL, sensor
dropout, 80k steps) with the true environment (h0, hr0, c, c2/c, k) appended to
every sensor token. Checkpoints `baselines/b2/acoustic_tone_nll_drop_env{,_s1}.pt`.

| B2 | All 50 | 3 / 5 / 7 sensors lost | Range from centroid 0–100 / 100–200 / 200–300 / 300–400 / >400 m |
|---|---|---|---|
| Regressor, environment-blind | 3.62 | 5.81 / 12.45 / 68.8 | 0.57 / 1.05 / 1.88 / 4.30 / 10.53 |
| Regressor, true environment (s0, s1) | **1.29, 1.33** | 1.80 / 3.55 / 31.2 | 0.40 / 0.48 / 0.76 / 1.27 / 3.61 |
| PFN `PD_B_pfnset_fixdrop` (s0, s1) | 4.14, 3.90 | 7.2 / 18.1 / 87 | 4.66 / 3.12 / 3.01 / 3.40 / 7.44 (s0) |
| PFN + blind regressor, fused by predicted variance | 2.43, 2.41 | | 0.56 / 0.98 / 1.57 / 2.83 / 6.21 |

The bound is for an exact environment and the oracle regressor is very
sensitive to it (`env_sensitivity.log`): it falls back to the blind
regressor's 3.6 m with an error of about 0.8 m in depth, 10 m/s in sound speed,
0.02 in the spreading factor or 0.8 m in receiver height (the bottom speed does
not matter). Fusion controls (`fusion.log`): two regressors fused 3.51, two PFN
seeds 3.46, so the 2.4 m is not an ensembling effect; fusing the PFN with the
true-environment regressor adds nothing (1.31), so what the PFN adds to the
blind regressor is environment information.

### 13.2 What the trained PFN does with its context (`ctx_controls.py`)

| Context given to the PFN | B2 (s0, s1) | B1 (`CL_B_pfnset_d256_l6`) |
|---|---|---|
| Calibration, 10 points | 4.14, 3.90 | 3.72 |
| 1 / 2 / 5 / 20 / 40 points | 4.87 / 4.45 / 4.21 / 4.11 / 4.12 (s0) | 3.72 at 1, 2 and 40 |
| 10 points of the target transit itself (prefix / interleaved) | 4.10 / 4.21 | 3.71 / 3.71 |
| Attention to the context switched off | 67.5, 66.1 | 4.17 |
| Context labels permuted among the 10 points | 7.74, 7.48 | 3.72 |
| Labels of another deployment's transit | 44.0, 43.6 | 4.12 |
| Context acoustics replaced by noise | 64.6, 119.2 | 7.01 |
| Context of another deployment | 85.3, 83.3 | 6.38 |
| Context labels shifted by 10 m: share passed to the predictions | 0.375, 0.380 | 0.014 |
| No-context regressor, same transits | 3.62 (median point 1.52; PFN 3.00) | 2.88 (raw; median 1.17; PFN 1.59) |

- **B1: the PFN ignores its context** and is a slightly worse regressor
  (residual correlation with the regressor 0.69). The environment is fixed, so
  there is nothing for the context to supply.
- **B2: the PFN depends on its context completely and uses it to resolve the
  water depth.** With another deployment's context (same array, other
  environment) the error is explained by the depth difference (R² 0.72; 24 m in
  the closest decile of depth difference, 153 m in the farthest quartile); the
  other four parameters have no effect.
- **Its front end does not localize on its own.** An MLP probe on the frozen
  per-point encoder output (`probe_frontend.py`) reaches 9.04 / 8.81 m (linear
  15 m); the regressor gets 3.62 m from the same single point. The PFN decodes
  positions relative to the context, at a flat 3 to 4.7 m at every range, and loses the
  0.6 to 1.9 m accuracy the regressor has inside 300 m (62% of the points).
  Beyond 300 m the PFN is better than the blind regressor.
- The gain from more context points is uniform over the points (about 0.7 m
  from 1 to 10, nothing after), with no dependence on the distance to the
  context labels.
- **No memorization**: on 400 training deployments the PFN scores 3.91 / 3.69
  and the regressor 3.54 (held-out 4.14 / 3.90 and 3.62). Both underfit.

### 13.3 Reading

On B2 the context has 2.3 m to give and the PFN reaches the information (the
fusion shows it), but its design consults the context only after each point has
been compressed by a context-blind front end, and pays with a 3 m floor. On B1
the gap to the regressor (0.85 m) is a training or capacity cost, not a context
effect. Open confound: the B2 regressor reads tone features and the PFN raw
records (equal for the regressor on B1, not checked on B2).

### 13.4 Input-format and seed controls for the B2 regressor

Same recipe as `tone_nll_drop` (NLL, sensor dropout 0.3, 80k steps); logs
`scratch/context_diag/train_b2_{raw_nll_drop,tone_nll_drop_s1}.log`.

| B2 regressor | All 50 | Points 11–50 | 3 / 5 / 7 sensors lost |
|---|---|---|---|
| `tone_nll_drop` (seed 0) | 3.62 | 4.29 | 5.81 / 12.45 / 68.8 |
| `tone_nll_drop_s1` | 3.60 | 4.25 | 5.75 / 12.20 / 67.1 |
| `raw_nll_drop` (raw records, as the NPs read) | 3.88 | 4.50 | 6.07 / 12.56 / 67.8 |

The regressor's seed noise is negligible (0.02 m). On B2 the raw records cost
0.26 m against the tone features (on B1 they were equal), so the fair
no-context reference for the raw-input PFN (4.14 / 3.90) is 3.88 m, not 3.62:
most of the PFN's deficit in the mean is the input format. Its error profile
is still different (flat floor near the array, better far away; section 13.2).

### 13.5 Context-conditioned regressor (`ctxreg`, new opt-in architecture)

`src/models/ctxreg.py`, `config/model/spatial_ctxreg.yaml` (registered in
`registry.py` and in the trainer's list of split architectures). A sensor-set
regressor (256 × 6) whose sensor tokens receive a deployment embedding through a
zero-initialized gate; the embedding comes from a second sensor-set encoder that
reads each context point with its label, followed by a 2-layer transformer over
the context points. It is a CNP with early conditioning (11.4 M parameters).
`spatial.ctx_mode=attn` replaces the summary by per-target attention to the
context tokens before every encoder layer (PFN-like, 12.9 M);
`spatial.ctx_gate=false` removes the context branch. 10% of the training samples
have their context hidden (`ctx_drop`). Launch: `scratch/context_diag/run_ctxreg_b2.sh
<seed> <name> [overrides]` (the B2 PFN recipe with `model=spatial_ctxreg`);
evaluation `eval_ctxreg_b2.sh`; queue `chain_b2.sh`.

Summary mode, `PD_B_ctxreg` (s0, s1), B2, all 50 points (`b2_ctxreg.log`,
`b2_ctxreg_drop{0,3,5,7}/`):

| | `ctxreg` summary | PFN `PD_B_pfnset_fixdrop` | Regressor raw / tone |
|---|---|---|---|
| Calibration, 10 points | 4.11, 3.90 | 4.14, 3.90 | 3.88 / 3.62 |
| Median point | 2.83, 2.71 | 3.00, 2.85 | 2.33 (11–50) / 1.52 |
| 1 / 2 / 5 / 20 / 40 points (s0) | 6.34 / 5.21 / 4.42 / 3.97 / 3.87 | 4.87 / 4.45 / 4.21 / 4.11 / 4.12 | |
| Range 0–100 / 100–200 / 200–300 / 300–400 / >400 m (s0) | 3.95 / 2.76 / 2.88 / 3.36 / 8.41 | 4.66 / 3.12 / 3.01 / 3.40 / 7.44 | tone: 0.57 / 1.05 / 1.88 / 4.30 / 10.53 |
| 3 / 5 / 7 sensors lost | 6.4 / 13.7 / 65.6 | 7.2 / 18.1 / 87 | raw: 6.1 / 12.6 / 67.8 |
| Empty context | 23.4, 23.9 | (67 with attention off) | |
| Labels permuted / other deployment's context | 18.7, 18.0 / 160, 163 | 7.7, 7.5 / 85, 83 | |
| Share of a 10 m label shift passed on | 0.34, 0.35 | 0.375, 0.380 | |
| 90% coverage | 0.91 | 0.94–0.96 | |

**Early conditioning does not fix it: the model reproduces the PFN.** Same
error, same flat 3 to 4 m profile near the array, same total dependence on the
context (residual correlation with the PFN 0.77; PFN seed to seed 0.84; with
the regressor 0.27). This overturns the reading of section 13.3: the floor is
not caused by consulting the context after compression, and the label-shift
share is not point-to-point anchoring (a summary cannot match points); it comes
with inferring the environment from labelled points. What the two models share
is the NP trainer's recipe (30k steps at lr 1e-4, batches of 128 transits, raw
records) and the reliance on an estimated environment. With its context hidden
10% of the time the model reaches only 23 m without context, about what the
regressor reaches after 10% of its steps: the budget is the suspect, and the
gate-closed control (`PD_B_ctxreg_nogate`) is the test. Queued: attention mode
(`PD_B_ctxreg_attn`), then the control.

### 13.6 Attention mode and the gate-closed control (14 November 2026)

Same trainer, recipe and transits as section 13.5; two seeds each
(`b2_ctxreg_attn.log`, `b2_ctxreg_nogate.log`, `*_drop{0,3,5,7}/`).

| B2, NP trainer (200 epochs, lr 1e-4, raw records) | All 50 (s0, s1) | 1 / 5 / 20 / 40 points (s0) | 3 / 5 / 7 sensors lost | Range 0–100 / 100–200 / 200–300 / 300–400 / >400 m (s0) |
|---|---|---|---|---|
| `PD_B_ctxreg_nogate` (no context branch: the plain regressor) | **9.50, 9.23** | | 14.4 / 27.5 / 97 | 8.82 / 8.71 / 7.42 / 8.37 / 14.73 |
| PFN `PD_B_pfnset_fixdrop` | 4.14, 3.90 | 4.87 / 4.21 / 4.11 / 4.12 | 7.2 / 18.1 / 87 | 4.66 / 3.12 / 3.01 / 3.40 / 7.44 |
| `PD_B_ctxreg` (summary) | 4.11, 3.90 | 6.34 / 4.42 / 3.97 / 3.87 | 6.4 / 13.7 / 65.6 | 3.95 / 2.76 / 2.88 / 3.36 / 8.41 |
| `PD_B_ctxreg_attn` (attention) | **3.79, 3.70** | 6.00 / 4.08 / 3.64 / 3.55 | 5.9 / 12.8 / 62.7 | 3.78 / 2.65 / 2.82 / 3.31 / 7.13 |
| Regressor in its own recipe (80k steps, lr 3e-4, iid points), raw / tone | 3.88 / 3.62 | | 6.1 / 12.6 / 68 (raw) | tone: 0.57 / 1.05 / 1.88 / 4.30 / 10.53 |

- **At equal recipe the context is worth 5.5 m.** The same backbone with the
  context branch removed reaches 9.4 m in the NP trainer; the three context
  models reach 3.7 to 4.1 m. The earlier "the context does not pay" compared
  context models trained with this recipe against a regressor trained with a
  much better one.
- **The NP trainer's recipe is the limit.** It takes the plain regressor from
  3.88 m (own recipe, same raw input) to 9.4 m. Every context model so far has
  been trained with it, so they are all far from converged (validation error
  still falling at epoch 200). Which part matters (30k against 80k steps,
  lr 1e-4 against 3e-4 with warm-up, batches of 128 transits against 2048
  independent points) is not isolated.
- **Attention is slightly better than the summary**: 0.26 m in the mean of two
  seeds, both attention seeds below both summary seeds, the same at every
  context size. With 20 or more context points it matches the tone regressor
  (3.64 / 3.54) and with 40 it passes it (3.55 / 3.46), in a worse recipe and
  on raw records. Its control profile is the same as the summary's (empty
  context 24 m, other deployment 160 m, label-shift share 0.35).
- Answered in section 13.7 (a context model trained with the regressor's budget). Bracket was: 3.62
  (blind) to 1.3 (true environment).

### 13.7 The regressor's budget in the NP trainer (15 November 2026)

Recipe (`scratch/context_diag/run_ctxreg_b2_r80k.sh`): 48 transits per batch,
200 epochs (400 steps per epoch, 80k steps), lr 3e-4 with a 10-epoch linear
warm-up then cosine (`training.warmup_epochs`, new opt-in, default 0), no early
stopping. Everything else as sections 13.5 and 13.6: raw records, calibration
context of 2 to 40 points, rotation augmentation, sensor dropout 0.3, context
hidden for 10% of samples. Runs `PD_B_ctxreg_attn_r80k` and
`PD_B_ctxreg_nogate_r80k` (2026-11-14); logs `b2_r80k_s0.log`,
`b2_ctxreg_attn_r80k.log` (both attention seeds, all conditions),
`b2_ctxreg_attn_r80k_drop*/`, `b2_ctxreg_nogate_r80k_drop*/`. Attention has two
seeds, the gate-closed ablation one.

| B2, all 50 points, 10 calibration points | MAE | Median | p90 | Range 0–100 / 100–200 / 200–300 / 300–400 / >400 m |
|---|---|---|---|---|
| Attention, 80k recipe (s0; s1 = **1.72**, median 1.05) | **1.68** | 1.01 | 3.22 | 0.68 / 0.82 / 1.07 / 1.62 / 4.27 |
| Gate closed, 80k recipe (s0) | **4.24** | 2.19 | 9.90 | 0.93 / 1.84 / 2.49 / 4.79 / 11.18 |
| Tone regressor, own recipe (s0) | 3.62 | 1.52 | 9.08 | 0.57 / 1.05 / 1.88 / 4.30 / 10.53 |
| Regressor given the true environment (13.1) | 1.29, 1.33 | | | 0.40 / 0.48 / 0.76 / 1.27 / 3.61 |
| Attention, old recipe (s0), for reference | 3.79 | 2.76 | 6.64 | 3.78 / 2.65 / 2.82 / 3.31 / 7.13 |

Attention, 80k recipe, seed 0, by condition (all 50 points):

| Context | 1 | 2 | 5 | 10 | 20 | 40 points | spread 10 | empty |
|---|---|---|---|---|---|---|---|---|
| MAE (m) | 2.40 | 2.02 | 1.77 | 1.68 | 1.62 | 1.60 | 1.71 | 8.01 |

| Control | labels permuted | other deployment | other deployment's labels / records | noise records | labels +10 / +30 m | own transit, prefix / spread |
|---|---|---|---|---|---|---|
| MAE (m) | 20.1 | 174 | 54.8 / 164 | 48.0 | 7.71 / 20.8 (share 0.33 / 0.31) | 1.66 / 1.71 |

- **The context model was undertrained, not useless.** With the regressor's
  budget the attention model goes from 3.79 to 1.68 m and the gate-closed
  ablation from 9.50 to 4.24 m. At equal recipe the context is worth 2.56 m
  (a factor 2.5), and the attention model sits 0.35 to 0.4 m above the
  true-environment bound of section 13.1. It beats the tone regressor on every
  one of the 800 held-out deployments and on 96% of the transits.
- **The flat 3 to 4 m floor near the array is gone** (0.68 m at 0–100 m): it
  was undertraining, as the gate-closed model of 13.6 suggested.
- **One labelled point already gives 2.40 m**; the gain saturates by 10 to 20
  points. Context from the target's own transit is no better than from another
  transit (1.66 against 1.68): the model uses the deployment, not the track.
- Controls keep the profile of 13.2 and 13.5: wrong deployment 174 m, permuted
  labels 20 m, label-shift share 0.33. Without context the model gives 8.0 m
  (24 m in the old recipe), still twice the gate-closed model: a deployable
  no-context fallback would need more context dropout or a fusion with the
  ablation.
- The gate-closed model in the NP trainer (4.24 m; sensors lost 3 / 5 / 7:
  6.6 / 13.7 / 73) is 0.36 m above the raw-record regressor in its own trainer
  (3.88 m); one seed, not explained (batches of whole transits, label scaling).
- Residual correlation: attention with gate closed 0.46, with tone regressor
  0.47, gate closed with tone regressor 0.85.
- **Seed 1 confirms**: 1.72 m (1 / 5 / 20 / 40 points 2.46 / 1.82 / 1.67 /
  1.63; empty 8.41; other deployment 176; permuted labels 19.6; shift share
  0.33; own transit 1.71). Seed to seed 0.04 m.
- **Sensors lost** (attention, both seeds; three draws each):

  | Sensors lost | 0 | 3 | 5 | 7 |
  |---|---|---|---|---|
  | Attention, 80k recipe (s0, s1) | 1.68, 1.72 | 2.46, 2.53 | 4.86, 5.04 | 35.2, 37.4 |
  | Gate closed, 80k recipe (s0) | 4.24 | 6.60 | 13.7 | 73.0 |
  | Tone regressor, own recipe | 3.62 | 5.75 | 12.2 | 67.1 |
  | Regressor given the true environment | 1.3 | 1.80 | 3.55 | 31.2 |

  With 5 sensors lost the attention model (4.9 m) is near the blind regressor
  with all 10 (3.6 m). 90% interval coverage 0.905 with all sensors, 0.89 with
  3 or 5 lost, 0.94 with 7 lost.
- Launched 15 November 17:49 (`scratch/context_diag/queue_r80k_2.sh`, log
  `queue_r80k_2.log`), two runs at a time: B2 gate-closed seed 1
  (`PD_B_ctxreg_nogate_r80k`), the original PFN with this recipe, two seeds
  (`PD_B_pfnset_r80k`, `MODEL=spatial_pfn_set`), then B1 attention and
  gate-closed, two seeds each (`CL_B_ctxreg_{attn,nogate}_r80k`; 48 per batch,
  80 epochs = 80k steps, 4-epoch warm-up). Evaluations follow each pair
  (`b2_ctxreg_nogate_r80k`, `b2_pfnset_r80k`, `b1_ctxreg_{attn,nogate}_r80k`).
- **Gate-closed seed 1** (15 November 19:22): 4.25 m (seed 0 4.24); sensors
  lost 3 / 5 / 7: 6.55 / 13.4 / 70.6. At equal recipe the context is worth
  2.5 m with two seeds per side (4.24, 4.25 against 1.68, 1.72).
- **Original PFN with this recipe, seed 0** (`PD_B_pfnset_r80k`, 20:18; first
  look `b2_pfnset_r80k_s0.log`): **1.82 m** (old recipe 4.14), median 1.06,
  by range 0.86 / 0.96 / 1.10 / 1.63 / 4.71. Context size 1 / 2 / 5 / 10 / 20 /
  40 points: 1.97 / 1.88 / 1.84 / 1.82 / 1.81 / 1.80. Controls: attention off
  59 m, permuted labels 8.5, other deployment 77, shift share 0.33, own
  transit 1.81. So the existing PFN was undertrained too; no new architecture
  is needed to pass the regressor. Against the attention model it is 0.10 to
  0.14 m worse with 10 points and 0.4 m better with one (1.97 against 2.40),
  with half the parameters (6.11 M against 12.95 M); residual correlation
  between the two 0.76 (attention seed to seed 0.85). One PFN seed; seed 1
  running.
- B1 confirmed at 48 000 training transits (1000 steps per epoch); B1
  attention seed 0 started 20:18 at about 107 s per epoch.
- **Original PFN with this recipe, two seeds** (15 November 21:54;
  `b2_pfnset_r80k.log`, `b2_pfnset_r80k_drop*/`): **1.82, 1.76 m**.

  | B2, 80k recipe (s0, s1) | 10 points | 1 point | 40 points | 3 sensors lost | 5 lost | 7 lost | 90% coverage (0 / 3 / 5 lost) |
  |---|---|---|---|---|---|---|---|
  | Original PFN (6.11 M) | 1.82, 1.76 | 1.97, 1.91 | 1.80, 1.74 | 3.00, 2.92 | 7.98, 7.80 | 60.5, 59.5 | 0.98 / 0.95 / 0.93 |
  | Attention (12.95 M) | 1.68, 1.72 | 2.40, 2.46 | 1.60, 1.63 | 2.46, 2.53 | 4.86, 5.04 | 35.2, 37.4 | 0.905 / 0.89 / 0.89 |
  | Gate closed (4.92 M) | 4.24, 4.25 | | | 6.59, 6.55 | 13.7, 13.4 | 73.0, 70.6 | 0.89 / 0.88 / 0.90 |

  - With 10 points the attention model is 0.09 m better (paired over the 800
    held-out deployments: 0.095 +- 0.012 m; both attention seeds below both
    PFN seeds, but two seeds each). The clear differences are elsewhere: the
    PFN is better with one context point (1.9 against 2.4 m) and nearly flat
    in context size; the attention model keeps improving with context and
    degrades much less with lost sensors (4.9 against 7.9 m with 5 lost).
  - The PFN's intervals are too wide with all sensors (90% interval covers
    0.98, 50% covers 0.76); the attention model's are calibrated.
  - PFN controls, both seeds: attention off 59 / 61 m, other deployment 77 /
    76, permuted labels 8.5 / 7.7, shift share 0.33, own transit 1.81 / 1.74.
  - Averaging the predicted means: two PFN seeds 1.66, two attention seeds
    1.52, all four 1.46 m.

### 13.8 B1 with the same recipe: the context has nothing to do (16 November 2026)

Runs `CL_B_ctxreg_attn_r80k` and `CL_B_ctxreg_nogate_r80k`, two seeds each
(48 transits per batch, 80 epochs = 80k steps, lr 3e-4, 4-epoch warm-up);
logs `b1_ctxreg_{attn,nogate}_r80k.log`, `*_drop{0,3,5,7}/`. 12 000 held-out
transits, all 50 points, 10 calibration points.

| B1 (s0, s1) | MAE | Median | 3 / 5 / 7 sensors lost | Range 0–100 / 100–200 / 200–300 / 300–400 / >400 m (s0) |
|---|---|---|---|---|
| Attention, 80k recipe | 3.25, 3.20 | 1.29, 1.29 | 4.68, 4.59 / 8.18, 7.99 / 42.6, 42.3 | 0.84 / 0.91 / 1.50 / 3.28 / 9.29 |
| Gate closed, 80k recipe | 3.24, 3.18 | 1.27, 1.26 | 4.66, 4.59 / 8.10, 8.07 / 42.1, 42.8 | 0.82 / 0.88 / 1.47 / 3.26 / 9.37 |
| Raw-record regressor `acoustic_raw`, own recipe (Euclidean loss, no sensor dropout, 60k steps) | 2.88 | 1.17 | | 0.72 / 0.77 / 1.38 / 3.13 / 7.96 |
| Old PFN `CL_B_pfnset_d256_l6` | 3.72 | 1.59 | 5.3 / 9.7 / 49 | 1.59 / 1.27 / 1.79 / 3.57 / 10.13 |

- **With and without context the model is the same on B1**: 3.25, 3.20 against
  3.24, 3.18 m, and the same with sensors lost. The attention model's error
  does not move under any context manipulation: 1, 2, 5, 40 points, empty,
  permuted labels, another deployment, noise records all give 3.25 (s0) and
  3.20 (s1); label-shift share 0.000. Individual predictions differ between
  conditions only at the level of bfloat16 rounding (section 13.9). Residual
  correlation attention with gate closed 0.84, attention seed to seed 0.86.
- So B1 (fixed environment, random layouts with known positions) gives the
  context nothing to infer, and B2 (unknown environment) is where it is
  essential (2.5 m, section 13.7). This answers section 10.
- Both are 0.5 m better than the old PFN and 0.3 m behind the regressor in
  its own trainer. That regressor reads raw records too (on B1 raw records
  and tone features give the same, section 3.2), so the input format does
  not explain the gap; it was trained with a Euclidean loss and no sensor
  dropout, and no B1 regressor with the NLL and sensor dropout exists.
  (Corrected 16 November: this row was first labelled "tone regressor".)

### 13.9 bfloat16 rounds the outputs (evaluation-only check, 16 November 2026)

`ctxreg` runs its output head inside the bfloat16 autocast, and so does the
no-context regressor (`acoustic_regressor.py`); the PFN's head is in float32
(only its sensor-set encoder is in bfloat16). bfloat16 has 8 significant bits:
in model units (194 m per unit) a coordinate beyond 194 m is rounded to steps
of 1.5 m, 97 to 194 m to 0.76 m. Same checkpoints, B2, 10 calibration points
(`scratch/context_diag/precision_check.py`, `precision_check_reg.py`):

| B2 MAE (m) | bfloat16 (as reported so far) | head in float32 | all float32 |
|---|---|---|---|
| Attention 80k, s0 / s1 | 1.68 / 1.72 | 1.62 / 1.65 | **1.57 / 1.59** |
| Gate closed 80k, s0 | 4.24 | 4.21 | 4.15 |
| Original PFN 80k, s0 | 1.82 | | 1.81 |
| Tone regressor, s0 / s1 | 3.62 / 3.60 | | 3.55 / 3.52 |
| Raw-record regressor | 3.88 | | 3.80 |

- Every model with a bfloat16 output gains 0.07 to 0.13 m when evaluated in
  float32, without retraining; the PFN gains 0.02. No conclusion changes; the
  attention model's margin over the PFN grows from 0.09 to about 0.2 m.
- All numbers in sections 13.1 to 13.8 are the bfloat16 ones. The
  true-environment bound (1.3 m) comes from the same regressor code and is
  probably rounded too (not rechecked).
- **Applied 16 November** (approved by the user): `ctxreg` computes its head in
  float32 and uses bfloat16 only while training; `eval_suite.load_model`
  switches bfloat16 off for every loaded model (`EVAL_BF16=1` restores it);
  `acoustic_regressor.py` computes its head in float32 and evaluates in
  float32; same in `ctx_controls.py`, `eval_env_regressor.py`, `fusion.py`.
  Re-evaluation `scratch/context_diag/reeval_fp32.sh` (tags `*_fp32`,
  `results_<tag>_robust_fp32.csv`); retraining test with the float32 head
  `PD_B_ctxreg_attn_r80k_h32` (seed 0).

**B2 in float32** (all 50 points, 10 calibration points; s0, s1). These replace
the bfloat16 numbers of sections 13.7 and 13.1 as the reference:

| B2, float32 | MAE | 1 / 5 / 40 points (s0) | 3 sensors lost | 5 lost | 7 lost | 90% coverage (0 / 3 / 5 lost) |
|---|---|---|---|---|---|---|
| Attention, 80k (12.95 M) | **1.57, 1.59** | 2.32 / 1.67 / 1.49 | 2.37, 2.43 | 4.85, 5.02 | 35.0, 37.0 | 0.94 / 0.92 / 0.91 |
| Original PFN, 80k (6.11 M) | **1.81, 1.75** | 1.95 / 1.82 / 1.78 | 2.98, 2.91 | 7.91, 7.73 | 59.9, 59.1 | 0.98 / 0.95 / 0.93 |
| Gate closed, 80k (4.92 M) | 4.14, 4.16 | | 6.51, 6.48 | 13.6, 13.3 | 72.4, 70.6 | 0.91 / 0.89 / 0.91 |
| Tone regressor, own trainer | 3.55, 3.52 | | 5.75, 5.69 | 12.4, 12.2 | 68.7, 67.0 | 0.91 / 0.90 / 0.90 |
| Raw-record regressor, own trainer | 3.80 | | 6.01 | 12.5 | 67.8 | 0.90 |
| Regressor given the true environment | **1.13, 1.17** | | 1.66, 1.74 | 3.43, 3.76 | 31.1, 32.1 | 0.96 / 0.94 / 0.93 |

**B1 in float32** (s0, s1): attention 3.16, 3.10 (3 / 5 / 7 sensors lost 4.60, 4.50 /
8.12, 7.92 / 42.6, 42.3); gate closed 3.13, 3.07 (4.57, 4.48 / 8.01, 7.98 / 42.1,
42.7); regressors in their own trainer: raw records 2.80, 2.79, tone features
2.82. Context still makes no difference on B1 (empty or another deployment's
context: same error).

**Retraining with the float32 head** (`PD_B_ctxreg_attn_r80k_h32`, seed 0, log
`b2_ctxreg_attn_r80k_h32.log`): 1.55 m against 1.57 for the old seed 0
evaluated in float32 (1 / 5 / 40 points 2.30 / 1.66 / 1.45; sensors lost 2.36 /
4.78 / 34.4; 90% coverage 0.92 against 0.94; empty context 7.60). The 0.02 m is
within seed noise: training with rounded outputs did no lasting harm, and the
existing checkpoints need no retraining, only float32 evaluation.

Cost (measured 16 November on the shared GPU, B2): inference 25 / 26 / 21
microseconds per position for PFN / attention / gate closed (1200 transits,
10 context points); training 80k steps with another job on the GPU: PFN
2 h 28, attention 2 h 06 (1 h 28 mostly alone), gate closed 1 h 21 to 1 h 33.
PFN and attention cost the same; the attention model's extra parameters are
its separate context encoder.

### 13.10 Where the attention model's parameters are, and what the trained model uses (16 November 2026)

The user's concern: 12.95 M parameters, twice the PFN, for 0.2 m. Breakdown
(`scratch/context_diag/trim_probe.py`, log `trim_probe.log`):

| Component | Original PFN (6.11 M) | Attention model (12.95 M) |
|---|---|---|
| Sensor encoder, 256 × 6 | 4.74 M, one for context and target points | 4.74 M for target points + **4.74 M second copy for context points** |
| Input maps, norms, head | 0.19 M | 0.31 M |
| Mixing across the context points | 6 blocks at width 128, 1.19 M (also the readout) | transformer over context points, 2 layers at width 256, **1.58 M** |
| Context to target | inside those 6 blocks | 6 cross-attention blocks at width 256, **1.58 M** |

Compute is equal (section 13.9): both pass every point through a 256 × 6 sensor
encoder. Evaluation-only probes on the two trained attention seeds (B2, float32,
intact 1.57 / 1.59 m; removing a part without retraining):

| Probe | s0 | s1 |
|---|---|---|
| Drop the cross-attention block before layer 1 / 2 / 3 / 4 / 5 / 6 | 130 / 17.2 / 2.12 / 1.63 / 3.87 / 5.62 | 124 / 17.4 / 2.72 / 1.68 / 3.29 / 9.85 |
| Context transformer: no layers / first layer only | 73 / 2.80 | 106 / 3.12 |
| Context encoder truncated to 5 / 4 / 3 layers | 2.70 / 4.65 / 8.95 | 2.84 / 5.18 / 8.14 |
| Context encoder running on the target encoder's layers | 44 | 21 |

No part of the trained model is idle, so nothing can be cut without
retraining. The only cheap hint: the cross-attention blocks before layers 3 and
4 carry least (0.06 to 1.1 m), those before layers 1 and 2 are essential.
Implemented 16 November (approved): three opt-in options in `ctxreg`
(`config/model/spatial_ctxreg.yaml`), defaults unchanged and old checkpoints
load and score as before (1.57 m): `ctx_share` (context points run through the
target encoder's layers; own input map, readout token and norm),
`ctx_pool_layers: 0` (no transformer over the context points, attn mode only),
`ctx_attn_layers` (which target-encoder layers, from 1, get a context
attention). Sizes in attn mode: full 12.95 M; A shared encoder 8.21 M; B shared
+ 1-layer context transformer 7.42 M; C shared, no context transformer,
attention before layers 1, 2, 5, 6: 6.10 M (the PFN has 6.11 M).

Launched 12:05 (`scratch/context_diag/chain_trim.sh`, log `chain_trim.log`),
B2, 80k recipe, seed 0, side by side at 37 s per epoch:
A `PD_B_ctxreg_attn_share_r80k`, C `PD_B_ctxreg_attn_lean_r80k`; evaluation
`b2_ctxreg_trim.log`, `b2_ctxreg_trim_drop*/`. References (float32): full model
1.57, 1.59; original PFN 1.81, 1.75.

**Results (finished 14:25, one seed each; B2 held-out, float32, MAE in m, all 50
points; sensor loss with evaluation seed 0):**

| Model | Parameters | 10 points | 1 / 2 / 5 / 40 points | Lost 3 / 5 / 7 | 90% coverage |
|---|---|---|---|---|---|
| Full attention (s0, s1) | 12.95 M | 1.57, 1.59 | s0 2.32 / 1.93 / 1.67 / 1.49 | 2.37, 2.43 / 4.85, 5.02 / 35.0, 37.0 | 0.94 |
| A, shared encoder (s0) | 8.21 M | 1.62 | 2.49 / 1.99 / 1.72 / 1.52 | 2.48 / 5.15 / 37.2 | 0.92 |
| **C, lean (s0)** | **6.10 M** | **1.61** | 2.30 / 1.96 / 1.72 / 1.53 | 2.48 / 5.19 / 37.3 | 0.92 |
| Original PFN (s0, s1) | 6.11 M | 1.81, 1.75 | s0 1.95 / 1.86 / 1.82 / 1.78 | 2.98, 2.91 / 7.91, 7.73 / 59.9, 59.1 | 0.98 |

Controls for A / C: empty context 8.68 / 8.53; permuted labels 18.3 / 16.2;
other deployment 174 / 175; shift share 0.33 / 0.33; own-transit context 1.60 /
1.60. Same behaviour as the full model. Best validation error 1.600 (A), 1.599
(C), against 1.532 for the full model retrained with the float32 head (the
like-for-like validation number).

Reading: the second sensor encoder is not needed (A), and neither are the
transformer over the context points nor the attention blocks before layers 3
and 4 (C equals A). At the PFN's size, C costs 0.02 to 0.05 m at 10 points and
0.2 to 0.3 m with 5 sensors lost against the full model, and keeps its whole
advantage over the original PFN (0.14 to 0.20 m at 10 points, 5.2 against 7.8 m
with 5 sensors lost, calibrated intervals). The PFN stays better with 1 or 2
context points. One seed: the cost against the full model is within about twice
its seed spread, so a second seed of C is needed before stating it.

Compute does not change with the trimming: inference on an idle GPU is 11 us
per position for the PFN, the full model, A and C alike
(`bench_trim.py` in the session scratchpad, 1200 transits x 50 points; the 25 us
of section 13.9 was measured with a training run on the same GPU), and A and C
trained side by side in 2 h 03 min. The saving is in parameters only.

**Second seed of C (approved, 16 November 14:48 to 15:56;**
`scratch/context_diag/chain_lean_s1.sh`, 1 h 04 min of training alone on the
GPU; two-seed evaluation `b2_ctxreg_lean_r80k.log`, `b2_ctxreg_lean_r80k_drop*/`):

| Model | Parameters | 10 points | 1 point | 40 points | Lost 3 | Lost 5 | Lost 7 | 90% coverage |
|---|---|---|---|---|---|---|---|---|
| **C, lean (s0, s1)** | **6.10 M** | **1.61, 1.57** | 2.30, 2.26 | 1.53, 1.48 | 2.48, 2.40 | 5.19, 4.97 | 37.3, 36.1 | 0.92, 0.92 |
| Full attention (s0, s1) | 12.95 M | 1.57, 1.59 | 2.32, 2.37 | 1.49, 1.50 | 2.37, 2.43 | 4.85, 5.02 | 35.0, 37.0 | 0.94, 0.94 |
| Original PFN (s0, s1) | 6.11 M | 1.81, 1.75 | 1.95, 1.89 | 1.78, 1.72 | 2.98, 2.91 | 7.91, 7.73 | 59.9, 59.1 | 0.98, 0.98 |

Seed 1 controls: empty 7.99, permuted labels 15.9, other deployment 170, shift
share 0.33, own-transit context 1.55; best validation error 1.558.

Conclusion: with two seeds C equals the full model (means 1.59 and 1.58 m at 10
points, 5.08 and 4.94 m with 5 sensors lost, both inside the seed spread) at
less than half the parameters and the PFN's size. **C is the main model**:
`model=spatial_ctxreg model.spatial.ctx_mode=attn model.spatial.ctx_share=true
model.spatial.ctx_pool_layers=0 model.spatial.ctx_attn_layers=[1,2,5,6]` with
the 80k recipe (`run_ctxreg_b2_r80k.sh`). Its gate-closed ablation is the
existing `PD_B_ctxreg_nogate_r80k` (4.14, 4.16 m): with the gate closed no
context module exists, so the ablation is the same network for the full model
and for C. Only remaining weakness against the original PFN: 1 or 2 context
points.

### 13.11 Chirp B2 with the 80k recipe and model C (launched 16 November 2026, 17:56)

Approved. `scratch/context_diag/queue_chirp.sh` (log `queue_chirp.log`), data
`data/pathD_chirp/hermite_distinct/processed_delay/geometry_split`, two runs at
a time: C `PD_B_ctxreg_attn_lean_chirp_r80k` (s0, s1; 6.00 M with the 4 delay
features), then gate closed `PD_B_ctxreg_nogate_chirp_r80k` (s0, s1). Evaluation
tags `b2c_ctxreg_lean_r80k` and `b2c_ctxreg_nogate_r80k` (controls and 0/3/5/7
sensors lost). `run_ctxreg_b2.sh` now takes `TRAIN_DD=<dir>`.

Reference, evaluation only, float32 (`b2_chirp_pilot_fp32`): the old-recipe
chirp pilot PFN (`PD_B_pfnset_chirp`, section 7) gives 2.91 m with 10 context
points, 3.10 with 1 point, 3.03 with 5, 2.62 with 40; empty context 3.10,
permuted labels 2.98, other deployment 9.91. So that pilot took only 0.2 m from
its context at 10 points: on chirp the gap between C and its ablation may be
much smaller than on tone (2.6 m). Results pending.
