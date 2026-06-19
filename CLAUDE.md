# ECG Selective Prediction — Project Spec

## Project Goal

Build a benchmark of selective prediction methods for multi-label ECG classification on PTB-XL. The deliverable is a reproducible comparison of calibration and abstention methods (temperature scaling, MC dropout, deep ensembles, conformal risk control), evaluated via risk-coverage curves, with a methodological contribution in class-conditional conformal abstention for the multi-label setting. Target venue: Q1 or Q2 Journal. Target timeline: one week of experiments.

This is NOT a new-architecture paper. The model is frozen early and treated as a black box; all the work is in the post-hoc uncertainty/calibration/abstention layer on top of it.

**Scope discipline:** helme/PTB-XL benchmark infrastructure (preprocessing, labels, scaler, the 0.925 reference number) exists to support this goal — it is not the goal itself. The deliverable is the selective prediction methodology (calibration, conformal risk control, abstention), not a faithful xresnet1d101 reproduction. If porting the real architecture (Phase 1b) starts consuming disproportionate time, the correct move is to fall back to the placeholder `ECGResNet` as the permanent backbone, report its AUROC honestly (without claiming benchmark parity), and move forward into calibration/selective-prediction work. A slightly lower backbone AUROC does not block the paper; a missing selective-prediction result does.

## Environment

- Hardware: MacBook Pro, M3 Max, 36 GB (use `mps` backend, not `cpu`, for all training/inference)
- Package manager: `uv`, in a dedicated virtual environment separate from the `helme` repo (avoid fastai dependency conflicts entirely — never import anything from `helme/code`, only read its output files)
- Python: 3.11.15
- Core dependencies: torch, numpy, scikit-learn, pandas, wfdb, matplotlib, tqdm, scipy
- Later-phase dependencies (do not install yet): `nonconformist` (conformal prediction), possibly `netcal` (calibration metrics)

## Data

- Source: PTB-XL raw waveforms at `data/ptbxl/records100/` (100 Hz, already present and verified working)
- Task: 5-class superdiagnostic classification (NORM, MI, STTC, CD, HYP), multi-label
- Labels: preprocessed by the `helme` repo, copied into this repo at `output/ecg_ptbxl_benchmarking/exp0/data/` as `y_train.npy`, `y_val.npy`, `y_test.npy`, plus `mlb.pkl` (class ordering) and `standard_scaler.pkl` — these are ground truth, never regenerated
- DECISION: `src/utils/data.py`'s `aggregate_labels()` is NOT used in the active pipeline. It independently reconstructs labels from `scp_codes`/`scp_statements.csv` and is not guaranteed to match helme's column ordering or thresholding exactly. It may be kept only as an offline cross-check against helme's `y_*.npy`, never as the training-time label source.
- DECISION: normalization uses helme's `standard_scaler.pkl` (loaded via pickle, applied with `data.py`'s `apply_standardizer`), not a freshly-fit scaler, to guarantee exact match with benchmark preprocessing. Before trusting it, verify `scaler.mean_.shape == (12,)` and that it was fit on the same channel ordering as `load_signals` produces.
- Splits: use PTB-XL's native `strat_fold` column from `ptbxl_database.csv` via `data.py`'s `get_fold_splits()` — folds 1–8 train, 9 val, 10 test (matches helme convention, required for valid comparison to Strodthoff benchmark numbers). Sanity check required: `X[splits["train"]].shape[0] == y_train.shape[0]` (and same for val/test) before trusting any downstream result.
- Waveforms are NOT precomputed by helme (only labels/scalers are) — loaded from raw `.dat`/`.hea` files via `data.py`'s `load_signals()` (wraps `wfdb.rdsamp`), cached locally as `.npy` after first load (one-time cost, ~2 min)
- `load_signals` / `data.py` returns shape `(N, T, 12)` (channels last). `ECGResNet` expects `(N, 12, T)` for `Conv1d` — an explicit `.transpose(0, 2, 1)` is required after standardization and is easy to silently forget; verify shape before passing to the model.
- Expected shapes: X = (N, 12, 1000), y = (N, 5)

## Repo Structure

Actual layout (supersedes any earlier draft):

```
repo/
├── CLAUDE.md
├── rules.md                     # working agreement for Claude Code sessions — respect this
├── config.py                    # all paths and hyperparameters, single source of truth
├── data/
│   └── ptbxl/                    # raw PTB-XL: records100/, records500/, ptbxl_database.csv, scp_statements.csv
├── output/
│   └── ecg_ptbxl_benchmarking/exp0/data/   # helme artifacts: y_*.npy, mlb.pkl, standard_scaler.pkl, *_bootstrap_ids.npy
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── models/
│   │   └── resnet1d.py           # ECGResNet, returns raw logits (no sigmoid)
│   └── utils/
│       └── data.py               # canonical: load_ptbxl_metadata, load_signals, get_fold_splits,
│                                  # fit_standardizer/apply_standardizer. aggregate_labels present but unused (see Data section)
├── checkpoints/                  # model weights, logits, normalization stats — gitignored
├── results/                      # plots, metrics tables
├── check.py / sanity_check.py    # existing verification scripts
└── documentation/                # (typo'd as `documentatoin` on disk — leave as-is unless you rename deliberately)
```

## Model

DECISION: the real baseline is **xresnet1d101**, ported into pure PyTorch (no fastai dependency). Target: macro AUROC ≈ 0.925 on the superdiagnostic test set, matching Strodthoff et al.'s reported number — this is the only result that counts as "reproduces the benchmark."

`ECGResNet` (currently in `src/models/resnet1d.py`: lightweight 1D ResNet, 4 residual blocks, 64→128→256→512 channels, stride-2 downsampling, global average pool, linear head) is a **temporary placeholder**, used only to validate the data pipeline end-to-end before investing in the real architecture port. Its AUROC is not meaningful and must never be cited as a baseline or benchmark comparison in any figure, table, or write-up.

Both models: input `(batch, 12, 1000)`, output raw logits `(batch, 5)` — sigmoid applied outside the model, never inside, so logits remain available for temperature scaling. Loss: `BCEWithLogitsLoss` (multi-label, not softmax/cross-entropy). Train from scratch in pure PyTorch; do not use fastai or helme's training code.

## Phase Plan

**Phase 1a — Pipeline validation (current focus, fast)**
Run the existing glue code + placeholder `ECGResNet` end to end: load metadata/signals/splits via `data.py`, load helme's `y_*.npy` and `standard_scaler.pkl`, transpose to `(N, 12, T)`, assert shape alignment, train a few epochs. Goal is only to confirm the pipeline is correct (no crashes, AUROC moving in a sane direction, shapes consistent) — not to produce a reportable number.

**Phase 1b — Real baseline**
Port `xresnet1d101` into pure PyTorch (stem, bottleneck blocks, layer depths, ResNet-D-style downsampling — check helme's fastai-based implementation for the exact spec, then reimplement without the fastai dependency). Train on the validated pipeline from 1a. Target macro AUROC ≈ 0.925 on test. This is the number that goes in the paper as "reproduces Strodthoff."

TIMEBOX: this phase should not run more than ~2-3 days. If the port stalls (architecture bugs, AUROC not converging near target after reasonable debugging), stop and fall back to the placeholder `ECGResNet` as the permanent backbone for the rest of the project — see Scope Discipline note above. Do not let baseline fidelity delay the start of Phase 1c.

**Phase 1c — Calibration**
Save val/test logits and labels to disk (plain `.npy`, see Coding Conventions). Implement temperature scaling (single scalar parameter, fit on val logits via LBFGS minimizing NLL). Report ECE before and after scaling — this is the first metric that goes into `src/metrics.py`. Exit criterion: saved checkpoint, saved logits, an ECE table comparing raw vs. temperature-scaled.

**Phase 2 — MC Dropout**
Enable dropout at inference (`model.train()` during forward passes only, no gradient updates), run N=30 stochastic forward passes per test sample, use predictive entropy as the uncertainty/selection score.

**Phase 3 — Deep Ensembles**
Train 5 seeds of the identical (real xresnet1d101) architecture. Use prediction variance across seeds as the selection score. This is the strongest baseline — budget real time for this (5x Phase 1b training cost).

**Phase 4 — Conformal Risk Control (core methodological contribution)**
Adapt Angelopoulos-style conformal risk control to the multi-label setting with class-conditional coverage guarantees, prioritizing actionable classes (MI, CD as proxies for high-risk findings). Abstain on a record if any high-priority class's nonconformity score exceeds its class-specific threshold.

**Phase 5 — Evaluation**
Risk-coverage curves for every method on one plot. AURC as the single-number summary. Per-class ECE and adaptive ECE. Subgroup coverage analysis (age bins from `ptbxl_database.csv`, sex). Clinical-utility framing: sensitivity gain on actionable classes at fixed abstention budgets (e.g. 10%, 20%).

## Coding Conventions

Keep every phase's output (logits, labels, scores) saved as plain `.npy` in `checkpoints/` so later phases never need to re-run earlier ones (decision: plain arrays, not a structured pickle class — optimizing for speed now, revisit only if it becomes genuinely painful). Each calibration/abstention method should be a standalone module exposing a consistent interface: fit on val logits + labels, then score test logits → per-sample selection score + per-sample calibrated probability. This lets `evaluate.py` loop over all methods generically rather than special-casing each one.

Metrics (ECE, adaptive ECE, AURC, anything beyond plain sklearn AUROC) live in `src/metrics.py`, created when first needed (Phase 1c). Every function's docstring must cite the source paper for its definition — this is non-negotiable per `rules.md`, since a wrong formula here silently invalidates results downstream.

Do not modify anything inside the `helme` repo directory, or `data/` (raw datasets) — read-only, reference only. Do not modify any `.pkl` files already produced by a training run (expensive to regenerate) without explicit instruction. Do not invent new metric definitions without asking first. Never use random splits — always `strat_fold`. Do not change the baseline architecture (`xresnet1d101` once ported, per Phase 1b) without explicit instruction.

## Working Agreement for Claude Code Sessions

See `rules.md` for the full list. Key points that affect how this file should be used: give Claude Code precise specs, not goals; never let it launch training runs (it writes code, the human runs it); every metric/Predictor needs a unit test on synthetic data with known properties; review diffs closely on anything touching metrics or conformal code; update this file whenever a design decision is made, so future sessions don't relitigate it.

## Current Status / Next Action

Phase 1a in progress. Resolved: helme label files copied in; canonical data path is `data.py` (not `loader.py`); normalization uses helme's `standard_scaler.pkl`; output format for logits/labels is plain `.npy`; baseline architecture is real `xresnet1d101` (Phase 1b), with the current placeholder `ECGResNet` used only to validate the pipeline in 1a.

Not yet written: the glue code in `src/train.py` that (1) loads metadata + signals + splits via `data.py`, (2) loads `y_train/val/test.npy` from helme's output directory, (3) loads and applies helme's `standard_scaler.pkl`, (4) transposes X to `(N, 12, T)`, (5) asserts shape alignment between X splits and y splits before training starts.

Next action: implement that glue code in `src/train.py` using the placeholder `ECGResNet`, get a human review of the diff, then human runs `python src/train.py` and reports back whether shapes/assertions pass and AUROC trends sanely (not the number itself). Once that's confirmed, move to Phase 1b: port `xresnet1d101`.
