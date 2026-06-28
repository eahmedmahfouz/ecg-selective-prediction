# ECG Selective Prediction — Project Spec

## Project Goal

Build calibration and selective-prediction methods for multi-label ECG classification on PTB-XL: temperature scaling, MC dropout, deep ensembles, conformal risk control, evaluated via risk-coverage curves. Target venue: IEEE JBHI or Computers in Biology and Medicine. Target: Q2 submission.

This is NOT a new-architecture paper. The backbone model is frozen and treated as a black box; the contribution is the post-hoc calibration/abstention layer on top of it.

**Scope discipline:** helme/PTB-XL infrastructure (preprocessing, labels, scaler, benchmark reference numbers) supports this goal — it is not the goal. A slightly-off backbone AUROC does not block the paper; a missing selective-prediction result does.

## Environment

- MacBook Pro M3 Max — use `mps` backend, not `cpu`.
- `uv`, dedicated venv, separate from the `helme` repo (avoid fastai entirely — never import from `helme/code`, only read its output files).
- Python 3.11.15. Core deps: torch, numpy, scikit-learn, pandas, wfdb, matplotlib, tqdm, scipy.

## Data

- Raw PTB-XL waveforms: `data/ptbxl/records100/` (100Hz).
- Baseline task: 5-class superdiagnostic (NORM, MI, STTC, CD, HYP), multi-label. Labels: helme's precomputed `y_train/val/test.npy` in `output/ecg_ptbxl_benchmarking/exp0/data/` — ground truth, never regenerated.
- `aggregate_labels()` in `data.py` is NOT used for the baseline (helme's files are trusted there). It IS used for Contribution 1's subdiagnostic+rhythm labels, where no helme-precomputed equivalent exists.
- DECISION (validated): positivity rule is presence-in-`scp_codes` regardless of stored likelihood value (including 0.0), uniformly across all tasks — no per-task or per-code special-casing. PTB-XL stores some confirmed findings at likelihood=0.0; this was discovered via mismatches against helme's trusted superdiagnostic counts and published rhythm-task prevalences, both now matching within known dataset-version-gap noise (~38 records).
- Normalization: helme's `standard_scaler.pkl` (a single global scaler, confirmed to match helme's own implementation).
- Splits: PTB-XL's `strat_fold` via `data.py`'s `get_fold_splits()` — folds 1–8 train, 9 val, 10 test. Always assert shape alignment before trusting any result.
- `load_signals()` returns `(N, T, 12)`; models expect `(N, 12, T)` — transpose explicitly.

## Repo Structure

```
repo/
├── CLAUDE.md
├── rules.md                      # working agreement for Claude Code — respect this
├── config.py
├── data/ptbxl/                   # raw PTB-XL (read-only)
├── output/ecg_ptbxl_benchmarking/exp0/   # helme artifacts (read-only)
├── src/
│   ├── train.py                  # baseline (5-class)
│   ├── train_acuity.py           # Contribution 1 (35-class)
│   ├── models/resnet1d.py        # ECGResNet — raw logits, no sigmoid inside
│   ├── calibration/risk_control.py   # Contribution 1: calibrate_threshold()
│   ├── plot_risk_coverage.py         # Fig: operating points scatter (tier-colored, no lines)
│   ├── plot_risk_coverage_curves.py  # Fig: 34 certified pts × 3 configs, background bands
│   ├── plot_class_auroc.py           # Fig: 3-panel ROC curves, 24 supported classes
│   └── utils/
│       ├── data.py               # load_ptbxl_metadata, load_signals, get_fold_splits,
│       │                          fit_standardizer/apply_standardizer, aggregate_labels
│       └── labels_acuity.py      # generate_combined_labels() -> (N,35)
├── tests/
├── checkpoints/                  # weights, logits, norm stats, acuity_labels/ — gitignored
└── results/
    ├── contribution1/            # sensitivity_sweep.csv, scorecard.csv, etc.
    └── figures/                  # risk_coverage.pdf/png, risk_coverage_curves.pdf/png,
                                  # class_auroc.pdf/png, uniform_vs_tiered.pdf/png
```

`loader.py` (an earlier draft) is superseded by `data.py` — do not recreate it.

## Baseline Model & Status

Target baseline: real `xresnet1d101`, ported to pure PyTorch, reference macro AUROC ≈0.925 (Strodthoff et al.). Current placeholder: `ECGResNet`, pipeline validated end-to-end and trained successfully — a real, usable backbone, not the official benchmark figure.

helme's own checkpoint (`exp0/models/fastai_xresnet1d101.pth`) is a 71-class backbone (helme's "exp0" task), incompatible with our 5-class labels — not usable as a shortcut.

**Open decision (intentionally unresolved):** whether to port real `xresnet1d101`, or formally adopt the placeholder as the permanent baseline. Does not block Contribution 1.

Both models: input `(batch,12,1000)`, output raw logits `(batch, n_classes)`, `BCEWithLogitsLoss`.

## Phase Plan (baseline track)

1. Calibration — temperature scaling, ECE before/after. First entry in `src/metrics.py`.
2. MC Dropout — predictive entropy from N=30 stochastic forward passes.
3. Deep Ensembles — 5 seeds, prediction variance as selection score.
4. Conformal Risk Control — class-conditional FNR guarantees, uniform α (comparison point against Contribution 1's acuity-weighted version).
5. Evaluation — risk-coverage curves, AURC, per-class ECE, subgroup coverage, clinical-utility framing.

## Research Contributions

1. **Acuity-weighted class-conditional risk control** (current focus)
2. **Hierarchy-constrained conformal sets for multi-label ECG**
3. **Three-signal composite abstention rule** (synthesis layer; exact signals TBD — do not invent without instruction)

### Contribution 1 — Acuity-weighted class-conditional risk control

**Granularity:** combined 23-class subdiagnostic + 12-class rhythm (35 classes, multi-label) — finer than the 5-class baseline, because acuity tiers map onto distinct classes here (rhythm classes like AFib, sinus bradycardia are separated, not lumped). Needs its own 35-logit model/training run; does not replace the baseline. Labels via `generate_combined_labels()` (see Data section), same splits as baseline; an all-zero row is valid.

**Confirmed dataset constraint:** PTB-XL's rhythm task has no VT or VF (the 12 rhythm classes are SR, AFIB, AFLT, BIGU, PACE, PSVT, SARRH, SBRAD, STACH, SVARR, SVTAC, TRIGU — verified against published sources). AV block is one merged `_AVB` bucket, not split by severity. "Critical" tier below means *most severe available in this dataset*, not ICU-grade emergency — state this plainly in the writeup.

**Acuity tiers** (self-defined; precedent: Kwon & Kim, *Sci Rep* 16:10016, 2026; directionally informed — not formally derived — by Sandau et al., *Circulation* 2017 and Wagner et al., *JACC* 2009 Part VI [Wagner is a diagnostic-criteria document, not a risk-stratification one — keep that distinction precise in the writeup]. All three references verified):

| Tier | Classes | Target α | Rationale |
|---|---|---|---|
| Critical | AMI, IMI, LMI, PMI, AFIB, AFLT, `_AVB` | 0.02 | MI subtypes and `_AVB` can't be separated by acuity at this label granularity — treated conservatively by necessity, stated as a judgment call. |
| Important | ISCA, ISCI, ISC_, CLBBB, WPW, PSVT, SVTAC, SVARR, BIGU, TRIGU | 0.05 | Ischemia patterns; CLBBB can mask ischemia; WPW carries arrhythmia risk; supraventricular tachyarrhythmias/ectopy markers. |
| Benign | NST_, STTC, LVH, RVH, SEHYP, IRBBB, ILBBB, IVCD, LAFB/LPFB, CRBBB, LAO/LAE, RAO/RAE, SARRH, SBRAD, STACH, PACE | 0.10 | Chronic/structural or physiologic-variant findings; PACE is a device-status marker, already clinically managed. |
| Excluded (this framework only) | NORM, SR | n/a | Negative/baseline classes. Scope note: this exclusion applies to Contribution 1's risk-control framework only — do not assume exclusion in Contributions 2/3. |

**This table is the single source of truth for class→tier→α.** Any code that calibrates per-class thresholds must derive its class list directly from this table (or a file generated from it) — never hardcode a separate copy. A prior run drifted from this table (wrong tier for AMI/IMI/AFIB, an unsupported class included, a supported class omitted) and produced an untrustworthy result; that class of bug is the reason for this rule.

**Supported/unsupported rule:** a class is calibration-supported if its fold-9 (val) positive count is ≥10; otherwise it must be reported as statistically unsupported rather than given a confident α target (or its α loosened / cross-dataset data pooled, as a documented alternative). Per-class val counts are data-dependent and must be regenerated and checked against this threshold fresh each time, not copied from a prior run.

**Calibration method:** `src/calibration/risk_control.py::calibrate_threshold(scores, labels, alpha, delta, method, bonferroni_grid)` — Learn-then-Test (Angelopoulos & Bates 2021).
- `bonferroni_grid=False` (default): exploits monotonicity of FNR(λ) — single boundary test at full δ, not grid-wide Bonferroni. Equivalent to fixed-sequence testing / the UCB-inversion approach in Bates, Angelopoulos, Lei, Romano & Tibshirani, "Distribution-Free, Risk-Controlling Prediction Sets," JACM 2021 (RCPS). `bonferroni_grid=True` preserved for any future non-monotone risk.
- `method="binomial"` (default): exact one-sided binomial tail p-value — the data is exactly Bernoulli, so this is valid and substantially tighter than Hoeffding, especially at small α. `method="hoeffding"` preserved as an option.
- Between-class correction: δ/n_classes (Bonferroni across all supported classes, for a simultaneous guarantee) — this correction is NOT optional and must not be dropped.
- Any change to this function requires a fresh 1000-trial synthetic coverage simulation (violation rate ≤ δ) before being trusted on real data — this has caught real bugs every time it's been required so far.

**Sensitivity sweep — complete.** Primary (0.02/0.05/0.10), stricter (0.01/0.03/0.08), looser (0.05/0.08/0.15). Derived from CLAUDE.md tier table; val counts computed fresh; δ=0.10/24 (24 supported classes). Unsupported (n_val+ <10, all cut=−1 across all configs): PMI(2), AFLT(7), WPW(7), PSVT(3), SVTAC(3), BIGU(8), TRIGU(2), SEHYP(3), ILBBB(7).

**Primary calibration (method=binomial, bonferroni_grid=False, δ=0.10/24):**

| Class | Tier | α | n_val+ | λ_c | Test FNR | Flag% |
|---|---|---|---|---|---|---|
| AMI | Critical | 0.02 | 306 | 0.0060 | 0.0131 | 54.0% |
| IMI | Critical | 0.02 | 326 | 0.0020 | 0.0061 | 76.1% |
| AFIB | Critical | 0.02 | 151 | 0.0000 | 0.0000 | 100.0% |
| LMI | Critical | 0.02 | 20 | 0.0000 | 0.0000 | 100.0% |
| _AVB | Critical | 0.02 | 83 | 0.0000 | 0.0000 | 100.0% |
| ISCA | Important | 0.05 | 92 | 0.0000 | 0.0000 | 100.0% |
| ISCI | Important | 0.05 | 39 | 0.0000 | 0.0000 | 100.0% |
| ISC_ | Important | 0.05 | 125 | 0.0000 | 0.0000 | 100.0% |
| CLBBB | Important | 0.05 | 54 | 0.0000 | 0.0000 | 100.0% |
| SVARR | Important | 0.05 | 15 | 0.0000 | 0.0000 | 100.0% |
| NST_ | Benign | 0.10 | 75 | 0.0010 | 0.0000 | 70.9% |
| STTC | Benign | 0.10 | 225 | 0.0310 | 0.0901 | 42.4% |
| LVH | Benign | 0.10 | 210 | 0.0260 | 0.0514 | 29.8% |
| RVH | Benign | 0.10 | 12 | 0.0000 | 0.0000 | 100.0% |
| IRBBB | Benign | 0.10 | 112 | 0.0160 | 0.0089 | 26.3% |
| IVCD | Benign | 0.10 | 78 | 0.0010 | 0.0127 | 96.1% |
| LAFB/LPFB | Benign | 0.10 | 181 | 0.0731 | 0.0615 | 17.5% |
| CRBBB | Benign | 0.10 | 55 | 0.0000 | 0.0000 | 100.0% |
| LAO/LAE | Benign | 0.10 | 43 | 0.0000 | 0.0000 | 100.0% |
| RAO/RAE | Benign | 0.10 | 10 | 0.0000 | 0.0000 | 100.0% |
| SARRH | Benign | 0.10 | 77 | 0.0070 | 0.0260 | 66.6% |
| SBRAD | Benign | 0.10 | 64 | 0.0010 | 0.0156 | 51.3% |
| STACH | Benign | 0.10 | 83 | 0.0030 | 0.0244 | 9.8% |
| PACE | Benign | 0.10 | 29 | 0.0000 | 0.0000 | 100.0% |

Flag% = fraction of test set with score ≥ λ_c (predicted positive), uniform formula, no special-casing.

**Stability summary across all three configs:**

- **Stable certified (all 3, 7 classes — all Benign):** IRBBB, IVCD, LAFB/LPFB, LVH, SARRH, STACH, STTC. These are the primary paper result — certify regardless of α choice.
- **α-sensitive — certify at primary, drop under stricter (4):** AMI, IMI (Critical: certify at α=0.02 but not α=0.01, min_n jumps to 546); NST_, SBRAD (Benign: marginal, cut=1 at primary). Flag in writeup.
- **α-sensitive — certify only under looser (5):** AFIB (Critical, α=0.05: 151 > min_n=107); ISCA, ISC_ (Important); CRBBB, LAO/LAE (Benign — very low λ_c, borderline). Flag in writeup.
- **Stable uncertified (all 3, 8 classes):** LMI(20), _AVB(83) [Critical]; CLBBB(54), ISCI(39), SVARR(15) [Important]; PACE(29), RAO/RAE(10), RVH(12) [Benign].

**Critical tier finding (state plainly in writeup):** AMI (n=306) and IMI (n=326) are the only Critical-tier classes with enough val positives to certify at α=0.02 (min_n=272). _AVB (n=83) never certifies across any config — even at the loosest α=0.05 it needs 107, falls 24 short. This is a dataset-scale constraint, not a methods failure. AFIB (n=151) certifies only at α=0.05 (looser). The Critical tier therefore splits into two empirical sub-groups: MI subtypes with sufficient prevalence (AMI, IMI) vs. rhythm/conduction classes with insufficient prevalence at PTB-XL fold-9 scale (_AVB, AFIB at primary α, LMI).

### Contribution 1 — Figures (complete)

Three paper figures generated; all save to `results/figures/`:

| File | Script | What it shows |
|---|---|---|
| `risk_coverage.pdf` | `plot_risk_coverage.py` | Single-panel scatter: (flag-rate, FNR) operating points for all 34 certified points, tier-colored, background FNR bands, α ceiling lines |
| `risk_coverage_curves.pdf` | `plot_risk_coverage_curves.py` | Same 34 points × 3 α sweep configs; marker shape encodes config (▼ stricter, ● tiered, ▲ looser); background bands |
| `class_auroc.pdf` | `plot_class_auroc.py` | 3-panel ROC curves (Critical / Important / Benign) for all 24 supported classes; solid = certified, dashed = uncertifiable/trivial |

**Locked decisions for all figures:**

- **"Certified" definition:** status == `"certified"` at **tiered/primary α only** — not "certified under any of the 3 sweep configs". All figures, tables, captions, and paper text use this definition uniformly. Any deviation must be stated explicitly in the caption and cross-referenced.
- **`_AVB` matplotlib gotcha (permanent coding note):** matplotlib's `ax.get_legend_handles_labels()` silently drops any artist whose label starts with `_`. The class `_AVB` triggers this. **Never use `get_legend_handles_labels()` in any plot that may include `_AVB`.** Instead, collect `(line, label)` tuples manually during the plot loop and pass them directly to `ax.legend()`.

### Contributions 2 & 3 — not yet scoped

Contribution 2 will use the subdiagnostic↔superdiagnostic hierarchy in `scp_statements.csv`. Contribution 3's three signals are undefined — ask before implementing either.

## Coding Conventions

Save logits/labels/scores as plain `.npy` in `checkpoints/`. Each calibration/abstention method: standalone module, fit-on-val / score-on-test interface.

Metrics beyond plain AUROC live in `src/metrics.py`, every function citing its source paper.

Do not modify `helme/`, `data/`, or any existing `.pkl` without explicit instruction. Do not invent metric or abstention-rule definitions without asking. Always `strat_fold`, never random splits. Do not change the baseline architecture without explicit instruction.

## Working Agreement

See `rules.md`. In short: give Claude Code specs not goals; it writes code, the human runs training; every metric/Predictor gets a synthetic-data unit test; review diffs on metrics/conformal code closely — including cross-checking any class/tier list against this document, not just the code's internal logic; update this file on every design decision.

## Next Action

Contribution 1 calibration, sensitivity sweep, and all three paper figures are complete and locked. Next: Phase 1 of the baseline track — temperature scaling on the 5-class superdiagnostic model. Implement `src/metrics.py` with `temperature_scale()` (fit T on val logits, apply to test logits) and `expected_calibration_error()` (ECE, cite Guo et al. 2017). Report ECE before/after on the baseline model's val and test sets. Do not touch the acuity model or Contribution 1 code.
