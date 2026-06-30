"""Temperature-scaling selective-prediction baseline — 35-class acuity model.

Step 1: Fit scalar T via L-BFGS-B minimising multi-label NLL on val logits
        (Guo et al. 2017). Report per-class ECE before/after for all 35 classes
        on both val and test splits.
        → results/ece_temperature_scaling.csv

Step 2: Naive threshold per each of the 24 supported classes:
        largest λ such that empirical FNR_val(λ) ≤ tier α.
        No statistical correction. Evaluate λ on test.
        → results/baseline_temp_scaling.csv

Sanity check: stops with non-zero exit if overall val ECE does not decrease.

Run: python src/calibration/temperature_scaling.py
"""

import csv
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from src.metrics import expected_calibration_error

# ── paths ─────────────────────────────────────────────────────────────────────
CKP     = pathlib.Path("checkpoints")
RESULTS = pathlib.Path("results")
C1      = RESULTS / "contribution1"
RESULTS.mkdir(exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
val_logits  = np.load(CKP / "acuity_val_logits.npy")    # (N_val, 35)
val_labels  = np.load(CKP / "acuity_val_labels.npy")    # (N_val, 35)
test_logits = np.load(CKP / "acuity_test_logits.npy")   # (N_test, 35)
test_labels = np.load(CKP / "acuity_test_labels.npy")   # (N_test, 35)
class_names = json.load(open(CKP / "acuity_labels" / "class_names.json"))

# supported classes → tier / alpha (derived from primary_calibration.csv, not hardcoded)
supported = {}
for row in csv.DictReader(open(C1 / "primary_calibration.csv")):
    supported[row["class"]] = {
        "tier":      row["tier"],
        "alpha":     float(row["alpha"]),
        "n_val_pos": int(row["n_val_pos"]),
    }

# ── 1. Fit temperature T via L-BFGS-B ─────────────────────────────────────────

def _nll_and_grad(log_T_arr: np.ndarray,
                  logits: np.ndarray,
                  labels: np.ndarray):
    """NLL and analytic gradient w.r.t. log(T).

    p = sigmoid(logits/T), T = exp(log_T).
    dp/d(log_T) = p*(1-p)*(-logits/T).
    d(NLL)/d(log_T) = mean[(y - p) * logits / T].
    """
    T       = float(np.exp(log_T_arr[0]))
    p       = expit(logits / T)
    p       = np.clip(p, 1e-12, 1 - 1e-12)
    loss    = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    grad    = float(np.mean((labels - p) * logits / T))
    return loss, np.array([grad])


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Return optimal T > 0 (T=1 means no change needed)."""
    logits = val_logits.astype(np.float64)
    labels = val_labels.astype(np.float64)
    result = minimize(
        _nll_and_grad, x0=[0.0],
        args=(logits, labels),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
    )
    return float(np.exp(result.x[0]))


print("Fitting temperature T on acuity val logits (L-BFGS-B) …")
T = fit_temperature(val_logits, val_labels)
print(f"  T = {T:.6f}")

# ── 2. Per-class ECE before/after on val and test ─────────────────────────────

def per_class_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    return [
        expected_calibration_error(probs[:, c:c+1], labels[:, c:c+1], n_bins)
        for c in range(probs.shape[1])
    ]


val_probs_raw  = expit(val_logits)
val_probs_cal  = expit(val_logits  / T)
test_probs_raw = expit(test_logits)
test_probs_cal = expit(test_logits / T)

val_ece_before  = per_class_ece(val_probs_raw,  val_labels)
val_ece_after   = per_class_ece(val_probs_cal,  val_labels)
test_ece_before = per_class_ece(test_probs_raw, test_labels)
test_ece_after  = per_class_ece(test_probs_cal, test_labels)

# Sanity check: overall val ECE must decrease
overall_before = expected_calibration_error(val_probs_raw, val_labels)
overall_after  = expected_calibration_error(val_probs_cal, val_labels)
print(f"  Overall val ECE: {overall_before:.4f} → {overall_after:.4f}  "
      f"(Δ = {overall_after - overall_before:+.4f})")
ECE_INCREASE_TOL = 1e-5   # allow numerical noise; flag only genuine increases
if overall_after > overall_before + ECE_INCREASE_TOL:
    print("ERROR: overall val ECE increased after temperature scaling.")
    print("The fitting procedure may have failed. Stopping.")
    sys.exit(1)
if abs(T - 1.0) < 0.01:
    print("  NOTE: T ≈ 1.0 — acuity model is already well-calibrated; "
          "scaling has negligible effect.")
print("  Sanity check PASSED.")

# Save per-class ECE
ece_path = RESULTS / "ece_temperature_scaling.csv"
with open(ece_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "class", "ece_before", "ece_after", "delta_ece", "temperature"])
    for split, before, after in (
        ("val",  val_ece_before,  val_ece_after),
        ("test", test_ece_before, test_ece_after),
    ):
        for i, cls in enumerate(class_names):
            w.writerow([split, cls,
                        f"{before[i]:.6f}", f"{after[i]:.6f}",
                        f"{after[i] - before[i]:.6f}", f"{T:.6f}"])
print(f"  Saved: {ece_path}")

# Save calibrated probs
np.save(CKP / "acuity_val_probs_cal.npy",  val_probs_cal.astype(np.float32))
np.save(CKP / "acuity_test_probs_cal.npy", test_probs_cal.astype(np.float32))
print("  Saved acuity_val_probs_cal.npy, acuity_test_probs_cal.npy")

# ── 3. Naive threshold per supported class ────────────────────────────────────

def naive_threshold(scores_val: np.ndarray,
                    labels_val: np.ndarray,
                    alpha: float) -> float:
    """Largest λ s.t. empirical FNR_val(λ) ≤ alpha. No statistical correction.

    FNR(λ) = |{i : label_i=1, score_i < λ}| / |{i : label_i=1}|
    Strategy: sort positive scores ascending; allow floor(alpha * n_pos) misses.
    """
    pos_scores = np.sort(scores_val[labels_val == 1])
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0
    max_fn = int(np.floor(alpha * n_pos))   # max false negatives allowed
    if max_fn >= n_pos:
        return 1.0
    return float(pos_scores[max_fn])


def evaluate_threshold(scores_test: np.ndarray,
                       labels_test: np.ndarray,
                       lambda_c: float):
    """Return (test_fnr, flag_pct) for a given threshold."""
    n_pos = int(labels_test.sum())
    if n_pos == 0:
        return 0.0, float((scores_test >= lambda_c).mean() * 100)
    fn      = ((labels_test == 1) & (scores_test < lambda_c)).sum()
    test_fnr = float(fn) / n_pos
    flag_pct = float((scores_test >= lambda_c).mean()) * 100
    return test_fnr, flag_pct


print("\nComputing naive thresholds on temperature-scaled val probs …")
rows_out = []
n_violations = 0
for cls, info in supported.items():
    idx   = class_names.index(cls)
    alpha = info["alpha"]

    lambda_c          = naive_threshold(val_probs_cal[:, idx], val_labels[:, idx], alpha)
    test_fnr, flag_pct = evaluate_threshold(test_probs_cal[:, idx], test_labels[:, idx], lambda_c)
    violates           = test_fnr > alpha
    n_violations      += int(violates)

    rows_out.append({
        "class": cls, "tier": info["tier"], "alpha": alpha,
        "n_val_pos": info["n_val_pos"],
        "lambda_c":  round(lambda_c,  6),
        "test_fnr":  round(test_fnr,  6),
        "flag_pct":  round(flag_pct,  2),
        "violates_alpha": violates,
    })
    sym = "FAIL" if violates else "ok  "
    print(f"  [{sym}] {cls:12s}  α={alpha:.2f}  λ={lambda_c:.4f}"
          f"  test_fnr={test_fnr:.4f}  flag={flag_pct:.1f}%")

out_path = RESULTS / "baseline_temp_scaling.csv"
fields   = ["class", "tier", "alpha", "n_val_pos",
            "lambda_c", "test_fnr", "flag_pct", "violates_alpha"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print(f"\nTemp-scaling naive: {n_violations}/{len(rows_out)} classes violate "
      f"their target α on the test set.")
print(f"Saved: {out_path}")
