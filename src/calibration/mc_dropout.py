"""MC Dropout selective-prediction baseline — 35-class acuity model.

Enables Dropout at inference (BN stays in eval mode) and runs N=30 stochastic
forward passes on val/test. Mean probability across passes is used as the per-class
score. Applies the identical naive-threshold procedure as temperature_scaling.py —
only the score source differs.

Sanity check: reports mean absolute difference between MC-dropout mean probs and
the deterministic sigmoid(logits); exits if dropout appears inactive (MAD < 1e-5).

Outputs:
  checkpoints/acuity_val_mc_probs.npy   (N_val,  35)
  checkpoints/acuity_test_mc_probs.npy  (N_test, 35)
  results/baseline_mc_dropout.csv
  results/baseline_comparison.csv       (combined: LTT + temp-scaling + MC-dropout)

Run: python src/calibration/mc_dropout.py
     (requires temperature_scaling.py to have run first)
"""

import csv
import json
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from config import PTBXL_PATH, HELME_DATA, CHECKPOINT_DIR
from src.models.resnet1d import ECGResNet
from src.utils.data import load_ptbxl_metadata, get_fold_splits, apply_standardizer

# ── constants ─────────────────────────────────────────────────────────────────
N_PASSES     = 30
BATCH_SIZE   = 64
N_CLASSES    = 35

CKP     = pathlib.Path(CHECKPOINT_DIR)
RESULTS = pathlib.Path("results")
C1      = RESULTS / "contribution1"

# ── load supported classes / alpha targets from primary_calibration.csv ───────
supported = {}
for row in csv.DictReader(open(C1 / "primary_calibration.csv")):
    supported[row["class"]] = {
        "tier":      row["tier"],
        "alpha":     float(row["alpha"]),
        "n_val_pos": int(row["n_val_pos"]),
    }

class_names = json.load(open(CKP / "acuity_labels" / "class_names.json"))
val_logits  = np.load(CKP / "acuity_val_logits.npy")
val_labels  = np.load(CKP / "acuity_val_labels.npy")
test_labels = np.load(CKP / "acuity_test_labels.npy")

# ── 1. Load and preprocess signals (mirrors train_acuity.py) ──────────────────
print("Loading signals from X_raw.npy …")
X = np.load(CKP / "X_raw.npy")                           # (N, T, 12)  raw
with open(pathlib.Path(HELME_DATA) / "standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X = apply_standardizer(X, scaler)                        # (N, T, 12)  normalised
X = np.ascontiguousarray(X.transpose(0, 2, 1), dtype=np.float32)  # (N, 12, T)

meta   = load_ptbxl_metadata(PTBXL_PATH)
splits = get_fold_splits(meta)
X_val  = X[splits["val"]]    # (N_val,  12, 1000)
X_test = X[splits["test"]]   # (N_test, 12, 1000)
print(f"  X_val:  {X_val.shape}   X_test: {X_test.shape}")

assert X_val.shape[0]  == val_logits.shape[0],  "Val size mismatch"
assert X_test.shape[0] == test_labels.shape[0], "Test size mismatch"

# ── 2. Load model — BN in eval, Dropout in train ──────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = ECGResNet(num_classes=N_CLASSES).to(device)
model.load_state_dict(
    torch.load(CKP / "acuity_model.pt", map_location=device, weights_only=True)
)
model.eval()                          # BN uses running statistics
for m in model.modules():            # only Dropout layers go stochastic
    if isinstance(m, nn.Dropout):
        m.train()

# ── 3. N_PASSES stochastic forward passes ─────────────────────────────────────

def mc_probs(X_np: np.ndarray) -> np.ndarray:
    """Run N_PASSES stochastic passes; return mean sigmoid probs (N, 35)."""
    X_t  = torch.from_numpy(X_np)
    ds   = TensorDataset(X_t)
    dl   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    acc  = np.zeros((N_PASSES, X_np.shape[0], N_CLASSES), dtype=np.float32)

    with torch.no_grad():
        for p in range(N_PASSES):
            offset = 0
            for (batch,) in dl:
                logits = model(batch.to(device)).cpu().numpy()
                probs  = 1 / (1 + np.exp(-logits))
                n      = logits.shape[0]
                acc[p, offset:offset + n] = probs
                offset += n
            if (p + 1) % 10 == 0:
                print(f"    pass {p+1}/{N_PASSES}")
    return acc.mean(axis=0)   # (N, 35)


print(f"\nRunning {N_PASSES} stochastic passes on val set …")
val_mc_probs = mc_probs(X_val)
print(f"Running {N_PASSES} stochastic passes on test set …")
test_mc_probs = mc_probs(X_test)

# ── 4. Sanity check: MC probs must differ from deterministic ──────────────────
det_val_probs = 1 / (1 + np.exp(-val_logits))   # sigmoid of saved logits
mad = float(np.mean(np.abs(val_mc_probs - det_val_probs)))
print(f"\nSanity check: mean abs diff (MC-dropout vs deterministic) = {mad:.6f}")
if mad < 1e-5:
    print("ERROR: MC-dropout probs are nearly identical to deterministic probs.")
    print("Dropout may not be active. Check that model.train() is set on Dropout layers.")
    sys.exit(1)
print("  Sanity check PASSED: dropout is active.")

# Save MC probs
np.save(CKP / "acuity_val_mc_probs.npy",  val_mc_probs.astype(np.float32))
np.save(CKP / "acuity_test_mc_probs.npy", test_mc_probs.astype(np.float32))
print("  Saved acuity_val_mc_probs.npy, acuity_test_mc_probs.npy")

# ── 5. Naive threshold — identical logic to temperature_scaling.py ────────────

def naive_threshold(scores_val: np.ndarray,
                    labels_val: np.ndarray,
                    alpha: float) -> float:
    pos_scores = np.sort(scores_val[labels_val == 1])
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0
    max_fn = int(np.floor(alpha * n_pos))
    if max_fn >= n_pos:
        return 1.0
    return float(pos_scores[max_fn])


def evaluate_threshold(scores_test: np.ndarray,
                       labels_test: np.ndarray,
                       lambda_c: float):
    n_pos = int(labels_test.sum())
    if n_pos == 0:
        return 0.0, float((scores_test >= lambda_c).mean() * 100)
    fn       = ((labels_test == 1) & (scores_test < lambda_c)).sum()
    test_fnr = float(fn) / n_pos
    flag_pct = float((scores_test >= lambda_c).mean()) * 100
    return test_fnr, flag_pct


print("\nComputing naive thresholds on MC-dropout val probs …")
rows_out    = []
n_violations = 0
for cls, info in supported.items():
    idx   = class_names.index(cls)
    alpha = info["alpha"]

    lambda_c            = naive_threshold(val_mc_probs[:, idx], val_labels[:, idx], alpha)
    test_fnr, flag_pct  = evaluate_threshold(test_mc_probs[:, idx], test_labels[:, idx], lambda_c)
    violates             = test_fnr > alpha
    n_violations        += int(violates)

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

mc_path = RESULTS / "baseline_mc_dropout.csv"
fields  = ["class", "tier", "alpha", "n_val_pos",
           "lambda_c", "test_fnr", "flag_pct", "violates_alpha"]
with open(mc_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print(f"\nMC-dropout naive: {n_violations}/{len(rows_out)} classes violate "
      f"their target α on the test set.")
print(f"Saved: {mc_path}")

# ── 6. Baseline comparison CSV ────────────────────────────────────────────────
print("\nGenerating baseline_comparison.csv …")

ltt_rows = {r["class"]: r
            for r in csv.DictReader(open(C1 / "primary_calibration.csv"))}
ts_rows  = {r["class"]: r
            for r in csv.DictReader(open(RESULTS / "baseline_temp_scaling.csv"))}
mc_rows  = {r["class"]: r
            for r in csv.DictReader(open(mc_path))}

methods = [
    ("ltt_binomial",      ltt_rows),
    ("temp_scaling_naive", ts_rows),
    ("mc_dropout_naive",   mc_rows),
]

comp_rows   = []
violations  = {m: 0 for m, _ in methods}
for cls, info in supported.items():
    alpha = info["alpha"]
    tier  = info["tier"]
    for method, rows in methods:
        r        = rows[cls]
        test_fnr = float(r["test_fnr"])
        flag_pct = float(r["flag_pct"])
        violates = test_fnr > alpha
        violations[method] += int(violates)
        comp_rows.append({
            "class":          cls,
            "tier":           tier,
            "alpha":          alpha,
            "method":         method,
            "test_fnr":       round(test_fnr, 6),
            "flag_pct":       round(flag_pct, 2),
            "violates_alpha": violates,
        })

comp_path  = RESULTS / "baseline_comparison.csv"
comp_fields = ["class", "tier", "alpha", "method", "test_fnr", "flag_pct", "violates_alpha"]
with open(comp_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=comp_fields)
    w.writeheader()
    w.writerows(comp_rows)
print(f"Saved: {comp_path}")

n_cls = len(supported)
print(f"\n{'Method':<22s}  {'Violations':>10s}  {'Rate':>8s}")
print("-" * 46)
for method, _ in methods:
    v = violations[method]
    print(f"  {method:<20s}  {v:>4d}/{n_cls:<4d}   {v/n_cls*100:>5.1f}%")
