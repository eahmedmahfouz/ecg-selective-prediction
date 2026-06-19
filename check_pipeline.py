"""
Dry-run: verify imports + data pipeline up to shape checks.
Does NOT build the model or run any training.
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

print("── imports ──────────────────────────────────────")
from config import PTBXL_PATH, HELME_DATA, CHECKPOINT_DIR, SAMPLING_RATE, NUM_CLASSES
from src.utils.data import (
    load_ptbxl_metadata, load_signals, get_fold_splits,
    aggregate_labels, apply_standardizer,
)
from src.models.resnet1d import ECGResNet
print("  OK")

print("── metadata + splits ────────────────────────────")
meta = load_ptbxl_metadata(PTBXL_PATH)
splits = get_fold_splits(meta)
print(f"  records: {len(meta)}")
print(f"  train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}")

print("── signals (cache) ──────────────────────────────")
cache_path = Path(CHECKPOINT_DIR) / "X_raw.npy"
if cache_path.exists():
    X = np.load(cache_path)
    print(f"  loaded from cache — shape: {X.shape}, dtype: {X.dtype}")
else:
    print("  cache missing — loading from disk (~2 min)...")
    import os; os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    X = load_signals(meta, PTBXL_PATH, sampling_rate=SAMPLING_RATE)
    np.save(cache_path, X)
    print(f"  cached — shape: {X.shape}")

print("── superdiagnostic labels ───────────────────────")
scp = pd.read_csv(Path(PTBXL_PATH) / "scp_statements.csv", index_col=0)
labelled = aggregate_labels(meta, scp, task="super")
label_cols = [c for c in labelled.columns if c not in list(meta.columns) + ["label_primary"]]
print(f"  classes ({len(label_cols)}): {label_cols}")
Y = labelled[label_cols].values.astype(np.float32)
y_train = Y[splits["train"]]
y_val   = Y[splits["val"]]
y_test  = Y[splits["test"]]
print(f"  y_train: {y_train.shape}  y_val: {y_val.shape}  y_test: {y_test.shape}")

print("── scaler ───────────────────────────────────────")
with open(Path(HELME_DATA) / "standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
scaler_type = "global" if scaler.mean_.shape == (1,) else "per-channel"
print(f"  mean_.shape: {scaler.mean_.shape} ({scaler_type})")

print("── standardize + transpose ──────────────────────")
X = apply_standardizer(X, scaler)
X = np.ascontiguousarray(X.transpose(0, 2, 1))
print(f"  X after: shape={X.shape}, dtype={X.dtype}, contiguous={X.flags['C_CONTIGUOUS']}")

print("── shape alignment assertions ───────────────────")
assert X[splits["train"]].shape[0] == y_train.shape[0], \
    f"train: {X[splits['train']].shape[0]} vs {y_train.shape[0]}"
assert X[splits["val"]].shape[0] == y_val.shape[0], \
    f"val: {X[splits['val']].shape[0]} vs {y_val.shape[0]}"
assert X[splits["test"]].shape[0] == y_test.shape[0], \
    f"test: {X[splits['test']].shape[0]} vs {y_test.shape[0]}"
assert y_train.shape[1] == NUM_CLASSES, \
    f"expected {NUM_CLASSES} classes, got {y_train.shape[1]}"
print("  OK")

print("── ECGDataset + one batch ───────────────────────")
from torch.utils.data import DataLoader
from src.train import ECGDataset
ds = ECGDataset(X[splits["val"]], y_val)
loader = DataLoader(ds, batch_size=8, num_workers=0)
X_b, y_b = next(iter(loader))
print(f"  X_b: {X_b.shape} {X_b.dtype}   y_b: {y_b.shape} {y_b.dtype}")

print("── model forward pass (CPU, 1 batch) ────────────")
model = ECGResNet(num_classes=NUM_CLASSES)
model.eval()
with torch.no_grad():
    logits = model(X_b)
print(f"  logits: {logits.shape} {logits.dtype}")

print("\n✓ pipeline check passed — safe to run training")
