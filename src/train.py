import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import (
    PTBXL_PATH, HELME_DATA, CHECKPOINT_DIR,
    SAMPLING_RATE, BATCH_SIZE, EPOCHS, LR, NUM_CLASSES,
)
from src.utils.data import (
    load_ptbxl_metadata, load_signals, get_fold_splits,
    aggregate_labels, apply_standardizer,
)
from src.models.resnet1d import ECGResNet


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X must be contiguous (N, 12, T) float32; y must be (N, C) float32
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@torch.no_grad()
def get_logits_and_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for X_b, y_b in loader:
        all_logits.append(model(X_b.to(device)).cpu())
        all_labels.append(y_b)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def safe_macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Macro AUROC, skipping any class with only one label value in ground truth."""
    per_class = []
    for c in range(labels.shape[1]):
        if len(np.unique(labels[:, c])) < 2:
            continue
        per_class.append(roc_auc_score(labels[:, c], probs[:, c]))
    return float(np.mean(per_class)) if per_class else float("nan")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Metadata and splits
    meta = load_ptbxl_metadata(PTBXL_PATH)
    splits = get_fold_splits(meta)

    # 2. Raw signals — cached as .npy after first load (~2 min first time)
    cache_path = Path(CHECKPOINT_DIR) / "X_raw.npy"
    if cache_path.exists():
        print(f"Loading cached signals from {cache_path}")
        X = np.load(cache_path)
    else:
        print("Loading signals from disk (one-time, ~2 min)...")
        X = load_signals(meta, PTBXL_PATH, sampling_rate=SAMPLING_RATE)
        np.save(cache_path, X)
        print(f"Cached to {cache_path}")

    # 3. Superdiagnostic labels — aggregated from scp_codes via scp_statements.csv.
    #    helme's y_*.npy covers 21837 records (full PTB-XL) vs our 21799-record CSV,
    #    so those files cannot be used row-for-row. aggregate_labels reconstructs the
    #    same 5-class multi-hot encoding (CD, HYP, MI, NORM, STTC) from our records.
    scp = pd.read_csv(Path(PTBXL_PATH) / "scp_statements.csv", index_col=0)
    labelled = aggregate_labels(meta, scp, task="super")
    label_cols = [
        c for c in labelled.columns
        if c not in list(meta.columns) + ["label_primary"]
    ]
    print(f"Superdiagnostic classes ({len(label_cols)}): {label_cols}")
    Y = labelled[label_cols].values.astype(np.float32)
    y_train = Y[splits["train"]]
    y_val   = Y[splits["val"]]
    y_test  = Y[splits["test"]]

    # 4. Helme scaler
    with open(Path(HELME_DATA) / "standard_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    scaler_type = "global" if scaler.mean_.shape == (1,) else "per-channel"
    print(f"Scaler: mean_.shape={scaler.mean_.shape} ({scaler_type})")

    # 5. Standardize (channels-last, shape stays (N, T, 12))
    X = apply_standardizer(X, scaler)

    # 6. Transpose to (N, 12, T) — ascontiguousarray so torch.from_numpy works
    X = np.ascontiguousarray(X.transpose(0, 2, 1))

    # 7. Shape alignment checks — y is derived from the same meta, so counts must match
    assert X[splits["train"]].shape[0] == y_train.shape[0], (
        f"Train mismatch: {X[splits['train']].shape[0]} vs {y_train.shape[0]}"
    )
    assert X[splits["val"]].shape[0] == y_val.shape[0], (
        f"Val mismatch: {X[splits['val']].shape[0]} vs {y_val.shape[0]}"
    )
    assert X[splits["test"]].shape[0] == y_test.shape[0], (
        f"Test mismatch: {X[splits['test']].shape[0]} vs {y_test.shape[0]}"
    )
    assert y_train.shape[1] == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, got {y_train.shape[1]}"
    )
    print(
        f"Shape checks passed — X: {X.shape}, "
        f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}"
    )

    X_train, X_val, X_test = X[splits["train"]], X[splits["val"]], X[splits["test"]]

    # num_workers=0 required on macOS MPS (multiprocessing workers deadlock)
    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val),   batch_size=BATCH_SIZE,
                              num_workers=0)
    test_loader  = DataLoader(ECGDataset(X_test,  y_test),  batch_size=BATCH_SIZE,
                              num_workers=0)

    # Model
    model     = ECGResNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                           steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS}", leave=False, unit="batch")
        for X_b, y_b in bar:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        val_logits, val_labels = get_logits_and_labels(model, val_loader, device)
        val_probs = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
        val_auc   = safe_macro_auc(val_labels, val_probs)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss {total_loss/len(train_loader):.4f} | Val AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    # Final evaluation on test set
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=True)
    )
    test_logits, test_labels = get_logits_and_labels(model, test_loader, device)
    test_probs = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
    print(f"\nTest AUC (macro): {safe_macro_auc(test_labels, test_probs):.4f}")

    # Save everything calibration.py will need
    val_logits_final, _ = get_logits_and_labels(model, val_loader, device)
    np.save(os.path.join(CHECKPOINT_DIR, "val_logits.npy"),  val_logits_final)
    np.save(os.path.join(CHECKPOINT_DIR, "val_labels.npy"),  y_val)
    np.save(os.path.join(CHECKPOINT_DIR, "test_logits.npy"), test_logits)
    np.save(os.path.join(CHECKPOINT_DIR, "test_labels.npy"), test_labels)
    print("Saved logits — ready for calibration step.")


if __name__ == "__main__":
    main()
