import pandas as pd
from pathlib import Path

from src.data import load_ptbxl_metadata, aggregate_labels, load_signals, get_fold_splits

data_dir = Path("data/ptbxl")

meta = load_ptbxl_metadata(data_dir)

scp_statements = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    
labels = aggregate_labels(meta, scp_statements, task="super")


print(labels.shape)
# To see the sum of the binary class columns (excluding metadata columns)
label_cols = [c for c in labels.columns if c not in meta.columns and c != "label_primary"]
print(labels[label_cols].sum(axis=0))
