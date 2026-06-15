import pandas as pd

df = pd.read_csv('data/ptbxl/ptbxl_database.csv', index_col='ecg_id')
print(f"Total rows in CSV: {len(df)}")

# Check if some filename_lr entries are missing/NaN
print(f"Missing filename_lr: {df['filename_lr'].isna().sum()}")

# Check fold distribution
print("\nRecords per fold:")
print(df['strat_fold'].value_counts().sort_index())

# Check which version you have
import os
version_file = 'data/ptbxl/SHA256SUMS.txt'
if os.path.exists(version_file):
    with open(version_file) as f:
        print(f"\nFirst line of SHA256SUMS: {f.readline().strip()}")
