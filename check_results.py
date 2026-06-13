import os
import pandas as pd

csv_files = []
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))

print("CSV files found:", len(csv_files))
for f in csv_files:
    print(" -", f)

print("\nSearching for CIFAR-100 / TASK_ADAPTER / bottleneck-related rows...\n")

keywords = [
    "cifar100",
    "TASK_ADAPTER",
    "task_adapter",
    "bottleneck",
    "0.4169",
    "0.1003",
    "0.1048",
    "0.1062"
]

for f in csv_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue

    text_df = df.astype(str)
    mask = pd.Series([False] * len(df))

    for kw in keywords:
        mask = mask | text_df.apply(
            lambda row: row.str.contains(kw, case=False, na=False).any(),
            axis=1
        )

    matched = df[mask]

    if len(matched) > 0:
        print("=" * 100)
        print("FILE:", f)
        print("COLUMNS:", list(df.columns))
        print(matched.to_string(index=False))
