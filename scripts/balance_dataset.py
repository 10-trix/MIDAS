"""
scripts/balance_dataset.py

Fixes class imbalance by capping every class at MAX_SAMPLES.
Also removes number classes (0-9) if you don't need them.

Run this BEFORE train_model.py.

Usage:
    python scripts/balance_dataset.py --lang ASL
    python scripts/balance_dataset.py --lang ASL --max 200 --keep-numbers
"""

import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lang",         default="ASL", choices=["ASL","ISL"])
parser.add_argument("--max",          type=int, default=200,
                    help="Max samples per class (default 200)")
parser.add_argument("--keep-numbers", action="store_true",
                    help="Keep number signs 0-9 (removed by default)")
args = parser.parse_args()

LANG        = args.lang
MAX_SAMPLES = args.max
KEEP_NUMS   = args.keep_numbers

IN_CSV  = os.path.join("data", "processed", f"final_train_{LANG}.csv")
OUT_CSV = os.path.join("data", "processed", f"final_train_{LANG}.csv")  # overwrite

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"\nLoading {IN_CSV} ...")
df = pd.read_csv(IN_CSV)
print(f"Before — {len(df)} samples, {df['label'].nunique()} classes")
print(f"\nBefore balance:")
print(df['label'].value_counts().to_string())

# ── Remove number classes (0-9) unless --keep-numbers ─────────────────────────
if not KEEP_NUMS:
    numbers = [str(i) for i in range(10)]
    before  = len(df)
    df      = df[~df['label'].isin(numbers)]
    print(f"\nRemoved number classes (0-9): {before - len(df)} rows dropped")

# ── Cap each class at MAX_SAMPLES ─────────────────────────────────────────────
df_balanced = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(min(len(x), MAX_SAMPLES), random_state=42))
      .reset_index(drop=True)
)

# ── Shuffle ────────────────────────────────────────────────────────────────────
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nAfter balance (max {MAX_SAMPLES} per class):")
vc = df_balanced['label'].value_counts().sort_index()
for label, count in vc.items():
    bar = "█" * int(count / MAX_SAMPLES * 20)
    flag = " ← LOW (collect more!)" if count < 100 else ""
    print(f"  {label:<20} {count:>5}  {bar}{flag}")

print(f"\nAfter  — {len(df_balanced)} samples, {df_balanced['label'].nunique()} classes")

# ── Save ───────────────────────────────────────────────────────────────────────
df_balanced.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")
print(f"\nNext: python scripts/train_model.py --lang {LANG}")
