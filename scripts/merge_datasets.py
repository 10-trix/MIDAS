"""
scripts/merge_datasets.py

Merges all data sources into one final training CSV:
  1. data/raw/ASL/*.csv        — your own recorded ASL signs
  2. data/raw/ISL/*.csv        — your own recorded ISL signs
  3. data/processed/asl_landmarks.csv  — downloaded ASL dataset (if exists)
  4. data/processed/Indian Sign Language Gesture Landmarks.csv — downloaded ISL dataset

Output:
  data/processed/final_train_ASL.csv
  data/processed/final_train_ISL.csv

Usage:
    python scripts/merge_datasets.py
"""

import pandas as pd
import os
import glob

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_ASL_DIR    = os.path.join("data", "raw", "ASL")
RAW_ISL_DIR    = os.path.join("data", "raw", "ISL")
PROCESSED_DIR  = os.path.join("data", "processed")

ASL_DOWNLOADED = os.path.join(PROCESSED_DIR, "asl_landmarks.csv")
ISL_DOWNLOADED = os.path.join(PROCESSED_DIR, "Indian Sign Language Gesture Landmarks.csv")

OUT_ASL = os.path.join(PROCESSED_DIR, "final_train_ASL.csv")
OUT_ISL = os.path.join(PROCESSED_DIR, "final_train_ISL.csv")

# ── Expected columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [f"{c}{i}" for i in range(21) for c in ["x","y","z"]]
LABEL_COL    = "label"
ALL_COLS     = FEATURE_COLS + [LABEL_COL]

def load_csv_safe(path, source_name):
    """Load a CSV, auto-detect label column, return clean dataframe."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [ERROR] Could not read {path}: {e}")
        return None

    print(f"  Loaded {source_name}: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns[:5])} ...")

    # Auto-detect label column (last column or column named 'label'/'class'/'sign')
    label_col = None
    for candidate in ["label", "class", "sign", "gesture", "word"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df.columns[-1]  # fallback: last column

    print(f"  Label column: '{label_col}'")

    # Rename label column to 'label'
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    # Keep only numeric feature columns + label
    feature_cols = [c for c in df.columns if c != "label"]
    df_clean = df[feature_cols + ["label"]].copy()

    # Drop rows with missing values
    before = len(df_clean)
    df_clean = df_clean.dropna()
    dropped = before - len(df_clean)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing values")

    # Normalize label: lowercase + strip whitespace
    df_clean["label"] = df_clean["label"].astype(str).str.lower().str.strip()

    print(f"  Classes: {df_clean['label'].nunique()} | Samples: {len(df_clean)}")
    return df_clean


def load_folder_csvs(folder, source_name):
    """Load all CSVs from a folder (each file = one sign class)."""
    if not os.path.exists(folder):
        print(f"  [SKIP] Folder not found: {folder}")
        return None

    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print(f"  [SKIP] No CSV files in {folder}")
        return None

    dfs = []
    for csv_path in sorted(csv_files):
        df = pd.read_csv(csv_path)
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df = df.dropna()
        dfs.append(df)
        sign = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"    {sign}: {len(df)} samples")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  {source_name}: {len(combined)} total samples, {combined['label'].nunique()} classes")
    return combined


def merge_and_save(frames, output_path, lang):
    """Merge list of dataframes, print summary, save to CSV."""
    valid = [df for df in frames if df is not None and len(df) > 0]

    if not valid:
        print(f"\n[WARNING] No data found for {lang} — nothing saved.")
        return

    merged = pd.concat(valid, ignore_index=True)

    # Drop duplicates
    before = len(merged)
    merged = merged.drop_duplicates()
    print(f"\n  Removed {before - len(merged)} duplicate rows")

    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    merged.to_csv(output_path, index=False)

    print(f"\n{'='*52}")
    print(f"  {lang} FINAL DATASET")
    print(f"  Total samples : {len(merged)}")
    print(f"  Total classes : {merged['label'].nunique()}")
    print(f"\n  Samples per class:")
    vc = merged["label"].value_counts()
    for label, count in vc.items():
        bar = "█" * min(int(count / max(vc) * 30), 30)
        print(f"    {label:<20} {count:>5}  {bar}")
    print(f"\n  Saved → {output_path}")
    print(f"{'='*52}\n")


# ── ASL ────────────────────────────────────────────────────────────────────────
print("\n" + "="*52)
print("  MERGING ASL DATA")
print("="*52)

asl_sources = []

print("\n[1] Your own ASL recordings:")
asl_sources.append(load_folder_csvs(RAW_ASL_DIR, "Own ASL recordings"))

print("\n[2] Downloaded ASL landmarks CSV:")
if os.path.exists(ASL_DOWNLOADED):
    asl_sources.append(load_csv_safe(ASL_DOWNLOADED, "Downloaded ASL"))
else:
    print(f"  [SKIP] Not found: {ASL_DOWNLOADED}")
    print(f"  Run extract_landmarks.py first if you want to include it")

merge_and_save(asl_sources, OUT_ASL, "ASL")

# ── ISL ────────────────────────────────────────────────────────────────────────
print("="*52)
print("  MERGING ISL DATA")
print("="*52)

isl_sources = []

print("\n[1] Your own ISL recordings:")
isl_sources.append(load_folder_csvs(RAW_ISL_DIR, "Own ISL recordings"))

print("\n[2] Downloaded ISL landmarks CSV:")
if os.path.exists(ISL_DOWNLOADED):
    isl_sources.append(load_csv_safe(ISL_DOWNLOADED, "Downloaded ISL"))
else:
    # Try alternate filename
    alt = os.path.join(PROCESSED_DIR, "isl_landmarks.csv")
    if os.path.exists(alt):
        isl_sources.append(load_csv_safe(alt, "Downloaded ISL"))
    else:
        print(f"  [SKIP] ISL CSV not found")
        print(f"  Expected: {ISL_DOWNLOADED}")

merge_and_save(isl_sources, OUT_ISL, "ISL")

print("Done! Next step: python scripts/train_model.py")
