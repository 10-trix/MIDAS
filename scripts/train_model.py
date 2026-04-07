"""
scripts/train_model.py

Trains a RandomForest classifier on the merged landmark CSV.
Saves the trained model + label encoder to models/

Usage:
    python scripts/train_model.py --lang ASL
    python scripts/train_model.py --lang ISL
    python scripts/train_model.py --lang ASL --trees 300
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
import time

from sklearn.ensemble               import RandomForestClassifier
from sklearn.model_selection        import train_test_split, cross_val_score
from sklearn.preprocessing          import LabelEncoder
from sklearn.metrics                import (accuracy_score,
                                            classification_report,
                                            confusion_matrix)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves PNG without display
import matplotlib.pyplot as plt
import seaborn as sns

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lang",  default="ASL", choices=["ASL", "ISL"])
parser.add_argument("--trees", type=int, default=300, help="Number of trees in forest")
args = parser.parse_args()

LANG  = args.lang
N_EST = args.trees

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_CSV   = os.path.join("data", "processed", f"final_train_{LANG}.csv")
MODEL_OUT  = os.path.join("models", f"{LANG.lower()}_classifier.pkl")
ENCODER_OUT= os.path.join("models", "label_encoder.pkl")
REPORT_OUT = os.path.join("models", f"{LANG.lower()}_confusion_matrix.png")
os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"\n{'='*54}")
print(f"  Training {LANG} Sign Language Classifier")
print(f"{'='*54}\n")

if not os.path.exists(DATA_CSV):
    print(f"[ERROR] Training data not found: {DATA_CSV}")
    print(f"  Run these first:")
    print(f"    python scripts/collect_data.py --sign <sign> --lang {LANG}")
    print(f"    python scripts/merge_datasets.py")
    exit(1)

print(f"[1] Loading data from {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV)
print(f"    Rows     : {len(df)}")
print(f"    Columns  : {df.shape[1]}")
print(f"    Classes  : {df['label'].nunique()}")
print(f"    Labels   : {sorted(df['label'].unique().tolist())}")

# ── Check class balance ────────────────────────────────────────────────────────
vc = df["label"].value_counts()
min_samples = vc.min()
max_samples = vc.max()
print(f"\n    Min samples per class : {min_samples} ({vc.idxmin()})")
print(f"    Max samples per class : {max_samples} ({vc.idxmax()})")

if min_samples < 20:
    print(f"\n[WARNING] Some classes have fewer than 20 samples.")
    print(f"    Low-sample classes: {vc[vc < 20].index.tolist()}")
    print(f"    Accuracy may be low for these. Collect more data for them.\n")

# ── Features and labels ────────────────────────────────────────────────────────
X = df.drop("label", axis=1).values.astype(np.float32)
y_raw = df["label"].values

# Encode string labels to integers
le = LabelEncoder()
y  = le.fit_transform(y_raw)

print(f"\n[2] Feature matrix shape : {X.shape}")
print(f"    Label classes         : {list(le.classes_)}")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Split → Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n[4] Training RandomForest (n_estimators={N_EST}) ...")
t0    = time.time()
model = RandomForestClassifier(
    n_estimators=N_EST,
    max_depth=None,         # let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",    # standard for classification
    n_jobs=-1,              # use all CPU cores
    random_state=42,
    class_weight="balanced" # handles class imbalance automatically
)
model.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"    Training time : {elapsed:.1f}s")

# ── Evaluate ───────────────────────────────────────────────────────────────────
print(f"\n[5] Evaluating on test set ...")
y_pred    = model.predict(X_test)
y_pred_lbl= le.inverse_transform(y_pred)
y_test_lbl= le.inverse_transform(y_test)

accuracy  = accuracy_score(y_test, y_pred)
print(f"\n    ┌─────────────────────────────┐")
print(f"    │  Test Accuracy : {accuracy*100:6.2f}%      │")
print(f"    └─────────────────────────────┘")

# Grade
if accuracy >= 0.97:
    grade = "Excellent — ready for demo"
elif accuracy >= 0.90:
    grade = "Good — ready for demo"
elif accuracy >= 0.80:
    grade = "Acceptable — collect more data"
else:
    grade = "Needs improvement — collect more data"
print(f"    Grade : {grade}\n")

# Per-class report
print(classification_report(y_test_lbl, y_pred_lbl, zero_division=0))

# ── Cross-validation (more reliable than single split) ────────────────────────
print(f"[6] Cross-validation (5-fold) ...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print(f"    CV scores : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"    CV mean   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n")

# ── Confusion matrix ───────────────────────────────────────────────────────────
print(f"[7] Saving confusion matrix → {REPORT_OUT}")
cm = confusion_matrix(y_test_lbl, y_pred_lbl, labels=le.classes_)
fig_size = max(10, len(le.classes_) // 2)
fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"{LANG} Sign Classifier — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%")
plt.tight_layout()
plt.savefig(REPORT_OUT, dpi=100)
plt.close()
print(f"    Saved → {REPORT_OUT}")

# ── Feature importance (top 10) ────────────────────────────────────────────────
importances = model.feature_importances_
feat_names  = [c for c in df.columns if c != "label"]
top_idx     = np.argsort(importances)[::-1][:10]
print(f"\n[8] Top 10 most important landmarks:")
for i, idx in enumerate(top_idx):
    print(f"    {i+1:>2}. {feat_names[idx]:<10}  importance: {importances[idx]:.4f}")

# ── Save model ─────────────────────────────────────────────────────────────────
print(f"\n[9] Saving model ...")
joblib.dump(model, MODEL_OUT)
joblib.dump(le,    ENCODER_OUT)
print(f"    Model saved   → {MODEL_OUT}")
print(f"    Encoder saved → {ENCODER_OUT}")

# Model size
size_kb = os.path.getsize(MODEL_OUT) / 1024
print(f"    Model size    : {size_kb:.1f} KB")

print(f"\n{'='*54}")
print(f"  TRAINING COMPLETE")
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  Model    : {MODEL_OUT}")
print(f"{'='*54}")
print(f"\nNext step: python scripts/live_predict.py --lang {LANG}\n")
