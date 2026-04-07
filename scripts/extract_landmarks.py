"""
scripts/extract_landmarks.py

These images are pre-drawn MediaPipe skeletons (colored dots on black bg).
We extract landmark positions by detecting each colored dot's center pixel.

Strategy: detect all colored blobs, find their centers, normalize positions.
z is set to 0 (not recoverable from 2D skeleton images).

Usage:
    python scripts/extract_landmarks.py
"""

import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\apexc\MIDAS\data\processed\processed_combine_asl_dataset"
OUTPUT_CSV  = r"C:\Users\apexc\MIDAS\data\processed\asl_landmarks.csv"

# ── Column names: x0,y0,z0 ... x20,y20,z20, label ─────────────────────────────
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

# ── HSV color ranges for all dot colors visible in skeleton images ─────────────
DOT_COLORS_HSV = [
    (np.array([0,   150, 100]), np.array([10,  255, 255])),   # red
    (np.array([170, 150, 100]), np.array([180, 255, 255])),   # red (wrap)
    (np.array([11,  150, 100]), np.array([22,  255, 255])),   # orange
    (np.array([23,  150, 100]), np.array([35,  255, 255])),   # yellow
    (np.array([36,  80,  80]),  np.array([85,  255, 255])),   # green
    (np.array([86,  80,  80]),  np.array([130, 255, 255])),   # blue
    (np.array([131, 60,  60]),  np.array([165, 255, 255])),   # purple
    (np.array([0,   0,   180]), np.array([180, 40,  255])),   # white/cream
]

def get_dot_centers(img):
    """Detect all colored dot centers in a skeleton image."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lo, hi in DOT_COLORS_HSV:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # Close small gaps within each dot
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Find connected components
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    centers = []
    for i in range(1, len(centroids)):       # skip background
        if stats[i, cv2.CC_STAT_AREA] >= 10: # minimum dot size
            centers.append((float(centroids[i][0]), float(centroids[i][1])))

    return centers


def centers_to_feature_vector(centers, img_w, img_h):
    """
    Convert dot centers to a 63-value normalized feature vector.
    Sorts dots by position and pads/trims to exactly 21 landmarks.
    """
    if len(centers) < 5:
        return None

    # Sort top→bottom, left→right (consistent across images)
    centers_sorted = sorted(centers, key=lambda p: (int(p[1] / 15), p[0]))

    # Normalize to [0,1]
    normed = []
    for (cx, cy) in centers_sorted[:21]:     # take at most 21
        normed += [cx / img_w, cy / img_h, 0.0]

    # Pad to 63 if fewer than 21 dots found
    while len(normed) < 63:
        normed.append(0.0)

    # Subtract wrist (first point) to make position-independent
    wx, wy = normed[0], normed[1]
    for i in range(21):
        normed[i*3]     -= wx
        normed[i*3 + 1] -= wy

    return normed


# ── Main loop ──────────────────────────────────────────────────────────────────
rows        = []
skipped     = 0
total       = 0

sign_folders = sorted(os.listdir(DATASET_DIR))
print(f"Found {len(sign_folders)} sign classes: {sign_folders}\n")

for sign_label in sign_folders:
    folder_path = os.path.join(DATASET_DIR, sign_label)
    if not os.path.isdir(folder_path):
        continue

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    sign_count = 0
    for img_file in tqdm(image_files, desc=f"{sign_label.upper():>3}", ncols=72):
        img = cv2.imread(os.path.join(folder_path, img_file))
        if img is None:
            skipped += 1
            continue

        h, w   = img.shape[:2]
        centers = get_dot_centers(img)
        vec     = centers_to_feature_vector(centers, w, h)

        if vec is None:
            skipped += 1
            continue

        rows.append(vec + [sign_label])
        sign_count += 1
        total      += 1

    print(f"  {sign_label.upper()} → {sign_count} extracted\n")

# ── Save CSV ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("=" * 50)
print(f"Total extracted : {total}")
print(f"Skipped         : {skipped}")
print(f"Classes         : {df['label'].nunique()}")
print(f"\nSamples per class:")
print(df['label'].value_counts().to_string())
print(f"\nSaved → {OUTPUT_CSV}")
