"""
scripts/debug_images.py
Checks a sample of images to understand why MediaPipe detects 0 landmarks.
Run this BEFORE extract_landmarks.py to diagnose the problem.
"""

import cv2
import mediapipe as mp
import os
import numpy as np

DATASET_DIR = r"C:\Users\apexc\MIDAS\data\processed\processed_combine_asl_dataset"

mp_hands = mp.solutions.hands

# ── Test with different confidence thresholds ──────────────────────────────────
def test_image(img_path, label, confidence):
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=confidence,
    )
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Could not read image: {img_path}")
        return False

    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h} pixels")
    print(f"  Image dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

    # Try normal
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    detected = result.multi_hand_landmarks is not None
    print(f"  Confidence={confidence} → {'DETECTED' if detected else 'NOT DETECTED'}")

    # Try padded (add white border — helps MediaPipe find hands in tight crops)
    padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255,255,255])
    rgb_padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    result2 = hands.process(rgb_padded)
    detected2 = result2.multi_hand_landmarks is not None
    print(f"  Padded+Confidence={confidence} → {'DETECTED ✓' if detected2 else 'NOT DETECTED'}")

    # Try flipped
    flipped = cv2.flip(img, 1)
    rgb_flip = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    result3 = hands.process(rgb_flip)
    detected3 = result3.multi_hand_landmarks is not None
    print(f"  Flipped+Confidence={confidence} → {'DETECTED ✓' if detected3 else 'NOT DETECTED'}")

    # Try resized larger (MediaPipe works better on larger images)
    resized = cv2.resize(img, (400, 400))
    rgb_res = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    result4 = hands.process(rgb_res)
    detected4 = result4.multi_hand_landmarks is not None
    print(f"  Resized 400x400+Confidence={confidence} → {'DETECTED ✓' if detected4 else 'NOT DETECTED'}")

    hands.close()

    # Save a copy so you can visually inspect it
    save_path = f"debug_{label}_sample.jpg"
    cv2.imwrite(save_path, img)
    print(f"  Saved sample image → {save_path} (open this to see what the image looks like)")

    return detected or detected2 or detected3 or detected4

# ── Test one image from each of 3 folders ─────────────────────────────────────
test_labels = ['a', 'b', 'c']
print("=" * 60)
for label in test_labels:
    folder = os.path.join(DATASET_DIR, label)
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not images:
        print(f"[{label.upper()}] No images found")
        continue

    img_path = os.path.join(folder, images[0])
    print(f"\n[{label.upper()}] Testing: {images[0]}")
    for conf in [0.1, 0.3, 0.5]:
        test_image(img_path, label, conf)
    print("=" * 60)

print("\nCheck the saved debug_*.jpg files to see what the images look like.")
print("Then tell me what you see — are they hand photos or something else?")
