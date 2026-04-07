"""
scripts/collect_data.py

Records hand landmark data from your webcam for model training.
Each sample = 63 numbers (21 landmarks x x,y,z) + label saved to CSV.

Usage:
    python scripts/collect_data.py --sign "hello" --samples 100 --lang ASL
    python scripts/collect_data.py --sign "namaste" --samples 100 --lang ISL

Controls during recording:
    SPACE  — start / pause recording
    Q      — quit and save
    R      — reset current session (discard and restart)
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
import os
import time

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sign",    required=True,  help="Sign label e.g. hello")
parser.add_argument("--samples", type=int, default=100, help="How many samples to collect")
parser.add_argument("--lang",    default="ASL",  choices=["ASL","ISL"])
args = parser.parse_args()

SIGN_LABEL   = args.sign.lower().strip()
TARGET_COUNT = args.samples
LANG         = args.lang

# ── Output path ────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join("data", "raw", LANG)
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"{SIGN_LABEL}.csv")

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ── Webcam ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit(1)

# ── Column names ───────────────────────────────────────────────────────────────
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

# ── State ──────────────────────────────────────────────────────────────────────
collected  = []
recording  = False
countdown  = 0
last_saved = 0

print(f"\n[INFO] Collecting '{SIGN_LABEL}' ({LANG}) — target: {TARGET_COUNT} samples")
print(f"[INFO] Press SPACE to start | Q to quit & save | R to reset\n")

def extract_landmarks(hand_landmarks, img_w, img_h):
    """Extract normalized 63-value feature vector from hand landmarks."""
    raw = []
    for lm in hand_landmarks.landmark:
        raw += [lm.x, lm.y, lm.z]

    # Normalize relative to wrist (landmark 0)
    wx, wy, wz = raw[0], raw[1], raw[2]
    normalized = []
    for i in range(21):
        normalized += [
            raw[i*3 + 0] - wx,
            raw[i*3 + 1] - wy,
            raw[i*3 + 2] - wz,
        ]
    return normalized

# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    h, w     = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results  = hands.process(rgb)
    rgb.flags.writeable = True

    hand_detected = False
    landmark_vec  = None

    if results.multi_hand_landmarks:
        hand_detected = True
        lm = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, lm, mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        landmark_vec = extract_landmarks(lm, w, h)

        # Auto-collect if recording
        if recording and landmark_vec:
            now = time.time()
            if now - last_saved > 0.05:   # max 20 samples/sec to avoid duplicates
                collected.append(landmark_vec + [SIGN_LABEL])
                last_saved = now

    # ── HUD ───────────────────────────────────────────────────────────────────
    count     = len(collected)
    progress  = count / TARGET_COUNT
    bar_w     = int((w - 40) * min(progress, 1.0))

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Sign name + count
    cv2.putText(frame, f"Sign: {SIGN_LABEL.upper()}  ({LANG})",
                (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{count} / {TARGET_COUNT}",
                (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,255,150) if recording else (180,180,180), 2, cv2.LINE_AA)

    # Progress bar
    cv2.rectangle(frame, (20, h-24), (w-20, h-10), (60,60,60), -1)
    if bar_w > 0:
        color = (0,200,100) if recording else (100,100,200)
        cv2.rectangle(frame, (20, h-24), (20+bar_w, h-10), color, -1)

    # Status badge
    if count >= TARGET_COUNT:
        badge = "DONE! Press Q to save"
        badge_col = (0, 180, 80)
        recording = False
    elif recording:
        badge = "RECORDING..." if hand_detected else "RECORDING — show your hand!"
        badge_col = (0, 100, 220) if hand_detected else (0, 100, 255)
    else:
        badge = "PAUSED — press SPACE to start"
        badge_col = (120, 120, 120)

    cv2.rectangle(frame, (w-360, 8), (w-8, 38), badge_col, -1, cv2.LINE_AA)
    cv2.putText(frame, badge, (w-354, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

    # No hand warning
    if not hand_detected:
        cv2.putText(frame, "No hand detected",
                    (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 2, cv2.LINE_AA)

    # Bottom hint
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h-34), (w, h-26), (0,0,0), -1)
    cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "SPACE: start/pause   Q: save & quit   R: reset",
                (12, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)

    cv2.imshow(f"Collecting: {SIGN_LABEL.upper()}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if count < TARGET_COUNT:
            recording = not recording
            print(f"[{'RECORDING' if recording else 'PAUSED'}]")
    elif key == ord('r'):
        collected = []
        recording = False
        print("[RESET] Cleared all samples")

# ── Save ───────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()

if collected:
    df_new = pd.DataFrame(collected, columns=columns)

    # Append to existing CSV if it already exists (from previous sessions)
    if os.path.exists(OUT_CSV):
        df_old = pd.read_csv(OUT_CSV)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
        print(f"\n[INFO] Appended to existing file")

    df_new.to_csv(OUT_CSV, index=False)
    print(f"[SAVED] {len(df_new)} total samples for '{SIGN_LABEL}' → {OUT_CSV}")
else:
    print("\n[INFO] No samples collected — nothing saved.")
