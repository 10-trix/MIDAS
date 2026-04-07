"""
scripts/test_mediapipe.py
Week 1 — MediaPipe hand landmark detection (smoothed + polished)

Controls:
  Q     — quit
  S     — save screenshot
  1     — ISL mode
  2     — ASL mode
  D     — toggle debug (show raw unsmoothed dots in red)
"""

import cv2
import mediapipe as mp
import time
import os
import numpy as np
from collections import deque

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,            # 0=fast 1=balanced — drop to 0 if laggy
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,  # higher = smoother
)

# ── Smoothing (Exponential Moving Average) ─────────────────────────────────────
# 0.0 = no smoothing (jittery), 1.0 = frozen
# 0.6 is the sweet spot — reduces jitter without adding noticeable lag
SMOOTH_FACTOR = 0.60
smoothed = {}   # { hand_idx: np.array (21, 2) }

def smooth_landmarks(hand_idx, landmarks, w, h):
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
    if hand_idx not in smoothed:
        smoothed[hand_idx] = pts
    else:
        smoothed[hand_idx] = SMOOTH_FACTOR * smoothed[hand_idx] + (1 - SMOOTH_FACTOR) * pts
    return smoothed[hand_idx]

# ── Webcam setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep buffer at 1 to reduce lag

if not cap.isOpened():
    print("[ERROR] Webcam not found. Try changing VideoCapture(0) to VideoCapture(1).")
    exit(1)

os.makedirs("screenshots", exist_ok=True)

# ── Hand connections ───────────────────────────────────────────────────────────
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
FINGERTIPS  = [4, 8, 12, 16, 20]

# Per-finger colors (BGR) — thumb, index, middle, ring, pinky
FINGER_COLORS = [
    (0,   210, 255),   # thumb  — amber
    (0,   255, 140),   # index  — green
    (255, 180,  0),    # middle — blue
    (255,  80, 180),   # ring   — pink
    (120, 140, 255),   # pinky  — lavender
]
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

def draw_hand(frame, pts, label, conf):
    """Draw colored finger skeleton + landmarks on smoothed points."""
    p = pts.astype(int)

    # Base connections in light gray (palm web)
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(p[a]), tuple(p[b]),
                 (160, 160, 160), 1, cv2.LINE_AA)

    # Colored finger lines (drawn on top)
    for chain, color in zip(FINGER_CHAINS, FINGER_COLORS):
        for i in range(len(chain) - 1):
            cv2.line(frame, tuple(p[chain[i]]), tuple(p[chain[i+1]]),
                     color, 3, cv2.LINE_AA)

    # Landmark dots
    for i, pt in enumerate(p):
        is_tip = i in FINGERTIPS
        radius = 8 if is_tip else 5
        color  = (0, 255, 220) if is_tip else (255, 255, 255)
        cv2.circle(frame, tuple(pt), radius, color,  -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pt), radius, (0,0,0), 1, cv2.LINE_AA)

    # Label near wrist
    wx, wy = p[0]
    cv2.putText(frame, f"{label}  {conf:.0%}",
                (wx - 40, wy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, fps, hand_count, mode, debug):
    h, w = frame.shape[:2]

    # Top bar — semi-transparent black
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"FPS  {fps:.0f}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hands  {hand_count}",
                (140, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    if debug:
        cv2.putText(frame, "DEBUG ON",
                    (290, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2, cv2.LINE_AA)

    # Mode badge top-right
    badge_col = (30, 100, 210) if mode == "ISL" else (180, 80, 20)
    cv2.rectangle(frame, (w - 112, 8), (w - 8, 44), badge_col, -1, cv2.LINE_AA)
    cv2.putText(frame, mode, (w - 90, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom hint bar
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "Q: quit   S: screenshot   1: ISL   2: ASL   D: debug",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)


# ── Main loop ──────────────────────────────────────────────────────────────────
fps_buf   = deque(maxlen=30)
prev_time = time.time()
mode      = "ASL"
debug     = False
shot_n    = 0

print("[INFO] Running — Q=quit  S=screenshot  1=ISL  2=ASL  D=debug")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    h, w      = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    hand_count   = 0
    detected_ids = set()

    if results.multi_hand_landmarks:
        for i, (lm, info) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            hand_count += 1
            detected_ids.add(i)
            label = info.classification[0].label
            conf  = info.classification[0].score

            pts = smooth_landmarks(i, lm, w, h)
            draw_hand(frame, pts, label, conf)

            if debug:
                for lmk in lm.landmark:
                    cv2.circle(frame,
                               (int(lmk.x * w), int(lmk.y * h)),
                               3, (0, 0, 255), -1)

    # Clean up stale smoothed state
    for k in list(smoothed.keys()):
        if k not in detected_ids:
            del smoothed[k]

    # FPS (rolling average)
    now = time.time()
    fps_buf.append(1 / (now - prev_time + 1e-9))
    prev_time = now
    fps = sum(fps_buf) / len(fps_buf)

    draw_hud(frame, fps, hand_count, mode, debug)

    cv2.imshow("Sign Language Translator — Week 1", frame)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('s'):
        p = f"screenshots/shot_{shot_n:03d}.jpg"
        cv2.imwrite(p, frame)
        print(f"[Saved] {p}")
        shot_n += 1
    elif key == ord('1'): mode = "ISL"; print("[Mode] ISL")
    elif key == ord('2'): mode = "ASL"; print("[Mode] ASL")
    elif key == ord('d'):
        debug = not debug
        print(f"[Debug] {'ON' if debug else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("[INFO] Done.")
