"""
scripts/live_predict.py

Opens webcam, runs MediaPipe on every frame, feeds landmarks into
the trained RandomForest model, and speaks the prediction aloud.

Usage:
    python scripts/live_predict.py --lang ASL
    python scripts/live_predict.py --lang ISL
    python scripts/live_predict.py --lang ASL --no-tts   (silent mode)

Controls:
    Q       — quit
    S       — screenshot
    SPACE   — clear sentence
    1       — ISL mode
    2       — ASL mode
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
import os
import time
from collections import deque, Counter

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lang",   default="ASL", choices=["ASL","ISL"])
parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
args = parser.parse_args()

LANG   = args.lang
USE_TTS= not args.no_tts

# ── TTS setup ──────────────────────────────────────────────────────────────────
tts_engine = None
if USE_TTS:
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 150)   # speaking speed
        tts_engine.setProperty("volume", 1.0)
        print("[TTS] pyttsx3 ready")
    except Exception as e:
        print(f"[TTS] Could not init pyttsx3: {e} — running silent")
        USE_TTS = False

def speak(text):
    if tts_engine and USE_TTS:
        tts_engine.say(text)
        tts_engine.runAndWait()

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", f"{LANG.lower()}_classifier.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    print(f"  Run first: python scripts/train_model.py --lang {LANG}")
    exit(1)

print(f"[INFO] Loading model: {MODEL_PATH}")
model   = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
CLASSES = list(encoder.classes_)
print(f"[INFO] Loaded — {len(CLASSES)} classes: {CLASSES}\n")

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ── Webcam ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit(1)

os.makedirs("screenshots", exist_ok=True)

# ── Smoothing ──────────────────────────────────────────────────────────────────
SMOOTH_FACTOR  = 0.60
smoothed_pts   = {}

STABILITY_FRAMES = 20      # sign must be held for this many frames to confirm
CONFIRM_COOLDOWN = 1.5     # seconds between confirmed words

prediction_buffer = deque(maxlen=STABILITY_FRAMES)
last_confirmed    = ""
last_confirm_time = 0
sentence          = []

fps_buf   = deque(maxlen=30)
prev_time = time.time()
shot_n    = 0

# ── Helper: extract + normalize landmarks ─────────────────────────────────────
def extract_features(hand_lm):
    raw = []
    for lm in hand_lm.landmark:
        raw += [lm.x, lm.y, lm.z]
    wx, wy, wz = raw[0], raw[1], raw[2]
    normed = []
    for i in range(21):
        normed += [
            raw[i*3+0] - wx,
            raw[i*3+1] - wy,
            raw[i*3+2] - wz,
        ]
    return normed

def smooth_hand(idx, lm, w, h):
    pts = np.array([[p.x*w, p.y*h] for p in lm.landmark])
    if idx not in smoothed_pts:
        smoothed_pts[idx] = pts
    else:
        smoothed_pts[idx] = SMOOTH_FACTOR*smoothed_pts[idx] + (1-SMOOTH_FACTOR)*pts
    return smoothed_pts[idx]

# ── Finger colors for skeleton ─────────────────────────────────────────────────
FINGER_CHAINS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
FINGER_COLORS = [(0,210,255),(0,255,140),(255,180,0),(255,80,180),(120,140,255)]
FINGERTIPS    = {4,8,12,16,20}

def draw_hand(frame, pts):
    p = pts.astype(int)
    for (a,b) in mp_hands.HAND_CONNECTIONS:
        cv2.line(frame, tuple(p[a]), tuple(p[b]), (160,160,160), 1, cv2.LINE_AA)
    for chain, color in zip(FINGER_CHAINS, FINGER_COLORS):
        for i in range(len(chain)-1):
            cv2.line(frame, tuple(p[chain[i]]), tuple(p[chain[i+1]]), color, 3, cv2.LINE_AA)
    for i, pt in enumerate(p):
        r = 8 if i in FINGERTIPS else 5
        c = (0,255,220) if i in FINGERTIPS else (255,255,255)
        cv2.circle(frame, tuple(pt), r, c,  -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pt), r, (0,0,0), 1, cv2.LINE_AA)

print("[INFO] Running — Q=quit  SPACE=clear  S=screenshot")
print("[INFO] Hold a sign steady for ~1 second to confirm it\n")

# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # ── Prediction ────────────────────────────────────────────────────────────
    predicted_label = None
    confidence      = 0.0
    hand_detected   = False
    detected_ids    = set()

    if results.multi_hand_landmarks:
        for idx, lm in enumerate(results.multi_hand_landmarks):
            hand_detected = True
            detected_ids.add(idx)

            # Draw smoothed skeleton
            pts = smooth_hand(idx, lm, w, h)
            draw_hand(frame, pts)

            # Extract features and predict
            features = extract_features(lm)
            proba    = model.predict_proba([features])[0]
            pred_idx = np.argmax(proba)
            confidence      = proba[pred_idx]
            predicted_label = encoder.inverse_transform([pred_idx])[0]

    # Clean stale smoothed hands
    for k in list(smoothed_pts):
        if k not in detected_ids:
            del smoothed_pts[k]

    # ── Stability buffer: confirm only if same sign held long enough ──────────
    if predicted_label and confidence > 0.60:
        prediction_buffer.append(predicted_label)
    else:
        prediction_buffer.append(None)

    confirmed_now = None
    if len(prediction_buffer) == STABILITY_FRAMES:
        counts     = Counter(p for p in prediction_buffer if p is not None)
        if counts:
            top_sign, top_count = counts.most_common(1)[0]
            # Confirm if same sign appears in 80%+ of buffer frames
            if top_count >= STABILITY_FRAMES * 0.80:
                now = time.time()
                if (top_sign != last_confirmed or
                        now - last_confirm_time > CONFIRM_COOLDOWN):
                    confirmed_now   = top_sign
                    last_confirmed  = top_sign
                    last_confirm_time = now
                    sentence.append(top_sign)
                    print(f"[CONFIRMED] {top_sign}  (conf: {confidence:.0%})")
                    speak(top_sign)

    # ── FPS ───────────────────────────────────────────────────────────────────
    now = time.time()
    fps_buf.append(1 / (now - prev_time + 1e-9))
    prev_time = now
    fps = sum(fps_buf) / len(fps_buf)

    # ── HUD ───────────────────────────────────────────────────────────────────
    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0,0), (w,58), (0,0,0), -1)
    cv2.addWeighted(bar, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS {fps:.0f}",
                (14,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,150), 2, cv2.LINE_AA)

    # Live prediction label
    if predicted_label and hand_detected:
        conf_color = (0,220,100) if confidence > 0.80 else (0,180,255)
        cv2.putText(frame,
                    f"{predicted_label.upper()}  {confidence:.0%}",
                    (w//2 - 120, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, conf_color, 3, cv2.LINE_AA)

    # Lang badge
    badge_col = (30,100,210) if LANG=="ISL" else (180,80,20)
    cv2.rectangle(frame, (w-108,8), (w-8,44), badge_col, -1, cv2.LINE_AA)
    cv2.putText(frame, LANG, (w-92,34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Stability bar (fills as sign is held)
    stable_count = sum(1 for p in prediction_buffer if p == predicted_label)
    stable_frac  = stable_count / STABILITY_FRAMES if predicted_label else 0
    bar_w        = int((w - 40) * stable_frac)
    cv2.rectangle(frame, (20, 62), (w-20, 72), (50,50,50), -1)
    if bar_w > 0:
        color = (0,220,80) if stable_frac >= 0.80 else (0,150,255)
        cv2.rectangle(frame, (20,62), (20+bar_w,72), color, -1)

    # Confirmed flash
    if confirmed_now:
        flash = frame.copy()
        cv2.rectangle(flash, (0,0), (w,h), (0,200,80), -1)
        cv2.addWeighted(flash, 0.15, frame, 0.85, 0, frame)

    # Sentence box at bottom
    sent_str  = " ".join(sentence[-8:])   # last 8 words
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0,h-70), (w,h), (0,0,0), -1)
    cv2.addWeighted(bar2, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Sentence:", (14, h-46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(frame, sent_str if sent_str else "—",
                (14, h-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q:quit  SPACE:clear sentence  S:screenshot",
                (w-430, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130,130,130), 1, cv2.LINE_AA)

    if not hand_detected:
        cv2.putText(frame, "Show your hand",
                    (w//2-100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,255), 2, cv2.LINE_AA)

    cv2.imshow(f"Sign Language Translator — {LANG}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        sentence.clear()
        prediction_buffer.clear()
        last_confirmed = ""
        print("[CLEARED] Sentence reset")
    elif key == ord('s'):
        p = f"screenshots/predict_{shot_n:03d}.jpg"
        cv2.imwrite(p, frame)
        print(f"[SAVED] {p}")
        shot_n += 1
    elif key == ord('1'):
        LANG = "ISL"
        print("[MODE] Switched to ISL")
    elif key == ord('2'):
        LANG = "ASL"
        print("[MODE] Switched to ASL")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()

if sentence:
    full = " ".join(sentence)
    print(f"\n[SESSION] Full sentence: {full}")
    speak(full)

print("[INFO] Done.")
