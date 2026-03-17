"""
scripts/test_mediapipe.py
Week 1 — Verify MediaPipe hand landmark detection works on your webcam.

Controls:
  Q     — quit
  S     — save screenshot
  1     — ISL mode
  2     — ASL mode
"""

import cv2
import mediapipe as mp
import time, os

mp_hands      = mp.solutions.hands
mp_drawing    = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)")
    exit(1)

os.makedirs("screenshots", exist_ok=True)
prev_time, mode, shot_n = time.time(), "ASL", 0

print("[INFO] Running. Press Q to quit | S to screenshot | 1 = ISL | 2 = ASL")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    hand_count = 0
    if results.multi_hand_landmarks:
        for lm, info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_count += 1
            label = info.classification[0].label
            conf  = info.classification[0].score

            mp_drawing.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw_style.get_default_hand_landmarks_style(),
                mp_draw_style.get_default_hand_connections_style(),
            )

            wrist = lm.landmark[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f"{label} ({conf:.2f})",
                        (wx - 30, wy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            print(f"\r[Wrist] x:{wrist.x:.3f}  y:{wrist.y:.3f}  z:{wrist.z:.3f}", end="", flush=True)

    fps = 1 / (time.time() - prev_time + 1e-9)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Hands: {hand_count}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)

    color = (255,150,50) if mode == "ISL" else (50,150,255)
    cv2.rectangle(frame, (w-110, 8), (w-8, 40), color, -1)
    cv2.putText(frame, f"Mode: {mode}", (w-105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, "Q:quit  S:screenshot  1:ISL  2:ASL",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    cv2.imshow("SLT — MediaPipe Test", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        p = f"screenshots/shot_{shot_n:03d}.jpg"
        cv2.imwrite(p, frame)
        print(f"\n[Saved] {p}")
        shot_n += 1
    elif key == ord('1'):
        mode = "ISL"
    elif key == ord('2'):
        mode = "ASL"

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n[INFO] Done.")
