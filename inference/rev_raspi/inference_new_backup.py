#!/usr/bin/env python3
import cv2
import time
import sys
import os
import csv
import numpy as np
import joblib
import mediapipe as mp
from collections import Counter, deque
from datetime import datetime

# CONFIG (RASPI SAFE)
BASE_DIR = "processed_data_v2_1"
BASE_SESSION_DIR = os.path.expanduser("~/engagement_sessions")

SCALER_PATH = f"{BASE_DIR}/preprocess/scaler.pkl"
PCA_PATH    = f"{BASE_DIR}/preprocess/pca.pkl"
MODEL_PATH  = f"{BASE_DIR}/models/mlp_engagement_v2_1.pkl"

TARGET_FPS = 3
PROCESS_INTERVAL = 1.0 / TARGET_FPS

CAM_W, CAM_H = 640, 480
SIDEBAR_W = 280
FRAME_SIZE = (CAM_W, CAM_H)

FONT = cv2.FONT_HERSHEY_DUPLEX

# LOAD MODEL PIPELINE
try:
    scaler = joblib.load(SCALER_PATH)
    pca    = joblib.load(PCA_PATH)
    model  = joblib.load(MODEL_PATH)
except Exception as e:
    sys.exit(f"[ERROR] Model pipeline gagal dimuat: {e}")

# SESSION INPUT (TERMINAL)
print("=== SESSION SETUP ===")
session_date = input("Tanggal (YYYY-MM-DD) [kosong = hari ini]: ").strip()
if not session_date:
    session_date = datetime.now().strftime("%Y-%m-%d")

user_name = input("Nama user: ").strip()
session_id = input("Sesi ke-: ").strip()

if not user_name or not session_id:
    sys.exit("[ERROR] Nama dan Session ID wajib diisi")

# SESSION DIRECTORY
session_root = os.path.join(
    BASE_SESSION_DIR,
    session_date,
    f"{user_name}_Session-{session_id}"
)

engagement_dir = os.path.join(session_root, "engagement")
os.makedirs(session_root, exist_ok=True)
for i in range(4):
    os.makedirs(os.path.join(engagement_dir, str(i)), exist_ok=True)

csv_path = os.path.join(session_root, "engagement_results.csv")
video_path = os.path.join(session_root, "session_video.avi")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Timestamp", "Frame",
        "Engagement Level", "Confidence", "FPS"
    ])

# MEDIAPIPE INIT (LIGHT)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

GEOMETRY_INDEX = [33, 133, 1, 61, 291, 199]

def extract_feature(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    lm = np.array([[p.x, p.y] for p in result.multi_face_landmarks[0].landmark])
    lm -= lm.mean(axis=0)
    scale = np.linalg.norm(lm, axis=1).max()
    if scale < 1e-6:
        return None
    lm /= scale

    geo = []
    for i in range(len(GEOMETRY_INDEX)):
        for j in range(i + 1, len(GEOMETRY_INDEX)):
            geo.append(np.linalg.norm(
                lm[GEOMETRY_INDEX[i]] - lm[GEOMETRY_INDEX[j]]
            ))

    feat = np.hstack([lm.flatten(), geo]).reshape(1, -1)
    feat = scaler.transform(feat)
    feat = pca.transform(feat)
    return feat

# CAMERA & WRITER
cap = cv2.VideoCapture(0)
video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    TARGET_FPS,
    FRAME_SIZE
)

pred_buffer = deque(maxlen=5)
engagement_counter = Counter()
confidence_list = []

last_process = 0
frame_count = 0
pred, conf = 0, 0.0

# MAIN LOOP
print("[INFO] Tekan Q untuk mengakhiri sesi")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    frame_count += 1

    if now - last_process >= PROCESS_INTERVAL:
        last_process = now
        feat = extract_feature(frame)

        if feat is not None:
            raw_pred = int(model.predict(feat)[0])
            raw_conf = float(model.predict_proba(feat).max())

            pred_buffer.append(raw_pred)
            pred = Counter(pred_buffer).most_common(1)[0][0]
            conf = raw_conf

            engagement_counter[pred] += 1
            confidence_list.append(conf)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(),
                    frame_count,
                    pred,
                    f"{conf:.4f}",
                    f"{TARGET_FPS:.2f}"
                ])

    video_writer.write(frame)
    cv2.imshow("Engagement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
cap.release()
video_writer.release()
cv2.destroyAllWindows()
face_mesh.close()

# FINAL REPORT (TERMINAL)
print("\n=== SESSION REPORT ===")
for lvl in range(4):
    print(f"Level {lvl}: {engagement_counter.get(lvl, 0)} frame")

if confidence_list:
    print(f"Average confidence: {np.mean(confidence_list):.4f}")

print(f"[INFO] Data tersimpan di: {session_root}")