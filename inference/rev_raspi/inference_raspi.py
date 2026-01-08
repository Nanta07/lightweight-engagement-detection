# inference/rev_raspi/inference_local.py
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
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkcalendar import Calendar

# CONFIGURATION
BASE_DIR = "processed_data_v2_1"
BASE_SESSION_DIR = r"C:\Users\Ananta\Documents\GitHub\lightweight-engagement-detection\data_sessions"

SCALER_PATH = f"{BASE_DIR}/preprocess/scaler.pkl"
PCA_PATH    = f"{BASE_DIR}/preprocess/pca.pkl"
MODEL_PATH  = f"{BASE_DIR}/models/mlp_engagement_v2_1.pkl"

TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

CAM_W, CAM_H = 640, 480
SIDEBAR_W = 300
FRAME_SIZE = (CAM_W, CAM_H)

FONT = cv2.FONT_HERSHEY_DUPLEX

# LOAD PIPELINE
try:
    scaler = joblib.load(SCALER_PATH)
    pca    = joblib.load(PCA_PATH)
    model  = joblib.load(MODEL_PATH)
except Exception as e:
    sys.exit(f"[ERROR] Failed loading pipeline: {e}")

# UI INPUT (DATE, NAME, SESSION)
root = tk.Tk()
root.withdraw()

session_date = None

def pick_date():
    global session_date
    win = tk.Toplevel(root)
    win.title("Session Setup - Date")
    win.geometry("300x300")
    win.grab_set()

    cal = Calendar(win, selectmode='day', date_pattern='yyyy-mm-dd')
    cal.pack(pady=10)

    def confirm():
        global session_date
        session_date = cal.get_date()
        win.destroy()

    tk.Button(win, text="OK", command=confirm).pack(pady=10)
    root.wait_window(win)

pick_date()

if not session_date:
    messagebox.showerror("Error", "Tanggal wajib dipilih!")
    sys.exit(1)

user_name = simpledialog.askstring("Session Setup", "Masukkan nama user:", parent=root)
if not user_name:
    messagebox.showerror("Error", "Nama user wajib diisi!")
    sys.exit(1)

session_id = simpledialog.askstring("Session Setup", "Sesi ke-:", parent=root)
if not session_id:
    messagebox.showerror("Error", "Session ID wajib diisi!")
    sys.exit(1)

print(f"[INFO] Session: {session_date} | {user_name} | Session-{session_id}")

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
video_path = os.path.join(session_root, "session_video.mp4")

# === ADDED: open CSV writer ONCE ===
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp", "Time", "Frame",
    "Engagement Level", "Confidence",
    "Response Time", "FPS"
])

# MEDIAPIPE INIT
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
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
            geo.append(np.linalg.norm(lm[GEOMETRY_INDEX[i]] - lm[GEOMETRY_INDEX[j]]))

    feat = np.hstack([lm.flatten(), geo]).reshape(1, -1)
    feat = scaler.transform(feat)
    feat = pca.transform(feat)
    return feat

# CAMERA & MAIN LOOP
cap = cv2.VideoCapture(0)
video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    TARGET_FPS,
    FRAME_SIZE
)

pred_buffer = deque(maxlen=5)
engagement_counter = Counter()
confidence_list = []

last_time = 0
pred, conf = 0, 0.0

# === ADDED ===
frame_id = 0
session_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    if now - last_time >= PROCESS_INTERVAL:
        last_time = now

        feat = extract_feature(frame)
        if feat is not None:
            raw_pred = int(model.predict(feat)[0])
            raw_conf = float(model.predict_proba(feat).max())

            pred_buffer.append(raw_pred)
            pred = Counter(pred_buffer).most_common(1)[0][0]
            conf = raw_conf

            engagement_counter[pred] += 1
            confidence_list.append(conf)

            # === ADDED: SAVE FRAME ===
            frame_name = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(
                os.path.join(engagement_dir, str(pred), frame_name),
                frame
            )

            # === ADDED: WRITE CSV ===
            csv_writer.writerow([
                int(now),
                round(now - session_start, 2),
                frame_id,
                pred,
                round(conf, 4),
                round(PROCESS_INTERVAL, 3),
                TARGET_FPS
            ])

            frame_id += 1

    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
cap.release()
video_writer.release()
csv_file.close()   # === ADDED ===
cv2.destroyAllWindows()
face_mesh.close()

print("[INFO] Session saved & outputs written correctly")