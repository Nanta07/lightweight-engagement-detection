#File name: inference/rev_raspi/completefunction_webcam_logreg.py
#Complete function for Raspberry Pi webcam engagement detection using Logistic Regression model
#!/usr/bin/env python3

import os
import time
import cv2
import mediapipe as mp
import joblib
import csv
import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar
from PIL import Image, ImageTk
import numpy as np

# ============================================================
# GLOBAL CONFIG
# ============================================================
BASE_FOLDER = "/home/pi/engagement_sessions"
MODEL_PATH  = "models/v3_logreg_engagement.pkl"
SCALER_PATH = "models/v3_scaler_engagement.pkl"
PCA_PATH    = "models/v3_pca_engagement.pkl"

RAW_FEATURES = 936
TARGET_FPS = 4
FRAME_INTERVAL = 1.0 / TARGET_FPS

csv_file_path = None
video_writer = None
stop_recording = False

# ============================================================
# INIT MEDIAPIPE
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)

# ============================================================
# LOAD MODEL PIPELINE
# ============================================================
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca    = joblib.load(PCA_PATH)

# ============================================================
# UTILITIES
# ============================================================
def format_timestamp(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))

def extract_features_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face = result.multi_face_landmarks[0]
    landmarks = [(lm.x, lm.y) for lm in face.landmark]
    features = np.array(landmarks).flatten().astype(np.float32)

    if features.shape[0] != RAW_FEATURES:
        return None

    return features.reshape(1, -1)

# ============================================================
# SESSION START
# ============================================================
def start_session(selected_date):
    global csv_file_path, video_writer, stop_recording
    stop_recording = False

    session_folder = os.path.join(BASE_FOLDER, f"session_{selected_date}")
    os.makedirs(session_folder, exist_ok=True)

    engagement_dir = os.path.join(session_folder, "engagement")
    for lvl in ["0", "1", "2", "3"]:
        os.makedirs(os.path.join(engagement_dir, lvl), exist_ok=True)

    csv_file_path = os.path.join(session_folder, "engagement_results.csv")
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Time", "Frame",
            "Engagement", "Confidence",
            "ResponseTime", "FPS"
        ])

    video_path = os.path.join(session_folder, "session_video.mp4")
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (640, 480)
    )

    capture_frames(session_folder)

# ============================================================
# CAMERA LOOP
# ============================================================
def capture_frames(session_folder):
    global stop_recording

    cap = cv2.VideoCapture(0)
    last_time = 0

    cam_window = tk.Toplevel()
    cam_window.title("Engagement Camera")

    label = tk.Label(cam_window)
    label.pack()

    tk.Button(
        cam_window,
        text="Stop Session",
        command=lambda: stop_camera(cam_window)
    ).pack(pady=10)

    def loop():
        nonlocal last_time

        if stop_recording:
            return

        ret, frame = cap.read()
        if not ret:
            return

        now = time.time()
        if now - last_time >= FRAME_INTERVAL:
            last_time = now
            start = time.time()

            features = extract_features_from_frame(frame)
            if features is not None:
                scaled = scaler.transform(features)
                pca_feat = pca.transform(scaled)

                pred = int(model.predict(pca_feat)[0])
                conf = float(np.max(model.predict_proba(pca_feat)))

                ts = int(time.time() * 1000)
                fname = f"frame_{ts}.jpg"

                folder = os.path.join(
                    session_folder, "engagement", str(pred)
                )
                cv2.imwrite(os.path.join(folder, fname), frame)

                resp = time.time() - start
                fps = 1 / resp if resp > 0 else 0

                with open(csv_file_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        ts,
                        format_timestamp(ts // 1000),
                        fname,
                        pred,
                        conf,
                        resp,
                        fps
                    ])

                video_writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=img)
        label.image = img

        cam_window.after(10, loop)

    loop()
    cam_window.mainloop()

    cap.release()
    video_writer.release()
    display_summary()

# ============================================================
# STOP & SUMMARY
# ============================================================
def stop_camera(win):
    global stop_recording
    stop_recording = True
    win.destroy()

def display_summary():
    counts = {"0":0,"1":0,"2":0,"3":0}
    with open(csv_file_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            lvl = r["Engagement"]
            counts[str(lvl)] += 1

    low = counts["0"] + counts["1"]
    high = counts["2"] + counts["3"]

    result = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"

    messagebox.showinfo(
        "Session Result",
        f"""
Engagement 0: {counts['0']}
Engagement 1: {counts['1']}
Engagement 2: {counts['2']}
Engagement 3: {counts['3']}

FINAL RESULT: {result}
"""
    )

# ============================================================
# TKINTER UI
# ============================================================
root = tk.Tk()
root.title("Engagement Session")

tk.Label(root, text="Select Date").pack(pady=5)
calendar = Calendar(root, date_pattern="yyyy-mm-dd")
calendar.pack(pady=10)

tk.Button(
    root,
    text="Start Session",
    command=lambda: start_session(calendar.get_date())
).pack(pady=20)

root.mainloop()