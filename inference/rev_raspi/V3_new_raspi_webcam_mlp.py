# Updated Code is on ignore.py 

#Jumlah detection jangan terlalu banyak permenitnya saat in jumlah pengambilan selama 9 menit mendapatkan kurang lebih 1927 frame gambar (harus diperkecilagi agar ideal). Misal nanti perlu dibuat sesi untuk testing selama 1 menit persesi
#untuk session info ditambahkan lagi untuk date participant name, session keberapa, start dan end time. dan juga hasil dari evaluasi dari model selama sesi engagement yang telah dilakukan
# Total 5 sesi untuk setiap engagement level. Untuk setiap sesi selama 1 menit. Misal engagement 0, maka akan memeperagakan engagement level 0 dengan kondisi tertentu dan bermacam-macam pose. Tapi bagian ini hiraukan saja terlebih dahulu karena nanti setiap sesinya akan distop secara manual oleh pengguna
# Penyimpanan video perdurasi juga perlu dikurangi agar tidak terlalu besar, misal durasi 2/3 dari total sesi pengambilan data
# Berikan FPS yang ideal untuk pengambilan data dari sesi. Pertimbangkan yang ideal seberapa
# Buatkan tampilan UI sebelum menjalankan sesi pengambilan data dari user seperti data, nama dan sesi keberapa. Jadi lebih memudahkan untuk diinput oleh pengguna
# Berikan code final yang sudah fix, jangan ubah pada bagian yang tidak saya minta. INI SANGAT PENTING

#!/usr/bin/env python3
import sys
import os
import time
import csv
import cv2
import numpy as np
import joblib
from collections import Counter


# CONFIG

MODEL_PATH  = "models/v2_mlp_engagement.pkl"
SCALER_PATH = "models/v2_scaler_mlp_engagement.pkl"

RAW_FEATURES = 936
TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

SIDEBAR_W = 300
FONT = cv2.FONT_HERSHEY_DUPLEX


# SESSION METADATA INPUT

print("\n=== ENGAGEMENT SESSION SETUP ===")
SESSION_DATE = input("Enter session date (YYYY-MM-DD): ").strip() or time.strftime("%Y-%m-%d")
SESSION_NAME = input("Enter participant name       : ").strip() or "Unknown"
SESSION_NOTE = input("Enter session label/name     : ").strip() or "Session"

print("\n[INFO] Session Initialized")
print(f" Date        : {SESSION_DATE}")
print(f" Participant : {SESSION_NAME}")
print(f" Session     : {SESSION_NOTE}")
input("[INFO] Press ENTER to start detection...")


# SESSION OUTPUT DIR

SESSION_ID = time.strftime("%H%M%S")
BASE_DIR = f"sessions/{SESSION_DATE}_{SESSION_NAME}_{SESSION_NOTE}_{SESSION_ID}"

VIDEO_PATH = os.path.join(BASE_DIR, "session_video.mp4")
CSV_PATH   = os.path.join(BASE_DIR, "engagement_results.csv")
FRAME_DIR  = os.path.join(BASE_DIR, "frames")

for i in range(4):
    os.makedirs(os.path.join(FRAME_DIR, str(i)), exist_ok=True)


# SAVE SESSION INFO

with open(os.path.join(BASE_DIR, "session_info.txt"), "w") as f:
    f.write("ENGAGEMENT SESSION INFO\n")
    f.write("=======================\n")
    f.write(f"Date        : {SESSION_DATE}\n")
    f.write(f"Participant : {SESSION_NAME}\n")
    f.write(f"Session     : {SESSION_NOTE}\n")
    f.write(f"Start Time  : {time.strftime('%H:%M:%S')}\n")


# FEATURE EXTRACTOR (LOCKED)

def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (36, 26))
    features = resized.flatten().astype(np.float32)
    if features.shape[0] != RAW_FEATURES:
        raise ValueError("Feature size mismatch")
    return features


# RESIZE

def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x, y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas


# SIDEBAR UI (IDENTIK REFERENSI)

def draw_sidebar(panel, pred, conf, fps):
    panel[:] = (30, 30, 30)
    label = "LOW ENGAGEMENT" if pred <= 1 else "HIGH ENGAGEMENT"

    cv2.putText(panel, "ENGAGEMENT MONITOR", (15, 40),
                FONT, 0.75, (255,255,255), 2)
    cv2.putText(panel, label, (15, 120),
                FONT, 0.9, (255,255,255), 2)
    cv2.putText(panel, f"Class : {pred}", (15, 180),
                FONT, 0.7, (255,255,255), 1)
    cv2.putText(panel, f"Conf  : {conf:.2f}", (15, 230),
                FONT, 0.7, (255,255,255), 1)
    cv2.putText(panel, f"FPS   : {fps:.1f}", (15, 280),
                FONT, 0.7, (255,255,255), 1)
    cv2.putText(panel, "Model : MLP", (15, 330),
                FONT, 0.6, (200,200,200), 1)
    cv2.putText(panel, "Press Q to Stop", (15, panel.shape[0]-20),
                FONT, 0.6, (180,180,180), 1)


# FINAL REPORT (MATCHING REFERENSI)

def show_final_report(counter, confs, fps_list, rt_list):
    screen = np.zeros((540, 900, 3), dtype=np.uint8)
    screen[:] = (30,30,30)

    total = sum(counter.values())
    low = counter.get(0,0) + counter.get(1,0)
    high = counter.get(2,0) + counter.get(3,0)

    avg_conf = np.mean(confs)
    avg_fps  = np.mean(fps_list)
    avg_rt   = np.mean(rt_list)

    result = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"

    lines = [
        "ENGAGEMENT MODEL EVALUATION",
        "",
        f"Date        : {SESSION_DATE}",
        f"Participant : {SESSION_NAME}",
        f"Session     : {SESSION_NOTE}",
        "",
        f"Total Frames      : {total}",
        f"Low Engagement    : {(low/total)*100:.2f} %",
        f"High Engagement   : {(high/total)*100:.2f} %",
        "",
        f"Avg Confidence    : {avg_conf:.3f}",
        f"Avg FPS           : {avg_fps:.2f}",
        f"Avg Response Time : {avg_rt:.4f}s",
        "",
        f"FINAL RESULT : {result}",
        "",
        "Press ESC to Exit"
    ]

    y = 60
    for t in lines:
        cv2.putText(screen, t, (100, y),
                    FONT, 0.8, (255,255,255), 2)
        y += 40

    while True:
        cv2.imshow("Final Report", screen)
        if cv2.waitKey(10) == 27:
            break


# LOAD PIPELINE

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# CAMERA & VIDEO

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    sys.exit("[ERROR] Camera not available")

h, w = frame.shape[:2]

video_writer = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    TARGET_FPS,
    (w, h)
)


# CSV

csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp","Time","Frame",
    "Engagement Level","Confidence",
    "Response Time","FPS"
])


# RUNTIME STORAGE

counter = Counter()
conf_list = []
fps_list = []
rt_list = []

last_time = 0
pred, conf = 0, 0.0


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_time >= PROCESS_INTERVAL:
        last_time = now

        start = time.time()
        feats = extract_features(frame).reshape(1, -1)
        feats = scaler.transform(feats)

        pred = int(model.predict(feats)[0])
        conf = float(np.max(model.predict_proba(feats)))

        rt = time.time() - start
        fps = 1.0 / rt if rt > 0 else 0

        ts_ms = int(now * 1000)
        frame_name = f"frame_{ts_ms}.jpg"

        cv2.imwrite(
            os.path.join(FRAME_DIR, str(pred), frame_name),
            frame
        )

        csv_writer.writerow([
            ts_ms,
            time.strftime("%H:%M:%S", time.localtime(now)),
            frame_name,
            pred,
            round(conf,4),
            round(rt,4),
            round(fps,2)
        ])

        counter[pred] += 1
        conf_list.append(conf)
        fps_list.append(fps)
        rt_list.append(rt)

    video_writer.write(frame)

    cam_view = resize_with_aspect_ratio(frame, w - SIDEBAR_W, h)
    sidebar = np.zeros((h, SIDEBAR_W, 3), dtype=np.uint8)
    draw_sidebar(sidebar, pred, conf, TARGET_FPS)

    ui = np.hstack((cam_view, sidebar))
    cv2.imshow("Engagement Detection - MLP", ui)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP

cap.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()

show_final_report(counter, conf_list, fps_list, rt_list)