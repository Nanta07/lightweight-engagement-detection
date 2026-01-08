#!/usr/bin/env python3

# FINAL REALTIME INFERENCE + SESSION UI v2.1 (RASPI READY)
# Multiclass Engagement Detection (0–3)

import cv2
import time
import sys
import numpy as np
import joblib
import mediapipe as mp
from collections import Counter, deque

# CONFIGURATION
BASE_DIR = "processed_data_v2_1"

SCALER_PATH = f"{BASE_DIR}/preprocess/scaler.pkl"
PCA_PATH    = f"{BASE_DIR}/preprocess/pca.pkl"
MODEL_PATH  = f"{BASE_DIR}/models/mlp_engagement_v2_1.pkl"

TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

SIDEBAR_W = 300
CAM_W, CAM_H = 640, 480

FONT = cv2.FONT_HERSHEY_DUPLEX

# LOAD PIPELINE
try:
    scaler = joblib.load(SCALER_PATH)
    pca    = joblib.load(PCA_PATH)
    model  = joblib.load(MODEL_PATH)
except Exception as e:
    sys.exit(f"[ERROR] Failed loading model pipeline: {e}")

print("[INFO] Model, scaler, and PCA loaded successfully")

# MEDIAPIPE INIT (LIGHTWEIGHT)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# SESSION STORAGE
engagement_counter = Counter()
confidence_list = []
pred_buffer = deque(maxlen=5)

# FEATURE EXTRACTION (MATCH TRAINING v2.1)
GEOMETRY_INDEX = [33, 133, 1, 61, 291, 199]

def extract_feature(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = np.array(
        [[p.x, p.y] for p in result.multi_face_landmarks[0].landmark],
        dtype=np.float32
    )

    # normalize
    landmarks -= landmarks.mean(axis=0)
    scale = np.linalg.norm(landmarks, axis=1).max()
    if scale < 1e-6:
        return None
    landmarks /= scale

    # geometric features
    geo_feat = []
    for i in range(len(GEOMETRY_INDEX)):
        for j in range(i + 1, len(GEOMETRY_INDEX)):
            geo_feat.append(
                np.linalg.norm(
                    landmarks[GEOMETRY_INDEX[i]] -
                    landmarks[GEOMETRY_INDEX[j]]
                )
            )

    feat = np.hstack([landmarks.flatten(), geo_feat]).reshape(1, -1)

    # safety check
    if feat.shape[1] != scaler.n_features_in_:
        raise ValueError("Feature dimension mismatch")

    feat = scaler.transform(feat)
    feat = pca.transform(feat)

    return feat

# UI UTILITIES
def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x, y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def draw_sidebar(panel, pred, conf, fps):
    panel[:] = (30, 30, 30)

    status = "LOW ENGAGEMENT" if pred <= 1 else "HIGH ENGAGEMENT"

    cv2.putText(panel, "ENGAGEMENT MONITOR", (15, 40),
                FONT, 0.75, (255, 255, 255), 2)

    cv2.putText(panel, f"Status : {status}", (15, 100),
                FONT, 0.7, (255, 255, 255), 2)

    cv2.putText(panel, f"Class : {pred}", (15, 150),
                FONT, 0.7, (255, 255, 255), 1)

    cv2.putText(panel, "Confidence", (15, 200),
                FONT, 0.6, (200, 200, 200), 1)

    bx, by, bw, bh = 15, 220, 260, 20
    cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (120, 120, 120), 1)
    cv2.rectangle(panel, (bx, by),
                  (bx + int(bw * conf), by + bh),
                  (220, 220, 220), -1)

    cv2.putText(panel, f"{conf:.2f}", (bx + 180, by + 16),
                FONT, 0.6, (0, 0, 0), 1)

    cv2.putText(panel, f"FPS : {fps:.1f}", (15, 280),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, "Model : MLP + PCA", (15, 310),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, "Press Q to Quit", (15, panel.shape[0] - 20),
                FONT, 0.6, (180, 180, 180), 1)

# CAMERA INIT
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Camera not accessible")

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

        feat = extract_feature(frame)
        if feat is not None:
            raw_pred = int(model.predict(feat)[0])
            raw_conf = float(model.predict_proba(feat).max())

            pred_buffer.append(raw_pred)
            pred = Counter(pred_buffer).most_common(1)[0][0]
            conf = raw_conf

            engagement_counter[pred] += 1
            confidence_list.append(conf)

    fps = 1.0 / PROCESS_INTERVAL

    cam_view = resize_with_aspect_ratio(frame, CAM_W, CAM_H)
    sidebar = np.zeros((CAM_H, SIDEBAR_W, 3), dtype=np.uint8)
    draw_sidebar(sidebar, pred, conf, fps)

    ui = np.hstack((cam_view, sidebar))
    cv2.imshow("Engagement Detection v2.1", ui)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# FINAL SESSION REPORT
def show_final_report(counter, confidences):
    screen = np.zeros((500, 900, 3), dtype=np.uint8)
    screen[:] = (30, 30, 30)

    avg_conf = np.mean(confidences) if confidences else 0.0
    y = 80

    cv2.putText(screen, "ENGAGEMENT DETECTION REPORT", (120, y),
                FONT, 1.0, (255, 255, 255), 2)
    y += 60

    for level in range(4):
        cv2.putText(screen, f"Level {level} : {counter.get(level, 0)}",
                    (200, y), FONT, 0.9, (255, 255, 255), 2)
        y += 40

    y += 20
    cv2.putText(screen, f"Avg Confidence : {avg_conf:.3f}",
                (200, y), FONT, 0.9, (255, 255, 255), 2)

    while True:
        cv2.imshow("Final Report", screen)
        if cv2.waitKey(10) == 27:
            break

show_final_report(engagement_counter, confidence_list)