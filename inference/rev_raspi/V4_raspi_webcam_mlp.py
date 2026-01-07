#!/usr/bin/env python3
import sys
import time
import cv2
import numpy as np
import joblib
from collections import Counter

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = r"C:\Users\Ananta\Documents\GitHub\lightweight-engagement-detection\processed_data_v2\v2_1"

MODEL_PATH  = f"{BASE_DIR}/models/mlp_engagement_v2_1.pkl"
SCALER_PATH = f"{BASE_DIR}/scaler.pkl"
PCA_PATH    = f"{BASE_DIR}/pca.pkl"

RAW_FEATURES = 936
TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

SIDEBAR_W = 300
FONT = cv2.FONT_HERSHEY_DUPLEX

# =====================================================
# LOAD PIPELINE
# =====================================================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
except Exception as e:
    sys.exit(f"[ERROR] Failed loading model pipeline: {e}")

print("[INFO] Model, scaler, and PCA loaded successfully")

# =====================================================
# SESSION STORAGE
# =====================================================
engagement_counter = Counter()
confidence_list = []

# =====================================================
# FEATURE EXTRACTOR (LOCKED / SAME AS TRAINING)
# =====================================================
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (36, 26))
    features = resized.flatten().astype(np.float32)

    if features.shape[0] != RAW_FEATURES:
        raise ValueError("Feature size mismatch")

    return features

# =====================================================
# ASPECT RATIO SAFE RESIZE
# =====================================================
def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized

    return canvas

# =====================================================
# SIDEBAR UI
# =====================================================
def draw_sidebar(panel, pred, conf, fps):
    panel[:] = (30, 30, 30)
    label = "LOW ENGAGEMENT" if pred <= 1 else "HIGH ENGAGEMENT"

    cv2.putText(panel, "ENGAGEMENT MONITOR", (15, 40),
                FONT, 0.75, (255, 255, 255), 2)

    cv2.putText(panel, f"Status : {label}", (15, 100),
                FONT, 0.7, (255, 255, 255), 2)

    cv2.putText(panel, f"Class : {pred}", (15, 150),
                FONT, 0.7, (255, 255, 255), 1)

    cv2.putText(panel, "Confidence", (15, 200),
                FONT, 0.6, (200, 200, 200), 1)

    bar_x, bar_y, bar_w, bar_h = 15, 220, 260, 20
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + int(bar_w * conf), bar_y + bar_h),
                  (220, 220, 220), -1)

    cv2.putText(panel, f"{conf:.2f}", (bar_x + 180, bar_y + 16),
                FONT, 0.6, (0, 0, 0), 1)

    cv2.putText(panel, f"FPS : {fps:.1f}", (15, 280),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, "Model : MLP + PCA", (15, 310),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, "Press Q to Quit", (15, panel.shape[0] - 20),
                FONT, 0.6, (180, 180, 180), 1)

# =====================================================
# CAMERA INIT
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Camera not opened")

last_time = 0
pred, conf = 0, 0.0

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_time >= PROCESS_INTERVAL:
        last_time = now

        # Feature extraction
        feats = extract_features(frame).reshape(1, -1)

        # PIPELINE (MATCH TRAINING ORDER)
        feats_pca = pca.transform(feats)
        feats_scaled = scaler.transform(feats_pca)

        pred = int(model.predict(feats_scaled)[0])
        conf = float(np.max(model.predict_proba(feats_scaled)))

        engagement_counter[pred] += 1
        confidence_list.append(conf)

    fps = 1.0 / PROCESS_INTERVAL

    cam_view = resize_with_aspect_ratio(frame, 640, 480)
    sidebar = np.zeros((480, SIDEBAR_W, 3), dtype=np.uint8)
    draw_sidebar(sidebar, pred, conf, fps)

    ui = np.hstack((cam_view, sidebar))
    cv2.imshow("Engagement Detection v2.1", ui)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================
# FINAL REPORT
# =====================================================
def show_final_report(counter, confidences):
    screen = np.zeros((500, 900, 3), dtype=np.uint8)
    screen[:] = (30, 30, 30)

    total = sum(counter.values())
    avg_conf = sum(confidences) / max(len(confidences), 1)

    y = 80
    cv2.putText(screen, "ENGAGEMENT DETECTION REPORT", (120, y),
                FONT, 1.0, (255, 255, 255), 2)
    y += 50

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