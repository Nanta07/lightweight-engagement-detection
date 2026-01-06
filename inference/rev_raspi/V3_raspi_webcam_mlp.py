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
MODEL_PATH  = "models/v2_mlp_engagement.pkl"
SCALER_PATH = "models/v2_scaler_mlp_engagement.pkl"

RAW_FEATURES = 936
TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

SIDEBAR_W = 300
MIN_WINDOW_W = 900
MIN_WINDOW_H = 500

FONT = cv2.FONT_HERSHEY_DUPLEX   # ← PALING AMAN & ENAK (tanpa install)

# =====================================================
# SESSION STORAGE
# =====================================================
engagement_counter = Counter()
confidence_list = []

# =====================================================
# FEATURE EXTRACTOR (LOCKED)
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

    cv2.putText(panel, "Status", (15, 90),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, label, (15, 130),
                FONT, 0.9, (255, 255, 255), 2)

    cv2.putText(panel, f"Class : {pred}", (15, 190),
                FONT, 0.7, (255, 255, 255), 1)

    cv2.putText(panel, "Confidence", (15, 230),
                FONT, 0.6, (200, 200, 200), 1)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 15, 250, 260, 20
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    fill_w = int(bar_w * conf)
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h), (220, 220, 220), -1)

    cv2.putText(panel, f"{conf:.2f}", (bar_x + 180, bar_y + 17),
                FONT, 0.6, (0, 0, 0), 1)

    cv2.putText(panel, f"FPS : {fps:.1f}", (15, 320),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, "Model : MLP", (15, 350),
                FONT, 0.6, (200, 200, 200), 1)

    cv2.putText(panel, "Press Q to Stop", (15, panel.shape[0] - 20),
                FONT, 0.6, (180, 180, 180), 1)

# =====================================================
# FINAL REPORT SCREEN
# =====================================================
def show_final_report(counter, confidences):
    screen = np.zeros((500, 900, 3), dtype=np.uint8)
    screen[:] = (30, 30, 30)

    total = sum(counter.values())
    low  = counter.get(0, 0) + counter.get(1, 0)
    high = counter.get(2, 0) + counter.get(3, 0)

    low_pct  = (low / total) * 100 if total else 0
    high_pct = (high / total) * 100 if total else 0
    avg_conf = sum(confidences) / len(confidences)

    result = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"

    lines = [
        "ENGAGEMENT SESSION SUMMARY",
        "",
        f"Low Engagement  : {low_pct:.2f} %",
        f"High Engagement : {high_pct:.2f} %",
        "",
        f"Final Result   : {result}",
        f"Avg Confidence : {avg_conf:.3f}",
        "",
        "Press ESC to Exit"
    ]

    y = 80
    for text in lines:
        cv2.putText(screen, text, (120, y),
                    FONT, 0.9, (255, 255, 255), 2)
        y += 45

    while True:
        cv2.imshow("Engagement Report", screen)
        if cv2.waitKey(10) == 27:
            break

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================================
# CAMERA INIT
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Camera not opened")

cv2.namedWindow("Engagement Detection - MLP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Engagement Detection - MLP", 1000, 600)

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

        feats = extract_features(frame).reshape(1, -1)
        feats = scaler.transform(feats)
        pred = int(model.predict(feats)[0])
        conf = float(np.max(model.predict_proba(feats)))

        engagement_counter[pred] += 1
        confidence_list.append(conf)

    fps = 1.0 / PROCESS_INTERVAL

    _, _, win_w, win_h = cv2.getWindowImageRect("Engagement Detection - MLP")
    win_w = max(win_w, MIN_WINDOW_W)
    win_h = max(win_h, MIN_WINDOW_H)

    cam_w = max(win_w - SIDEBAR_W, 1)
    cam_h = max(win_h, 1)

    cam_view = resize_with_aspect_ratio(frame, cam_w, cam_h)

    sidebar = np.zeros((cam_h, SIDEBAR_W, 3), dtype=np.uint8)
    draw_sidebar(sidebar, pred, conf, fps)

    ui = np.hstack((cam_view, sidebar))
    cv2.imshow("Engagement Detection - MLP", ui)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
show_final_report(engagement_counter, confidence_list)