import sys
import time
import cv2
import numpy as np
import joblib
from collections import Counter

# ===============================
# SESSION STORAGE
# ===============================
engagement_counter = Counter()
confidence_list = []

# ===============================
# CONFIG
# ===============================
MODEL_PATH  = "models/v3_logreg_engagement.pkl"
SCALER_PATH = "models/v3_scaler_engagement.pkl"
PCA_PATH    = "models/v3_pca_engagement.pkl"

RAW_FEATURES = 936
TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240

# ===============================
# UI UTILITIES
# ===============================
def draw_overlay(frame, lines, alpha=0.6):
    overlay = frame.copy()
    height = 30 + 30 * len(lines)
    cv2.rectangle(overlay, (5, 5), (310, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (15, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_confidence_bar(frame, confidence):
    x, y, w, h = 15, 140, 280, 18
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    fill = int(w * confidence)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), (0, 200, 0), -1)


def show_final_report(counter, confidences):
    report = np.zeros((420, 640, 3), dtype=np.uint8)

    total = sum(counter.values())
    low  = counter.get(0, 0) + counter.get(1, 0)
    high = counter.get(2, 0) + counter.get(3, 0)

    low_pct  = (low / total) * 100 if total else 0
    high_pct = (high / total) * 100 if total else 0

    conclusion = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"
    avg_conf = sum(confidences) / len(confidences)

    lines = [
        "ENGAGEMENT SESSION REPORT (LOGREG)",
        "",
        f"Low Engagement  : {low_pct:.2f} %",
        f"High Engagement : {high_pct:.2f} %",
        "",
        "FINAL RESULT:",
        conclusion,
        "",
        f"Average Confidence : {avg_conf:.3f}",
        "",
        "Press ESC to exit"
    ]

    y = 40
    for line in lines:
        cv2.putText(report, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        y += 32

    while True:
        cv2.imshow("Engagement Report", report)
        if cv2.waitKey(10) == 27:
            break


# ===============================
# FEATURE EXTRACTOR (LOCKED)
# ===============================
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (36, 26))
    features = resized.flatten().astype(np.float32)

    if features.shape[0] != RAW_FEATURES:
        raise ValueError("Feature size mismatch")

    return features


# ===============================
# LOAD PIPELINE
# ===============================
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca    = joblib.load(PCA_PATH)

# ===============================
# CAMERA INIT
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    sys.exit("[ERROR] Camera not opened")

last_process_time = 0
pred = 0
confidence = 0.0

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    if now - last_process_time >= PROCESS_INTERVAL:
        last_process_time = now

        features = extract_features(frame).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        pred = int(model.predict(features_pca)[0])
        confidence = float(np.max(model.predict_proba(features_pca)))

        engagement_counter[pred] += 1
        confidence_list.append(confidence)

    fps = 1.0 / PROCESS_INTERVAL
    label = "LOW" if pred <= 1 else "HIGH"

    draw_overlay(frame, [
        f"Engagement : {label}",
        f"Class      : {pred}",
        f"Confidence : {confidence:.2f}",
        f"FPS        : {fps:.1f}"
    ])

    draw_confidence_bar(frame, confidence)

    cv2.imshow("Engagement Detection - LOGREG", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
show_final_report(engagement_counter, confidence_list)
