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

print("[INFO] RasPi Engagement Detection - Logistic Regression")

# ===============================
# CONFIG
# ===============================
MODEL_PATH  = "models/v3_logreg_engagement.pkl"
SCALER_PATH = "models/v3_scaler_engagement.pkl"
PCA_PATH    = "models/v3_pca_engagement.pkl"

RAW_FEATURES = 936
PCA_FEATURES = 64

# RasPi performance control
TARGET_FPS = 4                  # inference FPS
PROCESS_INTERVAL = 1.0 / TARGET_FPS

CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240

print(f"[CONFIG] Target inference FPS: {TARGET_FPS}")

# ===============================
# FEATURE EXTRACTOR (LOCKED)
# ===============================
def extract_features(frame):
    """
    MUST be identical to training & local test
    Output: 936 features
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (36, 26))  # 36 × 26 = 936
    features = resized.flatten().astype(np.float32)

    if features.shape[0] != RAW_FEATURES:
        raise ValueError("Feature size mismatch")

    return features

# ===============================
# LOAD PIPELINE
# ===============================
print("[LOAD] Logistic Regression model")
model = joblib.load(MODEL_PATH)

print("[LOAD] Scaler")
scaler = joblib.load(SCALER_PATH)

print("[LOAD] PCA")
pca = joblib.load(PCA_PATH)

print("[OK] Pipeline loaded")
print("[INFO] Model expects PCA features:", model.n_features_in_)

# ===============================
# CAMERA INIT
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("[ERROR] Camera not opened")
    sys.exit(1)

print("[OK] Camera opened")

last_process_time = 0
display_text = "Initializing..."

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame read failed")
        break

    now = time.time()

    # === CONTROLLED INFERENCE RATE ===
    if now - last_process_time >= PROCESS_INTERVAL:
        last_process_time = now

        # 1️⃣ Feature extraction
        features = extract_features(frame).reshape(1, -1)

        # 2️⃣ Scaling
        features_scaled = scaler.transform(features)

        # 3️⃣ PCA
        features_pca = pca.transform(features_scaled)

        # 4️⃣ Prediction
        pred = int(model.predict(features_pca)[0])

        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(features_pca)))
        else:
            confidence = 0.0

        # Store session statistics
        engagement_counter[pred] += 1
        confidence_list.append(confidence)

        display_text = f"Eng: {pred} | Conf: {confidence:.2f}"

        print(f"[PRED] class={pred}, conf={confidence:.3f}")

    # ===============================
    # VISUAL DISPLAY
    # ===============================
    cv2.putText(
        frame,
        display_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("RasPi Engagement - LOGREG", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Exit requested")
        break

# ===============================
# SESSION SUMMARY
# ===============================
def print_session_summary(counter, confidences):
    print("\n===== SESSION SUMMARY =====")

    total = sum(counter.values())
    if total == 0:
        print("No predictions made.")
        return

    for level in sorted(counter.keys()):
        count = counter[level]
        percent = (count / total) * 100
        print(f"Engagement {level}: {count} frames ({percent:.2f}%)")

    low  = counter.get(0, 0) + counter.get(1, 0)
    high = counter.get(2, 0) + counter.get(3, 0)

    conclusion = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"
    avg_conf = sum(confidences) / len(confidences)

    print("\n===== FINAL CONCLUSION =====")
    print(f"Session Engagement Level : {conclusion}")
    print(f"Average Confidence       : {avg_conf:.3f}")

print_session_summary(engagement_counter, confidence_list)

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("[DONE] Program finished")