import sys
import time
import cv2
import numpy as np
import joblib
from collections import Counter

# --------------------
# Session storage
# --------------------
engagement_counter = Counter()
confidence_list = []

print("STEP 1: Script started (MLP)")

# --------------------
# CONFIG paths
# --------------------
MODEL_PATH  = "models/v2_mlp_engagement.pkl"
SCALER_PATH = "models/v2_scaler_mlp_engagement.pkl"

RAW_FEATURES = 936

TARGET_FPS = 5
PROCESS_INTERVAL = 1.0 / TARGET_FPS

print("STEP 2: Config loaded")
print(f"Target FPS: {TARGET_FPS}")

# --------------------
# Feature extraction
# --------------------
def extract_features(frame):
    """
    Must match training (936).
    If you have real training extractor, replace this logic.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (36, 26))  # 36x26 == 936
    features = resized.flatten().astype(np.float32)

    if features.shape[0] != RAW_FEATURES:
        raise ValueError("Feature size mismatch")

    return features

# --------------------
# Load model + scaler
# --------------------
print("STEP 3: Loading MLP model")
mlp_model = joblib.load(MODEL_PATH)

print("STEP 4: Loading scaler")
scaler = joblib.load(SCALER_PATH)

print("STEP 5: Pipeline loaded")

# --------------------
# Camera init
# --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not opened")
    sys.exit(1)

print("STEP 6: Camera opened")

last_process_time = 0
display_text = "Initializing..."

print("STEP 7: Entering main loop")

# --------------------
# Main loop
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Frame not read")
        break

    current_time = time.time()

    if current_time - last_process_time >= PROCESS_INTERVAL:
        last_process_time = current_time

        # 1) Feature extraction
        features = extract_features(frame).reshape(1, -1)

        # 2) Scale
        features_scaled = scaler.transform(features)

        # 3) Predict
        pred = int(mlp_model.predict(features_scaled)[0])
        if hasattr(mlp_model, "predict_proba"):
            confidence = float(np.max(mlp_model.predict_proba(features_scaled)))
        else:
            confidence = 0.0

        engagement_counter[pred] += 1
        confidence_list.append(confidence)

        display_text = f"MLP Eng: {pred} | Conf: {confidence:.2f}"
        print(f"PREDICT → class={pred}, confidence={confidence:.3f}")

    cv2.putText(
        frame,
        display_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 200, 0),
        2
    )

    cv2.imshow("MLP Engagement Test (FPS Controlled)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("STOP requested by user")
        break

# --------------------
# Session summary
# --------------------
def print_session_summary(counter, confidences):
    print("\n===== SESSION SUMMARY (MLP) =====")

    total = sum(counter.values())
    if total == 0:
        print("No valid engagement detected.")
        return

    for level in sorted(counter.keys()):
        count = counter[level]
        percent = (count / total) * 100
        print(f"Engagement {level}: {count} frames ({percent:.2f}%)")

    low = counter.get(0, 0) + counter.get(1, 0)
    high = counter.get(2, 0) + counter.get(3, 0)

    print("\nLow Engagement Frames :", low)
    print("High Engagement Frames:", high)

    conclusion = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"
    avg_conf = sum(confidences) / len(confidences)

    print("\n===== FINAL CONCLUSION =====")
    print(f"Session Engagement Level : {conclusion}")
    print(f"Average Confidence       : {avg_conf:.3f}")

print_session_summary(engagement_counter, confidence_list)

cap.release()
cv2.destroyAllWindows()
print("STEP 8: Finished (MLP)")