import os
import time
import cv2
import csv
import joblib
import numpy as np
from collections import Counter

# ===============================
# CONFIG
# ===============================
BASE_DIR = "/home/pi/engagement_project"
MODEL_DIR = os.path.join(BASE_DIR, "models")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")

MODEL_PATH  = os.path.join(MODEL_DIR, "mlp_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_PATH    = os.path.join(MODEL_DIR, "pca.pkl")

RAW_FEATURES = 936
PCA_FEATURES = 64

FRAME_INTERVAL = 0.2   # 0.2s → ~5 FPS (AMAN untuk Raspi)

# ===============================
# INIT SESSION
# ===============================
timestamp = time.strftime("%Y%m%d_%H%M%S")
session_path = os.path.join(SESSION_DIR, f"session_{timestamp}")
os.makedirs(session_path, exist_ok=True)

eng_folder = os.path.join(session_path, "engagement")
for i in range(4):
    os.makedirs(os.path.join(eng_folder, str(i)), exist_ok=True)

csv_path = os.path.join(session_path, "engagement_log.csv")
video_path = os.path.join(session_path, "session_video.mp4")

# ===============================
# LOAD MODEL PIPELINE
# ===============================
print("[INFO] Loading MLP pipeline...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

print("[OK] Model loaded")
print("Model expects:", model.n_features_in_)

# ===============================
# FEATURE EXTRACTOR
# ===============================
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (36, 26))  # 36x26 = 936
    features = resized.flatten().astype(np.float32)

    if features.shape[0] != RAW_FEATURES:
        raise ValueError("Feature mismatch")

    return features

# ===============================
# CSV INIT
# ===============================
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "engagement_level",
        "confidence",
        "fps"
    ])

# ===============================
# CAMERA INIT
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, 10, (640, 480))

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

print("[OK] Camera started")

# ===============================
# SESSION STORAGE
# ===============================
counter = Counter()
conf_list = []

last_time = time.time()

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_time < FRAME_INTERVAL:
        continue
    last_time = now

    start = time.time()

    # Feature pipeline
    features = extract_features(frame).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Predict
    pred = int(model.predict(features_pca)[0])
    prob = model.predict_proba(features_pca)
    confidence = float(np.max(prob))

    fps = 1.0 / (time.time() - start)

    # Store stats
    counter[pred] += 1
    conf_list.append(confidence)

    # Save frame
    frame_name = f"{int(time.time()*1000)}.jpg"
    cv2.imwrite(
        os.path.join(eng_folder, str(pred), frame_name),
        frame
    )

    # CSV log
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            int(time.time()),
            pred,
            round(confidence, 3),
            round(fps, 2)
        ])

    # Overlay
    cv2.putText(
        frame,
        f"Engagement: {pred} | Conf: {confidence:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    video_writer.write(frame)
    cv2.imshow("RASPI MLP DEPLOY", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===============================
# SESSION SUMMARY
# ===============================
print("\n===== SESSION SUMMARY =====")
total = sum(counter.values())

for k in sorted(counter.keys()):
    print(f"Engagement {k}: {counter[k]} frames")

low = counter.get(0, 0) + counter.get(1, 0)
high = counter.get(2, 0) + counter.get(3, 0)

result = "HIGH ENGAGEMENT" if high > low else "LOW ENGAGEMENT"
avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0

print("Result :", result)
print("Avg confidence:", round(avg_conf, 3))

# ===============================
# CLEANUP
# ===============================
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("[DONE] Session saved at:", session_path)