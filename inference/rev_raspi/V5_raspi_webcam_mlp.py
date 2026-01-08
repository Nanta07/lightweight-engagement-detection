# V5_raspi_webcam_mlp.py

#!/usr/bin/env python3

# FINAL REALTIME INFERENCE PIPELINE v2.1 (RASPBERRY PI READY)
# Multiclass Engagement Detection (0–3)
import cv2
import time
import sys
import numpy as np
import joblib
import mediapipe as mp
from collections import Counter, deque

# Configuration
BASE_DIR = "processed_data_v2_1"

SCALER_PATH = f"{BASE_DIR}/preprocess/scaler.pkl"
PCA_PATH    = f"{BASE_DIR}/preprocess/pca.pkl"
MODEL_PATH  = f"{BASE_DIR}/models/mlp_engagement_v2_1.pkl"

TARGET_FPS = 4
PROCESS_INTERVAL = 1.0 / TARGET_FPS

FONT = cv2.FONT_HERSHEY_DUPLEX

# LOAD MODELS
try:
    scaler = joblib.load(SCALER_PATH)
    pca    = joblib.load(PCA_PATH)
    model  = joblib.load(MODEL_PATH)
except Exception as e:
    sys.exit(f"[ERROR] Failed to load model/preprocess files: {e}")

print("[INFO] Model, scaler, and PCA loaded successfully")

# MEDIAPIPE FACE MESH INIT (LIGHTWEIGHT)
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

# smoothing buffer (anti-flicker)
pred_buffer = deque(maxlen=5)

# FEATURE EXTRACTION
# landmark indices used in training (DO NOT CHANGE)
GEOMETRY_INDEX = [33, 133, 1, 61, 291, 199]

def extract_feature(frame):
    """
    Extract landmark + geometric features
    Output shape BEFORE scaler: (1, N_features)
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    # landmark (x, y)
    landmarks = np.array(
        [[p.x, p.y] for p in result.multi_face_landmarks[0].landmark],
        dtype=np.float32
    )

    # normalize (translation & scale invariant)
    landmarks -= landmarks.mean(axis=0)

    norm = np.linalg.norm(landmarks, axis=1).max()
    if norm < 1e-6:
        return None
    landmarks /= norm

    # geometric distances
    geo_feat = []
    for i in range(len(GEOMETRY_INDEX)):
        for j in range(i + 1, len(GEOMETRY_INDEX)):
            geo_feat.append(
                np.linalg.norm(
                    landmarks[GEOMETRY_INDEX[i]] -
                    landmarks[GEOMETRY_INDEX[j]]
                )
            )

    geo_feat = np.array(geo_feat, dtype=np.float32)

    # concatenate
    feat = np.hstack([landmarks.flatten(), geo_feat]).reshape(1, -1)

    # SAFETY CHECK (ACADEMIC VALIDATION)
    if feat.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature mismatch: got {feat.shape[1]}, "
            f"expected {scaler.n_features_in_}"
        )

    # preprocess (LOCKED PIPELINE)
    feat = scaler.transform(feat)
    feat = pca.transform(feat)

    return feat

# CAMERA INIT
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Camera not accessible")

last_time = 0
pred, conf = 0, 0.0

print("[INFO] Starting realtime engagement detection...")

#Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_time >= PROCESS_INTERVAL:
        last_time = now

        try:
            feat = extract_feature(frame)
            if feat is not None:
                raw_pred = int(model.predict(feat)[0])
                raw_conf = float(model.predict_proba(feat).max())

                # smoothing
                pred_buffer.append(raw_pred)
                pred = Counter(pred_buffer).most_common(1)[0][0]
                conf = raw_conf

                engagement_counter[pred] += 1
                confidence_list.append(conf)

        except Exception as e:
            print("[WARN]", e)

    # UI overlay
    cv2.putText(
        frame,
        f"Engagement Level : {pred} | Conf : {conf:.2f}",
        (20, 40),
        FONT,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Engagement Detection (MLP v2.1)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# FINAL SESSION REPORT
print("\n========== SESSION REPORT ==========")
total = sum(engagement_counter.values())

for level in range(4):
    print(f"Engagement Level {level} : {engagement_counter[level]}")

if confidence_list:
    print(f"Average Confidence : {np.mean(confidence_list):.3f}")

print("===================================")