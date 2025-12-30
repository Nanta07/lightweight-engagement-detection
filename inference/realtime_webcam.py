import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/logistic_regression_engagement.pkl"
CAMERA_INDEX = 0

LABEL_MAP = {
    0: "Not Engaged",
    1: "Low Engagement",
    2: "Engaged",
    3: "High Engagement"
}

# =========================
# LOAD MODEL
# =========================
print("[INFO] Loading model...")
model = joblib.load(MODEL_PATH)

# =========================
# MEDIAPIPE
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1
)

# =========================
# HELPER
# =========================
def extract_landmark_vector(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0]
    vector = []

    for lm in landmarks.landmark:
        vector.extend([lm.x, lm.y])

    if len(vector) != 936:
        return None

    return np.array(vector).reshape(1, -1)

# =========================
# CAMERA LOOP
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Camera not detected")

print("[INFO] Starting real-time inference...")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vec = extract_landmark_vector(frame)

    if vec is not None:
        probs = model.predict_proba(vec)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

        label_text = f"{LABEL_MAP[pred]} ({conf:.2f})"
    else:
        label_text = "Face Not Detected"

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay
    cv2.putText(frame, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Engagement Detection (LogReg)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()