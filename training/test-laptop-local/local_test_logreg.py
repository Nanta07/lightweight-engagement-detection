import cv2
import mediapipe as mp
import numpy as np
import joblib

# ============================
# LOAD MODEL
# ============================

model  = joblib.load("v3_logreg_engagement.pkl")
scaler = joblib.load("v3_scaler_engagement.pkl")
pca    = joblib.load("v3_pca_engagement.pkl")

print("[OK] LogReg model loaded")

# ============================
# MEDIAPIPE
# ============================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================
# CAMERA
# ============================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        features = np.array([[p.x, p.y] for p in lm]).flatten()

        cv2.putText(frame, f"Landmarks: {len(lm)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        if features.shape[0] == 936:
            X = scaler.transform([features])
            X = pca.transform(X)

            probs = model.predict_proba(X)[0]
            pred  = model.predict(X)[0]
            conf  = np.max(probs)

            cv2.putText(frame, f"Pred: {pred}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,255), 2)

    cv2.imshow("LogReg Local Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()