#extract_landmarks.py

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


def extract_landmark_vector(image_path):
    print("[DEBUG] Start extract_landmark_vector")

    img = cv2.imread(image_path)
    print("[DEBUG] Image loaded:", img is not None)

    if img is None:
        print("[ERROR] Image not found")
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("[DEBUG] Creating FaceMesh")

    # 🔴 BUAT FaceMesh DI DALAM FUNCTION
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False
    ) as face_mesh:

        print("[DEBUG] Processing image with MediaPipe")
        result = face_mesh.process(rgb)

    print("[DEBUG] MediaPipe finished")

    if not result.multi_face_landmarks:
        print("[DEBUG] No face detected")
        return None

    landmarks = result.multi_face_landmarks[0]
    vector = []

    for lm in landmarks.landmark:
        vector.extend([lm.x, lm.y])

    print("[DEBUG] Landmark count:", len(vector))

    return np.array(vector)