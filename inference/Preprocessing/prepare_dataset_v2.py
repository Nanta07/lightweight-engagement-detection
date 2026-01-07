#!/usr/bin/env python3
"""
prepare_dataset_v2.py
Pipeline Dataset Engineering v2 untuk Engagement Detection (DAiSEE)

Author  : Ananta Boemi Adji
Purpose : Improve dataset quality before retraining MLP & Logistic Regression
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

# =========================================================
# CONFIGURATION
# =========================================================
DATASET_ROOT = r"C:/Users/Ananta/Documents/1. Collage/PKL/Student Employee/DATASET_SEKUNDER/Dataset/Dataset_Sekunder/Dataset_Daisee_Kurasi"

SPLITS = {
    "train": "Sampled_Dataset_Train",
    "val": "Sampled_Dataset_Validation",
    "test": "Sampled_Dataset_Test"
}

OUT_DIR = "processed_data_v2"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FACES = 1
LANDMARK_DIM = 468
RANDOM_STATE = 42
PCA_COMPONENTS = 120   # kompromi antara performa & efisiensi

# =========================================================
# INITIALIZE MEDIAPIPE
# =========================================================
mp_face_mesh = mp.solutions.face_mesh

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def parse_label(folder_name):
    if folder_name.isdigit():
        return int(folder_name)
    if "engagement" in folder_name:
        return int(folder_name.split("_")[-1])
    raise ValueError(f"Cannot parse label: {folder_name}")


def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=MAX_FACES,
        refine_landmarks=False
    ) as face_mesh:
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    coords = np.array([[p.x, p.y] for p in lm])
    return coords


def normalize_landmarks(landmarks):
    """
    Translation & scale normalization
    """
    center = np.mean(landmarks, axis=0)
    landmarks = landmarks - center

    scale = np.linalg.norm(landmarks, axis=1).max()
    if scale > 0:
        landmarks = landmarks / scale

    return landmarks


def geometric_features(landmarks):
    """
    Feature engineering:
    - Pairwise distances (important points only)
    - Ratios to make scale-invariant
    """
    # Indices landmark penting (mata, hidung, mulut)
    key_idx = [33, 133, 1, 61, 291, 199]
    key_pts = landmarks[key_idx]

    features = []

    for i in range(len(key_pts)):
        for j in range(i + 1, len(key_pts)):
            dist = np.linalg.norm(key_pts[i] - key_pts[j])
            features.append(dist)

    return np.array(features)


# =========================================================
# DATA EXTRACTION PIPELINE
# =========================================================
def extract_dataset(split_name, split_folder):
    print(f"\n========== [STAGE 1] Extracting {split_name.upper()} ==========")

    X_raw, X_geo, y = [], [], []
    failed = 0

    split_path = os.path.join(DATASET_ROOT, split_folder)

    for class_folder in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        label = parse_label(class_folder)
        files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]

        for f in tqdm(files, desc=f"{split_name} | {class_folder}", unit="img"):
            img_path = os.path.join(class_path, f)
            lm = extract_landmarks(img_path)
            if lm is None:
                failed += 1
                continue

            lm_norm = normalize_landmarks(lm)
            geo_feat = geometric_features(lm_norm)

            X_raw.append(lm_norm.flatten())
            X_geo.append(geo_feat)
            y.append(label)

    print(f"[INFO] {split_name} extracted")
    print(f"       Total samples : {len(y)}")
    print(f"       Failed images : {failed}")

    return np.array(X_raw), np.array(X_geo), np.array(y)


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    print("\n============================================")
    print(" DATASET ENGINEERING PIPELINE v2 STARTED ")
    print("============================================")

    datasets = {}

    # Stage 1: Extraction & Feature Engineering
    for split, folder in SPLITS.items():
        X_raw, X_geo, y = extract_dataset(split, folder)
        X = np.hstack([X_raw, X_geo])
        datasets[split] = {"X": X, "y": y}

    print("\n========== [STAGE 2] Feature Scaling ==========")
    scaler = StandardScaler()
    datasets["train"]["X"] = scaler.fit_transform(datasets["train"]["X"])
    datasets["val"]["X"] = scaler.transform(datasets["val"]["X"])
    datasets["test"]["X"] = scaler.transform(datasets["test"]["X"])
    print("[DONE] Feature scaling completed")

    print("\n========== [STAGE 3] Dimensionality Reduction (PCA) ==========")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    datasets["train"]["X"] = pca.fit_transform(datasets["train"]["X"])
    datasets["val"]["X"] = pca.transform(datasets["val"]["X"])
    datasets["test"]["X"] = pca.transform(datasets["test"]["X"])
    print(f"[DONE] PCA applied → {PCA_COMPONENTS} dimensions")

    print("\n========== [STAGE 4] Class Balancing (TRAIN ONLY) ==========")
    X_bal, y_bal = [], []

    classes = np.unique(datasets["train"]["y"])
    max_count = int(np.mean([np.sum(datasets["train"]["y"] == c) for c in classes]))

    for c in classes:
        X_c = datasets["train"]["X"][datasets["train"]["y"] == c]
        y_c = datasets["train"]["y"][datasets["train"]["y"] == c]

        X_res, y_res = resample(
            X_c, y_c,
            replace=True,
            n_samples=max_count,
            random_state=RANDOM_STATE
        )

        X_bal.append(X_res)
        y_bal.append(y_res)

    datasets["train"]["X"] = np.vstack(X_bal)
    datasets["train"]["y"] = np.hstack(y_bal)

    print("[DONE] Train set balanced")
    print("       New train size:", datasets["train"]["X"].shape[0])

    print("\n========== [STAGE 5] Saving Dataset v2 ==========")
    for split in datasets:
        np.save(os.path.join(OUT_DIR, f"X_{split}_v2.npy"), datasets[split]["X"])
        np.save(os.path.join(OUT_DIR, f"y_{split}_v2.npy"), datasets[split]["y"])
        print(f"[SAVED] {split} dataset")

    print("\n============================================")
    print(" DATASET ENGINEERING PIPELINE v2 FINISHED ")
    print("============================================")


if __name__ == "__main__":
    main()