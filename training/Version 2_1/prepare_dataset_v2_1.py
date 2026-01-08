#!/usr/bin/env python3
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

# ================= CONFIG =================
DATASET_ROOT = r"C:/Users/Ananta/Documents/1. Collage/PKL/Student Employee/DATASET_SEKUNDER/Dataset/Dataset_Sekunder/Dataset_Daisee_Kurasi"

SPLITS = {
    "train": "Sampled_Dataset_Train",
    "val": "Sampled_Dataset_Validation",
    "test": "Sampled_Dataset_Test"
}

OUT_DIR = "processed_data_v2_1"
MODEL_DIR = os.path.join(OUT_DIR, "preprocess")
os.makedirs(MODEL_DIR, exist_ok=True)

PCA_COMPONENTS = 120
RANDOM_STATE = 42
MAX_FACES = 1

mp_face_mesh = mp.solutions.face_mesh

# ================= UTILS =================
def parse_label(name):
    return int(name.split("_")[-1]) if "engagement" in name else int(name)

def extract_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x, p.y] for p in lm])

def normalize(lm):
    lm = lm - np.mean(lm, axis=0)
    s = np.linalg.norm(lm, axis=1).max()
    return lm / s if s > 0 else lm

def geometric(lm):
    idx = [33,133,1,61,291,199]
    pts = lm[idx]
    feats = []
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            feats.append(np.linalg.norm(pts[i]-pts[j]))
    return np.array(feats)

def extract_split(folder):
    X, y = [], []
    for cls in sorted(os.listdir(folder)):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            continue
        label = parse_label(cls)
        for f in tqdm(os.listdir(path), desc=cls):
            if not f.endswith(".jpg"):
                continue
            lm = extract_landmarks(os.path.join(path, f))
            if lm is None:
                continue
            lm = normalize(lm)
            X.append(np.hstack([lm.flatten(), geometric(lm)]))
            y.append(label)
    return np.array(X), np.array(y)

# ================= MAIN =================
datasets = {}
for s, f in SPLITS.items():
    X, y = extract_split(os.path.join(DATASET_ROOT, f))
    datasets[s] = {"X": X, "y": y}

# ===== Scaling (FIT ONLY TRAIN) =====
scaler = StandardScaler()
datasets["train"]["X"] = scaler.fit_transform(datasets["train"]["X"])
datasets["val"]["X"]   = scaler.transform(datasets["val"]["X"])
datasets["test"]["X"]  = scaler.transform(datasets["test"]["X"])

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ===== PCA (FIT ONLY TRAIN) =====
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
datasets["train"]["X"] = pca.fit_transform(datasets["train"]["X"])
datasets["val"]["X"]   = pca.transform(datasets["val"]["X"])
datasets["test"]["X"]  = pca.transform(datasets["test"]["X"])

joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))

# ===== Save =====
for s in datasets:
    np.save(os.path.join(OUT_DIR, f"X_{s}_v2_1.npy"), datasets[s]["X"])
    np.save(os.path.join(OUT_DIR, f"y_{s}_v2_1.npy"), datasets[s]["y"])

print("[DONE] Dataset v2.1 READY")