# ============================================================
# DATASET ENGINEERING v2.1
# Scaling + PCA (120)
# ============================================================

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = r"C:\Users\Ananta\Documents\GitHub\lightweight-engagement-detection\processed_data_v2"
OUTPUT_DIR = os.path.join(DATA_DIR, "v2_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load raw feature datasets (sebelum scaling & PCA)
X_train = np.load(os.path.join(DATA_DIR, "X_train_v2.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val_v2.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test_v2.npy"))

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# ============================================================
# PCA
# ============================================================

pca = PCA(n_components=120, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

joblib.dump(pca, os.path.join(OUTPUT_DIR, "pca.pkl"))

# ============================================================
# SAVE DATASET v2.1
# ============================================================

np.save(os.path.join(OUTPUT_DIR, "X_train_v2_1.npy"), X_train_pca)
np.save(os.path.join(OUTPUT_DIR, "X_val_v2_1.npy"),   X_val_pca)
np.save(os.path.join(OUTPUT_DIR, "X_test_v2_1.npy"),  X_test_pca)

print("[DONE] Dataset Engineering v2.1 completed")