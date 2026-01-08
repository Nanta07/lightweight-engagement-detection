#train_mlp_raspi_pkl.py

# FINAL BEST MLP TRAINING (Raspberry Pi Compatible)
import os
import numpy as np
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Load dataset (.npy – SESUAI PROJECT)
DATA_DIR = "processed_data"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("[OK] Dataset loaded")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Train distribution:", Counter(y_train))

# Scaling (WAJIB untuk MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Sample weight (AMAN & RASPI-SAFE)
sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# BEST MLP CONFIG (HASIL TERBAIK SEPANJANG EKSPERIMEN)
mlp = MLPClassifier(
    hidden_layer_sizes=(64,),     # 1 layer = paling stabil
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

print("\n[TRAINING] MLP (BEST CONFIG)...")
mlp.fit(X_train, y_train, sample_weight=sample_weight)

# VALIDATION
y_val_pred = mlp.predict(X_val)

print("\n=== VALIDATION RESULTS ===")
print(classification_report(y_val, y_val_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# TEST
y_test_pred = mlp.predict(X_test)

print("\n=== TEST RESULTS ===")
print(classification_report(y_test, y_test_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# SAVE MODEL
MODEL_PATH  = os.path.join(DATA_DIR, "v2_mlp_engagement.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "v2_scaler_mlp_engagement.pkl")

joblib.dump(mlp, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n[OK] BEST MLP model saved:")
print("Model :", MODEL_PATH)
print("Scaler:", SCALER_PATH)