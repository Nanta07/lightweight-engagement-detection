# ============================================================
# FINAL MODEL TRAINING PIPELINE v2 (RASPBERRY PI READY)
# Multiclass Engagement Detection (0–3)
# ============================================================

import os
import numpy as np
import joblib
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\Ananta\Documents\GitHub\lightweight-engagement-detection\processed_data_v2"
OUTPUT_DIR = os.path.join(DATA_DIR, "models_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# STAGE 1: LOAD DATASET v2
# ============================================================

print("\n========== [STAGE 1] Load Dataset v2 ==========")

X_train = np.load(os.path.join(DATA_DIR, "X_train_v2.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train_v2.npy"))

X_val = np.load(os.path.join(DATA_DIR, "X_val_v2.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val_v2.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test_v2.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test_v2.npy"))

print("[INFO] Dataset shapes:")
print(" Train:", X_train.shape)
print(" Val  :", X_val.shape)
print(" Test :", X_test.shape)

print("[INFO] Train distribution:", Counter(y_train))
print("[INFO] Val distribution  :", Counter(y_val))
print("[INFO] Test distribution :", Counter(y_test))

print("[DONE] Dataset v2 loaded successfully")

# ============================================================
# STAGE 2: PREPARE SAMPLE WEIGHT (MULTICLASS SAFE)
# ============================================================

print("\n========== [STAGE 2] Compute Sample Weight ==========")

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("[DONE] Sample weight computed")

# ============================================================
# STAGE 3: TRAIN LOGISTIC REGRESSION v2
# ============================================================

print("\n========== [STAGE 3] Train Logistic Regression v2 ==========")

logreg = LogisticRegression(
    max_iter=400,
    solver="saga",
    multi_class="multinomial",
    class_weight="balanced",
    C=0.3,                    # stronger regularization (lebih stabil)
    tol=1e-3,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

logreg.fit(X_train, y_train, sample_weight=sample_weight)

print("[DONE] Logistic Regression training finished")

# ============================================================
# STAGE 4: EVALUATION - LOGISTIC REGRESSION
# ============================================================

print("\n========== [STAGE 4] Evaluation - Logistic Regression ==========")

def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n--- {name} ---")
    print(classification_report(y, y_pred, digits=4))
    print("Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

evaluate_model(logreg, X_val, y_val, "VALIDATION")
evaluate_model(logreg, X_test, y_test, "TEST")

# ============================================================
# STAGE 5: SAVE LOGREG MODEL
# ============================================================

logreg_path = os.path.join(OUTPUT_DIR, "logreg_engagement_v2.pkl")
joblib.dump(logreg, logreg_path)

print(f"[SAVED] Logistic Regression model → {logreg_path}")

# ============================================================
# STAGE 6: TRAIN MLP v2 (FINAL CONFIG)
# ============================================================

print("\n========== [STAGE 6] Train MLP v2 ==========")

mlp = MLPClassifier(
    hidden_layer_sizes=(64,),      # paling stabil untuk RasPi
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=RANDOM_STATE,
    verbose=True
)

mlp.fit(X_train, y_train, sample_weight=sample_weight)

print("[DONE] MLP training finished")

# ============================================================
# STAGE 7: EVALUATION - MLP
# ============================================================

print("\n========== [STAGE 7] Evaluation - MLP ==========")

evaluate_model(mlp, X_val, y_val, "VALIDATION")
evaluate_model(mlp, X_test, y_test, "TEST")

# ============================================================
# STAGE 8: SAVE MLP MODEL
# ============================================================

mlp_path = os.path.join(OUTPUT_DIR, "mlp_engagement_v2.pkl")
joblib.dump(mlp, mlp_path)

print(f"[SAVED] MLP model → {mlp_path}")

# ============================================================
# PIPELINE FINISHED
# ============================================================

print("\n============================================")
print(" MODEL TRAINING PIPELINE v2 FINISHED SUCCESSFULLY")
print("============================================")