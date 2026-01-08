# ============================================================
# FINAL MLP TRAINING PIPELINE v2.1 (RASPBERRY PI READY)
# ============================================================

import os
import numpy as np
import joblib
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR = r"C:\Users\Ananta\Documents\GitHub\lightweight-engagement-detection\processed_data_v2\v2_1"
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# LOAD DATASET v2.1
# ============================================================

X_train = np.load(os.path.join(DATA_DIR, "X_train_v2_1.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val_v2_1.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test_v2_1.npy"))

y_train = np.load(os.path.join(DATA_DIR, "..", "y_train_v2.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "..", "y_val_v2.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "..", "y_test_v2.npy"))

print("Train shape:", X_train.shape)
print("Class distribution:", Counter(y_train))

# ============================================================
# SAMPLE WEIGHT
# ============================================================

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# ============================================================
# TRAIN MLP
# ============================================================

mlp = MLPClassifier(
    hidden_layer_sizes=(64,),
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

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n{name}")
    print(classification_report(y, y_pred, digits=4))
    print("Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

evaluate(mlp, X_val, y_val, "VALIDATION")
evaluate(mlp, X_test, y_test, "TEST")

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_engagement_v2_1.pkl"))

print("[SAVED] MLP v2.1 model saved successfully")