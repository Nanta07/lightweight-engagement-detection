import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report

DATA_DIR = "processed_data_v2_1"
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, "X_train_v2_1.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train_v2_1.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val_v2_1.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val_v2_1.npy"))

sample_weight = compute_sample_weight("balanced", y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(64,),
    max_iter=400,
    early_stopping=True,
    random_state=42,
    verbose=True
)

mlp.fit(X_train, y_train, sample_weight=sample_weight)

print(classification_report(y_val, mlp.predict(X_val)))

joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_engagement_v2_1.pkl"))
print("[DONE] MLP v2.1 trained & saved")