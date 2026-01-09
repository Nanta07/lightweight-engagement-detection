#train_logreg_raspi_pkl.py
import os
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample
import joblib

# -----------------------------
# Load processed data
# -----------------------------
DATA_DIR = os.path.join(os.getcwd(), "processed_data")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("TRAIN distribution:", Counter(y_train))
print("VAL distribution:", Counter(y_val))
print("TEST distribution:", Counter(y_test))

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Dimensionality reduction (optional)
# -----------------------------
pca_dim = 64  # tweakable, RasPi-friendly
pca = PCA(n_components=pca_dim, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"[INFO] PCA output dim: {X_train_pca.shape[1]}")

# -----------------------------
# Optional: Oversampling minor classes
# -----------------------------
# Uncomment if minor classes are extremely underrepresented
# from sklearn.utils import resample
# X_train_combined = np.hstack((X_train_pca, y_train.reshape(-1,1)))
# df = pd.DataFrame(X_train_combined)
# df_majority = df[df.iloc[:, -1] != 0]
# df_minority = df[df.iloc[:, -1] == 0]
# df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
# df_balanced = pd.concat([df_majority, df_minority_upsampled])
# X_train_pca = df_balanced.iloc[:, :-1].values
# y_train = df_balanced.iloc[:, -1].astype(int).values

# -----------------------------
# Logistic Regression (Improved)
# -----------------------------
clf = LogisticRegression(
    max_iter=300,
    solver='saga',           # RasPi-friendly, supports multinomial
    multi_class='multinomial',
    class_weight='balanced',  # important for imbalance
    C=0.5,                    # tweakable regularization
    n_jobs=-1,
    tol=1e-3,
    random_state=42
)

print("\n[TRAINING] Logistic Regression...")
clf.fit(X_train_pca, y_train)

# -----------------------------
# Evaluate
# -----------------------------
def evaluate(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    print(f"\n=== {dataset_name} RESULTS ===")
    print(classification_report(y, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

evaluate(clf, X_val_pca, y_val, "VALIDATION")
evaluate(clf, X_test_pca, y_test, "TEST")

# -----------------------------
# Save model, scaler, PCA
# -----------------------------
os.makedirs(DATA_DIR, exist_ok=True)
joblib.dump(clf, os.path.join(DATA_DIR, "v3_logreg_engagement.pkl"))
joblib.dump(scaler, os.path.join(DATA_DIR, "v3_scaler_engagement.pkl"))
joblib.dump(pca, os.path.join(DATA_DIR, "v3_pca_engagement.pkl"))
print("\n[OK] Logistic Regression FINAL IMPROVED model saved (RasPi-safe)")