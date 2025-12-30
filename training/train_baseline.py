import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X_train = np.load("processed_data/X_train.npy")
y_train = np.load("processed_data/y_train.npy")

X_val = np.load("processed_data/X_val.npy")
y_val = np.load("processed_data/y_val.npy")

model = LogisticRegression(
    max_iter=3000,
    n_jobs=-1,
    class_weight="balanced",
    solver="lbfgs",
    multi_class="auto"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("\n=== VALIDATION RESULTS ===")
print(classification_report(y_val, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))