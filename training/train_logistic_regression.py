import numpy as np
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import load_config, get_image_paths
from extract_landmarks import extract_landmark_vector

cfg = load_config()

train_path = f"{cfg['dataset_root']}/{cfg['train_dir']}"

image_paths, labels = get_image_paths(train_path)

X, y = [], []

print("[INFO] Extracting landmarks...")
for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    vec = extract_landmark_vector(img_path)
    if vec is not None:
        X.append(vec)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"[INFO] Dataset shape: {X.shape}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1
    ))
])

print("[INFO] Training Logistic Regression...")
pipeline.fit(X, y)

joblib.dump(pipeline, cfg["model_output"])
print(f"[SUCCESS] Model saved to {cfg['model_output']}")