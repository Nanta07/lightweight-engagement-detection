import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils import load_config, get_image_paths
from extract_landmarks import extract_landmark_vector

cfg = load_config()

model = joblib.load(cfg["model_output"])

test_path = f"{cfg['dataset_root']}/{cfg['test_dir']}"
image_paths, labels = get_image_paths(test_path)

X_test, y_test = [], []

for img_path, label in zip(image_paths, labels):
    vec = extract_landmark_vector(img_path)
    if vec is not None:
        X_test.append(vec)
        y_test.append(label)

X_test = np.array(X_test)

pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))