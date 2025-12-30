# training/extract_dataset.py
import os
import numpy as np
from tqdm import tqdm
from extract_landmarks import extract_landmark_vector

# CONFIG: sesuaikan jika tidak pakai config.yaml
DATASET_ROOT = r"C:/Users/Ananta/Documents/1. Collage/PKL/Student Employee/DATASET_SEKUNDER/Dataset/Dataset_Sekunder/Dataset_Daisee_Kurasi"
SUBFOLDERS = {
    "Sampled_Dataset_Train": "train",
    "Sampled_Dataset_Validation": "val",
    "Sampled_Dataset_Test": "test"
}
# Pilih folder mana yang akan diekstrak. Untuk permulaan, extract Test dulu (lebih cepat).
TARGET_DIR = os.path.join(DATASET_ROOT, "Sampled_Dataset_Test")

OUT_DIR = os.path.join(os.getcwd(), "processed_data")
os.makedirs(OUT_DIR, exist_ok=True)

X_list = []
y_list = []
failed = []

# asumsi struktur: TARGET_DIR/<class_folder>/<imagename.jpg>
for class_name in sorted(os.listdir(TARGET_DIR)):
    class_path = os.path.join(TARGET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # Try to parse label if folder named '0','1','2','3' or 'engagement_0' etc.
    try:
        if class_name.isdigit():
            label = int(class_name)
        elif "engagement" in class_name:
            # take last integer part
            label = int(class_name.split("_")[-1])
        else:
            # fallback: index order
            label = None
    except Exception:
        label = None

    files = [f for f in os.listdir(class_path) if f.lower().endswith(".jpg")]
    for f in tqdm(files, desc=f"Class {class_name}", unit="img"):
        img_path = os.path.join(class_path, f)
        vec = extract_landmark_vector(img_path)
        if vec is None:
            failed.append(img_path)
            continue
        X_list.append(vec)
        y_list.append(label if label is not None else class_name)

X = np.stack(X_list) if X_list else np.zeros((0, 936))
y = np.array(y_list)

np.save(os.path.join(OUT_DIR, "X_test.npy"), X)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y)

with open(os.path.join(OUT_DIR, "failed_images.txt"), "w", encoding="utf-8") as fw:
    for p in failed:
        fw.write(p + "\n")

print("Done. Saved:", os.path.join(OUT_DIR, "X_test.npy"), X.shape)
print("Failed images:", len(failed))