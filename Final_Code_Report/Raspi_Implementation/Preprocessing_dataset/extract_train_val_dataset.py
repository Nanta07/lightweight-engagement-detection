# training/extract_dataset_train_val.py
# Name: extract_train_val_dataset.py
import os
import numpy as np
from tqdm import tqdm
from extract_landmarks import extract_landmark_vector

DATASET_ROOT = r"C:/Users/Ananta/Documents/1. Collage/PKL/Student Employee/DATASET_SEKUNDER/Dataset/Dataset_Sekunder/Dataset_Daisee_Kurasi"

SPLITS = {
    "train": "Sampled_Dataset_Train",
    "val": "Sampled_Dataset_Validation"
}

OUT_DIR = os.path.join(os.getcwd(), "processed_data")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_label(class_name):
    if class_name.isdigit():
        return int(class_name)
    if "engagement" in class_name:
        return int(class_name.split("_")[-1])
    raise ValueError(f"Cannot parse label from {class_name}")


def extract_split(split_name, split_folder):
    print(f"\n=== Extracting {split_name.upper()} dataset ===")

    split_dir = os.path.join(DATASET_ROOT, split_folder)

    X_list, y_list, failed = [], [], []

    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        try:
            label = parse_label(class_name)
        except Exception as e:
            print(f"[WARN] Skip folder {class_name}: {e}")
            continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith(".jpg")]

        for f in tqdm(files, desc=f"{split_name} | {class_name}", unit="img"):
            img_path = os.path.join(class_path, f)
            vec = extract_landmark_vector(img_path)
            if vec is None:
                failed.append(img_path)
                continue

            X_list.append(vec)
            y_list.append(label)

    X = np.stack(X_list) if X_list else np.zeros((0, 936))
    y = np.array(y_list)

    np.save(os.path.join(OUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y_{split_name}.npy"), y)

    with open(os.path.join(OUT_DIR, f"failed_{split_name}.txt"), "w", encoding="utf-8") as fw:
        for p in failed:
            fw.write(p + "\n")

    print(f"[DONE] {split_name}: X shape = {X.shape}, failed = {len(failed)}")


if __name__ == "__main__":
    for split, folder in SPLITS.items():
        extract_split(split, folder)