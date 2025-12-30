import yaml
import os

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_image_paths(base_dir):
    image_paths = []
    labels = []

    for label in sorted(os.listdir(base_dir)):
        class_dir = os.path.join(base_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith(".jpg"):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(int(label))

    return image_paths, labels