import numpy as np
from collections import Counter

for split in ["train", "val", "test"]:
    try:
        y = np.load(f"processed_data/y_{split}.npy")
        print(f"\n{split.upper()} distribution:")
        print(Counter(y))
    except:
        pass