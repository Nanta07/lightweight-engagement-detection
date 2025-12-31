# lightweight-engagement-detection
Exploration of lightweight machine learning models for facial engagement detection as an alternative to Random Forest, using image-based datasets.


# Lightweight Engagement Detection using Logistic Regression

This project explores the use of a lightweight machine learning model for facial engagement detection as an alternative to Random Forest. The study is conducted as part of an experimental research project focusing on efficiency and deployability on low-resource devices.

## Background

Engagement detection is commonly used to analyze user attention and interaction levels, particularly in scenarios such as online learning, customer service, and human-computer interaction. Previous works often employ ensemble-based models such as Random Forest due to their robustness and high accuracy.

However, Random Forest models tend to be computationally heavy and memory-intensive, which limits their deployment on edge devices such as Raspberry Pi. Therefore, this project investigates Logistic Regression as a lightweight alternative while maintaining acceptable performance.

## Engagement Levels

The engagement levels are defined into four classes:

| Label | Description |
|------|------------|
| 0 | Not Engaged |
| 1 | Low Engagement |
| 2 | Medium Engagement |
| 3 | High Engagement |

## Dataset

The dataset used in this project is a curated subset derived from the DAiSEE dataset.

Dataset structure:
```

Sampled_Dataset_Train/
Sampled_Dataset_Validation/
Sampled_Dataset_Test/

```

Each dataset contains four folders corresponding to engagement levels (0–3).

Dataset size:
- Training set: 16,209 image frames
- Validation set: 4,378 image frames
- Test set: 5,184 image frames

Due to storage and licensing considerations, the full dataset is not included in this repository. Only a small sample is provided for demonstration purposes.

## Feature Extraction

Facial features are extracted using MediaPipe Face Mesh, which produces 468 facial landmarks per frame. Each landmark consists of (x, y) coordinates, resulting in a 936-dimensional feature vector per image.

These feature vectors are used as input for the classification model.

## Model Selection

### Why Not Random Forest?

Random Forest was not selected for this project due to:
- High memory usage when deployed
- Slower inference time on low-power devices
- Less suitable for real-time edge deployment

### Why Logistic Regression?

Logistic Regression was chosen because:
- Lightweight and fast inference
- Low memory footprint
- Suitable for real-time applications
- Easier deployment on embedded systems
- Provides probabilistic output for confidence analysis

Although Logistic Regression is simpler than Random Forest, this study aims to evaluate whether acceptable performance can still be achieved under resource constraints.

## Project Scope

This repository focuses on:
- Dataset exploration and preprocessing
- Feature extraction using facial landmarks
- Training and evaluation of Logistic Regression models

Integration with Raspberry Pi and server-side systems will be addressed in future stages of the research.

## Repository Structure

- `preprocessing/` : Feature extraction scripts
- `training/` : Model training and evaluation
- `data/` : Sample dataset structure
- `docs/` : Methodology and dataset description
- `experiments/` : Experiment notes and observations

## Future Work

- Model optimization and hyperparameter tuning
- Comparison with other lightweight models (e.g., Linear SVM)
- Deployment and benchmarking on Raspberry Pi
- Real-time engagement visualization


---
# Dataset Information

This project uses a curated version of the DAiSEE dataset for engagement detection.

The full dataset is not included in this repository. Please refer to the original DAiSEE dataset for complete access.

Only a small subset of sample images is provided for code demonstration and testing purposes.
```

https://drive.google.com/drive/folders/1ydBo1g9RmWN0Ka55rGMLZ15aj7PUpTVJ
