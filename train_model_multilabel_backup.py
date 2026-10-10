
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from joblib import dump
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ---------------- CONFIGURATION ----------------
CSV_PATH = Path("labels.csv")
IMAGE_DIR = Path("images")
MODEL_DIR = Path("saved_model_cnn_rf")

IMAGE_SIZE = 224
BATCH_SIZE = 16
N_ESTIMATORS = 100
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Set to 1000 for a quick pipeline test, or None for all images.
MAX_IMAGES = None

# Optional: use a smaller number if your PC has limited RAM.
# ------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", DEVICE)

# 1. Load metadata and locate images
df = pd.read_csv(CSV_PATH)

required = ["Image Index", "Finding Labels", "Patient ID"]
for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing CSV column: {column}")

df["Image Index"] = df["Image Index"].astype(str).str.strip()
df["Finding Labels"] = df["Finding Labels"].fillna("")

image_map = {
    p.name: p
    for p in IMAGE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
}

df["image_path"] = df["Image Index"].map(image_map)
print("Missing images:", int(df["image_path"].isna().sum()))
df = df.dropna(subset=["image_path"]).copy()
df = df.drop_duplicates(subset=["Image Index"])

if MAX_IMAGES is not None:
    df = df.sample(
        n=min(MAX_IMAGES, len(df)),
        random_state=RANDOM_STATE
    )

df = df.reset_index(drop=True)

if len(df) < 10:
    raise ValueError("Not enough matching images.")

# 2. Create multilabel targets
label_lists = df["Finding Labels"].apply(
    lambda value: [
        label.strip()
        for label in str(value).split("|")
        if label.strip()
    ]
).tolist()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_lists).astype(np.uint8)

print("Images:", len(df))
print("Labels:", list(mlb.classes_))

# 3. Split by patient BEFORE model evaluation
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        np.zeros(len(df)),
        y,
        groups=df["Patient ID"]
    )
)

# Keep test patients out of all supervised model fitting.
print("Training images:", len(train_idx))
print("Testing images:", len(test_idx))

# 4. Create pretrained DenseNet121 feature extractor
weights = DenseNet121_Weights.DEFAULT
cnn = densenet121(weights=weights)

# Save the feature extractor weights so Streamlit can use
# exactly the same CNN without downloading them again.
torch.save(
    cnn.features.state_dict(),
    MODEL_DIR / "densenet_features.pth"
)

cnn.classifier = nn.Identity()
cnn = cnn.to(DEVICE)
cnn.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class XrayDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = preprocess(image)
            return tensor, index
        except Exception as error:
            raise RuntimeError(
                f"Could not read image {path}: {error}"
            ) from error


# 5. Extract CNN features in batches
@torch.inference_mode()
def extract_features(paths):
    dataset = XrayDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []

    for batch_number, (images, _) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        features = cnn.features(images)
        features = torch.relu(features)
        features = torch.nn.functional.adaptive_avg_pool2d(
            features, (1, 1)
        )
        features = torch.flatten(features, 1)

        all_features.append(
            features.cpu().numpy().astype(np.float32)
        )

        if batch_number % 50 == 0:
            print("Feature batches processed:", batch_number)

    return np.concatenate(all_features, axis=0)


print("Extracting CNN features...")
X = extract_features(df["image_path"].tolist())
print("CNN feature matrix:", X.shape)

# Release CNN memory before Random Forest training
del cnn
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 6. Train one balanced binary Random Forest per finding
base_rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

model = MultiOutputClassifier(
    base_rf,
    n_jobs=1
)

print("Training CNN-feature Random Forest classifiers...")
model.fit(X_train, y_train)

# 7. Obtain probabilities and evaluate at threshold 0.5
probability_columns = []

for estimator in model.estimators_:
    proba = estimator.predict_proba(X_test)

    positive_class = np.where(estimator.classes_ == 1)[0]

    if len(positive_class):
        probability_columns.append(proba[:, positive_class[0]])
    else:
        probability_columns.append(
            np.zeros(len(X_test), dtype=float)
        )

y_scores = np.column_stack(probability_columns)
y_pred = (y_scores >= 0.5).astype(np.uint8)

print("\n===== TEST CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    target_names=mlb.classes_,
    zero_division=0
))

macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
micro_f1 = f1_score(
    y_test, y_pred, average="micro", zero_division=0
)

print(f"Macro F1 at threshold 0.5: {macro_f1:.4f}")
print(f"Micro F1 at threshold 0.5: {micro_f1:.4f}")

# Per-label ROC-AUC and Average Precision.
# These are undefined if the test set contains only one class.
auc_values = []
ap_values = []

print("\n===== PER-LABEL RANKING METRICS =====")
for i, label in enumerate(mlb.classes_):
    if len(np.unique(y_test[:, i])) < 2:
        auc = np.nan
        print(f"{label}: ROC-AUC undefined (one test class)")
    else:
        auc = roc_auc_score(y_test[:, i], y_scores[:, i])
        ap = average_precision_score(y_test[:, i], y_scores[:, i])
        auc_values.append(auc)
        ap_values.append(ap)
        print(f"{label}: ROC-AUC={auc:.4f}, AP={ap:.4f}")

print(
    "Macro ROC-AUC:",
    round(float(np.mean(auc_values)), 4)
    if auc_values else "undefined"
)
print(
    "Macro Average Precision:",
    round(float(np.mean(ap_values)), 4)
    if ap_values else "undefined"
)

# Save model, labels and preprocessing settings
dump(model, MODEL_DIR / "random_forest.joblib")
dump(mlb, MODEL_DIR / "label_binarizer.joblib")

with open(MODEL_DIR / "config.json", "w") as file:
    json.dump({
        "image_size": IMAGE_SIZE,
        "labels": mlb.classes_.tolist(),
        "threshold": 0.5,
        "feature_dim": int(X.shape[1])
    }, file, indent=2)

# Save evaluation outputs for comparison and reporting
metrics_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0
    )
).T

metrics_df.to_csv(MODEL_DIR / "classification_report.csv")

pd.DataFrame({
    "label": mlb.classes_,
    "roc_auc": [
        roc_auc_score(y_test[:, i], y_scores[:, i])
        if len(np.unique(y_test[:, i])) == 2 else np.nan
        for i in range(len(mlb.classes_))
    ],
    "average_precision": [
        average_precision_score(y_test[:, i], y_scores[:, i])
        for i in range(len(mlb.classes_))
    ]
}).to_csv(MODEL_DIR / "ranking_metrics.csv", index=False)

np.save(MODEL_DIR / "test_labels.npy", y_test)
np.save(MODEL_DIR / "test_scores.npy", y_scores)

print("\nTraining and evaluation completed.")
print("Saved all artifacts in:", MODEL_DIR.resolve())
