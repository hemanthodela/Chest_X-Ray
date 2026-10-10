"""
Binary Chest X-Ray Pneumonia Detection: DenseNet121 + Random Forest
===================================================================
Pipeline:
    Chest X-ray image -> Preprocessing (224x224, ImageNet Norm)
    -> Pretrained DenseNet121 Feature Extractor (1024 features)
    -> Balanced RandomForestClassifier -> Binary Prediction (Normal vs Pneumonia)

Outputs:
    - saved_model_binary/random_forest.joblib
    - saved_model_binary/model_config.json
    - saved_model_binary/densenet_features.pth
    - evaluation_results/metrics.json
    - evaluation_results/classification_report.csv
    - evaluation_results/confusion_matrix.png
    - evaluation_results/roc_curve.png
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for generating images
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
N_ESTIMATORS = 150
RANDOM_STATE = 42

MODEL_DIR = Path("saved_model_binary")
RESULTS_DIR = Path("evaluation_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using compute device: {DEVICE}")


# -------------------------------------------------------------
# 1. Dataset Directory Detection and Validation
# -------------------------------------------------------------
def locate_dataset_root():
    """Locate the dataset root folder containing train, val, and test splits."""
    candidate_roots = [
        Path("chest_xray"),
        Path("chect_xray"),
        Path("chest_xray/chest_xray"),
        Path("."),
    ]
    for root in candidate_roots:
        if (root / "train").exists() and (root / "test").exists():
            return root
    raise FileNotFoundError(
        "Could not find dataset with train/ and test/ folders. Checked: "
        + ", ".join(str(c) for c in candidate_roots)
    )

DATASET_ROOT = locate_dataset_root()
print(f"[Dataset] Found dataset at: {DATASET_ROOT.resolve()}")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def gather_split_images(split_name: str):
    """
    Collect image paths and labels for a split.
    Label 0: NORMAL (No Pneumonia)
    Label 1: PNEUMONIA (Pneumonia)
    """
    split_dir = DATASET_ROOT / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Case-insensitive lookup for NORMAL and PNEUMONIA subfolders
    normal_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.upper() == "NORMAL"]
    pneumonia_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.upper() == "PNEUMONIA"]

    if not normal_dirs or not pneumonia_dirs:
        raise ValueError(f"Could not find NORMAL and PNEUMONIA folders inside {split_dir}")

    normal_dir = normal_dirs[0]
    pneumonia_dir = pneumonia_dirs[0]

    normal_files = [p for p in normal_dir.glob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    pneumonia_files = [p for p in pneumonia_dir.glob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    paths = normal_files + pneumonia_files
    labels = [0] * len(normal_files) + [1] * len(pneumonia_files)

    return paths, labels, len(normal_files), len(pneumonia_files)


# Count images in each split
train_paths, train_labels, n_train_norm, n_train_pneu = gather_split_images("train")
val_paths, val_labels, n_val_norm, n_val_pneu = gather_split_images("val")
test_paths, test_labels, n_test_norm, n_test_pneu = gather_split_images("test")

print("\n" + "="*60)
print("DATASET SPLIT BREAKDOWN")
print("="*60)
print(f"  Training set   : {len(train_paths):>5} images (Normal: {n_train_norm:>4}, Pneumonia: {n_train_pneu:>4})")
print(f"  Validation set : {len(val_paths):>5} images (Normal: {n_val_norm:>4}, Pneumonia: {n_val_pneu:>4})")
print(f"  Test set       : {len(test_paths):>5} images (Normal: {n_test_norm:>4}, Pneumonia: {n_test_pneu:>4})")
print(f"  Total images   : {len(train_paths) + len(val_paths) + len(test_paths):>5}")
print("="*60 + "\n")


# -------------------------------------------------------------
# 2. Preprocessing and DenseNet121 Feature Extractor
# -------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ChestXrayDataset(Dataset):
    def __init__(self, paths, labels=None):
        self.paths = list(paths)
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor = preprocess(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {e}") from e

        if self.labels is not None:
            return tensor, self.labels[idx]
        return tensor


def init_feature_extractor():
    """Initialize DenseNet121 with pretrained weights."""
    print("[CNN] Loading pretrained DenseNet121...")
    # Check if we already have saved weights locally
    saved_weights_candidates = [
        MODEL_DIR / "densenet_features.pth",
        Path("saved_model_cnn_rf") / "densenet_features.pth"
    ]
    
    local_weights = None
    for cand in saved_weights_candidates:
        if cand.exists():
            local_weights = cand
            break

    if local_weights is not None:
        print(f"[CNN] Loading local DenseNet weights from {local_weights}")
        cnn = densenet121(weights=None)
        state_dict = torch.load(local_weights, map_location="cpu")
        cnn.features.load_state_dict(state_dict)
    else:
        print("[CNN] Downloading ImageNet pretrained weights (DenseNet121_Weights.DEFAULT)...")
        weights = DenseNet121_Weights.DEFAULT
        cnn = densenet121(weights=weights)

    # Save feature weights in target model directory for self-contained deployment
    features_weight_path = MODEL_DIR / "densenet_features.pth"
    if not features_weight_path.exists():
        print(f"[CNN] Saving feature weights to {features_weight_path}...")
        torch.save(cnn.features.state_dict(), features_weight_path)

    cnn.classifier = nn.Identity()
    cnn = cnn.to(DEVICE)
    cnn.eval()
    return cnn


cnn = init_feature_extractor()


@torch.inference_mode()
def extract_features(paths, split_title="dataset"):
    """Extract 1024-dim features for all images in paths using DenseNet121."""
    dataset = ChestXrayDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    total_batches = len(loader)
    start_time = time.time()

    print(f"[Feature Extraction] Starting {split_title} ({len(paths)} images, {total_batches} batches)...")
    for b_idx, images in enumerate(loader, start=1):
        images = images.to(DEVICE)
        feats = cnn.features(images)
        feats = torch.relu(feats)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        feats = torch.flatten(feats, 1)

        all_features.append(feats.cpu().numpy().astype(np.float32))

        if b_idx % 25 == 0 or b_idx == total_batches:
            elapsed = time.time() - start_time
            rate = (b_idx * BATCH_SIZE) / max(elapsed, 0.001)
            print(f"  [{split_title}] Batch {b_idx}/{total_batches} completed ({rate:.1f} imgs/s)")

    features_matrix = np.concatenate(all_features, axis=0)
    print(f"[Feature Extraction] Finished {split_title}: shape {features_matrix.shape}")
    return features_matrix


# -------------------------------------------------------------
# 3. Extract Features for Train, Val, and Test Splits
# -------------------------------------------------------------
X_train = extract_features(train_paths, "Training Set")
y_train = np.array(train_labels, dtype=np.int64)

X_val = extract_features(val_paths, "Validation Set")
y_val = np.array(val_labels, dtype=np.int64)

X_test = extract_features(test_paths, "Test Set")
y_test = np.array(test_labels, dtype=np.int64)

# Free PyTorch memory
del cnn
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# -------------------------------------------------------------
# 4. Train Random Forest Classifier
# -------------------------------------------------------------
print("\n" + "="*60)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*60)
print(f"Hyperparameters: n_estimators={N_ESTIMATORS}, max_features='sqrt', min_samples_leaf=2, class_weight='balanced'")

rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

t0 = time.time()
rf_model.fit(X_train, y_train)
train_duration = time.time() - t0
print(f"Training completed in {train_duration:.2f} seconds.")


# -------------------------------------------------------------
# 5. Validation Check
# -------------------------------------------------------------
val_preds = rf_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"[Validation Check] Validation Accuracy ({len(y_val)} samples): {val_acc * 100:.2f}%")


# -------------------------------------------------------------
# 6. Comprehensive Test Set Evaluation
# -------------------------------------------------------------
print("\n" + "="*60)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("="*60)

test_preds = rf_model.predict(X_test)
test_probas = rf_model.predict_proba(X_test)
test_scores_pneumonia = test_probas[:, 1]  # Score for Pneumonia class

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
rec = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
f1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)

cm = confusion_matrix(y_test, test_preds)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

try:
    auc = roc_auc_score(y_test, test_scores_pneumonia)
except Exception:
    auc = float("nan")

# Classification report dictionary & DataFrame
report_dict = classification_report(
    y_test,
    test_preds,
    target_names=["No Pneumonia (Normal)", "Pneumonia"],
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report_dict).T

# Save classification report CSV
report_csv_path = RESULTS_DIR / "classification_report.csv"
report_df.to_csv(report_csv_path)

# Metrics dictionary
metrics_data = {
    "dataset": {
        "root": str(DATASET_ROOT),
        "train_samples": len(y_train),
        "train_normal": int(n_train_norm),
        "train_pneumonia": int(n_train_pneu),
        "val_samples": len(y_val),
        "val_normal": int(n_val_norm),
        "val_pneumonia": int(n_val_pneu),
        "test_samples": len(y_test),
        "test_normal": int(n_test_norm),
        "test_pneumonia": int(n_test_pneu)
    },
    "test_metrics": {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }
}

metrics_json_path = RESULTS_DIR / "metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(metrics_data, f, indent=4)


# -------------------------------------------------------------
# 7. Generate Evaluation Visualizations (Confusion Matrix & ROC)
# -------------------------------------------------------------
# 7a. Confusion Matrix Plot
plt.figure(figsize=(7, 6))
cm_labels = ["No Pneumonia", "Pneumonia"]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=cm_labels,
    yticklabels=cm_labels,
    annot_kws={"size": 14, "weight": "bold"}
)
plt.title(f"Confusion Matrix (Test Set)\nAccuracy: {acc*100:.2f}% | Specificity: {specificity*100:.2f}% | Sensitivity: {rec*100:.2f}%", fontsize=12, pad=12)
plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
plt.ylabel("True Class", fontsize=11, labelpad=8)
plt.tight_layout()
cm_plot_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=300)
plt.close()

# 7b. ROC Curve Plot
plt.figure(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_test, test_scores_pneumonia)
plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, pad=12)
plt.legend(loc="lower right", fontsize=10, frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
roc_plot_path = RESULTS_DIR / "roc_curve.png"
plt.savefig(roc_plot_path, dpi=300)
plt.close()


# -------------------------------------------------------------
# 8. Save Model and Configuration
# -------------------------------------------------------------
model_path = MODEL_DIR / "random_forest.joblib"
joblib.dump(rf_model, model_path)

config_data = {
    "model_architecture": "DenseNet121 Feature Extractor + RandomForestClassifier",
    "classes": ["No Pneumonia", "Pneumonia"],
    "label_mapping": {
        "0": "No Pneumonia",
        "1": "Pneumonia"
    },
    "folder_mapping": {
        "NORMAL": 0,
        "PNEUMONIA": 1
    },
    "image_size": IMAGE_SIZE,
    "feature_dim": 1024,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "rf_hyperparameters": {
        "n_estimators": N_ESTIMATORS,
        "class_weight": "balanced",
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE
    },
    "evaluation_metrics": metrics_data["test_metrics"]
}

config_json_path = MODEL_DIR / "model_config.json"
with open(config_json_path, "w") as f:
    json.dump(config_data, f, indent=4)


# -------------------------------------------------------------
# 9. Print Terminal Summary & Explanations
# -------------------------------------------------------------
print("\n" + "="*70)
print(f"{'METRIC':<30} | {'VALUE':<12} | {'PERCENTAGE':<12}")
print("-"*70)
print(f"{'Accuracy':<30} | {acc:<12.4f} | {acc*100:>6.2f}%")
print(f"{'Precision (Pneumonia)':<30} | {prec:<12.4f} | {prec*100:>6.2f}%")
print(f"{'Recall / Sensitivity':<30} | {rec:<12.4f} | {rec*100:>6.2f}%")
print(f"{'Specificity (Normal)':<30} | {specificity:<12.4f} | {specificity*100:>6.2f}%")
print(f"{'F1-Score':<30} | {f1:<12.4f} | {f1*100:>6.2f}%")
print(f"{'ROC-AUC':<30} | {auc:<12.4f} | {'N/A':<12}")
print("-"*70)
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print("="*70)

print("\n" + "="*70)
print("EXPLANATION OF EVALUATION METRICS:")
print("="*70)
print("1. Accuracy:")
print("   The overall proportion of correct predictions across both classes.")
print("2. Recall (Sensitivity for Pneumonia):")
print("   Crucial in medicine: of all actual pneumonia cases, how many did the")
print("   model catch? High recall means very few pneumonia patients are missed.")
print("3. Specificity (True Negative Rate for Normal):")
print("   Of all healthy/normal individuals, how many were correctly classified")
print("   as No Pneumonia. High specificity means healthy patients aren't falsely alarmed.")
print("4. Precision:")
print("   When the model flags an X-ray as Pneumonia, how often is it actually correct?")
print("5. F1-Score:")
print("   The harmonic mean of precision and recall, balancing false alarms and missed cases.")
print("6. ROC-AUC:")
print("   Quantifies the model's overall diagnostic discrimination ability across all")
print("   possible decision thresholds (1.0 = perfect ranking, 0.5 = random guess).")
print("="*70)

print("\n[Artifacts Saved Successfully]")
print(f"  Model File           : {model_path.resolve()}")
print(f"  Configuration File   : {config_json_path.resolve()}")
print(f"  Metrics JSON         : {metrics_json_path.resolve()}")
print(f"  Report CSV           : {report_csv_path.resolve()}")
print(f"  Confusion Matrix PNG : {cm_plot_path.resolve()}")
print(f"  ROC Curve PNG        : {roc_plot_path.resolve()}")
print("="*70 + "\n")
