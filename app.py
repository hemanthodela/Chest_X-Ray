"""
Chest X-Ray Pneumonia Detection Web Application
==============================================
Binary Classification using DenseNet121 Feature Extractor + Random Forest Classifier.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
import joblib

import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision import transforms
import streamlit as st

# -------------------------------------------------------------
# App Configuration & Page Layout
# -------------------------------------------------------------
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished medical AI dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        border-left: 6px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .main-header p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 14px;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-densenet {
        background-color: #0284c7;
        color: white;
    }
    .badge-rf {
        background-color: #059669;
        color: white;
    }
    .badge-binary {
        background-color: #7c3aed;
        color: white;
    }

    .prediction-box-pneumonia {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 18px;
    }
    
    .prediction-box-normal {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 18px;
    }
    
    .prediction-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .pred-pneumonia-title { color: #b91c1c; }
    .pred-normal-title { color: #15803d; }
    
    .prediction-result {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }
    .pred-pneumonia-text { color: #991b1b; }
    .pred-normal-text { color: #166534; }

    .score-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .score-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .score-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    
    .disclaimer-card {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        color: #92400e;
        font-size: 13px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Paths and Device Setup
# -------------------------------------------------------------
MODEL_DIR = Path("saved_model_binary")
RESULTS_DIR = Path("evaluation_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------
# Load Pretrained Artifacts
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DenseNet121 & Random Forest models...")
def load_model_artifacts():
    """Load the trained Random Forest model, config, and DenseNet121 feature extractor."""
    model_file = MODEL_DIR / "random_forest.joblib"
    config_file = MODEL_DIR / "model_config.json"
    weights_file = MODEL_DIR / "densenet_features.pth"
    
    # Check fallback directory if needed
    if not weights_file.exists():
        fallback_weights = Path("saved_model_cnn_rf") / "densenet_features.pth"
        if fallback_weights.exists():
            weights_file = fallback_weights

    if not model_file.exists() or not config_file.exists():
        raise FileNotFoundError(
            f"Trained artifacts not found in '{MODEL_DIR}'. "
            "Please run 'python train_model.py' to generate and save the binary classification model."
        )

    # 1. Load configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    # 2. Load Random Forest model
    rf_model = joblib.load(model_file)

    # 3. Load DenseNet121 feature extractor
    cnn = densenet121(weights=None)
    if weights_file.exists():
        state = torch.load(weights_file, map_location="cpu")
        cnn.features.load_state_dict(state)
    cnn.classifier = nn.Identity()
    cnn = cnn.to(DEVICE)
    cnn.eval()

    # 4. Preprocessing transform
    image_size = config.get("image_size", 224)
    norm_mean = config.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
    norm_std = config.get("normalization", {}).get("std", [0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return rf_model, config, cnn, preprocess


@torch.inference_mode()
def extract_single_image_features(image: Image.Image, cnn, preprocess):
    """Preprocess image and extract 1024-dimensional feature vector."""
    rgb_img = image.convert("RGB")
    tensor = preprocess(rgb_img).unsqueeze(0).to(DEVICE)
    features = cnn.features(tensor)
    features = torch.relu(features)
    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    features = torch.flatten(features, 1)
    return features.cpu().numpy().astype(np.float32)


# -------------------------------------------------------------
# Sidebar: System & Architecture Info
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=64)
    st.title("Project Details")
    st.markdown("""
    **Architecture:**
    - Pretrained **DenseNet121** (ImageNet)
    - 1,024 Feature Extractor (Frozen)
    - **Random Forest Classifier**
    - Balanced Class Weighting
    """)
    st.markdown("---")
    
    st.subheader("Target Classes")
    st.markdown("""
    - **0: No Pneumonia (NORMAL)**
    - **1: Pneumonia (PNEUMONIA)**
    """)
    st.markdown("---")
    
    st.subheader("Compute Device")
    st.caption(f"Torch Device: `{DEVICE}`")
    
    st.markdown("---")
    st.caption("Educational Machine Learning Project\nAntigravity IDE • Python 3.14")


# -------------------------------------------------------------
# Main Application Header
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🩻 Chest X-Ray Pneumonia Detection</h1>
    <p>Predicting Pneumonia vs No Pneumonia using DenseNet121 Feature Extraction and Random Forest Classification.</p>
    <div style="margin-top: 12px;">
        <span class="status-badge badge-densenet">DenseNet121 (1024-D)</span>
        <span class="status-badge badge-rf">Random Forest Classifier</span>
        <span class="status-badge badge-binary">Binary Classification</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Attempt Artifact Loading
# -------------------------------------------------------------
try:
    rf_model, config, cnn, preprocess = load_model_artifacts()
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False
    st.error("⚠️ **Model Artifacts Not Yet Available**")
    st.info(
        f"{e}\n\n"
        "To generate the trained model, run the training pipeline in your terminal:\n\n"
        "```powershell\npython train_model.py\n```"
    )


# -------------------------------------------------------------
# Main Tabs: 1. Prediction & 2. Evaluation Metrics
# -------------------------------------------------------------
tab_predict, tab_evaluation = st.tabs([
    "🔍 Image Diagnosis & Score Prediction",
    "📊 Model Performance & Test Set Metrics"
])


# =============================================================
# TAB 1: Prediction
# =============================================================
with tab_predict:
    if not artifacts_ready:
        st.warning("Please complete model training before performing inference.")
    else:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("1. Upload Chest X-Ray")
            uploaded_file = st.file_uploader(
                "Upload a frontal chest X-ray image (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
                help="Accepts chest radiographic images in standard image formats."
            )

            # Convenience: Allow user to pick a sample image from the test set
            st.markdown("##### Or pick a sample test image:")
            sample_col1, sample_col2 = st.columns(2)
            
            sample_image_path = None
            dataset_root = Path("chest_xray")
            if not dataset_root.exists() and Path("chest_xray/chest_xray").exists():
                dataset_root = Path("chest_xray/chest_xray")

            if dataset_root.exists():
                test_norm_dir = dataset_root / "test" / "NORMAL"
                test_pneu_dir = dataset_root / "test" / "PNEUMONIA"

                norm_samples = list(test_norm_dir.glob("*.jpeg")) + list(test_norm_dir.glob("*.jpg")) + list(test_norm_dir.glob("*.png"))
                pneu_samples = list(test_pneu_dir.glob("*.jpeg")) + list(test_pneu_dir.glob("*.jpg")) + list(test_pneu_dir.glob("*.png"))

                with sample_col1:
                    if st.button("📁 Load Sample Normal X-Ray", use_container_width=True):
                        if norm_samples:
                            st.session_state["sample_img"] = norm_samples[0]
                with sample_col2:
                    if st.button("📁 Load Sample Pneumonia X-Ray", use_container_width=True):
                        if pneu_samples:
                            st.session_state["sample_img"] = pneu_samples[0]

            # Determine which image to process
            active_image = None
            image_source_label = ""

            if uploaded_file is not None:
                try:
                    active_image = Image.open(uploaded_file)
                    image_source_label = f"Uploaded file: {uploaded_file.name}"
                except Exception as img_err:
                    st.error(f"❌ Could not read the uploaded image file: {img_err}")
            elif "sample_img" in st.session_state and st.session_state["sample_img"].exists():
                try:
                    active_image = Image.open(st.session_state["sample_img"])
                    image_source_label = f"Sample: {st.session_state['sample_img'].name}"
                except Exception as img_err:
                    st.error(f"❌ Could not read sample image: {img_err}")

            if active_image is not None:
                st.image(
                    active_image,
                    caption=f"{image_source_label} ({active_image.size[0]}×{active_image.size[1]} px)",
                    use_container_width=True
                )
                
                predict_clicked = st.button("🚀 Run Pneumonia Prediction", type="primary", use_container_width=True)

        with col_result:
            st.subheader("2. Model Prediction & Scores")
            
            if active_image is None:
                st.info("👈 Please upload a chest X-ray image or click one of the sample buttons to see model predictions.")
            elif not predict_clicked and "last_pred" not in st.session_state:
                st.caption("Click **'Run Pneumonia Prediction'** to extract features and compute class scores.")
            else:
                try:
                    with st.spinner("Extracting 1,024 DenseNet121 features and running Random Forest..."):
                        # Extract features
                        features = extract_single_image_features(active_image, cnn, preprocess)

                        # Validate feature dimensionality
                        expected_dim = config.get("feature_dim", 1024)
                        if features.shape[1] != expected_dim:
                            st.error(f"Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}")
                            st.stop()

                        # Predict probabilities: [Prob(Normal), Prob(Pneumonia)]
                        probas = rf_model.predict_proba(features)[0]
                        pred_class_idx = int(np.argmax(probas))
                        
                        normal_score = float(probas[0])
                        pneumonia_score = float(probas[1])

                        # Map class names
                        classes = config.get("classes", ["No Pneumonia", "Pneumonia"])
                        predicted_label = classes[pred_class_idx]
                        is_pneumonia = (pred_class_idx == 1)

                    # Display Top Result Box
                    if is_pneumonia:
                        st.markdown(f"""
                        <div class="prediction-box-pneumonia">
                            <div class="prediction-title pred-pneumonia-title">Prediction Result</div>
                            <h2 class="prediction-result pred-pneumonia-text">⚠️ {predicted_label}</h2>
                            <p style="margin: 6px 0 0 0; color: #b91c1c; font-size: 14px;">
                                Pneumonia probability score: <strong>{pneumonia_score*100:.2f}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box-normal">
                            <div class="prediction-title pred-normal-title">Prediction Result</div>
                            <h2 class="prediction-result pred-normal-text">✅ {predicted_label}</h2>
                            <p style="margin: 6px 0 0 0; color: #15803d; font-size: 14px;">
                                No Pneumonia probability score: <strong>{normal_score*100:.2f}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display Specific Scores as requested in prompt
                    st.markdown("#### Model Class Scores")
                    score_col1, score_col2 = st.columns(2)
                    with score_col1:
                        st.markdown(f"""
                        <div class="score-card">
                            <div class="score-label">No Pneumonia Score</div>
                            <div class="score-value" style="color: #16a34a;">{normal_score:.4f}</div>
                            <small style="color: #64748b;">({normal_score*100:.2f}%)</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with score_col2:
                        st.markdown(f"""
                        <div class="score-card">
                            <div class="score-label">Pneumonia Score</div>
                            <div class="score-value" style="color: #dc2626;">{pneumonia_score:.4f}</div>
                            <small style="color: #64748b;">({pneumonia_score*100:.2f}%)</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # Score Comparison Visualization (Bar Chart)
                    st.markdown("#### Score Comparison Chart")
                    chart_df = pd.DataFrame({
                        "Class": ["No Pneumonia", "Pneumonia"],
                        "Model Score": [normal_score, pneumonia_score],
                        "Percentage (%)": [round(normal_score * 100, 2), round(pneumonia_score * 100, 2)]
                    })
                    
                    st.bar_chart(
                        chart_df,
                        x="Class",
                        y="Model Score",
                        color="Class",
                        use_container_width=True
                    )

                    # Detailed Probability Breakdown Table
                    with st.expander("Detailed Probability Breakdown"):
                        st.dataframe(
                            chart_df,
                            use_container_width=True,
                            hide_index=True
                        )

                except Exception as pred_err:
                    st.error(f"❌ Prediction failed: {pred_err}")

        # Educational Disclaimer
        st.markdown("""
        <div class="disclaimer-card">
            <strong>⚠️ Educational & Research Disclaimer:</strong><br>
            This software is an educational machine learning project demonstrating CNN feature extraction combined with Random Forest classification. 
            The output scores represent Random Forest probability estimates based on training distribution and are <strong>not calibrated clinical probabilities</strong>. 
            This application is <strong>not intended for clinical diagnosis, patient management, or medical decision-making</strong>.
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# TAB 2: Model Performance & Test Set Metrics
# =============================================================
with tab_evaluation:
    st.subheader("Model Evaluation on Held-Out Test Set")
    st.caption("All metrics shown below are calculated strictly on the held-out test dataset (624 test images), without any data leakage.")

    metrics_file = RESULTS_DIR / "metrics.json"
    report_file = RESULTS_DIR / "classification_report.csv"
    cm_img_file = RESULTS_DIR / "confusion_matrix.png"
    roc_img_file = RESULTS_DIR / "roc_curve.png"

    if not metrics_file.exists():
        st.warning(
            "Evaluation results not found yet. Run `python train_model.py` to evaluate the model on the test split."
        )
    else:
        with open(metrics_file, "r") as f:
            eval_data = json.load(f)

        test_m = eval_data.get("test_metrics", {})
        ds_info = eval_data.get("dataset", {})

        # Top KPI Metrics Row
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        with m_col1:
            st.metric("Accuracy", f"{test_m.get('accuracy', 0)*100:.2f}%")
        with m_col2:
            st.metric("Sensitivity (Recall)", f"{test_m.get('recall_sensitivity', 0)*100:.2f}%")
        with m_col3:
            st.metric("Specificity", f"{test_m.get('specificity', 0)*100:.2f}%")
        with m_col4:
            st.metric("Precision", f"{test_m.get('precision', 0)*100:.2f}%")
        with m_col5:
            st.metric("F1-Score", f"{test_m.get('f1_score', 0):.4f}")
        with m_col6:
            st.metric("ROC-AUC", f"{test_m.get('roc_auc', 0):.4f}")

        st.markdown("---")

        # Visualizations: Confusion Matrix and ROC Curve
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.markdown("##### Confusion Matrix (Test Set)")
            if cm_img_file.exists():
                st.image(str(cm_img_file), use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")
        
        with viz_col2:
            st.markdown("##### Receiver Operating Characteristic (ROC) Curve")
            if roc_img_file.exists():
                st.image(str(roc_img_file), use_container_width=True)
            else:
                st.info("ROC curve image not found.")

        st.markdown("---")

        # Classification Report Table & Dataset Splits
        table_col1, table_col2 = st.columns([3, 2])
        with table_col1:
            st.markdown("##### Detailed Classification Report")
            if report_file.exists():
                report_df = pd.read_csv(report_file, index_col=0)
                st.dataframe(report_df, use_container_width=True)
            else:
                st.info("Classification report CSV not found.")

        with table_col2:
            st.markdown("##### Dataset Split Breakdown")
            split_df = pd.DataFrame([
                {"Split": "Training", "Normal": ds_info.get("train_normal", "-"), "Pneumonia": ds_info.get("train_pneumonia", "-"), "Total": ds_info.get("train_samples", "-")},
                {"Split": "Validation", "Normal": ds_info.get("val_normal", "-"), "Pneumonia": ds_info.get("val_pneumonia", "-"), "Total": ds_info.get("val_samples", "-")},
                {"Split": "Test", "Normal": ds_info.get("test_normal", "-"), "Pneumonia": ds_info.get("test_pneumonia", "-"), "Total": ds_info.get("test_samples", "-")},
            ])
            st.dataframe(split_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### Explanation of Evaluation Metrics")
        st.markdown("""
        - **Accuracy:** The overall percentage of correctly classified test images across both Normal and Pneumonia classes.
        - **Sensitivity / Recall (Pneumonia):** The proportion of actual pneumonia cases that were correctly identified. In a medical context, high recall is vital to minimize missed diagnoses (false negatives).
        - **Specificity (Normal):** The proportion of actual healthy/normal cases correctly identified as No Pneumonia, minimizing unnecessary alarms and interventions.
        - **Precision:** Of all chest X-rays flagged by the model as Pneumonia, the fraction that genuinely had pneumonia.
        - **F1-Score:** The harmonic mean of precision and recall, reflecting the balanced performance on positive cases.
        - **ROC-AUC:** Area Under the Receiver Operating Characteristic curve. Measures the model's discriminative power across all decision thresholds.
        """)
