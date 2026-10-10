
from pathlib import Path
import json

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from joblib import load
from torchvision.models import densenet121
from torchvision import transforms

MODEL_DIR = Path("saved_model_cnn_rf")

st.set_page_config(
    page_title="Chest X-Ray CNN + Random Forest",
    page_icon="🩻",
    layout="centered"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@st.cache_resource
def load_artifacts():
    model = load(MODEL_DIR / "random_forest.joblib")
    mlb = load(MODEL_DIR / "label_binarizer.joblib")

    with open(MODEL_DIR / "config.json") as file:
        config = json.load(file)

    cnn = densenet121(weights=None)

    state = torch.load(
        MODEL_DIR / "densenet_features.pth",
        map_location="cpu",
        weights_only=True
    )
    cnn.features.load_state_dict(state)
    cnn.classifier = nn.Identity()
    cnn = cnn.to(DEVICE)
    cnn.eval()

    preprocess = transforms.Compose([
        transforms.Resize((
            config["image_size"],
            config["image_size"]
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, mlb, config, cnn, preprocess


@torch.inference_mode()
def extract_one_feature(image, cnn, preprocess):
    image_tensor = preprocess(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    features = cnn.features(image_tensor)
    features = torch.relu(features)
    features = torch.nn.functional.adaptive_avg_pool2d(
        features, (1, 1)
    )
    features = torch.flatten(features, 1)

    return features.cpu().numpy().astype(np.float32)


st.title("🩻 Chest X-Ray Classification")
st.write(
    "CNN feature extraction + Random Forest multi-label prediction"
)

try:
    model, mlb, config, cnn, preprocess = load_artifacts()
except Exception as error:
    st.error(
        "Unable to load the trained artifacts. Run "
        "`python train_model.py` successfully first."
    )
    st.caption(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"]
)

threshold = st.slider(
    "Prediction threshold (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded X-ray",
            use_container_width=True
        )

        if st.button("Predict findings", type="primary"):
            with st.spinner("Extracting features and predicting..."):
                features = extract_one_feature(
                    image, cnn, preprocess
                )

                if features.shape[1] != config["feature_dim"]:
                    st.error("Feature dimension does not match the model.")
                    st.stop()

                probability_columns = []

                for estimator in model.estimators_:
                    proba = estimator.predict_proba(features)
                    positive_class = np.where(
                        estimator.classes_ == 1
                    )[0]

                    if len(positive_class):
                        score = float(proba[0, positive_class[0]])
                    else:
                        score = 0.0

                    probability_columns.append(score)

                scores = np.asarray(probability_columns)

            results = sorted(
                [
                    {
                        "Finding": label,
                        "Score (%)": round(float(score) * 100, 2),
                        "Prediction": (
                            "Yes"
                            if score >= threshold / 100
                            else "No"
                        )
                    }
                    for label, score in zip(mlb.classes_, scores)
                ],
                key=lambda row: row["Score (%)"],
                reverse=True
            )

            st.subheader("Predicted findings")

            selected = [
                row for row in results
                if row["Prediction"] == "Yes"
            ]

            if selected:
                for row in selected:
                    st.write(
                        f"**{row['Finding']}** — "
                        f"{row['Score (%)']:.2f}%"
                    )
                    st.progress(row["Score (%)"] / 100)
            else:
                st.info(
                    "No labels reached this threshold. "
                    "This does not mean the image is normal."
                )

            with st.expander("All finding scores"):
                st.dataframe(
                    results,
                    hide_index=True,
                    use_container_width=True
                )

            st.caption(
                "Scores are Random Forest outputs, not calibrated "
                "disease probabilities."
            )

            st.warning(
                "Educational research prototype only. Not intended "
                "for diagnosis or clinical decisions."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
