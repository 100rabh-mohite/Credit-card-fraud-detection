"""Streamlit demo for the credit-card fraud classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

MODEL_DIR = Path("models")


@st.cache_resource
def load_artifacts():
    required = [
        "model.pkl", "scaler.pkl", "threshold.pkl", "feature_names.pkl"
    ]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {', '.join(missing)}. "
            "Run python src/train.py and python src/evaluate.py first."
        )

    return (
        joblib.load(MODEL_DIR / "model.pkl"),
        joblib.load(MODEL_DIR / "scaler.pkl"),
        float(joblib.load(MODEL_DIR / "threshold.pkl")),
        joblib.load(MODEL_DIR / "feature_names.pkl"),
    )


st.title("💳 Credit Card Fraud Detection")
st.caption("Machine-learning demonstration for imbalanced transaction classification.")
st.warning(
    "Educational portfolio project only. This application is not a production "
    "banking fraud-control system and must not be used to approve or decline real payments."
)

try:
    model, scaler, threshold, features = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.info(
    "The dataset's V1–V28 columns are anonymized PCA-transformed features. "
    "For a meaningful prediction, enter values from a compatible transaction dataset."
)

st.subheader("Transaction features")
st.write("Enter all feature values. The model uses the same feature order as training.")

with st.form("transaction_form"):
    input_data = {}
    columns = st.columns(3)

    for index, feature in enumerate(features):
        with columns[index % 3]:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"feature_{feature}",
            )

    submitted = st.form_submit_button("Predict transaction", type="primary")

if submitted:
    try:
        input_df = pd.DataFrame(
            [[input_data[name] for name in features]],
            columns=features,
        )

        values = input_df.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            st.error("Please enter finite numeric values for every feature.")
            st.stop()

        scaled = scaler.transform(input_df)
        fraud_probability = float(model.predict_proba(scaled)[0, 1])
        prediction = int(fraud_probability >= threshold)

        st.subheader("Prediction result")
        left, middle, right = st.columns(3)
        left.metric("Fraud probability", f"{fraud_probability:.2%}")
        middle.metric("Decision threshold", f"{threshold:.2%}")
        right.metric("Model decision", "Fraud" if prediction else "Normal")

        if prediction:
            st.error("🚨 Model flagged this transaction as potentially fraudulent.")
        else:
            st.success("✅ Model classified this transaction as normal.")

        st.caption(
            "A probability is a model estimate, not a guarantee. "
            "The threshold was selected using validation data."
        )
    except (ValueError, TypeError) as error:
        st.error(f"Could not process these inputs: {error}")

with st.expander("About this project"):
    st.markdown(
        """
        - Training-only feature scaling and SMOTE resampling
        - Separate training, validation, and untouched test sets
        - Model selection and threshold tuning on validation data
        - Precision, recall, F1, ROC-AUC and PR-AUC evaluation
        """
    )
