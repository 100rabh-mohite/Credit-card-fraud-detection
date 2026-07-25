import streamlit as st
import joblib
import pandas as pd


# -----------------------------
# Load Model and Scaler
# -----------------------------

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"


@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler



model, scaler = load_model()



# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)


st.title(
    "💳 Credit Card Fraud Detection"
)


st.write(
    "Enter transaction details to predict whether the transaction is Fraud or Normal."
)



# -----------------------------
# Input Features
# -----------------------------

st.subheader(
    "Transaction Details"
)


features = [

    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"

]



input_data = {}



for feature in features:

    input_data[feature] = st.number_input(
        feature,
        value=0.0
    )



# Convert input to dataframe

input_df = pd.DataFrame(
    [input_data]
)



# -----------------------------
# Prediction
# -----------------------------

if st.button(
    "Predict Transaction"
):

    try:

        # Scale input

        scaled_data = scaler.transform(
            input_df
        )


        # Prediction

        prediction = model.predict(
            scaled_data
        )


        probability = model.predict_proba(
            scaled_data
        )[0][1]



        st.subheader(
            "Prediction Result"
        )


        if prediction[0] == 1:

            st.error(
                "🚨 Fraud Transaction Detected"
            )

        else:

            st.success(
                "✅ Normal Transaction"
            )



        st.write(
            f"Fraud Probability: {probability:.4f}"
        )



    except Exception as e:

        st.error(
            f"Error: {e}"
        )