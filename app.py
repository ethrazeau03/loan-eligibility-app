from pathlib import Path
import pandas as pd
import streamlit as st

from src.data import load_loan_data, preprocess_loan_data
from src.predict import predict_loan
from src.train import train_and_save_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")
st.title("Loan Eligibility Prediction")

if st.button("Train / Refresh Model"):
    accuracy = train_and_save_model()
    st.success(f"Training complete. Accuracy: {accuracy:.2%}")

data_path = ROOT / "data" / "raw" / "credit.csv"
if data_path.exists():
    raw_df = load_loan_data(data_path)
    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head())

    processed_df = preprocess_loan_data(raw_df)
    feature_cols = [c for c in processed_df.columns if c != "Loan_Approved"]

    st.subheader("Prediction Form")
    user_input = {}
    for col in feature_cols:
        user_input[col] = st.number_input(col, value=float(processed_df[col].median()))
    input_df = pd.DataFrame([user_input])

    if st.button("Predict Approval"):
        try:
            pred, proba = predict_loan(input_df)
            label = "Approved" if pred == 1 else "Not Approved"
            st.success(f"Prediction: {label}")
            st.write(f"Approval probability: {proba:.2%}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.warning("Place credit.csv inside data/raw before using the app.")
