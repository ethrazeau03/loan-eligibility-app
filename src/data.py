from pathlib import Path
import pandas as pd


def load_loan_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def preprocess_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].astype("object")
    if "Loan_Amount_Term" in df.columns:
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    fill_mode_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History"]
    for col in fill_mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "LoanAmount" in df.columns:
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    cat_cols = [col for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"] if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)

    if "Loan_Approved" in df.columns:
        df["Loan_Approved"] = df["Loan_Approved"].replace({"Y": 1, "N": 0})

    return df
