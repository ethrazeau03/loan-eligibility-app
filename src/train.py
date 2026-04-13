from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .data import load_loan_data, preprocess_loan_data
from .logger import setup_logger

LOGGER = setup_logger("loan_train")
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def train_and_save_model():
    df = load_loan_data(ROOT / "data" / "raw" / "credit.csv")
    df = preprocess_loan_data(df)

    if "Loan_Approved" not in df.columns:
        raise ValueError("Target column 'Loan_Approved' not found.")

    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    model = LogisticRegression(max_iter=1000)
    model.fit(xtrain_scaled, ytrain)

    preds = model.predict(xtest_scaled)
    accuracy = float(accuracy_score(ytest, preds))
    LOGGER.info("Loan model accuracy: %.4f", accuracy)

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), ARTIFACT_DIR / "feature_columns.joblib")
    joblib.dump({"accuracy": accuracy}, ARTIFACT_DIR / "metrics.joblib")

    return accuracy


if __name__ == "__main__":
    train_and_save_model()
