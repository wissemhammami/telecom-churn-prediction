import logging
import os
import joblib
import pandas as pd

from src.features.preprocessing import prepare_features
from src.serving.config import (
    CHURN_THRESHOLD,
    MODEL_PATH,
    NEW_CUSTOMERS_PATH,
    PREDICTIONS_PATH,
)
from src.serving.utils import get_risk_level

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NEW_DATA_PATH = NEW_CUSTOMERS_PATH
OUTPUT_PATH = PREDICTIONS_PATH


def load_artifacts():
    """Load the production champion pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Artifact not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Champion loaded.")
    return model


def load_customers(path: str) -> pd.DataFrame:
    """Load customer rows from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"{len(df)} customers loaded.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw customer rows for the champion pipeline."""
    logger.info("Preprocessing applied.")
    return prepare_features(df)


def predict(model, X, threshold: float = CHURN_THRESHOLD):
    """Generate labels and probabilities at the selected threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    labels = (probabilities >= threshold).astype(int)
    logger.info(f"Predictions generated for {len(labels)} customers.")
    return labels, probabilities


def save_results(df_original: pd.DataFrame, labels, probabilities, output_path: str):
    """Save predictions and risk levels alongside the original rows."""
    results = df_original.copy()
    results["Churn_Predicted"] = labels
    results["Churn_Probability"] = probabilities.round(4)
    results["Risk_Level"] = results["Churn_Probability"].apply(get_risk_level)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Results saved: {output_path}")
    total = len(labels)
    churners = labels.sum()
    logger.info(
        f"Summary: {churners}/{total} customers at risk ({churners / total * 100:.1f}%)"
    )
    return results


def main():
    logger.info("Starting batch inference...")
    model = load_artifacts()
    df_customers = load_customers(NEW_DATA_PATH)
    X = preprocess(df_customers.copy())
    labels, probabilities = predict(model, X)
    results = save_results(df_customers, labels, probabilities, OUTPUT_PATH)
    print("\nResults:")
    print(results[["Churn_Predicted", "Churn_Probability", "Risk_Level"]].to_string())


if __name__ == "__main__":
    main()
