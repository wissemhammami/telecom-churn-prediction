"""Create a held-out probability calibration report for the saved model."""

import os
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.preprocessing import prepare_features
from src.serving.config import (
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)

matplotlib.use("Agg")


def main() -> None:
    """Evaluate held-out probabilities and save a calibration curve."""
    df = pd.read_csv(RAW_DATA_PATH)
    y = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    X = prepare_features(df)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probabilities)

    calibration = pd.DataFrame(
        {"probability": probabilities, "actual": y_test.to_numpy()}
    )
    calibration["bin"] = pd.qcut(
        calibration["probability"], q=10, labels=False, duplicates="drop"
    )
    curve = (
        calibration.groupby("bin", as_index=False)
        .agg(mean_predicted=("probability", "mean"), fraction_positive=("actual", "mean"))
    )
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(curve["mean_predicted"], curve["fraction_positive"], "o-", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed churn rate")
    plt.title("Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "calibration.png"), dpi=150)
    plt.close()

    mean_predicted = probabilities.mean()
    observed = y_test.mean()
    confidence = (
        "over-confident"
        if abs(mean_predicted - observed) > 0.02 and mean_predicted > observed
        else "under-confident"
        if abs(mean_predicted - observed) > 0.02
        else "approximately aligned"
    )
    print(f"Brier score: {brier:.6f}")
    print(f"Calibration assessment: {confidence}")


if __name__ == "__main__":
    main()
