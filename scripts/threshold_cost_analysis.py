"""Compare retention thresholds using training CV and held-out test data."""

import os
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

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

RETENTION_OFFER_COST = 20
CUSTOMER_LIFETIME_VALUE = 300
RETENTION_SUCCESS_RATE = 0.3


def net_benefit(y_true, probabilities, threshold):
    """Calculate expected net benefit at one threshold."""
    labels = probabilities >= threshold
    true_positive = ((labels == 1) & (y_true == 1)).sum()
    false_positive = ((labels == 1) & (y_true == 0)).sum()
    return (
        true_positive * RETENTION_SUCCESS_RATE * CUSTOMER_LIFETIME_VALUE
        - (true_positive + false_positive) * RETENTION_OFFER_COST
    )


def main() -> None:
    """Select a threshold on training CV predictions and assess it on test data."""
    df = pd.read_csv(RAW_DATA_PATH)
    y = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    X = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = joblib.load(MODEL_PATH)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = [
        {
            "threshold": threshold,
            "net_benefit": net_benefit(
                y_train.to_numpy(), train_probabilities, threshold
            ),
        }
        for threshold in thresholds
    ]
    report = pd.DataFrame(rows)
    best = report.loc[report["net_benefit"].idxmax()]
    model.fit(X_train, y_train)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_labels = test_probabilities >= best["threshold"]
    test_benefit = net_benefit(y_test.to_numpy(), test_probabilities, best["threshold"])
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report.to_csv(os.path.join(REPORTS_DIR, "threshold_cost_analysis.csv"), index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(report["threshold"], report["net_benefit"], "o-")
    plt.axvline(0.5, linestyle="--", label="Production threshold 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Expected net benefit")
    plt.title("Retention Threshold Cost Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "threshold_cost_curve.png"), dpi=150)
    plt.close()
    print(f"Recommended training-CV threshold: {best['threshold']:.2f}")
    print(f"Training-CV net benefit: {best['net_benefit']:.2f}")
    print(f"Test precision: {precision_score(y_test, test_labels):.4f}")
    print(f"Test recall: {recall_score(y_test, test_labels):.4f}")
    print(f"Test net benefit: {test_benefit:.2f}")
    print("Production threshold remains 0.50 until business costs are validated.")


if __name__ == "__main__":
    main()
