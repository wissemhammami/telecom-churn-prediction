"""Shared raw-data preparation used by training and inference."""

import pandas as pd

from src.features.feature_engineering import appliquer_feature_engineering


def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer rows and apply the project's feature engineering."""
    result = df.copy()
    for column in ["customerID", "Churn"]:
        if column in result.columns:
            result = result.drop(columns=[column])
    if "TotalCharges" in result.columns:
        result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    return appliquer_feature_engineering(result)