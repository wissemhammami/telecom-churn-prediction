"""Shared SHAP explanation helpers for API and Streamlit serving."""

from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from src.serving.config import (
    BACKGROUND_PATH,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


NUMERIC_FINAL = NUMERIC_FEATURES + [
    "ChargesMoyennes",
    "NbServices",
    "SansInternet",
    "ContratLong",
]
CATEGORICAL_FINAL = CATEGORICAL_FEATURES + ["SegmentTenure"]

COLUMN_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
    "ChargesMoyennes": "Average monthly charge ($)",
    "NbServices": "Number of optional services",
    "SansInternet": "No internet service",
    "ContratLong": "Long contract (1-2 years)",
    "SegmentTenure": "Tenure segment",
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}

VALUE_LABELS = {
    "SegmentTenure": {
        "Nouveau": "New (12 months or less)",
        "Intermediaire": "Intermediate (13-36 months)",
        "Fidele": "Loyal (over 36 months)",
    },
    "SansInternet": {1: "Yes", 0: "No"},
    "ContratLong": {1: "Yes", 0: "No"},
    "SeniorCitizen": {1: "Yes", 0: "No"},
}


def _to_dense(values):
    """Convert sparse transformed values to a dense array when needed."""
    return values.toarray() if hasattr(values, "toarray") else values


def source_column_for_feature(feature_name: str) -> str:
    """Map one transformed model feature to its prepared source column."""
    if feature_name.startswith("num__"):
        source_column = feature_name.removeprefix("num__")
        if source_column not in NUMERIC_FINAL:
            raise ValueError(
                f"Unknown numeric transformed feature: {feature_name}"
            )
        return source_column

    if feature_name.startswith("cat__"):
        category_feature = feature_name.removeprefix("cat__")
        matches = [
            column
            for column in CATEGORICAL_FINAL
            if category_feature.startswith(f"{column}_")
        ]
        if matches:
            return max(matches, key=len)

    raise ValueError(f"Unable to map transformed feature to source column: {feature_name}")


def _format_value(column: str, value) -> str:
    """Format a prepared source value for human-readable explanations."""
    label = VALUE_LABELS.get(column, {}).get(value)
    if label is not None:
        return label
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    return str(value)


def build_readable_explanation(
    explanation_row: shap.Explanation,
    feature_names: list[str],
    X_prepared_row: pd.DataFrame,
) -> list[dict]:
    """Group transformed SHAP values and attach readable source-column values."""
    values = np.asarray(explanation_row.values)
    if values.ndim == 2:
        values = values[0]
    grouped_values: dict[str, float] = {}
    for feature_name, value in zip(feature_names, values):
        source_column = source_column_for_feature(feature_name)
        grouped_values[source_column] = grouped_values.get(source_column, 0.0) + float(
            value
        )

    raw_sum = float(values.sum())
    grouped_sum = sum(grouped_values.values())
    if not np.isclose(raw_sum, grouped_sum, atol=1e-9, rtol=0):
        raise ValueError(
            "Grouped SHAP values do not preserve the raw SHAP sum: "
            f"{grouped_sum} != {raw_sum}"
        )

    row = X_prepared_row.iloc[0]
    readable = []
    for column, shap_value in sorted(
        grouped_values.items(), key=lambda item: abs(item[1]), reverse=True
    ):
        value = row[column]
        readable.append(
            {
                "feature": column,
                "label": COLUMN_LABELS[column],
                "value": _format_value(column, value),
                "shap": round(shap_value, 4),
                "direction": (
                    "increases risk"
                    if shap_value > 1e-6
                    else "reduces risk"
                    if shap_value < -1e-6
                    else "neutral"
                ),
            }
        )
    return readable


def build_summary(readable_list: list[dict], top_k: int = 3) -> str:
    """Build a concise summary of factors affecting the predicted risk score."""
    increasing = [item for item in readable_list if item["direction"] == "increases risk"][
        :top_k
    ]
    reducing = [item for item in readable_list if item["direction"] == "reduces risk"][
        :top_k
    ]

    def format_items(items: list[dict]) -> str:
        return ", ".join(f"{item['label']}: {item['value']}" for item in items)

    parts = []
    if increasing:
        parts.append(
            f"Main factors increasing the predicted risk score: {format_items(increasing)}."
        )
    if reducing:
        parts.append(
            f"Main factors reducing the predicted risk score: {format_items(reducing)}."
        )
    summary = " ".join(parts) or "No material factors affect the predicted risk score."
    return summary if len(summary) <= 250 else f"{summary[:247].rstrip()}..."


def build_waterfall_explanation(
    readable_list: list[dict], base_value: float
) -> shap.Explanation:
    """Build a waterfall-compatible explanation using original customer values."""
    return shap.Explanation(
        values=np.array([item["shap"] for item in readable_list], dtype=float),
        base_values=float(base_value),
        data=None,
        feature_names=[
            f"{item['label']} = {item['value']}" for item in readable_list
        ],
    )


def build_explainer(model, background_df: pd.DataFrame):
    """Build a SHAP explainer using the committed background sample."""
    pipeline = model.named_steps["preprocessor"]
    background_transformed = _to_dense(pipeline.transform(background_df))
    estimator = model.named_steps["clf"]
    try:
        return shap.Explainer(estimator, background_transformed)
    except Exception:
        return shap.TreeExplainer(
            estimator,
            data=background_transformed,
            model_output="probability",
        )


@lru_cache(maxsize=1)
def get_explainer(model):
    """Load the background sample and build one explainer per process."""
    try:
        background_df = pd.read_csv(BACKGROUND_PATH)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"SHAP background file not found: {BACKGROUND_PATH}"
        ) from error
    return build_explainer(model, background_df)


def explain_row(explainer, model, X_prepared: pd.DataFrame) -> shap.Explanation:
    """Explain prepared rows for the churn class."""
    pipeline = model.named_steps["preprocessor"]
    transformed = _to_dense(pipeline.transform(X_prepared))
    explanation = explainer(transformed)
    values = explanation.values
    base_values = explanation.base_values
    if values.ndim == 3:
        values = values[:, :, 1]
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed,
        feature_names=list(pipeline.get_feature_names_out()),
    )
