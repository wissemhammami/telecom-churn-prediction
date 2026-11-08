"""Regression tests for readable SHAP explanations and waterfall inputs."""

import pandas as pd

from src.features.preprocessing import prepare_features
from src.serving.explain import (
    build_readable_explanation,
    build_summary,
    build_waterfall_explanation,
    explain_row,
    get_explainer,
)
from src.serving.main import model, predict_one
from src.tests.test_predict import CLIENT_EXEMPLE


def _readable_explanation():
    """Build the readable explanation for the shared example customer."""
    prepared = prepare_features(pd.DataFrame([CLIENT_EXEMPLE]))
    explanation = explain_row(get_explainer(model), model, prepared)
    readable = build_readable_explanation(
        explanation, list(explanation.feature_names), prepared
    )
    return prepared, explanation, readable


def test_waterfall_endpoint_matches_model_probability():
    """All grouped SHAP values preserve the model output within tolerance."""
    _, explanation, readable = _readable_explanation()
    waterfall = build_waterfall_explanation(
        readable, float(explanation.base_values[0])
    )
    probability = predict_one(CLIENT_EXEMPLE).churn_probability
    endpoint = float(waterfall.base_values + waterfall.values.sum())

    assert abs(endpoint - probability) < 0.02


def test_waterfall_contains_all_grouped_entries():
    """The waterfall input includes every distinct prepared source column."""
    prepared, explanation, readable = _readable_explanation()
    waterfall = build_waterfall_explanation(
        readable, float(explanation.base_values[0])
    )

    assert len(waterfall.values) == len(prepared.columns)
    assert len(waterfall.values) > 10


def test_summary_uses_label_value_format():
    """Summaries separate readable labels and values with a colon."""
    _, _, readable = _readable_explanation()
    summary = build_summary(readable)

    assert ": " in summary
    assert "))" not in summary
    assert ") (" not in summary
