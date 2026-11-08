"""Shared serving helpers for thresholds and risk labels."""

import logging

from src.serving.config import (
    CHURN_THRESHOLD,
    LOG_LEVEL,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)

logger = logging.getLogger(__name__)


def predict_proba_with_threshold(model, X, threshold: float = CHURN_THRESHOLD) -> tuple:
    """Return binary labels and churn probabilities at the configured threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    return (probabilities >= threshold).astype(int), probabilities


def get_risk_level(probability: float) -> str:
    """Convert a churn probability into Low, Medium, or High."""
    if probability >= RISK_HIGH_THRESHOLD:
        return "High"
    if probability >= RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"
