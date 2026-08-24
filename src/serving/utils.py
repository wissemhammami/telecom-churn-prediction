"""Shared serving helpers for preprocessing, thresholds, and risk labels."""

import logging

import numpy as np
import pandas as pd

from src.features.preprocessing import preparer_features
from src.serving.config import LOG_FORMAT, LOG_LEVEL, SEUIL_CHURN

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def preprocesser_client(data: dict, pipeline) -> np.ndarray:
    """Prepare one raw customer and apply the fitted preprocessing pipeline."""
    transformed = pipeline.transform(preparer_features(pd.DataFrame([data])))
    return transformed.toarray() if hasattr(transformed, "toarray") else transformed


def preprocesser_batch(df: pd.DataFrame, pipeline) -> np.ndarray:
    """Prepare raw customers and apply the fitted preprocessing pipeline."""
    transformed = pipeline.transform(preparer_features(df))
    return transformed.toarray() if hasattr(transformed, "toarray") else transformed


def predire_proba(model, X, seuil: float = SEUIL_CHURN) -> tuple:
    """Return binary labels and churn probabilities at the configured threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    return (probabilities >= seuil).astype(int), probabilities


def niveau_risque(probabilite: float) -> str:
    """Convert a churn probability into Faible, Moyen, or Élevé."""
    if probabilite >= 0.7:
        return "Élevé"
    if probabilite >= 0.4:
        return "Moyen"
    return "Faible"
