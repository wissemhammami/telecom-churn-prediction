"""Generate global and individual SHAP reports for the selected champion."""

import logging
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from src.features.preprocessing import preparer_features
from src.serving.config import FEATURES_PATH, RAW_DATA_PATH, REPORTS_DIR, MODEL_PATH

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
NB_CLIENTS_SHAP = 500


def charger_artefacts():
    paths = [MODEL_PATH, FEATURES_PATH]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefact introuvable : {path}")
    model = joblib.load(MODEL_PATH)
    return model, model.named_steps["preprocessor"], joblib.load(FEATURES_PATH)


def charger_donnees() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    return preparer_features(df)


def plot_feature_importance(model, feature_names, output_dir: str) -> None:
    estimator = model.named_steps.get("clf", model)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = estimator.coef_[0]
    else:
        logger.warning("Le champion ne fournit pas d'importances de features.")
        return
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": abs(importances)}).sort_values("Importance", ascending=False).head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", palette="magma", legend=False)
    plt.title("Top 20 Features - Importance du champion")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()


def plot_shap_summary(model, pipeline, X_shap: pd.DataFrame, output_dir: str):
    transformed = pipeline.transform(X_shap)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    estimator = model.named_steps.get("clf", model)
    explanation = shap.Explainer(estimator, transformed)(transformed)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), bbox_inches="tight")
    plt.close()
    return explanation


def plot_shap_waterfall(shap_values, output_dir: str, nb_clients: int = 5) -> None:
    for index in range(min(nb_clients, len(shap_values))):
        shap.plots.waterfall(shap_values[index], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_waterfall_client_{index + 1}.png"), bbox_inches="tight")
        plt.close()


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    model, pipeline, feature_names = charger_artefacts()
    X = charger_donnees()
    X_shap = X.sample(n=min(NB_CLIENTS_SHAP, len(X)), random_state=42)
    plot_feature_importance(model, feature_names, REPORTS_DIR)
    shap_values = plot_shap_summary(model, pipeline, X_shap, REPORTS_DIR)
    plot_shap_waterfall(shap_values, REPORTS_DIR)
    logger.info("Analyse d'interpretabilite terminee : %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
