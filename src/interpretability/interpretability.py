# src/interpretability/interpretability.py
"""
Interpretabilité — Telecom Churn Prediction
--------------------------------------------
Génère les visualisations SHAP pour expliquer les prédictions du modèle.

Graphiques produits :
    - feature_importance.png  : Top 20 features par gain XGBoost
    - shap_summary.png        : Impact global des features (beeswarm)
    - shap_waterfall_*.png    : Explication individuelle par client

Usage :
    python src/interpretability/interpretability.py
"""

from src.serving.config import (
    MODEL_PATH,
    FEATURES_PATH,
    PROCESSED_DATA_PATH,
    REPORTS_DIR
)
import os
import logging
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Backend non-interactif — pas besoin d'interface graphique
matplotlib.use("Agg")


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration visuelle
# -------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# Nombre de clients pour SHAP — limité pour la vitesse
NB_CLIENTS_SHAP = 500


# -------------------------
# Chargement des artefacts
# -------------------------

def charger_artefacts():
    """
    Charge le modèle XGBoost et la liste des features.

    Returns:
        Tuple (model, feature_columns)

    Raises:
        FileNotFoundError : si un artefact est absent
    """
    for path in [MODEL_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefact introuvable : {path}")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    # Extraire le modèle XGBoost si encapsulé dans un pipeline
    xgb_model = (
        model.named_steps["clf"]
        if hasattr(model, "named_steps")
        else model
    )

    logger.info("Modèle et features chargés.")
    return xgb_model, feature_columns


# -------------------------
# Chargement des données
# -------------------------

def charger_donnees(feature_columns: list) -> pd.DataFrame:
    """
    Charge les données preprocessées et vérifie les colonnes.

    Args:
        feature_columns (list) : Colonnes attendues par le modèle

    Returns:
        pd.DataFrame : Features prêtes pour SHAP
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Données preprocessées introuvables : {PROCESSED_DATA_PATH}\n"
            "Lance d'abord : python src/data/preprocess.py"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Vérifier que toutes les features sont présentes
    manquantes = set(feature_columns) - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans les données : {manquantes}")

    X = df[feature_columns]
    logger.info(f"Données chargées — {X.shape[0]} lignes, {X.shape[1]} features.")
    return X


# -------------------------
# Feature Importance (Gain XGBoost)
# -------------------------

def plot_feature_importance(xgb_model, output_dir: str) -> None:
    """
    Génère le graphique des 20 features les plus importantes par gain.

    Le gain mesure l'amélioration de performance apportée par chaque feature.

    Args:
        xgb_model  : Modèle XGBoost entraîné
        output_dir : Dossier de sauvegarde
    """
    logger.info("Génération feature importance...")

    try:
        booster = xgb_model.get_booster()
        importance_dict = booster.get_score(importance_type="gain")

        importance_df = (
            pd.DataFrame(importance_dict.items(), columns=["Feature", "Gain"])
            .sort_values("Gain", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df,
            x="Gain",
            y="Feature",
            hue="Feature",
            palette="magma",
            legend=False
        )
        plt.title("Top 20 Features — Importance par Gain", fontsize=14)
        plt.xlabel("Gain total")
        plt.ylabel("")
        plt.tight_layout()

        path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(path)
        plt.close()

        logger.info(f"Feature importance sauvegardée : {path}")

    except Exception as e:
        logger.error(f"Erreur feature importance : {e}")


# -------------------------
# SHAP — Explication globale
# -------------------------

def plot_shap_summary(xgb_model, X_shap: pd.DataFrame, output_dir: str) -> tuple:
    """
    Calcule les valeurs SHAP et génère le beeswarm plot global.

    Le beeswarm montre l'impact de chaque feature sur toutes les prédictions.
    Couleur rouge = valeur feature élevée, bleu = valeur basse.

    Args:
        xgb_model  : Modèle XGBoost entraîné
        X_shap     : Echantillon de données pour SHAP
        output_dir : Dossier de sauvegarde

    Returns:
        Tuple (explainer, shap_values) — réutilisés pour les waterfall plots
    """
    logger.info(f"Calcul SHAP sur {len(X_shap)} clients...")

    explainer = shap.Explainer(xgb_model, X_shap)
    shap_values = explainer(X_shap)

    # Beeswarm — vue globale
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP — Impact global des features", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info(f"SHAP summary sauvegardé : {path}")
    return explainer, shap_values


# -------------------------
# SHAP — Explication individuelle
# -------------------------

def plot_shap_waterfall(shap_values, output_dir: str, nb_clients: int = 5) -> None:
    """
    Génère les waterfall plots pour les N premiers clients.

    Chaque waterfall montre comment chaque feature pousse la prédiction
    vers le churn (rouge) ou vers la rétention (bleu).

    Args:
        shap_values : Valeurs SHAP calculées
        output_dir  : Dossier de sauvegarde
        nb_clients  : Nombre de clients à expliquer (défaut=5)
    """
    logger.info(f"Génération waterfall pour {nb_clients} clients...")

    for i in range(min(nb_clients, len(shap_values))):
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i], max_display=15, show=False)
        plt.title(f"Client {i + 1} — Décomposition de la prédiction", fontsize=13)
        plt.tight_layout()

        path = os.path.join(output_dir, f"shap_waterfall_client_{i + 1}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    logger.info(f"Waterfall plots sauvegardés pour {nb_clients} clients.")


# -------------------------
# Main
# -------------------------

def main():
    logger.info("Démarrage de l'analyse d'interpretabilité...")

    # Créer dossier reports si absent
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Charger artefacts
    xgb_model, feature_columns = charger_artefacts()

    # 2. Charger données
    X = charger_donnees(feature_columns)

    # 3. Feature importance
    plot_feature_importance(xgb_model, REPORTS_DIR)

    # 4. Echantillon pour SHAP — limité pour la vitesse
    X_shap = X.sample(n=min(NB_CLIENTS_SHAP, len(X)), random_state=42)

    # 5. SHAP global
    _, shap_values = plot_shap_summary(xgb_model, X_shap, REPORTS_DIR)

    # 6. SHAP individuel — 5 premiers clients
    plot_shap_waterfall(shap_values, REPORTS_DIR, nb_clients=5)

    logger.info("Analyse d'interpretabilité terminée.")
    logger.info(f"Graphiques disponibles dans : {REPORTS_DIR}")


if __name__ == "__main__":
    main()
