# src/serving/utils.py
"""
Fonctions utilitaires — Telecom Churn Prediction
-------------------------------------------------
Fonctions réutilisables partagées entre les modules
predict.py, main.py et interpretability.py.
"""

import os
import logging
import joblib
import pandas as pd
from src.serving.config import (
    MODEL_PATH,
    PIPELINE_PATH,
    SEUIL_CHURN,
    LOG_FORMAT,
    LOG_LEVEL
)

# -------------------------
# Logging
# -------------------------


def creer_logger(nom: str) -> logging.Logger:
    """
    Crée et retourne un logger configuré.

    Args:
        nom (str) : Nom du module appelant

    Returns:
        logging.Logger
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )
    return logging.getLogger(nom)


logger = creer_logger(__name__)


# -------------------------
# Chargement des artefacts
# -------------------------

def charger_modele():
    """
    Charge le modèle XGBoost sauvegardé.

    Returns:
        model : XGBClassifier entraîné

    Raises:
        FileNotFoundError : si le fichier modèle est absent
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lance d'abord : python src/training/train.py"
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé.")
    return model


def charger_pipeline():
    """
    Charge le pipeline de preprocessing sauvegardé.

    Returns:
        pipeline : Pipeline sklearn (StandardScaler + OneHotEncoder)

    Raises:
        FileNotFoundError : si le fichier pipeline est absent
    """
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Pipeline introuvable : {PIPELINE_PATH}\n"
            "Lance d'abord : python src/data/preprocess.py"
        )
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("Pipeline chargé.")
    return pipeline


# -------------------------
# Preprocessing
# -------------------------

def preprocesser_client(data: dict, pipeline) -> any:
    """
    Applique le pipeline de preprocessing sur les données d'un client.

    Args:
        data (dict)  : Features brutes du client
        pipeline     : Pipeline sklearn chargé

    Returns:
        numpy array : Features transformées prêtes pour le modèle
    """
    df = pd.DataFrame([data])

    # Correction TotalCharges — même traitement qu'en training
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Supprimer colonnes inutiles si présentes
    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = pipeline.transform(df)

    # Convertir matrice sparse si nécessaire
    if hasattr(X, "toarray"):
        X = X.toarray()

    return X


def preprocesser_batch(df: pd.DataFrame, pipeline) -> any:
    """
    Applique le pipeline de preprocessing sur un DataFrame complet.

    Args:
        df (pd.DataFrame) : Données brutes de plusieurs clients
        pipeline          : Pipeline sklearn chargé

    Returns:
        numpy array : Features transformées
    """
    df = df.copy()

    # Correction TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Supprimer colonnes inutiles
    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = pipeline.transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    return X


# -------------------------
# Prédiction
# -------------------------

def predire_proba(model, X, seuil: float = SEUIL_CHURN) -> tuple:
    """
    Génère label et probabilité de churn pour une ou plusieurs lignes.

    Args:
        model       : Modèle XGBoost chargé
        X           : Features preprocessées
        seuil       : Seuil de décision (défaut = 0.5)

    Returns:
        Tuple (labels, probabilites) — deux arrays numpy
    """
    probabilites = model.predict_proba(X)[:, 1]
    labels = (probabilites >= seuil).astype(int)
    return labels, probabilites


# -------------------------
# Niveau de risque
# -------------------------

def niveau_risque(probabilite: float) -> str:
    """
    Convertit une probabilité de churn en niveau de risque lisible.

    Niveaux :
        >= 0.7  → Élevé
        >= 0.4  → Moyen
        < 0.4   → Faible

    Args:
        probabilite (float) : Probabilité de churn (0-1)

    Returns:
        str : "Élevé", "Moyen" ou "Faible"
    """
    if probabilite >= 0.7:
        return "Élevé"
    elif probabilite >= 0.4:
        return "Moyen"
    else:
        return "Faible"


# -------------------------
# Résumé des prédictions
# -------------------------

def afficher_resume(labels, probabilites) -> None:
    """
    Affiche un résumé rapide des prédictions batch dans les logs.

    Args:
        labels       : Array de labels prédits (0/1)
        probabilites : Array de probabilités
    """
    total = len(labels)
    churnes = labels.sum()
    pourcentage = churnes / total * 100

    logger.info(f"Résumé : {churnes}/{total} clients à risque ({pourcentage:.1f}%)")
    logger.info(f"Probabilité moyenne : {probabilites.mean():.4f}")
    logger.info(f"Probabilité max     : {probabilites.max():.4f}")
