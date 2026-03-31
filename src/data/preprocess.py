# src/data/preprocess.py

import os
import logging
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.serving.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, PIPELINE_PATH,
    SCALER_PATH, MODELS_DIR, TARGET_COL,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from src.features.feature_engineering import appliquer_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

NUMERIC_FINAL = NUMERIC_FEATURES + ["ChargesMoyennes", "NbServices", "SansInternet", "ContratLong"]
CATEGORICAL_FINAL = CATEGORICAL_FEATURES + ["SegmentTenure"]


def charger_donnees() -> pd.DataFrame:
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Fichier brut introuvable : {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Données chargées — {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df


def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nb_manquants = df["TotalCharges"].isna().sum()
    if nb_manquants > 0:
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        logger.info(f"TotalCharges : {nb_manquants} valeurs manquantes imputées.")
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def construire_pipeline() -> Pipeline:
    preprocesseur = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FINAL),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FINAL)
    ])
    return Pipeline([("preprocessor", preprocesseur)])


def sauvegarder_processed(X_transformed, y: pd.Series) -> None:
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    df_processed = pd.DataFrame(X_transformed)
    df_processed[TARGET_COL] = y.values
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Données preprocessées sauvegardées : {PROCESSED_DATA_PATH}")


def sauvegarder_artefacts(pipeline: Pipeline) -> None:
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info(f"Pipeline sauvegardé : {PIPELINE_PATH}")
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler sauvegardé : {SCALER_PATH}")


def main():
    logger.info("Démarrage du preprocessing...")
    df = charger_donnees()
    df = nettoyer(df)
    df = appliquer_feature_engineering(df)
    X = df[NUMERIC_FINAL + CATEGORICAL_FINAL]
    y = df[TARGET_COL]
    logger.info(f"Features : {X.shape[1]} colonnes | Churn : {y.value_counts().to_dict()}")
    pipeline = construire_pipeline()
    pipeline.fit(X)
    X_transformed = pipeline.transform(X)
    sauvegarder_processed(X_transformed, y)
    sauvegarder_artefacts(pipeline)
    logger.info("Preprocessing terminé.")


if __name__ == "__main__":
    main()