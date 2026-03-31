# src/inference/predict.py

import os
import logging
import joblib
import pandas as pd

from src.features.feature_engineering import appliquer_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new", "new_customers.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "new", "new_customers_predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_churn_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessor_pipeline.pkl")
SEUIL_CHURN = 0.5


def charger_artefacts():
    for path in [MODEL_PATH, PIPELINE_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefact introuvable : {path}")
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("Modèle et pipeline chargés.")
    return model, pipeline


def charger_clients(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    logger.info(f"{len(df)} clients chargés.")
    return df


def preprocesser(df: pd.DataFrame, pipeline) -> any:
    df = df.copy()
    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df = appliquer_feature_engineering(df)
    X = pipeline.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    logger.info("Preprocessing appliqué.")
    return X


def predire(model, X, seuil: float = SEUIL_CHURN):
    probabilites = model.predict_proba(X)[:, 1]
    labels = (probabilites >= seuil).astype(int)
    logger.info(f"Prédictions générées pour {len(labels)} clients.")
    return labels, probabilites


def sauvegarder(df_original: pd.DataFrame, labels, probabilites, output_path: str):
    resultats = df_original.copy()
    resultats["Churn_Predit"] = labels
    resultats["Churn_Probabilite"] = probabilites.round(4)
    resultats["Niveau_Risque"] = resultats["Churn_Probabilite"].apply(
        lambda p: "Élevé" if p >= 0.7 else ("Moyen" if p >= 0.4 else "Faible")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    resultats.to_csv(output_path, index=False)
    logger.info(f"Résultats sauvegardés : {output_path}")
    total = len(labels)
    churnes = labels.sum()
    logger.info(f"Résumé : {churnes}/{total} clients à risque ({churnes / total * 100:.1f}%)")
    return resultats


def main():
    logger.info("Démarrage de l'inférence batch...")
    model, pipeline = charger_artefacts()
    df_clients = charger_clients(NEW_DATA_PATH)
    X = preprocesser(df_clients.copy(), pipeline)
    labels, probs = predire(model, X)
    resultats = sauvegarder(df_clients, labels, probs, OUTPUT_PATH)
    print("\nRésultats :")
    print(resultats[["Churn_Predit", "Churn_Probabilite", "Niveau_Risque"]].to_string())


if __name__ == "__main__":
    main()