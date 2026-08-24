# src/inference/predict.py

import os
import logging
import joblib
import pandas as pd

from src.features.preprocessing import preparer_features
from src.serving.config import MODEL_PATH, NEW_CUSTOMERS_PATH, PREDICTIONS_PATH, SEUIL_CHURN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEW_DATA_PATH = NEW_CUSTOMERS_PATH
OUTPUT_PATH = PREDICTIONS_PATH


def charger_artefacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Artefact introuvable : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Champion charge.")
    return model


def charger_clients(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    logger.info(f"{len(df)} clients chargés.")
    return df


def preprocesser(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing applique.")
    return preparer_features(df)


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
    model = charger_artefacts()
    df_clients = charger_clients(NEW_DATA_PATH)
    X = preprocesser(df_clients.copy())
    labels, probs = predire(model, X)
    resultats = sauvegarder(df_clients, labels, probs, OUTPUT_PATH)
    print("\nRésultats :")
    print(resultats[["Churn_Predit", "Churn_Probabilite", "Niveau_Risque"]].to_string())


if __name__ == "__main__":
    main()