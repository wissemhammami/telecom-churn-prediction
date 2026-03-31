# src/training/train.py

import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from src.serving.config import (
    PROCESSED_DATA_PATH, MODELS_DIR, TARGET_COL, RANDOM_STATE, TEST_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "xgb_churn_model.pkl")
FEATURES_SAVE_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def charger_donnees() -> tuple:
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Données introuvables : {PROCESSED_DATA_PATH}\n"
            "Lance d'abord : python src/data/preprocess.py"
        )
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"Données chargées — {X.shape[0]} lignes, {X.shape[1]} features.")
    return X, y


def evaluer_modele(nom: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(f"--- {nom} --- AUC={auc:.4f} | F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}


def entrainer_baseline(X_train, X_test, y_train, y_test) -> tuple:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    return baseline, evaluer_modele("Baseline", baseline, X_test, y_test)


def entrainer_xgboost(X_train, X_test, y_train, y_test) -> tuple:
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight : {ratio:.2f}")

    xgb = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)

    param_dist = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [1, ratio]
    }

    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=15, cv=5, scoring="roc_auc",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    logger.info(f"Meilleurs hyperparamètres : {search.best_params_}")

    metriques = evaluer_modele("XGBoost", best, X_test, y_test)

    joblib.dump(best, MODEL_SAVE_PATH)
    joblib.dump(X_train.columns.tolist(), FEATURES_SAVE_PATH)
    logger.info(f"Modèle sauvegardé : {MODEL_SAVE_PATH}")

    return best, metriques


def afficher_comparaison(baseline_metrics: dict, xgb_metrics: dict) -> None:
    logger.info(f"{'Métrique':<12} {'Baseline':>10} {'XGBoost':>10} {'Gain':>10}")
    for m in ["auc", "f1", "precision", "recall"]:
        gain = xgb_metrics[m] - baseline_metrics[m]
        logger.info(f"{m:<12} {baseline_metrics[m]:>10.4f} {xgb_metrics[m]:>10.4f} {gain:>+10.4f}")


def main():
    logger.info("Démarrage de l'entraînement...")
    X, y = charger_donnees()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
    _, baseline_metrics = entrainer_baseline(X_train, X_test, y_train, y_test)
    _, xgb_metrics = entrainer_xgboost(X_train, X_test, y_train, y_test)
    afficher_comparaison(baseline_metrics, xgb_metrics)
    logger.info("Entraînement terminé.")


if __name__ == "__main__":
    main()