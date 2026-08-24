"""Train, compare, evaluate, and package churn models."""

import json
import logging
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.preprocessing import preparer_features
from src.serving.config import (
    CATEGORICAL_FEATURES, FEATURES_PATH, METADATA_PATH, MODEL_PATH,
    MODELS_DIR, NUMERIC_FEATURES, RANDOM_STATE, RAW_DATA_PATH,
    REPORTS_DIR, TARGET_COL, TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPORT_SAVE_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")
NUMERIC_FINAL = NUMERIC_FEATURES + ["ChargesMoyennes", "NbServices", "SansInternet", "ContratLong"]
CATEGORICAL_FINAL = CATEGORICAL_FEATURES + ["SegmentTenure"]


def charger_donnees() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW_DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible absente : {TARGET_COL}")
    y = df[TARGET_COL].map({"Yes": 1, "No": 0}).fillna(df[TARGET_COL]).astype(int)
    X = preparer_features(df)
    logger.info("Donnees chargees : %s lignes, %s features brutes.", len(X), X.shape[1])
    return X, y


def construire_preprocesseur() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FINAL),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), CATEGORICAL_FINAL),
    ])


def construire_pipeline(classifier) -> Pipeline:
    return Pipeline([("preprocessor", construire_preprocesseur()), ("clf", classifier)])


def tune_model(name, classifier, parameters, X_train, y_train, cv) -> tuple[Pipeline, dict]:
    search = RandomizedSearchCV(
        construire_pipeline(classifier),
        param_distributions={f"clf__{key}": value for key, value in parameters.items()},
        n_iter=15, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, refit=True,
    )
    search.fit(X_train, y_train)
    std = search.cv_results_["std_test_score"][search.best_index_]
    logger.info("%s : CV ROC-AUC %.4f (+/- %.4f)", name, search.best_score_, std)
    return search.best_estimator_, {"model": name, "cv_mean_roc_auc": search.best_score_, "cv_std_roc_auc": std}


def baseline(X_train, y_train, cv) -> tuple[Pipeline, dict]:
    model = construire_pipeline(DummyClassifier(strategy="most_frequent"))
    scores = []
    for train_index, valid_index in cv.split(X_train, y_train):
        model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        probabilities = model.predict_proba(X_train.iloc[valid_index])[:, 1]
        scores.append(roc_auc_score(y_train.iloc[valid_index], probabilities))
    model.fit(X_train, y_train)
    return model, {"model": "DummyClassifier", "cv_mean_roc_auc": sum(scores) / len(scores), "cv_std_roc_auc": pd.Series(scores).std()}


def evaluate(model, X_test, y_test, threshold=0.5) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    labels = (probabilities >= threshold).astype(int)
    return {
        "test_roc_auc": roc_auc_score(y_test, probabilities),
        "test_pr_auc": average_precision_score(y_test, probabilities),
        "test_f1": f1_score(y_test, labels, zero_division=0),
        "test_precision": precision_score(y_test, labels, zero_division=0),
        "test_recall": recall_score(y_test, labels, zero_division=0),
    }


def threshold_analysis(model, X_test, y_test) -> pd.DataFrame:
    probabilities = model.predict_proba(X_test)[:, 1]
    rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        labels = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, labels, zero_division=0),
            "recall": recall_score(y_test, labels, zero_division=0),
            "f1": f1_score(y_test, labels, zero_division=0),
        })
    return pd.DataFrame(rows)


def main():
    X, y = charger_donnees()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), {
            "C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"], "class_weight": [None, "balanced"]}),
        ("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), {
            "n_estimators": [100, 200, 300], "max_depth": [None, 8, 16], "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"], "class_weight": [None, "balanced"]}),
        ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1), {
            "n_estimators": [100, 200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "scale_pos_weight": [1, ratio]}),
    ]
    models, rows = {}, []
    baseline_model, baseline_row = baseline(X_train, y_train, cv)
    models[baseline_row["model"]], rows = baseline_model, [baseline_row]
    for name, classifier, parameters in candidates:
        models[name], row = tune_model(name, classifier, parameters, X_train, y_train, cv)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    comparison.to_csv(REPORT_SAVE_PATH, index=False)
    champion_name = comparison.iloc[comparison["cv_mean_roc_auc"].idxmax()]["model"]
    champion = models[champion_name]
    metrics = evaluate(champion, X_test, y_test)
    thresholds = threshold_analysis(champion, X_test, y_test)
    thresholds.to_csv(os.path.join(REPORTS_DIR, "threshold_analysis.csv"), index=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(champion, MODEL_PATH)
    joblib.dump(champion.named_steps["preprocessor"].get_feature_names_out().tolist(), FEATURES_PATH)
    champion_row = comparison[comparison["model"] == champion_name].iloc[0]
    metadata = {
        "model_name": champion_name, "selection_metric": "roc_auc",
        "mean_cv_roc_auc": float(champion_row["cv_mean_roc_auc"]), "cv_std": float(champion_row["cv_std_roc_auc"]),
        "threshold": 0.5, "random_state": RANDOM_STATE,
        **{key: float(value) for key, value in metrics.items()},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    logger.info("Champion selectionne : %s", champion_name)
    logger.info("Test metrics : %s", metrics)
    logger.info("Threshold analysis :\n%s", thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
