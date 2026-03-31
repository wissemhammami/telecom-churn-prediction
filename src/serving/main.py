# src/serving/main.py

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.schemas import (
    CustomerInput, BatchCustomerInput,
    PredictionOutput, BatchPredictionOutput,
)
from src.serving.utils import niveau_risque
from src.features.feature_engineering import appliquer_feature_engineering

logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
MODEL_PATH    = BASE_DIR / "models" / "xgb_churn_model.pkl"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessor_pipeline.pkl"
SEUIL_CHURN   = 0.5

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predicts churn probability for telecom customers using XGBoost.",
    version="1.0.0",
)

try:
    model    = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("Model and pipeline loaded successfully.")
except Exception as e:
    logger.error("Startup error: %s", e)
    model    = None
    pipeline = None


def preprocess(data: dict):
    df = pd.DataFrame([data])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df = appliquer_feature_engineering(df)
    X = pipeline.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def predict_one(data: dict) -> PredictionOutput:
    X    = preprocess(data)
    prob = float(model.predict_proba(X)[:, 1][0])
    label = int(prob >= SEUIL_CHURN)
    return PredictionOutput(
        churn_label=label,
        churn_probability=round(prob, 4),
        niveau_risque=niveau_risque(prob),
    )


@app.get("/")
def root():
    return {"status": "Telecom Churn Prediction API is running."}


@app.get("/health")
def health():
    return {
        "model_loaded":    model is not None,
        "pipeline_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        result = predict_one(customer.model_dump())
        logger.info("Prediction: label=%s prob=%s", result.churn_label, result.churn_probability)
        return result
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchCustomerInput):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        results      = [predict_one(c.model_dump()) for c in batch.customers]
        total        = len(results)
        nb_churners  = sum(r.churn_label for r in results)
        logger.info("Batch: %s clients processed.", total)
        return BatchPredictionOutput(
            results=results,
            total=total,
            nb_churners=nb_churners,
            taux_churn=round(nb_churners / total * 100, 2),
        )
    except Exception as e:
        logger.error("Batch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interpret")
def interpret(customer: CustomerInput, top_n: int = 5):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        import shap
        X             = preprocess(customer.model_dump())
        result        = predict_one(customer.model_dump())
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        explainer     = shap.TreeExplainer(model)
        shap_values   = explainer.shap_values(X)
        shap_dict     = dict(zip(feature_names, shap_values[0]))
        top_features  = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        )
        return {
            "churn_label":       result.churn_label,
            "churn_probability": result.churn_probability,
            "niveau_risque":     result.niveau_risque,
            "top_features":      {k: round(float(v), 4) for k, v in top_features.items()},
        }
    except Exception as e:
        logger.error("Interpret error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))