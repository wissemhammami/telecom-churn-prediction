import logging

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.features.preprocessing import prepare_features
from src.serving.config import (
    CHURN_THRESHOLD,
    LOG_FORMAT,
    LOG_LEVEL,
    MODEL_PATH,
)
from src.serving.explain import (
    build_readable_explanation,
    build_summary,
    explain_row,
    get_explainer,
)
from src.serving.schemas import (
    BatchCustomerInput,
    BatchPredictionOutput,
    CustomerInput,
    PredictionOutput,
)
from src.serving.utils import get_risk_level

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predicts churn probability using the selected champion model.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
    pipeline = model.named_steps["preprocessor"]
    logger.info("Model and pipeline loaded successfully.")
except Exception:
    logger.exception("Startup error")
    model = None
    pipeline = None


def preprocess(data: dict):
    """Prepare one raw customer for the loaded champion pipeline."""
    return prepare_features(pd.DataFrame([data]))


def predict_one(data: dict) -> PredictionOutput:
    """Return a prediction for one raw customer."""
    X = preprocess(data)
    prob = float(model.predict_proba(X)[:, 1][0])
    label = int(prob >= CHURN_THRESHOLD)
    return PredictionOutput(
        churn_label=label,
        churn_probability=round(prob, 4),
        risk_level=get_risk_level(prob),
    )


def get_model():
    """Return the loaded model or raise a service-unavailable error."""
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return model


@app.get("/")
def root():
    """Return the API status message."""
    return {"status": "Telecom Churn Prediction API is running."}


@app.get("/health")
def health():
    """Report whether the champion and its preprocessor are loaded."""
    response = {
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
    }
    if model is None or pipeline is None:
        return JSONResponse(status_code=503, content=response)
    return response


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput, _model=Depends(get_model)):
    try:
        result = predict_one(customer.model_dump())
        logger.info(
            "Prediction: label=%s prob=%s", result.churn_label, result.churn_probability
        )
        return result
    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchCustomerInput, _model=Depends(get_model)):
    try:
        results = [predict_one(c.model_dump()) for c in batch.customers]
        total = len(results)
        num_churners = sum(r.churn_label for r in results)
        logger.info("Batch: %s customers processed.", total)
        return BatchPredictionOutput(
            results=results,
            total=total,
            num_churners=num_churners,
            churn_rate_percent=round(num_churners / total * 100, 2),
        )
    except Exception:
        logger.exception("Batch error")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/interpret")
def interpret(
    customer: CustomerInput, top_n: int = 5, _model=Depends(get_model)
):
    """Return the top SHAP features for one customer."""
    try:
        X = preprocess(customer.model_dump())
        result = predict_one(customer.model_dump())
        shap_values = explain_row(get_explainer(model), model, X)
        shap_dict = dict(zip(shap_values.feature_names, shap_values.values[0]))
        top_features = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        )
        readable = build_readable_explanation(
            shap_values, list(shap_values.feature_names), X
        )[:top_n]
        return {
            "churn_label": result.churn_label,
            "churn_probability": result.churn_probability,
            "risk_level": result.risk_level,
            "top_features": {k: round(float(v), 4) for k, v in top_features.items()},
            "summary": build_summary(readable),
            "explanation": readable,
        }
    except Exception:
        logger.exception("Interpret error")
        raise HTTPException(status_code=500, detail="Internal prediction error")
