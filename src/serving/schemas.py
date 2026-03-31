# src/serving/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class CustomerInput(BaseModel):
    tenure: float = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = Field(None, ge=0, le=1)
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={"example": {
            "tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 845.5,
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }}
    )


class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1)


class PredictionOutput(BaseModel):
    churn_label: int = Field(..., description="0 = no churn, 1 = churn")
    churn_probability: float = Field(..., description="Churn probability (0-1)")
    niveau_risque: str = Field(..., description="Faible / Moyen / Élevé")


class BatchPredictionOutput(BaseModel):
    results: List[PredictionOutput]
    total: int
    nb_churners: int
    taux_churn: float