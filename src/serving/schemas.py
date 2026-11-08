from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerInput(BaseModel):
    """Input schema for one customer prediction."""

    tenure: float = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.5,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        }
    )


class BatchCustomerInput(BaseModel):
    """Input schema for a non-empty batch of customers."""

    customers: List[CustomerInput] = Field(..., min_length=1, max_length=1000)


class PredictionOutput(BaseModel):
    """Public response schema for one prediction."""

    churn_label: int = Field(..., description="0 = no churn, 1 = churn")
    churn_probability: float = Field(..., description="Churn probability (0-1)")
    risk_level: Literal["Low", "Medium", "High"]


class BatchPredictionOutput(BaseModel):
    """Public response schema for batch predictions."""

    results: List[PredictionOutput]
    total: int
    num_churners: int
    churn_rate_percent: float
