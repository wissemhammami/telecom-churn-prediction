# Telecom Churn Prediction

End-to-end machine learning project predicting customer churn for a telecom company.
Built with XGBoost, FastAPI, and Streamlit.

## Live Demo

[Open the app](https://churn-wissem.streamlit.app)

---

## Problem Statement

Telecom companies lose revenue when customers cancel their subscriptions.
This project predicts which customers are likely to churn so the business can intervene early
and take targeted retention actions.

**Dataset** : [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Target** : Binary classification — Churn (1) vs No Churn (0)  
**Dataset size** : 7,043 customers — 21 features

---

## Results

| Metric    | Baseline | XGBoost |
|-----------|----------|---------|
| AUC       | 0.50     | 0.85    |
| F1        | 0.00     | 0.63    |
| Precision | 0.00     | 0.52    |
| Recall    | 0.00     | 0.81    |

---

## Project Structure
```
TELECOM-CHURN-PREDICTION/
│
├── data/
│   ├── raw/                        # Original dataset
│   ├── processed/                  # Preprocessed dataset
│   └── new/                        # New customers for inference
│
├── models/
│   ├── xgb_churn_model.pkl         # Trained XGBoost model
│   ├── preprocessor_pipeline.pkl   # Sklearn preprocessing pipeline
│   ├── scaler.pkl                  # Standard scaler
│   └── feature_columns.pkl         # Feature names
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── reports/                        # SHAP plots and feature importance
│
├── scripts/
│   └── check_data.py               # Data validation script
│
├── src/
│   ├── data/
│   │   └── preprocess.py           # Data cleaning and pipeline
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering
│   ├── training/
│   │   └── train.py                # XGBoost training with RandomizedSearchCV
│   ├── inference/
│   │   └── predict.py              # Batch inference
│   ├── interpretability/
│   │   └── interpretability.py     # SHAP visualizations
│   └── serving/
│       ├── main.py                 # FastAPI application
│       ├── schemas.py              # Pydantic schemas
│       ├── config.py               # Central configuration
│       └── utils.py                # Shared utilities
│
├── app.py                          # Streamlit frontend
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## Installation
```bash
git clone https://github.com/wissemhammami/telecom-churn-prediction.git
cd telecom-churn-prediction

python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

**1 — Preprocess data**
```bash
python -m src.data.preprocess
```

**2 — Train the model**
```bash
python -m src.training.train
```

**3 — Generate SHAP reports**
```bash
python -m src.interpretability.interpretability
```

**4 — Launch the API**
```bash
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at `http://localhost:8000/docs`

**5 — Launch the Streamlit app**
```bash
python -m streamlit run app.py
```
App available at `http://localhost:8501`

**6 — Run tests**
```bash
pytest src/tests/test_predict.py -v
```

---

## Docker
```bash
# Build
docker build -t churn-app .

# Run
docker run -p 8000:8000 -p 8501:8501 churn-app
```

---

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Check API status                     |
| POST   | `/predict`        | Predict churn for a single customer  |
| POST   | `/predict/batch`  | Predict churn for multiple customers |
| POST   | `/interpret`      | SHAP explanation for a customer      |

**Example request**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 5,
    "MonthlyCharges": 85.6,
    "TotalCharges": 428.0,
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
    "PaymentMethod": "Electronic check"
  }'
```

**Example response**
```json
{
  "churn_label": 1,
  "churn_probability": 0.87,
  "niveau_risque": "Élevé"
}
```

---

## ML Pipeline
```
Raw Data
  └── Preprocessing         → TotalCharges fix, target encoding
        └── Feature Engineering  → ChargesMoyennes, NbServices,
                                    SansInternet, ContratLong, SegmentTenure
              └── Pipeline        → StandardScaler + OneHotEncoder
                    └── XGBoost   → RandomizedSearchCV (15 iterations, 5-fold CV)
                          └── SHAP → Global + individual explanations
```

---

## Key Insights from EDA

- Contract type is the strongest predictor — month-to-month customers churn at ~43%
- Tenure is negatively correlated with churn — new customers churn more
- Monthly charges above $70 increase churn risk significantly
- Fiber optic customers churn more than DSL customers

---

## Tech Stack

| Category          | Tools                          |
|-------------------|--------------------------------|
| Language          | Python 3.13                    |
| ML                | XGBoost, Scikit-learn          |
| Explainability    | SHAP                           |
| API               | FastAPI, Pydantic, Uvicorn     |
| Frontend          | Streamlit                      |
| Containerization  | Docker                         |
| Visualization     | Matplotlib, Seaborn            |

---

## Author

**Wissem Hammami**  
Machine Learning Engineer | Data Science | ESSAI, University of Carthage  
[GitHub](https://github.com/wissemhammami)