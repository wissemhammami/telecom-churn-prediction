# Telecom Churn Prediction

End-to-end machine learning project predicting customer churn for a telecom company.
Built with scikit-learn, XGBoost, FastAPI, and Streamlit.

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

## Model Selection

The training workflow compares Logistic Regression, Random Forest, and XGBoost
against a DummyClassifier baseline. Each candidate is tuned with
RandomizedSearchCV using the same 5-fold StratifiedKFold and ROC-AUC scoring.
The champion is selected from training CV results, then evaluated once on the
untouched test set. Final reporting includes ROC-AUC, PR-AUC, F1, precision,
recall, and a simple threshold analysis. The selected champion is deployed by
batch inference, FastAPI, and Streamlit.

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
│   ├── champion_model.pkl          # Selected fitted model pipeline
│   ├── model_metadata.json          # Champion and evaluation metrics
│   └── feature_columns.pkl         # Feature names
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── reports/
│   ├── model_comparison.csv         # CV comparison
│   └── threshold_analysis.csv       # Test threshold analysis
│
├── scripts/
│   └── check_data.py               # Data validation script
│
├── src/
│   ├── data/
│   │   └── __init__.py
│   ├── features/
│   │   ├── feature_engineering.py  # Feature engineering
│   │   └── preprocessing.py        # Shared raw-data preparation
│   ├── training/
│   │   └── train.py                # Multi-model CV tuning and champion selection
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

Requires Python 3.12 (Streamlit Cloud and local installs fail on Python 3.14 due to shap/numba/llvmlite incompatibility).

python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

**1 — Train the model**
```bash
python -m src.training.train
```

**2 — Generate SHAP reports**
```bash
python -m src.interpretability.interpretability
```

**3 — Launch the API**
```bash
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at `http://localhost:8000/docs`

**4 — Launch the Streamlit app**
```bash
python -m streamlit run app.py
```
App available at `http://localhost:8501`

**5 — Run tests**
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
  └── Train/Test split → test set held untouched
    └── Shared feature preparation
      └── 5-fold Stratified CV + RandomizedSearchCV
        └── Logistic Regression / Random Forest / XGBoost
          └── CV comparison → champion selection
            └── Untouched test evaluation + threshold analysis
              └── Champion packaging → API / Streamlit / batch / SHAP
```

### Final Results

The champion is selected by mean training CV ROC-AUC. The current generated
artifacts select Random Forest.

| Model | Mean CV ROC-AUC | Std |
|---|---:|---:|
| DummyClassifier | 0.5000 | 0.0000 |
| LogisticRegression | 0.8474 | 0.0113 |
| RandomForest | 0.8477 | 0.0100 |
| XGBoost | 0.8469 | 0.0104 |

The champion's one-time untouched-test results are ROC-AUC `0.8442`, PR-AUC
`0.6520`, F1 `0.6414`, precision `0.5392`, and recall `0.7914`, using threshold
`0.5`. Threshold results for `0.30`, `0.40`, `0.50`, `0.60`, and `0.70` are
saved in `reports/threshold_analysis.csv`.

SHAP uses the fitted champion estimator and explains the churn class. The
standalone report script and the FastAPI and Streamlit explanation paths work
with linear or tree-based champions.

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