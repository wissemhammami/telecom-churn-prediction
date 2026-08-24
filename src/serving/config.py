# src/serving/config.py
"""
Configuration centrale du projet — Telecom Churn Prediction
Tous les chemins, constantes ML et paramètres API sont définis ici.
Importer ce fichier dans les autres modules au lieu de redéfinir les chemins.
"""

import os

# -------------------------
# Répertoire racine du projet
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# -------------------------
# Chemins — Data
# -------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "churn.csv")
NEW_CUSTOMERS_PATH = os.path.join(DATA_DIR, "new", "new_customers.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "new", "new_customers_predictions.csv")

# -------------------------
# Chemins — Modèles
# -------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "champion_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# -------------------------
# Chemins — Reports
# -------------------------
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# -------------------------
# Constantes ML
# -------------------------
TARGET_COL = "Churn"
SEUIL_CHURN = 0.5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colonnes numériques et catégorielles — identiques au preprocessing
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# -------------------------
# Paramètres API
# -------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Telecom Churn Prediction API"
API_VERSION = "1.0.0"

# -------------------------
# Logging
# -------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
