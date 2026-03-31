# tests/test_predict.py
"""
Tests unitaires — Telecom Churn Prediction
------------------------------------------
Teste les fonctions principales du pipeline d'inférence.

Usage :
    pytest tests/test_predict.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# -------------------------
# Données de test
# -------------------------

# Client type pour les tests
CLIENT_EXEMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
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
    "MonthlyCharges": 85.6,
    "TotalCharges": 1027.2
}

# DataFrame batch de test
DF_BATCH = pd.DataFrame([CLIENT_EXEMPLE, CLIENT_EXEMPLE])


# -------------------------
# Tests — niveau_risque
# -------------------------

class TestNiveauRisque:
    """Tests pour la fonction niveau_risque."""

    def test_risque_eleve(self):
        """Probabilité >= 0.7 → Élevé"""
        from src.serving.utils import niveau_risque
        assert niveau_risque(0.85) == "Élevé"

    def test_risque_moyen(self):
        """Probabilité entre 0.4 et 0.7 → Moyen"""
        from src.serving.utils import niveau_risque
        assert niveau_risque(0.55) == "Moyen"

    def test_risque_faible(self):
        """Probabilité < 0.4 → Faible"""
        from src.serving.utils import niveau_risque
        assert niveau_risque(0.2) == "Faible"

    def test_seuil_exact_eleve(self):
        """Probabilité exactement à 0.7 → Élevé"""
        from src.serving.utils import niveau_risque
        assert niveau_risque(0.7) == "Élevé"

    def test_seuil_exact_moyen(self):
        """Probabilité exactement à 0.4 → Moyen"""
        from src.serving.utils import niveau_risque
        assert niveau_risque(0.4) == "Moyen"


# -------------------------
# Tests — predire_proba
# -------------------------

class TestPredireProba:
    """Tests pour la fonction predire_proba."""

    def test_labels_binaires(self):
        """Les labels retournés doivent être 0 ou 1 uniquement."""
        from src.serving.utils import predire_proba

        # Mock modèle
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        X = np.zeros((2, 10))
        labels, probs = predire_proba(model, X)

        assert set(labels).issubset({0, 1})

    def test_probabilites_entre_0_et_1(self):
        """Les probabilités doivent être entre 0 et 1."""
        from src.serving.utils import predire_proba

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.4, 0.6], [0.9, 0.1]])

        X = np.zeros((2, 10))
        labels, probs = predire_proba(model, X)

        assert all(0 <= p <= 1 for p in probs)

    def test_seuil_05(self):
        """Avec seuil=0.5, prob=0.7 → label=1."""
        from src.serving.utils import predire_proba

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        X = np.zeros((1, 10))
        labels, probs = predire_proba(model, X, seuil=0.5)

        assert labels[0] == 1

    def test_seuil_personnalise(self):
        """Avec seuil=0.8, prob=0.7 → label=0."""
        from src.serving.utils import predire_proba

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        X = np.zeros((1, 10))
        labels, probs = predire_proba(model, X, seuil=0.8)

        assert labels[0] == 0


# -------------------------
# Tests — preprocesser_client
# -------------------------

class TestPreprocesserClient:
    """Tests pour la fonction preprocesser_client."""

    def test_supprime_customerID(self):
        """customerID doit être supprimé avant le pipeline."""
        from src.serving.utils import preprocesser_client

        # Mock pipeline
        pipeline = MagicMock()
        pipeline.transform.return_value = np.zeros((1, 20))

        data = CLIENT_EXEMPLE.copy()
        data["customerID"] = "7590-VHVEG"

        preprocesser_client(data, pipeline)

        # Vérifier que transform a été appelé sans customerID
        appel_df = pipeline.transform.call_args[0][0]
        assert "customerID" not in appel_df.columns

    def test_supprime_churn(self):
        """Colonne Churn doit être supprimée si présente."""
        from src.serving.utils import preprocesser_client

        pipeline = MagicMock()
        pipeline.transform.return_value = np.zeros((1, 20))

        data = CLIENT_EXEMPLE.copy()
        data["Churn"] = 1

        preprocesser_client(data, pipeline)

        appel_df = pipeline.transform.call_args[0][0]
        assert "Churn" not in appel_df.columns

    def test_totalcharges_converti(self):
        """TotalCharges doit être converti en float."""
        from src.serving.utils import preprocesser_client

        pipeline = MagicMock()
        pipeline.transform.return_value = np.zeros((1, 20))

        data = CLIENT_EXEMPLE.copy()
        data["TotalCharges"] = "1027.2"  # string comme dans le CSV brut

        preprocesser_client(data, pipeline)

        appel_df = pipeline.transform.call_args[0][0]
        assert pd.api.types.is_float_dtype(appel_df["TotalCharges"])


# -------------------------
# Tests — feature_engineering
# -------------------------

class TestFeatureEngineering:
    """Tests pour les fonctions de feature engineering."""

    def test_charges_moyennes_tenure_normal(self):
        """ChargesMoyennes = TotalCharges / tenure pour tenure > 0."""
        from src.features.feature_engineering import ajouter_charge_moyenne

        df = pd.DataFrame([{"TotalCharges": 1000.0, "tenure": 10}])
        df = ajouter_charge_moyenne(df)

        assert df["ChargesMoyennes"].iloc[0] == 100.0

    def test_charges_moyennes_tenure_zero(self):
        """ChargesMoyennes = 0 si tenure = 0 (nouveau client)."""
        from src.features.feature_engineering import ajouter_charge_moyenne

        df = pd.DataFrame([{"TotalCharges": 0.0, "tenure": 0}])
        df = ajouter_charge_moyenne(df)

        assert df["ChargesMoyennes"].iloc[0] == 0.0

    def test_segment_tenure_nouveau(self):
        """tenure <= 12 → Nouveau."""
        from src.features.feature_engineering import ajouter_segment_tenure

        df = pd.DataFrame([{"tenure": 6}])
        df = ajouter_segment_tenure(df)

        assert df["SegmentTenure"].iloc[0] == "Nouveau"

    def test_segment_tenure_fidele(self):
        """tenure > 36 → Fidele."""
        from src.features.feature_engineering import ajouter_segment_tenure

        df = pd.DataFrame([{"tenure": 50}])
        df = ajouter_segment_tenure(df)

        assert df["SegmentTenure"].iloc[0] == "Fidele"

    def test_nb_services(self):
        """Client avec 3 services Yes → NbServices = 3."""
        from src.features.feature_engineering import ajouter_nb_services

        df = pd.DataFrame([{
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No"
        }])
        df = ajouter_nb_services(df)

        assert df["NbServices"].iloc[0] == 3

    def test_contrat_long_two_year(self):
        """Contrat Two year → ContratLong = 1."""
        from src.features.feature_engineering import ajouter_contrat_long

        df = pd.DataFrame([{"Contract": "Two year"}])
        df = ajouter_contrat_long(df)

        assert df["ContratLong"].iloc[0] == 1

    def test_contrat_long_month_to_month(self):
        """Contrat Month-to-month → ContratLong = 0."""
        from src.features.feature_engineering import ajouter_contrat_long

        df = pd.DataFrame([{"Contract": "Month-to-month"}])
        df = ajouter_contrat_long(df)

        assert df["ContratLong"].iloc[0] == 0
