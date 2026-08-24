"""Test the main inference and feature-engineering behavior."""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock


# Test data.

# Representative customer record.
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
    "TotalCharges": 1027.2,
}

# Sample batch.
DF_BATCH = pd.DataFrame([CLIENT_EXEMPLE, CLIENT_EXEMPLE])


class TestNiveauRisque:
    """Test risk-level classification."""

    def test_risque_eleve(self):
        """Probabilities at least 0.7 are high risk."""
        from src.serving.utils import niveau_risque

        assert niveau_risque(0.85) == "Élevé"

    def test_risque_moyen(self):
        """Probabilities from 0.4 to below 0.7 are medium risk."""
        from src.serving.utils import niveau_risque

        assert niveau_risque(0.55) == "Moyen"

    def test_risque_faible(self):
        """Probabilities below 0.4 are low risk."""
        from src.serving.utils import niveau_risque

        assert niveau_risque(0.2) == "Faible"

    def test_seuil_exact_eleve(self):
        """A probability of exactly 0.7 is high risk."""
        from src.serving.utils import niveau_risque

        assert niveau_risque(0.7) == "Élevé"

    def test_seuil_exact_moyen(self):
        """A probability of exactly 0.4 is medium risk."""
        from src.serving.utils import niveau_risque

        assert niveau_risque(0.4) == "Moyen"


class TestPredireProba:
    """Test probability-to-label conversion."""

    def test_labels_binaires(self):
        """Returned labels are binary."""
        from src.serving.utils import predire_proba

        # Use a minimal model double.
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        X = np.zeros((2, 10))
        labels, probs = predire_proba(model, X)

        assert set(labels).issubset({0, 1})

    def test_probabilites_entre_0_et_1(self):
        """Returned probabilities are bounded between zero and one."""
        from src.serving.utils import predire_proba

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.4, 0.6], [0.9, 0.1]])

        X = np.zeros((2, 10))
        labels, probs = predire_proba(model, X)

        assert all(0 <= p <= 1 for p in probs)

    def test_seuil_05(self):
        """A probability above the default threshold gets label one."""
        from src.serving.utils import predire_proba

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        X = np.zeros((1, 10))
        labels, probs = predire_proba(model, X, seuil=0.5)

        assert labels[0] == 1

    def test_seuil_personnalise(self):
        """A probability below a custom threshold gets label zero."""
        from src.serving.utils import predire_proba

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        X = np.zeros((1, 10))
        labels, probs = predire_proba(model, X, seuil=0.8)

        assert labels[0] == 0


class TestPreprocesserClient:
    """Test single-customer preprocessing."""

    def test_supprime_customerID(self):
        """customerID is removed before transformation."""
        from src.serving.utils import preprocesser_client

        # Use a minimal pipeline double.
        pipeline = MagicMock()
        pipeline.transform.return_value = np.zeros((1, 20))

        data = CLIENT_EXEMPLE.copy()
        data["customerID"] = "7590-VHVEG"

        preprocesser_client(data, pipeline)

        # Verify the transformer did not receive customerID.
        appel_df = pipeline.transform.call_args[0][0]
        assert "customerID" not in appel_df.columns

    def test_supprime_churn(self):
        """Churn is removed when present."""
        from src.serving.utils import preprocesser_client

        pipeline = MagicMock()
        pipeline.transform.return_value = np.zeros((1, 20))

        data = CLIENT_EXEMPLE.copy()
        data["Churn"] = 1

        preprocesser_client(data, pipeline)

        appel_df = pipeline.transform.call_args[0][0]
        assert "Churn" not in appel_df.columns

    def test_totalcharges_converti(self):
        """TotalCharges is converted to a numeric dtype."""
        from src.serving.utils import preprocesser_client

        pipeline = MagicMock()
        pipeline.transform.return_value = np.zeros((1, 20))

        data = CLIENT_EXEMPLE.copy()
        data["TotalCharges"] = "1027.2"  # Raw CSV values may be strings.

        preprocesser_client(data, pipeline)

        appel_df = pipeline.transform.call_args[0][0]
        assert pd.api.types.is_float_dtype(appel_df["TotalCharges"])


class TestFeatureEngineering:
    """Test derived feature creation."""

    def test_charges_moyennes_tenure_normal(self):
        """Average charges equal total charges divided by tenure."""
        from src.features.feature_engineering import ajouter_charge_moyenne

        df = pd.DataFrame([{"TotalCharges": 1000.0, "tenure": 10}])
        df = ajouter_charge_moyenne(df)

        assert df["ChargesMoyennes"].iloc[0] == 100.0

    def test_charges_moyennes_tenure_zero(self):
        """Average charges are zero for zero-tenure customers."""
        from src.features.feature_engineering import ajouter_charge_moyenne

        df = pd.DataFrame([{"TotalCharges": 0.0, "tenure": 0}])
        df = ajouter_charge_moyenne(df)

        assert df["ChargesMoyennes"].iloc[0] == 0.0

    def test_segment_tenure_nouveau(self):
        """Tenure up to 12 months is the new-customer segment."""
        from src.features.feature_engineering import ajouter_segment_tenure

        df = pd.DataFrame([{"tenure": 6}])
        df = ajouter_segment_tenure(df)

        assert df["SegmentTenure"].iloc[0] == "Nouveau"

    def test_segment_tenure_fidele(self):
        """Tenure above 36 months is the loyal-customer segment."""
        from src.features.feature_engineering import ajouter_segment_tenure

        df = pd.DataFrame([{"tenure": 50}])
        df = ajouter_segment_tenure(df)

        assert df["SegmentTenure"].iloc[0] == "Fidele"

    def test_nb_services(self):
        """Three Yes service values produce a count of three."""
        from src.features.feature_engineering import ajouter_nb_services

        df = pd.DataFrame(
            [
                {
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                }
            ]
        )
        df = ajouter_nb_services(df)

        assert df["NbServices"].iloc[0] == 3

    def test_contrat_long_two_year(self):
        """A two-year contract is marked as long term."""
        from src.features.feature_engineering import ajouter_contrat_long

        df = pd.DataFrame([{"Contract": "Two year"}])
        df = ajouter_contrat_long(df)

        assert df["ContratLong"].iloc[0] == 1

    def test_contrat_long_month_to_month(self):
        """A month-to-month contract is not marked as long term."""
        from src.features.feature_engineering import ajouter_contrat_long

        df = pd.DataFrame([{"Contract": "Month-to-month"}])
        df = ajouter_contrat_long(df)

        assert df["ContratLong"].iloc[0] == 0


class TestChampionAPI:
    """Smoke tests for the deployed champion artifact and API handlers."""

    def test_health_reports_loaded_artifacts(self):
        from src.serving.main import health

        result = health()
        assert result["model_loaded"] is True
        assert result["pipeline_loaded"] is True

    def test_prediction_and_batch_contracts(self):
        from src.serving.main import interpret, predict, predict_batch
        from src.serving.schemas import BatchCustomerInput, CustomerInput

        customer = CustomerInput(**CLIENT_EXEMPLE)
        single = predict(customer)
        batch = predict_batch(BatchCustomerInput(customers=[customer, customer]))

        assert 0 <= single.churn_probability <= 1
        assert single.churn_label in [0, 1]
        assert batch.total == 2
        assert len(batch.results) == 2
        explanation = interpret(customer)
        assert explanation["churn_probability"] == single.churn_probability
        assert explanation["top_features"]

    def test_shared_preparation_removes_unwanted_columns(self):
        from src.features.preprocessing import preparer_features

        prepared = preparer_features(
            pd.DataFrame([{**CLIENT_EXEMPLE, "customerID": "abc", "Churn": "Yes"}])
        )
        assert "customerID" not in prepared.columns
        assert "Churn" not in prepared.columns
