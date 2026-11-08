"""Test the main inference and feature-engineering behavior."""

import numpy as np
import pandas as pd
import pytest
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
        from src.serving.utils import get_risk_level

        assert get_risk_level(0.85) == "High"

    def test_risque_moyen(self):
        """Probabilities from 0.4 to below 0.7 are medium risk."""
        from src.serving.utils import get_risk_level

        assert get_risk_level(0.55) == "Medium"

    def test_risque_faible(self):
        """Probabilities below 0.4 are low risk."""
        from src.serving.utils import get_risk_level

        assert get_risk_level(0.2) == "Low"

    def test_seuil_exact_eleve(self):
        """A probability of exactly 0.7 is high risk."""
        from src.serving.utils import get_risk_level

        assert get_risk_level(0.7) == "High"

    def test_seuil_exact_moyen(self):
        """A probability of exactly 0.4 is medium risk."""
        from src.serving.utils import get_risk_level

        assert get_risk_level(0.4) == "Medium"


class TestPredireProba:
    """Test probability-to-label conversion."""

    def test_labels_binaires(self):
        """Returned labels are binary."""
        from src.serving.utils import predict_proba_with_threshold

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        X = np.zeros((2, 10))
        labels, probs = predict_proba_with_threshold(model, X)

        assert set(labels).issubset({0, 1})

    def test_probabilites_entre_0_et_1(self):
        """Returned probabilities are bounded between zero and one."""
        from src.serving.utils import predict_proba_with_threshold

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.4, 0.6], [0.9, 0.1]])

        X = np.zeros((2, 10))
        labels, probs = predict_proba_with_threshold(model, X)

        assert all(0 <= p <= 1 for p in probs)

    def test_seuil_05(self):
        """A probability above the default threshold gets label one."""
        from src.serving.utils import predict_proba_with_threshold

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        X = np.zeros((1, 10))
        labels, probs = predict_proba_with_threshold(model, X, threshold=0.5)

        assert labels[0] == 1

    def test_seuil_personnalise(self):
        """A probability below a custom threshold gets label zero."""
        from src.serving.utils import predict_proba_with_threshold

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        X = np.zeros((1, 10))
        labels, probs = predict_proba_with_threshold(model, X, threshold=0.8)

        assert labels[0] == 0


class TestPreprocessing:
    """Test shared raw-data preprocessing."""

    def test_drops_unwanted_columns(self):
        """customerID and Churn are removed before transformation."""
        from src.features.preprocessing import prepare_features

        prepared = prepare_features(
            pd.DataFrame([{**CLIENT_EXEMPLE, "customerID": "abc", "Churn": "Yes"}])
        )

        assert "customerID" not in prepared.columns
        assert "Churn" not in prepared.columns

    def test_totalcharges_converti(self):
        """TotalCharges is converted to a numeric dtype."""
        from src.features.preprocessing import prepare_features

        data = CLIENT_EXEMPLE.copy()
        data["TotalCharges"] = "1027.2"  # Raw CSV values may be strings.

        prepared = prepare_features(pd.DataFrame([data]))

        assert pd.api.types.is_float_dtype(prepared["TotalCharges"])

    def test_blank_totalcharges_becomes_nan(self):
        """Blank TotalCharges values become NaN."""
        from src.features.preprocessing import prepare_features

        data = CLIENT_EXEMPLE.copy()
        data["TotalCharges"] = " "

        prepared = prepare_features(pd.DataFrame([data]))

        assert pd.isna(prepared["TotalCharges"].iloc[0])


class TestFeatureEngineering:
    """Test derived feature creation."""

    def test_charges_moyennes_tenure_normal(self):
        """Average charges equal total charges divided by tenure."""
        from src.features.feature_engineering import add_average_charge

        df = pd.DataFrame([{"TotalCharges": 1000.0, "tenure": 10}])
        df = add_average_charge(df)

        assert df["ChargesMoyennes"].iloc[0] == 100.0

    def test_charges_moyennes_tenure_zero(self):
        """Average charges are zero for zero-tenure customers."""
        from src.features.feature_engineering import add_average_charge

        df = pd.DataFrame([{"TotalCharges": 0.0, "tenure": 0}])
        df = add_average_charge(df)

        assert df["ChargesMoyennes"].iloc[0] == 0.0

    def test_charges_moyennes_tenure_zero_nan(self):
        """Average charges are zero for zero-tenure customers with NaN totals."""
        from src.features.feature_engineering import add_average_charge

        df = pd.DataFrame([{"TotalCharges": np.nan, "tenure": 0}])
        df = add_average_charge(df)

        assert df["ChargesMoyennes"].iloc[0] == 0.0

    def test_segment_tenure_nouveau(self):
        """Tenure up to 12 months is the new-customer segment."""
        from src.features.feature_engineering import add_tenure_segment

        df = pd.DataFrame([{"tenure": 6}])
        df = add_tenure_segment(df)

        assert df["SegmentTenure"].iloc[0] == "Nouveau"

    def test_segment_tenure_fidele(self):
        """Tenure above 36 months is the loyal-customer segment."""
        from src.features.feature_engineering import add_tenure_segment

        df = pd.DataFrame([{"tenure": 50}])
        df = add_tenure_segment(df)

        assert df["SegmentTenure"].iloc[0] == "Fidele"

    def test_tenure_segment_boundaries(self):
        """Tenure boundaries map to the expected segments."""
        from src.features.feature_engineering import add_tenure_segment

        df = add_tenure_segment(pd.DataFrame({"tenure": [12, 13, 36, 37]}))

        assert df["SegmentTenure"].tolist() == [
            "Nouveau",
            "Intermediaire",
            "Intermediaire",
            "Fidele",
        ]

    def test_nb_services(self):
        """Three Yes service values produce a count of three."""
        from src.features.feature_engineering import add_num_services

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
        df = add_num_services(df)

        assert df["NbServices"].iloc[0] == 3

    def test_no_internet_has_no_services(self):
        """Customers without internet have no optional services counted."""
        from src.features.feature_engineering import (
            add_no_internet,
            add_num_services,
        )

        df = pd.DataFrame(
            [
                {
                    "InternetService": "No",
                    "OnlineSecurity": "No internet service",
                    "OnlineBackup": "No internet service",
                    "DeviceProtection": "No internet service",
                    "TechSupport": "No internet service",
                    "StreamingTV": "No internet service",
                    "StreamingMovies": "No internet service",
                }
            ]
        )
        df = add_num_services(df)
        df = add_no_internet(df)

        assert df["SansInternet"].iloc[0] == 1
        assert df["NbServices"].iloc[0] == 0

    @pytest.mark.skipif(
        not pd.io.common.file_exists("data/raw/churn.csv"),
        reason="Raw dataset is not available",
    )
    def test_vectorized_features_match_rowwise_reference(self):
        """Vectorized feature calculations match the former row-wise behavior."""
        from src.features.feature_engineering import (
            add_average_charge,
            add_num_services,
        )

        source = pd.read_csv("data/raw/churn.csv").head(500)
        source["TotalCharges"] = pd.to_numeric(source["TotalCharges"], errors="coerce")
        services = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        expected_average = source.apply(
            lambda row: (
                row["TotalCharges"] / row["tenure"]
                if row["tenure"] > 0
                else 0
            ),
            axis=1,
        )
        expected_services = source[services].apply(
            lambda row: (row == "Yes").sum(), axis=1
        )
        actual = add_average_charge(source.copy())
        actual = add_num_services(actual)

        assert np.allclose(
            actual["ChargesMoyennes"],
            expected_average,
            equal_nan=True,
        )
        assert actual["NbServices"].equals(expected_services)

    def test_contrat_long_two_year(self):
        """A two-year contract is marked as long term."""
        from src.features.feature_engineering import add_long_contract

        df = pd.DataFrame([{"Contract": "Two year"}])
        df = add_long_contract(df)

        assert df["ContratLong"].iloc[0] == 1

    def test_contrat_long_month_to_month(self):
        """A month-to-month contract is not marked as long term."""
        from src.features.feature_engineering import add_long_contract

        df = pd.DataFrame([{"Contract": "Month-to-month"}])
        df = add_long_contract(df)

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

    def test_interpret_values_are_not_all_zero(self):
        from src.serving.main import interpret
        from src.serving.schemas import CustomerInput

        explanation = interpret(CustomerInput(**CLIENT_EXEMPLE))

        assert any(abs(value) > 1e-6 for value in explanation["top_features"].values())

    def test_shap_additivity(self):
        from src.serving.explain import explain_row, get_explainer
        from src.features.preprocessing import prepare_features
        from src.serving.main import model, predict_one
        from src.serving.schemas import CustomerInput

        customer = CustomerInput(**CLIENT_EXEMPLE)
        prepared = prepare_features(pd.DataFrame([customer.model_dump()]))
        explanation = explain_row(get_explainer(model), model, prepared)
        probability = predict_one(customer.model_dump()).churn_probability

        assert abs(
            explanation.base_values[0]
            + explanation.values[0].sum()
            - probability
        ) < 0.02

    def test_interpret_differs_between_customers(self):
        from src.serving.explain import explain_row, get_explainer
        from src.features.preprocessing import prepare_features
        from src.serving.main import model, predict_one
        from src.serving.schemas import CustomerInput

        high_risk = CLIENT_EXEMPLE.copy()
        high_risk.update(
            {
                "tenure": 2,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
            }
        )
        low_risk = CLIENT_EXEMPLE.copy()
        low_risk.update(
            {
                "tenure": 60,
                "Contract": "Two year",
                "InternetService": "DSL",
                "PaymentMethod": "Credit card (automatic)",
            }
        )

        high_customer = CustomerInput(**high_risk)
        low_customer = CustomerInput(**low_risk)
        high_prepared = prepare_features(pd.DataFrame([high_risk]))
        low_prepared = prepare_features(pd.DataFrame([low_risk]))
        explainer = get_explainer(model)
        high_explanation = explain_row(explainer, model, high_prepared)
        low_explanation = explain_row(explainer, model, low_prepared)
        high_features = {
            name: round(float(value), 4)
            for name, value in zip(
                high_explanation.feature_names, high_explanation.values[0]
            )
        }
        low_features = {
            name: round(float(value), 4)
            for name, value in zip(
                low_explanation.feature_names, low_explanation.values[0]
            )
        }

        assert high_features != low_features
        assert high_explanation.values[0].sum() > low_explanation.values[0].sum()
        assert predict_one(high_customer.model_dump()).churn_probability > predict_one(
            low_customer.model_dump()
        ).churn_probability

    def test_readable_explanation_groups_and_labels_features(self):
        from src.features.preprocessing import prepare_features
        from src.serving.explain import (
            build_readable_explanation,
            build_summary,
            explain_row,
            get_explainer,
        )
        from src.serving.main import model

        prepared = prepare_features(pd.DataFrame([CLIENT_EXEMPLE]))
        explanation = explain_row(get_explainer(model), model, prepared)
        readable = build_readable_explanation(
            explanation, list(explanation.feature_names), prepared
        )

        assert {item["feature"] for item in readable} == set(prepared.columns)
        assert all(
            set(item) == {"feature", "label", "value", "shap", "direction"}
            for item in readable
        )
        assert any(item["feature"] == "Contract" for item in readable)
        assert any(item["feature"] == "tenure" for item in readable)
        assert len(build_summary(readable)) <= 250

    def test_waterfall_uses_readable_feature_names(self):
        from src.features.preprocessing import prepare_features
        from src.serving.explain import (
            build_readable_explanation,
            build_waterfall_explanation,
            explain_row,
            get_explainer,
        )
        from src.serving.main import model

        prepared = prepare_features(pd.DataFrame([CLIENT_EXEMPLE]))
        explanation = explain_row(get_explainer(model), model, prepared)
        readable = build_readable_explanation(
            explanation, list(explanation.feature_names), prepared
        )[:3]
        waterfall = build_waterfall_explanation(readable, explanation.base_values[0])

        assert all(" = " in name for name in waterfall.feature_names)

    def test_shared_preparation_removes_unwanted_columns(self):
        from src.features.preprocessing import prepare_features

        prepared = prepare_features(
            pd.DataFrame([{**CLIENT_EXEMPLE, "customerID": "abc", "Churn": "Yes"}])
        )
        assert "customerID" not in prepared.columns
        assert "Churn" not in prepared.columns