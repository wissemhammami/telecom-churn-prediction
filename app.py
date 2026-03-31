# app.py

import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

from src.features.feature_engineering import appliquer_feature_engineering
from src.serving.utils import niveau_risque

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="centered",
)

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "models" / "xgb_churn_model.pkl"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessor_pipeline.pkl"


@st.cache_resource
def load_artifacts():
    model    = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


try:
    model, pipeline = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Failed to load model artifacts: {e}")


# --------------------------------------------------
# Preprocessing helper
# --------------------------------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = appliquer_feature_engineering(df)
    X = pipeline.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Telecom Churn Prediction")
st.markdown("Predict customer churn risk using a trained XGBoost model.")
st.divider()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "SHAP Explanation"])


# ══════════════════════════════════════════════════
# TAB 1 — Single prediction
# ══════════════════════════════════════════════════
with tab1:
    st.subheader("Customer Profile")

    with st.form("single_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure          = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment         = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        with col2:
            gender        = st.selectbox("Gender", ["Female", "Male"])
            senior        = st.selectbox("Senior Citizen", [0, 1])
            partner       = st.selectbox("Partner", ["Yes", "No"])
            dependents    = st.selectbox("Dependents", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            paperless     = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown("**Optional Services**")
        col3, col4, col5 = st.columns(3)
        with col3:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup   = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
        with col4:
            device_prot  = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        with col5:
            streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted and artifacts_loaded:
        input_data = pd.DataFrame([{
            "tenure":           tenure,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
            "gender":           gender,
            "SeniorCitizen":    senior,
            "Partner":          partner,
            "Dependents":       dependents,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
        }])

        try:
            X      = preprocess(input_data)
            prob   = float(model.predict_proba(X)[:, 1][0])
            label  = int(prob >= 0.5)
            risque = niveau_risque(prob)

            st.divider()
            st.subheader("Results")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Churn Probability", f"{prob:.1%}")
            with col_b:
                st.metric("Prediction", "Churn" if label == 1 else "No Churn")
            with col_c:
                if risque == "Faible":
                    st.success("Low Risk")
                elif risque == "Moyen":
                    st.warning("Medium Risk")
                else:
                    st.error("High Risk")

            st.progress(prob)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════
# TAB 2 — Batch prediction
# ══════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV file with the same columns as the training dataset.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file and artifacts_loaded:
        df_batch = pd.read_csv(uploaded_file)
        st.markdown(f"**{len(df_batch)} customers loaded.**")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("Run Batch Prediction", use_container_width=True):
            try:
                X_batch = preprocess(df_batch.copy())
                probs   = model.predict_proba(X_batch)[:, 1]
                labels  = (probs >= 0.5).astype(int)

                df_results = df_batch.copy()
                df_results["Churn_Predicted"]   = labels
                df_results["Churn_Probability"] = probs.round(4)
                df_results["Risk_Level"]        = [niveau_risque(p) for p in probs]

                total       = len(labels)
                nb_churners = labels.sum()

                st.divider()
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Customers", total)
                with col_b:
                    st.metric("Predicted Churners", int(nb_churners))
                with col_c:
                    st.metric("Churn Rate", f"{nb_churners / total:.1%}")

                st.dataframe(
                    df_results[["Churn_Predicted", "Churn_Probability", "Risk_Level"]],
                    use_container_width=True
                )

                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


# ══════════════════════════════════════════════════
# TAB 3 — SHAP explanation
# ══════════════════════════════════════════════════
with tab3:
    st.subheader("SHAP Feature Explanation")
    st.markdown("Explain the prediction for a single customer.")

    use_example = st.checkbox("Use example customer", value=True)

    if use_example:
        df_shap = pd.DataFrame([{
            "tenure": 5, "MonthlyCharges": 85.6, "TotalCharges": 428.0,
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }])
    else:
        shap_file = st.file_uploader("Upload single-row CSV", type="csv", key="shap_upload")
        df_shap   = pd.read_csv(shap_file) if shap_file else None

    if df_shap is not None and artifacts_loaded:
        if st.button("Explain Prediction", use_container_width=True):
            try:
                X_shap        = preprocess(df_shap.copy())
                prob          = float(model.predict_proba(X_shap)[:, 1][0])
                label         = int(prob >= 0.5)
                feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)

                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Churn Probability", f"{prob:.1%}")
                with col_b:
                    st.metric("Prediction", "Churn" if label == 1 else "No Churn")

                st.markdown("**Top features driving this prediction:**")

                indices = pd.Series(shap_values[0]).abs().sort_values(ascending=False).index
                shap_df = pd.DataFrame({
                    "Feature":    [feature_names[i] for i in indices[:10]],
                    "SHAP Value": [shap_values[0][i] for i in indices[:10]]
                }).reset_index(drop=True)

                st.dataframe(shap_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=X_shap[0],
                        feature_names=list(feature_names),
                    ),
                    max_display=10,
                    show=False,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close()

            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")