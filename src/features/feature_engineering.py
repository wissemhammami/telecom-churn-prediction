# src/features/feature_engineering.py
"""
Feature Engineering — Telecom Churn Prediction
-----------------------------------------------
Crée de nouvelles features à partir des données brutes
pour améliorer les performances du modèle.

Usage :
    Appelé depuis preprocess.py avant le pipeline sklearn.
"""

import pandas as pd


# -------------------------
# Feature principale : charges moyennes par mois
# -------------------------

def ajouter_charge_moyenne(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la charge mensuelle moyenne réelle.
    Evite la division par zéro pour les nouveaux clients (tenure=0).

    Logique : TotalCharges / tenure
    """
    df["ChargesMoyennes"] = df.apply(
        lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else 0,
        axis=1
    )
    return df


# -------------------------
# Feature : client fidèle ou non
# -------------------------

def ajouter_segment_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmente les clients par ancienneté (tenure).

    Segments :
        - Nouveau     : tenure <= 12 mois
        - Intermédiaire : 12 < tenure <= 36 mois
        - Fidèle      : tenure > 36 mois
    """
    def segmenter(tenure):
        if tenure <= 12:
            return "Nouveau"
        elif tenure <= 36:
            return "Intermediaire"
        else:
            return "Fidele"

    df["SegmentTenure"] = df["tenure"].apply(segmenter)
    return df


# -------------------------
# Feature : nombre de services souscrits
# -------------------------

def ajouter_nb_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compte le nombre de services optionnels souscrits par client.

    Services comptés : OnlineSecurity, OnlineBackup, DeviceProtection,
                       TechSupport, StreamingTV, StreamingMovies
    """
    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    # Compte les colonnes avec valeur "Yes"
    df["NbServices"] = df[services].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    return df


# -------------------------
# Feature : client sans services internet
# -------------------------

def ajouter_sans_internet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indique si le client n'a pas de service internet.
    Ces clients ont un profil de churn très différent.
    """
    df["SansInternet"] = (df["InternetService"] == "No").astype(int)
    return df


# -------------------------
# Feature : contrat long terme
# -------------------------

def ajouter_contrat_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indique si le client a un contrat long terme (1 an ou 2 ans).
    Les contrats long terme réduisent fortement le churn.
    """
    df["ContratLong"] = (
        df["Contract"].isin(["One year", "Two year"])
    ).astype(int)
    return df


# -------------------------
# Pipeline complet feature engineering
# -------------------------

def appliquer_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les transformations de feature engineering.
    À appeler sur le DataFrame brut avant le pipeline sklearn.

    Args:
        df (pd.DataFrame) : DataFrame brut (après nettoyage)

    Returns:
        pd.DataFrame : DataFrame enrichi avec nouvelles features
    """
    df = ajouter_charge_moyenne(df)
    df = ajouter_segment_tenure(df)
    df = ajouter_nb_services(df)
    df = ajouter_sans_internet(df)
    df = ajouter_contrat_long(df)

    return df
