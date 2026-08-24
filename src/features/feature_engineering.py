"""Create derived customer features shared by training and inference."""

import pandas as pd


def ajouter_charge_moyenne(df: pd.DataFrame) -> pd.DataFrame:
    """Add average monthly charges while handling zero-tenure customers."""
    df["ChargesMoyennes"] = df.apply(
        lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else 0,
        axis=1,
    )
    return df


def ajouter_segment_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Add a tenure segment for new, intermediate, and loyal customers."""

    def segmenter(tenure):
        if tenure <= 12:
            return "Nouveau"
        elif tenure <= 36:
            return "Intermediaire"
        else:
            return "Fidele"

    df["SegmentTenure"] = df["tenure"].apply(segmenter)
    return df


def ajouter_nb_services(df: pd.DataFrame) -> pd.DataFrame:
    """Add the number of optional services marked as Yes."""
    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["NbServices"] = df[services].apply(lambda row: (row == "Yes").sum(), axis=1)
    return df


def ajouter_sans_internet(df: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator for customers without internet service."""
    df["SansInternet"] = (df["InternetService"] == "No").astype(int)
    return df


def ajouter_contrat_long(df: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator for one-year and two-year contracts."""
    df["ContratLong"] = (df["Contract"].isin(["One year", "Two year"])).astype(int)
    return df


def appliquer_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all derived-feature transformations to a customer DataFrame."""
    df = ajouter_charge_moyenne(df)
    df = ajouter_segment_tenure(df)
    df = ajouter_nb_services(df)
    df = ajouter_sans_internet(df)
    df = ajouter_contrat_long(df)

    return df
