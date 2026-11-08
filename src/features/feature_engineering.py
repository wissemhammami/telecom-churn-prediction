"""Create derived customer features shared by training and inference."""

import numpy as np
import pandas as pd


def add_average_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Add average monthly charges while handling zero-tenure customers."""
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"].replace(0, np.nan),
        0,
    )
    return df


def add_tenure_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Add a tenure segment for new, intermediate, and loyal customers."""

    def segment_tenure(tenure):
        if tenure <= 12:
            return "New"
        elif tenure <= 36:
            return "Intermediate"
        else:
            return "Loyal"

    df["TenureSegment"] = df["tenure"].apply(segment_tenure)
    return df


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """Add the number of optional services marked as Yes."""
    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["NumServices"] = (df[services] == "Yes").sum(axis=1)
    return df


def add_no_internet(df: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator for customers without internet service."""
    df["NoInternet"] = (df["InternetService"] == "No").astype(int)
    return df


def add_long_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator for one-year and two-year contracts."""
    df["LongContract"] = (df["Contract"].isin(["One year", "Two year"])).astype(int)
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all derived-feature transformations to a customer DataFrame."""
    df = add_average_charge(df)
    df = add_tenure_segment(df)
    df = add_num_services(df)
    df = add_no_internet(df)
    df = add_long_contract(df)

    return df
