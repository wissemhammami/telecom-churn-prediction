# scripts/check_data.py
"""Check the availability and basic quality of project datasets."""

import os
import pandas as pd

# Dataset paths.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_processed.csv")
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new", "new_customers.csv")


def verifier(path: str, nom: str) -> None:
    """Print basic quality information for one dataset."""
    print(f"\n{'-' * 40}")
    print(f"{nom}")
    print(f"{'-' * 40}")

    if not os.path.exists(path):
        print(f"Introuvable : {path}")
        return

    df = pd.read_csv(path)
    nb_nan = df.isna().sum().sum()
    nb_duplic = df.duplicated().sum()

    print(f"Lignes   : {df.shape[0]}")
    print(f"Colonnes : {df.shape[1]}")
    print(f"NaN      : {nb_nan}")
    print(f"Doublons : {nb_duplic}")

    # Show target distribution when available.
    if "Churn" in df.columns:
        counts = df["Churn"].value_counts()
        pcts = df["Churn"].value_counts(normalize=True) * 100
        print(f"\nChurn :")
        for val in counts.index:
            print(f"  {val} : {counts[val]} ({pcts[val]:.1f}%)")

    print(f"\n{df.head(3).to_string()}")


def main():
    """Check all configured datasets."""
    verifier(RAW_DATA_PATH, "RAW       — data/raw/churn.csv")
    verifier(PROCESSED_DATA_PATH, "PROCESSED — data/processed/churn_processed.csv")
    verifier(NEW_DATA_PATH, "NEW       — data/new/new_customers.csv")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
