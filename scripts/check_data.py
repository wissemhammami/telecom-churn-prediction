# scripts/check_data.py
"""
Vérification rapide des données — Telecom Churn Prediction
Usage : python scripts/check_data.py
"""

import os
import pandas as pd

# -------------------------
# Chemins
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_processed.csv")
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new", "new_customers.csv")


# -------------------------
# Vérification fichier
# -------------------------

def verifier(path: str, nom: str) -> None:
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

    # Distribution target si presente
    if "Churn" in df.columns:
        counts = df["Churn"].value_counts()
        pcts = df["Churn"].value_counts(normalize=True) * 100
        print(f"\nChurn :")
        for val in counts.index:
            print(f"  {val} : {counts[val]} ({pcts[val]:.1f}%)")

    print(f"\n{df.head(3).to_string()}")


# -------------------------
# Main
# -------------------------

def main():
    verifier(RAW_DATA_PATH, "RAW       — data/raw/churn.csv")
    verifier(PROCESSED_DATA_PATH, "PROCESSED — data/processed/churn_processed.csv")
    verifier(NEW_DATA_PATH, "NEW       — data/new/new_customers.csv")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
