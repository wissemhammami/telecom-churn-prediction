# test.py
"""
📌 Data Validation Script for Telecom Churn Prediction
------------------------------------------------------
This script performs a comprehensive validation of the processed dataset
used for training and inference in the Telecom Churn Prediction project.

Validation checks include:
1️⃣ Missing values (NaNs)
2️⃣ Column data types
3️⃣ Target column correctness
4️⃣ Basic statistics for numeric columns
5️⃣ Dataset integrity and consistency

Usage:
------
python test.py
"""

import os
import pandas as pd

# -------------------------
# 1️⃣ Paths
# -------------------------
# Define the base directory of the project and locate the processed dataset
BASE_DIR = os.path.abspath(".")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_processed.csv")

# -------------------------
# 2️⃣ Load processed dataset
# -------------------------


def load_data(path):
    """
    Loads the processed dataset from disk.

    Args:
        path (str): Path to the CSV file containing the processed dataset

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If the dataset does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}. Run preprocessing first.")
    df = pd.read_csv(path)
    return df

# -------------------------
# 3️⃣ Validation checks
# -------------------------


def check_nans(df):
    """
    Checks for missing values (NaNs) in all columns.

    Args:
        df (pd.DataFrame): Dataset to validate
    """
    nan_counts = df.isna().sum()
    print("📝 NaN values per column:\n", nan_counts)
    if df.isna().any().any():
        print("\n❌ There are missing values in the dataset.")
    else:
        print("\n✅ No missing values detected.")


def check_types(df):
    """
    Displays data types for all columns to ensure preprocessing consistency.

    Args:
        df (pd.DataFrame): Dataset to validate
    """
    print("\n🔎 Column data types:\n", df.dtypes)


def check_target(df, target="Churn"):
    """
    Validates the target column:
    - Exists in the dataset
    - Encoded correctly as 0/1

    Args:
        df (pd.DataFrame): Dataset to validate
        target (str): Target column name
    """
    if target not in df.columns:
        print(f"\n❌ Target column '{target}' not found in dataset.")
        return
    unique_vals = df[target].unique()
    print(f"\n🎯 Target column '{target}' unique values: {unique_vals}")
    if set(unique_vals) <= {0, 1}:
        print(f"✅ Target column '{target}' is correctly encoded as 0/1.")
    else:
        print(f"❌ Target column '{target}' contains unexpected values.")


def check_numeric_stats(df):
    """
    Prints basic statistics for numeric columns to detect anomalies.

    Args:
        df (pd.DataFrame): Dataset to validate
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        print("\n📊 Basic statistics for numeric columns:")
        print(df[numeric_cols].describe())
    else:
        print("\n❌ No numeric columns found.")

# -------------------------
# 4️⃣ Main validation workflow
# -------------------------


def main():
    """
    Main entry point for data validation.
    Loads the dataset and runs all validation checks.
    """
    print("🚀 Starting data validation for Telecom Churn Dataset...\n")
    df = load_data(PROCESSED_DATA_PATH)

    check_nans(df)
    check_types(df)
    check_target(df)
    check_numeric_stats(df)

    print("\n✅ Data validation completed successfully.")


# -------------------------
# 5️⃣ Script entry point
# -------------------------
if __name__ == "__main__":
    main()
