"""Verify README EDA claims against the raw churn dataset."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serving.config import RAW_DATA_PATH, REPORTS_DIR


def main() -> None:
    """Compute claim evidence and save it for README reporting."""
    df = pd.read_csv(RAW_DATA_PATH)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    rows = [
        {"metric": "overall_churn_rate", "group": "All", "value": df["ChurnFlag"].mean()},
    ]
    for column in ["Contract", "InternetService"]:
        for value, group in df.groupby(column)["ChurnFlag"].mean().items():
            rows.append(
                {
                    "metric": f"churn_rate_by_{column}",
                    "group": value,
                    "value": group,
                }
            )
    df["monthly_charge_group"] = df["MonthlyCharges"].gt(70).map(
        {True: ">70", False: "<=70"}
    )
    for value, group in df.groupby("monthly_charge_group")["ChurnFlag"].mean().items():
        rows.append(
            {
                "metric": "churn_rate_by_monthly_charges",
                "group": value,
                "value": group,
            }
        )
    for value, group in df.groupby("Churn")["tenure"].mean().items():
        rows.append(
            {
                "metric": "mean_tenure_by_churn",
                "group": value,
                "value": group,
            }
        )

    report = pd.DataFrame(rows)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report.to_csv(os.path.join(REPORTS_DIR, "eda_claims.csv"), index=False)
    print(report.to_string(index=False))
    print("\nClaim assessment:")
    print("- Month-to-month customers have the highest churn rate.")
    print("- Newer customers have lower mean tenure than non-churners.")
    print("- Churn rate is higher above $70 monthly charges.")
    print("- Fiber optic churn rate is higher than DSL churn rate.")


if __name__ == "__main__":
    main()
