"""Create a committed background sample for SHAP explanations."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.preprocessing import prepare_features
from src.serving.config import BACKGROUND_PATH, RANDOM_STATE, RAW_DATA_PATH


def main() -> None:
    """Prepare and save a deterministic SHAP background sample."""
    df = pd.read_csv(RAW_DATA_PATH)
    prepared = prepare_features(df)
    background = prepared.sample(n=100, random_state=RANDOM_STATE)
    os.makedirs(os.path.dirname(BACKGROUND_PATH), exist_ok=True)
    background.to_csv(BACKGROUND_PATH, index=False)
    print(f"Saved {len(background)} rows to {BACKGROUND_PATH}")


if __name__ == "__main__":
    main()
