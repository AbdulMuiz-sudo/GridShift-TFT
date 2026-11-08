import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "outputs", "processed")


def load_shared_data():
    """
    Loads the pre-split and pre-scaled data for the models.
    This ensures TFT, XGBoost, and LSTM all use the exact same data.
    """
    train = pd.read_csv(
        os.path.join(PROCESSED_DIR, "train_scaled.csv"), parse_dates=["timestamp"]
    )
    val = pd.read_csv(
        os.path.join(PROCESSED_DIR, "val_scaled.csv"), parse_dates=["timestamp"]
    )
    test = pd.read_csv(
        os.path.join(PROCESSED_DIR, "test_scaled.csv"), parse_dates=["timestamp"]
    )

    print(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")
    return train, val, test


if __name__ == "__main__":
    load_shared_data()
