import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import itertools

# Add project root to path to import prepare_data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data.prepare_data import load_shared_data

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_features(df):
    """
    Create lag features and rolling statistics.
    Crucial fix: We switch the target to the DIFFERENCE between t and t+1
    to handle distribution shifts and capacity growth.
    """
    df = df.copy()

    # Target: The change in load from current hour to next hour
    df["target_diff"] = df["load"].shift(-1) - df["load"]

    # Keep actual for final evaluation reconstruction
    df["target_actual"] = df["load"].shift(-1)

    # Standard Lags
    df["load_lag_1"] = df["load"].shift(1)
    df["load_lag_2"] = df["load"].shift(2)
    df["load_lag_24"] = df["load"].shift(24)
    df["solar_lag_1"] = df["solar"].shift(1)
    df["wind_lag_1"] = df["wind"].shift(1)

    # New: Rolling Averages (smooths out noise and provides recent baseline)
    df["load_roll_mean_24"] = df["load"].rolling(window=24).mean()
    df["load_roll_mean_168"] = df["load"].rolling(window=168).mean()  # 7-day trend

    # Drop rows with NaN (created by shifting/rolling)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    print("1. Loading shared data...")
    train, val, test = load_shared_data()

    print("2. Engineering temporal features and rolling statistics...")
    train = create_features(train)
    val = create_features(val)
    test = create_features(test)

    # Define features
    features = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "solar",
        "wind",
        "load_lag_1",
        "load_lag_2",
        "load_lag_24",
        "solar_lag_1",
        "wind_lag_1",
        "load_roll_mean_24",
        "load_roll_mean_168",
    ]

    # We train on the difference, not the absolute value
    target = "target_diff"

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]

    print(
        f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}"
    )

    print("\n3. Starting Hyperparameter Tuning (with Regularization)...")
    # Expanded grid with regularization to prevent overfitting to the training set noise
    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8],  # Row sampling to prevent overfitting
        "colsample_bytree": [0.8],  # Feature sampling to force generalization
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tuning_results = []
    best_mae = float("inf")
    best_params = None

    for params in experiments:
        model = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, **params
        )
        model.fit(X_train, y_train)

        # Predict the DIFFERENCE
        val_diff_preds = model.predict(X_val)

        # Reconstruct the absolute prediction: Current Load + Predicted Difference
        val_preds_absolute = val["load"] + val_diff_preds

        # Evaluate on the actual absolute load
        val_mae = mean_absolute_error(val["target_actual"], val_preds_absolute)

        result_row = params.copy()
        result_row["val_mae"] = val_mae
        tuning_results.append(result_row)

        print(f"Tested {params} -> Val MAE: {val_mae:.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(os.path.join(RESULTS_DIR, "xgboost_tuning.csv"), index=False)
    print(f"\nBest Params: {best_params} (Val MAE: {best_mae:.2f})")

    print("\n4. Training Final Model with Best Params...")
    final_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, **best_params
    )
    final_model.fit(X_train, y_train)

    print("\n5. Evaluating Final Model (Absolute Reconstruction)...")
    # Predict differences
    val_diff_preds = final_model.predict(X_val)
    test_diff_preds = final_model.predict(X_test)

    # Reconstruct absolute predictions
    val_preds = val["load"] + val_diff_preds
    test_preds = test["load"] + test_diff_preds

    # Validation Metrics
    val_actual = val["target_actual"]
    val_mae = mean_absolute_error(val_actual, val_preds)
    val_rmse = np.sqrt(mean_squared_error(val_actual, val_preds))
    val_mape = mean_absolute_percentage_error(val_actual, val_preds)

    # Test Metrics
    test_actual = test["target_actual"]
    test_mae = mean_absolute_error(test_actual, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
    test_mape = mean_absolute_percentage_error(test_actual, test_preds)

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "validation",
                "MAE": val_mae,
                "RMSE": val_rmse,
                "MAPE": f"{val_mape*100:.2f}%",
            },
            {
                "dataset": "test",
                "MAE": test_mae,
                "RMSE": test_rmse,
                "MAPE": f"{test_mape*100:.2f}%",
            },
        ]
    )

    metrics_df.to_csv(os.path.join(RESULTS_DIR, "xgboost_metrics.csv"), index=False)
    print("\n" + metrics_df.to_string(index=False))

    pd.DataFrame({"actual": test_actual, "predicted": test_preds}).to_csv(
        os.path.join(RESULTS_DIR, "xgboost_test_predictions.csv"), index=False
    )
    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
