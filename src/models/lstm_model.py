"""Train a one-hour-ahead LSTM on the shared GridShift dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs" / "processed" / "grid_hourly_merged.csv"
RESULTS = ROOT / "outputs" / "results"

FEATURES = [
    "load", "solar", "wind",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

TRIALS = [
    {
        "name": "small", "hidden_size": 32, "layers": 1,
        "dropout": 0.0, "learning_rate": 0.001, "batch_size": 128,
    },
    {
        "name": "balanced", "hidden_size": 64, "layers": 1,
        "dropout": 0.1, "learning_rate": 0.001, "batch_size": 128,
    },
    {
        "name": "deeper", "hidden_size": 64, "layers": 2,
        "dropout": 0.1, "learning_rate": 0.0005, "batch_size": 128,
    },
]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--lookback-hours", type=int, default=168)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_frame(path):
    df = pd.read_csv(path)
    required = ["timestamp", "Demand (MW)", "Solar (MW)", "Wind (MW)"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = df.rename(columns={
        "Demand (MW)": "load",
        "Solar (MW)": "solar",
        "Wind (MW)": "wind",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for column in ("load", "solar", "wind"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "load", "solar", "wind"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps; resolve before training.")

    ts = df["timestamp"]
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12)
    return df


def make_windows(features, target, indices, lookback, device):
    # Inputs end immediately before the hour being predicted.
    x = np.stack([features[i - lookback:i] for i in indices]).astype("float32")
    y = target[indices].astype("float32")
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def predict(model, x, target_mean, target_std):
    model.eval()
    predictions = []

    with torch.no_grad():
        loader = DataLoader(
            TensorDataset(x),
            batch_size=512,
            shuffle=False,
        )
        for (batch,) in loader:
            predictions.append(model(batch).cpu().numpy())

    return np.concatenate(predictions) * target_std + target_mean


def main():
    args = parse_args()

    if args.lookback_hours < 1:
        raise ValueError("--lookback-hours must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    df = load_frame(args.data)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    expected = (25678, 17974, 21826, 3852, 3852)
    actual = (n, train_end, val_end, val_end - train_end, n - val_end)
    if actual != expected:
        raise ValueError(
            "Data/split mismatch. Expected "
            f"(rows, train_end, val_end, validation, test)={expected}; "
            f"got {actual}. Use the same complete-row CSV as TFT."
        )

    if train_end <= args.lookback_hours:
        raise ValueError("Training split is shorter than the lookback window.")

    print(
        f"Complete rows: {n}; train/validation/test: "
        f"{train_end}/{val_end - train_end}/{n - val_end}"
    )

    raw_features = df[FEATURES].to_numpy(dtype=np.float64)

    # Fit feature and target scaling using training rows only.
    feature_mean = raw_features[:train_end].mean(axis=0)
    feature_std = raw_features[:train_end].std(axis=0)
    feature_std[feature_std == 0] = 1
    features = ((raw_features - feature_mean) / feature_std).astype("float32")

    actual_load = df["load"].to_numpy(dtype=np.float64)
    target_mean = float(actual_load[:train_end].mean())
    target_std = float(actual_load[:train_end].std()) or 1.0
    target = ((actual_load - target_mean) / target_std).astype("float32")

    train_indices = np.arange(args.lookback_hours, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n)

    x_train, y_train = make_windows(
        features, target, train_indices, args.lookback_hours, device
    )
    x_val, y_val = make_windows(
        features, target, val_indices, args.lookback_hours, device
    )
    x_test, y_test = make_windows(
        features, target, test_indices, args.lookback_hours, device
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    tuning_rows = []
    best = None

    for config in TRIALS:
        print(f"\nTraining {config['name']}: {config}")

        model = LSTMForecaster(
            len(FEATURES),
            config["hidden_size"],
            config["layers"],
            config["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"]
        )
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=config["batch_size"],
            shuffle=True,
        )

        best_val_mae = float("inf")
        best_state = None
        stale_epochs = 0
        history = []

        for epoch in range(args.max_epochs):
            model.train()
            losses = []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))

            val_predictions = predict(model, x_val, target_mean, target_std)
            val_actual = actual_load[val_indices]
            val_mae = float(mean_absolute_error(val_actual, val_predictions))

            history.append({
                "epoch": epoch + 1,
                "train_mse_scaled": float(np.mean(losses)),
                "validation_mae_mw": val_mae,
            })
            print(f"Epoch {epoch + 1:02d}: val MAE {val_mae:.2f} MW")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= args.patience:
                break

        model.load_state_dict(best_state)
        val_predictions = predict(model, x_val, target_mean, target_std)
        val_rmse = float(
            mean_squared_error(
                actual_load[val_indices], val_predictions
            ) ** 0.5
        )

        row = {
            **config,
            "validation_mae_mw": best_val_mae,
            "validation_rmse_mw": val_rmse,
            "epochs_trained": len(history),
        }
        tuning_rows.append(row)

        if best is None or best_val_mae < best["row"]["validation_mae_mw"]:
            best = {
                "row": row,
                "model": model,
                "history": history,
                "val_predictions": val_predictions,
            }

    pd.DataFrame(tuning_rows).sort_values(
        "validation_mae_mw"
    ).to_csv(RESULTS / "lstm_tuning.csv", index=False)

    model = best["model"]
    test_predictions = predict(model, x_test, target_mean, target_std)
    val_predictions = best["val_predictions"]

    def prediction_frame(indices, predictions):
        return pd.DataFrame({
            "timestamp": df["timestamp"].iloc[indices].to_numpy(),
            "actual": actual_load[indices],
            "predicted": predictions,
        })

    val_frame = prediction_frame(val_indices, val_predictions)
    test_frame = prediction_frame(test_indices, test_predictions)
    val_frame.to_csv(RESULTS / "lstm_validation_predictions.csv", index=False)
    test_frame.to_csv(RESULTS / "lstm_test_predictions.csv", index=False)

    metrics = pd.DataFrame([
        {
            "dataset": "validation",
            "MAE_MW": mean_absolute_error(
                actual_load[val_indices], val_predictions
            ),
            "RMSE_MW": mean_squared_error(
                actual_load[val_indices], val_predictions
            ) ** 0.5,
            "trial": best["row"]["name"],
        },
        {
            "dataset": "test",
            "MAE_MW": mean_absolute_error(
                actual_load[test_indices], test_predictions
            ),
            "RMSE_MW": mean_squared_error(
                actual_load[test_indices], test_predictions
            ) ** 0.5,
            "trial": best["row"]["name"],
        },
    ])
    metrics.to_csv(RESULTS / "lstm_metrics.csv", index=False)

    torch.save({
        "state_dict": model.state_dict(),
        "config": best["row"],
        "features": FEATURES,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "lookback_hours": args.lookback_hours,
    }, RESULTS / "lstm_best.pt")

    pd.DataFrame(best["history"]).to_csv(
        RESULTS / "lstm_training_history.csv", index=False
    )

    settings = {
        **best["row"],
        "target": "Demand (MW), one hour ahead",
        "lookback_hours": args.lookback_hours,
        "split": "chronological 70/15/15",
        "rows": n,
        "split_rows": {
            "train": train_end,
            "validation": val_end - train_end,
            "test": n - val_end,
        },
        "device": str(device),
        "features": FEATURES,
        "scaling": "mean/std fitted on training rows only",
    }
    (RESULTS / "lstm_selected_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print(f"\nSelected configuration: {best['row']['name']}")
    print(metrics.to_string(index=False))
    print(f"\nResults saved to: {RESULTS}")


if __name__ == "__main__":
    main()