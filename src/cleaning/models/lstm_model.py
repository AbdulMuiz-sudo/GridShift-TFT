"""Train an LSTM on the same pre-split CSVs, features, and target as TFT/XGBoost.

Run from the project root:
    python src/models/lstm_model.py

Required files: outputs/processed/train_scaled.csv, val_scaled.csv, test_scaled.csv
Dependencies: pandas numpy scikit-learn torch
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "outputs" / "processed" / "train_scaled.csv").exists():
            return parent
    return Path.cwd()


ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.prepare_data import load_shared_data

PROCESSED = ROOT / "outputs" / "processed"
RESULTS = ROOT / "outputs" / "results"
FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "solar", "wind", "load_lag_1", "load_lag_2", "load_lag_24",
    "solar_lag_1", "wind_lag_1", "load_roll_mean_24", "load_roll_mean_168",
]
TRIALS = [
    {"name": "small", "hidden_size": 32, "layers": 1, "dropout": 0.0,
     "learning_rate": 0.001, "batch_size": 128},
    {"name": "balanced", "hidden_size": 64, "layers": 1, "dropout": 0.0,
     "learning_rate": 0.001, "batch_size": 128},
    {"name": "deeper", "hidden_size": 64, "layers": 2, "dropout": 0.1,
     "learning_rate": 0.0005, "batch_size": 128},
]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-hours", type=int, default=168)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_partition_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    required = {"timestamp", "load", "solar", "wind", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "month_sin", "month_cos"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} split is missing columns: {sorted(missing)}")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    # These transformations match src/models/xgboost_model.py and tft_model.py.
    frame["target_diff"] = frame["load"].shift(-1) - frame["load"]
    frame["target_actual"] = frame["load"].shift(-1)
    frame["load_lag_1"] = frame["load"].shift(1)
    frame["load_lag_2"] = frame["load"].shift(2)
    frame["load_lag_24"] = frame["load"].shift(24)
    frame["solar_lag_1"] = frame["solar"].shift(1)
    frame["wind_lag_1"] = frame["wind"].shift(1)
    frame["load_roll_mean_24"] = frame["load"].rolling(24).mean()
    frame["load_roll_mean_168"] = frame["load"].rolling(168).mean()
    frame = frame.dropna(subset=["target_diff", "target_actual", *FEATURES]).reset_index(drop=True)
    frame["partition"] = name
    return frame


def make_windows(frame: pd.DataFrame, lookback: int, feature_mean, feature_std,
                 target_mean, target_std, device):
    raw = frame[FEATURES].to_numpy(dtype=np.float64)
    x_all = ((raw - feature_mean) / feature_std).astype(np.float32)
    y_all = ((frame["target_diff"].to_numpy(dtype=np.float64) - target_mean) / target_std).astype(np.float32)
    # Each split is prepared independently by the partner models. Keep its boundary
    # intact and use only past rows from that same split for every sequence.
    indices = np.arange(lookback - 1, len(frame))
    if len(indices) == 0:
        raise ValueError(f"{frame['partition'].iloc[0]} split has no rows after lookback={lookback}.")
    x = np.stack([x_all[i-lookback+1:i+1] for i in indices])
    return (torch.from_numpy(x).to(device), torch.from_numpy(y_all[indices]).to(device), indices)


def predict(model, x, target_mean, target_std):
    model.eval()
    outputs = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(x), batch_size=512, shuffle=False):
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs) * target_std + target_mean


def main():
    args = parse_args()
    if args.lookback_hours < 1:
        raise ValueError("--lookback-hours must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train_raw, val_raw, test_raw = load_shared_data()
    parts = {
        "train": load_partition_frame(train_raw, "train"),
        "validation": load_partition_frame(val_raw, "validation"),
        "test": load_partition_frame(test_raw, "test"),
    }
    print("Rows after matching XGBoost feature engineering:", ", ".join(f"{name}={len(frame)}" for name, frame in parts.items()))

    train = parts["train"]
    raw_train_x = train[FEATURES].to_numpy(dtype=np.float64)
    feature_mean = raw_train_x.mean(axis=0)
    feature_std = raw_train_x.std(axis=0); feature_std[feature_std == 0] = 1.0
    target_mean = float(train["target_diff"].mean())
    target_std = float(train["target_diff"].std()) or 1.0
    windows = {name: make_windows(frame, args.lookback_hours, feature_mean, feature_std,
                                   target_mean, target_std, device)
               for name, frame in parts.items()}
    x_train, y_train, train_idx = windows["train"]
    x_val, y_val, val_idx = windows["validation"]
    x_test, y_test, test_idx = windows["test"]
    print(f"Device: {device}; lookback: {args.lookback_hours} hours; samples train/val/test: {len(x_train)}/{len(x_val)}/{len(x_test)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    tuning, best = [], None
    for config in TRIALS:
        print(f"\nTraining {config['name']}: {config}")
        model = LSTMForecaster(len(FEATURES), config["hidden_size"], config["layers"], config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config["batch_size"], shuffle=True)
        best_val_mae, best_state, stale, history = float("inf"), None, 0, []
        for epoch in range(args.max_epochs):
            model.train(); losses = []
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(bx), by)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach()))
            val_diff = predict(model, x_val, target_mean, target_std)
            val_frame = parts["validation"].iloc[val_idx]
            val_pred = val_frame["load"].to_numpy() + val_diff
            val_actual = val_frame["target_actual"].to_numpy()
            val_mae = float(mean_absolute_error(val_actual, val_pred))
            history.append({"epoch": epoch + 1, "train_mse_scaled": float(np.mean(losses)), "validation_mae_mw": val_mae})
            print(f"Epoch {epoch+1:02d}: val MAE {val_mae:.2f} MW")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= args.patience: break
        model.load_state_dict(best_state)
        val_diff = predict(model, x_val, target_mean, target_std)
        val_frame = parts["validation"].iloc[val_idx]
        val_pred = val_frame["load"].to_numpy() + val_diff
        val_actual = val_frame["target_actual"].to_numpy()
        val_rmse = float(mean_squared_error(val_actual, val_pred) ** 0.5)
        row = {**config, "validation_mae_mw": best_val_mae, "validation_rmse_mw": val_rmse, "epochs_trained": len(history)}
        tuning.append(row)
        if best is None or best_val_mae < best["row"]["validation_mae_mw"]:
            best = {"row": row, "model": model, "history": history, "val_pred": val_pred}

    pd.DataFrame(tuning).sort_values("validation_mae_mw").to_csv(RESULTS / "lstm_tuning.csv", index=False)
    model = best["model"]
    test_diff = predict(model, x_test, target_mean, target_std)
    test_frame = parts["test"].iloc[test_idx]
    test_pred = test_frame["load"].to_numpy() + test_diff
    val_frame = parts["validation"].iloc[val_idx]
    val_actual, test_actual = val_frame["target_actual"].to_numpy(), test_frame["target_actual"].to_numpy()
    val_predictions = pd.DataFrame({"timestamp": val_frame.timestamp, "actual": val_actual, "predicted": best["val_pred"]})
    test_predictions = pd.DataFrame({"timestamp": test_frame.timestamp, "actual": test_actual, "predicted": test_pred})
    val_predictions.to_csv(RESULTS / "lstm_validation_predictions.csv", index=False)
    test_predictions.to_csv(RESULTS / "lstm_test_predictions.csv", index=False)
    metrics = pd.DataFrame([
        {"dataset":"validation", "MAE":mean_absolute_error(val_actual, best["val_pred"]), "RMSE":mean_squared_error(val_actual, best["val_pred"])**0.5, "trial":best["row"]["name"]},
        {"dataset":"test", "MAE":mean_absolute_error(test_actual, test_pred), "RMSE":mean_squared_error(test_actual, test_pred)**0.5, "trial":best["row"]["name"]},
    ])
    metrics.to_csv(RESULTS / "lstm_metrics.csv", index=False)
    torch.save({"state_dict":model.state_dict(), "config":best["row"], "features":FEATURES,
                "feature_mean":feature_mean.tolist(), "feature_std":feature_std.tolist(),
                "target_mean":target_mean, "target_std":target_std, "lookback_hours":args.lookback_hours}, RESULTS / "lstm_best.pt")
    pd.DataFrame(best["history"]).to_csv(RESULTS / "lstm_training_history.csv", index=False)
    settings = {**best["row"], "target":"load(t+1)-load(t), reconstructed to Demand (MW)",
                "lookback_hours":args.lookback_hours, "split":"provided chronological train/validation/test CSVs",
                "source_files":["outputs/processed/train_scaled.csv", "outputs/processed/val_scaled.csv", "outputs/processed/test_scaled.csv"],
                "source_rows":{"train":len(train_raw), "validation":len(val_raw), "test":len(test_raw)},
                "model_rows":{name:int(len(frame)) for name,frame in parts.items()},
                "predicted_rows":{"train":len(train_idx),"validation":len(val_idx),"test":len(test_idx)},
                "features":FEATURES,"scaling":"feature and target mean/std fit on training split only"}
    (RESULTS / "lstm_selected_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("\nSelected configuration:", best["row"]["name"])
    print(metrics.to_string(index=False))
    print("\nResults saved to:", RESULTS)


if __name__ == "__main__":
    main()


