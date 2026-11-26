"""GPU TFT training using the same shared data partitions as XGBoost.

Both models import ``src.data.prepare_data.load_shared_data()``. That loader
reads train_scaled.csv, val_scaled.csv, and test_scaled.csv; it does not create
a new split. This script also uses XGBoost's one-hour demand-change target and
its engineered, forecast-available input columns.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

# PyTorch Forecasting can invoke this harmless scikit-learn warning per batch.
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
except ImportError:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE


def project_root() -> Path:
    """Support both src/models/tft_model.py and this delivered output copy."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "src" / "data" / "prepare_data.py").exists():
            return folder
    return Path.cwd()


BASE_DIR = project_root()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from src.data.prepare_data import load_shared_data

RESULTS_DIR = BASE_DIR / "outputs" / "results"

# Copied exactly from src/models/xgboost_model.py.
FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "solar", "wind", "load_lag_1", "load_lag_2", "load_lag_24",
    "solar_lag_1", "wind_lag_1", "load_roll_mean_24", "load_roll_mean_168",
]

TRIALS = [
    {"name": "small_fast", "hidden_size": 16, "attention_head_size": 2,
     "hidden_continuous_size": 8, "dropout": 0.10, "learning_rate": 0.010, "batch_size": 64},
    {"name": "balanced", "hidden_size": 32, "attention_head_size": 4,
     "hidden_continuous_size": 16, "dropout": 0.10, "learning_rate": 0.003, "batch_size": 64},
    {"name": "regularized", "hidden_size": 32, "attention_head_size": 4,
     "hidden_continuous_size": 16, "dropout": 0.20, "learning_rate": 0.001, "batch_size": 128},
    {"name": "larger", "hidden_size": 48, "attention_head_size": 4,
     "hidden_continuous_size": 24, "dropout": 0.15, "learning_rate": 0.002, "batch_size": 64},
]


class LearningCurve(Callback):
    def __init__(self):
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val = metrics.get("val_loss")
        train = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        if val is not None:
            self.rows.append({
                "epoch": trainer.current_epoch + 1,
                "train_loss": float(train.detach().cpu()) if train is not None else np.nan,
                "validation_loss": float(val.detach().cpu()),
            })


class EpochProgress(Callback):
    """Readable epoch summaries alongside Lightning's batch progress bar."""

    def __init__(self, trial_name: str):
        self.trial_name = trial_name
        self.started_at = None

    def on_fit_start(self, trainer, pl_module):
        self.started_at = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Lightning runs one validation sanity check before epoch 1; omit it.
        if trainer.sanity_checking or self.started_at is None:
            return
        completed = trainer.current_epoch + 1
        elapsed = time.perf_counter() - self.started_at
        seconds_per_epoch = elapsed / completed
        remaining = seconds_per_epoch * max(trainer.max_epochs - completed, 0)
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        val_loss = metrics.get("val_loss")
        train_text = f"{float(train_loss.detach().cpu()):.2f}" if train_loss is not None else "n/a"
        val_text = f"{float(val_loss.detach().cpu()):.2f}" if val_loss is not None else "n/a"
        print(
            f"[{self.trial_name}] epoch {completed}/{trainer.max_epochs} complete | "
            f"train loss={train_text} | val loss={val_text} | "
            f"elapsed={elapsed / 60:.1f} min | est. remaining={remaining / 60:.1f} min",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--encoder-hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exact XGBoost target, horizon, lags, and rolling-statistic features."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "load", "solar", "wind", *FEATURES[:6]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Shared split is missing columns: {sorted(missing)}")
    df["target_diff"] = df["load"].shift(-1) - df["load"]
    df["target_actual"] = df["load"].shift(-1)
    df["load_lag_1"] = df["load"].shift(1)
    df["load_lag_2"] = df["load"].shift(2)
    df["load_lag_24"] = df["load"].shift(24)
    df["solar_lag_1"] = df["solar"].shift(1)
    df["wind_lag_1"] = df["wind"].shift(1)
    df["load_roll_mean_24"] = df["load"].rolling(window=24).mean()
    df["load_roll_mean_168"] = df["load"].rolling(window=168).mean()
    return df.dropna(subset=["target_diff", "target_actual", *FEATURES]).reset_index(drop=True)


def prepare_shared_partitions():
    """Load XGBoost's fixed CSVs and create exactly its valid model rows."""
    train_raw, val_raw, test_raw = load_shared_data()
    partitions = {
        "train": create_features(train_raw),
        "validation": create_features(val_raw),
        "test": create_features(test_raw),
    }
    for name, frame in partitions.items():
        frame["series"] = "india_national"
        frame["partition"] = name
    combined = pd.concat([partitions["train"], partitions["validation"], partitions["test"]], ignore_index=True)
    combined["time_idx"] = np.arange(len(combined), dtype=np.int64)
    starts, cursor = {}, 0
    for name in ("train", "validation", "test"):
        starts[name] = cursor
        cursor += len(partitions[name])
    manifest = {
        "loader": "src.data.prepare_data.load_shared_data",
        "source_files": ["outputs/processed/train_scaled.csv", "outputs/processed/val_scaled.csv", "outputs/processed/test_scaled.csv"],
        "target": "target_diff = load(t+1) - load(t); reported prediction = load(t) + predicted_diff",
        "horizon": "1 hour",
        "feature_columns": FEATURES,
        "partitions": {
            name: {
                "raw_rows": int(len(raw)),
                "model_rows_after_XGBoost_feature_lags": int(len(partitions[name])),
                "first_timestamp": str(partitions[name].timestamp.iloc[0]),
                "last_timestamp": str(partitions[name].timestamp.iloc[-1]),
            }
            for name, raw in (("train", train_raw), ("validation", val_raw), ("test", test_raw))
        },
    }
    return combined, starts, manifest


def make_datasets(combined: pd.DataFrame, starts: dict[str, int], encoder_hours: int):
    """TFT datasets whose validation/test predictions stay in fixed partitions."""
    train_end, val_end = starts["validation"], starts["test"]
    if train_end <= encoder_hours:
        raise ValueError("Fixed training split is shorter than encoder-hours.")
    training = TimeSeriesDataSet(
        combined.iloc[:train_end], time_idx="time_idx", target="target_diff", group_ids=["series"],
        min_encoder_length=encoder_hours, max_encoder_length=encoder_hours,
        min_prediction_length=1, max_prediction_length=1,
        # At prediction t, these are available exactly as they are to XGBoost.
        time_varying_known_reals=["time_idx", *FEATURES],
        time_varying_unknown_reals=["target_diff"],
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, combined.iloc[:val_end], min_prediction_idx=starts["validation"], stop_randomization=True
    )
    testing = TimeSeriesDataSet.from_dataset(
        training, combined, min_prediction_idx=starts["test"], stop_randomization=True
    )
    return training, validation, testing


def prediction_frame(model, loader, combined: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct next-hour MW demand exactly as the XGBoost scorer does."""
    result = model.predict(loader, mode="prediction", return_index=True)
    output = result.output if hasattr(result, "output") else result[0]
    index = result.index if hasattr(result, "index") else result[1]
    time_idx = pd.DataFrame(index)["time_idx"].to_numpy()
    rows = combined.set_index("time_idx").loc[time_idx, ["timestamp", "load", "target_actual"]].reset_index()
    rows["predicted_diff"] = np.asarray(output).reshape(-1)
    rows["predicted"] = rows["load"] + rows["predicted_diff"]
    rows = rows.rename(columns={"target_actual": "actual"})
    return rows[["timestamp", "time_idx", "actual", "predicted", "predicted_diff"]]


def train_trial(name, config, training, validation, combined, args, accelerator):
    train_loader = training.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=0)
    checkpoint = ModelCheckpoint(
        dirpath=RESULTS_DIR / "checkpoints" / name, monitor="val_loss", mode="min", save_top_k=1
    )
    curve = LearningCurve()
    progress = EpochProgress(name)
    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator=accelerator, devices=1,
        callbacks=[checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), curve, progress],
        logger=False, enable_model_summary=False, gradient_clip_val=0.1, deterministic=False,
    )
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=config["learning_rate"], hidden_size=config["hidden_size"],
        attention_head_size=config["attention_head_size"], hidden_continuous_size=config["hidden_continuous_size"],
        dropout=config["dropout"], loss=MAE(), log_interval=-1, reduce_on_plateau_patience=3,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    val_pred = prediction_frame(best, val_loader, combined)
    return best, checkpoint.best_model_path, val_pred, mean_absolute_error(val_pred.actual, val_pred.predicted), mean_squared_error(val_pred.actual, val_pred.predicted) ** 0.5, curve.rows


def save_curve(history, path: Path):
    history = pd.DataFrame(history)
    plt.figure(figsize=(8, 4))
    if not history.empty:
        if history.train_loss.notna().any():
            plt.plot(history.epoch, history.train_loss, label="training loss")
        plt.plot(history.epoch, history.validation_loss, label="validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MAE loss"); plt.title("Selected TFT training curve")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, workers=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {accelerator}; CUDA available: {torch.cuda.is_available()}", flush=True)

    combined, starts, manifest = prepare_shared_partitions()
    (RESULTS_DIR / "tft_shared_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Fixed shared model rows: " + ", ".join(
        f"{name}={entry['model_rows_after_XGBoost_feature_lags']}" for name, entry in manifest["partitions"].items()
    ))
    training, validation, testing = make_datasets(combined, starts, args.encoder_hours)

    trials, best_record = [], None
    for config in TRIALS:
        print(f"\nTraining {config['name']}: {config}")
        model, ckpt, val_pred, val_mae, val_rmse, history = train_trial(
            config["name"], config, training, validation, combined, args, accelerator
        )
        row = {**config, "validation_mae_mw": val_mae, "validation_rmse_mw": val_rmse, "checkpoint": str(ckpt)}
        trials.append(row)
        if best_record is None or val_mae < best_record["validation_mae_mw"]:
            best_record = {**row, "model": model, "val_pred": val_pred, "history": history}
            print(f"New best validation MAE: {val_mae:.2f} MW")

    pd.DataFrame(trials).sort_values("validation_mae_mw").to_csv(RESULTS_DIR / "tft_tuning.csv", index=False)
    test_loader = testing.to_dataloader(train=False, batch_size=best_record["batch_size"], num_workers=0)
    test_pred = prediction_frame(best_record["model"], test_loader, combined)
    test_mae = mean_absolute_error(test_pred.actual, test_pred.predicted)
    test_rmse = mean_squared_error(test_pred.actual, test_pred.predicted) ** 0.5
    best_record["val_pred"].to_csv(RESULTS_DIR / "tft_validation_predictions.csv", index=False)
    test_pred.to_csv(RESULTS_DIR / "tft_test_predictions.csv", index=False)
    pd.DataFrame([
        {"dataset": "validation", "MAE_MW": best_record["validation_mae_mw"], "RMSE_MW": best_record["validation_rmse_mw"], "trial": best_record["name"]},
        {"dataset": "test", "MAE_MW": test_mae, "RMSE_MW": test_rmse, "trial": best_record["name"]},
    ]).to_csv(RESULTS_DIR / "tft_metrics.csv", index=False)
    save_curve(best_record["history"], RESULTS_DIR / "tft_training_curve.png")
    os.replace(best_record["checkpoint"], RESULTS_DIR / "tft_best.ckpt")
    selected = {key: best_record[key] for key in TRIALS[0]}
    selected.update({"target": manifest["target"], "loss": "MAE", "optimizer": "Adam (TFT default)",
                     "encoder_hours": args.encoder_hours, "shared_split_manifest": "outputs/results/tft_shared_split_manifest.json",
                     "device": accelerator})
    (RESULTS_DIR / "tft_selected_settings.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"\nBest trial: {best_record['name']} | val MAE={best_record['validation_mae_mw']:.2f} MW | test MAE={test_mae:.2f} MW")
    print(f"Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
