"""Train and evaluate a leakage-safe Temporal Fusion Transformer (TFT).

Shared handoff implemented here
-------------------------------
* source: outputs/processed/grid_hourly_merged.csv
* target: electricity demand (MW) one hour ahead
* split: chronological 70% / 15% / 15%; latest 15% is the test period
* available-at-forecast inputs: history of load/solar/wind and calendar values

The script writes tuning results, final metrics/predictions, a learning-curve
plot, architecture metadata, and a Lightning checkpoint to outputs/results/.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:  # supports both current and older Lightning installs
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
except ImportError:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

# PyTorch Forecasting passes unnamed NumPy arrays to scikit-learn scalers that
# it fitted from DataFrame columns. That harmless internal mismatch otherwise
# emits this warning once per batch and floods the terminal.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\\.utils\\.validation",
)


def project_root() -> Path:
    """Work both from outputs/tft_model.py and src/models/tft_model.py."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "outputs" / "processed" / "grid_hourly_merged.csv").exists():
            return folder
    return Path.cwd()


PROJECT_ROOT = project_root()
DEFAULT_DATA = PROJECT_ROOT / "outputs" / "processed" / "grid_hourly_merged.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# All trials use the same data, horizon, and feature availability.  Change this
# list to add experiments; results are written to tft_tuning.csv.
TRIALS = [
    {"name": "small_fast", "hidden_size": 16, "attention_head_size": 2,
     "hidden_continuous_size": 8, "dropout": 0.10, "learning_rate": 0.010,
     "batch_size": 64},
    {"name": "balanced", "hidden_size": 32, "attention_head_size": 4,
     "hidden_continuous_size": 16, "dropout": 0.10, "learning_rate": 0.003,
     "batch_size": 64},
    {"name": "regularized", "hidden_size": 32, "attention_head_size": 4,
     "hidden_continuous_size": 16, "dropout": 0.20, "learning_rate": 0.001,
     "batch_size": 128},
    {"name": "larger", "hidden_size": 48, "attention_head_size": 4,
     "hidden_continuous_size": 24, "dropout": 0.15, "learning_rate": 0.002,
     "batch_size": 64},
]


class LearningCurve(Callback):
    """Collect epoch losses without requiring an external experiment service."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to grid_hourly_merged.csv")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--encoder-hours", type=int, default=168,
                        help="History given to TFT (default: 7 days)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even when CUDA is available")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    """Read Abdullah's merged CSV, normalize its known headers, and validate it."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"timestamp", "Demand (MW)", "Solar (MW)", "Wind (MW)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected CSV headers; missing {sorted(missing)}. Found {list(df.columns)}")

    df = df.rename(columns={"Demand (MW)": "load", "Solar (MW)": "solar", "Wind (MW)": "wind"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ("load", "solar", "wind"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "load", "solar", "wind"]).sort_values("timestamp")
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found. Resolve them in the data-preparation step first.")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological after sorting.")

    # Calendar fields are known for every future timestamp. Solar, wind and load
    # deliberately remain *unknown* future reals, preventing forecast leakage.
    df["hour_sin"] = np.sin(2 * np.pi * df.timestamp.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.timestamp.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.timestamp.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.timestamp.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df.timestamp.dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df.timestamp.dt.month - 1) / 12)
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    df["series"] = "india_national"
    return df.reset_index(drop=True)


def make_datasets(df: pd.DataFrame, encoder_hours: int):
    """Build datasets with the exact 70/15/15 chronological handoff split."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end <= encoder_hours:
        raise ValueError("Training portion is shorter than the encoder history.")

    known_reals = ["time_idx", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    # Only historical values are usable for these. They are excluded from decoder inputs.
    unknown_reals = ["load", "solar", "wind"]

    training = TimeSeriesDataSet(
        df.iloc[:train_end],
        time_idx="time_idx", target="load", group_ids=["series"],
        min_encoder_length=encoder_hours, max_encoder_length=encoder_hours,
        min_prediction_length=1, max_prediction_length=1,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    # Include earlier rows solely as encoder history; min_prediction_idx ensures
    # validation and test targets are strictly confined to their own time periods.
    validation = TimeSeriesDataSet.from_dataset(
        training, df.iloc[:val_end], min_prediction_idx=train_end, stop_randomization=True
    )
    testing = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=val_end, stop_randomization=True
    )
    return training, validation, testing, train_end, val_end


def predict_frame(model, loader, source: pd.DataFrame) -> pd.DataFrame:
    """Return timestamp-aligned, inverse-transformed one-step predictions in MW."""
    prediction = model.predict(loader, mode="prediction", return_index=True)
    # pytorch-forecasting returns a Prediction named tuple in current releases.
    values = prediction.output if hasattr(prediction, "output") else prediction[0]
    index = prediction.index if hasattr(prediction, "index") else prediction[1]
    values = np.asarray(values).reshape(-1)
    index = pd.DataFrame(index)
    pred_time_idx = index["time_idx"].to_numpy()
    actual = source.set_index("time_idx").loc[pred_time_idx, ["timestamp", "load"]].reset_index()
    actual["predicted"] = values
    actual = actual.rename(columns={"load": "actual"})
    return actual[["timestamp", "time_idx", "actual", "predicted"]]


def train_trial(name, config, training, validation, val_source, args, accelerator):
    """Fit one configuration and return validation MAE plus its best checkpoint."""
    train_loader = training.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=0)
    trial_dir = RESULTS_DIR / "checkpoints" / name
    checkpoint = ModelCheckpoint(dirpath=trial_dir, monitor="val_loss", mode="min", save_top_k=1)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)
    curve = LearningCurve()
    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator=accelerator, devices=1,
        callbacks=[checkpoint, early_stop, curve], logger=False, enable_model_summary=False,
        # TFT's CUDA interpolation backward pass has no deterministic kernel.
        # Disabling strict determinism permits GPU training; the fixed seed still
        # makes runs reasonably comparable, though not bit-for-bit identical.
        gradient_clip_val=0.1, deterministic=False,
    )
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=config["learning_rate"], hidden_size=config["hidden_size"],
        attention_head_size=config["attention_head_size"],
        hidden_continuous_size=config["hidden_continuous_size"], dropout=config["dropout"],
        loss=MAE(), log_interval=-1, reduce_on_plateau_patience=3,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    val_predictions = predict_frame(best, val_loader, val_source)
    val_mae = mean_absolute_error(val_predictions.actual, val_predictions.predicted)
    val_rmse = mean_squared_error(val_predictions.actual, val_predictions.predicted) ** 0.5
    return best, checkpoint.best_model_path, val_predictions, val_mae, val_rmse, curve.rows


def save_curve(history, destination: Path):
    """Save the selected trial's epoch-by-epoch training and validation losses."""
    history = pd.DataFrame(history)
    plt.figure(figsize=(8, 4))
    if not history.empty:
        if history["train_loss"].notna().any():
            plt.plot(history["epoch"], history["train_loss"], label="training loss")
        plt.plot(history["epoch"], history["validation_loss"], label="validation loss")
    plt.title("TFT training run")
    plt.ylabel("MAE loss (normalized by target scale during training)")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, workers=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"
    print(f"Device: {accelerator}; CUDA available: {torch.cuda.is_available()}")

    df = load_data(args.data)
    training, validation, testing, train_end, val_end = make_datasets(df, args.encoder_hours)
    print(f"Rows: train={train_end}, validation={val_end-train_end}, test={len(df)-val_end}")

    tuning_rows, best_record = [], None
    for trial in TRIALS:
        print(f"\nTraining {trial['name']}: {trial}")
        model, checkpoint, val_pred, val_mae, val_rmse, history = train_trial(
            trial["name"], trial, training, validation, df, args, accelerator
        )
        row = {**trial, "validation_mae_mw": val_mae, "validation_rmse_mw": val_rmse,
               "checkpoint": str(checkpoint)}
        tuning_rows.append(row)
        if best_record is None or val_mae < best_record["validation_mae_mw"]:
            best_record = {**row, "model": model, "val_predictions": val_pred, "history": history}
            print(f"New best validation MAE: {val_mae:.2f} MW")

    tuning = pd.DataFrame(tuning_rows).sort_values("validation_mae_mw")
    tuning.to_csv(RESULTS_DIR / "tft_tuning.csv", index=False)
    best = best_record
    test_loader = testing.to_dataloader(train=False, batch_size=best["batch_size"], num_workers=0)
    test_pred = predict_frame(best["model"], test_loader, df)
    test_mae = mean_absolute_error(test_pred.actual, test_pred.predicted)
    test_rmse = mean_squared_error(test_pred.actual, test_pred.predicted) ** 0.5

    best["val_predictions"].to_csv(RESULTS_DIR / "tft_validation_predictions.csv", index=False)
    test_pred.to_csv(RESULTS_DIR / "tft_test_predictions.csv", index=False)
    pd.DataFrame([
        {"dataset": "validation", "MAE_MW": best["validation_mae_mw"], "RMSE_MW": best["validation_rmse_mw"], "trial": best["name"]},
        {"dataset": "test", "MAE_MW": test_mae, "RMSE_MW": test_rmse, "trial": best["name"]},
    ]).to_csv(RESULTS_DIR / "tft_metrics.csv", index=False)
    save_curve(best["history"], RESULTS_DIR / "tft_training_curve.png")
    # Keep a stable, obvious checkpoint name as the saved model weight.
    Path(best["checkpoint"]).replace(RESULTS_DIR / "tft_best.ckpt")
    architecture = {k: v for k, v in best.items() if k in TRIALS[0]}
    architecture.update({"target": "load one hour ahead (MW)", "loss": "MAE", "optimizer": "Adam (TFT default)",
                         "encoder_hours": args.encoder_hours, "split": "chronological 70/15/15",
                         "device": accelerator})
    (RESULTS_DIR / "tft_selected_settings.json").write_text(json.dumps(architecture, indent=2), encoding="utf-8")
    print(f"\nBest trial: {best['name']} | val MAE={best['validation_mae_mw']:.2f} MW | test MAE={test_mae:.2f} MW")
    print(f"Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
