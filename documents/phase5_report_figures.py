"""Generate figures for the Phase 5 paper from checked-in experiment outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Held-out metrics copied from the saved CSV files; sample counts come from the
# cleaning report and selected LSTM settings JSON.
metrics = {
    "TFT": {"validation": (1566.804643, 3316.469378, 3694),
            "test": (2985.180968, 6272.787726, 3695)},
    "LSTM": {"validation": (1998.046745, 3762.900039, 3527),
             "test": (3567.733493, 6641.465927, 3528)},
    "XGBoost": {"validation": (3604.191197, 6188.099941, 3694),
                "test": (4619.619574, 7930.160182, 3695)},
}
models = list(metrics)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
x = np.arange(len(models))
width = 0.34
for ax, metric_i, title in [(axes[0], 0, "Mean absolute error"),
                            (axes[1], 1, "Root mean squared error")]:
    val = [metrics[m]["validation"][metric_i] for m in models]
    test = [metrics[m]["test"][metric_i] for m in models]
    ax.bar(x - width/2, val, width, color="#4b84a8", label="Validation")
    ax.bar(x + width/2, test, width, color="#e18b42", label="Test")
    ax.set_xticks(x, models)
    ax.set_ylabel("Error (MW)")
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", color="#dce2e8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
fig.suptitle("Model performance on the shared chronological partitions", x=0.06,
             ha="left", weight="bold", fontsize=12)
fig.text(0.06, 0.005,
         "Prediction rows: TFT/XGBoost 3,694 validation / 3,695 test; "
         "LSTM 3,527 / 3,528 (168-row sequence window).",
         fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.07, 1, 0.93))
fig.savefig(FIGURES / "phase5_model_metrics.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# LSTM epoch history: keep scaled training MSE and validation MAE on separate
# axes because they are in different units.
history = pd.read_csv(RESULTS / "lstm_training_history.csv")
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
axes[0].plot(history.epoch, history.train_mse_scaled, color="#146c94", linewidth=1.6)
axes[0].set_title("LSTM training loss", loc="left", weight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Scaled training MSE")
axes[0].grid(axis="y", color="#dce2e8", linewidth=0.6)
best = history.validation_mae_mw.idxmin()
axes[1].plot(history.epoch, history.validation_mae_mw, color="#146c94", linewidth=1.6)
axes[1].scatter(history.loc[best, "epoch"], history.loc[best, "validation_mae_mw"],
                color="#e18b42", zorder=3, label=f"Best epoch {int(history.loc[best, 'epoch'])}")
axes[1].set_title("LSTM validation error", loc="left", weight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation MAE (MW)")
axes[1].legend(frameon=False)
axes[1].grid(axis="y", color="#dce2e8", linewidth=0.6)
fig.tight_layout()
fig.savefig(FIGURES / "phase5_lstm_history.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Forecast illustration on a timestamp-aligned contiguous interval. XGBoost's
# saved predictions contain no timestamps, so it is intentionally omitted.
tft = pd.read_csv(RESULTS / "tft_test_predictions.csv", parse_dates=["timestamp"])
lstm = pd.read_csv(RESULTS / "lstm_test_predictions.csv", parse_dates=["timestamp"])
paired = tft[["timestamp", "actual", "predicted"]].merge(
    lstm[["timestamp", "actual", "predicted"]], on="timestamp",
    suffixes=("_tft", "_lstm"), how="inner").sort_values("timestamp").reset_index(drop=True)
if not np.allclose(paired.actual_tft, paired.actual_lstm, rtol=0, atol=1e-6):
    raise ValueError("TFT/LSTM actual values disagree on common timestamps.")
hour_step = paired.timestamp.diff().dt.total_seconds().to_numpy() == 3600
breaks = np.r_[0, np.where(~hour_step[1:])[0] + 1, len(paired)]
runs = [(int(a), int(b)) for a, b in zip(breaks[:-1], breaks[1:]) if b-a >= 72]
if not runs:
    raise ValueError("No common contiguous 72-hour test interval was found.")
start, end = max(runs, key=lambda ab: ab[1]-ab[0])
window = paired.iloc[start:start+72]
fig, ax = plt.subplots(figsize=(9.0, 3.4))
ax.plot(window.timestamp, window.actual_tft, color="#333942", linewidth=1.5,
        label="Observed demand")
ax.plot(window.timestamp, window.predicted_tft, color="#146c94", linewidth=1.25,
        label="TFT")
ax.plot(window.timestamp, window.predicted_lstm, color="#e18b42", linewidth=1.25,
        label="LSTM")
ax.set_title("Timestamp-aligned example: contiguous 72-hour test window", loc="left", weight="bold")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Demand (MW)")
ax.grid(axis="y", color="#dce2e8", linewidth=0.6)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(FIGURES / "phase5_test_forecasts.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Generated Phase 5 report figures:")
for name in ("phase5_model_metrics.png", "phase5_lstm_history.png", "phase5_test_forecasts.png"):
    print(FIGURES / name)
print(f"Aligned forecast segment: {window.timestamp.iloc[0]} to {window.timestamp.iloc[-1]}")
