# GridShift: Hourly Electricity Demand Forecasting

GridShift is a research project that forecasts India's hourly electricity demand from historical demand, solar generation, wind generation, and calendar patterns. It compares a Temporal Fusion Transformer (TFT) with a Long Short-Term Memory network (LSTM) using a chronological holdout, and studies how forecast error changes in later observations.

This repository contains data-audit and preprocessing materials, model-training scripts, experiment outputs, and the Phase 1-4 research report. It is an offline research prototype; it is not intended for grid dispatch or other operational decisions.

## Research question

How accurately can sequence models forecast electricity demand one hour ahead, and how does accuracy differ between validation data and a later held-out test period?

The study is observational. Relationships between demand and renewable generation do not, by themselves, show that solar or wind generation caused demand changes.

## Current comparable results

TFT and LSTM use the same modeling file, target, one-hour forecast horizon, and chronological split: **25,678 complete rows**, divided into **17,974 training / 3,852 validation / 3,852 test rows**. Values are in megawatts (MW).

| Model | Split | MAE (MW) | RMSE (MW) | Selected configuration | Device |
|---|---|---:|---:|---|---|
| TFT | Validation | 1,874.62 | 3,616.16 | Larger configuration; selected by validation MAE | GPU |
| TFT | Test | 3,542.41 | 6,610.82 | Held-out evaluation | GPU |
| LSTM | Validation | 2,128.01 | 3,770.96 | Small configuration; selected by validation MAE | CPU |
| LSTM | Test | 4,258.34 | 6,929.33 | Held-out evaluation | CPU |

On this split, TFT has lower reported MAE and RMSE on both validation and test data. The LSTM run lasted 30 epochs with a 168-hour lookback; its best validation MAE occurred at epoch 27. Training MAE and RMSE were not recorded for that run. Its history contains scaled training MSE and validation MAE instead.

The legacy XGBoost experiment is **excluded from this comparison**. The cleaning report in the supplied project archive records 12,984 rows for that experiment, so its scores are not directly comparable to the 25,678-row TFT and LSTM results.

## Dataset

The project uses the hourly India electricity-demand, solar-generation, and wind-generation dataset by Mukherjee, Kalita, and Kumar (2025), Mendeley Data, Version 2. The dataset record describes observations from the combined five regional grid networks.

- DOI: <https://doi.org/10.17632/y58jknpgs8.2>
- Dataset page: <https://data.mendeley.com/datasets/y58jknpgs8/2>
- License listed by the dataset record: CC BY 4.0

The dataset record has a date-range discrepancy: its title and file list refer to September 2021 through June 2025, while its description text refers to September 2021 through December 2023. Confirm coverage from workbook timestamps when reproducing the raw-data workflow.

### Modeling data

The ready-to-run LSTM script reads:

```text
outputs/processed/grid_hourly_merged.csv
```

The script drops rows without a valid timestamp, demand, solar, or wind value, then checks that the resulting data has exactly 25,678 rows and the expected split. It stops if the data does not match. Keep the CSV used for the TFT experiment unchanged when reproducing this comparison.

## Methods

- **Forecast target:** Demand (MW) one hour ahead.
- **Input window:** the preceding 168 hours (seven days).
- **Split:** chronological 70/15/15; no random shuffling across time periods.
- **Model selection:** validation MAE. The test set is held out for final evaluation.
- **LSTM inputs:** demand, solar, wind, and sine/cosine encodings for hour of day, weekday, and month.
- **Scaling:** the LSTM script estimates feature and target means and standard deviations from training rows only.
- **Metrics:** MAE and RMSE in MW. Classification accuracy is not applicable to this regression task.

At each one-hour forecast, the LSTM uses historical observations up to the forecast origin. Later measured values are used only as history for subsequent one-hour-ahead forecasts, consistent with rolling one-step-ahead evaluation.

## Repository layout

```text
.
├── src/
│   ├── data/
│   │   └── prepare_data.py
│   └── models/
│       ├── lstm_model.py
│       ├── tft_model.py
│       └── xgboost_model.py
├── outputs/
│   ├── processed/
│   │   └── grid_hourly_merged.csv
│   └── results/
├── figures/
├── audit_india_dataset.py
├── clean_india_grid.py
├── gridshift_phase1_phase4.tex
└── gridshift_phase3_4_report.pdf
```

The precise files included may vary by project checkout. The XGBoost script is retained as a project artifact, but its legacy run is not part of the current TFT/LSTM findings.

## Run the LSTM

Run commands from the project root, the folder containing `src/` and `outputs/`.

### Install dependencies

```powershell
python -m pip install pandas numpy scikit-learn torch
```

For GPU training, install the PyTorch build that matches your CUDA and driver setup using the official PyTorch installation instructions. Check whether PyTorch can see CUDA with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Train and evaluate

```powershell
python src/models/lstm_model.py
```

The script uses CUDA automatically when available. To force CPU:

```powershell
python src/models/lstm_model.py --cpu
```

Optional parameters:

```powershell
python src/models/lstm_model.py --lookback-hours 168 --max-epochs 30 --patience 5 --seed 42
```

The default run trains three LSTM configurations and selects the best one using validation MAE. It then evaluates that selected configuration on the test split.

## LSTM output files

The script writes experiment artifacts to `outputs/results/`:

| File | Contents |
|---|---|
| `lstm_metrics.csv` | Validation and test MAE/RMSE in MW |
| `lstm_tuning.csv` | Validation results for each tested configuration |
| `lstm_selected_settings.json` | Selected configuration, device, row counts, and split sizes |
| `lstm_training_history.csv` | Epoch-level scaled training MSE and validation MAE |
| `lstm_validation_predictions.csv` | Actual and predicted validation demand |
| `lstm_test_predictions.csv` | Actual and predicted test demand |
| `lstm_best.pt` | Selected model weights and scaling parameters |

For a reported run, preserve the metrics, settings, prediction files, and training history together. The run history supplied for the results above does not contain training MAE/RMSE.

## Run the TFT

The TFT training script is `src/models/tft_model.py`. It uses the same merged CSV by default, automatically chooses GPU when available, and writes its selected settings, metrics, predictions, tuning results, checkpoint, and training curve under `outputs/results/`.

Install compatible PyTorch Lightning and PyTorch Forecasting dependencies for the local PyTorch version, then run:

```powershell
python src/models/tft_model.py
```

Use `--cpu` to force CPU or `--data path\to\grid_hourly_merged.csv` to provide a different input file. The TFT run evaluates several configurations, selects by validation MAE, and evaluates the selected model on the test period.

## Raw-data audit and preprocessing

The repository also includes scripts for auditing and cleaning the original Excel workbooks. The workbook workflow requires the raw dataset files and the configured sheet, header, timestamp, and measurement columns to match the actual workbook structure. Inspect the audit report before running the cleaner; do not assume workbook headers or sheet layouts.

Typical setup commands are:

```powershell
python -m pip install pandas numpy openpyxl matplotlib seaborn scikit-learn joblib
python audit_india_dataset.py --input-dir data/raw --output-dir outputs/audit
python clean_india_grid.py --input-dir data/raw --output-dir outputs/processed
```

The cleaned dataset used for the reported model comparison is already provided at `outputs/processed/grid_hourly_merged.csv` in the project archive. Re-running raw preprocessing may create a different row set; if so, the LSTM script's row-count guard will stop rather than silently run a non-matching comparison.

## Research report

- `gridshift_phase1_phase4.tex` is the editable LaTeX source covering project scope, data, models, and results.
- `gridshift_phase3_4_report.pdf` is the rendered report.
- `figures/tft_training_curve.png` is used by the LaTeX source for the TFT training curve.

Compile the LaTeX source from the project root after placing the TFT curve image in `figures/`:

```powershell
pdflatex gridshift_phase1_phase4.tex
pdflatex gridshift_phase1_phase4.tex
```

The LSTM epoch curves are defined in the LaTeX source from the run history. The contribution table lists project modules and implementation paths.

## References

Mukherjee, D., Kalita, K., & Kumar, S. (2025). *Electricity Demand, Solar and Wind Generation Data (September 2021-June 2025) of India at 1-hour interval* (Version 2). Mendeley Data. <https://doi.org/10.17632/y58jknpgs8.2>.
