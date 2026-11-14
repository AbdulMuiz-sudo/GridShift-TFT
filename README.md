# GridShift Electricity Demand Forecasting

GridShift compares one-hour-ahead forecasts of India electricity demand using XGBoost, a Long Short-Term Memory network (LSTM), and a Temporal Fusion Transformer (TFT). This is an offline research prototype, not an operational grid-control system.

## Current dataset and split

All three models load the same prepared files through `src/data/prepare_data.py`:

- `outputs/processed/train_scaled.csv`: 18,022 rows
- `outputs/processed/val_scaled.csv`: 3,862 rows
- `outputs/processed/test_scaled.csv`: 3,863 rows
- Total prepared rows: 25,747

The current source is `outputs/processed/grid_hourly_merged.csv` (37,392 hourly timestamps). `cleaneddata.py` resamples hourly, interpolates internal gaps up to three hours, drops rows with unresolved missing demand, creates cyclical calendar features, and writes chronological 70/15/15 splits. The scaler is fit on train only. The previous 12,984-row audit summary described an unrelated workbook; the corrected current cleaning record is `outputs/processed/audit/cleaning_report.json` and its readable companion is `outputs/processed/audit/cleaning_report.md`.

After target, lag, and rolling features are created, model-ready counts are 17,854 train, 3,694 validation, and 3,695 test. The LSTM also requires a 168-row sequence and therefore predicts 17,687 / 3,527 / 3,528 samples. Its validation and test metrics cover fewer, later timestamps than TFT/XGBoost; keep that sample-count difference in mind when comparing scores.

The retained validation timestamps contain 4 long gaps and the test timestamps contain 19. Targets use the next retained row, so the few labels immediately before these gaps are not exactly one clock hour ahead. The model reports therefore describe a nominal one-hour forecast with these gap cases disclosed.

## Latest model results

MAE and RMSE are in MW. Lower is better. Model selection uses validation MAE; test remains held out.

| Model | Split | MAE | RMSE | Selected trial |
|---|---|---:|---:|---|
| TFT | Validation | 1,566.80 | 3,316.47 | larger |
| TFT | Test | 2,985.18 | 6,272.79 | larger |
| LSTM | Validation | 1,998.05 | 3,762.90 | balanced |
| LSTM | Test | 3,567.73 | 6,641.47 | balanced |
| XGBoost | Validation | 3,604.19 | 6,188.10 | depth 6, lr 0.05, 200 trees |
| XGBoost | Test | 4,619.62 | 7,930.16 | selected on validation MAE |

TFT has the lowest reported MAE and RMSE on both periods; LSTM is second and XGBoost is third. These scores use the same source splits, but LSTM scores use fewer prediction timestamps because of its sequence window.

## Modeling approach

The forecast is the next hour's demand, reconstructed as current demand plus the predicted one-hour demand change. Calendar sine/cosine terms, solar, wind, load lags, and rolling load means provide forecast-available features.

- **XGBoost:** `XGBRegressor`, squared-error objective, tuned on eight combinations of max depth (4/6), learning rate (0.05/0.10), and trees (200/400); subsample and column sample each 0.8.
- **LSTM:** PyTorch `nn.LSTM` plus a linear regression head. It uses a 168-hour lookback, MSE on training-scaled demand changes, Adam, gradient clipping at 1.0, and validation MAE for early stopping and selection.
- **TFT:** PyTorch Forecasting `TemporalFusionTransformer`, 168-hour encoder, MAE loss, Adam, gradient clipping, and validation MAE selection.

For LSTM, three settings were tested: 32 hidden units/one layer/no dropout/lr 0.001; 64/one/no dropout/lr 0.001; and 64/two/dropout 0.1/lr 0.0005. All use batch size 128. The balanced 64-unit, one-layer trial was selected; training stopped after 16 epochs and the best checkpoint was epoch 11.

For TFT, four configurations were tested. The selected larger trial uses hidden size 48, four attention heads, hidden continuous size 24, dropout 0.15, learning rate 0.002, and batch size 64.

## Evaluation and error analysis

This is continuous-value regression. MAE gives average absolute error in MW and is easy to interpret; RMSE penalizes large misses more heavily. Precision, recall, F1, and ROC-AUC require class labels or a defined threshold and are not appropriate for the continuous-demand target. MAPE is reported for XGBoost as a secondary percentage measure.

On LSTM predictions, mean signed error (prediction minus actual) is -114.50 MW on validation and -208.54 MW on test, indicating slight average underprediction. Test MAE by actual-demand quartile is highest in the lowest-demand quartile (4,365 MW); the model tends to overpredict that group and underpredict the highest-demand quartile. Hour 23 has the largest LSTM MAE: 6,241 MW on validation and 12,063 MW on test. TFT predictions show the same late-hour challenge, with test MAE 12,640 MW at hour 23. This is a targeted bias check by demand level and time of day; the dataset has no demographic groups to test.

All models have higher errors on the later test period than validation. This supports caution about temporal distribution shift; one holdout does not establish causation or guarantee future performance.

## Run the pipeline

Install dependencies from `requirements.txt`. Choose a PyTorch build compatible with your hardware/CUDA setup if you plan to use GPU training.

```powershell
python cleaneddata.py --input-file outputs/processed/grid_hourly_merged.csv --output-dir outputs/processed
python src/models/xgboost_model.py
python src/models/lstm_model.py
python src/models/tft_model.py
```

Run `cleaneddata.py` first only when intentionally rebuilding the shared splits. Running it overwrites the three model input files and invalidates results from previous split versions. The model scripts write metrics, settings, tuning results, predictions, and checkpoints to `outputs/results/`.

## Phase 3 and 4 report

- `gridshift_phase1_phase4.tex`: editable report source.
- `gridshift_phase3_4_report.pdf`: Phase 3/4 architecture, tuning, evaluation, curves, error analysis, and Git contribution table.

## Git contribution map

The table in the report follows Git commit authors exactly as recorded in this repository. Key commits include `1e9fc02` and `f7f866b` (Abdullah Aftab: shared data preparation and XGBoost), `8daf0c4` and `f6c3cdb` (Abdul Muiz: TFT), and `7c7db34` and `014b6c1` (WikidSmellz: LSTM results/model updates and prior report). The account name `WikidSmellz` is shown as Git records it; it is not expanded into a person’s name. This report and cleaning-report refresh are not yet in Git history and should be committed under the actual contributor account(s).

## Dataset citation

Mukherjee, D., Kalita, K., & Kumar, S. (2025). *Electricity Demand, Solar and Wind Generation Data (September 2021-June 2025) of India at 1-hour interval* (Version 2). Mendeley Data. [doi:10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2).
