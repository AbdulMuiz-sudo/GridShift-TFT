# Cleaning and split report

## Dataset used by the models

The current model run starts from `outputs/processed/grid_hourly_merged.csv`. It contains 37,392 hourly timestamps from 2021-09-01 00:00 through 2025-12-06 23:00. The old 12,984-row report described a different workbook and does not describe this modeling run.

## Cleaning and preparation

`cleaneddata.py` sorts and deduplicates timestamps, resamples to hourly frequency, interpolates internal gaps of up to three hours, and removes rows whose demand is still missing. It creates calendar sine/cosine features and splits chronologically. The numeric scaler is fit only on training rows and then applied to validation and test rows.

| Stage | Count |
|---|---:|
| Merged input timestamps | 37,392 |
| Rows after hourly resampling | 37,392 |
| Missing values before short interpolation, per measurement | 11,714 |
| Missing values after interpolation, per measurement | 11,645 |
| Rows retained after unresolved demand gaps are removed | 25,747 |

## Shared model splits

| Split | Rows | File |
|---|---:|---|
| Train | 18,022 | `train_scaled.csv` |
| Validation | 3,862 | `val_scaled.csv` |
| Test | 3,863 | `test_scaled.csv` |
| **Total** | **25,747** | |

Retained split timestamps are hourly except for 4 gaps in validation and 19 in test. Because the model scripts form the target with a one-row shift, labels immediately before those gaps refer to the next retained observation rather than exactly one clock hour later. The report records these cases so the nominal one-hour horizon is interpreted accurately.

After the shared one-hour target and 168-hour lag/rolling features are formed, model-ready rows are 17,854 train, 3,694 validation, and 3,695 test. The LSTM additionally uses a 168-row input sequence and reports 17,687, 3,527, and 3,528 predictions, respectively.

`cleaneddata.py` now writes these counts to `outputs/processed/audit/cleaning_report.json` each time the shared splits are regenerated.
