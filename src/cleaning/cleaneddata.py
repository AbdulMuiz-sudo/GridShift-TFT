"""
GridShift TFT — Clean, Split & Scale.
Consumes the pre-processed 'grid_hourly_merged.csv', enforces strict hourly continuity,
cleans missing data, engineers time features, splits chronologically (70/15/15),
and scales without data leakage.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
MAX_INTERPOLATION_GAP = 3  # Maximum consecutive missing hours to interpolate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", default="outputs/processed/grid_hourly_merged.csv"
    )
    parser.add_argument("--output-dir", default="outputs/processed")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading merged dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(
            f"[!] Could not find {input_path}. Ensure your first script ran successfully."
        )
        return

    source_rows = int(len(df))

    # 1. STANDARDIZE & PREPARE
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col_map = {"Demand (MW)": "load", "Solar (MW)": "solar", "Wind (MW)": "wind"}
    df = df.rename(columns=col_map)
    numeric_cols = ["load", "solar", "wind"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. DATA CLEANING & CONTINUITY ENFORCEMENT
    print("Cleaning data and enforcing continuous hourly index...")

    df = df.dropna(subset=["timestamp"])
    rows_after_timestamp_parse = int(len(df))
    duplicate_timestamps = int(df.duplicated(subset=["timestamp"]).sum())

    # Drop exact duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    # Set index to resample and enforce strict 1-hour frequency
    df = df.set_index("timestamp")
    df = df.resample("1H").mean()
    rows_after_hourly_resample = int(len(df))

    missing_before_interpolation = {
        col: int(value) for col, value in df[numeric_cols].isna().sum().items()
    }

    # Interpolate short missing gaps (e.g., 1 to 3 hours)
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit=MAX_INTERPOLATION_GAP, limit_area="inside"
    )

    missing_after_interpolation = {
        col: int(value) for col, value in df[numeric_cols].isna().sum().items()
    }

    # Drop rows where target variable (load) is still missing after interpolation
    df = df.dropna(subset=["load"]).reset_index()
    final_rows = int(len(df))

    # 3. FEATURE ENGINEERING
    print("Generating cyclical time features...")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Sine/Cosine transforms for continuous cyclical representation
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # 4. CHRONOLOGICAL SPLIT
    print("Splitting dataset chronologically (70/15/15)...")
    train_end = int(len(df) * TRAIN_FRACTION)
    val_end = train_end + int(len(df) * VAL_FRACTION)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    non_hourly_gaps = {
        name: int((part["timestamp"].diff().dropna() != pd.Timedelta(hours=1)).sum())
        for name, part in (("train", train), ("validation", val), ("test", test))
    }

    # 5. SCALING (Fitted only on train to prevent look-ahead bias)
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    train_scaled[[f"{c}_scaled" for c in numeric_cols]] = scaler.fit_transform(
        train[numeric_cols]
    )
    val_scaled[[f"{c}_scaled" for c in numeric_cols]] = scaler.transform(
        val[numeric_cols]
    )
    test_scaled[[f"{c}_scaled" for c in numeric_cols]] = scaler.transform(
        test[numeric_cols]
    )

    # Save the scaler for inference
    joblib.dump(scaler, out_dir / "standard_scaler_train_only.joblib")

    # Export finalized datasets
    train_scaled.to_csv(out_dir / "train_scaled.csv", index=False)
    val_scaled.to_csv(out_dir / "val_scaled.csv", index=False)
    test_scaled.to_csv(out_dir / "test_scaled.csv", index=False)

    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cleaning_report = {
        "report_version": 2,
        "dataset": "GridShift shared modeling dataset",
        "source_file": str(input_path),
        "source_rows": source_rows,
        "rows_after_timestamp_parse": rows_after_timestamp_parse,
        "duplicate_timestamp_rows_detected_and_dropped": duplicate_timestamps,
        "rows_after_hourly_resample": rows_after_hourly_resample,
        "missing_before_interpolation": missing_before_interpolation,
        "max_interpolation_gap_hours": MAX_INTERPOLATION_GAP,
        "missing_after_short_interpolation": missing_after_interpolation,
        "rows_dropped_for_missing_load": rows_after_hourly_resample - final_rows,
        "final_rows": final_rows,
        "date_start": str(df["timestamp"].min()),
        "date_end": str(df["timestamp"].max()),
        "split_method": "chronological; 70% train, 15% validation, 15% test",
        "split_rows": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "non_hourly_timestamp_gaps_by_split": non_hourly_gaps,
        "split_files": [
            "outputs/processed/train_scaled.csv",
            "outputs/processed/val_scaled.csv",
            "outputs/processed/test_scaled.csv",
        ],
        "scaler": "StandardScaler fit on training rows only; applied to validation and test",
        "scaled_columns": ["load_scaled", "solar_scaled", "wind_scaled"],
        "time_features": [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
        ],
        "outlier_policy": "No rows are removed by outlier thresholds in this preparation script.",
        "target_horizon_note": "Model target creation shifts to the next retained row. When adjacent timestamps are separated by a long unresolved gap, that label is not exactly one clock hour ahead.",
        "note": "This report describes the shared split files consumed by TFT, XGBoost, and LSTM. It supersedes the unrelated 12,984-row workbook report.",
    }
    (audit_dir / "cleaning_report.json").write_text(
        json.dumps(cleaning_report, indent=2), encoding="utf-8"
    )

    print("\n===================================")
    print("DATASET PREPARATION COMPLETE")
    print("===================================")
    print(f"Total valid hours: {len(df)}")
    print(f"Train set: {len(train)} rows")
    print(f"Validation set: {len(val)} rows")
    print(f"Test set: {len(test)} rows")
    print(f"\nFiles saved directly to {out_dir}")


if __name__ == "__main__":
    main()
