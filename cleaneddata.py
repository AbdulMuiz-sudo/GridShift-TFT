"""
GridShift TFT — Clean, Split & Scale.
Consumes the pre-processed 'grid_hourly_merged.csv', enforces strict hourly continuity,
cleans missing data, engineers time features, splits chronologically (70/15/15),
and scales without data leakage.
"""

import argparse
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

    # Drop exact duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    # Set index to resample and enforce strict 1-hour frequency
    df = df.set_index("timestamp")
    df = df.resample("1H").mean()

    # Interpolate short missing gaps (e.g., 1 to 3 hours)
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit=MAX_INTERPOLATION_GAP, limit_area="inside"
    )

    # Drop rows where target variable (load) is still missing after interpolation
    df = df.dropna(subset=["load"]).reset_index()

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
