"""
GridShift TFT — configurable preprocessing for India's hourly electricity data.

IMPORTANT:
1. The workbook audit has been inspected.
2. This version reads only the hourly Report sheet in January 2024- June 2025.xlsx.
3. Older monthly workbooks are intentionally excluded because their sampling intervals
   and column layouts differ from this hourly table.
4. The script does not claim to prove solar adoption caused a shift.
5. Outliers are flagged by default, not deleted/clipped, because real demand peaks
   may be valid observations.

Install:
    pip install pandas numpy openpyxl matplotlib seaborn scikit-learn joblib

Example:
    python clean_india_grid.py --input-dir data/raw --output-dir outputs/processed
"""
from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

# Exact names observed in workbook_audit.json for the hourly Report sheet.
TIMESTAMP_COL = "Timestamp"
LOAD_COL = "Demand (MW)"
SOLAR_COL = "Solar (MW)"
WIND_COL = "Wind (MW)"
SHEET_NAME = "Report"
HEADER_ROW = 0
DATA_FILE = "January 2024- June 2025.xlsx"
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"
MAX_SHORT_GAP_TO_INTERPOLATE = 2
TEST_FRACTION = 0.20
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_POLICY = "flag"  # "flag" or "clip"; prefer "flag" for the first analysis

def read_all_excel(input_dir, sheet_name=SHEET_NAME, header=HEADER_ROW):
    """Read only the audited hourly Report sheet, not incompatible older workbooks."""
    path = Path(input_dir) / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.resolve()}. Put '{DATA_FILE}' inside the data/raw folder."
        )
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read sheet '{sheet_name}' from {path.name}: {exc}"
        ) from exc
    df["__source_file"] = path.name
    print(f"Read {path.name} | sheet={sheet_name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df

def require_config():
    missing = [name for name, value in {
        "TIMESTAMP_COL": TIMESTAMP_COL,
        "LOAD_COL": LOAD_COL,
        "SOLAR_COL": SOLAR_COL,
    }.items() if not value]
    if missing:
        raise SystemExit(
            "Configure these exact column names after running audit_india_dataset.py: "
            + ", ".join(missing)
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/processed")
    args = parser.parse_args()
    require_config()

    out = Path(args.output_dir)
    figdir = out / "figures"
    auditdir = out / "audit"
    figdir.mkdir(parents=True, exist_ok=True)
    auditdir.mkdir(parents=True, exist_ok=True)

    raw = read_all_excel(args.input_dir)
    # Remove completely empty rows/columns; retain source-file provenance for audit.
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    raw_columns = [str(c) for c in raw.columns]
    (auditdir / "columns_seen.json").write_text(
        json.dumps(raw_columns, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    required = [TIMESTAMP_COL, LOAD_COL, SOLAR_COL]
    if WIND_COL:
        required.append(WIND_COL)
    absent = [c for c in required if c not in raw.columns]
    if absent:
        raise SystemExit(f"Configured columns not found: {absent}. See outputs/audit/columns_seen.json")

    # Preserve a raw copy for auditing; standardize only the selected analysis columns.
    selected = [TIMESTAMP_COL, LOAD_COL, SOLAR_COL] + ([WIND_COL] if WIND_COL else [])
    if "__source_file" in raw.columns:
        selected.append("__source_file")
    df = raw[selected].copy()
    df = df.rename(columns={
        TIMESTAMP_COL: "timestamp",
        LOAD_COL: "load",
        SOLAR_COL: "solar",
        **({WIND_COL: "wind"} if WIND_COL else {}),
    })

    # Parse timestamps; unparseable timestamps are reported and excluded.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    bad_timestamp_count = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"]).copy()

    numeric_cols = ["load", "solar"] + (["wind"] if "wind" in df.columns else [])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert common textual placeholders to missing values via numeric coercion above.
    # Duplicate timestamp rows are reported. If the source contains regional rows,
    # do NOT simply drop them: add a region key and define an aggregation first.
    duplicate_timestamp_rows = int(df.duplicated(subset=["timestamp"], keep="first").sum())
    if duplicate_timestamp_rows:
        # The default is safe only for a single national aggregate series.
        # Stop to avoid silently discarding regional records.
        raise SystemExit(
            f"Found {duplicate_timestamp_rows} repeated timestamps. "
            "Check whether rows represent different regions/series. Add a region key "
            "and aggregate deliberately before removing duplicates."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    missing_before = df[numeric_cols].isna().sum().to_dict()

    # EDA on raw selected data before filling.
    missing_plot = df[numeric_cols].isna().mean().mul(100)
    plt.figure(figsize=(8, 4))
    missing_plot.plot(kind="bar")
    plt.ylabel("Missing values (%)")
    plt.title("Missingness in selected raw variables")
    plt.tight_layout()
    plt.savefig(figdir / "raw_missingness.png", dpi=160)
    plt.close()

    df[numeric_cols].hist(figsize=(10, 6), bins=40)
    plt.suptitle("Raw variable distributions")
    plt.tight_layout()
    plt.savefig(figdir / "raw_distributions.png", dpi=160)
    plt.close()

    if len(numeric_cols) >= 2:
        plt.figure(figsize=(7, 5))
        sns.heatmap(df[numeric_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Raw-variable correlation (descriptive, not causal)")
        plt.tight_layout()
        plt.savefig(figdir / "raw_correlation.png", dpi=160)
        plt.close()

    plt.figure(figsize=(12, 5))
    for col in numeric_cols:
        plt.plot(df["timestamp"], df[col], label=col, linewidth=0.7)
    plt.xlabel("Timestamp")
    plt.ylabel("Value in source units")
    plt.title("Raw hourly series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "raw_time_series.png", dpi=160)
    plt.close()

    # Fill only short internal gaps by time interpolation; longer gaps remain missing.
    df = df.set_index("timestamp")
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit=MAX_SHORT_GAP_TO_INTERPOLATE, limit_area="inside"
    )
    missing_after_interpolation = df[numeric_cols].isna().sum().to_dict()

    # Outlier audit using IQR bounds. Flag first; do not automatically delete true peaks.
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        flag = (df[col] < lower) | (df[col] > upper)
        df[f"{col}_outlier_flag"] = flag.fillna(False)
        outlier_summary[col] = {
            "q1": None if pd.isna(q1) else float(q1),
            "q3": None if pd.isna(q3) else float(q3),
            "iqr_lower_bound": None if pd.isna(lower) else float(lower),
            "iqr_upper_bound": None if pd.isna(upper) else float(upper),
            "flagged_count": int(flag.sum()),
            "policy": OUTLIER_POLICY,
        }
        if OUTLIER_POLICY == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)

    # Keep rows only when forecasting target is available. Missing feature values
    # can be handled downstream, but report them explicitly.
    before_target_drop = len(df)
    df = df.dropna(subset=["load"])
    dropped_missing_target = before_target_drop - len(df)

    # Time features: cyclical encodings avoid treating hour 23 and hour 0 as far apart.
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df = df.reset_index()
    # Chronological split; fit scaler on training data only to prevent leakage.
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_at = int(len(df) * (1 - TEST_FRACTION))
    if split_at < 1 or split_at >= len(df):
        raise SystemExit("Not enough rows for configured chronological split.")
    train = df.iloc[:split_at].copy()
    test = df.iloc[split_at:].copy()

    scale_cols = [c for c in numeric_cols if c in train.columns]
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[[f"{c}_scaled" for c in scale_cols]] = scaler.fit_transform(train[scale_cols])
    test_scaled[[f"{c}_scaled" for c in scale_cols]] = scaler.transform(test[scale_cols])
    joblib.dump(scaler, out / "standard_scaler_train_only.joblib")

    df.to_csv(out / "cleaned_hourly_data.csv", index=False)
    train_scaled.to_csv(out / "train_scaled.csv", index=False)
    test_scaled.to_csv(out / "test_scaled.csv", index=False)

    report = {
        "source_files": [DATA_FILE],
        "source_sheet": SHEET_NAME,
        "timestamp_format": TIMESTAMP_FORMAT,
        "configured_columns": {"timestamp": TIMESTAMP_COL, "load": LOAD_COL,
                               "solar": SOLAR_COL, "wind": WIND_COL},
        "rows_after_read_and_empty_row_removal": int(len(raw)),
        "rows_after_timestamp_parse": int(len(df)),
        "unparseable_timestamp_rows": bad_timestamp_count,
        "duplicate_timestamp_rows_detected": duplicate_timestamp_rows,
        "missing_before_interpolation": {k: int(v) for k, v in missing_before.items()},
        "missing_after_short_interpolation": {k: int(v) for k, v in missing_after_interpolation.items()},
        "rows_dropped_for_missing_load": int(dropped_missing_target),
        "final_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_method": "chronological; final 20% held out",
        "scaler": "StandardScaler fit on training rows only, then applied to test rows",
        "outlier_summary": outlier_summary,
        "notes": [
            "No real-data results are asserted in this script.",
            "IQR flags are screening indicators, not proof of data errors.",
            "Repeated timestamps cause a stop to prevent accidental loss of regional records.",
            "If regional data are present, define region handling and aggregation before modelling."
        ],
    }
    (auditdir / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nOutputs written to {out.resolve()}")
    print(f"Charts saved to {figdir.resolve()}")

if __name__ == "__main__":
    main()
