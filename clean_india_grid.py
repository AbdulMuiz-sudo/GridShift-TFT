import pandas as pd
import numpy as np
import json
import ast
import argparse
import warnings
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
SCADA_TAG_MAP = {
    "SCADA/ANALOG/044MQ067/0": "Demand (MW)",
    "SCADA/ANALOG/044MQ206/0": "Solar (MW)",
    "SCADA/ANALOG/044MQ070/0": "Wind (MW)",
}

STANDARD_COLUMNS = ["Demand (MW)", "Solar (MW)", "Wind (MW)"]

TIME_ALIASES = ["time", "timestamp", "datetime", "date", "date time"]

VALUE_ALIASES = {
    "demand (mw)": "Demand (MW)",
    "load (mw)": "Demand (MW)",
    "demand": "Demand (MW)",
    "load": "Demand (MW)",
    "solar (mw)": "Solar (MW)",
    "solar": "Solar (MW)",
    "wind (mw)": "Wind (MW)",
    "wind": "Wind (MW)",
}

MAX_GAP = 2


# ==========================================
# JSON CLEANING
# ==========================================


def clean_cell(value):

    text = str(value).strip()
    text = text.replace("\ufeff", "")

    for i in range(2):
        if len(text) >= 2:
            if text[0] == text[-1] and text[0] in ["'", '"']:
                text = text[1:-1].strip()

    return text


def decode_json(value):

    if not isinstance(value, str):
        return None

    text = clean_cell(value)

    if not text or text[0] not in ["{", "["]:
        return None

    candidates = [text, text.replace('\\"', '"')]

    for candidate in candidates:

        try:
            data = json.loads(candidate, strict=False)

            for i in range(2):
                if isinstance(data, str):
                    data = json.loads(data, strict=False)

            return data

        except (json.JSONDecodeError, TypeError):
            pass

    try:
        return ast.literal_eval(text)

    except (ValueError, SyntaxError):
        return None


# ==========================================
# EXTRACT SCADA RECORDS
# ==========================================


def extract_records(data):

    records = []

    if isinstance(data, list):

        for item in data:
            records.extend(extract_records(item))

    elif isinstance(data, dict):

        tag = data.get("ck", data.get("tag", data.get("Tag", data.get("name"))))

        timestamp = data.get(
            "ts", data.get("timestamp", data.get("time", data.get("Time")))
        )

        value = data.get("v", data.get("value", data.get("Value")))

        if tag is not None and timestamp is not None and value is not None:

            records.append(
                {"tag": str(tag).strip(), "timestamp": timestamp, "value": value}
            )

        for child in data.values():

            if isinstance(child, (dict, list)):
                records.extend(extract_records(child))

    return records


# ==========================================
# READ EXCEL WORKBOOK
# ==========================================


def read_workbook(path):

    excel = pd.ExcelFile(path)

    preferred = ["Report", "Sheet1"]

    sheets = [s for s in preferred if s in excel.sheet_names]

    sheets += [s for s in excel.sheet_names if s not in sheets]

    for sheet in sheets:

        df = pd.read_excel(excel, sheet_name=sheet, header=None, dtype=object)

        yield sheet, df


# ==========================================
# DETECT TABULAR FORMAT
# ==========================================


def detect_tabular(path):

    for sheet, raw in read_workbook(path):

        if raw.empty:
            continue

        for row in range(min(6, len(raw))):

            headers = [
                str(x).strip().lower() if pd.notna(x) else "" for x in raw.iloc[row]
            ]

            if any(h in TIME_ALIASES for h in headers):

                return sheet, raw, row

    return None, None, None


# ==========================================
# PROCESS TABULAR DATA
# ==========================================


def process_tabular(raw, path, header_row):

    headers = raw.iloc[header_row].tolist()

    description_row = None
    data_start = header_row + 1

    # Check whether the next row contains measurement descriptions

    if data_start < len(raw):

        next_row = raw.iloc[data_start].tolist()

        first_value = str(next_row[0]).strip().lower()

        if first_value not in TIME_ALIASES:

            description_row = next_row
            data_start += 1

    selected_columns = {}

    for i, header in enumerate(headers):

        if pd.isna(header):
            continue

        header = str(header).strip()
        normalized = header.lower()

        # Identify timestamp

        if normalized in TIME_ALIASES:

            selected_columns[i] = "timestamp"
            continue

        # Match actual SCADA tag IDs

        if header in SCADA_TAG_MAP:

            selected_columns[i] = SCADA_TAG_MAP[header]
            continue

        # Match standard tabular column names

        if normalized in VALUE_ALIASES:

            selected_columns[i] = VALUE_ALIASES[normalized]
            continue

        # Match descriptions from the second header row

        if description_row is not None:

            description = str(description_row[i]).strip().upper()

            if description == "NLDC_DEMAND|P":
                selected_columns[i] = "Demand (MW)"

            elif description == "ALL_IND_SOLAR|P":
                selected_columns[i] = "Solar (MW)"

            elif description == "ALL_INDIA_WIND|P":
                selected_columns[i] = "Wind (MW)"

    if "timestamp" not in selected_columns.values():

        raise ValueError("Timestamp column not found")

    # Extract actual data rows

    data = raw.iloc[data_start:].copy()

    result = pd.DataFrame()

    for index, standard_name in selected_columns.items():

        result[standard_name] = data.iloc[:, index].values

    # Ensure all required columns exist

    for col in STANDARD_COLUMNS:

        if col not in result.columns:
            result[col] = np.nan

    # Clean timestamps

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    # Convert measurements to numeric

    for col in STANDARD_COLUMNS:

        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(subset=["timestamp"])

    print("\nProcessed:", path.name)
    print("Rows extracted:", len(result))

    return result[["timestamp"] + STANDARD_COLUMNS]


# ==========================================
# PROCESS JSON SCADA DATA
# ==========================================


def process_json(path):

    records = []
    discovered_tags = set()

    for sheet, raw in read_workbook(path):

        for cell in raw.to_numpy().ravel():

            if not isinstance(cell, str):
                continue

            data = decode_json(cell)

            if data is None:
                continue

            extracted = extract_records(data)

            for record in extracted:

                records.append(record)

                discovered_tags.add(record["tag"])

    if not records:

        warnings.warn("No valid SCADA records found in " + path.name)

        return None

    df = pd.DataFrame(records)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["value"] = pd.to_numeric(
        df["value"].astype(str).str.replace(",", ""), errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "value"])

    if df.empty:
        return None

    # Pivot SCADA tags into individual columns

    pivot = df.pivot_table(
        index="timestamp", columns="tag", values="value", aggfunc="mean"
    )

    pivot = pivot.sort_index()

    # Resample high-frequency data to hourly averages

    hourly = pivot.resample("1h").mean()

    # Map raw SCADA tags to standard column names

    hourly = hourly.rename(columns=SCADA_TAG_MAP)

    hourly = hourly.loc[:, ~hourly.columns.duplicated()]

    hourly = hourly.reset_index()

    print("\nSCADA tags discovered in", path.name)

    for tag in sorted(discovered_tags):

        mapped = SCADA_TAG_MAP.get(tag)

        if mapped:
            print(tag, "->", mapped)
        else:
            print(tag, "-> NOT MAPPED")

    return hourly


# ==========================================
# AUTO-DETECT AND PROCESS FILES
# ==========================================


def process_file(path):

    sheet, raw, header_row = detect_tabular(path)

    if raw is not None:

        print("\n[TABULAR DATA]", path.name)
        print("Sheet:", sheet)

        return process_tabular(raw, path, header_row)

    print("\n[JSON SCADA]", path.name)

    return process_json(path)


# ==========================================
# MERGE ALL YEARS
# ==========================================


def merge_all_files(input_dir):

    input_dir = Path(input_dir)

    files = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.suffix.lower() in [".xlsx", ".xls"] and not p.name.startswith("~$")
        ]
    )

    if not files:

        raise FileNotFoundError("No Excel files found in " + str(input_dir))

    all_data = []

    for path in files:

        try:

            df = process_file(path)

            if df is not None and not df.empty:

                all_data.append(df)

        except Exception as e:

            warnings.warn(f"Could not process {path.name}: {e}")

    if not all_data:

        raise ValueError("No valid data extracted from Excel files")

    combined = pd.concat(all_data, ignore_index=True, sort=False)

    # Ensure standard columns exist

    for col in STANDARD_COLUMNS:

        if col not in combined.columns:
            combined[col] = np.nan

        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Parse timestamps

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")

    combined = combined.dropna(subset=["timestamp"])

    combined = combined[["timestamp"] + STANDARD_COLUMNS]

    # Aggregate duplicate timestamps

    combined = combined.groupby("timestamp")[STANDARD_COLUMNS].mean().sort_index()

    # Create continuous hourly index

    combined = combined.resample("1h").mean()

    # Interpolate gaps up to two hours

    combined[STANDARD_COLUMNS] = combined[STANDARD_COLUMNS].interpolate(
        method="time", limit=MAX_GAP, limit_area="inside"
    )

    return combined.reset_index()


# ==========================================
# MAIN
# ==========================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", default="data/raw")

    parser.add_argument("--output-dir", default="outputs/processed")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting electricity data preprocessing...\n")

    df = merge_all_files(args.input_dir)

    output_file = output_dir / "grid_hourly_merged.csv"

    df.to_csv(output_file, index=False)

    print("\n===================================")
    print("PREPROCESSING COMPLETED")
    print("===================================")

    print("Total hourly records:", len(df))

    print("Start:", df["timestamp"].min())

    print("End:", df["timestamp"].max())

    print("\nMissing values:")

    print(df[STANDARD_COLUMNS].isna().sum())

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    main()
