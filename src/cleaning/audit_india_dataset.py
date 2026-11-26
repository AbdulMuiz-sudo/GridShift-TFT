"""
GridShift TFT — raw workbook audit
Run this FIRST, before cleaning. It reports workbook sheets, dimensions,
column names, and sample rows so the cleaning configuration can be grounded
in the actual downloaded files rather than guessed column names.

Usage:
    pip install pandas openpyxl
    python audit_india_dataset.py --input-dir data/raw --output-dir outputs/audit
"""
from pathlib import Path
import argparse
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/audit")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".xlsx", ".xls"}])
    if not files:
        raise SystemExit(f"No Excel files found under: {input_dir.resolve()}")

    report = []
    for file in files:
        print(f"\n=== FILE: {file} ===")
        try:
            book = pd.ExcelFile(file)
        except Exception as exc:
            print(f"Could not open workbook: {exc}")
            report.append({"file": str(file), "error": str(exc)})
            continue

        for sheet in book.sheet_names:
            try:
                # Header row is intentionally left at the default for this initial audit.
                df = pd.read_excel(file, sheet_name=sheet, nrows=8)
                info = {
                    "file": str(file),
                    "sheet": sheet,
                    "columns": [str(c) for c in df.columns],
                    "sample_rows": df.fillna("").astype(str).to_dict(orient="records"),
                }
                report.append(info)
                print(f"\nSheet: {sheet}")
                print(f"Columns: {list(df.columns)}")
                print(df.head(5).to_string(index=False))
            except Exception as exc:
                print(f"Could not read sheet {sheet!r}: {exc}")
                report.append({"file": str(file), "sheet": sheet, "error": str(exc)})

    out_json = output_dir / "workbook_audit.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAudit saved to: {out_json.resolve()}")
    print("Inspect the actual headers and sheet structure before configuring the cleaning script.")

if __name__ == "__main__":
    main()
