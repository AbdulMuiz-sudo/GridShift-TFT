# GridShift TFT — Phase 1 and Phase 2 starter files

## Dataset source

Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash (2025).
*Electricity Demand, Solar and Wind Generation Data (September 2021–June 2025) of India at 1-hour interval*.
Mendeley Data, Version 2. DOI: https://doi.org/10.17632/y58jknpgs8.2

Dataset page: https://data.mendeley.com/datasets/y58jknpgs8/2
The repository page describes data from North-Eastern Regional Load Despatch Centre, Grid-India, covering combined five regional grid networks. Licence listed: CC BY 4.0.

IMPORTANT metadata caution: the record title and file list refer to September 2021–June 2025, but the description text on the page still says September 2021–December 2023. The file list includes `January 2024- June 2025.xlsx`. This project report explicitly notes the inconsistency instead of silently resolving it.

## Folder setup

1. Download the dataset files from the Mendeley page.
2. Put the original Excel files unchanged in `data/raw/`.
3. Install dependencies:

   `pip install pandas numpy openpyxl matplotlib seaborn scikit-learn joblib`

4. Run the workbook audit first:

   `python audit_india_dataset.py --input-dir data/raw --output-dir outputs/audit`

5. Open `outputs/audit/workbook_audit.json`. Confirm the actual sheet layouts, column names, units, and whether each row is a national aggregate or a regional record.
6. Set `TIMESTAMP_COL`, `LOAD_COL`, `SOLAR_COL`, `WIND_COL`, `SHEET_NAME`, and `HEADER_ROW` in `clean_india_grid.py` to values actually present in your files. Do not guess these values.
7. If the workbook has multi-row headers, merged cells, repeated header rows, or separate date/time columns, adapt the reader based on the audit before running cleaning.
8. Run:

   `python clean_india_grid.py --input-dir data/raw --output-dir outputs/processed`

## What the cleaning script does

- reads Excel files using configured sheet/header settings;
- removes fully empty rows and columns;
- parses timestamps and reports invalid timestamps;
- coerces load/solar/wind fields to numeric;
- stops if timestamps repeat, because duplicates may be legitimate regional records and should not be silently dropped;
- produces raw-data EDA figures;
- interpolates only short internal gaps (at most two hourly rows) for numeric variables;
- flags IQR outliers without deleting them by default;
- removes rows with missing load targets after short-gap interpolation;
- creates cyclical calendar features;
- splits data chronologically, holding out the final 20%;
- fits `StandardScaler` on training rows only, then applies it to the test rows;
- writes a JSON cleaning audit report and CSV outputs.

The script is intentionally configuration-driven because exact workbook headers and sheet layouts have not been independently verified. The audit must be run before configuring it. Do not report actual missingness counts, outlier counts, or model results until the script has run on the downloaded files.

## EDA figure files

The script saves:
- `raw_missingness.png`
- `raw_distributions.png`
- `raw_correlation.png`
- `raw_time_series.png`

Copy the verified figures into the LaTeX project's `figures/` folder if you want them included in the PDF.

## LaTeX

`gridshift_phase1_phase2.tex` is a source document. Compile with `pdflatex` twice, or use Overleaf. It includes figures only when matching files exist in `figures/`. Replace team/instructor placeholders and add verified results after running the scripts. Do not treat the proposed methods as experimental findings.
