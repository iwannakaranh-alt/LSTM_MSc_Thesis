
from pathlib import Path
import pandas as pd
from lstm import (
    load_series, run_grid, train_final)

DATA_PATH  = Path(r"C:\code\portfolio\DATA_LINEAR.xlsx")
SHEET_NAME = "H28"
DATE_COL   = "Timestamp"
TARGET_COL = "H28"

TITLE   = "Hourly Energy Forecast"
OUTBASE = Path("results_hourly")
LOOKBACKS   = [720]
TRAIN_SIZES = [0.5]
EPOCHS      = [10]
BATCHES     = [8]
# --------------------------------

def _load_series_from_excel(path, sheet, date_col, target_col):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if date_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"Missing columns. Found: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    s = pd.to_numeric(df[target_col], errors="coerce").dropna()
    return s 

def main():
    s = _load_series_from_excel(DATA_PATH, SHEET_NAME, DATE_COL, TARGET_COL)

    print(f"Running sheet='{SHEET_NAME}' (n={len(s)}) …")
    best = run_grid(s, LOOKBACKS, TRAIN_SIZES, EPOCHS, BATCHES)
    if best is None:
        raise RuntimeError("No valid combos. Maybe increase data length or lower lookback.")
    best_mse, params = best
    print("Best combo:", params, f"mse={best_mse:.6f}")
    out_dir = OUTBASE / DATA_PATH.stem / SHEET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    train_final(
        s,
        lookback=params["lookback"],
        train_size=params["train_size"],
        epochs=params["epochs"],
        batch=params["batch"],
        title=f"{TITLE} — {SHEET_NAME}",
        out=out_dir,
    )
    print(f"Saved artifacts to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
