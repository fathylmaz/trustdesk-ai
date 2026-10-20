"""Validate and load original BANKING77 CSV schema."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    filename = Path(path)
    if not filename.is_file():
        raise FileNotFoundError(f"Data missing: {filename}. Run: python -m trustdesk.download_data")
    data = pd.read_csv(filename)
    if not {"text", "category"}.issubset(data.columns):
        raise ValueError(f"Expected text/category columns in {filename}, got {list(data.columns)}")
    data = data[["text", "category"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["category"] = data["category"].astype(str).str.strip()
    data = data.loc[(data["text"] != "") & (data["category"] != "")]
    if data.empty:
        raise ValueError(f"No valid rows in {filename}")
    return data.reset_index(drop=True)
