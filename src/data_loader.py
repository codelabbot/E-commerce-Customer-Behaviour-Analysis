"""Data loader and non-destructive inspection utilities for e-commerce customer behavior data.

Provides reusable modular functions for dataset ingestion, schema validation,
missingness auditing, and distribution inspection with random_state=42.
"""

import os
import glob
import pandas as pd
import numpy as np

RANDOM_STATE = 42

def find_raw_data_file(data_dir: str = "data/raw") -> str:
    """Locate the raw dataset file (.csv or .xlsx) in data/raw."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Prioritize CSV if available, then XLSX
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    matched = []
    for pat in patterns:
        matched.extend(glob.glob(os.path.join(data_dir, pat)))

    if not matched:
        raise FileNotFoundError(
            f"No raw data file (.csv, .xlsx) found in {data_dir}."
        )
    return matched[0]


def load_raw_data(file_path: str = None, data_dir: str = "data/raw") -> pd.DataFrame:
    """Load the raw customer dataset without modifying any underlying records."""
    if file_path is None:
        file_path = find_raw_data_file(data_dir)

    print(f"Loading raw dataset from: {file_path} ...")
    if file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    return df


def audit_raw_data(df: pd.DataFrame) -> dict:
    """Perform a comprehensive, non-destructive audit of dataset structure and integrity.
    
    Returns a dictionary containing shape, dtypes, nullness, unique counts,
    and anomaly detection metrics.
    """
    total_records, total_columns = df.shape

    # Missing value analysis
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_records) * 100
    missing_summary = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_pct.round(2)
    })

    # Numeric anomaly audits
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    anomalies = {}
    for col in numeric_cols:
        anomalies[f"{col}_zero_count"] = int((df[col] == 0).sum())
        anomalies[f"{col}_neg_count"] = int((df[col] < 0).sum())

    # Categorical and entity distinct counts
    unique_counts = {col: int(df[col].nunique()) for col in df.columns}

    return {
        "shape": (total_records, total_columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_summary": missing_summary,
        "unique_counts": unique_counts,
        "anomalies": anomalies,
    }
