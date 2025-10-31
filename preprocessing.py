#!/usr/bin/env python3
"""
Preprocessing for the pedestrian count dataset (NO feature expansion, NO imputation).

What it does:
  • Loads a CSV (expects a target column, default: 'pm_tot').
  • Normalizes column names (strip spaces).
  • Coerces the target to numeric and drops rows where the target is missing/non-numeric.
  • Keeps ONLY numeric feature columns (drops all non-numeric to avoid feature expansion).
  • Does NOT perform any imputation (assumes no missing values in features).

Outputs:
  - processed.csv  -> [pm_tot, <numeric features...>]
  - X.csv          -> numeric features only
  - y.csv          -> target column only
  - meta.json      -> simple manifest (no imputation performed)

Usage:
  python preprocessing.py --input ./df1_v1a_out.csv --output ./processed --target pm_tot
"""

import argparse
import os
import json
import pandas as pd
import numpy as np


def preprocess(input_path: str, output_dir: str, target_col: str = "pm_tot") -> None:
    # --- Load CSV ---
    df = pd.read_csv(input_path)
    if not isinstance(df.columns, pd.Index):
        raise ValueError("Input does not appear to be a tabular CSV.")

    # Normalize column names (strip surrounding whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns (first 20): {list(df.columns)[:20]}"
        )

    # --- Coerce target to numeric; drop rows with missing/non-numeric target (safety) ---
    before_na = df[target_col].isna().sum()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    made_na = int(df[target_col].isna().sum() - before_na)
    df = df.dropna(subset=[target_col]).copy()

    # --- Keep only numeric features (exclude the target itself) ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    if len(num_cols) == 0:
        raise ValueError(
            "No numeric feature columns found besides the target. "
            "Please ensure your dataset has numeric predictors."
        )

    X = df[num_cols]
    y = df[target_col].astype(float)

    # --- NO IMPUTATION: assume no missing values in features ---
    # If there were any NaNs, they will remain as-is by design.

    # --- Assemble processed DataFrame with target first ---
    processed = pd.concat([y.rename(target_col), X], axis=1)

    # --- Save outputs ---
    os.makedirs(output_dir, exist_ok=True)
    processed_path = os.path.join(output_dir, "processed.csv")
    X_path = os.path.join(output_dir, "X.csv")
    y_path = os.path.join(output_dir, "y.csv")
    meta_path = os.path.join(output_dir, "meta.json")

    processed.to_csv(processed_path, index=False)
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False, header=[target_col])

    meta = {
        "input_path": os.path.abspath(input_path),
        "target_col": target_col,
        "n_rows": int(processed.shape[0]),
        "n_features": len(num_cols),
        "feature_names": num_cols,
        "target_non_numeric_converted_to_na": made_na if made_na > 0 else 0,
        "imputation_performed": False,
        "notes": [
            "Dropped all non-numeric columns to avoid feature expansion.",
            "No imputation performed; assumes no missing values in features."
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {processed_path}, {X_path}, {y_path}, {meta_path}. No imputation performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess pedestrian count CSV without feature expansion or imputation."
    )
    parser.add_argument("--input", "-i", default="df1_v1a_out.csv", help="Path to input CSV file.")
    parser.add_argument("--output", "-o", default="processed", help="Output directory for processed files.")
    parser.add_argument("--target", "-t", default="pm_tot", help="Target column name (default: pm_tot).")
    args = parser.parse_args()

    preprocess(input_path=args.input, output_dir=args.output, target_col=args.target)
