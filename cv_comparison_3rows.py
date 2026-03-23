"""
CV Comparison: First 3 rows of comparison table
All evaluated with Repeated 3×10 CV (same folds) for fair comparison.

Row 1: PoissonRegressor (GLM baseline) with R-code nb3a features
Row 2: HistGB_Poisson (default params) with R-code nb3a features
Row 3: HistGB_Poisson (default params) with L1 Lasso 18 features

Usage:
  python cv_comparison_3rows.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "df1_v1a_out.csv"
RANDOM_STATE = 42
CATEGORICAL_COLS = [
    "Street Nam", "_Date", "season", "geometry",
    "class_type", "speed_type", "crossing_class"
]


# ============================================================================
# METRICS
# ============================================================================

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def standard_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

def smape_score(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

def correlation_r2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2

def mcfadden_r2(y_true, y_pred):
    """McFadden's pseudo-R2 (deviance-based, matching R code)."""
    y_pred_safe = np.maximum(y_pred, 1e-10)
    y_bar = max(np.mean(y_true), 1e-10)
    dev_model = 2.0 * np.sum(
        np.where(y_true > 0,
                 y_true * np.log(y_true / y_pred_safe) - (y_true - y_pred_safe),
                 y_pred_safe)
    )
    dev_null = 2.0 * np.sum(
        np.where(y_true > 0,
                 y_true * np.log(y_true / y_bar) - (y_true - y_bar),
                 y_bar)
    )
    if dev_null == 0:
        return 0.0
    return 1.0 - (dev_model / dev_null)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("CV COMPARISON: Repeated 3×10 (same folds for all)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data — ALL 101 samples (CV uses everything)
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["pm_tot", "holdout"])
    y = df["pm_tot"].values
    print(f"\nSamples: {len(y)}")

    # ------------------------------------------------------------------
    # 2. Create R-code derived features
    # ------------------------------------------------------------------
    df["log_Number"] = np.log(df["Number of"])
    df["has_com_retail"] = (
        (df["Commercial Area"] + df["Retail Area"]) > 0
    ).astype(int)
    df["log_stv_ann_plus1"] = np.log(df["stv_ann"] + 1.0)

    # ------------------------------------------------------------------
    # 3. Prepare R-code nb3a features
    # ------------------------------------------------------------------
    features_nb3a = [
        "log_Number", "log_stv_ann_plus1", "dist_CBD",
        "has_com_retail", "crossing_class", "speed_type",
        "Retail Area_qm", "transit_stops_qm", "med_inc_qm"
    ]
    cat_nb3a = ["crossing_class", "speed_type"]
    num_nb3a = [c for c in features_nb3a if c not in cat_nb3a]

    pre_rcode = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_nb3a),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_nb3a),
    ], remainder="drop")

    X_rcode = pre_rcode.fit_transform(df[features_nb3a])
    print(f"R-code nb3a features: {X_rcode.shape[1]} columns after encoding")

    # ------------------------------------------------------------------
    # 4. Prepare L1 Lasso 18 features
    # ------------------------------------------------------------------
    derived_cols = [
        "log_Number", "has_com_retail", "log_stv_ann_plus1"
    ]
    drop_cols = [
        c for c in ["pm_tot", "holdout", "site_id"] + derived_cols
        if c in df.columns
    ]
    X_raw = df.drop(columns=drop_cols)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    pre_l1 = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], remainder="drop")

    X_all = pre_l1.fit_transform(X_raw)
    feature_names = pre_l1.get_feature_names_out().tolist()

    # L1 Lasso feature selection
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_all, np.log1p(y))
    feat_idx_18 = np.argsort(np.abs(lasso.coef_))[::-1][:18]
    X_l1_18 = X_all[:, feat_idx_18]

    selected_names = [feature_names[i] for i in feat_idx_18]
    print(f"L1 Lasso: 18 features selected")
    for i, name in enumerate(selected_names):
        print(f"  {i+1:2d}. {name}")

    # ------------------------------------------------------------------
    # 5. Define 3 configurations
    # ------------------------------------------------------------------
    configs = [
        {
            "name": "PoissonRegressor (GLM baseline)",
            "features": "R-code nb3a features",
            "X": X_rcode,
            "model": PoissonRegressor(alpha=1.0, max_iter=5000),
        },
        {
            "name": "HistGB_Poisson (Default)",
            "features": "R-code nb3a features",
            "X": X_rcode,
            "model": HistGradientBoostingRegressor(
                loss="poisson", max_iter=300, learning_rate=0.05,
                random_state=RANDOM_STATE
            ),
        },
        {
            "name": "HistGB_Poisson (Default)",
            "features": "L1 Lasso, 18 features",
            "X": X_l1_18,
            "model": HistGradientBoostingRegressor(
                loss="poisson", max_iter=500,
                random_state=RANDOM_STATE
            ),
        },
    ]

    # ------------------------------------------------------------------
    # 6. Run all with SAME Repeated 3×10 CV folds
    # ------------------------------------------------------------------
    cv = RepeatedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    all_results = []

    for cfg in configs:
        X = cfg["X"]
        model = cfg["model"]

        rmse_list = []
        mape_list = []
        smape_list = []
        r2_list = []
        mf_list = []

        for train_idx, val_idx in cv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)
            y_pred_val = np.maximum(m.predict(X_val), 0)
            y_pred_tr = np.maximum(m.predict(X_tr), 0)

            rmse_list.append(rmse_score(y_val, y_pred_val))
            mape_list.append(standard_mape(y_val, y_pred_val))
            smape_list.append(smape_score(y_val, y_pred_val))
            r2_list.append(correlation_r2(y_val, y_pred_val))
            mf_list.append(mcfadden_r2(y_tr, y_pred_tr))

        result = {
            "model": cfg["name"],
            "features": cfg["features"],
            "RMSE_mean": np.mean(rmse_list),
            "RMSE_std": np.std(rmse_list),
            "MAPE_mean": np.mean(mape_list),
            "MAPE_std": np.std(mape_list),
            "SMAPE_mean": np.mean(smape_list),
            "SMAPE_std": np.std(smape_list),
            "Corr_R2_mean": np.mean(r2_list),
            "Corr_R2_std": np.std(r2_list),
            "McFadden_R2_mean": np.mean(mf_list),
            "McFadden_R2_std": np.std(mf_list),
        }
        all_results.append(result)

        print(f"\n  {cfg['name']} | {cfg['features']}")
        print(f"  {'─' * 60}")
        print(f"  RMSE:      {result['RMSE_mean']:7.2f} ± {result['RMSE_std']:.2f}")
        print(f"  MAPE:      {result['MAPE_mean']:7.2f} ± {result['MAPE_std']:.2f}")
        print(f"  SMAPE:     {result['SMAPE_mean']:7.2f} ± {result['SMAPE_std']:.2f}")
        print(f"  Corr R²:   {result['Corr_R2_mean']:7.4f} ± {result['Corr_R2_std']:.4f}")
        print(f"  McF R²:    {result['McFadden_R2_mean']:7.4f} ± {result['McFadden_R2_std']:.4f}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("cv_comparison_3rows.csv", index=False)
    print(f"\nResults saved to: cv_comparison_3rows.csv")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()