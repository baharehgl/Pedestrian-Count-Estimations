"""
Stratified CV Comparison: All 4 rows of comparison table
Stratified Repeated 3x10 CV (binned by pedestrian count volume)
Same folds for all models — fair comparison.

Row 1: GLM NegativeBinomial (statsmodels) with R-code nb3a features
Row 2: HistGB_Poisson (default params) with R-code nb3a features
Row 3: HistGB_Poisson (default params) with L1 Lasso 18 features
Row 4: HistGB_Poisson (tuned params) with L1 Lasso 20 features  ★

Stratification: pedestrian counts are binned into 5 equal groups
so every fold gets a balanced mix of low and high traffic sites.

Usage:
  pip install statsmodels
  python stratified_cv_comparison.py
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedStratifiedKFold
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
    print("STRATIFIED CV COMPARISON")
    print("Stratified Repeated 3x10 (binned by pedestrian count)")
    print("Same folds for all models")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data — ALL 101 samples
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["pm_tot", "holdout"])
    y = df["pm_tot"].values
    print(f"\nSamples: {len(y)}")

    # ------------------------------------------------------------------
    # 2. Create stratification bins (5 equal groups by count)
    # ------------------------------------------------------------------
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    print(f"Stratification bins: {np.bincount(y_bins)}")

    # ------------------------------------------------------------------
    # 3. R-code nb3a features
    # ------------------------------------------------------------------
    df["log_Number"] = np.log(df["Number of"])
    df["has_com_retail"] = (
        (df["Commercial Area"] + df["Retail Area"]) > 0
    ).astype(int)
    df["log_stv_ann_plus1"] = np.log(df["stv_ann"] + 1.0)

    num_nb3a = [
        "log_Number", "log_stv_ann_plus1", "dist_CBD",
        "has_com_retail", "Retail Area_qm", "transit_stops_qm", "med_inc_qm"
    ]
    cat_nb3a = ["crossing_class", "speed_type"]

    # Unscaled for GLM (matching R exactly)
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(df[cat_nb3a])
    X_num = df[num_nb3a].fillna(df[num_nb3a].median()).values
    X_rcode_unscaled = np.hstack([X_num, X_cat])
    print(f"R-code nb3a (unscaled for GLM): {X_rcode_unscaled.shape[1]} columns")

    # Scaled for HistGB
    pre_rcode_scaled = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_nb3a),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_nb3a),
    ], remainder="drop")
    X_rcode_scaled = pre_rcode_scaled.fit_transform(df[num_nb3a + cat_nb3a])
    print(f"R-code nb3a (scaled for HistGB): {X_rcode_scaled.shape[1]} columns")

    # ------------------------------------------------------------------
    # 4. L1 Lasso 18 and 20 features
    # ------------------------------------------------------------------
    derived_cols = ["log_Number", "has_com_retail", "log_stv_ann_plus1"]
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

    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_all, np.log1p(y))
    sorted_idx = np.argsort(np.abs(lasso.coef_))[::-1]

    X_l1_18 = X_all[:, sorted_idx[:18]]
    X_l1_20 = X_all[:, sorted_idx[:20]]

    print(f"L1 Lasso: 18 and 20 features selected")

    # ------------------------------------------------------------------
    # 5. Stratified Repeated 3x10 CV — same folds for ALL
    # ------------------------------------------------------------------
    cv = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    configs = [
        {
            "name": "GLM_NegBinomial (statsmodels) | R-code nb3a features",
            "X": X_rcode_unscaled,
            "type": "glm",
        },
        {
            "name": "HistGB_Poisson (Default) | R-code nb3a features",
            "X": X_rcode_scaled,
            "type": "histgb_default",
        },
        {
            "name": "HistGB_Poisson (Default) | L1 Lasso 18 features",
            "X": X_l1_18,
            "type": "histgb_default_l1",
        },
        {
            "name": "HistGB_Poisson (TUNED) | L1 Lasso 20 features  *",
            "X": X_l1_20,
            "type": "histgb_tuned",
        },
    ]

    all_results = []

    for cfg in configs:
        X = cfg["X"]
        rmse_list, mape_list, smape_list, r2_list, mf_list = [], [], [], [], []
        fail_count = 0

        for train_idx, val_idx in cv.split(X, y_bins):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            try:
                if cfg["type"] == "glm":
                    # GLM NegativeBinomial — matching R code exactly
                    X_tr_c = sm.add_constant(X_tr)
                    X_val_c = sm.add_constant(X_val)
                    glm = sm.GLM(
                        y_tr, X_tr_c,
                        family=sm.families.NegativeBinomial(alpha=1.0)
                    )
                    result = glm.fit(maxiter=200, disp=False)
                    y_pred_val = np.maximum(result.predict(X_val_c), 0)
                    y_pred_val = np.clip(y_pred_val, 0, y_tr.max() * 5)
                    y_pred_tr = np.maximum(result.predict(X_tr_c), 0)

                elif cfg["type"] == "histgb_default":
                    m = HistGradientBoostingRegressor(
                        loss="poisson", max_iter=300,
                        learning_rate=0.05, random_state=RANDOM_STATE
                    )
                    m.fit(X_tr, y_tr)
                    y_pred_val = np.maximum(m.predict(X_val), 0)
                    y_pred_tr = np.maximum(m.predict(X_tr), 0)

                elif cfg["type"] == "histgb_default_l1":
                    m = HistGradientBoostingRegressor(
                        loss="poisson", max_iter=500,
                        random_state=RANDOM_STATE
                    )
                    m.fit(X_tr, y_tr)
                    y_pred_val = np.maximum(m.predict(X_val), 0)
                    y_pred_tr = np.maximum(m.predict(X_tr), 0)

                elif cfg["type"] == "histgb_tuned":
                    m = HistGradientBoostingRegressor(
                        loss="poisson", learning_rate=0.01,
                        max_iter=500, max_depth=7,
                        min_samples_leaf=10, l2_regularization=1.0,
                        random_state=RANDOM_STATE
                    )
                    m.fit(X_tr, y_tr)
                    y_pred_val = np.maximum(m.predict(X_val), 0)
                    y_pred_tr = np.maximum(m.predict(X_tr), 0)

                rmse_list.append(rmse_score(y_val, y_pred_val))
                mape_list.append(standard_mape(y_val, y_pred_val))
                smape_list.append(smape_score(y_val, y_pred_val))
                r2_list.append(correlation_r2(y_val, y_pred_val))
                mf_list.append(mcfadden_r2(y_tr, y_pred_tr))

            except Exception:
                fail_count += 1

        n_ok = len(rmse_list)
        result = {
            "model": cfg["name"],
            "RMSE_mean": np.mean(rmse_list), "RMSE_std": np.std(rmse_list),
            "MAPE_mean": np.mean(mape_list), "MAPE_std": np.std(mape_list),
            "SMAPE_mean": np.mean(smape_list), "SMAPE_std": np.std(smape_list),
            "Corr_R2_mean": np.mean(r2_list), "Corr_R2_std": np.std(r2_list),
            "McFadden_R2_mean": np.mean(mf_list), "McFadden_R2_std": np.std(mf_list),
        }
        all_results.append(result)

        print(f"\n  {cfg['name']}")
        print(f"  {'=' * 60}")
        if fail_count > 0:
            print(f"  Successful folds: {n_ok}/{n_ok + fail_count}")
        print(f"  RMSE:      {result['RMSE_mean']:7.2f} +/- {result['RMSE_std']:.2f}")
        print(f"  MAPE:      {result['MAPE_mean']:7.2f} +/- {result['MAPE_std']:.2f}")
        print(f"  SMAPE:     {result['SMAPE_mean']:7.2f} +/- {result['SMAPE_std']:.2f}")
        print(f"  Corr R2:   {result['Corr_R2_mean']:7.4f} +/- {result['Corr_R2_std']:.4f}")
        print(f"  McF R2:    {result['McFadden_R2_mean']:7.4f} +/- {result['McFadden_R2_std']:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("stratified_cv_comparison.csv", index=False)
    print(f"\nResults saved to: stratified_cv_comparison.csv")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()