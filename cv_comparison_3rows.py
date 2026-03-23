"""
CV Comparison: First 3 rows of comparison table
All evaluated with Repeated 3x10 CV (same folds) for fair comparison.

Row 1: GLM NegativeBinomial (statsmodels -- matching R code) with R-code nb3a features
Row 2: HistGB_Poisson (default params) with R-code nb3a features
Row 3: HistGB_Poisson (default params) with L1 Lasso 18 features

Usage:
  pip install statsmodels
  python cv_comparison_3rows.py
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


def run_cv(X, y, cv, model_fn, model_name):
    """
    Run CV for a given model.
    model_fn(X_train, y_train) -> fitted object with .predict()
    """
    rmse_list, mape_list, smape_list, r2_list, mf_list = [], [], [], [], []
    fail_count = 0

    for train_idx, val_idx in cv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        try:
            model = model_fn(X_tr, y_tr)
            y_pred_val = np.maximum(model.predict(X_val), 0)
            y_pred_val = np.clip(y_pred_val, 0, y_tr.max() * 5)
            y_pred_tr = np.maximum(model.predict(X_tr), 0)

            rmse_list.append(rmse_score(y_val, y_pred_val))
            mape_list.append(standard_mape(y_val, y_pred_val))
            smape_list.append(smape_score(y_val, y_pred_val))
            r2_list.append(correlation_r2(y_val, y_pred_val))
            mf_list.append(mcfadden_r2(y_tr, y_pred_tr))
        except Exception:
            fail_count += 1

    n_ok = len(rmse_list)
    print(f"\n  {model_name}")
    print(f"  {'=' * 60}")
    if fail_count > 0:
        print(f"  Successful folds: {n_ok}/{n_ok + fail_count}")
    print(f"  RMSE:      {np.mean(rmse_list):7.2f} +/- {np.std(rmse_list):.2f}")
    print(f"  MAPE:      {np.mean(mape_list):7.2f} +/- {np.std(mape_list):.2f}")
    print(f"  SMAPE:     {np.mean(smape_list):7.2f} +/- {np.std(smape_list):.2f}")
    print(f"  Corr R2:   {np.mean(r2_list):7.4f} +/- {np.std(r2_list):.4f}")
    print(f"  McF R2:    {np.mean(mf_list):7.4f} +/- {np.std(mf_list):.4f}")

    return {
        "model": model_name,
        "RMSE_mean": np.mean(rmse_list), "RMSE_std": np.std(rmse_list),
        "MAPE_mean": np.mean(mape_list), "MAPE_std": np.std(mape_list),
        "SMAPE_mean": np.mean(smape_list), "SMAPE_std": np.std(smape_list),
        "Corr_R2_mean": np.mean(r2_list), "Corr_R2_std": np.std(r2_list),
        "McFadden_R2_mean": np.mean(mf_list), "McFadden_R2_std": np.std(mf_list),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("CV COMPARISON: Repeated 3x10 (same folds for all)")
    print("Row 1: GLM_NegBinomial (statsmodels) | R-code nb3a features")
    print("Row 2: HistGB_Poisson (Default)      | R-code nb3a features")
    print("Row 3: HistGB_Poisson (Default)      | L1 Lasso 18 features")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data -- ALL 101 samples (CV uses everything)
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["pm_tot", "holdout"])
    y = df["pm_tot"].values
    print(f"\nSamples: {len(y)}")

    # ------------------------------------------------------------------
    # 2. R-code nb3a features (NO scaling -- matching R exactly)
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

    # One-hot encode categoricals (drop="first" matches R's factor())
    # No scaling for GLM -- R doesn't scale
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(df[cat_nb3a])
    X_num = df[num_nb3a].fillna(df[num_nb3a].median()).values
    X_rcode_unscaled = np.hstack([X_num, X_cat])
    print(f"R-code nb3a (unscaled for GLM): {X_rcode_unscaled.shape[1]} columns")

    # Scaled version for HistGB (works better with scaling)
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
    # 3. L1 Lasso 18 features
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
    feat_idx_18 = np.argsort(np.abs(lasso.coef_))[::-1][:18]
    X_l1_18 = X_all[:, feat_idx_18]

    selected_names = [feature_names[i] for i in feat_idx_18]
    print(f"L1 Lasso: 18 features selected")
    for i, name in enumerate(selected_names):
        print(f"  {i+1:2d}. {name}")

    # ------------------------------------------------------------------
    # 4. Same Rep 3x10 folds for ALL
    # ------------------------------------------------------------------
    cv = RepeatedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    results = []

    # ------------------------------------------------------------------
    # ROW 1: GLM NegativeBinomial (statsmodels -- exact R match)
    # Uses unscaled features, add_constant, NegBinomial family
    # ------------------------------------------------------------------
    def fit_glm_nb(X_tr, y_tr):
        X_tr_c = sm.add_constant(X_tr)
        glm = sm.GLM(
            y_tr, X_tr_c,
            family=sm.families.NegativeBinomial(alpha=1.0)
        )
        result = glm.fit(maxiter=200, disp=False)

        class GLMPredictor:
            def __init__(self, fitted):
                self.fitted = fitted
            def predict(self, X):
                return self.fitted.predict(sm.add_constant(X))

        return GLMPredictor(result)

    r1 = run_cv(
        X_rcode_unscaled, y, cv, fit_glm_nb,
        "GLM_NegBinomial (statsmodels) | R-code nb3a features"
    )
    results.append(r1)

    # ------------------------------------------------------------------
    # ROW 2: HistGB_Poisson (Default) with R-code features
    # ------------------------------------------------------------------
    def fit_histgb_rcode(X_tr, y_tr):
        m = HistGradientBoostingRegressor(
            loss="poisson", max_iter=300, learning_rate=0.05,
            random_state=RANDOM_STATE
        )
        m.fit(X_tr, y_tr)
        return m

    r2 = run_cv(
        X_rcode_scaled, y, cv, fit_histgb_rcode,
        "HistGB_Poisson (Default) | R-code nb3a features"
    )
    results.append(r2)

    # ------------------------------------------------------------------
    # ROW 3: HistGB_Poisson (Default) with L1 Lasso 18 features
    # ------------------------------------------------------------------
    def fit_histgb_l1(X_tr, y_tr):
        m = HistGradientBoostingRegressor(
            loss="poisson", max_iter=500,
            random_state=RANDOM_STATE
        )
        m.fit(X_tr, y_tr)
        return m

    r3 = run_cv(
        X_l1_18, y, cv, fit_histgb_l1,
        "HistGB_Poisson (Default) | L1 Lasso 18 features"
    )
    results.append(r3)

    # ------------------------------------------------------------------
    # Print reference row
    # ------------------------------------------------------------------
    print(f"\n  {'=' * 60}")
    print(f"  For reference (already computed):")
    print(f"  * HistGB_Poisson (TUNED) | L1 Lasso 20 feat | Rep 3x10")
    print(f"  RMSE: 78.72 | MAPE: 51.79 | SMAPE: 49.93 | McF R2: 0.997")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    df_results.to_csv("cv_comparison_3rows.csv", index=False)
    print(f"\nResults saved to: cv_comparison_3rows.csv")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()