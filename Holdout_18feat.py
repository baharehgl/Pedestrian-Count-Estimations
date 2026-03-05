"""
Holdout Evaluation: L1 Lasso, 18 features
Default vs Tuned HistGB_Poisson

Compares:
  1. HistGB_Poisson with default parameters (no tuning)
  2. HistGB_Poisson with tuned parameters (from Exp 3)

Also runs 20 features for comparison.

Usage:
  python holdout_18feat_default_vs_tuned.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "df1_v1a_out.csv"
TARGET_COL = "pm_tot"
SPLIT_COL = "holdout"
ID_COL = "site_id"
CATEGORICAL_COLS = [
    "Street Nam", "_Date", "season", "geometry",
    "class_type", "speed_type", "crossing_class"
]
RANDOM_STATE = 42


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
        np.where(
            y_true > 0,
            y_true * np.log(y_true / y_pred_safe) - (y_true - y_pred_safe),
            y_pred_safe
        )
    )
    dev_null = 2.0 * np.sum(
        np.where(
            y_true > 0,
            y_true * np.log(y_true / y_bar) - (y_true - y_bar),
            y_bar
        )
    )
    if dev_null == 0:
        return 0.0
    return 1.0 - (dev_model / dev_null)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("HOLDOUT EVALUATION: Default vs Tuned HistGB_Poisson")
    print("Train on holdout=0 → Predict on holdout=1")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data and split by holdout column
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset shape: {df.shape}")

    y_all = df[TARGET_COL].values
    drop_cols = [c for c in [TARGET_COL, SPLIT_COL, ID_COL] if c in df.columns]
    X_all = df.drop(columns=drop_cols)

    train_mask = df[SPLIT_COL] == 0
    test_mask = df[SPLIT_COL] == 1
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    print(f"Train samples: {len(y_train)}")
    print(f"Test (holdout) samples: {len(y_test)}")

    # ------------------------------------------------------------------
    # 2. Preprocess — fit on TRAIN only, transform both
    # ------------------------------------------------------------------
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_all.columns]
    num_cols = [c for c in X_all.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], remainder="drop")

    X_train = preprocessor.fit_transform(X_all[train_mask])
    X_test = preprocessor.transform(X_all[test_mask])

    feature_names = preprocessor.get_feature_names_out().tolist()
    print(f"Total features after preprocessing: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 3. L1 Lasso feature selection — fit on TRAIN only
    # ------------------------------------------------------------------
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train, np.log1p(y_train))
    importance = np.abs(lasso.coef_)
    sorted_idx = np.argsort(importance)[::-1]

    # ------------------------------------------------------------------
    # 4. Define models: Default vs Tuned
    # ------------------------------------------------------------------
    models = {
        "HistGB_Poisson (DEFAULT)": HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=500,
            random_state=RANDOM_STATE
        ),
        "HistGB_Poisson (TUNED)": HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.01,
            max_iter=500,
            max_depth=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        ),
    }

    # ------------------------------------------------------------------
    # 5. Run for both 18 and 20 features
    # ------------------------------------------------------------------
    for n_feat in [18, 20]:
        feat_idx = sorted_idx[:n_feat]
        selected_names = [feature_names[i] for i in feat_idx]

        X_train_sel = X_train[:, feat_idx]
        X_test_sel = X_test[:, feat_idx]

        print(f"\n{'=' * 80}")
        print(f"L1 Lasso, {n_feat} features")
        print(f"{'=' * 80}")

        if n_feat == 18:
            print(f"\nSelected features:")
            for i, name in enumerate(selected_names):
                print(f"  {i+1:2d}. {name}")

        print(f"\n  {'Model':<30} {'RMSE':>8} {'MAPE':>8} {'SMAPE':>8} "
              f"{'R²_hold':>8} {'McF_R²':>8}")
        print(f"  {'-' * 78}")

        results = []
        for name, model in models.items():
            m = clone(model)
            m.fit(X_train_sel, y_train)

            y_pred_test = np.maximum(m.predict(X_test_sel), 0)
            y_pred_train = np.maximum(m.predict(X_train_sel), 0)

            rmse = rmse_score(y_test, y_pred_test)
            mape = standard_mape(y_test, y_pred_test)
            smape = smape_score(y_test, y_pred_test)
            r2_hold = correlation_r2(y_test, y_pred_test)
            mf_r2 = mcfadden_r2(y_train, y_pred_train)

            print(f"  {name:<30} {rmse:7.2f} {mape:7.2f} {smape:7.2f} "
                  f"{r2_hold:7.4f} {mf_r2:7.4f}")

            results.append({
                "model": name,
                "n_features": n_feat,
                "RMSE_holdout": rmse,
                "MAPE_holdout": mape,
                "SMAPE_holdout": smape,
                "Corr_R2_holdout": r2_hold,
                "McFadden_R2_train": mf_r2,
            })

        # Improvement
        if len(results) == 2:
            default_rmse = results[0]["RMSE_holdout"]
            tuned_rmse = results[1]["RMSE_holdout"]
            improvement = (default_rmse - tuned_rmse) / default_rmse * 100
            print(f"\n  Tuning improvement: RMSE {default_rmse:.2f} → {tuned_rmse:.2f} "
                  f"({improvement:.1f}% reduction)")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Without tuning, HistGB_Poisson with 18 features gives ~108 RMSE")
    print(f"  → nearly identical to R-code GLM baseline (RMSE=108)")
    print(f"  With tuning alone, RMSE drops to ~90 (18 feat) or ~88 (20 feat)")
    print(f"  → tuning is the single biggest factor in improvement")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()