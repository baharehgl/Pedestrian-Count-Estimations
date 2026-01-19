# ============================================
# ALL-FEATURES: 5-fold CV comparison
# Models: RandomForest, HistGB (Poisson), GLM NegativeBinomial
# Metrics:
#   - Pseudo-R2 (TRAIN fold)
#   - SMAPE (VAL fold)  [R-style symmetric MAPE]
#   - RMSE (VAL fold)
# Data: df1_v1a_out.csv
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm


# ============================================================
# 1) Metrics
# ============================================================

def smape(y_true, y_pred):
    """
    Symmetric MAPE (same idea as your R cvstats):
    mean( abs(pred-obs) / (obs/2 + pred/2) * 100 )
    Avoids explosions when obs ~ 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-9, denom)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pseudo_r2_like_R(y_true, y_pred, y_null_mean):
    """
    Approximate mf_r2 = 1 - deviance/null.deviance
    using Poisson deviance computed from predictions.
    (Works for non-GLM models too as an approximation.)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-9
    mu = np.clip(y_pred, eps, None)
    mu0 = np.clip(float(y_null_mean), eps, None)

    def poisson_deviance(y, mu):
        term = np.where(
            y == 0,
            mu,
            y * np.log(y / mu) - (y - mu)
        )
        return 2.0 * np.sum(term)

    dev_full = poisson_deviance(y_true, mu)
    dev_null = poisson_deviance(y_true, np.full_like(y_true, mu0))

    return 1.0 - (dev_full / dev_null)


# ============================================================
# 2) Load data (ignore holdout for CV)
# ============================================================

df = pd.read_csv("df1_v1a_out.csv")
df = df.dropna(subset=["pm_tot"]).copy()

target_col = "pm_tot"

# OPTIONAL: drop ID-like columns if present
id_cols = ["site_id"]

# Use ALL features except target and IDs (keep holdout column out of features too)
drop_cols = [target_col, "holdout"] + [c for c in id_cols if c in df.columns]

feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols].copy()
y_all = df[target_col].astype(float).values

print("Total rows:", X_all.shape[0])
print("Raw feature count:", X_all.shape[1])

# Identify categorical vs numeric on the full dataset (ok; encoding is refit inside fold)
categorical_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_all.columns if c not in categorical_cols]

print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(categorical_cols))


# ============================================================
# 3) Preprocessor builder
# ============================================================

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


# ============================================================
# 4) 5-fold CV
# ============================================================

K = 5
SEED = 42
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all[train_idx]
    X_val   = X_all.iloc[val_idx].copy()
    y_val   = y_all[val_idx]

    y_train_mean = float(np.mean(y_train))

    # Fit preprocessor ONLY on train fold (no leakage)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)

    # ---------------- Random Forest ----------------
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train_enc, y_train)

    y_train_pred_rf = rf.predict(X_train_enc)
    y_val_pred_rf   = rf.predict(X_val_enc)

    rf_r2_train = pseudo_r2_like_R(y_train, y_train_pred_rf, y_train_mean)
    rf_smape_val = smape(y_val, y_val_pred_rf)
    rf_rmse_val  = rmse(y_val, y_val_pred_rf)

    fold_results.append({
        "fold": fold, "model": "RandomForest",
        "pseudo_R2_train": rf_r2_train,
        "SMAPE_val": rf_smape_val,
        "RMSE_val": rf_rmse_val
    })

    # ---------------- HistGB Poisson ----------------
    hgb = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        random_state=SEED
    )
    hgb.fit(X_train_enc, y_train)

    y_train_pred_hgb = hgb.predict(X_train_enc)
    y_val_pred_hgb   = hgb.predict(X_val_enc)

    hgb_r2_train = pseudo_r2_like_R(y_train, y_train_pred_hgb, y_train_mean)
    hgb_smape_val = smape(y_val, y_val_pred_hgb)
    hgb_rmse_val  = rmse(y_val, y_val_pred_hgb)

    fold_results.append({
        "fold": fold, "model": "HistGB_Poisson",
        "pseudo_R2_train": hgb_r2_train,
        "SMAPE_val": hgb_smape_val,
        "RMSE_val": hgb_rmse_val
    })

    # ---------------- GLM Negative Binomial ----------------
    # Add intercept column
    X_train_glm = sm.add_constant(X_train_enc, has_constant="add")
    X_val_glm   = sm.add_constant(X_val_enc,   has_constant="add")

    glm_nb = sm.GLM(y_train, X_train_glm, family=sm.families.NegativeBinomial())
    glm_res = glm_nb.fit()

    y_train_pred_glm = glm_res.predict(X_train_glm)
    y_val_pred_glm   = glm_res.predict(X_val_glm)

    # For GLM we can compute the true McFadden pseudo-R2 directly
    try:
        glm_r2_train = float(1.0 - glm_res.deviance / glm_res.null_deviance)
    except Exception:
        glm_r2_train = pseudo_r2_like_R(y_train, y_train_pred_glm, y_train_mean)

    glm_smape_val = smape(y_val, y_val_pred_glm)
    glm_rmse_val  = rmse(y_val, y_val_pred_glm)

    fold_results.append({
        "fold": fold, "model": "GLM_NegativeBinomial",
        "pseudo_R2_train": glm_r2_train,
        "SMAPE_val": glm_smape_val,
        "RMSE_val": glm_rmse_val
    })

# ============================================================
# 5) Save fold results + summary
# ============================================================

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv("cv_allfeatures_fold_results.csv", index=False)

summary_df = (
    fold_df
    .groupby("model", as_index=False)
    .agg(
        pseudo_R2_train_mean=("pseudo_R2_train", "mean"),
        pseudo_R2_train_std=("pseudo_R2_train", "std"),
        SMAPE_val_mean=("SMAPE_val", "mean"),
        SMAPE_val_std=("SMAPE_val", "std"),
        RMSE_val_mean=("RMSE_val", "mean"),
        RMSE_val_std=("RMSE_val", "std"),
    )
)

summary_df.to_csv("cv_allfeatures_summary_results.csv", index=False)

print("\nSaved:")
print(" - cv_allfeatures_fold_results.csv")
print(" - cv_allfeatures_summary_results.csv")
print("\nSummary:")
print(summary_df)
