import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV

# ============================================================
# 0) Metrics
# ============================================================

def smape(y_true, y_pred):
    """
    SMAPE (like your R CV code):
      mean( abs(pred-obs) / (abs(obs)+abs(pred))/2 ) * 100
    Does NOT explode when obs == 0.
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
    1 - dev_full / dev_null using Poisson deviance
    (approximation for non-GLM models; for GLM you can compute deviance directly).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-9
    mu = np.clip(y_pred, eps, None)
    mu0 = np.clip(float(y_null_mean), eps, None)

    def poisson_deviance(y, mu):
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
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
# 1) Load data and define features / target
# ============================================================

df = pd.read_csv("df1_v1a_out.csv")
df = df.dropna(subset=["pm_tot"]).copy()

target_col = "pm_tot"

# columns we explicitly do NOT want as features
drop_cols = [
    target_col,
    "holdout",     # ignore holdout for CV (if exists)
    "site_id",     # ID
    "geometry",    # WKT point string
    "Street Nam",  # street name text column
    "_Date"        # date string
]

drop_cols = [c for c in drop_cols if c in df.columns]  # keep only those that exist
feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols].copy()
y_all = df[target_col].astype(float).values

print("Total rows:", X_all.shape[0])
print("Raw feature count:", X_all.shape[1])

# Identify categorical vs numeric by dtype (will be used consistently across folds)
categorical_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_all.columns if c not in categorical_cols]

print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(categorical_cols))


# ============================================================
# 2) Preprocessor (dense output so HistGB works)
# ============================================================

def make_onehot():
    # sklearn newer versions use sparse_output; older use sparse
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot())
    ])

    # sparse_threshold=0 forces dense output
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0
    )


# ============================================================
# 3) Base model for evaluation (HistGB Poisson)
# ============================================================

def train_and_evaluate_histgb(X_train_sel, X_val_sel, y_train, y_val):
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_sel, y_train)

    y_train_pred = model.predict(X_train_sel)
    y_val_pred   = model.predict(X_val_sel)

    # Pseudo-R2 on TRAIN fold (like your earlier convention)
    y_train_mean = float(np.mean(y_train))
    r2_train = pseudo_r2_like_R(y_train, y_train_pred, y_train_mean)

    # Validation metrics
    smape_val = smape(y_val, y_val_pred)
    rmse_val  = rmse(y_val, y_val_pred)

    return r2_train, smape_val, rmse_val


# ============================================================
# 4) 5-fold CV for feature selection comparison
# ============================================================

K = 5
SEED = 42
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

# How many features to select
K_TOP = 20

fold_rows = []                 # per-fold metrics
selected_rows = []             # per-fold selected feature names
freq_counter = {}              # (method, feature_name) -> count

methods = ["All_features", "MutualInformation", "L1_Lasso", "RandomForest"]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
    print(f"\n========== Fold {fold}/{K} ==========")

    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all[train_idx]
    X_val   = X_all.iloc[val_idx].copy()
    y_val   = y_all[val_idx]

    # Fit preprocessor on TRAIN fold only
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)

    # Just in case, ensure dense
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
        X_val_enc   = X_val_enc.toarray()

    feature_names_enc = preprocessor.get_feature_names_out()
    n_all = X_train_enc.shape[1]

    # -----------------------
    # A) Baseline: All features
    # -----------------------
    r2_all, smape_all, rmse_all = train_and_evaluate_histgb(X_train_enc, X_val_enc, y_train, y_val)

    fold_rows.append({
        "fold": fold,
        "feature_selection": "All_features",
        "n_features": n_all,
        "pseudo_R2_train": r2_all,
        "SMAPE_val": smape_all,
        "RMSE_val": rmse_all
    })

    # store selected features
    for f in feature_names_enc:
        selected_rows.append({"fold": fold, "feature_selection_method": "All_features", "feature_name": f})

    # -----------------------
    # B) Mutual Information (top K_TOP)
    # -----------------------
    mi_scores = mutual_info_regression(X_train_enc, y_train, random_state=SEED)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)

    k_mi = min(K_TOP, n_all)
    mi_idx = np.argsort(mi_scores)[::-1][:k_mi]
    mi_names = feature_names_enc[mi_idx]

    X_train_mi = X_train_enc[:, mi_idx]
    X_val_mi   = X_val_enc[:, mi_idx]

    r2_mi, smape_mi, rmse_mi = train_and_evaluate_histgb(X_train_mi, X_val_mi, y_train, y_val)

    fold_rows.append({
        "fold": fold,
        "feature_selection": "MutualInformation",
        "n_features": len(mi_idx),
        "pseudo_R2_train": r2_mi,
        "SMAPE_val": smape_mi,
        "RMSE_val": rmse_mi
    })

    for f in mi_names:
        selected_rows.append({"fold": fold, "feature_selection_method": "MutualInformation", "feature_name": f})
        freq_counter[("MutualInformation", f)] = freq_counter.get(("MutualInformation", f), 0) + 1

    # -----------------------
    # C) L1 (Lasso) selection (non-zero coefs) on standardized data
    # -----------------------
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train_enc)
    X_val_std   = scaler.transform(X_val_enc)

    lasso = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=20000)
    lasso.fit(X_train_std, y_train)

    coef = lasso.coef_
    lasso_idx = np.where(np.abs(coef) > 1e-6)[0]

    # fallback if everything is shrunk to zero
    if len(lasso_idx) == 0:
        k_lasso = min(K_TOP, len(coef))
        lasso_idx = np.argsort(np.abs(coef))[::-1][:k_lasso]

    lasso_names = feature_names_enc[lasso_idx]

    X_train_l1 = X_train_enc[:, lasso_idx]
    X_val_l1   = X_val_enc[:, lasso_idx]

    r2_l1, smape_l1, rmse_l1 = train_and_evaluate_histgb(X_train_l1, X_val_l1, y_train, y_val)

    fold_rows.append({
        "fold": fold,
        "feature_selection": "L1_Lasso",
        "n_features": len(lasso_idx),
        "pseudo_R2_train": r2_l1,
        "SMAPE_val": smape_l1,
        "RMSE_val": rmse_l1
    })

    for f in lasso_names:
        selected_rows.append({"fold": fold, "feature_selection_method": "L1_Lasso", "feature_name": f})
        freq_counter[("L1_Lasso", f)] = freq_counter.get(("L1_Lasso", f), 0) + 1

    # -----------------------
    # D) Random Forest importance (top K_TOP)
    # -----------------------
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_enc, y_train)

    importances = rf.feature_importances_
    k_rf = min(K_TOP, n_all)
    rf_idx = np.argsort(importances)[::-1][:k_rf]
    rf_names = feature_names_enc[rf_idx]

    X_train_rf = X_train_enc[:, rf_idx]
    X_val_rf   = X_val_enc[:, rf_idx]

    r2_rf, smape_rf, rmse_rf = train_and_evaluate_histgb(X_train_rf, X_val_rf, y_train, y_val)

    fold_rows.append({
        "fold": fold,
        "feature_selection": "RandomForest",
        "n_features": len(rf_idx),
        "pseudo_R2_train": r2_rf,
        "SMAPE_val": smape_rf,
        "RMSE_val": rmse_rf
    })

    for f in rf_names:
        selected_rows.append({"fold": fold, "feature_selection_method": "RandomForest", "feature_name": f})
        freq_counter[("RandomForest", f)] = freq_counter.get(("RandomForest", f), 0) + 1


# ============================================================
# 5) Save outputs
# ============================================================

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("feature_selection_cv_fold_results.csv", index=False)
print("\nSaved: feature_selection_cv_fold_results.csv")

summary_df = (
    fold_df.groupby("feature_selection", as_index=False)
    .agg(
        n_features_mean=("n_features", "mean"),
        n_features_std=("n_features", "std"),
        pseudo_R2_train_mean=("pseudo_R2_train", "mean"),
        pseudo_R2_train_std=("pseudo_R2_train", "std"),
        SMAPE_val_mean=("SMAPE_val", "mean"),
        SMAPE_val_std=("SMAPE_val", "std"),
        RMSE_val_mean=("RMSE_val", "mean"),
        RMSE_val_std=("RMSE_val", "std"),
    )
    .sort_values("SMAPE_val_mean")  # lower is better
)

summary_df.to_csv("feature_selection_cv_summary.csv", index=False)
print("Saved: feature_selection_cv_summary.csv")
print("\nSummary:\n", summary_df)

selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv("selected_features_by_method_cv.csv", index=False)
print("Saved: selected_features_by_method_cv.csv")

# Frequency table (excluding All_features because it's huge)
freq_rows = []
for (method, feat), cnt in freq_counter.items():
    freq_rows.append({"feature_selection_method": method, "feature_name": feat, "selected_in_folds": cnt})

freq_df = pd.DataFrame(freq_rows).sort_values(
    ["feature_selection_method", "selected_in_folds", "feature_name"],
    ascending=[True, False, True]
)

freq_df.to_csv("selected_features_frequency_cv.csv", index=False)
print("Saved: selected_features_frequency_cv.csv")
