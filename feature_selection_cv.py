import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV

# ============================================================
# 0) Metrics (match your R symmetric MAPE)
# ============================================================

def smape_like_R(y_true, y_pred):
    """
    Matches your R CV code:
      mean(abs(pred - obs) / (obs/2 + pred/2) * 100)

    We add eps to avoid division by zero when obs=pred=0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) / 2.0) + (np.abs(y_pred) / 2.0)
    denom = np.where(denom == 0, 1e-9, denom)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pseudo_r2_like_R(y_true, y_pred, y_null_mean):
    """
    1 - dev_full / dev_null using Poisson deviance.
    (Approximation for non-GLM models; consistent with your earlier approach.)
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
# 1) Load data
# ============================================================

df = pd.read_csv("df1_v1a_out.csv").copy()
df = df.dropna(subset=["pm_tot"]).copy()

target_col = "pm_tot"

# Drop columns you said you don't want
drop_cols = [
    target_col,
    "holdout",     # not used anymore (CV instead)
    "site_id",
    "geometry",
    "Street Nam",
    "_Date"
]
drop_cols = [c for c in drop_cols if c in df.columns]

feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols].copy()
y_all = df[target_col].astype(float).values

print("Total rows:", X_all.shape[0])
print("Raw feature count:", X_all.shape[1])

# Identify numeric/categorical globally (consistent columns across folds)
categorical_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_all.columns if c not in categorical_cols]

print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(categorical_cols))


# ============================================================
# 2) Preprocessor (dense output so HistGB works)
# ============================================================

def make_onehot_dense():
    # sklearn >= 1.2 uses sparse_output; older uses sparse
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
        ("onehot", make_onehot_dense())
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0  # force dense output
    )


# ============================================================
# 3) Evaluation model (HistGB Poisson)
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

    # pseudo-R2 on TRAIN (your convention)
    y_train_mean = float(np.mean(y_train))
    r2_train = pseudo_r2_like_R(y_train, y_train_pred, y_train_mean)

    # validation metrics
    smape_val = smape_like_R(y_val, y_val_pred)
    rmse_val  = rmse(y_val, y_val_pred)

    return r2_train, smape_val, rmse_val


# ============================================================
# 4) Repeated K-Fold CV (like caret repeatedcv)
# ============================================================

K = 5
R = 5
SEED = 42
TOPK = 20

rkf = RepeatedKFold(n_splits=K, n_repeats=R, random_state=SEED)
n_splits = K * R

fold_rows = []        # metrics per split
selected_rows = []    # selected feature names per split
freq_counter = {}     # (method, feature) -> count

# Lasso inner CV (within train fold) for stability
lasso_inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

split_id = 0
for train_idx, val_idx in rkf.split(X_all):
    split_id += 1
    split_name = f"Split{split_id:02d}"

    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all[train_idx]
    X_val   = X_all.iloc[val_idx].copy()
    y_val   = y_all[val_idx]

    # Fit preprocessing on TRAIN only
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)

    # ensure dense
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
        X_val_enc   = X_val_enc.toarray()

    feat_names = preprocessor.get_feature_names_out()
    n_all = X_train_enc.shape[1]

    # ---------- A) All features ----------
    r2_all, smape_all, rmse_all = train_and_evaluate_histgb(X_train_enc, X_val_enc, y_train, y_val)

    fold_rows.append({
        "split": split_name,
        "feature_selection": "All_features",
        "n_features": n_all,
        "pseudo_R2_train": r2_all,
        "SMAPE_val": smape_all,
        "RMSE_val": rmse_all
    })

    # not storing all-features list per split (too big); if you want it, uncomment:
    # for f in feat_names:
    #     selected_rows.append({"split": split_name, "feature_selection_method": "All_features", "feature_name": f})

    # ---------- B) Mutual Information (top TOPK) ----------
    mi_scores = mutual_info_regression(X_train_enc, y_train, random_state=SEED)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)

    k_mi = min(TOPK, n_all)
    mi_idx = np.argsort(mi_scores)[::-1][:k_mi]
    mi_names = feat_names[mi_idx]

    X_train_mi = X_train_enc[:, mi_idx]
    X_val_mi   = X_val_enc[:, mi_idx]

    r2_mi, smape_mi, rmse_mi = train_and_evaluate_histgb(X_train_mi, X_val_mi, y_train, y_val)

    fold_rows.append({
        "split": split_name,
        "feature_selection": "MutualInformation",
        "n_features": len(mi_idx),
        "pseudo_R2_train": r2_mi,
        "SMAPE_val": smape_mi,
        "RMSE_val": rmse_mi
    })

    for f in mi_names:
        selected_rows.append({"split": split_name, "feature_selection_method": "MutualInformation", "feature_name": f})
        freq_counter[("MutualInformation", f)] = freq_counter.get(("MutualInformation", f), 0) + 1

    # ---------- C) L1 (Lasso) (non-zero coefs) ----------
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train_enc)
    X_val_std   = scaler.transform(X_val_enc)

    lasso = LassoCV(cv=lasso_inner_cv, n_jobs=-1, random_state=SEED, max_iter=20000)
    lasso.fit(X_train_std, y_train)

    coef = lasso.coef_
    l1_idx = np.where(np.abs(coef) > 1e-6)[0]

    if len(l1_idx) == 0:
        # fallback: take TOPK by abs(coef)
        k_l1 = min(TOPK, len(coef))
        l1_idx = np.argsort(np.abs(coef))[::-1][:k_l1]

    l1_names = feat_names[l1_idx]

    X_train_l1 = X_train_enc[:, l1_idx]
    X_val_l1   = X_val_enc[:, l1_idx]

    r2_l1, smape_l1, rmse_l1 = train_and_evaluate_histgb(X_train_l1, X_val_l1, y_train, y_val)

    fold_rows.append({
        "split": split_name,
        "feature_selection": "L1_Lasso",
        "n_features": len(l1_idx),
        "pseudo_R2_train": r2_l1,
        "SMAPE_val": smape_l1,
        "RMSE_val": rmse_l1
    })

    for f in l1_names:
        selected_rows.append({"split": split_name, "feature_selection_method": "L1_Lasso", "feature_name": f})
        freq_counter[("L1_Lasso", f)] = freq_counter.get(("L1_Lasso", f), 0) + 1

    # ---------- D) Random Forest importance (top TOPK) ----------
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_enc, y_train)

    importances = rf.feature_importances_
    k_rf = min(TOPK, n_all)
    rf_idx = np.argsort(importances)[::-1][:k_rf]
    rf_names = feat_names[rf_idx]

    X_train_rf = X_train_enc[:, rf_idx]
    X_val_rf   = X_val_enc[:, rf_idx]

    r2_rf, smape_rf, rmse_rf = train_and_evaluate_histgb(X_train_rf, X_val_rf, y_train, y_val)

    fold_rows.append({
        "split": split_name,
        "feature_selection": "RandomForest",
        "n_features": len(rf_idx),
        "pseudo_R2_train": r2_rf,
        "SMAPE_val": smape_rf,
        "RMSE_val": rmse_rf
    })

    for f in rf_names:
        selected_rows.append({"split": split_name, "feature_selection_method": "RandomForest", "feature_name": f})
        freq_counter[("RandomForest", f)] = freq_counter.get(("RandomForest", f), 0) + 1


# ============================================================
# 5) Save outputs
# ============================================================

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("feature_selection_repeatedcv_fold_results.csv", index=False)
print("\nSaved: feature_selection_repeatedcv_fold_results.csv")

summary_df = (
    fold_df.groupby("feature_selection", as_index=False)
    .agg(
        n_splits=("split", "count"),
        n_features_mean=("n_features", "mean"),
        n_features_std=("n_features", "std"),
        pseudo_R2_train_mean=("pseudo_R2_train", "mean"),
        pseudo_R2_train_std=("pseudo_R2_train", "std"),
        SMAPE_val_mean=("SMAPE_val", "mean"),
        SMAPE_val_std=("SMAPE_val", "std"),
        RMSE_val_mean=("RMSE_val", "mean"),
        RMSE_val_std=("RMSE_val", "std"),
    )
)

# add SE (like R often reports)
summary_df["pseudo_R2_train_se"] = summary_df["pseudo_R2_train_std"] / np.sqrt(summary_df["n_splits"])
summary_df["SMAPE_val_se"] = summary_df["SMAPE_val_std"] / np.sqrt(summary_df["n_splits"])
summary_df["RMSE_val_se"] = summary_df["RMSE_val_std"] / np.sqrt(summary_df["n_splits"])

# sort: lower SMAPE is better
summary_df = summary_df.sort_values("SMAPE_val_mean")

summary_df.to_csv("feature_selection_repeatedcv_summary.csv", index=False)
print("Saved: feature_selection_repeatedcv_summary.csv")
print("\nSummary:\n", summary_df)

selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv("selected_features_by_method_repeatedcv.csv", index=False)
print("Saved: selected_features_by_method_repeatedcv.csv")

freq_rows = []
for (method, feat), cnt in freq_counter.items():
    freq_rows.append({
        "feature_selection_method": method,
        "feature_name": feat,
        "selected_count": cnt,
        "selected_rate": cnt / n_splits
    })

freq_df = pd.DataFrame(freq_rows).sort_values(
    ["feature_selection_method", "selected_count", "feature_name"],
    ascending=[True, False, True]
)

freq_df.to_csv("selected_features_frequency_repeatedcv.csv", index=False)
print("Saved: selected_features_frequency_repeatedcv.csv")
