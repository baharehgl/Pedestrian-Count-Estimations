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
# 0) Metrics
# ============================================================

def smape_like_R(y_true, y_pred):
    """
    Matches your R symmetric MAPE:
      mean(abs(pred - obs) / (obs/2 + pred/2) * 100)
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
    1 - dev_full / dev_null using Poisson deviance (approx).
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

# drop columns you do NOT want as features
drop_cols = [target_col, "holdout", "site_id", "geometry", "Street Nam", "_Date"]
drop_cols = [c for c in drop_cols if c in df.columns]

feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols].copy()
y_all = df[target_col].astype(float).values

print("Total rows:", len(df))
print("Raw feature count:", X_all.shape[1])

# ============================================================
# 2) Load selected features frequency file (NON-repeated CV)
# ============================================================

freq_df = pd.read_csv("selected_features_frequency_cv.csv")

TOPK = 20  # change if you want (e.g. 10, 15, 25)

selected_by_method = {}
for method in ["MutualInformation", "L1_Lasso", "RandomForest"]:
    top_feats = (
        freq_df[freq_df["feature_selection_method"] == method]
        .sort_values(["selected_count", "feature_name"], ascending=[False, True])
        .head(TOPK)["feature_name"]
        .tolist()
    )
    selected_by_method[method] = top_feats

print("\nSelected encoded features used for prediction (top frequency):")
for m, feats in selected_by_method.items():
    print(f"- {m}: {len(feats)}")

# ============================================================
# 3) Preprocessor (fit inside each fold)
# ============================================================

def make_onehot_dense():
    # for sklearn versions that use sparse_output instead of sparse
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

categorical_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_all.columns if c not in categorical_cols]

print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(categorical_cols))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", make_onehot_dense())
])

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0  # force dense
    )

# ============================================================
# 4) CV loop (K=5)
# ============================================================

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

rows = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all[train_idx]
    X_val   = X_all.iloc[val_idx].copy()
    y_val   = y_all[val_idx]

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)

    # ensure dense array
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
        X_val_enc = X_val_enc.toarray()

    feat_names = preprocessor.get_feature_names_out()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    y_train_mean = float(np.mean(y_train))

    for fs_method, selected_feats in selected_by_method.items():
        # keep only features that exist in this fold encoding
        selected_feats_existing = [f for f in selected_feats if f in name_to_idx]
        sel_idx = np.array([name_to_idx[f] for f in selected_feats_existing], dtype=int)

        X_train_sel = X_train_enc[:, sel_idx]
        X_val_sel   = X_val_enc[:, sel_idx]

        # ---------------- RandomForest ----------------
        rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(X_train_sel, y_train)

        y_train_pred = rf.predict(X_train_sel)
        y_val_pred   = rf.predict(X_val_sel)

        rows.append({
            "fold": fold,
            "feature_selection": fs_method,
            "model": "RandomForest",
            "n_features_used": len(sel_idx),
            "pseudo_R2_train": pseudo_r2_like_R(y_train, y_train_pred, y_train_mean),
            "SMAPE_val": smape_like_R(y_val, y_val_pred),
            "RMSE_val": rmse(y_val, y_val_pred)
        })

        # ---------------- HistGB Poisson ----------------
        hgb = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, random_state=42)
        hgb.fit(X_train_sel, y_train)

        y_train_pred = hgb.predict(X_train_sel)
        y_val_pred   = hgb.predict(X_val_sel)

        rows.append({
            "fold": fold,
            "feature_selection": fs_method,
            "model": "HistGB_Poisson",
            "n_features_used": len(sel_idx),
            "pseudo_R2_train": pseudo_r2_like_R(y_train, y_train_pred, y_train_mean),
            "SMAPE_val": smape_like_R(y_val, y_val_pred),
            "RMSE_val": rmse(y_val, y_val_pred)
        })

        # ---------------- GLM Negative Binomial ----------------
        X_train_glm = sm.add_constant(X_train_sel, has_constant="add")
        X_val_glm   = sm.add_constant(X_val_sel, has_constant="add")

        glm_nb = sm.GLM(y_train, X_train_glm, family=sm.families.NegativeBinomial())
        glm_res = glm_nb.fit()

        y_train_pred = glm_res.predict(X_train_glm)
        y_val_pred   = glm_res.predict(X_val_glm)

        # prevent negative/zero predictions from breaking deviance logic
        y_train_pred = np.clip(y_train_pred, 1e-9, None)
        y_val_pred   = np.clip(y_val_pred, 1e-9, None)

        rows.append({
            "fold": fold,
            "feature_selection": fs_method,
            "model": "GLM_NegativeBinomial",
            "n_features_used": len(sel_idx),
            "pseudo_R2_train": pseudo_r2_like_R(y_train, y_train_pred, y_train_mean),
            "SMAPE_val": smape_like_R(y_val, y_val_pred),
            "RMSE_val": rmse(y_val, y_val_pred)
        })

# ============================================================
# 5) Save fold results + summary CSV
# ============================================================

results_df = pd.DataFrame(rows)
results_df.to_csv("cv_selected_features_fold_results.csv", index=False)
print("\nSaved: cv_selected_features_fold_results.csv")

summary_df = (
    results_df
    .groupby(["feature_selection", "model"], as_index=False)
    .agg(
        pseudo_R2_train_mean=("pseudo_R2_train", "mean"),
        pseudo_R2_train_std=("pseudo_R2_train", "std"),
        SMAPE_val_mean=("SMAPE_val", "mean"),
        SMAPE_val_std=("SMAPE_val", "std"),
        RMSE_val_mean=("RMSE_val", "mean"),
        RMSE_val_std=("RMSE_val", "std"),
        n_features_used_mean=("n_features_used", "mean")
    )
)

summary_df = summary_df.sort_values(["SMAPE_val_mean", "RMSE_val_mean"])
summary_df.to_csv("cv_selected_features_summary.csv", index=False)

print("Saved: cv_selected_features_summary.csv")
print("\nSummary:\n", summary_df)
