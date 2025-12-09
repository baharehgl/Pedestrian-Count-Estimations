import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# ============================================================
# 1. Load data and RF-selected features
# ============================================================

# Main data (after preprocessing)
df = pd.read_csv("df1_v1a_out.csv")

# CSV created by previous feature-selection script
sel_feats_df = pd.read_csv("selected_features_by_method.csv")

# Filter to RandomForest-based selection
rf_feats = sel_feats_df.loc[
    sel_feats_df["feature_selection_method"] == "RandomForest",
    "feature_name"
].tolist()

if len(rf_feats) == 0:
    raise ValueError("No features found for method 'RandomForest' in selected_features_by_method.csv")

print(f"Number of RF-selected encoded features: {len(rf_feats)}")

# Ensure required columns exist
assert "pm_tot" in df.columns, "pm_tot not in data!"
assert "holdout" in df.columns, "holdout not in data!"

# Drop rows with missing target or holdout
df = df.dropna(subset=["pm_tot", "holdout"])

target_col = "pm_tot"
split_col = "holdout"
id_cols = ["site_id"]  # optional

# columns we explicitly do NOT want as features
drop_cols = [
    target_col,
    split_col,
    "site_id",     # ID
    "geometry",    # WKT point string
    "Street Nam",  # street name text column
    "_Date"        # date as string (the one that became cat___Date_...)
]

# All feature columns = everything except these
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[target_col].values

# ============================================================
# 2. Train / Test split by holdout
# ============================================================

train_mask = df[split_col] == 0
test_mask  = df[split_col] == 1

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Total raw features:", X_train.shape[1])

y_train_mean = y_train.mean()

# ============================================================
# 3. Identify numeric and categorical columns
# ============================================================

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

# ============================================================
# 4. Preprocessor (same structure as in feature-selection script)
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Fit on TRAIN only
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

feature_names_enc = preprocessor.get_feature_names_out()
print("Encoded train shape:", X_train_enc.shape)

# ============================================================
# 5. Map RF-selected feature names to indices
# ============================================================

name_to_idx = {name: i for i, name in enumerate(feature_names_enc)}

missing = [f for f in rf_feats if f not in name_to_idx]
if missing:
    print("Warning: The following RF-selected features were not found in current encoding:")
    for m in missing:
        print("  -", m)
    # Keep only those that exist
    rf_feats = [f for f in rf_feats if f in name_to_idx]

rf_indices = np.array([name_to_idx[f] for f in rf_feats], dtype=int)

print(f"Using {len(rf_indices)} RF-selected encoded features.")

X_train_sel = X_train_enc[:, rf_indices]
X_test_sel  = X_test_enc[:, rf_indices]

# ============================================================
# 6. Metrics: MAPE, RMSE, pseudo-R2 (train, like R)
# ============================================================

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    denom = np.where(y_true == 0, eps, y_true)
    return np.mean(np.abs((y_pred - y_true) / denom) * 100.0)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pseudo_r2_like_R(y_true, y_pred, y_null_mean):
    """
    Approximate R-style:
        1 - dev_full / dev_null
    using Poisson deviance as deviance measure.
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
# 7. Train & evaluate 3 models on RF-selected features
# ============================================================

results = []

# ---------- 1) Random Forest ----------
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_sel, y_train)

y_train_pred_rf = rf.predict(X_train_sel)
y_test_pred_rf  = rf.predict(X_test_sel)

rf_pseudo_r2 = pseudo_r2_like_R(y_train, y_train_pred_rf, y_train_mean)
rf_mape = mape(y_test, y_test_pred_rf)
rf_rmse = rmse(y_test, y_test_pred_rf)

print("\nRandom Forest (on RF-selected features):")
print(f"  Pseudo-R² (train): {rf_pseudo_r2:.4f}")
print(f"  MAPE (test, %):    {rf_mape:.2f}")
print(f"  RMSE (test):       {rf_rmse:.4f}")

results.append({
    "model": "RandomForest",
    "pseudo_R2_train": rf_pseudo_r2,
    "MAPE_test": rf_mape,
    "RMSE_test": rf_rmse
})

# ---------- 2) HistGradientBoosting (Poisson) ----------
hgb = HistGradientBoostingRegressor(
    loss="poisson",
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

hgb.fit(X_train_sel, y_train)

y_train_pred_hgb = hgb.predict(X_train_sel)
y_test_pred_hgb  = hgb.predict(X_test_sel)

hgb_pseudo_r2 = pseudo_r2_like_R(y_train, y_train_pred_hgb, y_train_mean)
hgb_mape = mape(y_test, y_test_pred_hgb)
hgb_rmse = rmse(y_test, y_test_pred_hgb)

print("\nHistGradientBoosting (Poisson) (on RF-selected features):")
print(f"  Pseudo-R² (train): {hgb_pseudo_r2:.4f}")
print(f"  MAPE (test, %):    {hgb_mape:.2f}")
print(f"  RMSE (test):       {hgb_rmse:.4f}")

results.append({
    "model": "HistGB_Poisson",
    "pseudo_R2_train": hgb_pseudo_r2,
    "MAPE_test": hgb_mape,
    "RMSE_test": hgb_rmse
})

# ---------- 3) GLM Negative Binomial ----------
X_train_glm = sm.add_constant(X_train_sel, has_constant="add")
X_test_glm  = sm.add_constant(X_test_sel,  has_constant="add")

glm_nb = sm.GLM(y_train, X_train_glm, family=sm.families.NegativeBinomial())
glm_nb_res = glm_nb.fit()

y_train_pred_glm = glm_nb_res.predict(X_train_glm)
y_test_pred_glm  = glm_nb_res.predict(X_test_glm)

glm_pseudo_r2 = pseudo_r2_like_R(y_train, y_train_pred_glm, y_train_mean)
glm_mape = mape(y_test, y_test_pred_glm)
glm_rmse = rmse(y_test, y_test_pred_glm)

print("\nGLM Negative Binomial (on RF-selected features):")
print(f"  Pseudo-R² (train): {glm_pseudo_r2:.4f}")
print(f"  (Direct dev-based R-style: {1 - glm_nb_res.deviance / glm_nb_res.null_deviance:.4f})")
print(f"  MAPE (test, %):    {glm_mape:.2f}")
print(f"  RMSE (test):       {glm_rmse:.4f}")

results.append({
    "model": "GLM_NegativeBinomial",
    "pseudo_R2_train": glm_pseudo_r2,
    "MAPE_test": glm_mape,
    "RMSE_test": glm_rmse
})

# ============================================================
# 8. Summary table + save
# ============================================================

summary_df = pd.DataFrame(results)
print("\n==========================================")
print("Summary (3 models on RF-selected features):")
print(summary_df)

summary_df.to_csv("rf_selected_features_3models_comparison.csv", index=False)
print("\nSaved summary to 'rf_selected_features_3models_comparison.csv'")
