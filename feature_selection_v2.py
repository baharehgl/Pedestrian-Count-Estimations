import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV

# ============================================================
# 1. Load data and define features / target
# ============================================================

df = pd.read_csv("df1_v1a_out.csv")

# Make sure required columns exist
assert "pm_tot" in df.columns, "pm_tot not in data!"
assert "holdout" in df.columns, "holdout not in data!"

# Drop rows with missing target or holdout
df = df.dropna(subset=["pm_tot", "holdout"])

target_col = "pm_tot"
split_col = "holdout"
id_cols = ["site_id"]  # optional

# All features = everything except pm_tot, holdout, and optional IDs
feature_cols = [
    c for c in df.columns
    if c not in [target_col, split_col] + [c for c in id_cols if c in df.columns]
]

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
print("Total features:", X_train.shape[1])

y_train_mean = y_train.mean()

# ============================================================
# 3. Identify numeric and categorical columns
# ============================================================

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

# ============================================================
# 4. Preprocessor (impute + one-hot encode)
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse=False   # dense so we can work with numpy
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Fit on train only, transform train & test
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

# Get encoded feature names (so we know what we select)
feature_names_enc = preprocessor.get_feature_names_out()

print("Encoded train shape:", X_train_enc.shape)
print("Encoded feature example:", feature_names_enc[:10])

# ============================================================
# 5. Metrics: MAPE, RMSE, pseudo-R2 like R (train)
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
# 6. Base model to evaluate feature sets: HistGB_Poisson
# ============================================================

def train_and_evaluate_model(X_train_sel, X_test_sel, y_train, y_test, y_train_mean):
    """
    Train HistGradientBoostingRegressor (Poisson) on selected features
    and compute metrics.
    """
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_sel, y_train)

    y_train_pred = model.predict(X_train_sel)
    y_test_pred  = model.predict(X_test_sel)

    r2_train = pseudo_r2_like_R(y_train, y_train_pred, y_train_mean)
    mape_test = mape(y_test, y_test_pred)
    rmse_test = rmse(y_test, y_test_pred)

    return r2_train, mape_test, rmse_test


results = []
selected_features_dict = {}

# ============================================================
# 7. Baseline: ALL FEATURES (no feature selection)
# ============================================================

print("\n================ BASELINE: ALL FEATURES ================")

r2_all, mape_all, rmse_all = train_and_evaluate_model(
    X_train_enc, X_test_enc, y_train, y_test, y_train_mean
)

print(f"Baseline (All features) - Pseudo-R² train: {r2_all:.4f}, "
      f"MAPE test: {mape_all:.2f}, RMSE test: {rmse_all:.4f}")

results.append({
    "feature_selection": "All_features",
    "n_features": X_train_enc.shape[1],
    "pseudo_R2_train": r2_all,
    "MAPE_test": mape_all,
    "RMSE_test": rmse_all
})

selected_features_dict["All_features"] = list(feature_names_enc)

# ============================================================
# 8. Feature Selection 1: Univariate Mutual Information
# ============================================================

print("\n================ Univariate Mutual Information ================")

mi_scores = mutual_info_regression(X_train_enc, y_train, random_state=42)
mi_scores = np.nan_to_num(mi_scores, nan=0.0)

k = min(20, X_train_enc.shape[1])  # top-k features
mi_indices = np.argsort(mi_scores)[::-1][:k]

mi_selected_names = feature_names_enc[mi_indices]
print(f"Top {k} features by MI:")
for name, score in zip(mi_selected_names, mi_scores[mi_indices]):
    print(f"  {name}: {score:.4f}")

X_train_mi = X_train_enc[:, mi_indices]
X_test_mi  = X_test_enc[:, mi_indices]

r2_mi, mape_mi, rmse_mi = train_and_evaluate_model(
    X_train_mi, X_test_mi, y_train, y_test, y_train_mean
)

print(f"MI - Pseudo-R² train: {r2_mi:.4f}, MAPE test: {mape_mi:.2f}, RMSE test: {rmse_mi:.4f}")

results.append({
    "feature_selection": "MutualInformation",
    "n_features": len(mi_indices),
    "pseudo_R2_train": r2_mi,
    "MAPE_test": mape_mi,
    "RMSE_test": rmse_mi
})

selected_features_dict["MutualInformation"] = mi_selected_names.tolist()

# ============================================================
# 9. Feature Selection 2: L1 (Lasso)
# ============================================================

print("\n================ L1 (Lasso) ================")

# Standardize features for Lasso (important!)
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_std = scaler.fit_transform(X_train_enc)
X_test_std  = scaler.transform(X_test_enc)

lasso = LassoCV(
    cv=5,
    random_state=42,
    n_jobs=-1
)
lasso.fit(X_train_std, y_train)

coef = lasso.coef_
# select non-zero coefficients
lasso_indices = np.where(np.abs(coef) > 1e-6)[0]

if len(lasso_indices) == 0:
    # fallback: if everything shrinks to 0, take top |coef|
    print("Warning: Lasso shrank all coefficients to zero; using top 20 by |coef| as fallback.")
    k_lasso = min(20, len(coef))
    lasso_indices = np.argsort(np.abs(coef))[::-1][:k_lasso]

lasso_selected_names = feature_names_enc[lasso_indices]

print(f"Lasso selected {len(lasso_indices)} features:")
for name, c in zip(lasso_selected_names, coef[lasso_indices]):
    print(f"  {name}: coef={c:.4f}")

X_train_lasso = X_train_enc[:, lasso_indices]
X_test_lasso  = X_test_enc[:, lasso_indices]

r2_lasso, mape_lasso, rmse_lasso = train_and_evaluate_model(
    X_train_lasso, X_test_lasso, y_train, y_test, y_train_mean
)

print(f"L1 (Lasso) - Pseudo-R² train: {r2_lasso:.4f}, "
      f"MAPE test: {mape_lasso:.2f}, RMSE test: {rmse_lasso:.4f}")

results.append({
    "feature_selection": "L1_Lasso",
    "n_features": len(lasso_indices),
    "pseudo_R2_train": r2_lasso,
    "MAPE_test": mape_lasso,
    "RMSE_test": rmse_lasso
})

selected_features_dict["L1_Lasso"] = lasso_selected_names.tolist()

# ============================================================
# 10. Feature Selection 3: Random Forest importance
# ============================================================

print("\n================ Random Forest Importance ================")

rf_fs = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf_fs.fit(X_train_enc, y_train)

importances = rf_fs.feature_importances_
k_rf = min(20, X_train_enc.shape[1])
rf_indices = np.argsort(importances)[::-1][:k_rf]

rf_selected_names = feature_names_enc[rf_indices]

print(f"Top {k_rf} features by RF importance:")
for name, imp in zip(rf_selected_names, importances[rf_indices]):
    print(f"  {name}: {imp:.4f}")

X_train_rf = X_train_enc[:, rf_indices]
X_test_rf  = X_test_enc[:, rf_indices]

r2_rf, mape_rf, rmse_rf = train_and_evaluate_model(
    X_train_rf, X_test_rf, y_train, y_test, y_train_mean
)

print(f"RF FS - Pseudo-R² train: {r2_rf:.4f}, "
      f"MAPE test: {mape_rf:.2f}, RMSE test: {rmse_rf:.4f}")

results.append({
    "feature_selection": "RandomForest",
    "n_features": len(rf_indices),
    "pseudo_R2_train": r2_rf,
    "MAPE_test": mape_rf,
    "RMSE_test": rmse_rf
})

selected_features_dict["RandomForest"] = rf_selected_names.tolist()

# ============================================================
# 11. Summary table + save (metrics AND selected features)
# ============================================================

summary_df = pd.DataFrame(results)
print("\n================ SUMMARY ==================")
print(summary_df)

# 1) Metrics CSV
summary_df.to_csv("feature_selection_comparison.csv", index=False)
print("\nSaved metrics to 'feature_selection_comparison.csv'")

# 2) Selected features CSV (long format: one row per feature per method)
rows = []
for method, feats in selected_features_dict.items():
    for f in feats:
        rows.append({
            "feature_selection_method": method,
            "feature_name": f
        })

selected_features_df = pd.DataFrame(rows)
selected_features_df.to_csv("selected_features_by_method.csv", index=False)
print("Saved selected features to 'selected_features_by_method.csv'")