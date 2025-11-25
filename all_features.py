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
# 1. Load data
# ============================================================

# Use the preprocessed file you mentioned
df = pd.read_csv("df1_v1a_out.csv")

# Ensure target and split columns exist and drop missing
df = df.dropna(subset=["pm_tot", "holdout"])

target_col = "pm_tot"
split_col = "holdout"
id_cols = ["site_id"]  # optional: drop if present

# All feature columns = everything except target, holdout, and IDs
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
print("Number of features:", X_train.shape[1])

# ============================================================
# 3. Identify numeric and categorical features
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
        sparse=False  # ensure dense output
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Fit on train, transform train and test
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

print("Encoded train shape:", X_train_enc.shape)
print("Encoded test shape: ", X_test_enc.shape)

# ============================================================
# 5. Metric functions (MAPE, RMSE, pseudo-R2)
# ============================================================

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (%), similar to your R code:
    mean(abs((pred - obs) / obs) * 100)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    denom = np.where(y_true == 0, eps, y_true)
    return np.mean(np.abs((y_pred - y_true) / denom) * 100.0)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pseudo_r2_like_R(y_true, y_pred, y_null_mean):
    """
    Approximate the same structure as your R function:

        mf_r2 <- function(m) 1 - (m$deviance / m$null.deviance)

    Here we compute a Poisson deviance for:
      - full model (using y_pred as lambda_i)
      - null model (using constant lambda_0 = mean(y_train))

    Then pseudo-R2 = 1 - dev_full / dev_null
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-9
    mu = np.clip(y_pred, eps, None)
    mu0 = np.clip(float(y_null_mean), eps, None)

    def poisson_deviance(y, mu):
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        # D_i = 2 * [y_i * log(y_i / mu_i) - (y_i - mu_i)]
        # handle y_i == 0 safely
        term = np.where(
            y == 0,
            mu,  # limit case as y -> 0
            y * np.log(y / mu) - (y - mu)
        )
        return 2.0 * np.sum(term)

    dev_full = poisson_deviance(y_true, mu)
    dev_null = poisson_deviance(y_true, np.full_like(y_true, mu0))

    return 1.0 - (dev_full / dev_null)


y_train_mean = y_train.mean()

# ============================================================
# 6. Fit models with ALL FEATURES
#    - Random Forest
#    - HistGradientBoosting (Poisson)
#    - GLM Negative Binomial
# ============================================================

results = []

# -------------------- Random Forest -------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_enc, y_train)

y_train_pred_rf = rf.predict(X_train_enc)
y_test_pred_rf  = rf.predict(X_test_enc)

rf_pseudo_r2 = pseudo_r2_like_R(y_train, y_train_pred_rf, y_train_mean)  # train
rf_mape = mape(y_test, y_test_pred_rf)                                   # test
rf_rmse = rmse(y_test, y_test_pred_rf)                                   # test

print("\nRandom Forest:")
print(f"  Pseudo-R² (train): {rf_pseudo_r2:.4f}")
print(f"  MAPE (test, %):    {rf_mape:.2f}")
print(f"  RMSE (test):       {rf_rmse:.4f}")

results.append({
    "model": "RandomForest",
    "pseudo_R2_train": rf_pseudo_r2,
    "MAPE_test": rf_mape,
    "RMSE_test": rf_rmse
})

# ---------------- HistGradientBoosting (Poisson) ------------
hgb = HistGradientBoostingRegressor(
    loss="poisson",
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

hgb.fit(X_train_enc, y_train)

y_train_pred_hgb = hgb.predict(X_train_enc)
y_test_pred_hgb  = hgb.predict(X_test_enc)

hgb_pseudo_r2 = pseudo_r2_like_R(y_train, y_train_pred_hgb, y_train_mean)
hgb_mape = mape(y_test, y_test_pred_hgb)
hgb_rmse = rmse(y_test, y_test_pred_hgb)

print("\nHistGradientBoosting (Poisson):")
print(f"  Pseudo-R² (train): {hgb_pseudo_r2:.4f}")
print(f"  MAPE (test, %):    {hgb_mape:.2f}")
print(f"  RMSE (test):       {hgb_rmse:.4f}")

results.append({
    "model": "HistGB_Poisson",
    "pseudo_R2_train": hgb_pseudo_r2,
    "MAPE_test": hgb_mape,
    "RMSE_test": hgb_rmse
})

# ---------------- GLM Negative Binomial ---------------------
# Use statsmodels on the same encoded features
X_train_glm = sm.add_constant(X_train_enc, has_constant="add")
X_test_glm  = sm.add_constant(X_test_enc,  has_constant="add")

glm_nb = sm.GLM(y_train, X_train_glm, family=sm.families.NegativeBinomial())
glm_nb_res = glm_nb.fit()

y_train_pred_glm = glm_nb_res.predict(X_train_glm)
y_test_pred_glm  = glm_nb_res.predict(X_test_glm)

glm_pseudo_r2 = pseudo_r2_like_R(y_train, y_train_pred_glm, y_train_mean)
glm_mape = mape(y_test, y_test_pred_glm)
glm_rmse = rmse(y_test, y_test_pred_glm)

print("\nGLM Negative Binomial:")
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
# 7. Summary table + save as CSV
# ============================================================

summary_df = pd.DataFrame(results)
print("\n==========================================")
print("Summary table (all features, 3 models):")
print(summary_df)

summary_df.to_csv("all_features_model_comparison.csv", index=False)
print("\nSaved summary to 'all_features_model_comparison.csv'")
