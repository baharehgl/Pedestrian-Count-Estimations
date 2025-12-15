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
# 1) Load data + selected features list
# ============================================================

df = pd.read_csv("df1_v1a_out.csv")
sel_feats_df = pd.read_csv("selected_features_by_method.csv")

# Ensure required columns exist
assert "pm_tot" in df.columns, "pm_tot not in data!"
assert "holdout" in df.columns, "holdout not in data!"
assert {"feature_selection_method", "feature_name"}.issubset(sel_feats_df.columns), \
    "selected_features_by_method.csv must have columns: feature_selection_method, feature_name"

# Drop rows with missing target or holdout
df = df.dropna(subset=["pm_tot", "holdout"])

target_col = "pm_tot"
split_col = "holdout"

# Drop high-cardinality / leakage-ish columns (as you decided)
drop_cols = [
    target_col,
    split_col,
    "site_id",
    "geometry",
    "Street Nam",
    "_Date",
]

# All raw feature columns
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df[target_col].values

# ============================================================
# 2) Train/test split by holdout
# ============================================================

train_mask = df[split_col] == 0
test_mask  = df[split_col] == 1

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Raw feature count: {X_train.shape[1]}")

y_train_mean = float(np.mean(y_train))

# ============================================================
# 3) Preprocess: numeric impute + categorical one-hot
# ============================================================

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False))
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
name_to_idx = {name: i for i, name in enumerate(feature_names_enc)}

print("Encoded train shape:", X_train_enc.shape)

# ============================================================
# 4) Metrics: MAPE, RMSE, pseudo-R2 like your R structure (TRAIN)
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
    Pseudo-R2 = 1 - dev_full / dev_null
    using Poisson deviance computed from predictions.
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
# 5) Helper: get indices for a selection method
# ============================================================

def get_indices_for_method(method_name: str):
    feats = sel_feats_df.loc[
        sel_feats_df["feature_selection_method"] == method_name,
        "feature_name"
    ].tolist()

    if len(feats) == 0:
        raise ValueError(f"No features found for method '{method_name}' in selected_features_by_method.csv")

    # Filter out features not present (can happen if preprocessing differs)
    present = [f for f in feats if f in name_to_idx]
    missing = [f for f in feats if f not in name_to_idx]

    if missing:
        print(f"[{method_name}] Warning: {len(missing)} selected features not found in current encoding. Dropping them.")
        # Uncomment to see them:
        # for m in missing[:20]: print("  -", m)

    idx = np.array([name_to_idx[f] for f in present], dtype=int)
    return present, idx

# ============================================================
# 6) Train & evaluate 3 models on a given feature matrix
# ============================================================

def eval_three_models(Xtr, Xte, selection_name: str):
    out = []

    # ---- RandomForest ----
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    ytr_pred = rf.predict(Xtr)
    yte_pred = rf.predict(Xte)

    out.append({
        "selection": selection_name,
        "model": "RandomForest",
        "n_features": Xtr.shape[1],
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
    })

    # ---- HistGB (Poisson) ----
    hgb = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, random_state=42)
    hgb.fit(Xtr, y_train)
    ytr_pred = hgb.predict(Xtr)
    yte_pred = hgb.predict(Xte)

    out.append({
        "selection": selection_name,
        "model": "HistGB_Poisson",
        "n_features": Xtr.shape[1],
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
    })

    # ---- GLM Negative Binomial ----
    # statsmodels expects 2D numeric matrix
    Xtr_glm = sm.add_constant(Xtr, has_constant="add")
    Xte_glm = sm.add_constant(Xte, has_constant="add")

    glm_nb = sm.GLM(y_train, Xtr_glm, family=sm.families.NegativeBinomial())
    glm_nb_res = glm_nb.fit()

    ytr_pred = glm_nb_res.predict(Xtr_glm)
    yte_pred = glm_nb_res.predict(Xte_glm)

    out.append({
        "selection": selection_name,
        "model": "GLM_NegativeBinomial",
        "n_features": Xtr.shape[1],
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
        # optional extra (true GLM deviance-based)
        "pseudo_R2_train_glmdev": 1.0 - (glm_nb_res.deviance / glm_nb_res.null_deviance)
    })

    return out

# ============================================================
# 7) Run: RF vs L1 vs MI selected features (same 3 models each)
# ============================================================

methods_to_compare = ["RandomForest", "L1_Lasso", "MutualInformation"]  # add "All_features" if you want

all_results = []

for method in methods_to_compare:
    selected_names, idx = get_indices_for_method(method)
    print(f"\n[{method}] Using {len(idx)} encoded features.")

    X_train_sel = X_train_enc[:, idx]
    X_test_sel  = X_test_enc[:, idx]

    all_results.extend(eval_three_models(X_train_sel, X_test_sel, selection_name=method))

# ============================================================
# 8) Summary table + save
# ============================================================

summary_df = pd.DataFrame(all_results)

# Nice ordering
summary_df = summary_df.sort_values(by=["selection", "model"]).reset_index(drop=True)

print("\n==================== FINAL SUMMARY ====================")
print(summary_df)

summary_df.to_csv("selected_features_methods_x_models_comparison.csv", index=False)
print("\nSaved: selected_features_methods_x_models_comparison.csv")
