import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# =========================
# Config
# =========================
DATA_FILE = "df1_v1a_out.csv"
SELECTED_FILE = "selected_features_by_method.csv"

OUTDIR = "model_coefficients_importance_by_selection"
os.makedirs(OUTDIR, exist_ok=True)

methods_to_compare = ["RandomForest", "L1_Lasso", "MutualInformation"]

# =========================
# 1) Load data
# =========================
df = pd.read_csv(DATA_FILE)
sel = pd.read_csv(SELECTED_FILE)

assert "pm_tot" in df.columns, "pm_tot not in data!"
assert "holdout" in df.columns, "holdout not in data!"
assert {"feature_selection_method", "feature_name"}.issubset(sel.columns), \
    "selected_features_by_method.csv must include: feature_selection_method, feature_name"

df = df.dropna(subset=["pm_tot", "holdout"])

target_col = "pm_tot"
split_col = "holdout"

# Drop columns you decided not to use
drop_cols = [
    target_col, split_col,
    "site_id",
    "geometry",
    "Street Nam",
    "_Date"
]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[target_col].values

train_mask = df[split_col] == 0
test_mask  = df[split_col] == 1

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"Train rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")
print(f"Raw features: {X_train.shape[1]}")

# =========================
# 2) Preprocess: numeric impute + categorical one-hot
# =========================
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

feature_names_enc = preprocessor.get_feature_names_out()
name_to_idx = {n: i for i, n in enumerate(feature_names_enc)}

print("Encoded features:", X_train_enc.shape[1])

# =========================
# 3) Metrics helpers (optional summary)
# =========================
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    eps = 1e-9
    denom = np.where(y_true == 0, eps, y_true)
    return np.mean(np.abs((y_pred - y_true) / denom) * 100.0)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def pseudo_r2_like_R(y_true, y_pred, y_null_mean):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    eps = 1e-9
    mu = np.clip(y_pred, eps, None)
    mu0 = np.clip(float(y_null_mean), eps, None)

    def poisson_deviance(y, mu):
        term = np.where(y == 0, mu, y*np.log(y/mu) - (y-mu))
        return 2.0*np.sum(term)

    dev_full = poisson_deviance(y_true, mu)
    dev_null = poisson_deviance(y_true, np.full_like(y_true, mu0))
    return 1.0 - dev_full / dev_null

y_train_mean = float(np.mean(y_train))

# =========================
# 4) Helper: get selected feature indices
# =========================
def get_selected(method_name: str):
    feats = sel.loc[sel["feature_selection_method"] == method_name, "feature_name"].tolist()
    if not feats:
        raise ValueError(f"No features found for '{method_name}' in {SELECTED_FILE}")

    present = [f for f in feats if f in name_to_idx]
    missing = [f for f in feats if f not in name_to_idx]
    if missing:
        print(f"[{method_name}] Warning: {len(missing)} features missing in current encoding (dropping them).")

    idx = np.array([name_to_idx[f] for f in present], dtype=int)
    return present, idx

# =========================
# 5) Train models per selection + save coefficients/importance
# =========================
metrics_rows = []

for method in methods_to_compare:
    feat_names, idx = get_selected(method)
    Xtr = X_train_enc[:, idx]
    Xte = X_test_enc[:, idx]

    method_dir = os.path.join(OUTDIR, method)
    os.makedirs(method_dir, exist_ok=True)

    print(f"\n=== {method}: {Xtr.shape[1]} features ===")

    # -------------------------
    # A) RandomForest importance
    # -------------------------
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    ytr_pred = rf.predict(Xtr)
    yte_pred = rf.predict(Xte)

    rf_imp = pd.DataFrame({
        "feature": feat_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    rf_imp.to_csv(os.path.join(method_dir, "RF_model_importance.csv"), index=False)

    metrics_rows.append({
        "selection": method,
        "model": "RandomForest",
        "n_features": Xtr.shape[1],
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
    })

    # -------------------------
    # B) HistGB Poisson importance (permutation importance)
    # -------------------------
    hgb = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    hgb.fit(Xtr, y_train)
    ytr_pred = hgb.predict(Xtr)
    yte_pred = hgb.predict(Xte)

    # Permutation importance: measures how much performance drops when a feature is shuffled
    perm = permutation_importance(
        hgb, Xte, y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )

    hgb_imp = pd.DataFrame({
        "feature": feat_names,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std
    }).sort_values("perm_importance_mean", ascending=False)

    hgb_imp.to_csv(os.path.join(method_dir, "HistGB_permutation_importance.csv"), index=False)

    metrics_rows.append({
        "selection": method,
        "model": "HistGB_Poisson",
        "n_features": Xtr.shape[1],
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
    })

    # -------------------------
    # C) GLM Negative Binomial coefficients
    # -------------------------
    Xtr_glm = sm.add_constant(Xtr, has_constant="add")
    Xte_glm = sm.add_constant(Xte, has_constant="add")

    glm_nb = sm.GLM(y_train, Xtr_glm, family=sm.families.NegativeBinomial())
    res = glm_nb.fit()

    ytr_pred = res.predict(Xtr_glm)
    yte_pred = res.predict(Xte_glm)

    coef_names = ["const"] + feat_names
    glm_df = pd.DataFrame({
        "feature": coef_names,
        "coef": res.params,
        "std_err": res.bse,
        "p_value": res.pvalues
    })
    glm_df["abs_coef"] = np.abs(glm_df["coef"])

    # Sort by |coef| so big negatives also show up at the top
    glm_df = glm_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    glm_df.to_csv(os.path.join(method_dir, "GLM_NB_coefficients.csv"), index=False)

    metrics_rows.append({
        "selection": method,
        "model": "GLM_NegativeBinomial",
        "n_features": Xtr.shape[1],
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
        "pseudo_R2_train_glmdev": float(1.0 - res.deviance / res.null_deviance),
    })

# =========================
# 6) Save combined metrics table
# =========================
metrics_df = pd.DataFrame(metrics_rows).sort_values(["selection", "model"]).reset_index(drop=True)
metrics_df.to_csv(os.path.join(OUTDIR, "metrics_selection_x_model.csv"), index=False)

print("\nSaved metrics table:", os.path.join(OUTDIR, "metrics_selection_x_model.csv"))
print("Saved per-method files under:", OUTDIR)
