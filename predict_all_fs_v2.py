import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

# =========================
# 0) Files
# =========================
DATA_FILE = "df1_v1a_out.csv"
SELECTED_FILE = "selected_features_by_method.csv"

# =========================
# 1) Load data
# =========================
df = pd.read_csv(DATA_FILE)
sel = pd.read_csv(SELECTED_FILE)

assert "pm_tot" in df.columns
assert "holdout" in df.columns
assert {"feature_selection_method", "feature_name"}.issubset(sel.columns)

df = df.dropna(subset=["pm_tot", "holdout"])

target_col = "pm_tot"
split_col = "holdout"

# drop leakage/high-cardinality cols as you decided
drop_cols = [target_col, split_col, "site_id", "geometry", "Street Nam", "_Date"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[target_col].values

train_mask = df[split_col] == 0
test_mask  = df[split_col] == 1

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]
y_train_mean = float(np.mean(y_train))

# =========================
# 2) Preprocess: impute + one-hot
# =========================
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)
feature_names_enc = preprocessor.get_feature_names_out()
name_to_idx = {n: i for i, n in enumerate(feature_names_enc)}

print("Encoded shape train:", X_train_enc.shape, " test:", X_test_enc.shape)

# =========================
# 3) Metrics
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
    return 1.0 - dev_full/dev_null

# =========================
# 4) Helper: get selected indices for a method
# =========================
def get_selected(method):
    feats = sel.loc[sel["feature_selection_method"] == method, "feature_name"].tolist()
    if not feats:
        raise ValueError(f"No features for method {method} in {SELECTED_FILE}")
    present = [f for f in feats if f in name_to_idx]
    missing = [f for f in feats if f not in name_to_idx]
    if missing:
        print(f"[{method}] Warning: {len(missing)} features missing in current encoding. Dropping them.")
    idx = np.array([name_to_idx[f] for f in present], dtype=int)
    return present, idx

# =========================
# 5) Compute "feature selection scores" for ALL features,
#    then we will subset to the selected features
# =========================
print("\nComputing MI scores (train)...")
mi_scores_all = mutual_info_regression(X_train_enc, y_train, random_state=42)
mi_scores_all = np.nan_to_num(mi_scores_all, nan=0.0)

print("Computing LassoCV coefficients (train, standardized)...")
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_std = scaler.fit_transform(X_train_enc)
lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_train_std, y_train)
lasso_coef_all = lasso.coef_

print("Computing RF-selection importances (train)...")
rf_selector = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_enc, y_train)
rf_sel_importance_all = rf_selector.feature_importances_

# =========================
# 6) Train 3 models per feature-set + save importance/coefficients
# =========================
methods_to_compare = ["MutualInformation", "L1_Lasso", "RandomForest"]

metrics_rows = []
selection_score_rows = []   # MI/L1/RF-selection scores for selected features
rf_model_imp_rows = []      # RF model importance (trained on selected features)
hgb_perm_imp_rows = []      # HGB permutation importance (trained on selected features)
glm_coef_rows = []          # GLM coefficients (trained on selected features)

for method in methods_to_compare:
    feat_names, idx = get_selected(method)
    Xtr = X_train_enc[:, idx]
    Xte = X_test_enc[:, idx]
    k = Xtr.shape[1]
    print(f"\n=== {method}: using {k} selected features ===")

    # ---- save feature-selection scores for the selected features ----
    for f in feat_names:
        j = name_to_idx[f]
        selection_score_rows.append({
            "selection_method": method,
            "feature": f,
            "MI_score": float(mi_scores_all[j]),
            "L1_coef": float(lasso_coef_all[j]),              # coef on standardized inputs
            "RFsel_importance": float(rf_sel_importance_all[j])
        })

    # ------------------ Model 1: RandomForest ------------------
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    ytr_pred = rf.predict(Xtr)
    yte_pred = rf.predict(Xte)

    metrics_rows.append({
        "selection": method, "model": "RandomForest", "n_features": k,
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred)
    })

    # RF model importance for these selected features
    for f, imp in zip(feat_names, rf.feature_importances_):
        rf_model_imp_rows.append({
            "selection": method,
            "model": "RandomForest",
            "feature": f,
            "importance": float(imp)
        })

    # ------------------ Model 2: HistGB Poisson ------------------
    hgb = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, random_state=42)
    hgb.fit(Xtr, y_train)
    ytr_pred = hgb.predict(Xtr)
    yte_pred = hgb.predict(Xte)

    metrics_rows.append({
        "selection": method, "model": "HistGB_Poisson", "n_features": k,
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred)
    })

    # HGB permutation importance (on test set; more meaningful than impurity-based)
    perm = permutation_importance(
        hgb, Xte, y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )
    for f, mean_imp, std_imp in zip(feat_names, perm.importances_mean, perm.importances_std):
        hgb_perm_imp_rows.append({
            "selection": method,
            "model": "HistGB_Poisson",
            "feature": f,
            "perm_importance_mean": float(mean_imp),
            "perm_importance_std": float(std_imp)
        })

    # ------------------ Model 3: GLM Negative Binomial ------------------
    Xtr_glm = sm.add_constant(Xtr, has_constant="add")
    Xte_glm = sm.add_constant(Xte, has_constant="add")
    glm_nb = sm.GLM(y_train, Xtr_glm, family=sm.families.NegativeBinomial())
    res = glm_nb.fit()

    ytr_pred = res.predict(Xtr_glm)
    yte_pred = res.predict(Xte_glm)

    metrics_rows.append({
        "selection": method, "model": "GLM_NegativeBinomial", "n_features": k,
        "pseudo_R2_train": pseudo_r2_like_R(y_train, ytr_pred, y_train_mean),
        "MAPE_test": mape(y_test, yte_pred),
        "RMSE_test": rmse(y_test, yte_pred),
        "pseudo_R2_train_glmdev": float(1.0 - res.deviance / res.null_deviance)
    })

    # GLM coefficients with names
    coef_names = ["const"] + feat_names
    for nm, b, se, pv in zip(coef_names, res.params, res.bse, res.pvalues):
        glm_coef_rows.append({
            "selection": method,
            "model": "GLM_NegativeBinomial",
            "feature": nm,
            "coef": float(b),
            "std_err": float(se),
            "p_value": float(pv)
        })

# =========================
# 7) Save outputs
# =========================
OUTDIR = "importance_by_selection_method"
os.makedirs(OUTDIR, exist_ok=True)

# 1) Metrics table (single file)
metrics_df = pd.DataFrame(metrics_rows).sort_values(["selection", "model"])
metrics_df.to_csv("selection_x_models_metrics.csv", index=False)
print("Saved: selection_x_models_metrics.csv")

# 2) Save per-selection feature-selection scores (MI / L1 coef / RFsel importance)
selection_scores_df = pd.DataFrame(selection_score_rows)

for method in methods_to_compare:
    df_m = selection_scores_df[selection_scores_df["selection_method"] == method].copy()

    # Sort by the score that corresponds to that method
    if method == "MutualInformation":
        df_m = df_m.sort_values("MI_score", ascending=False)
    elif method == "L1_Lasso":
        df_m = df_m.assign(abs_L1=np.abs(df_m["L1_coef"])).sort_values("abs_L1", ascending=False).drop(columns=["abs_L1"])
    elif method == "RandomForest":
        df_m = df_m.sort_values("RFsel_importance", ascending=False)

    outpath = os.path.join(OUTDIR, f"selected_feature_scores__{method}.csv")
    df_m.to_csv(outpath, index=False)
    print("Saved:", outpath)

# 3) Save per-selection model importances (RF model)
rf_imp_df = pd.DataFrame(rf_model_imp_rows)

for method in methods_to_compare:
    df_m = rf_imp_df[(rf_imp_df["selection"] == method) & (rf_imp_df["model"] == "RandomForest")].copy()
    df_m = df_m.sort_values("importance", ascending=False)
    outpath = os.path.join(OUTDIR, f"importances__{method}__RF_model.csv")
    df_m.to_csv(outpath, index=False)
    print("Saved:", outpath)

# 4) Save per-selection model importances (HGB permutation)
hgb_perm_df = pd.DataFrame(hgb_perm_imp_rows)

for method in methods_to_compare:
    df_m = hgb_perm_df[(hgb_perm_df["selection"] == method) & (hgb_perm_df["model"] == "HistGB_Poisson")].copy()
    df_m = df_m.sort_values("perm_importance_mean", ascending=False)
    outpath = os.path.join(OUTDIR, f"importances__{method}__HGB_permutation.csv")
    df_m.to_csv(outpath, index=False)
    print("Saved:", outpath)

# 5) Save per-selection GLM coefficients
glm_coef_df = pd.DataFrame(glm_coef_rows)

for method in methods_to_compare:
    df_m = glm_coef_df[(glm_coef_df["selection"] == method) & (glm_coef_df["model"] == "GLM_NegativeBinomial")].copy()

    # Sort by absolute coefficient (most influential)
    df_m = df_m.assign(abs_coef=np.abs(df_m["coef"])).sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

    outpath = os.path.join(OUTDIR, f"importances__{method}__GLM_coefficients.csv")
    df_m.to_csv(outpath, index=False)
    print("Saved:", outpath)

print("\nAll per-method importance files saved under:", OUTDIR)
