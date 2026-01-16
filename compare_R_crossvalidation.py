# ============================================
# 3-stage model comparison with 5-fold CV
# Stages: nb1a, nb2a, nb3a feature sets
# Models: Random Forest, HistGB (Poisson), GLM Negative Binomial
# Metrics: McFadden pseudo-R2 (train fold), MAPE (val fold), RMSE (val fold)
# Data: df1_v1a_out.csv
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm


# -------------------------
# 1) Metrics
# -------------------------
# def mape(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     eps = 1e-9
#     denom = np.where(y_true == 0, eps, y_true)
#     return np.mean(np.abs((y_pred - y_true) / denom) * 100.0)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-9, denom)  # safe if both 0
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mcfadden_pseudo_r2_poisson_deviance(y_true, y_pred, y_null_mean):
    """
    Pseudo-R2 with Poisson deviance:
        1 - dev_full / dev_null
    Works as an approximation for non-GLM models too.
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


# -------------------------
# 2) Preprocessor builder
# -------------------------
def make_preprocessor(X, categorical_cols):
    """
    - numeric: median impute
    - categorical: most_frequent + one-hot
    """
    if len(categorical_cols) == 0:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        return ColumnTransformer([("num", num_pipe, X.columns.tolist())])

    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])


# -------------------------
# 3) Load data
# -------------------------
data_path = "df1_v1a_out.csv"
df = pd.read_csv(data_path)

# Only require pm_tot now (holdout is NOT used)
df = df.dropna(subset=["pm_tot"]).copy()

# -------------------------
# 4) Derived features to match your R formulas
# -------------------------
# Safer logs (avoid -inf if zeros)
df["log_Number"] = np.log(np.clip(df["Number of"].astype(float), 1e-9, None))
df["dist_CBD_mi"] = df["dist_CBD"].astype(float) / 5280.0
df["has_com_retail"] = ((df["Commercial Area"] + df["Retail Area"]) > 0).astype(int)
df["Retail_Area_qm_scaled"] = df["Retail Area_qm"].astype(float) / 10000.0
df["log_stv_ann_plus1"] = np.log(np.clip(df["stv_ann"].astype(float) + 1.0, 1e-9, None))

# -------------------------
# 5) Stage definitions (nb1a, nb2a, nb3a)
# -------------------------
features_nb1a = [
    "dist_CBD_mi",
    "log_Number",
    "has_com_retail",
    "crossing_class",
    "speed_type",
    "Retail_Area_qm_scaled",
    "transit_stops_qm"
]
categorical_nb1a = ["crossing_class", "speed_type"]

features_nb2a = [
    "log_Number",
    "log_stv_ann_plus1",
    "stv_mi_qm"
]
categorical_nb2a = []

features_nb3a = [
    "log_Number",
    "log_stv_ann_plus1",
    "dist_CBD",
    "has_com_retail",
    "crossing_class",
    "speed_type",
    "Retail Area_qm",
    "transit_stops_qm",
    "med_inc_qm"
]
categorical_nb3a = ["crossing_class", "speed_type"]

stage_definitions = [
    ("nb1a_features", features_nb1a, categorical_nb1a),
    ("nb2a_features", features_nb2a, categorical_nb2a),
    ("nb3a_features", features_nb3a, categorical_nb3a),
]

# -------------------------
# 6) Cross-validation setup
# -------------------------
K = 5
SEED = 42
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

# -------------------------
# 7) Evaluate function for each model
# -------------------------
def run_cv_for_stage_and_model(df, stage_name, feature_cols, cat_cols, model_name):
    """
    Returns fold-level results list of dicts.
    pseudo_R2 is computed on TRAIN fold.
    MAPE/RMSE computed on VAL fold.
    """
    X_all = df[feature_cols].copy()
    y_all = df["pm_tot"].astype(float).values

    fold_rows = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
        X_train = X_all.iloc[train_idx].copy()
        y_train = y_all[train_idx]
        X_val = X_all.iloc[val_idx].copy()
        y_val = y_all[val_idx]

        y_train_mean = float(np.mean(y_train))

        preprocessor = make_preprocessor(X_train, cat_cols)

        # ---- RandomForest ----
        if model_name == "RandomForest":
            model = Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=500,
                    random_state=SEED,
                    n_jobs=-1
                ))
            ])
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred   = model.predict(X_val)

            pseudo_r2_train = mcfadden_pseudo_r2_poisson_deviance(y_train, y_train_pred, y_train_mean)

        # ---- HistGB Poisson ----
        elif model_name == "HistGB_Poisson":
            model = Pipeline([
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_iter=300,
                    random_state=SEED
                ))
            ])
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred   = model.predict(X_val)

            pseudo_r2_train = mcfadden_pseudo_r2_poisson_deviance(y_train, y_train_pred, y_train_mean)

        # ---- GLM Negative Binomial ----
        elif model_name == "GLM_NegativeBinomial":
            # Fit-transform using same preprocessing as other models
            X_train_enc = preprocessor.fit_transform(X_train)
            X_val_enc   = preprocessor.transform(X_val)

            X_train_glm = sm.add_constant(X_train_enc, has_constant="add")
            X_val_glm   = sm.add_constant(X_val_enc, has_constant="add")

            glm = sm.GLM(y_train, X_train_glm, family=sm.families.NegativeBinomial())
            res = glm.fit()

            y_train_pred = res.predict(X_train_glm)
            y_val_pred   = res.predict(X_val_glm)

            # This is the true mf_r2 for the GLM (like R): 1 - deviance/null.deviance
            # (Most comparable to your R mf_r2 for glm.nb.)
            try:
                pseudo_r2_train = float(1.0 - res.deviance / res.null_deviance)
            except Exception:
                # fallback if null_deviance not present
                pseudo_r2_train = mcfadden_pseudo_r2_poisson_deviance(y_train, y_train_pred, y_train_mean)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        fold_rows.append({
            "stage": stage_name,
            "model": model_name,
            "fold": fold_i,
            "pseudo_R2_train": float(pseudo_r2_train),
            "MAPE_val": float(mape(y_val, y_val_pred)),
            "RMSE_val": float(rmse(y_val, y_val_pred)),
        })

    return fold_rows


# -------------------------
# 8) Run CV across stages + models
# -------------------------
all_fold_results = []

models = ["RandomForest", "HistGB_Poisson", "GLM_NegativeBinomial"]

for stage_name, feature_cols, cat_cols in stage_definitions:
    print("=" * 70)
    print(f"Stage: {stage_name}")
    print("Features:", feature_cols)
    print("Categorical:", cat_cols)

    for model_name in models:
        fold_rows = run_cv_for_stage_and_model(df, stage_name, feature_cols, cat_cols, model_name)
        all_fold_results.extend(fold_rows)

fold_df = pd.DataFrame(all_fold_results)
fold_df.to_csv("cv_fold_results.csv", index=False)

# Summary mean/std across folds
summary_df = (
    fold_df
    .groupby(["stage", "model"], as_index=False)
    .agg(
        pseudo_R2_train_mean=("pseudo_R2_train", "mean"),
        pseudo_R2_train_std=("pseudo_R2_train", "std"),
        MAPE_val_mean=("MAPE_val", "mean"),
        MAPE_val_std=("MAPE_val", "std"),
        RMSE_val_mean=("RMSE_val", "mean"),
        RMSE_val_std=("RMSE_val", "std"),
    )
)

summary_df.to_csv("cv_summary_results.csv", index=False)

print("\nSaved:")
print(" - cv_fold_results.csv")
print(" - cv_summary_results.csv")
print("\nSummary:")
print(summary_df)
