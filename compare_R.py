# ============================================
# 3-stage model comparison (nb1a, nb2a, nb3a feature sets)
# Models: Random Forest, HistGradientBoostingRegressor (Poisson loss)
# Metrics: McFadden's pseudo-R2, MAPE, RMSE
# Train / Test split: holdout == 0 (train), holdout == 1 (test)
# Data file: processed.csv
# ============================================
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# -------------------------
# 1. Helper metric functions
# -------------------------

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (%), like your R holdout code:
    mean(abs((pred - obs) / obs) * 100)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # avoid division by zero (if any pm_tot == 0, you may adjust this)
    eps = 1e-9
    denom = np.where(y_true == 0, eps, y_true)
    return np.mean(np.abs((y_pred - y_true) / denom) * 100.0)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mcfadden_pseudo_r2_poisson(y_true, y_pred, y_null_mean):
    """
    McFadden's pseudo-R2 for Poisson-type models:
        R2 = 1 - (logLik_full / logLik_null)

    Here we approximate Poisson log-likelihood:
        logLik = sum(y_i * log(lambda_i) - lambda_i)
    The constant -log(y_i!) cancels in the ratio.

    y_null_mean is the mean of y in the TRAIN set (scalar).
    We can evaluate R2 on the TEST set using:
        - y_true_test
        - y_pred_test
        - y_null_mean (from train)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    eps = 1e-9
    lam_pred = np.clip(y_pred, eps, None)
    lam_null = np.clip(float(y_null_mean), eps, None)

    # full model log-likelihood
    loglik_full = np.sum(y_true * np.log(lam_pred) - lam_pred)

    # null model log-likelihood (same mean lambda for all points)
    loglik_null = np.sum(y_true * np.log(lam_null) - lam_null)

    # to be safe if loglik_null is zero or positive
    if loglik_null == 0:
        return np.nan

    return 1.0 - (loglik_full / loglik_null)


# -------------------------
# 2. Load data
# -------------------------

# If you want to use the raw file instead, replace with: 'df1_v1a_out.csv'
#data_path = "processed.csv"
PROCESSED_DIR = "./processed"
data_path = os.path.join(PROCESSED_DIR, "processed.csv")
df = pd.read_csv(data_path)

# we assume these columns exist: 'pm_tot', 'holdout', and all feature columns
# drop any rows where pm_tot or holdout is missing
df = df.dropna(subset=["pm_tot", "holdout"])

# -------------------------
# 3. Create derived features to match nb1a/nb2a/nb3a
# -------------------------

# log(Number of)
df["log_Number"] = np.log(df["Number of"])

# distance in miles (for nb1a)
df["dist_CBD_mi"] = df["dist_CBD"] / 5280.0

# indicator: any commercial or retail area > 0
df["has_com_retail"] = ((df["Commercial Area"] + df["Retail Area"]) > 0).astype(int)

# scaled Retail Area for nb1a
df["Retail_Area_qm_scaled"] = df["Retail Area_qm"] / 10000.0

# log(stv_ann + 1)
df["log_stv_ann_plus1"] = np.log(df["stv_ann"] + 1.0)

# -------------------------
# 4. Define feature sets corresponding to nb1a, nb2a, nb3a
# -------------------------

# Stage 1: nb1a feature set
features_nb1a = [
    "dist_CBD_mi",          # I(dist_CBD / 5280)
    "log_Number",           # log(`Number of`)
    "has_com_retail",       # I((Commercial + Retail) > 0)
    "crossing_class",       # factor(crossing_class)
    "speed_type",           # factor(speed_type)
    "Retail_Area_qm_scaled",# I(Retail Area_qm / 10000)
    "transit_stops_qm"      # transit_stops_qm
]

categorical_nb1a = ["crossing_class", "speed_type"]

# Stage 2: nb2a feature set
features_nb2a = [
    "log_Number",           # log(`Number of`)
    "log_stv_ann_plus1",    # log(stv_ann + 1)
    "stv_mi_qm"             # stv_mi_qm
]

categorical_nb2a = []  # all numeric

# Stage 3: nb3a feature set
features_nb3a = [
    "log_Number",           # log(`Number of`)
    "log_stv_ann_plus1",    # log(stv_ann + 1)
    "dist_CBD",             # dist_CBD (raw)
    "has_com_retail",       # I((Commercial + Retail) > 0)
    "crossing_class",       # factor(crossing_class)
    "speed_type",           # factor(speed_type)
    "Retail Area_qm",       # Retail Area_qm (raw)
    "transit_stops_qm",     # transit_stops_qm
    "med_inc_qm"            # med_inc_qm
]

categorical_nb3a = ["crossing_class", "speed_type"]

# packed for easy looping
stage_definitions = [
    ("nb1a_features", features_nb1a, categorical_nb1a),
    ("nb2a_features", features_nb2a, categorical_nb2a),
    ("nb3a_features", features_nb3a, categorical_nb3a),
]

# -------------------------
# 5. Train / Test split by holdout
# -------------------------

train_df = df[df["holdout"] == 0].copy()
test_df  = df[df["holdout"] == 1].copy()

y_train = train_df["pm_tot"].values
y_test  = test_df["pm_tot"].values

y_train_mean = y_train.mean()  # for null model in pseudo-R2

# -------------------------
# 6. Function to build preprocessor for each stage
# -------------------------

def make_preprocessor(X, categorical_cols):
    """
    Build a ColumnTransformer that:
    - one-hot encodes categorical columns
    - imputes numeric columns with median
    """
    if len(categorical_cols) == 0:
        # only numeric features
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, X.columns.tolist())
            ]
        )
    else:
        numeric_cols = [c for c in X.columns if c not in categorical_cols]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )
    return preprocessor


# -------------------------
# 7. Loop over stages & models and report metrics
# -------------------------

results = []

for stage_name, feature_cols, cat_cols in stage_definitions:
    print("=" * 70)
    print(f"Stage: {stage_name}")
    print("Using features:", feature_cols)
    print("Categorical:", cat_cols)

    # subset X for this stage
    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    # build preprocessor
    preprocessor = make_preprocessor(X_train, cat_cols)

    # ---- Random Forest ----
    rf_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rf_r2   = mcfadden_pseudo_r2_poisson(y_test, y_pred_rf, y_train_mean)
    rf_mape = mape(y_test, y_pred_rf)
    rf_rmse = rmse(y_test, y_pred_rf)

    print("\nRandom Forest:")
    print(f"  McFadden pseudo-R2: {rf_r2:.4f}")
    print(f"  MAPE (%):           {rf_mape:.2f}")
    print(f"  RMSE:               {rf_rmse:.4f}")

    results.append({
        "stage": stage_name,
        "model": "RandomForest",
        "pseudo_R2": rf_r2,
        "MAPE": rf_mape,
        "RMSE": rf_rmse
    })

    # ---- HistGradientBoosting (Poisson) ----
    hgb_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=None,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        ))
    ])

    hgb_model.fit(X_train, y_train)
    y_pred_hgb = hgb_model.predict(X_test)

    hgb_r2   = mcfadden_pseudo_r2_poisson(y_test, y_pred_hgb, y_train_mean)
    hgb_mape = mape(y_test, y_pred_hgb)
    hgb_rmse = rmse(y_test, y_pred_hgb)

    print("\nHistGradientBoosting (Poisson):")
    print(f"  McFadden pseudo-R2: {hgb_r2:.4f}")
    print(f"  MAPE (%):           {hgb_mape:.2f}")
    print(f"  RMSE:               {hgb_rmse:.4f}")

    results.append({
        "stage": stage_name,
        "model": "HistGB_Poisson",
        "pseudo_R2": hgb_r2,
        "MAPE": hgb_mape,
        "RMSE": hgb_rmse
    })

print("\n" + "=" * 70)
print("Summary table:")
summary_df = pd.DataFrame(results)
print(summary_df)
