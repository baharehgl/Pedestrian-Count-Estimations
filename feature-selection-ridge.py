# === Feature Selection + RIDGE Regression (on preprocessed X.csv / y.csv) ===
# Requirements: pandas, numpy, scikit-learn, matplotlib
# No feature expansion, no imputation (assumes none needed).
# Metrics: MAE (↓), RMSE (↓), R² (↑), Poisson deviance (↓).
# ---------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_poisson_deviance, make_scorer
)

# ------------------------ Config ------------------------
PROCESSED_DIR = "./processed"   # <-- set to your preprocessing output folder
X_PATH = os.path.join(PROCESSED_DIR, "X.csv")
Y_PATH = os.path.join(PROCESSED_DIR, "y.csv")
TARGET_COL = "pm_tot"

OUTPUT_DIR = "./fs_ridge_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5                 # reduce to 3 for speed if needed
K_BEST = None                 # if None -> min(15, max(5, n_features//4))
RIDGE_ALPHAS = np.logspace(-3, 3, 50)  # CV over these L2 strengths
# --------------------------------------------------------

# --------- Load preprocessed features/target (numeric only) ----------
if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
    raise FileNotFoundError(
        f"Could not find {X_PATH} or {Y_PATH}. Run preprocessing first."
    )

X = pd.read_csv(X_PATH)
y_df = pd.read_csv(Y_PATH)
if TARGET_COL not in y_df.columns:
    TARGET_COL = y_df.columns[0]  # fallback to first column if name differs
y = y_df[TARGET_COL].astype(float).values

n_samples, n_features = X.shape
feat_names = list(X.columns)
if K_BEST is None:
    K_BEST = max(5, min(15, n_features // 4))

print(f"[INFO] Loaded X: {X.shape}, y: {y.shape}. Using K_BEST={K_BEST}")

# ------------------- Feature selectors -------------------
mi_fn = partial(mutual_exclude if False else mutual_info_regression, random_state=RANDOM_STATE)

def make_ridge():
    # Ridge with internal CV (on each training fold only — no leakage)
    return RidgeCV(alphas=RIDGE_ALPHAS, cv=5)

# All pipelines end with the SAME model (Ridge) to keep comparison fair.
pipelines = {
    # 1) Univariate Mutual Information -> top-K -> Standardize -> Ridge
    "MI(TopK)->Ridge": Pipeline([
        ("select", SelectKBest(score_func=mi_fn, k=K_BEST)),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", make_ridge()),
    ]),

    # 2) L1 (Lasso) as selector on scaled features -> Ridge
    "LassoSelect->Ridge": Pipeline([
        ("scale_sel", StandardScaler(with_mean=True, with_std=True)),
        ("select", SelectFromModel(
            estimator=LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
            threshold="median",   # keep features with |coef| above median
            prefit=False
        )),
        ("scale_model", StandardScaler(with_mean=True, with_std=True)),
        ("model", make_ridge()),
    ]),

    # 3) RF importance selector (no scaling needed for RF) -> Standardize -> Ridge
    "RFSelect->Ridge": Pipeline([
        ("select", SelectFromModel(
            estimator=RandomForestRegressor(
                n_estimators=500, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
            ),
            threshold="median",   # keep features with importance above median
            prefit=False
        )),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", make_ridge()),
    ]),
}

# ------------------- Scoring & CV -------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def poisson_dev(y_true, y_pred):
    # Ensure positive predictions for Poisson deviance
    y_pred = np.clip(y_pred, 1e-9, None)
    return mean_poisson_deviance(y_true, y_pred)

scoring = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(rmse, greater_is_better=False),
    "r2": "r2",
    "poisson_dev": make_scorer(poisson_dev, greater_is_better=False),
}

cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

results = []
for name, pipe in sorted(pipelines.items()):
    cv_res = cross_validate(
        pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    res = {
        "method": name,
        "MAE_mean":  -np.mean(cv_res["test_mae"]),
        "MAE_std":    np.std(-cv_res["test_mae"]),
        "RMSE_mean": -np.mean(cv_res["test_rmse"]),
        "RMSE_std":   np.std(-cv_res["test_rmse"]),
        "R2_mean":    np.mean(cv_res["test_r2"]),
        "R2_std":     np.std(cv_res["test_r2"]),
        "PoissonDev_mean": -np.mean(cv_res["test_poisson_dev"]),
        "PoissonDev_std":   np.std(-cv_res["test_poisson_dev"]),
        "n_folds":    len(cv_res["test_mae"]),
    }
    results.append(res)
    print(f"[CV] {name}: "
          f"MAE {res['MAE_mean']:.3f}±{res['MAE_std']:.3f} | "
          f"RMSE {res['RMSE_mean']:.3f}±{res['RMSE_std']:.3f} | "
          f"R² {res['R2_mean']:.3f}±{res['R2_std']:.3f} | "
          f"PoissonDev {res['PoissonDev_mean']:.3f}±{res['PoissonDev_std']:.3f}")

res_df = pd.DataFrame(results).sort_values("MAE_mean").reset_index(drop=True)
res_path = os.path.join(OUTPUT_DIR, "ridge_feature_selection_cv_results.csv")
res_df.to_csv(res_path, index=False)

best_method = res_df.iloc[0]["method"]
print("\n=== Best method by MAE (lower is better):", best_method, "===")

# ------------------- Fit best pipeline & OOF predictions -------------------
best_pipe = pipelines[best_method]
kf_pred = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_pred = cross_val_predict(best_pipe, X, y, cv=kf_pred, n_jobs=-1)

oof_mae  = mean_absolute_error(y, oof_pred)
oof_rmse = np.sqrt(mean_squared_error(y, oof_pred))
oof_r2   = r2_score(y, oof_pred)
oof_pdev = poisson_dev(y, oof_pred)

print(f"[OOF {best_method}] MAE={oof_mae:.3f} | RMSE={oof_rmse:.3f} | R²={oof_r2:.3f} | PoissonDev={oof_pdev:.3f}")

# ------------------- Plots -------------------
# Bar chart of MAE/RMSE/R² (mean ± std) by method
plt.figure(figsize=(9, 5))
x = np.arange(len(res_df))
w = 0.25
plt.bar(x - w, res_df["MAE_mean"],  width=w, label="MAE (↓)")
plt.bar(x,     res_df["RMSE_mean"], width=w, label="RMSE (↓)")
plt.bar(x + w, res_df["R2_mean"],   width=w, label="R² (↑)")
plt.xticks(x, res_df["method"], rotation=20, ha="right")
plt.ylabel("Score")
plt.title("Feature Selection + Ridge — Cross-Validated Performance")
plt.legend()
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "ridge_fs_metrics_bar.png")
plt.savefig(bar_path, dpi=150)
plt.show()

# Predicted vs Actual scatter for best method (OOF)
plt.figure(figsize=(5.5,5.5))
plt.scatter(y, oof_pred, alpha=0.85)
mn = min(np.min(y), np.min(oof_pred))
mx = max(np.max(y), np.max(oof_pred))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual pm_tot")
plt.ylabel(f"Predicted pm_tot (OOF, {best_method})")
plt.title(f"Predicted vs Actual — {best_method}\nMAE={oof_mae:.2f}, RMSE={oof_rmse:.2f}, R²={oof_r2:.2f}")
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "ridge_fs_best_pred_vs_actual.png")
plt.savefig(scatter_path, dpi=150)
plt.show()

# ------------------- Save summary -------------------
summary = {
    "best_method": str(best_method),
    "oof_metrics": {
        "MAE": float(oof_mae),
        "RMSE": float(oof_rmse),
        "R2": float(oof_r2),
        "PoissonDeviance": float(oof_pdev),
    },
    "cv_table_path": res_path,
    "plots": {
        "bar": bar_path,
        "pred_vs_actual": scatter_path,
    },
    "config": {
        "processed_dir": PROCESSED_DIR,
        "target_col": TARGET_COL,
        "n_samples": int(n_samples),
        "n_features_before": int(n_features),
        "k_best": int(K_BEST),
        "cv_splits": N_SPLITS,
        "cv_repeats": N_REPEATS,
        "ridge_alphas": list(map(float, RIDGE_ALPHAS)),
    }
}
with open(os.path.join(OUTPUT_DIR, "ridge_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n[Saved] Results → {res_path}")
print(f"[Saved] Plots → {bar_path}, {scatter_path}")
print(f"[Done] Best method: {best_method}")