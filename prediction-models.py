# === Statistically Sound Models on fs_outputs-selected features ===
# Uses features chosen in fs_outputs/summary.json
# Models: RandomForest, HistGB (Poisson), GLM Negative Binomial, GEE Negative Binomial (if site_id groups)
# CV: 5x5 RepeatedKFold; Metrics: MAE (↓), RMSE (↓), R² (↑), Poisson deviance (↓)
# ------------------------------------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, mutual_info_regression
from sklearn.linear_model import LassoCV, PoissonRegressor, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance, make_scorer

# ---------------- Config ----------------
PROCESSED_DIR = "./processed"       # expects X.csv, y.csv from your preprocessing step
FS_DIR        = "./fs_outputs"      # expects summary.json from the feature-selection step
TARGET_COL    = "pm_tot"
OUTPUT_DIR    = "./model_outputs"

RANDOM_STATE  = 42
N_SPLITS      = 5
N_REPEATS     = 5
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load preprocessed numeric-only data ----------
X = pd.read_csv(os.path.join(PROCESSED_DIR, "X.csv"))
y_df = pd.read_csv(os.path.join(PROCESSED_DIR, "y.csv"))
if TARGET_COL not in y_df.columns:
    TARGET_COL = y_df.columns[0]
y = y_df[TARGET_COL].astype(float).values
if np.any(y < 0):
    raise ValueError("Targets must be non-negative for Poisson/Tweedie models.")

n_samples, n_features = X.shape

# ---------- Read best selector from fs_outputs ----------
with open(os.path.join(FS_DIR, "summary.json"), "r", encoding="utf-8") as f:
    fs_summary = json.load(f)
best_selector_name = fs_summary["best_method"]
K_BEST = fs_summary.get("config", {}).get("k_best", max(5, min(15, n_features // 4)))
print(f"[INFO] Using selector from fs_outputs: {best_selector_name}, K_BEST={K_BEST}")

mi_fn = partial(mutual_info_regression, random_state=RANDOM_STATE)

def build_selector_pipeline(name: str) -> Pipeline:
    """
    Returns a *selector-only* pipeline that will run inside each fold (no leakage).
    """
    if name == "Univariate_MI+RF":
        return Pipeline([
            ("select", SelectKBest(score_func=mi_fn, k=int(K_BEST))),
        ])

    elif name == "Lasso_SelectFromModel+RF":
        return Pipeline([
            ("scale_sel", StandardScaler(with_mean=True, with_std=True)),
            ("select", SelectFromModel(
                estimator=LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
                threshold="median",
                prefit=False
            )),
        ])

    elif name == "RF_Importance_Select+RF":
        return Pipeline([
            ("select", SelectFromModel(
                estimator=RandomForestRegressor(
                    n_estimators=500, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
                ),
                threshold="median",
                prefit=False
            )),
        ])
    else:
        raise ValueError(f"Unknown selector name in fs_outputs: {name}")

selector_only = build_selector_pipeline(best_selector_name)

# ---------- Define models (final estimators) ----------
rf = RandomForestRegressor(n_estimators=600, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)

hgb = HistGradientBoostingRegressor(
    loss="poisson",          # count-aware loss
    max_depth=6,
    learning_rate=0.08,
    max_iter=300,
    random_state=RANDOM_STATE
)

# Linear count GLMs benefit from scaling of inputs
poisson_glm = Pipeline([
    ("scale_model", StandardScaler(with_mean=True, with_std=True)),
    ("model", PoissonRegressor(alpha=1.0, fit_intercept=True, max_iter=1000))
])

# Tweedie with 1<p<2 is compound Poisson (allows over-dispersion)
tweedie_glm = Pipeline([
    ("scale_model", StandardScaler(with_mean=True, with_std=True)),
    ("model", TweedieRegressor(power=1.5, alpha=1.0, link="log", max_iter=1000))
])

# ---------- Wrap selector + model into full pipelines ----------
pipelines = {
    "RandomForest": Pipeline([*selector_only.steps, ("model", rf)]),
    "HistGB_Poisson": Pipeline([*selector_only.steps, ("model", hgb)]),
    "Poisson_GLM": Pipeline([*selector_only.steps, *poisson_glm.steps]),
    "Tweedie_GLM(p=1.5)": Pipeline([*selector_only.steps, *tweedie_glm.steps]),
}

# ---------- Scoring & CV ----------
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def poisson_dev(y_true, y_pred):
    return mean_poisson_deviance(y_true, np.clip(y_pred, 1e-9, None))

scoring = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(lambda yt, yp: -rmse(yt, yp)),
    "r2":  "r2",
    "poisson_dev": make_scorer(lambda yt, yp: -poisson_dev(yt, yp)),
}

cv_rep = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
kf     = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rows = []
oof_preds = {}  # for plotting best model scatter

for name, pipe in pipelines.items():
    # Repeated CV summary
    cv_res = cross_validate(pipe, X, y, cv=cv_rep, scoring=scoring, n_jobs=-1, return_train_score=False)
    row = {
        "model": name,
        "MAE_mean":         -np.mean(cv_res["test_mae"]),
        "MAE_std":           np.std(-cv_res["test_mae"]),
        "RMSE_mean":        -np.mean(cv_res["test_rmse"]),
        "RMSE_std":          np.std(-cv_res["test_rmse"]),
        "R2_mean":           np.mean(cv_res["test_r2"]),
        "R2_std":            np.std(cv_res["test_r2"]),
        "PoissonDev_mean":  -np.mean(cv_res["test_poisson_dev"]),
        "PoissonDev_std":    np.std(-cv_res["test_poisson_dev"]),
        "n_folds":           len(cv_res["test_mae"]),
    }
    rows.append(row)
    print(f"[CV] {name}: MAE {row['MAE_mean']:.3f}±{row['MAE_std']:.3f} | "
          f"RMSE {row['RMSE_mean']:.3f}±{row['RMSE_std']:.3f} | "
          f"R² {row['R2_mean']:.3f}±{row['R2_std']:.3f} | "
          f"PDev {row['PoissonDev_mean']:.3f}±{row['PoissonDev_std']:.3f}")

    # Out-of-fold predictions for scatter (single KFold)
    oof = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)
    oof_preds[name] = oof

res_df = pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)
res_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
res_df.to_csv(res_path, index=False)

best_model_name = res_df.iloc[0]["model"]
print("\n=== Best model by MAE (lower is better):", best_model_name, "===")

# ---------- Plots ----------
# Bar chart of MAE/RMSE/R²
plt.figure(figsize=(9,5))
x = np.arange(len(res_df))
w = 0.25
plt.bar(x - w, res_df["MAE_mean"],  width=w, label="MAE (↓)")
plt.bar(x,     res_df["RMSE_mean"], width=w, label="RMSE (↓)")
plt.bar(x + w, res_df["R2_mean"],   width=w, label="R² (↑)")
plt.xticks(x, res_df["model"], rotation=15, ha="right")
plt.ylabel("Score")
plt.title("Model Comparison on Selected Features (fs_outputs)")
plt.legend()
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "model_comparison_bar.png")
plt.savefig(bar_path, dpi=150)
plt.show()

# Predicted vs Actual for best model
best_oof = oof_preds[best_model_name]
plt.figure(figsize=(5.5,5.5))
plt.scatter(y, best_oof, alpha=0.85)
mn = min(np.min(y), np.min(best_oof))
mx = max(np.max(y), np.max(best_oof))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual pm_tot")
plt.ylabel(f"Predicted pm_tot (OOF, {best_model_name})")
plt.title(f"Predicted vs Actual — {best_model_name}")
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "best_model_pred_vs_actual.png")
plt.savefig(scatter_path, dpi=150)
plt.show()

# ---------- Save summary ----------
summary = {
    "best_selector_from_fs_outputs": best_selector_name,
    "best_model_by_MAE": best_model_name,
    "results_csv": res_path,
    "plots": {
        "bar": bar_path,
        "pred_vs_actual": scatter_path,
    },
    "config": {
        "processed_dir": PROCESSED_DIR,
        "fs_dir": FS_DIR,
        "target_col": TARGET_COL,
        "cv_splits": N_SPLITS,
        "cv_repeats": N_REPEATS,
    }
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n[Saved] Results → {res_path}")
print(f"[Saved] Plots → {bar_path}, {scatter_path}")
print("[Done]")