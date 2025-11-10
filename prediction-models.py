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
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance, make_scorer

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------- Config ----------------
PROCESSED_DIR = "./processed"
FS_DIR        = "./fs_outputs"
RAW_CSV       = "./df1_v1a_out.csv"   # change if your raw file lives elsewhere
TARGET_COL    = "pm_tot"
OUTPUT_DIR    = "./model_outputs"

RANDOM_STATE  = 42
N_SPLITS      = 5
N_REPEATS     = 5
K_BEST        = None   # if None -> min(15, max(5, n_features//4))
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load data ----------
X = pd.read_csv(os.path.join(PROCESSED_DIR, "X.csv"))
y_df = pd.read_csv(os.path.join(PROCESSED_DIR, "y.csv"))
if TARGET_COL not in y_df.columns:
    TARGET_COL = y_df.columns[0]
y = y_df[TARGET_COL].astype(float).values

n_samples, n_features = X.shape
if K_BEST is None:
    K_BEST = max(5, min(15, n_features // 4))

# Try to load site_id from raw for clustering (won't be used as a feature)
site_id = None
if os.path.exists(RAW_CSV):
    try:
        raw = pd.read_csv(RAW_CSV)
        raw.columns = [str(c).strip() for c in raw.columns]
        cand = [c for c in raw.columns if c.lower() in {"site_id", "siteid", "site"}]
        if cand:
            site_col = cand[0]
            site_id = raw[site_col].astype(str).values
    except Exception as e:
        warnings.warn(f"Could not read RAW_CSV for site_id: {e}")

# ---------- Load best selector from fs_outputs ----------
with open(os.path.join(FS_DIR, "summary.json"), "r", encoding="utf-8") as f:
    fs_summary = json.load(f)
best_selector_name = fs_summary["best_method"]
print(f"[INFO] Using selector from fs_outputs: {best_selector_name}")

# ---------- Rebuild selector to get selected feature names ----------
mi_fn = partial(mutual_info_regression, random_state=RANDOM_STATE)

def build_selector(name: str):
    if name == "Univariate_MI+RF":
        return Pipeline([
            ("select", SelectKBest(score_func=mi_fn, k=K_BEST))
        ])
    elif name == "Lasso_SelectFromModel+RF":
        return Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("select", SelectFromModel(
                estimator=LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
                threshold="median", prefit=False
            ))
        ])
    elif name == "RF_Importance_Select+RF":
        return Pipeline([
            ("select", SelectFromModel(
                estimator=RandomForestRegressor(n_estimators=500, max_depth=6,
                                                random_state=RANDOM_STATE, n_jobs=-1),
                threshold="median", prefit=False
            ))
        ])
    else:
        raise ValueError(f"Unknown selector name: {name}")

sel_pipe = build_selector(best_selector_name)
sel_pipe.fit(X, y)

# Extract mask regardless of pipeline shape
if best_selector_name == "Univariate_MI+RF":
    mask = sel_pipe.named_steps["select"].get_support()
elif best_selector_name == "Lasso_SelectFromModel+RF":
    mask = sel_pipe.named_steps["select"].get_support()
else:  # RF_Importance_Select+RF
    mask = sel_pipe.named_steps["select"].get_support()

selected_features = list(X.columns[mask])
if len(selected_features) == 0:
    # Safety fallback: pick top-K variance features
    var_rank = X.var().sort_values(ascending=False)
    selected_features = list(var_rank.index[:max(5, min(10, X.shape[1]))])
    print("[WARN] Selector picked 0 features. Falling back to high-variance subset:", selected_features)

X_sel = X[selected_features].copy()
print(f"[INFO] Selected {len(selected_features)} features:", selected_features)

# ---------- Helpers ----------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def poisson_dev(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, None)
    return mean_poisson_deviance(y_true, y_pred)

scoring = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(lambda yt, yp: -rmse(yt, yp)),
    "r2":  "r2",
    "poisson_dev": make_scorer(lambda yt, yp: -poisson_dev(yt, yp)),
}

cv_rep = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
kf     = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ---------- 1) Random Forest (robust small-N) ----------
rf = RandomForestRegressor(n_estimators=600, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
rf_cv = cross_validate(rf, X_sel, y, cv=cv_rep, scoring=scoring, n_jobs=-1)
rf_oof = cross_val_predict(rf, X_sel, y, cv=kf, n_jobs=-1)

# ---------- 2) HistGradientBoosting (Poisson loss for counts) ----------
hgb = HistGradientBoostingRegressor(loss="poisson", max_depth=6, learning_rate=0.08,
                                    max_iter=300, random_state=RANDOM_STATE)
# Ensure y is non-negative for Poisson
if np.any(y < 0):
    raise ValueError("Poisson loss needs non-negative target.")
hgb_cv = cross_validate(hgb, X_sel, y, cv=cv_rep, scoring=scoring, n_jobs=-1)
hgb_oof = cross_val_predict(hgb, X_sel, y, cv=kf, n_jobs=-1)

# ---------- 3) GLM Negative Binomial (statsmodels), OOF loop ----------
def glm_nb_oof(X_df, y_vec, cv):
    preds = np.zeros_like(y_vec, dtype=float)
    for tr, te in cv.split(X_df):
        X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
        y_tr = y_vec[tr]
        # Add intercept
        X_tr_sm = sm.add_constant(X_tr, has_constant="add")
        X_te_sm = sm.add_constant(X_te, has_constant="add")
        # Fit NB GLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y_tr, X_tr_sm, family=sm.families.NegativeBinomial())
            res = model.fit(maxiter=200, disp=0)
        preds[te] = np.clip(res.predict(X_te_sm), 1e-9, None)
    return preds

nb_oof = glm_nb_oof(X_sel, y, kf)

# ---------- 4) GEE Negative Binomial with site_id clusters (if possible) ----------
def gee_nb_oof(X_df, y_vec, groups, cv):
    preds = np.zeros_like(y_vec, dtype=float)
    if groups is None:
        raise ValueError("No site_id available.")
    groups = pd.Series(groups, index=X_df.index)
    # Check repeated groups
    counts = groups.value_counts()
    if (counts <= 1).all():
        raise ValueError("All groups are singletons; GEE/GLMM not identifiable.")
    for tr, te in cv.split(X_df):
        X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
        y_tr = y_vec[tr]
        g_tr = groups.iloc[tr]
        X_tr_sm = sm.add_constant(X_tr, has_constant="add")
        X_te_sm = sm.add_constant(X_te, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Working independence; NB family; cluster by site
            model = sm.GEE(endog=y_tr, exog=X_tr_sm, groups=g_tr,
                           family=sm.families.NegativeBinomial())
            res = model.fit()
        preds[te] = np.clip(res.predict(X_te_sm), 1e-9, None)
    return preds

gee_supported = True
gee_oof = None
try:
    gee_oof = gee_nb_oof(X_sel, y, site_id, kf)
except Exception as e:
    gee_supported = False
    print(f"[INFO] Skipping GEE-NB (≈GLMM) due to: {e}")

# ---------- Collect metrics ----------
def summarize(name, oof_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y, oof_pred),
        "RMSE": rmse(y, oof_pred),
        "R2": r2_score(y, oof_pred),
        "PoissonDev": poisson_dev(y, oof_pred),
    }

rf_metrics  = summarize("RandomForest", rf_oof)
hgb_metrics = summarize("HistGB_Poisson", hgb_oof)
nb_metrics  = summarize("GLM_NegativeBinomial", nb_oof)
rows = [rf_metrics, hgb_metrics, nb_metrics]

# Add mean±std from CV tables for RF/HGB (optional columns)
def cv_stats(cv_res, prefix):
    return {
        f"{prefix}_MAE_mean":  -np.mean(cv_res["test_mae"]),
        f"{prefix}_MAE_std":    np.std(-cv_res["test_mae"]),
        f"{prefix}_RMSE_mean": -np.mean(cv_res["test_rmse"]),
        f"{prefix}_RMSE_std":   np.std(-cv_res["test_rmse"]),
        f"{prefix}_R2_mean":    np.mean(cv_res["test_r2"]),
        f"{prefix}_R2_std":     np.std(cv_res["test_r2"]),
        f"{prefix}_PDev_mean": -np.mean(cv_res["test_poisson_dev"]),
        f"{prefix}_PDev_std":   np.std(-cv_res["test_poisson_dev"]),
    }

rows[0].update(cv_stats(rf_cv, "cv"))
rows[1].update(cv_stats(hgb_cv, "cv"))

if gee_supported and gee_oof is not None:
    gee_metrics = summarize("GEE_NegativeBinomial(site_id)", gee_oof)
    rows.append(gee_metrics)

res_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
res_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
res_df.to_csv(res_path, index=False)
print("\n[RESULTS]\n", res_df)

# ---------- Plot comparison ----------
plt.figure(figsize=(9,5))
x = np.arange(len(res_df))
plt.bar(x - 0.3, res_df["MAE"],  width=0.28, label="MAE (↓)")
plt.bar(x - 0.0, res_df["RMSE"], width=0.28, label="RMSE (↓)")
plt.bar(x + 0.3, res_df["R2"],   width=0.28, label="R² (↑)")
plt.xticks(x, res_df["model"], rotation=15, ha="right")
plt.ylabel("Score")
plt.title("Model Comparison on Selected Features (fs_outputs)")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "model_comparison_bar.png")
plt.savefig(plot_path, dpi=150)
plt.show()

# ---------- Save summary JSON ----------
summary = {
    "selected_features": selected_features,
    "best_selector_from_fs_outputs": best_selector_name,
    "results_csv": res_path,
    "bar_plot": plot_path,
    "config": {
        "processed_dir": PROCESSED_DIR,
        "fs_dir": FS_DIR,
        "raw_csv": RAW_CSV,
        "target_col": TARGET_COL,
        "cv_splits": N_SPLITS,
        "cv_repeats": N_REPEATS
    }
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n[Saved] Results → {res_path}")
print(f"[Saved] Plot   → {plot_path}")
print("[Done]")
