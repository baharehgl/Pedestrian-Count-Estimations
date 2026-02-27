"""
Pedestrian Count Estimation - Final Best Model Selection
==========================================================
This script takes the results from all experiments and picks ONE best model
using a combined ranking across RMSE, MAPE, and SMAPE.

Then it trains that model on the full dataset, reports:
  - All metrics
  - Selected features ranked by importance
  - Best hyperparameters

"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import PoissonRegressor, LassoCV, RidgeCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "df1_v1a_out.csv"
TARGET_COL = "pm_tot"
SPLIT_COL = "holdout"
ID_COL = "site_id"
DROP_COLS = [TARGET_COL, SPLIT_COL, ID_COL]
CATEGORICAL_COLS = [
    "Street Nam", "_Date", "season", "geometry",
    "class_type", "speed_type", "crossing_class"
]
RANDOM_STATE = 42
OUTPUT_DIR = "final_best_model"


# ============================================================================
# METRICS (corrected to match R code)
# ============================================================================

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def standard_mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

def smape_score(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100

def mcfadden_r2_deviance(y_true, y_pred):
    y_pred_safe = np.maximum(y_pred, 1e-10)
    y_bar = max(np.mean(y_true), 1e-10)
    dev_model = 2.0 * np.sum(
        np.where(y_true > 0,
                 y_true * np.log(y_true / y_pred_safe) - (y_true - y_pred_safe),
                 y_pred_safe)
    )
    dev_null = 2.0 * np.sum(
        np.where(y_true > 0,
                 y_true * np.log(y_true / y_bar) - (y_true - y_bar),
                 y_bar)
    )
    if dev_null == 0:
        return 0.0
    return 1.0 - (dev_model / dev_null)

def correlation_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    y = df[TARGET_COL].values
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return X, y, preprocessor, num_cols, cat_cols


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_l1(X_transformed, y, feature_names, n_features=20):
    y_log = np.log1p(y)
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_transformed, y_log)
    importance = np.abs(lasso.coef_)
    top_idx = np.argsort(importance)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names

def select_features_mi(X_transformed, y, feature_names, n_features=20):
    mi_scores = mutual_info_regression(
        X_transformed, y, random_state=RANDOM_STATE, n_neighbors=5
    )
    top_idx = np.argsort(mi_scores)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names

def select_features_rf(X_transformed, y, feature_names, n_features=20):
    rf = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_transformed, y)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


# ============================================================================
# ALL CANDIDATE MODELS
# ============================================================================

def get_all_candidates():
    """
    Every model + feature selection + n_features combo we want to compare.
    Each is a complete configuration that can be evaluated independently.
    """
    candidates = []

    # Feature selection options
    fs_options = [
        ("L1_Lasso", select_features_l1, 15),
        ("L1_Lasso", select_features_l1, 20),
        ("L1_Lasso", select_features_l1, 25),
        ("MI", select_features_mi, 20),
        ("RF", select_features_rf, 20),
    ]

    # Model options (including tuned variants)
    model_options = [
        ("RandomForest_default", RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("RandomForest_tuned", RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("HistGB_Poisson_default", HistGradientBoostingRegressor(
            loss="poisson", max_iter=500, random_state=RANDOM_STATE
        )),
        ("HistGB_Poisson_tuned_v1", HistGradientBoostingRegressor(
            loss="poisson", max_iter=800, learning_rate=0.05,
            max_depth=7, min_samples_leaf=10, l2_regularization=1.0,
            random_state=RANDOM_STATE
        )),
        ("HistGB_Poisson_tuned_v2", HistGradientBoostingRegressor(
            loss="poisson", max_iter=500, learning_rate=0.1,
            max_depth=5, min_samples_leaf=20, l2_regularization=0.0,
            random_state=RANDOM_STATE
        )),
        ("HistGB_SquaredError", HistGradientBoostingRegressor(
            loss="squared_error", max_iter=500, learning_rate=0.05,
            max_depth=7, min_samples_leaf=10,
            random_state=RANDOM_STATE
        )),
        ("ExtraTrees", ExtraTreesRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("GradientBoosting_Huber", GradientBoostingRegressor(
            loss="huber", n_estimators=500, learning_rate=0.05,
            max_depth=5, min_samples_leaf=5, subsample=0.8,
            random_state=RANDOM_STATE
        )),
        ("PoissonRegressor", PoissonRegressor(
            alpha=0.01, max_iter=5000
        )),
        ("Bagging_HistGB", BaggingRegressor(
            estimator=HistGradientBoostingRegressor(
                loss="poisson", random_state=RANDOM_STATE,
                max_iter=500, learning_rate=0.05
            ),
            n_estimators=10, max_samples=0.8,
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("Stack_RF+HistGB+ET", StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(
                    n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
                )),
                ("hgb", HistGradientBoostingRegressor(
                    loss="poisson", max_iter=500, learning_rate=0.05,
                    random_state=RANDOM_STATE
                )),
                ("et", ExtraTreesRegressor(
                    n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
                )),
            ],
            final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
            cv=5, n_jobs=-1
        )),
    ]

    for fs_name, fs_func, n_feat in fs_options:
        for model_name, model in model_options:
            candidates.append({
                "id": f"{fs_name}_{n_feat}_{model_name}",
                "fs_name": fs_name,
                "fs_func": fs_func,
                "n_features": n_feat,
                "model_name": model_name,
                "model": model,
            })

    return candidates


# ============================================================================
# EVALUATE ONE CANDIDATE WITH FULL CV
# ============================================================================

def evaluate_candidate(X_transformed, y, feature_names, candidate, cv_strategy):
    """Evaluate a single candidate configuration with cross-validation."""
    # Select features
    feat_idx, feat_names_sel = candidate["fs_func"](
        X_transformed, y, feature_names, n_features=candidate["n_features"]
    )
    X_selected = X_transformed[:, feat_idx]

    rmse_list, mape_list, smape_list = [], [], []
    r2_train_list, corr_r2_list = [], []

    for train_idx, val_idx in cv_strategy.split(X_selected):
        X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = clone(candidate["model"])
        m.fit(X_tr, y_tr)

        y_pred_tr = np.maximum(m.predict(X_tr), 0)
        y_pred_val = np.maximum(m.predict(X_val), 0)

        rmse_list.append(rmse_score(y_val, y_pred_val))
        mape_list.append(standard_mape(y_val, y_pred_val))
        smape_list.append(smape_score(y_val, y_pred_val))
        r2_train_list.append(mcfadden_r2_deviance(y_tr, y_pred_tr))
        corr_r2_list.append(correlation_r2(y_val, y_pred_val))

    return {
        "id": candidate["id"],
        "feature_selection": candidate["fs_name"],
        "n_features": candidate["n_features"],
        "model": candidate["model_name"],
        "RMSE_mean": np.mean(rmse_list),
        "RMSE_std": np.std(rmse_list),
        "MAPE_mean": np.mean(mape_list),
        "MAPE_std": np.std(mape_list),
        "SMAPE_mean": np.mean(smape_list),
        "SMAPE_std": np.std(smape_list),
        "R2_train_mean": np.mean(r2_train_list),
        "Corr_R2_val_mean": np.mean(corr_r2_list),
    }


# ============================================================================
# COMBINED RANKING: pick the single best
# ============================================================================

def rank_and_select_best(df_results):
    """
    Rank all candidates by RMSE, MAPE, and SMAPE.
    The candidate with the best average rank wins.
    """
    df = df_results.copy()

    # Rank each metric (lower is better, so rank 1 = lowest value)
    df["rank_RMSE"] = df["RMSE_mean"].rank(method="min")
    df["rank_MAPE"] = df["MAPE_mean"].rank(method="min")
    df["rank_SMAPE"] = df["SMAPE_mean"].rank(method="min")

    # Average rank (equal weight to all three metrics)
    df["avg_rank"] = (df["rank_RMSE"] + df["rank_MAPE"] + df["rank_SMAPE"]) / 3.0

    # Sort by average rank
    df = df.sort_values("avg_rank").reset_index(drop=True)

    return df


# ============================================================================
# FEATURE IMPORTANCE FOR FINAL MODEL
# ============================================================================

def get_feature_importance(model, X_data, y_data, feat_names_list):
    """Train on full data and extract feature importance, ranked high to low."""
    m = clone(model)
    m.fit(X_data, y_data)

    if hasattr(m, "feature_importances_"):
        importance = m.feature_importances_
        method = "built-in (split importance)"
    elif hasattr(m, "coef_"):
        importance = np.abs(m.coef_)
        method = "absolute coefficients"
    else:
        perm = permutation_importance(
            m, X_data, y_data, n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error"
        )
        importance = perm.importances_mean
        method = "permutation importance"

    sorted_idx = np.argsort(importance)[::-1]
    df_imp = pd.DataFrame({
        "rank": range(1, len(feat_names_list) + 1),
        "feature": [feat_names_list[i] for i in sorted_idx],
        "importance": importance[sorted_idx],
    })
    return df_imp, method


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FINDING THE SINGLE BEST MODEL")
    print("Evaluating all candidates with Repeated 3x10 CV")
    print("Ranking by combined RMSE + MAPE + SMAPE")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: '{DATA_PATH}' not found.")
        return

    # Load data
    X, y, preprocessor, num_cols, cat_cols = load_and_preprocess(DATA_PATH)
    X_transformed = preprocessor.fit_transform(X)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = list(range(X_transformed.shape[1]))

    print(f"Dataset: {X_transformed.shape[0]} samples, "
          f"{len(feature_names)} features after preprocessing")

    # CV strategy: same for ALL candidates (fair comparison)
    cv_strategy = RepeatedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    # Get all candidates
    candidates = get_all_candidates()
    print(f"Total candidates to evaluate: {len(candidates)}")

    # ========================================================================
    # STEP 1: Evaluate every candidate
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Evaluating all candidates (this may take a while)...")
    print("=" * 70)

    all_results = []
    for i, cand in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] {cand['id']}...", end=" ", flush=True)
        try:
            res = evaluate_candidate(
                X_transformed, y, feature_names, cand, cv_strategy
            )
            all_results.append(res)
            print(f"RMSE={res['RMSE_mean']:.2f}  "
                  f"MAPE={res['MAPE_mean']:.2f}  "
                  f"SMAPE={res['SMAPE_mean']:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    df_results = pd.DataFrame(all_results)

    # ========================================================================
    # STEP 2: Rank and pick the winner
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Combined Ranking (RMSE + MAPE + SMAPE)")
    print("=" * 70)

    df_ranked = rank_and_select_best(df_results)

    # Save full ranking
    ranking_path = os.path.join(OUTPUT_DIR, "all_candidates_ranked.csv")
    df_ranked.to_csv(ranking_path, index=False)

    # Show top 15
    display_cols = [
        "id", "model", "feature_selection", "n_features",
        "RMSE_mean", "MAPE_mean", "SMAPE_mean",
        "rank_RMSE", "rank_MAPE", "rank_SMAPE", "avg_rank"
    ]
    print("\nTop 15 candidates by combined rank:")
    print(df_ranked.head(15)[display_cols].to_string(index=False))

    # ========================================================================
    # STEP 3: The winner
    # ========================================================================
    winner = df_ranked.iloc[0]
    winner_id = winner["id"]

    print("\n" + "=" * 70)
    print("THE SINGLE BEST MODEL")
    print("=" * 70)
    print(f"\n  Winner: {winner_id}")
    print(f"  Model:             {winner['model']}")
    print(f"  Feature Selection: {winner['feature_selection']}")
    print(f"  N Features:        {int(winner['n_features'])}")
    print(f"\n  RMSE:   {winner['RMSE_mean']:.2f} ± {winner['RMSE_std']:.2f}  "
          f"(rank {int(winner['rank_RMSE'])})")
    print(f"  MAPE:   {winner['MAPE_mean']:.2f} ± {winner['MAPE_std']:.2f}  "
          f"(rank {int(winner['rank_MAPE'])})")
    print(f"  SMAPE:  {winner['SMAPE_mean']:.2f} ± {winner['SMAPE_std']:.2f}  "
          f"(rank {int(winner['rank_SMAPE'])})")
    print(f"  R2:     {winner['R2_train_mean']:.4f}")
    print(f"  Corr R2:{winner['Corr_R2_val_mean']:.4f}")
    print(f"\n  Average Rank: {winner['avg_rank']:.2f} out of {len(df_ranked)}")

    # ========================================================================
    # STEP 4: Feature importance for the winner
    # ========================================================================
    print("\n" + "=" * 70)
    print("SELECTED FEATURES (ranked by importance, high to low)")
    print("=" * 70)

    # Find the winning candidate object
    winner_cand = None
    for cand in candidates:
        if cand["id"] == winner_id:
            winner_cand = cand
            break

    if winner_cand:
        feat_idx, feat_names_sel = winner_cand["fs_func"](
            X_transformed, y, feature_names,
            n_features=winner_cand["n_features"]
        )
        X_selected = X_transformed[:, feat_idx]

        df_imp, imp_method = get_feature_importance(
            winner_cand["model"], X_selected, y, feat_names_sel
        )

        print(f"  Importance method: {imp_method}\n")
        print(f"  {'Rank':<6}{'Feature':<45}{'Importance':<12}")
        print(f"  {'-'*63}")
        for _, row in df_imp.iterrows():
            bar = "█" * int(row["importance"] / df_imp["importance"].max() * 30)
            print(f"  {int(row['rank']):<6}{row['feature']:<45}"
                  f"{row['importance']:<12.4f} {bar}")

        # Save
        imp_path = os.path.join(OUTPUT_DIR, "best_model_feature_importance.csv")
        df_imp.to_csv(imp_path, index=False)
        print(f"\n  Saved: {imp_path}")

    # ========================================================================
    # STEP 5: Comparison with baselines
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)

    baselines = [
        ("R-code GLM_NB (10 feat, 5-fold CV)", 93.56, 49.75, None),
        ("L1+RF (20 feat, 5-fold CV)", 79.84, 50.48, None),
        ("L1+HistGB (18 feat, holdout)", 133.60, 45.11, None),
    ]

    print(f"\n  {'Model':<45}{'RMSE':<12}{'MAPE':<12}{'SMAPE':<12}")
    print(f"  {'-'*81}")
    for name, rmse_val, mape_val, smape_val in baselines:
        smape_str = f"{smape_val:.2f}" if smape_val else "N/A"
        print(f"  {name:<45}{rmse_val:<12.2f}{mape_val:<12.2f}{smape_str:<12}")

    print(f"  {'-'*81}")
    print(f"  {'>>> BEST MODEL: ' + winner['model']:<45}"
          f"{winner['RMSE_mean']:<12.2f}"
          f"{winner['MAPE_mean']:<12.2f}"
          f"{winner['SMAPE_mean']:<12.2f}")

    rmse_improvement = ((79.84 - winner['RMSE_mean']) / 79.84) * 100
    print(f"\n  RMSE improvement over previous best: "
          f"{rmse_improvement:.1f}%")

    # ========================================================================
    # STEP 6: Save final model summary
    # ========================================================================
    summary = {
        "model_name": winner["model"],
        "feature_selection": winner["feature_selection"],
        "n_features": int(winner["n_features"]),
        "cv_strategy": "RepeatedKFold 3x10",
        "RMSE_mean": winner["RMSE_mean"],
        "RMSE_std": winner["RMSE_std"],
        "MAPE_mean": winner["MAPE_mean"],
        "MAPE_std": winner["MAPE_std"],
        "SMAPE_mean": winner["SMAPE_mean"],
        "SMAPE_std": winner["SMAPE_std"],
        "R2_train": winner["R2_train_mean"],
        "Corr_R2_val": winner["Corr_R2_val_mean"],
    }

    summary_path = os.path.join(OUTPUT_DIR, "best_model_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"\n  All results saved to: {OUTPUT_DIR}/")
    print(f"    - all_candidates_ranked.csv      (full ranking)")
    print(f"    - best_model_feature_importance.csv (features)")
    print(f"    - best_model_summary.csv          (final answer)")
    print("\nDone!")


if __name__ == "__main__":
    main()