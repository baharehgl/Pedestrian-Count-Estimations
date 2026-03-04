"""
Pedestrian Count Estimation - Final Best Model Selection (v2)
==============================================================
Picks ONE best model using combined ranking across RMSE, MAPE, and SMAPE.

v2 changes:
  - Expanded feature counts: 10, 15, 20, 25, 30 (was 15, 20, 25)
  - Evaluates ALL candidates under ALL 4 CV strategies
  - Cross-CV consistency analysis: shows whether the winner is the same
    regardless of CV strategy (strong robustness evidence)
  - Final winner selected from the CV strategy with best stability

Reports: metrics, selected features, importance, baseline comparison.

Usage:
  python find_best_model_v2.py

Make sure df1_v1a_out.csv is in the same directory or update DATA_PATH.
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

    # Feature selection options (expanded to match Exp 2: 10, 15, 20, 25, 30)
    fs_options = [
        ("L1_Lasso", select_features_l1, 10),
        ("L1_Lasso", select_features_l1, 15),
        ("L1_Lasso", select_features_l1, 20),
        ("L1_Lasso", select_features_l1, 25),
        ("L1_Lasso", select_features_l1, 30),
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
    print("FINDING THE SINGLE BEST MODEL (v2)")
    print("Evaluating all candidates under ALL 4 CV strategies")
    print("Ranking by combined RMSE + MAPE + SMAPE")
    print("Cross-CV consistency check for robustness")
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

    # ---- ALL 4 CV strategies (matching Exp 1 & Exp 3) ----
    cv_strategies = {
        "5-Fold": KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        "10-Fold": KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
        "Repeated_5x10": RepeatedKFold(
            n_splits=10, n_repeats=5, random_state=RANDOM_STATE
        ),
        "Repeated_3x10": RepeatedKFold(
            n_splits=10, n_repeats=3, random_state=RANDOM_STATE
        ),
    }

    # Get all candidates
    candidates = get_all_candidates()
    n_cands = len(candidates)
    n_cv = len(cv_strategies)
    total_evals = n_cands * n_cv
    print(f"Candidates: {n_cands}")
    print(f"CV strategies: {n_cv}")
    print(f"Total evaluations: {total_evals}")

    # ========================================================================
    # STEP 1: Evaluate every candidate under every CV strategy
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Evaluating all candidates × all CV strategies...")
    print("=" * 70)

    all_results = []  # list of dicts, each includes cv_strategy column
    eval_counter = 0

    for cv_name, cv_strat in cv_strategies.items():
        print(f"\n--- CV Strategy: {cv_name} ---")
        for i, cand in enumerate(candidates):
            eval_counter += 1
            print(f"  [{eval_counter}/{total_evals}] {cand['id']}...",
                  end=" ", flush=True)
            try:
                res = evaluate_candidate(
                    X_transformed, y, feature_names, cand, cv_strat
                )
                res["cv_strategy"] = cv_name
                all_results.append(res)
                print(f"RMSE={res['RMSE_mean']:.2f}  "
                      f"MAPE={res['MAPE_mean']:.2f}  "
                      f"SMAPE={res['SMAPE_mean']:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")

    df_all = pd.DataFrame(all_results)

    # Save all raw results
    all_path = os.path.join(OUTPUT_DIR, "all_candidates_all_cv.csv")
    df_all.to_csv(all_path, index=False)
    print(f"\nSaved all results: {all_path}")

    # ========================================================================
    # STEP 2: Rank within each CV strategy separately
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Combined Ranking within each CV strategy")
    print("=" * 70)

    per_cv_winners = {}  # cv_name -> winner row
    per_cv_rankings = {}  # cv_name -> full ranked DataFrame

    for cv_name in cv_strategies:
        df_cv = df_all[df_all["cv_strategy"] == cv_name].copy()
        df_ranked = rank_and_select_best(df_cv)
        per_cv_rankings[cv_name] = df_ranked

        winner = df_ranked.iloc[0]
        per_cv_winners[cv_name] = winner

        # Save per-CV ranking
        cv_rank_path = os.path.join(
            OUTPUT_DIR, f"ranking_{cv_name.replace(' ', '_')}.csv"
        )
        df_ranked.to_csv(cv_rank_path, index=False)

        print(f"\n  {cv_name} winner: {winner['id']}")
        print(f"    RMSE={winner['RMSE_mean']:.2f}  "
              f"MAPE={winner['MAPE_mean']:.2f}  "
              f"SMAPE={winner['SMAPE_mean']:.2f}  "
              f"avg_rank={winner['avg_rank']:.2f}")

    # ========================================================================
    # STEP 3: Cross-CV Consistency Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Cross-CV Consistency Analysis")
    print("=" * 70)

    # Show winners side by side
    print(f"\n  {'CV Strategy':<20}{'Winner':<45}{'RMSE':<10}"
          f"{'MAPE':<10}{'SMAPE':<10}{'Avg Rank':<10}")
    print(f"  {'-'*105}")

    winner_ids = []
    for cv_name, winner in per_cv_winners.items():
        winner_ids.append(winner["id"])
        print(f"  {cv_name:<20}{winner['id']:<45}"
              f"{winner['RMSE_mean']:<10.2f}"
              f"{winner['MAPE_mean']:<10.2f}"
              f"{winner['SMAPE_mean']:<10.2f}"
              f"{winner['avg_rank']:<10.2f}")

    # Check consistency
    unique_winners = list(set(winner_ids))
    n_unique = len(unique_winners)

    if n_unique == 1:
        print(f"\n  ✓ PERFECT CONSISTENCY: Same winner across ALL 4 CV strategies!")
        print(f"    Winner: {unique_winners[0]}")
        consistency = "perfect"
    else:
        print(f"\n  {n_unique} different winners across 4 CV strategies.")
        # Count how often each winner appears
        from collections import Counter
        winner_counts = Counter(winner_ids)
        print("  Winner frequency:")
        for wid, count in winner_counts.most_common():
            print(f"    {wid}: {count}/4 CV strategies")
        consistency = "partial"

    # Also show: for each unique winner, what is their rank under OTHER CV strategies?
    print(f"\n  Cross-CV rank matrix (rank of each winner under all CV strategies):")
    print(f"  {'Candidate':<45}", end="")
    for cv_name in cv_strategies:
        print(f"{cv_name:<18}", end="")
    print()
    print(f"  {'-'*117}")

    for uid in unique_winners:
        print(f"  {uid:<45}", end="")
        for cv_name in cv_strategies:
            df_r = per_cv_rankings[cv_name]
            row = df_r[df_r["id"] == uid]
            if len(row) > 0:
                rank_val = row.index[0] + 1  # 1-indexed position in ranking
                avg_r = row.iloc[0]["avg_rank"]
                print(f"#{rank_val:<3}(avg={avg_r:.1f})  ", end="")
            else:
                print(f"{'N/A':<18}", end="")
        print()

    # ========================================================================
    # STEP 4: Select the overall winner
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Overall Winner Selection")
    print("=" * 70)

    if consistency == "perfect":
        # Easy: same winner everywhere
        overall_winner_id = unique_winners[0]
        overall_cv = "Repeated_3x10"  # use the most stable CV for final metrics
        print(f"\n  Winner is consistent across all CV strategies.")
        print(f"  Using {overall_cv} for final reported metrics (most stable).")
    else:
        # Pick the candidate that has the best average rank ACROSS all 4 CVs
        # Compute mean of avg_rank across CV strategies for each candidate
        cross_cv_scores = {}
        for uid in df_all["id"].unique():
            ranks_across_cv = []
            for cv_name in cv_strategies:
                df_r = per_cv_rankings[cv_name]
                row = df_r[df_r["id"] == uid]
                if len(row) > 0:
                    ranks_across_cv.append(row.iloc[0]["avg_rank"])
            if ranks_across_cv:
                cross_cv_scores[uid] = np.mean(ranks_across_cv)

        # Best = lowest mean avg_rank across all CVs
        overall_winner_id = min(cross_cv_scores, key=cross_cv_scores.get)
        best_cross_score = cross_cv_scores[overall_winner_id]

        print(f"\n  Winners differ across CV strategies.")
        print(f"  Selecting candidate with best MEAN avg_rank across all 4 CVs.")
        print(f"\n  Top 5 by cross-CV mean rank:")
        sorted_cross = sorted(cross_cv_scores.items(), key=lambda x: x[1])
        for uid, score in sorted_cross[:5]:
            marker = " <<<" if uid == overall_winner_id else ""
            print(f"    {uid:<50} mean_avg_rank={score:.2f}{marker}")

        overall_cv = "Repeated_3x10"  # use for final reported metrics

    # Get the winner's results under the chosen CV
    df_final_cv = per_cv_rankings[overall_cv]
    winner_row = df_final_cv[df_final_cv["id"] == overall_winner_id].iloc[0]

    print(f"\n  {'='*50}")
    print(f"  OVERALL WINNER: {overall_winner_id}")
    print(f"  {'='*50}")
    print(f"  Model:             {winner_row['model']}")
    print(f"  Feature Selection: {winner_row['feature_selection']}")
    print(f"  N Features:        {int(winner_row['n_features'])}")
    print(f"\n  Metrics (under {overall_cv}):")
    print(f"    RMSE:   {winner_row['RMSE_mean']:.2f} ± {winner_row['RMSE_std']:.2f}  "
          f"(rank {int(winner_row['rank_RMSE'])})")
    print(f"    MAPE:   {winner_row['MAPE_mean']:.2f} ± {winner_row['MAPE_std']:.2f}  "
          f"(rank {int(winner_row['rank_MAPE'])})")
    print(f"    SMAPE:  {winner_row['SMAPE_mean']:.2f} ± {winner_row['SMAPE_std']:.2f}  "
          f"(rank {int(winner_row['rank_SMAPE'])})")
    print(f"    R2:     {winner_row['R2_train_mean']:.4f}")
    print(f"    Corr R2:{winner_row['Corr_R2_val_mean']:.4f}")
    print(f"\n  Average Rank: {winner_row['avg_rank']:.2f} "
          f"out of {len(df_final_cv)} candidates")

    # Show metrics under ALL CV strategies for the winner
    print(f"\n  Winner's metrics across all CV strategies:")
    print(f"  {'CV Strategy':<20}{'RMSE':<12}{'MAPE':<12}{'SMAPE':<12}{'Avg Rank':<12}")
    print(f"  {'-'*68}")
    for cv_name in cv_strategies:
        df_r = per_cv_rankings[cv_name]
        row = df_r[df_r["id"] == overall_winner_id]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {cv_name:<20}{r['RMSE_mean']:<12.2f}"
                  f"{r['MAPE_mean']:<12.2f}{r['SMAPE_mean']:<12.2f}"
                  f"{r['avg_rank']:<12.2f}")

    # ========================================================================
    # STEP 5: Feature importance for the winner
    # ========================================================================
    print("\n" + "=" * 70)
    print("SELECTED FEATURES (ranked by importance, high to low)")
    print("=" * 70)

    winner_cand = None
    for cand in candidates:
        if cand["id"] == overall_winner_id:
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

        imp_path = os.path.join(OUTPUT_DIR, "best_model_feature_importance.csv")
        df_imp.to_csv(imp_path, index=False)
        print(f"\n  Saved: {imp_path}")

    # ========================================================================
    # STEP 6: Comparison with baselines
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
    print(f"  {'>>> BEST MODEL: ' + winner_row['model']:<45}"
          f"{winner_row['RMSE_mean']:<12.2f}"
          f"{winner_row['MAPE_mean']:<12.2f}"
          f"{winner_row['SMAPE_mean']:<12.2f}")

    rmse_improvement = ((79.84 - winner_row['RMSE_mean']) / 79.84) * 100
    print(f"\n  RMSE improvement over previous best: "
          f"{rmse_improvement:.1f}%")

    # ========================================================================
    # STEP 7: Save final summary
    # ========================================================================
    summary = {
        "model_name": winner_row["model"],
        "feature_selection": winner_row["feature_selection"],
        "n_features": int(winner_row["n_features"]),
        "cv_strategy_reported": overall_cv,
        "cross_cv_consistency": consistency,
        "n_cv_strategies_tested": n_cv,
        "RMSE_mean": winner_row["RMSE_mean"],
        "RMSE_std": winner_row["RMSE_std"],
        "MAPE_mean": winner_row["MAPE_mean"],
        "MAPE_std": winner_row["MAPE_std"],
        "SMAPE_mean": winner_row["SMAPE_mean"],
        "SMAPE_std": winner_row["SMAPE_std"],
        "R2_train": winner_row["R2_train_mean"],
        "Corr_R2_val": winner_row["Corr_R2_val_mean"],
    }

    summary_path = os.path.join(OUTPUT_DIR, "best_model_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # Save cross-CV winner comparison
    cross_cv_rows = []
    for cv_name, winner in per_cv_winners.items():
        cross_cv_rows.append({
            "cv_strategy": cv_name,
            "winner_id": winner["id"],
            "model": winner["model"],
            "feature_selection": winner["feature_selection"],
            "n_features": int(winner["n_features"]),
            "RMSE_mean": winner["RMSE_mean"],
            "MAPE_mean": winner["MAPE_mean"],
            "SMAPE_mean": winner["SMAPE_mean"],
            "avg_rank": winner["avg_rank"],
        })
    cross_cv_path = os.path.join(OUTPUT_DIR, "cross_cv_winners.csv")
    pd.DataFrame(cross_cv_rows).to_csv(cross_cv_path, index=False)

    print(f"\n  All results saved to: {OUTPUT_DIR}/")
    print(f"    - all_candidates_all_cv.csv        (all results, all CVs)")
    print(f"    - ranking_<cv_name>.csv            (per-CV rankings)")
    print(f"    - cross_cv_winners.csv             (winner per CV strategy)")
    print(f"    - best_model_feature_importance.csv (features)")
    print(f"    - best_model_summary.csv           (final answer)")
    print("\nDone!")


if __name__ == "__main__":
    main()