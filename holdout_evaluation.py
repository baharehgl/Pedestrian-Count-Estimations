"""
Holdout Evaluation: Train on train set, predict on holdout test set.
No cross-validation — pure train/test split using the 'holdout' column.

Tests all combinations of:
  - 4 feature selection methods (L1 Lasso, MI, RF Importance, F-Regression)
  - 5 feature counts (10, 15, 20, 25, 30)
  - 7 tuned models (best hyperparameters from Experiment 3)

Reports: RMSE, MAPE, SMAPE, R² on holdout for every combination.
Ranks and selects the best.

"""

import os
import warnings
import numpy as np
import pandas as pd

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
)
from sklearn.linear_model import PoissonRegressor, LassoCV
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "df1_v1a_out.csv"  # Update if needed
TARGET_COL = "pm_tot"
SPLIT_COL = "holdout"
ID_COL = "site_id"
CATEGORICAL_COLS = [
    "Street Nam", "_Date", "season", "geometry",
    "class_type", "speed_type", "crossing_class"
]
RANDOM_STATE = 42
OUTPUT_DIR = "holdout_results"


# ============================================================================
# METRICS
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

def correlation_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return corr ** 2


# ============================================================================
# FEATURE SELECTION METHODS (fit on TRAIN only)
# ============================================================================

def select_features_l1(X_train, y_train, feature_names, n_features=20):
    y_log = np.log1p(y_train)
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train, y_log)
    importance = np.abs(lasso.coef_)
    top_idx = np.argsort(importance)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names

def select_features_mi(X_train, y_train, feature_names, n_features=20):
    mi_scores = mutual_info_regression(
        X_train, y_train, random_state=RANDOM_STATE, n_neighbors=5
    )
    top_idx = np.argsort(mi_scores)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names

def select_features_rf(X_train, y_train, feature_names, n_features=20):
    rf = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_
    top_idx = np.argsort(importance)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names

def select_features_f_regression(X_train, y_train, feature_names, n_features=20):
    f_scores, _ = f_regression(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    top_idx = np.argsort(f_scores)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


# ============================================================================
# TUNED MODELS (best hyperparameters from Experiment 3)
# ============================================================================

def get_tuned_models():
    """Returns 7 models with their best tuned hyperparameters from Exp 3."""
    models = {}

    models["HistGB_Poisson"] = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.01, max_iter=500,
        max_depth=7, min_samples_leaf=10, l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=800, max_depth=25, min_samples_split=5,
        min_samples_leaf=1, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    models["HistGB_SquaredError"] = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=300,
        max_depth=7, min_samples_leaf=10, l2_regularization=0.0,
        random_state=RANDOM_STATE
    )

    models["GradBoost_Huber"] = GradientBoostingRegressor(
        loss="huber", learning_rate=0.01, n_estimators=500,
        max_depth=7, min_samples_split=10, min_samples_leaf=3,
        subsample=1.0, alpha=0.8,
        random_state=RANDOM_STATE
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=2,
        min_samples_leaf=1, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    models["PoissonRegressor"] = PoissonRegressor(
        alpha=10.0, max_iter=5000
    )

    models["Bagging_HistGB"] = BaggingRegressor(
        estimator=HistGradientBoostingRegressor(
            loss="poisson", random_state=RANDOM_STATE,
            max_iter=500, learning_rate=0.05
        ),
        n_estimators=10, max_features=0.8, max_samples=1.0,
        random_state=RANDOM_STATE, n_jobs=-1
    )

    return models


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("HOLDOUT EVALUATION")
    print("Train on train set → Predict on holdout test set")
    print("No cross-validation — pure train/test split")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data and split by holdout column
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found at '{DATA_PATH}'")
        print("Please update DATA_PATH at the top of this script.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset shape: {df.shape}")
    print(f"Holdout column values: {df[SPLIT_COL].value_counts().to_dict()}")

    # Split into train and test
    train_mask = df[SPLIT_COL] == 0
    test_mask = df[SPLIT_COL] == 1

    y_all = df[TARGET_COL].values
    drop_cols = [c for c in [TARGET_COL, SPLIT_COL, ID_COL] if c in df.columns]
    X_all = df.drop(columns=drop_cols)

    X_train_raw = X_all[train_mask]
    X_test_raw = X_all[test_mask]
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    print(f"\nTrain samples: {len(y_train)}")
    print(f"Test (holdout) samples: {len(y_test)}")
    print(f"Train target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"Test target:  mean={y_test.mean():.2f}, std={y_test.std():.2f}")

    # ------------------------------------------------------------------
    # 2. Preprocess — fit on TRAIN only, transform both
    # ------------------------------------------------------------------
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_all.columns]
    num_cols = [c for c in X_all.columns if c not in cat_cols]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # FIT on train, transform both
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        num_names = num_cols
        cat_names = []
        if cat_cols:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names = num_names + cat_names

    print(f"Total features after preprocessing: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 3. Define all combinations
    # ------------------------------------------------------------------
    fs_methods = {
        "L1_Lasso": select_features_l1,
        "Mutual_Information": select_features_mi,
        "RF_Importance": select_features_rf,
        "F_Regression": select_features_f_regression,
    }

    n_features_list = [10, 15, 20, 25, 30]
    models = get_tuned_models()

    total = len(fs_methods) * len(n_features_list) * len(models)
    print(f"\nTotal combinations: {len(fs_methods)} FS × {len(n_features_list)} counts × {len(models)} models = {total}")

    # ------------------------------------------------------------------
    # 4. Run all combinations
    # ------------------------------------------------------------------
    results = []
    run_count = 0

    for fs_name, fs_func in fs_methods.items():
        for n_feat in n_features_list:

            # Feature selection — fit on TRAIN only
            feat_idx, feat_names = fs_func(
                X_train, y_train, feature_names, n_features=n_feat
            )

            X_train_sel = X_train[:, feat_idx]
            X_test_sel = X_test[:, feat_idx]

            for model_name, model in models.items():
                run_count += 1

                # Train on train set
                m = clone(model)
                m.fit(X_train_sel, y_train)

                # Predict on train (for R²)
                y_pred_train = np.maximum(m.predict(X_train_sel), 0)

                # Predict on holdout test
                y_pred_test = np.maximum(m.predict(X_test_sel), 0)

                # Compute metrics on holdout
                rmse = rmse_score(y_test, y_pred_test)
                mape = standard_mape(y_test, y_pred_test)
                smape = smape_score(y_test, y_pred_test)
                corr_r2 = correlation_r2(y_test, y_pred_test)
                train_r2 = correlation_r2(y_train, y_pred_train)

                results.append({
                    "feature_selection": fs_name,
                    "n_features": n_feat,
                    "model": model_name,
                    "RMSE_holdout": rmse,
                    "MAPE_holdout": mape,
                    "SMAPE_holdout": smape,
                    "Corr_R2_holdout": corr_r2,
                    "Corr_R2_train": train_r2,
                })

                if run_count % 20 == 0 or run_count == total:
                    print(f"  [{run_count}/{total}] {fs_name}_{n_feat}_{model_name}: "
                          f"RMSE={rmse:.2f}, MAPE={mape:.2f}, SMAPE={smape:.2f}")

    # ------------------------------------------------------------------
    # 5. Build results DataFrame and rank
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    # Rank by each metric
    df_results["rank_RMSE"] = df_results["RMSE_holdout"].rank()
    df_results["rank_MAPE"] = df_results["MAPE_holdout"].rank()
    df_results["rank_SMAPE"] = df_results["SMAPE_holdout"].rank()
    df_results["avg_rank"] = (
        df_results["rank_RMSE"] +
        df_results["rank_MAPE"] +
        df_results["rank_SMAPE"]
    ) / 3.0

    # Sort by average rank
    df_results = df_results.sort_values("avg_rank").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 15 MODELS ON HOLDOUT (ranked by average rank)")
    print("=" * 80)

    top15 = df_results.head(15)
    for i, row in top15.iterrows():
        print(f"  #{i+1:2d}  {row['feature_selection']:<20} {row['n_features']:2.0f} feat  "
              f"{row['model']:<22} RMSE={row['RMSE_holdout']:7.2f}  "
              f"MAPE={row['MAPE_holdout']:6.2f}  SMAPE={row['SMAPE_holdout']:6.2f}  "
              f"AvgRank={row['avg_rank']:.1f}")

    # ------------------------------------------------------------------
    # 7. Best result details
    # ------------------------------------------------------------------
    best = df_results.iloc[0]
    print("\n" + "=" * 80)
    print("BEST MODEL ON HOLDOUT")
    print("=" * 80)
    print(f"  Model:             {best['model']}")
    print(f"  Feature Selection: {best['feature_selection']}")
    print(f"  N Features:        {best['n_features']:.0f}")
    print(f"  RMSE (holdout):    {best['RMSE_holdout']:.2f}")
    print(f"  MAPE (holdout):    {best['MAPE_holdout']:.2f}")
    print(f"  SMAPE (holdout):   {best['SMAPE_holdout']:.2f}")
    print(f"  Corr R² (holdout): {best['Corr_R2_holdout']:.4f}")
    print(f"  Corr R² (train):   {best['Corr_R2_train']:.4f}")
    print(f"  Average Rank:      {best['avg_rank']:.2f} / {len(df_results)}")

    # ------------------------------------------------------------------
    # 8. Best per model type
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST RESULT PER MODEL TYPE")
    print("=" * 80)

    for model_name in models.keys():
        sub = df_results[df_results["model"] == model_name]
        b = sub.iloc[0]  # already sorted by avg_rank
        print(f"  {model_name:<22} {b['feature_selection']:<20} {b['n_features']:2.0f} feat  "
              f"RMSE={b['RMSE_holdout']:7.2f}  MAPE={b['MAPE_holdout']:6.2f}  "
              f"SMAPE={b['SMAPE_holdout']:6.2f}")

    # ------------------------------------------------------------------
    # 9. Best per feature selection method
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST RESULT PER FEATURE SELECTION METHOD")
    print("=" * 80)

    for fs_name in fs_methods.keys():
        sub = df_results[df_results["feature_selection"] == fs_name]
        b = sub.iloc[0]
        print(f"  {fs_name:<22} {b['n_features']:2.0f} feat  {b['model']:<22} "
              f"RMSE={b['RMSE_holdout']:7.2f}  MAPE={b['MAPE_holdout']:6.2f}  "
              f"SMAPE={b['SMAPE_holdout']:6.2f}")

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, "holdout_all_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")

    # Save top 15
    top_path = os.path.join(OUTPUT_DIR, "holdout_top15.csv")
    top15.to_csv(top_path, index=False)
    print(f"Top 15 saved to: {top_path}")

    # Save best per model
    best_per_model = []
    for model_name in models.keys():
        sub = df_results[df_results["model"] == model_name]
        best_per_model.append(sub.iloc[0])
    pd.DataFrame(best_per_model).to_csv(
        os.path.join(OUTPUT_DIR, "holdout_best_per_model.csv"), index=False
    )

    print(f"\n{'=' * 80}")
    print(f"Done! {total} combinations evaluated on holdout set.")
    print(f"Results in: {OUTPUT_DIR}/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()