"""
Pedestrian Count Estimation - Hyperparameter Tuning & Advanced CV Experiments
==============================================================================
This script systematically tests improvements to reduce RMSE and MAPE:
  1. 10-fold cross-validation (instead of 5)
  2. Repeated cross-validation (e.g., 5x10-fold)
  3. Different feature counts (10, 15, 20, 25, 30)
  4. Hyperparameter grid search for each model
  5. Additional models: ExtraTrees, GradientBoosting, Ridge-based Poisson

"""

import os
import warnings
import numpy as np
import pandas as pd
from itertools import product as iter_product

from sklearn.model_selection import (
    KFold, RepeatedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
)
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
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, f_regression
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, make_scorer
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION - Update these paths as needed
# ============================================================================
DATA_PATH = "df1_v1a_out.csv"  # Update if needed
TARGET_COL = "pm_tot"
SPLIT_COL = "holdout"
ID_COL = "site_id"
DROP_COLS = [TARGET_COL, SPLIT_COL, ID_COL]
CATEGORICAL_COLS = [
    "Street Nam", "_Date", "season", "geometry",
    "class_type", "speed_type", "crossing_class"
]
RANDOM_STATE = 42
OUTPUT_DIR = "tuning_results"

# ============================================================================
# METRICS
# ============================================================================

def rmse_score(y_true, y_pred):
    """Lower is better."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape_score(y_true, y_pred):
    """Lower is better. Multiply by 100 for percentage."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mcfadden_r2(y_true, y_pred):
    """McFadden's pseudo-R² (deviance-based)."""
    y_pred_clipped = np.maximum(y_pred, 1e-8)
    y_mean = np.mean(y_true)
    y_mean = max(y_mean, 1e-8)
    # Poisson deviance
    ll_model = np.sum(y_true * np.log(y_pred_clipped) - y_pred_clipped)
    ll_null = np.sum(y_true * np.log(y_mean) - y_mean)
    if ll_null == 0:
        return 0.0
    return 1.0 - (ll_model / ll_null)

# Sklearn scorers (neg because sklearn maximizes)
neg_rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
neg_mape_scorer = make_scorer(mape_score, greater_is_better=False)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess(data_path):
    """Load data, identify feature types, build preprocessor."""
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {TARGET_COL}, mean={df[TARGET_COL].mean():.2f}, "
          f"median={df[TARGET_COL].median():.2f}, std={df[TARGET_COL].std():.2f}")

    y = df[TARGET_COL].values
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing)

    # Identify categorical vs numeric columns
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

    # Build preprocessing pipeline
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

    return X, y, preprocessor, num_cols, cat_cols


# ============================================================================
# FEATURE SELECTION METHODS
# ============================================================================

def select_features_l1(X_transformed, y, feature_names, n_features=20):
    """L1 (Lasso) based feature selection using Poisson regression."""
    from sklearn.linear_model import LassoCV
    # Use LassoCV on log(y+1) for feature ranking
    y_log = np.log1p(y)
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_transformed, y_log)
    importance = np.abs(lasso.coef_)
    top_idx = np.argsort(importance)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


def select_features_mi(X_transformed, y, feature_names, n_features=20):
    """Mutual Information based feature selection."""
    mi_scores = mutual_info_regression(
        X_transformed, y, random_state=RANDOM_STATE, n_neighbors=5
    )
    top_idx = np.argsort(mi_scores)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


def select_features_rf(X_transformed, y, feature_names, n_features=20):
    """Random Forest importance based feature selection."""
    rf = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_transformed, y)
    importance = rf.feature_importances_
    top_idx = np.argsort(importance)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


def select_features_f_regression(X_transformed, y, feature_names, n_features=20):
    """F-statistic based feature selection (new method)."""
    f_scores, _ = f_regression(X_transformed, y)
    f_scores = np.nan_to_num(f_scores)
    top_idx = np.argsort(f_scores)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


# ============================================================================
# MODEL DEFINITIONS WITH HYPERPARAMETER GRIDS
# ============================================================================

def get_model_configs():
    """
    Returns model configs with hyperparameter grids.
    Each entry: (name, base_model, param_grid)
    """
    configs = {}

    # --- Random Forest ---
    configs["RandomForest"] = {
        "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "n_estimators": [200, 500, 800],
            "max_depth": [None, 15, 25, 35],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
        },
        # Focused grid for faster search
        "param_grid_fast": {
            "n_estimators": [300, 500],
            "max_depth": [None, 20, 30],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.5],
        },
    }

    # --- HistGradientBoosting (Poisson) ---
    configs["HistGB_Poisson"] = {
        "model": HistGradientBoostingRegressor(
            loss="poisson", random_state=RANDOM_STATE
        ),
        "param_grid": {
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "max_iter": [200, 500, 800, 1000],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [5, 10, 20, 30],
            "max_leaf_nodes": [15, 31, 63, None],
            "l2_regularization": [0.0, 0.1, 1.0, 10.0],
        },
        "param_grid_fast": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [300, 500, 800],
            "max_depth": [5, 7, 10],
            "min_samples_leaf": [10, 20],
            "l2_regularization": [0.0, 1.0],
        },
    }

    # --- HistGradientBoosting (Squared Error) - NEW ---
    configs["HistGB_SquaredError"] = {
        "model": HistGradientBoostingRegressor(
            loss="squared_error", random_state=RANDOM_STATE
        ),
        "param_grid_fast": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [300, 500, 800],
            "max_depth": [5, 7, 10],
            "min_samples_leaf": [10, 20],
            "l2_regularization": [0.0, 1.0, 10.0],
        },
    }

    # --- ExtraTrees (often outperforms RF on small datasets) - NEW ---
    configs["ExtraTrees"] = {
        "model": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid_fast": {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 15, 25],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.5, 0.8],
        },
    }

    # --- GradientBoosting (sklearn's original, with Huber loss for robustness) - NEW ---
    configs["GradientBoosting_Huber"] = {
        "model": GradientBoostingRegressor(
            loss="huber", random_state=RANDOM_STATE
        ),
        "param_grid_fast": {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 500],
            "max_depth": [3, 5, 7],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [3, 5, 10],
            "subsample": [0.8, 1.0],
            "alpha": [0.8, 0.9],
        },
    }

    # --- Poisson Regression (regularized GLM) - NEW ---
    configs["PoissonRegressor"] = {
        "model": PoissonRegressor(max_iter=5000),
        "param_grid_fast": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        },
    }

    # --- Bagging + HistGB (reduce variance) - NEW ---
    configs["Bagging_HistGB"] = {
        "model": BaggingRegressor(
            estimator=HistGradientBoostingRegressor(
                loss="poisson", random_state=RANDOM_STATE,
                max_iter=500, learning_rate=0.05
            ),
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "param_grid_fast": {
            "n_estimators": [5, 10, 15],
            "max_samples": [0.7, 0.8, 1.0],
            "max_features": [0.7, 0.8, 1.0],
        },
    }

    return configs


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_cv_experiment(
    X_selected, y, model, cv_strategy, model_name="", return_predictions=False
):
    """
    Run cross-validation and return metrics.
    """
    rmse_scores = []
    mape_scores = []
    r2_scores = []

    for train_idx, val_idx in cv_strategy.split(X_selected):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_clone = _clone_model(model)
        model_clone.fit(X_train, y_train)

        y_pred_train = model_clone.predict(X_train)
        y_pred_val = model_clone.predict(X_val)

        # Clip predictions to be non-negative
        y_pred_val = np.maximum(y_pred_val, 0)
        y_pred_train = np.maximum(y_pred_train, 0)

        rmse_scores.append(rmse_score(y_val, y_pred_val))
        mape_scores.append(mape_score(y_val, y_pred_val))
        r2_scores.append(mcfadden_r2(y_train, y_pred_train))

    results = {
        "model": model_name,
        "R2_train_mean": np.mean(r2_scores),
        "R2_train_std": np.std(r2_scores),
        "MAPE_val_mean": np.mean(mape_scores),
        "MAPE_val_std": np.std(mape_scores),
        "RMSE_val_mean": np.mean(rmse_scores),
        "RMSE_val_std": np.std(rmse_scores),
    }
    return results


def _clone_model(model):
    """Clone a sklearn model."""
    from sklearn.base import clone
    return clone(model)


def run_hyperparameter_search(
    X_selected, y, model, param_grid, cv_strategy, model_name="",
    search_method="random", n_iter=50
):
    """
    Run hyperparameter search and return best model + results.
    """
    if search_method == "grid":
        search = GridSearchCV(
            model, param_grid, cv=cv_strategy,
            scoring=neg_rmse_scorer, n_jobs=-1,
            refit=True, verbose=0
        )
    else:
        search = RandomizedSearchCV(
            model, param_grid, cv=cv_strategy,
            scoring=neg_rmse_scorer, n_jobs=-1,
            n_iter=min(n_iter, _count_grid_combos(param_grid)),
            refit=True, random_state=RANDOM_STATE, verbose=0
        )

    search.fit(X_selected, y)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_rmse = -search.best_score_  # neg_rmse -> positive

    # Also compute MAPE with the best model via manual CV
    cv_results = run_cv_experiment(
        X_selected, y, best_model, cv_strategy, model_name
    )
    cv_results["best_params"] = str(best_params)
    cv_results["search_best_rmse"] = best_rmse

    return cv_results, best_model, best_params


def _count_grid_combos(param_grid):
    """Count total combinations in a parameter grid."""
    total = 1
    for key, values in param_grid.items():
        total *= len(values)
    return total


# ============================================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("PEDESTRIAN COUNT ESTIMATION - HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found at '{DATA_PATH}'")
        print("Please update DATA_PATH at the top of this script.")
        return

    X, y, preprocessor, num_cols, cat_cols = load_and_preprocess(DATA_PATH)

    # Fit preprocessor and transform
    X_transformed = preprocessor.fit_transform(X)

    # Get feature names after transformation
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
    print(f"Dataset size: {X_transformed.shape[0]} samples")

    # ========================================================================
    # EXPERIMENT 1: Compare CV strategies (5-fold vs 10-fold vs Repeated)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Cross-Validation Strategy Comparison")
    print("=" * 80)

    cv_strategies = {
        "5-Fold": KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        "10-Fold": KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
        "RepeatedKFold_5x10": RepeatedKFold(
            n_splits=10, n_repeats=5, random_state=RANDOM_STATE
        ),
        "RepeatedKFold_3x10": RepeatedKFold(
            n_splits=10, n_repeats=3, random_state=RANDOM_STATE
        ),
    }

    # Use L1 selected features (20) as baseline for this comparison
    print("\nSelecting top 20 features with L1 (Lasso)...")
    l1_idx, l1_names = select_features_l1(
        X_transformed, y, feature_names, n_features=20
    )
    X_l1_20 = X_transformed[:, l1_idx]

    baseline_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGB_Poisson": HistGradientBoostingRegressor(
            loss="poisson", max_iter=500, random_state=RANDOM_STATE
        ),
    }

    cv_comparison_results = []
    for cv_name, cv_strat in cv_strategies.items():
        for model_name, model in baseline_models.items():
            print(f"  Testing {model_name} with {cv_name}...", end=" ")
            res = run_cv_experiment(
                X_l1_20, y, model, cv_strat, model_name=f"{model_name}"
            )
            res["cv_strategy"] = cv_name
            cv_comparison_results.append(res)
            print(f"RMSE={res['RMSE_val_mean']:.2f}±{res['RMSE_val_std']:.2f}, "
                  f"MAPE={res['MAPE_val_mean']:.2f}±{res['MAPE_val_std']:.2f}")

    df_cv_comp = pd.DataFrame(cv_comparison_results)
    cv_comp_path = os.path.join(OUTPUT_DIR, "experiment1_cv_strategy_comparison.csv")
    df_cv_comp.to_csv(cv_comp_path, index=False)
    print(f"\nResults saved to {cv_comp_path}")

    # ========================================================================
    # EXPERIMENT 2: Feature Count Sweep (10, 15, 20, 25, 30)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Feature Count Sweep")
    print("=" * 80)

    feature_counts = [10, 15, 20, 25, 30]
    feature_methods = {
        "L1_Lasso": select_features_l1,
        "Mutual_Information": select_features_mi,
        "Random_Forest": select_features_rf,
        "F_Regression": select_features_f_regression,
    }

    # Use 10-fold CV for this experiment
    cv_10fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    feature_sweep_results = []
    for n_feat in feature_counts:
        for fs_name, fs_func in feature_methods.items():
            print(f"\n  Feature selection: {fs_name}, n_features={n_feat}")
            try:
                feat_idx, feat_names = fs_func(
                    X_transformed, y, feature_names, n_features=n_feat
                )
                X_selected = X_transformed[:, feat_idx]
            except Exception as e:
                print(f"    ERROR in feature selection: {e}")
                continue

            for model_name, model in baseline_models.items():
                print(f"    {model_name}...", end=" ")
                res = run_cv_experiment(
                    X_selected, y, model, cv_10fold,
                    model_name=model_name
                )
                res["feature_selection"] = fs_name
                res["n_features"] = n_feat
                feature_sweep_results.append(res)
                print(f"RMSE={res['RMSE_val_mean']:.2f}, "
                      f"MAPE={res['MAPE_val_mean']:.2f}")

    df_feat_sweep = pd.DataFrame(feature_sweep_results)
    feat_sweep_path = os.path.join(OUTPUT_DIR, "experiment2_feature_count_sweep.csv")
    df_feat_sweep.to_csv(feat_sweep_path, index=False)
    print(f"\nResults saved to {feat_sweep_path}")

    # Find best feature config
    if len(feature_sweep_results) > 0:
        best_rmse_row = df_feat_sweep.loc[df_feat_sweep["RMSE_val_mean"].idxmin()]
        best_mape_row = df_feat_sweep.loc[df_feat_sweep["MAPE_val_mean"].idxmin()]
        print(f"\n  Best RMSE: {best_rmse_row['RMSE_val_mean']:.2f} "
              f"({best_rmse_row['model']}, {best_rmse_row['feature_selection']}, "
              f"n={best_rmse_row['n_features']})")
        print(f"  Best MAPE: {best_mape_row['MAPE_val_mean']:.2f} "
              f"({best_mape_row['model']}, {best_mape_row['feature_selection']}, "
              f"n={best_mape_row['n_features']})")

    # ========================================================================
    # EXPERIMENT 3: Hyperparameter Tuning (Main Experiment)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Hyperparameter Tuning with RandomizedSearchCV")
    print("=" * 80)

    # Use 10-fold repeated CV for tuning
    cv_tuning = RepeatedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    # Select best feature configs to tune on
    # We'll test L1 with 20 features (your current best) + the best from Exp 2
    tune_configs = [
        ("L1_Lasso", select_features_l1, 20),
        ("L1_Lasso", select_features_l1, 15),
        ("L1_Lasso", select_features_l1, 25),
    ]

    # Add the best from feature sweep if different
    if len(feature_sweep_results) > 0:
        best_fs = best_rmse_row["feature_selection"]
        best_n = int(best_rmse_row["n_features"])
        if (best_fs, best_n) not in [(t[0], t[2]) for t in tune_configs]:
            fs_map = {
                "L1_Lasso": select_features_l1,
                "Mutual_Information": select_features_mi,
                "Random_Forest": select_features_rf,
                "F_Regression": select_features_f_regression,
            }
            if best_fs in fs_map:
                tune_configs.append((best_fs, fs_map[best_fs], best_n))

    model_configs = get_model_configs()
    tuning_results = []

    for fs_name, fs_func, n_feat in tune_configs:
        print(f"\n--- Feature Selection: {fs_name}, n_features={n_feat} ---")
        feat_idx, feat_names = fs_func(
            X_transformed, y, feature_names, n_features=n_feat
        )
        X_selected = X_transformed[:, feat_idx]

        for model_name, config in model_configs.items():
            grid = config.get("param_grid_fast", config.get("param_grid", {}))
            if not grid:
                continue

            n_combos = _count_grid_combos(grid)
            n_iter = min(40, n_combos)  # Cap iterations for speed

            print(f"  Tuning {model_name} ({n_combos} combos, "
                  f"testing {n_iter})...", end=" ", flush=True)

            try:
                res, best_model, best_params = run_hyperparameter_search(
                    X_selected, y, config["model"], grid,
                    cv_tuning, model_name=model_name,
                    search_method="random", n_iter=n_iter
                )
                res["feature_selection"] = fs_name
                res["n_features"] = n_feat
                tuning_results.append(res)
                print(f"RMSE={res['RMSE_val_mean']:.2f}, "
                      f"MAPE={res['MAPE_val_mean']:.2f}")
                print(f"    Best params: {best_params}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

    df_tuning = pd.DataFrame(tuning_results)
    tuning_path = os.path.join(OUTPUT_DIR, "experiment3_hyperparameter_tuning.csv")
    df_tuning.to_csv(tuning_path, index=False)
    print(f"\nResults saved to {tuning_path}")

    # ========================================================================
    # EXPERIMENT 4: Log-transform target (common for count data)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Log-Transform Target Variable")
    print("=" * 80)
    print("  Training on log(y+1) and back-transforming predictions")

    y_log = np.log1p(y)
    cv_10fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    log_transform_results = []
    # Use L1 20 features
    feat_idx_l1, _ = select_features_l1(
        X_transformed, y, feature_names, n_features=20
    )
    X_l1 = X_transformed[:, feat_idx_l1]

    log_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
            max_depth=None, min_samples_leaf=2
        ),
        "HistGB_Poisson": HistGradientBoostingRegressor(
            loss="poisson", max_iter=500, learning_rate=0.05,
            random_state=RANDOM_STATE
        ),
        "HistGB_SquaredError": HistGradientBoostingRegressor(
            loss="squared_error", max_iter=500, learning_rate=0.05,
            random_state=RANDOM_STATE
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    for model_name, model in log_models.items():
        rmse_scores = []
        mape_scores = []

        for train_idx, val_idx in cv_10fold.split(X_l1):
            X_train, X_val = X_l1[train_idx], X_l1[val_idx]
            y_train_log, y_val = y_log[train_idx], y[val_idx]

            m = _clone_model(model)
            m.fit(X_train, y_train_log)
            y_pred_log = m.predict(X_val)
            y_pred = np.expm1(y_pred_log)  # back-transform
            y_pred = np.maximum(y_pred, 0)

            rmse_scores.append(rmse_score(y_val, y_pred))
            mape_scores.append(mape_score(y_val, y_pred))

        res = {
            "model": model_name,
            "target_transform": "log1p",
            "RMSE_val_mean": np.mean(rmse_scores),
            "RMSE_val_std": np.std(rmse_scores),
            "MAPE_val_mean": np.mean(mape_scores),
            "MAPE_val_std": np.std(mape_scores),
            "n_features": 20,
            "feature_selection": "L1_Lasso",
        }
        log_transform_results.append(res)
        print(f"  {model_name}: RMSE={res['RMSE_val_mean']:.2f}±{res['RMSE_val_std']:.2f}, "
              f"MAPE={res['MAPE_val_mean']:.2f}±{res['MAPE_val_std']:.2f}")

    df_log = pd.DataFrame(log_transform_results)
    log_path = os.path.join(OUTPUT_DIR, "experiment4_log_transform.csv")
    df_log.to_csv(log_path, index=False)
    print(f"\nResults saved to {log_path}")

    # ========================================================================
    # EXPERIMENT 5: Stacking / Blending Ensemble
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Stacking Ensemble")
    print("=" * 80)

    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import RidgeCV

    stacking_configs = [
        {
            "name": "Stack_RF+HistGB+ExtraTrees",
            "estimators": [
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
            "final_estimator": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        },
        {
            "name": "Stack_RF+HistGB_Poisson+HistGB_SE",
            "estimators": [
                ("rf", RandomForestRegressor(
                    n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
                )),
                ("hgb_p", HistGradientBoostingRegressor(
                    loss="poisson", max_iter=500, learning_rate=0.05,
                    random_state=RANDOM_STATE
                )),
                ("hgb_se", HistGradientBoostingRegressor(
                    loss="squared_error", max_iter=500, learning_rate=0.05,
                    random_state=RANDOM_STATE
                )),
            ],
            "final_estimator": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        },
    ]

    stacking_results = []
    for stack_config in stacking_configs:
        stacker = StackingRegressor(
            estimators=stack_config["estimators"],
            final_estimator=stack_config["final_estimator"],
            cv=5, n_jobs=-1
        )
        print(f"  {stack_config['name']}...", end=" ", flush=True)
        res = run_cv_experiment(
            X_l1, y, stacker, cv_10fold,
            model_name=stack_config["name"]
        )
        res["n_features"] = 20
        res["feature_selection"] = "L1_Lasso"
        stacking_results.append(res)
        print(f"RMSE={res['RMSE_val_mean']:.2f}±{res['RMSE_val_std']:.2f}, "
              f"MAPE={res['MAPE_val_mean']:.2f}±{res['MAPE_val_std']:.2f}")

    df_stack = pd.DataFrame(stacking_results)
    stack_path = os.path.join(OUTPUT_DIR, "experiment5_stacking.csv")
    df_stack.to_csv(stack_path, index=False)
    print(f"\nResults saved to {stack_path}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL EXPERIMENTS")
    print("=" * 80)

    all_results = []
    for df_res, exp_name in [
        (df_cv_comp, "CV_Strategy"),
        (df_feat_sweep, "Feature_Sweep"),
        (df_tuning, "HP_Tuning"),
        (df_log, "Log_Transform"),
        (df_stack, "Stacking"),
    ]:
        df_tmp = df_res.copy()
        df_tmp["experiment"] = exp_name
        all_results.append(df_tmp)

    df_all = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(OUTPUT_DIR, "all_experiments_summary.csv")
    df_all.to_csv(summary_path, index=False)

    # Print top results
    print("\nTop 10 by RMSE (lower is better):")
    top_rmse = df_all.nsmallest(10, "RMSE_val_mean")[
        ["experiment", "model", "RMSE_val_mean", "RMSE_val_std",
         "MAPE_val_mean", "MAPE_val_std"]
    ]
    print(top_rmse.to_string(index=False))

    print("\nTop 10 by MAPE (lower is better):")
    top_mape = df_all.nsmallest(10, "MAPE_val_mean")[
        ["experiment", "model", "MAPE_val_mean", "MAPE_val_std",
         "RMSE_val_mean", "RMSE_val_std"]
    ]
    print(top_mape.to_string(index=False))

    # Baseline comparison
    print("\n" + "-" * 60)
    print("YOUR CURRENT BEST BASELINES (from presentations):")
    print("  L1 + RandomForest (20 feat, 5-fold CV): RMSE=79.84, MAPE=50.48")
    print("  L1 + HistGB_Poisson (18 feat, holdout): RMSE=133.60, MAPE=45.11")
    print("  R-code HistGB (10 feat, 5-fold CV):     RMSE=89.76, MAPE=49.75")
    print("-" * 60)

    overall_best_rmse = df_all.loc[df_all["RMSE_val_mean"].idxmin()]
    overall_best_mape = df_all.loc[df_all["MAPE_val_mean"].idxmin()]
    print(f"\nBEST RMSE FOUND: {overall_best_rmse['RMSE_val_mean']:.2f} "
          f"({overall_best_rmse['experiment']}, {overall_best_rmse['model']})")
    print(f"BEST MAPE FOUND: {overall_best_mape['MAPE_val_mean']:.2f} "
          f"({overall_best_mape['experiment']}, {overall_best_mape['model']})")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()