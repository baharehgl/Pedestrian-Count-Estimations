"""
Pedestrian Count Estimation - Complete Experiments
==============================================================================
Combined script that includes:
  - Corrected metrics: SMAPE (matching R code's cvstats), standard MAPE,
    McFadden R2 (deviance-based), and Correlation R2
  - Experiment 1: CV strategy comparison (5-fold, 10-fold, Repeated)
                  NOW with ALL 4 feature selection methods (L1, MI, RF, F-Reg)
  - Experiment 2: Feature count sweep (10, 15, 20, 25, 30)
  - Experiment 3: Hyperparameter tuning (RandomizedSearchCV)
                  NOW with ALL 4 CV strategies × ALL 5 feature counts
  - Experiment 4: Log-transform target
  - Experiment 5: Stacking ensembles

Changes from v1:
  - Exp 1: Tests all 4 feature selection methods (was only L1 Lasso)
  - Exp 3: Uses all 4 CV strategies from Exp 1 + all 5 feature counts from
           Exp 2 (was only Repeated 3x10 with 3 feature counts)

"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold, RepeatedKFold, GridSearchCV, RandomizedSearchCV
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
    StackingRegressor,
)
from sklearn.linear_model import PoissonRegressor, LassoCV, RidgeCV
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import clone

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
# CORRECTED METRICS (matching R code behavior)
# ============================================================================

def rmse_score(y_true, y_pred):
    """Root Mean Squared Error. Same in both R and Python."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def standard_mape(y_true, y_pred):
    """
    Standard MAPE: mean(|pred - obs| / obs) * 100
    Used in R code for HOLDOUT evaluation (nb1a, nb2a, nb3a).
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100


def smape_score(y_true, y_pred):
    """
    Symmetric MAPE: mean(|pred - obs| / ((obs + pred) / 2)) * 100

    This is what the R code uses in cvstats() for cross-validation:
      mape = mean(abs((pred - obs) / (obs/2 + pred/2) * 100))

    SMAPE is typically LOWER than standard MAPE because the denominator
    is larger. This is why our Python MAPE looked worse than R's CV MAPE.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(
        np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    ) * 100


def mcfadden_r2_deviance(y_true, y_pred):
    """
    McFadden's pseudo-R2 (deviance-based, matching R code).

    R code: mf_r2 = 1 - (model$deviance / model$null.deviance)

    For Poisson-family models:
      deviance = 2 * sum(y*log(y/mu) - (y - mu))
      null model predicts y_bar for everything.
    """
    y_pred_safe = np.maximum(y_pred, 1e-10)
    y_bar = np.mean(y_true)
    y_bar_safe = max(y_bar, 1e-10)

    # Poisson deviance for fitted model
    dev_model = 2.0 * np.sum(
        np.where(
            y_true > 0,
            y_true * np.log(y_true / y_pred_safe) - (y_true - y_pred_safe),
            y_pred_safe
        )
    )

    # Poisson deviance for null model (intercept only = mean)
    dev_null = 2.0 * np.sum(
        np.where(
            y_true > 0,
            y_true * np.log(y_true / y_bar_safe) - (y_true - y_bar_safe),
            y_bar_safe
        )
    )

    if dev_null == 0:
        return 0.0
    return 1.0 - (dev_model / dev_null)


def correlation_r2(y_true, y_pred):
    """
    Correlation-based R2: cor(pred, obs)^2
    Also reported in R code's cvstats.
    """
    if len(y_true) < 2:
        return np.nan
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return corr ** 2


# Sklearn scorer for hyperparameter search (neg because sklearn maximizes)
neg_rmse_scorer = make_scorer(rmse_score, greater_is_better=False)


# ============================================================================
# DEMO: Show SMAPE vs MAPE difference
# ============================================================================

def demonstrate_metric_difference():
    """Show how SMAPE and MAPE differ on example pedestrian count data."""
    print("\n" + "=" * 60)
    print("DEMO: SMAPE vs MAPE on example predictions")
    print("=" * 60)

    y_true = np.array([5, 10, 50, 100, 200, 500])
    y_pred = np.array([8, 15, 40, 120, 180, 450])

    mape_val = standard_mape(y_true, y_pred)
    smape_val = smape_score(y_true, y_pred)

    print(f"\n  obs:  {y_true}")
    print(f"  pred: {y_pred}")
    print(f"\n  Standard MAPE: {mape_val:.2f}%")
    print(f"  SMAPE (R CV):  {smape_val:.2f}%")
    print(f"  Difference:    {mape_val - smape_val:.2f}% (MAPE is always higher)")
    print(f"\n  -> When comparing to R code CV results, use SMAPE column.")


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

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

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
    """L1 (Lasso) based feature selection."""
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
    """F-statistic based feature selection."""
    f_scores, _ = f_regression(X_transformed, y)
    f_scores = np.nan_to_num(f_scores)
    top_idx = np.argsort(f_scores)[::-1][:n_features]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names


# ============================================================================
# MODEL DEFINITIONS WITH HYPERPARAMETER GRIDS
# ============================================================================

def get_model_configs():
    """Returns model configs with hyperparameter grids for tuning."""
    configs = {}

    # --- Random Forest ---
    configs["RandomForest"] = {
        "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
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
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [300, 500, 800],
            "max_depth": [5, 7, 10],
            "min_samples_leaf": [10, 20],
            "l2_regularization": [0.0, 1.0],
        },
    }

    # --- HistGradientBoosting (Squared Error) ---
    configs["HistGB_SquaredError"] = {
        "model": HistGradientBoostingRegressor(
            loss="squared_error", random_state=RANDOM_STATE
        ),
        "param_grid": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [300, 500, 800],
            "max_depth": [5, 7, 10],
            "min_samples_leaf": [10, 20],
            "l2_regularization": [0.0, 1.0, 10.0],
        },
    }

    # --- ExtraTrees ---
    configs["ExtraTrees"] = {
        "model": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 15, 25],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.5, 0.8],
        },
    }

    # --- GradientBoosting (Huber loss for robustness) ---
    configs["GradientBoosting_Huber"] = {
        "model": GradientBoostingRegressor(
            loss="huber", random_state=RANDOM_STATE
        ),
        "param_grid": {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 500],
            "max_depth": [3, 5, 7],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [3, 5, 10],
            "subsample": [0.8, 1.0],
            "alpha": [0.8, 0.9],
        },
    }

    # --- Poisson Regression (regularized GLM) ---
    configs["PoissonRegressor"] = {
        "model": PoissonRegressor(max_iter=5000),
        "param_grid": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        },
    }

    # --- Bagging + HistGB ---
    configs["Bagging_HistGB"] = {
        "model": BaggingRegressor(
            estimator=HistGradientBoostingRegressor(
                loss="poisson", random_state=RANDOM_STATE,
                max_iter=500, learning_rate=0.05
            ),
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "param_grid": {
            "n_estimators": [5, 10, 15],
            "max_samples": [0.7, 0.8, 1.0],
            "max_features": [0.7, 0.8, 1.0],
        },
    }

    return configs


# ============================================================================
# CV EXPERIMENT RUNNER (reports ALL metrics)
# ============================================================================

def run_cv_experiment(X_selected, y, model, cv_strategy, model_name=""):
    """
    Run cross-validation and report ALL metrics:
      RMSE, standard MAPE, SMAPE, McFadden R2, Correlation R2
    """
    rmse_list = []
    mape_list = []
    smape_list = []
    mf_r2_list = []
    corr_r2_list = []

    for train_idx, val_idx in cv_strategy.split(X_selected):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        m = clone(model)
        m.fit(X_train, y_train)

        y_pred_train = np.maximum(m.predict(X_train), 0)
        y_pred_val = np.maximum(m.predict(X_val), 0)

        rmse_list.append(rmse_score(y_val, y_pred_val))
        mape_list.append(standard_mape(y_val, y_pred_val))
        smape_list.append(smape_score(y_val, y_pred_val))
        mf_r2_list.append(mcfadden_r2_deviance(y_train, y_pred_train))
        corr_r2_list.append(correlation_r2(y_val, y_pred_val))

    return {
        "model": model_name,
        "R2_train_mean": np.mean(mf_r2_list),
        "R2_train_std": np.std(mf_r2_list),
        "Corr_R2_val_mean": np.mean(corr_r2_list),
        "Corr_R2_val_std": np.std(corr_r2_list),
        "MAPE_val_mean": np.mean(mape_list),
        "MAPE_val_std": np.std(mape_list),
        "SMAPE_val_mean": np.mean(smape_list),
        "SMAPE_val_std": np.std(smape_list),
        "RMSE_val_mean": np.mean(rmse_list),
        "RMSE_val_std": np.std(rmse_list),
    }


# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def _count_grid_combos(param_grid):
    """Count total combinations in a parameter grid."""
    total = 1
    for key, values in param_grid.items():
        total *= len(values)
    return total


def run_hyperparameter_search(
    X_selected, y, model, param_grid, cv_strategy, model_name="",
    search_method="random", n_iter=50
):
    """Run hyperparameter search and return best model + results."""
    n_combos = _count_grid_combos(param_grid)

    if search_method == "grid" or n_combos <= n_iter:
        search = GridSearchCV(
            model, param_grid, cv=cv_strategy,
            scoring=neg_rmse_scorer, n_jobs=-1,
            refit=True, verbose=0
        )
    else:
        search = RandomizedSearchCV(
            model, param_grid, cv=cv_strategy,
            scoring=neg_rmse_scorer, n_jobs=-1,
            n_iter=n_iter, refit=True,
            random_state=RANDOM_STATE, verbose=0
        )

    search.fit(X_selected, y)

    best_model = search.best_estimator_
    best_params = search.best_params_

    # Compute all metrics with best model via manual CV
    cv_results = run_cv_experiment(
        X_selected, y, best_model, cv_strategy, model_name
    )
    cv_results["best_params"] = str(best_params)

    return cv_results, best_model, best_params


# ============================================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("PEDESTRIAN COUNT ESTIMATION")
    print("Full Experiments with Corrected Metrics (SMAPE + Deviance R2)")
    print("=" * 80)

    # Show SMAPE vs MAPE demo
    demonstrate_metric_difference()

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

    # Use ALL 4 feature selection methods (20 features each) as baseline
    exp1_feature_methods = {
        "L1_Lasso": select_features_l1,
        "Mutual_Information": select_features_mi,
        "Random_Forest": select_features_rf,
        "F_Regression": select_features_f_regression,
    }

    baseline_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGB_Poisson": HistGradientBoostingRegressor(
            loss="poisson", max_iter=500, random_state=RANDOM_STATE
        ),
    }

    cv_comparison_results = []
    for fs_name, fs_func in exp1_feature_methods.items():
        print(f"\n  Selecting top 20 features with {fs_name}...")
        try:
            feat_idx, feat_names_sel = fs_func(
                X_transformed, y, feature_names, n_features=20
            )
            X_fs_20 = X_transformed[:, feat_idx]
        except Exception as e:
            print(f"    ERROR in {fs_name}: {e}")
            continue

        for cv_name, cv_strat in cv_strategies.items():
            for model_name, model in baseline_models.items():
                print(f"    {fs_name} | {model_name} | {cv_name}...", end=" ")
                res = run_cv_experiment(
                    X_fs_20, y, model, cv_strat, model_name=model_name
                )
                res["cv_strategy"] = cv_name
                res["feature_selection"] = fs_name
                res["n_features"] = 20
                cv_comparison_results.append(res)
                print(
                    f"RMSE={res['RMSE_val_mean']:.2f}  "
                    f"MAPE={res['MAPE_val_mean']:.2f}  "
                    f"SMAPE={res['SMAPE_val_mean']:.2f}"
                )

    # Keep L1 features for later experiments
    l1_idx_20, l1_names_20 = select_features_l1(
        X_transformed, y, feature_names, n_features=20
    )
    X_l1_20 = X_transformed[:, l1_idx_20]

    df_cv_comp = pd.DataFrame(cv_comparison_results)
    cv_comp_path = os.path.join(OUTPUT_DIR, "experiment1_cv_strategy_comparison.csv")
    df_cv_comp.to_csv(cv_comp_path, index=False)
    print(f"\nSaved: {cv_comp_path}")

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

    cv_10fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    feature_sweep_results = []
    for n_feat in feature_counts:
        for fs_name, fs_func in feature_methods.items():
            print(f"\n  {fs_name}, n_features={n_feat}")
            try:
                feat_idx, feat_names = fs_func(
                    X_transformed, y, feature_names, n_features=n_feat
                )
                X_selected = X_transformed[:, feat_idx]
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            for model_name, model in baseline_models.items():
                print(f"    {model_name}...", end=" ")
                res = run_cv_experiment(
                    X_selected, y, model, cv_10fold, model_name=model_name
                )
                res["feature_selection"] = fs_name
                res["n_features"] = n_feat
                res["cv_strategy"] = "10-Fold"
                feature_sweep_results.append(res)
                print(
                    f"RMSE={res['RMSE_val_mean']:.2f}  "
                    f"MAPE={res['MAPE_val_mean']:.2f}  "
                    f"SMAPE={res['SMAPE_val_mean']:.2f}"
                )

    df_feat_sweep = pd.DataFrame(feature_sweep_results)
    feat_sweep_path = os.path.join(OUTPUT_DIR, "experiment2_feature_count_sweep.csv")
    df_feat_sweep.to_csv(feat_sweep_path, index=False)
    print(f"\nSaved: {feat_sweep_path}")

    # Print best from sweep
    if len(feature_sweep_results) > 0:
        best_rmse_row = df_feat_sweep.loc[df_feat_sweep["RMSE_val_mean"].idxmin()]
        best_smape_row = df_feat_sweep.loc[df_feat_sweep["SMAPE_val_mean"].idxmin()]
        print(f"\n  Best RMSE: {best_rmse_row['RMSE_val_mean']:.2f} "
              f"({best_rmse_row['model']}, {best_rmse_row['feature_selection']}, "
              f"n={best_rmse_row['n_features']})")
        print(f"  Best SMAPE: {best_smape_row['SMAPE_val_mean']:.2f} "
              f"({best_smape_row['model']}, {best_smape_row['feature_selection']}, "
              f"n={best_smape_row['n_features']})")

    # ========================================================================
    # EXPERIMENT 3: Hyperparameter Tuning
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Hyperparameter Tuning with RandomizedSearchCV")
    print("=" * 80)

    # ---- Use ALL 4 CV strategies (like Exp 1) ----
    tuning_cv_strategies = {
        "5-Fold": KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        "10-Fold": KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
        "RepeatedKFold_5x10": RepeatedKFold(
            n_splits=10, n_repeats=5, random_state=RANDOM_STATE
        ),
        "RepeatedKFold_3x10": RepeatedKFold(
            n_splits=10, n_repeats=3, random_state=RANDOM_STATE
        ),
    }

    # ---- Use ALL 5 feature counts (like Exp 2) ----
    tuning_feature_counts = [10, 15, 20, 25, 30]

    # Build feature configs: L1 Lasso with all 5 feature counts
    tune_feature_configs = []
    for n_feat in tuning_feature_counts:
        tune_feature_configs.append(("L1_Lasso", select_features_l1, n_feat))

    # Add best from feature sweep if it uses a different method
    if len(feature_sweep_results) > 0:
        best_fs = best_rmse_row["feature_selection"]
        best_n = int(best_rmse_row["n_features"])
        fs_map = {
            "L1_Lasso": select_features_l1,
            "Mutual_Information": select_features_mi,
            "Random_Forest": select_features_rf,
            "F_Regression": select_features_f_regression,
        }
        if best_fs in fs_map and best_fs != "L1_Lasso":
            tune_feature_configs.append((best_fs, fs_map[best_fs], best_n))
            print(f"  Added best from Exp 2: {best_fs} with {best_n} features")

    model_configs = get_model_configs()
    tuning_results = []

    total_runs = len(tune_feature_configs) * len(model_configs) * len(tuning_cv_strategies)
    print(f"\n  Total tuning runs: {len(tune_feature_configs)} feature configs × "
          f"{len(model_configs)} models × {len(tuning_cv_strategies)} CV strategies "
          f"= {total_runs}")

    run_counter = 0
    for fs_name, fs_func, n_feat in tune_feature_configs:
        print(f"\n--- Feature Selection: {fs_name}, n_features={n_feat} ---")
        feat_idx, feat_names = fs_func(
            X_transformed, y, feature_names, n_features=n_feat
        )
        X_selected = X_transformed[:, feat_idx]

        for cv_name, cv_strat in tuning_cv_strategies.items():
            print(f"\n  CV Strategy: {cv_name}")

            for model_name, config in model_configs.items():
                grid = config.get("param_grid", {})
                if not grid:
                    continue

                n_combos = _count_grid_combos(grid)
                n_iter = min(40, n_combos)
                run_counter += 1

                print(f"    [{run_counter}/{total_runs}] {model_name} "
                      f"({n_combos} combos, testing {n_iter})...",
                      end=" ", flush=True)

                try:
                    res, best_model, best_params = run_hyperparameter_search(
                        X_selected, y, config["model"], grid,
                        cv_strat, model_name=model_name,
                        search_method="random", n_iter=n_iter
                    )
                    res["feature_selection"] = fs_name
                    res["n_features"] = n_feat
                    res["cv_strategy"] = cv_name
                    tuning_results.append(res)
                    print(
                        f"RMSE={res['RMSE_val_mean']:.2f}  "
                        f"MAPE={res['MAPE_val_mean']:.2f}  "
                        f"SMAPE={res['SMAPE_val_mean']:.2f}"
                    )
                    print(f"      Best params: {best_params}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

    df_tuning = pd.DataFrame(tuning_results)
    tuning_path = os.path.join(OUTPUT_DIR, "experiment3_hyperparameter_tuning.csv")
    df_tuning.to_csv(tuning_path, index=False)
    print(f"\nSaved: {tuning_path}")

    # ========================================================================
    # EXPERIMENT 4: Log-Transform Target Variable
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Log-Transform Target Variable")
    print("Training on log(y+1), back-transforming predictions")
    print("=" * 80)

    y_log = np.log1p(y)
    cv_10fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # Recompute L1 features for this experiment
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

    log_transform_results = []
    for model_name, model in log_models.items():
        rmse_list = []
        mape_list = []
        smape_list = []

        for train_idx, val_idx in cv_10fold.split(X_l1):
            X_train, X_val = X_l1[train_idx], X_l1[val_idx]
            y_train_log, y_val = y_log[train_idx], y[val_idx]

            m = clone(model)
            m.fit(X_train, y_train_log)
            y_pred_log = m.predict(X_val)
            y_pred = np.expm1(y_pred_log)  # back-transform
            y_pred = np.maximum(y_pred, 0)

            rmse_list.append(rmse_score(y_val, y_pred))
            mape_list.append(standard_mape(y_val, y_pred))
            smape_list.append(smape_score(y_val, y_pred))

        res = {
            "model": model_name,
            "R2_train_mean": np.nan,  # not directly comparable for log target
            "R2_train_std": np.nan,
            "Corr_R2_val_mean": np.nan,
            "Corr_R2_val_std": np.nan,
            "MAPE_val_mean": np.mean(mape_list),
            "MAPE_val_std": np.std(mape_list),
            "SMAPE_val_mean": np.mean(smape_list),
            "SMAPE_val_std": np.std(smape_list),
            "RMSE_val_mean": np.mean(rmse_list),
            "RMSE_val_std": np.std(rmse_list),
            "n_features": 20,
            "feature_selection": "L1_Lasso",
            "cv_strategy": "10-Fold",
            "target_transform": "log1p",
        }
        log_transform_results.append(res)
        print(
            f"  {model_name}: "
            f"RMSE={res['RMSE_val_mean']:.2f}  "
            f"MAPE={res['MAPE_val_mean']:.2f}  "
            f"SMAPE={res['SMAPE_val_mean']:.2f}"
        )

    df_log = pd.DataFrame(log_transform_results)
    log_path = os.path.join(OUTPUT_DIR, "experiment4_log_transform.csv")
    df_log.to_csv(log_path, index=False)
    print(f"\nSaved: {log_path}")

    # ========================================================================
    # EXPERIMENT 5: Stacking Ensembles
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Stacking Ensembles")
    print("=" * 80)

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
    cv_10fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

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
        res["cv_strategy"] = "10-Fold"
        stacking_results.append(res)
        print(
            f"RMSE={res['RMSE_val_mean']:.2f}  "
            f"MAPE={res['MAPE_val_mean']:.2f}  "
            f"SMAPE={res['SMAPE_val_mean']:.2f}"
        )

    df_stack = pd.DataFrame(stacking_results)
    stack_path = os.path.join(OUTPUT_DIR, "experiment5_stacking.csv")
    df_stack.to_csv(stack_path, index=False)
    print(f"\nSaved: {stack_path}")

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

    # Display columns
    display_cols = [
        "experiment", "model", "RMSE_val_mean", "RMSE_val_std",
        "MAPE_val_mean", "SMAPE_val_mean"
    ]
    available_cols = [c for c in display_cols if c in df_all.columns]

    print("\nTop 10 by RMSE (lower is better):")
    print(df_all.nsmallest(10, "RMSE_val_mean")[available_cols].to_string(index=False))

    print("\nTop 10 by Standard MAPE (lower is better):")
    print(df_all.nsmallest(10, "MAPE_val_mean")[available_cols].to_string(index=False))

    print("\nTop 10 by SMAPE (comparable to R code CV results, lower is better):")
    print(df_all.nsmallest(10, "SMAPE_val_mean")[available_cols].to_string(index=False))

    # MAPE vs SMAPE comparison
    print("\n" + "-" * 60)
    print("MAPE vs SMAPE gap (averaged across all experiments):")
    avg_mape = df_all["MAPE_val_mean"].mean()
    avg_smape = df_all["SMAPE_val_mean"].mean()
    print(f"  Average MAPE:  {avg_mape:.2f}%")
    print(f"  Average SMAPE: {avg_smape:.2f}%")
    print(f"  SMAPE is on average {avg_mape - avg_smape:.2f}% lower than MAPE")
    print("-" * 60)

    # Baselines
    print("\nYOUR CURRENT BEST BASELINES (from presentations):")
    print("  L1 + RandomForest (20 feat, 5-fold CV): RMSE=79.84, MAPE=50.48")
    print("  L1 + HistGB_Poisson (18 feat, holdout): RMSE=133.60, MAPE=45.11")
    print("  R-code HistGB (10 feat, 5-fold CV):     RMSE=89.76, MAPE=49.75")
    print("-" * 60)

    overall_best_rmse = df_all.loc[df_all["RMSE_val_mean"].idxmin()]
    overall_best_mape = df_all.loc[df_all["MAPE_val_mean"].idxmin()]
    overall_best_smape = df_all.loc[df_all["SMAPE_val_mean"].idxmin()]

    print(f"\nBEST RMSE:  {overall_best_rmse['RMSE_val_mean']:.2f} "
          f"({overall_best_rmse['experiment']}, {overall_best_rmse['model']})")
    print(f"BEST MAPE:  {overall_best_mape['MAPE_val_mean']:.2f} "
          f"({overall_best_mape['experiment']}, {overall_best_mape['model']})")
    print(f"BEST SMAPE: {overall_best_smape['SMAPE_val_mean']:.2f} "
          f"({overall_best_smape['experiment']}, {overall_best_smape['model']})")

    # ========================================================================
    # FEATURE IMPORTANCE REPORT (for best results)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SELECTED FEATURES & IMPORTANCE (for best configurations)")
    print("=" * 80)

    # --- Helper: train model on full data and extract importance ---
    def get_feature_importance(model, X_data, y_data, feat_names_list):
        """
        Train model on full data and return feature importances ranked high to low.
        Works for tree-based models (feature_importances_) and linear models (coef_).
        For models without built-in importance, uses permutation importance.
        """
        from sklearn.inspection import permutation_importance

        m = clone(model)
        m.fit(X_data, y_data)

        # Try built-in feature_importances_ (tree models)
        if hasattr(m, "feature_importances_"):
            importance = m.feature_importances_
            method = "built-in (Gini/split importance)"
        # Try coef_ (linear models)
        elif hasattr(m, "coef_"):
            importance = np.abs(m.coef_)
            method = "absolute coefficients"
        # Fallback: permutation importance
        else:
            perm = permutation_importance(
                m, X_data, y_data, n_repeats=10,
                random_state=RANDOM_STATE, scoring=neg_rmse_scorer
            )
            importance = perm.importances_mean
            method = "permutation importance"

        # Sort high to low
        sorted_idx = np.argsort(importance)[::-1]
        df_imp = pd.DataFrame({
            "rank": range(1, len(feat_names_list) + 1),
            "feature": [feat_names_list[i] for i in sorted_idx],
            "importance": importance[sorted_idx],
        })

        return df_imp, method

    # --- Report for each best configuration ---
    best_configs = [
        {
            "label": "Best RMSE (HP_Tuning, HistGB_Poisson)",
            "fs_func": select_features_l1,
            "n_features": 20,  # from HP_Tuning which used L1
            "model": HistGradientBoostingRegressor(
                loss="poisson", random_state=RANDOM_STATE,
                # Use tuned params if available from experiment 3
                max_iter=800, learning_rate=0.05, max_depth=7,
                min_samples_leaf=10, l2_regularization=1.0,
            ),
        },
        {
            "label": "Best SMAPE (CV_Strategy, HistGB_Poisson)",
            "fs_func": select_features_l1,
            "n_features": 20,
            "model": HistGradientBoostingRegressor(
                loss="poisson", max_iter=500, random_state=RANDOM_STATE,
            ),
        },
        {
            "label": "Best MAPE (HP_Tuning, HistGB_Poisson)",
            "fs_func": select_features_l1,
            "n_features": 15,  # HP_Tuning tested 15, 20, 25
            "model": HistGradientBoostingRegressor(
                loss="poisson", random_state=RANDOM_STATE,
                max_iter=800, learning_rate=0.05, max_depth=7,
                min_samples_leaf=10, l2_regularization=1.0,
            ),
        },
    ]

    # Also try to pull actual best params from tuning results if available
    if len(tuning_results) > 0:
        # Find best RMSE config from HP_Tuning
        df_tune = pd.DataFrame(tuning_results)
        best_tune_row = df_tune.loc[df_tune["RMSE_val_mean"].idxmin()]
        if "best_params" in best_tune_row and "n_features" in best_tune_row:
            print(f"\n  Note: Best HP_Tuning used n_features="
                  f"{best_tune_row.get('n_features', '?')}")
            print(f"  Best params: {best_tune_row.get('best_params', '?')}")

    all_importance_dfs = []

    for config in best_configs:
        print(f"\n--- {config['label']} ---")
        print(f"  Feature selection: L1 Lasso, n_features={config['n_features']}")

        # Select features
        feat_idx, feat_names_selected = config["fs_func"](
            X_transformed, y, feature_names, n_features=config["n_features"]
        )
        X_selected = X_transformed[:, feat_idx]

        # Get importance
        df_imp, imp_method = get_feature_importance(
            config["model"], X_selected, y, feat_names_selected
        )

        print(f"  Importance method: {imp_method}")
        print(f"\n  Features ranked by importance (high to low):")
        print(f"  {'Rank':<6}{'Feature':<40}{'Importance':<12}")
        print(f"  {'-'*56}")
        for _, row in df_imp.iterrows():
            print(f"  {int(row['rank']):<6}{row['feature']:<40}{row['importance']:.6f}")

        # Save to CSV
        df_imp["config"] = config["label"]
        all_importance_dfs.append(df_imp)

    # Save all importances to one CSV
    if all_importance_dfs:
        df_all_imp = pd.concat(all_importance_dfs, ignore_index=True)
        imp_path = os.path.join(OUTPUT_DIR, "feature_importance_ranked.csv")
        df_all_imp.to_csv(imp_path, index=False)
        print(f"\nFeature importances saved to: {imp_path}")

    # --- Also show which features are shared across best configs ---
    print("\n" + "-" * 60)
    print("FEATURES COMMON ACROSS ALL BEST CONFIGS:")
    print("-" * 60)

    feature_sets = []
    for config in best_configs:
        feat_idx, feat_names_selected = config["fs_func"](
            X_transformed, y, feature_names, n_features=config["n_features"]
        )
        feature_sets.append(set(feat_names_selected))

    common_features = feature_sets[0]
    for s in feature_sets[1:]:
        common_features = common_features.intersection(s)

    print(f"  {len(common_features)} features appear in ALL best configurations:")
    for f in sorted(common_features):
        print(f"    - {f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()