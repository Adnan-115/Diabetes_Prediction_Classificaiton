"""
Comprehensive In-Fold ADASYN + Optuna Feature Categorization Ablation & Double Meta-Learner Stacking Pipeline
Target: notebooks/abrar/
Hardware Acceleration: NVIDIA GeForce RTX 5060 Ti (CUDA 13.3) & Multi-Core CPU

Features & Rigor:
1. Leak-Free Nested Feature Selection: In-fold Mutual Information & RFE(LogisticRegression) on raw training folds.
2. 6 Feature Groups: Biological, Socioeconomic, Lifestyle, Combined-All, Hybrid MI, Hybrid RFE.
3. 5 Model Families: XGBoost (GPU CUDA), LightGBM (CPU subsample_freq=1), Random Forest, Logistic Regression, KNN (Tree-accelerated).
4. Multi-Criteria Statistical Evaluation: Paired Wilcoxon Signed-Rank, Cohen's dz Effect Size, Parsimony Efficiency Index.
5. Double Meta-Learner Stacking:
   - Design A: Homogeneous Stacking (5 diverse algorithms on winning feature group)
   - Design B: Heterogeneous Stacking (6 feature-group champions for representation diversity)
6. Nested 5-Fold CV Meta-Threshold Calibration: Zero meta-leakage calibration on meta-OOF probabilities.
7. Outer Full-Fit Rule for 20% Holdout Test Evaluation.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import ADASYN
import optuna

# Setup logging
log_file = Path("notebooks/abrar/optuna_run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w")
    ]
)
logger = logging.getLogger("Abrar_Feature_Ablation_Pipeline")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
RAW_DATA_PATH = Path("data/raw/diabetes_binary_health_indicators_BRFSS2015.csv")
TARGET = "diabetes_binary"
N_TRIALS = 60

# Domain boundaries for sanitation
CONTINUOUS_BOUNDS = {
    "bmi": (12, 98),
    "menthlth": (0, 30),
    "physhlth": (0, 30)
}
ORDINAL_COLS = {
    "genhlth": (1, 5),
    "age": (1, 13),
    "education": (1, 8),
    "income": (1, 8),
}
BINARY_COLS = [
    "highbp", "highchol", "cholcheck", "smoker", "stroke",
    "heartdiseaseorattack", "physactivity", "fruits", "veggies",
    "hvyalcoholconsump", "anyhealthcare", "nodocbccost", "diffwalk", "sex"
]

# Fixed Domain Groups
DOMAIN_GROUPS = {
    "biological": ["bmi", "highchol", "cholcheck", "highbp", "heartdiseaseorattack", "stroke", "age", "sex"],
    "socioeconomic": ["income", "education", "anyhealthcare", "nodocbccost"],
    "lifestyle": ["smoker", "physactivity", "fruits", "veggies", "hvyalcoholconsump", "menthlth", "physhlth", "diffwalk", "genhlth"]
}

def sanitize_resampled_data(X_resampled_df):
    """Post-ADASYN Discrete & Binary Sanitation Projection."""
    X_clean = X_resampled_df.copy()
    for col in BINARY_COLS:
        if col in X_clean.columns:
            X_clean[col] = (X_clean[col] >= 0.5).astype(int)
    for col, (min_val, max_val) in ORDINAL_COLS.items():
        if col in X_clean.columns:
            X_clean[col] = np.clip(np.round(X_clean[col]), min_val, max_val).astype(int)
    for col, (min_val, max_val) in CONTINUOUS_BOUNDS.items():
        if col in X_clean.columns:
            X_clean[col] = np.clip(X_clean[col], min_val, max_val)
    return X_clean

def prepare_data_and_nested_group_cache():
    """
    Loads raw data, splits 80/20 with stratification, and pre-computes
    the 6 feature groups across the 5 shared In-Fold ADASYN folds in memory.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: IN-FOLD NESTED DATA PREPARATION & SHARED FOLD CACHING")
    logger.info("=" * 70)
    
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df[TARGET]
    )
    
    assert len(set(train_df.index).intersection(set(holdout_df.index))) == 0, "Index overlap detected!"
    
    X_train = train_df.drop(columns=[TARGET]).reset_index(drop=True)
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    X_test = holdout_df.drop(columns=[TARGET]).reset_index(drop=True)
    y_test = holdout_df[TARGET].astype(int).reset_index(drop=True)
    all_features = list(X_train.columns)
    
    logger.info(f"Loaded Dataset: Train={X_train.shape}, Holdout={X_test.shape}")
    logger.info(f"Train Positive Rate: {y_train.mean():.4f}, Holdout Positive Rate: {y_test.mean():.4f}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    reference_splits = list(skf.split(X_train, y_train))
    
    # 1. Verify 5-fold disjoint partition integrity over full training set
    val_indices = [val_idx for _, val_idx in reference_splits]
    assert sum(len(v) for v in val_indices) == len(X_train), "Validation folds do not cover full training set!"
    assert len(set().union(*val_indices)) == len(X_train), "Validation folds are not mutually disjoint!"
    
    group_fold_cache = {
        "biological": [],
        "socioeconomic": [],
        "lifestyle": [],
        "combined_all": [],
        "hybrid_mi": [],
        "hybrid_rfe": []
    }
    
    nested_mi_features_by_fold = []
    nested_rfe_features_by_fold = []
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(reference_splits):
        f_num = fold + 1
        X_tr_raw = X_train.iloc[tr_idx].copy()
        y_tr_raw = y_train.iloc[tr_idx].copy()
        X_val_raw = X_train.iloc[val_idx].copy()
        y_val_raw = y_train.iloc[val_idx].copy()
        
        # 1. Scale on Train Fold
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        
        # 2. ADASYN on scaled Train Fold
        adasyn = ADASYN(random_state=SEED, n_neighbors=5)
        X_tr_res, y_tr_res = adasyn.fit_resample(X_tr_scaled, y_tr_raw)
        
        # 3. Discrete Projection Sanitation
        X_tr_res_df = pd.DataFrame(scaler.inverse_transform(X_tr_res), columns=all_features)
        X_tr_clean = sanitize_resampled_data(X_tr_res_df)
        
        # 4. Nested Feature Selection inside fold (Raw train data only)
        # MI
        mi_scores = mutual_info_classif(X_tr_raw, y_tr_raw, random_state=SEED)
        mi_top10 = list(pd.Series(mi_scores, index=all_features).sort_values(ascending=False).index[:10])
        nested_mi_features_by_fold.append(mi_top10)
        
        # RFE with Logistic Regression
        rfe_scaler = StandardScaler()
        rfe_clf = LogisticRegression(max_iter=500, random_state=SEED, solver="lbfgs")
        rfe = RFE(rfe_clf, n_features_to_select=10)
        rfe.fit(rfe_scaler.fit_transform(X_tr_raw), y_tr_raw)
        rfe_top10 = list(np.array(all_features)[rfe.support_])
        nested_rfe_features_by_fold.append(rfe_top10)
        
        # Build cached fold arrays for all 6 feature groups
        group_definitions = {
            "biological": DOMAIN_GROUPS["biological"],
            "socioeconomic": DOMAIN_GROUPS["socioeconomic"],
            "lifestyle": DOMAIN_GROUPS["lifestyle"],
            "combined_all": all_features,
            "hybrid_mi": mi_top10,
            "hybrid_rfe": rfe_top10
        }
        
        for g_name, g_cols in group_definitions.items():
            X_tr_sub = X_tr_clean[g_cols].values
            X_val_sub = X_val_raw[g_cols].values
            
            # Standardized versions for LR & KNN
            g_scaler = StandardScaler()
            X_tr_sub_scaled = g_scaler.fit_transform(X_tr_sub)
            X_val_sub_scaled = g_scaler.transform(X_val_sub)
            
            group_fold_cache[g_name].append({
                "fold": f_num,
                "tr_idx": tr_idx,
                "val_idx": val_idx,
                "feature_cols": g_cols,
                "X_tr_clean": np.ascontiguousarray(X_tr_sub, dtype=np.float32),
                "y_tr_res": np.ascontiguousarray(y_tr_res.values, dtype=np.int32),
                "X_val": np.ascontiguousarray(X_val_sub, dtype=np.float32),
                "y_val": np.ascontiguousarray(y_val_raw.values, dtype=np.int32),
                "X_tr_scaled": np.ascontiguousarray(X_tr_sub_scaled, dtype=np.float32),
                "X_val_scaled": np.ascontiguousarray(X_val_sub_scaled, dtype=np.float32),
            })
            
        logger.info(f"  Fold {f_num}/5 processed: MI Top10={mi_top10[:3]}... | RFE Top10={rfe_top10[:3]}...")
        
    # Programmatic assertion for shared fold coherence across all 6 groups
    for g_name, f_list in group_fold_cache.items():
        for k in range(5):
            assert np.array_equal(f_list[k]["val_idx"], reference_splits[k][1]), f"Fold {k} split desynchronized for {g_name}!"
            
    logger.info(f"All 6 Feature Groups cached across 5 folds in RAM in {time.time() - t0:.2f}s!")
    return X_train, y_train, X_test, y_test, group_fold_cache, nested_mi_features_by_fold, nested_rfe_features_by_fold, all_features

def evaluate_model_cv(model_name, params, fold_cache, trial=None):
    """Evaluates candidate parameters across 5 cached folds with Optuna early pruning."""
    fold_pr_aucs = []
    
    for step, f in enumerate(fold_cache):
        if model_name == "xgboost":
            clf = xgb.XGBClassifier(
                **params,
                tree_method="hist",
                device="cuda",
                scale_pos_weight=1.0,
                random_state=SEED
            )
            clf.fit(f["X_tr_clean"], f["y_tr_res"])
            val_probs = clf.predict_proba(f["X_val"])[:, 1]
            
        elif model_name == "lightgbm":
            clf = lgb.LGBMClassifier(
                **params,
                subsample_freq=1,
                scale_pos_weight=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1
            )
            clf.fit(f["X_tr_clean"], f["y_tr_res"])
            val_probs = clf.predict_proba(f["X_val"])[:, 1]
            
        elif model_name == "logistic_regression":
            clf = LogisticRegression(
                **params,
                max_iter=500,
                random_state=SEED,
                n_jobs=-1
            )
            clf.fit(f["X_tr_scaled"], f["y_tr_res"])
            val_probs = clf.predict_proba(f["X_val_scaled"])[:, 1]
            
        elif model_name == "random_forest":
            clf = RandomForestClassifier(
                **params,
                random_state=SEED,
                n_jobs=-1
            )
            clf.fit(f["X_tr_clean"], f["y_tr_res"])
            val_probs = clf.predict_proba(f["X_val"])[:, 1]
            
        elif model_name == "knn":
            clf = KNeighborsClassifier(
                **params,
                algorithm="auto",
                n_jobs=-1
            )
            clf.fit(f["X_tr_scaled"], f["y_tr_res"])
            val_probs = clf.predict_proba(f["X_val_scaled"])[:, 1]
            
        score = average_precision_score(f["y_val"], val_probs)
        fold_pr_aucs.append(score)
        
        if trial is not None:
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
    return np.mean(fold_pr_aucs), fold_pr_aucs

def compute_oof_predictions(model_name, best_params, fold_cache, n_samples):
    """Generates Out-of-Fold (OOF) validation probability predictions for meta-stacking."""
    oof_probs = np.zeros(n_samples, dtype=np.float32)
    per_fold_scores = []
    
    for f in fold_cache:
        val_idx = f["val_idx"]
        if model_name == "xgboost":
            clf = xgb.XGBClassifier(**best_params, tree_method="hist", device="cuda", scale_pos_weight=1.0, random_state=SEED)
            clf.fit(f["X_tr_clean"], f["y_tr_res"])
            probs = clf.predict_proba(f["X_val"])[:, 1]
        elif model_name == "lightgbm":
            clf = lgb.LGBMClassifier(**best_params, subsample_freq=1, scale_pos_weight=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
            clf.fit(f["X_tr_clean"], f["y_tr_res"])
            probs = clf.predict_proba(f["X_val"])[:, 1]
        elif model_name == "logistic_regression":
            clf = LogisticRegression(**best_params, max_iter=500, random_state=SEED, n_jobs=-1)
            clf.fit(f["X_tr_scaled"], f["y_tr_res"])
            probs = clf.predict_proba(f["X_val_scaled"])[:, 1]
        elif model_name == "random_forest":
            clf = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)
            clf.fit(f["X_tr_clean"], f["y_tr_res"])
            probs = clf.predict_proba(f["X_val"])[:, 1]
        elif model_name == "knn":
            clf = KNeighborsClassifier(**best_params, algorithm="auto", n_jobs=-1)
            clf.fit(f["X_tr_scaled"], f["y_tr_res"])
            probs = clf.predict_proba(f["X_val_scaled"])[:, 1]
            
        oof_probs[val_idx] = probs
        per_fold_scores.append(average_precision_score(f["y_val"], probs))
        
    return oof_probs, per_fold_scores

def calibrate_threshold_recursive(y_true, y_probs, init_low=0.05, init_high=0.60, steps=56):
    """Finds optimal decision threshold tau* to maximize F1-score with recursive auto-widening."""
    low, high = init_low, init_high
    for iteration in range(5):
        thresholds = np.linspace(low, high, steps)
        f1_scores = [f1_score(y_true, (y_probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        best_tau = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        if best_idx == 0 and low > 0.01:
            low = max(0.01, low - 0.05)
            continue
        elif best_idx == len(thresholds) - 1 and high < 0.90:
            high = min(0.95, high + 0.05)
            continue
        else:
            return float(best_tau), float(best_f1)
            
    return float(best_tau), float(best_f1)

def run_feature_group_ablation_and_stacking():
    """Main execution orchestrator."""
    X_train, y_train, X_test, y_test, group_fold_cache, mi_folds, rfe_folds, all_features = prepare_data_and_nested_group_cache()
    n_train_samples = len(X_train)
    
    logger.info("=" * 70)
    logger.info(f"PHASE 2: 6 FEATURE GROUPS x 5 MODEL FAMILIES OPTUNA TUNING ({N_TRIALS} Trials/Model)")
    logger.info("=" * 70)
    
    # Store all optimization results
    results = {}
    oof_predictions = {}
    fold_scores_map = {}
    
    feature_group_names = ["biological", "socioeconomic", "lifestyle", "combined_all", "hybrid_mi", "hybrid_rfe"]
    model_names = ["xgboost", "lightgbm", "logistic_regression", "random_forest", "knn"]
    
    for g_idx, group in enumerate(feature_group_names, 1):
        logger.info("-" * 60)
        logger.info(f"[{g_idx}/6] TUNING FEATURE GROUP: {group.upper()} (Cols: {len(group_fold_cache[group][0]['feature_cols'])})")
        logger.info("-" * 60)
        
        results[group] = {}
        oof_predictions[group] = {}
        fold_scores_map[group] = {}
        f_cache = group_fold_cache[group]
        
        for m_idx, model in enumerate(model_names, 1):
            logger.info(f"  --> [{m_idx}/5] Optimizing {model.upper()} ({group})...")
            
            def objective(trial):
                if model == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    }
                elif model == "lightgbm":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    }
                elif model == "logistic_regression":
                    params = {
                        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                    }
                elif model == "random_forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                        "max_depth": trial.suggest_int("max_depth", 5, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7]),
                    }
                elif model == "knn":
                    params = {
                        "n_neighbors": trial.suggest_int("n_neighbors", 5, 75, step=5),
                        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                        "p": trial.suggest_categorical("p", [1, 2]),
                    }
                mean_pr_auc, _ = evaluate_model_cv(model, params, f_cache, trial)
                return mean_pr_auc
                
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
                sampler=optuna.samplers.TPESampler(seed=SEED)
            )
            study.optimize(objective, n_trials=N_TRIALS)
            
            best_params = study.best_params
            best_val_pr_auc = study.best_value
            
            # Compute final OOF predictions with best params
            oof_p, fold_scores = compute_oof_predictions(model, best_params, f_cache, n_train_samples)
            oof_predictions[group][model] = oof_p
            fold_scores_map[group][model] = fold_scores
            
            results[group][model] = {
                "mean_cv_pr_auc": float(best_val_pr_auc),
                "fold_pr_aucs": [float(s) for s in fold_scores],
                "cv_pr_auc_std": float(np.std(fold_scores)),
                "best_params": best_params
            }
            logger.info(f"    -> {model.upper()} ({group}) Complete | Mean 5-Fold PR-AUC: {best_val_pr_auc:.4f} (+/- {np.std(fold_scores):.4f})")
            
    logger.info("=" * 70)
    logger.info("PHASE 3: MULTI-CRITERIA STATISTICAL COMPARISON & GROUP RANKING")
    logger.info("=" * 70)
    
    # Identify champion model per group
    group_champions = {}
    for group in feature_group_names:
        best_m = max(results[group].keys(), key=lambda m: results[group][m]["mean_cv_pr_auc"])
        group_champions[group] = {
            "model_name": best_m,
            "mean_pr_auc": results[group][best_m]["mean_cv_pr_auc"],
            "fold_scores": results[group][best_m]["fold_pr_aucs"],
            "std": results[group][best_m]["cv_pr_auc_std"],
            "num_features": len(group_fold_cache[group][0]["feature_cols"])
        }
        
    # Statistical paired tests vs Combined All
    combined_scores = group_champions["combined_all"]["fold_scores"]
    stat_comparisons = {}
    
    for group, data in group_champions.items():
        diff = np.array(data["fold_scores"]) - np.array(combined_scores)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 1e-6
        cohens_dz = mean_diff / std_diff if std_diff > 0 else 0.0
        
        # Paired Wilcoxon Signed-Rank Test
        try:
            w_stat, p_val = stats.wilcoxon(data["fold_scores"], combined_scores)
        except Exception:
            w_stat, p_val = 0.0, 1.0
            
        d_count = data["num_features"]
        parsimony_ratio = data["mean_pr_auc"] / np.sqrt(d_count / 21.0)
        
        stat_comparisons[group] = {
            "best_model": data["model_name"],
            "feature_count": d_count,
            "mean_pr_auc": float(data["mean_pr_auc"]),
            "std_pr_auc": float(data["std"]),
            "mean_diff_vs_combined": float(mean_diff),
            "cohens_dz": float(cohens_dz),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p_val": float(p_val),
            "parsimony_efficiency_index": float(parsimony_ratio)
        }
        logger.info(f"Group: {group:<15} | Best: {data['model_name']:<12} | PR-AUC: {data['mean_pr_auc']:.4f} (+/- {data['std']:.4f}) | Feats: {d_count:>2} | Cohen's dz: {cohens_dz:+.2f} | p-val: {p_val:.4f} | Parsimony Index: {parsimony_ratio:.4f}")
        
    # Select winning feature group
    sorted_groups = sorted(group_champions.keys(), key=lambda g: group_champions[g]["mean_pr_auc"], reverse=True)
    winner_group = sorted_groups[0]
    logger.info(f"\nFeature Group Champion: {winner_group.upper()} (PR-AUC = {group_champions[winner_group]['mean_pr_auc']:.4f})")
    
    logger.info("=" * 70)
    logger.info("PHASE 4: DOUBLE META-LEARNER STACKING (DESIGN A vs. DESIGN B)")
    logger.info("=" * 70)
    
    # Build Stacking Matrices
    # Design A: 5 models on winning feature group
    Z_A = np.column_stack([oof_predictions[winner_group][m] for m in model_names])
    
    # Design B: 6 feature-group champions
    Z_B = np.column_stack([oof_predictions[g][group_champions[g]["model_name"]] for g in feature_group_names])
    
    logger.info(f"Design A Stacking Matrix Shape: {Z_A.shape} (5 Base Algorithms on {winner_group})")
    logger.info(f"Design B Stacking Matrix Shape: {Z_B.shape} (6 Feature-Group Champions)")
    
    # Nested 5-Fold CV Threshold Calibration for Meta-Learners
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    def evaluate_meta_learner(Z_matrix, y_target):
        meta_oof_probs = np.zeros(len(y_target), dtype=np.float32)
        for tr_idx, val_idx in skf_meta.split(Z_matrix, y_target):
            meta_clf = LogisticRegression(max_iter=500, random_state=SEED, solver="lbfgs")
            meta_clf.fit(Z_matrix[tr_idx], y_target.iloc[tr_idx])
            meta_oof_probs[val_idx] = meta_clf.predict_proba(Z_matrix[val_idx])[:, 1]
            
        tau_star, f1_star = calibrate_threshold_recursive(y_target, meta_oof_probs)
        meta_pr_auc = average_precision_score(y_target, meta_oof_probs)
        meta_roc_auc = roc_auc_score(y_target, meta_oof_probs)
        
        # Fit final meta-learner on full Z
        final_meta = LogisticRegression(max_iter=500, random_state=SEED, solver="lbfgs")
        final_meta.fit(Z_matrix, y_target)
        
        return final_meta, meta_oof_probs, tau_star, f1_star, meta_pr_auc, meta_roc_auc
        
    meta_A, oof_meta_A, tau_A, f1_A, pr_auc_A, roc_auc_A = evaluate_meta_learner(Z_A, y_train)
    meta_B, oof_meta_B, tau_B, f1_B, pr_auc_B, roc_auc_B = evaluate_meta_learner(Z_B, y_train)
    
    logger.info(f"Meta-Learner Design A (Homogeneous)   | OOF PR-AUC: {pr_auc_A:.4f} | tau*: {tau_A:.2f} | OOF F1: {f1_A:.4f}")
    logger.info(f"Meta-Learner Design B (Heterogeneous) | OOF PR-AUC: {pr_auc_B:.4f} | tau*: {tau_B:.2f} | OOF F1: {f1_B:.4f}")
    logger.info(f"Design B Meta-Coefficients (Domain Importance Weights): {dict(zip(feature_group_names, np.round(meta_B.coef_[0], 4)))}")
    
    logger.info("=" * 70)
    logger.info("PHASE 5: OUTER FULL-FIT & FINAL 20% HOLDOUT EVALUATION (N=50,736)")
    logger.info("=" * 70)
    
    # Outer Full-Train Resampling
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    adasyn_full = ADASYN(random_state=SEED, n_neighbors=5)
    X_train_res_scaled, y_train_res = adasyn_full.fit_resample(X_train_scaled, y_train)
    X_train_res_clean = sanitize_resampled_data(pd.DataFrame(scaler_full.inverse_transform(X_train_res_scaled), columns=all_features))
    
    # Outer Feature Selection for Hybrid Groups on full 80% train partition
    mi_scores_outer = mutual_info_classif(X_train, y_train, random_state=SEED)
    mi_top10_outer = list(pd.Series(mi_scores_outer, index=all_features).sort_values(ascending=False).index[:10])
    
    rfe_clf_outer = LogisticRegression(max_iter=500, random_state=SEED, solver="lbfgs")
    rfe_outer = RFE(rfe_clf_outer, n_features_to_select=10)
    rfe_outer.fit(StandardScaler().fit_transform(X_train), y_train)
    rfe_top10_outer = list(np.array(all_features)[rfe_outer.support_])
    
    outer_group_cols = {
        "biological": DOMAIN_GROUPS["biological"],
        "socioeconomic": DOMAIN_GROUPS["socioeconomic"],
        "lifestyle": DOMAIN_GROUPS["lifestyle"],
        "combined_all": all_features,
        "hybrid_mi": mi_top10_outer,
        "hybrid_rfe": rfe_top10_outer
    }
    
    # Retrain Single Champion Model
    champ_g = winner_group
    champ_m = group_champions[champ_g]["model_name"]
    champ_params = results[champ_g][champ_m]["best_params"]
    champ_cols = outer_group_cols[champ_g]
    
    X_tr_champ = X_train_res_clean[champ_cols].values
    X_te_champ = X_test[champ_cols].values
    
    if champ_m == "xgboost":
        single_model = xgb.XGBClassifier(**champ_params, tree_method="hist", device="cuda", random_state=SEED)
        single_model.fit(X_tr_champ, y_train_res)
        y_test_probs_single = single_model.predict_proba(X_te_champ)[:, 1]
    elif champ_m == "lightgbm":
        single_model = lgb.LGBMClassifier(**champ_params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
        single_model.fit(X_tr_champ, y_train_res)
        y_test_probs_single = single_model.predict_proba(X_te_champ)[:, 1]
        
    # Calibrate single model tau*
    tau_star_single, _ = calibrate_threshold_recursive(y_train, oof_predictions[champ_g][champ_m])
    
    # Retrain Base Models for Design A Test Matrix
    Z_test_A_list = []
    for m in model_names:
        cols = outer_group_cols[winner_group]
        X_tr_m = X_train_res_clean[cols].values
        X_te_m = X_test[cols].values
        p = results[winner_group][m]["best_params"]
        
        if m == "xgboost":
            clf = xgb.XGBClassifier(**p, tree_method="hist", device="cuda", random_state=SEED).fit(X_tr_m, y_train_res)
            Z_test_A_list.append(clf.predict_proba(X_te_m)[:, 1])
        elif m == "lightgbm":
            clf = lgb.LGBMClassifier(**p, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_tr_m, y_train_res)
            Z_test_A_list.append(clf.predict_proba(X_te_m)[:, 1])
        elif m == "logistic_regression":
            s = StandardScaler()
            clf = LogisticRegression(**p, max_iter=500, random_state=SEED, n_jobs=-1).fit(s.fit_transform(X_tr_m), y_train_res)
            Z_test_A_list.append(clf.predict_proba(s.transform(X_te_m))[:, 1])
        elif m == "random_forest":
            clf = RandomForestClassifier(**p, random_state=SEED, n_jobs=-1).fit(X_tr_m, y_train_res)
            Z_test_A_list.append(clf.predict_proba(X_te_m)[:, 1])
        elif m == "knn":
            s = StandardScaler()
            clf = KNeighborsClassifier(**p, algorithm="auto", n_jobs=-1).fit(s.fit_transform(X_tr_m), y_train_res)
            Z_test_A_list.append(clf.predict_proba(s.transform(X_te_m))[:, 1])
            
    Z_test_A = np.column_stack(Z_test_A_list)
    y_test_probs_A = meta_A.predict_proba(Z_test_A)[:, 1]
    
    # Retrain Base Models for Design B Test Matrix
    Z_test_B_list = []
    for g in feature_group_names:
        m = group_champions[g]["model_name"]
        cols = outer_group_cols[g]
        X_tr_m = X_train_res_clean[cols].values
        X_te_m = X_test[cols].values
        p = results[g][m]["best_params"]
        
        if m == "xgboost":
            clf = xgb.XGBClassifier(**p, tree_method="hist", device="cuda", random_state=SEED).fit(X_tr_m, y_train_res)
            Z_test_B_list.append(clf.predict_proba(X_te_m)[:, 1])
        elif m == "lightgbm":
            clf = lgb.LGBMClassifier(**p, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_tr_m, y_train_res)
            Z_test_B_list.append(clf.predict_proba(X_te_m)[:, 1])
        elif m == "logistic_regression":
            s = StandardScaler()
            clf = LogisticRegression(**p, max_iter=500, random_state=SEED, n_jobs=-1).fit(s.fit_transform(X_tr_m), y_train_res)
            Z_test_B_list.append(clf.predict_proba(s.transform(X_te_m))[:, 1])
        elif m == "random_forest":
            clf = RandomForestClassifier(**p, random_state=SEED, n_jobs=-1).fit(X_tr_m, y_train_res)
            Z_test_B_list.append(clf.predict_proba(X_te_m)[:, 1])
        elif m == "knn":
            s = StandardScaler()
            clf = KNeighborsClassifier(**p, algorithm="auto", n_jobs=-1).fit(s.fit_transform(X_tr_m), y_train_res)
            Z_test_B_list.append(clf.predict_proba(s.transform(X_te_m))[:, 1])
            
    Z_test_B = np.column_stack(Z_test_B_list)
    y_test_probs_B = meta_B.predict_proba(Z_test_B)[:, 1]
    
    # Compute Final Evaluation Metrics
    def compute_full_eval_dict(name, y_probs, tau_star):
        y_pred_calib = (y_probs >= tau_star).astype(int)
        y_pred_def = (y_probs >= 0.50).astype(int)
        return {
            "name": name,
            "tau_star": float(tau_star),
            "holdout_pr_auc": float(average_precision_score(y_test, y_probs)),
            "holdout_roc_auc": float(roc_auc_score(y_test, y_probs)),
            "f1_calibrated": float(f1_score(y_test, y_pred_calib)),
            "recall_calibrated": float(recall_score(y_test, y_pred_calib)),
            "precision_calibrated": float(precision_score(y_test, y_pred_calib)),
            "f1_default": float(f1_score(y_test, y_pred_def)),
            "recall_default": float(recall_score(y_test, y_pred_def)),
            "precision_default": float(precision_score(y_test, y_pred_def)),
            "confusion_matrix_calibrated": confusion_matrix(y_test, y_pred_calib).tolist(),
            "confusion_matrix_default": confusion_matrix(y_test, y_pred_def).tolist(),
        }
        
    eval_single = compute_full_eval_dict(f"Single Champion ({champ_g.upper()} - {champ_m.upper()})", y_test_probs_single, tau_star_single)
    eval_meta_A = compute_full_eval_dict(f"Design A Meta-Learner (Homogeneous - {winner_group.upper()})", y_test_probs_A, tau_A)
    eval_meta_B = compute_full_eval_dict("Design B Meta-Learner (Heterogeneous Feature Diversity)", y_test_probs_B, tau_B)
    
    logger.info("-" * 60)
    logger.info("FINAL 20% HOLDOUT PERFORMANCE COMPARISON:")
    for ev in [eval_single, eval_meta_A, eval_meta_B]:
        logger.info(f"  {ev['name']:<50} | PR-AUC: {ev['holdout_pr_auc']:.4f} | ROC-AUC: {ev['holdout_roc_auc']:.4f} | F1 (tau*={ev['tau_star']:.2f}): {ev['f1_calibrated']:.4f} | Recall: {ev['recall_calibrated']:.4f}")
    logger.info("-" * 60)
    
    # Export full comprehensive artifact
    final_artifact = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grid_results": results,
        "statistical_comparisons": stat_comparisons,
        "group_champions": group_champions,
        "winning_feature_group": winner_group,
        "meta_learner_weights_design_b": dict(zip(feature_group_names, [float(c) for c in meta_B.coef_[0]])),
        "eval_single_champion": eval_single,
        "eval_meta_learner_design_a": eval_meta_A,
        "eval_meta_learner_design_b": eval_meta_B,
    }
    
    artifact_path = Path("notebooks/abrar/artifacts/feature_groups_evaluation.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w") as f:
        json.dump(final_artifact, f, indent=2)
        
    logger.info(f"Successfully exported final results to {artifact_path}!")
    return final_artifact

if __name__ == "__main__":
    run_feature_group_ablation_and_stacking()
