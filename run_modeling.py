"""
Phase 4 -- Modeling Pipeline (Baseline + Novel Contribution)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import cfg, SURVEY, set_global_seed, get_feature_set, get_output_dir
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import (
    handle_not_sure,
    handle_refused_as_missing,
    handle_skip_patterns,
    prepare_modeling_data,
    create_train_test_split,
)
from src.modeling import (
    fit_weighted_logistic,
    extract_odds_ratios,
    bootstrap_odds_ratios,
    fit_gradient_boosting,
    calibrate_model_prefit,
    evaluate_on_test,
    cv_evaluate,
    model_comparison_table,
    plot_odds_ratios,
    plot_calibration_comparison,
    plot_roc_comparison,
)
from src.evaluation import compute_ece
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

set_global_seed()
print("=" * 65)
print("PHASE 4 -- Modeling Pipeline")
print("=" * 65)

# ---- 1. Load and prepare data -------------------------------------------
print("\n--- Loading data ---")
df, meta = load_atp_w119()
df["y_apply"] = create_target_variable(df)

# impute_indicator regime; "Not sure" as own_category
print("\n--- Preparing modeling data (impute_indicator, own_category) ---")
X, y, weights, feature_names = prepare_modeling_data(
    df,
    feature_set="full",
    missingness_regime="impute_indicator",
    not_sure_treatment="own_category",
    apply_encoding=True,
    apply_one_hot=True,
    drop_first=True,  # avoid multicollinearity
)

print(f"Design matrix: {X.shape[0]} rows x {X.shape[1]} features")
print(f"Prevalence: {y.mean():.3f}")

# ---- 2. Train/test split -------------------------------------------------
print("\n--- Stratified train/test split (80/20) ---")
X_train, X_test, y_train, y_test, w_train, w_test = create_train_test_split(
    X, y, weights,
)
print(f"Train: {len(y_train):,} (prevalence {y_train.mean():.3f})")
print(f"Test:  {len(y_test):,} (prevalence {y_test.mean():.3f})")

figures_dir = get_output_dir("figures")
tables_dir = get_output_dir("tables")

all_test_results = []
calibration_data = []
roc_data = []


# =====================================================================
# 4.1  BASELINE: WEIGHTED LOGISTIC REGRESSION
# =====================================================================
print("\n" + "=" * 65)
print("4.1  BASELINE: Weighted Logistic Regression (L2)")
print("=" * 65)

# --- 4.1a  Default (no class_weight) ---
print("\n--- Logistic Regression (default) ---")
lr_default = fit_weighted_logistic(X_train, y_train, w_train, C=1.0, class_weight=None)
lr_default_metrics = evaluate_on_test(lr_default, X_test, y_test, w_test, model_name="LR (default)")
all_test_results.append(lr_default_metrics)

y_prob_lr = lr_default.predict_proba(X_test)[:, 1]
calibration_data.append({"model_name": "LR (default)", "y_true": y_test, "y_prob": y_prob_lr})
roc_data.append({"model_name": "LR (default)", "y_true": y_test, "y_prob": y_prob_lr})

print(f"  ROC-AUC (weighted): {lr_default_metrics['roc_auc_weighted']:.4f}")
print(f"  Brier (weighted):   {lr_default_metrics['brier_weighted']:.4f}")
print(f"  ECE (weighted):     {lr_default_metrics['ece_weighted']:.4f}")

# --- 4.1b  Balanced class weight ---
print("\n--- Logistic Regression (balanced) ---")
lr_balanced = fit_weighted_logistic(X_train, y_train, w_train, C=1.0, class_weight="balanced")
lr_balanced_metrics = evaluate_on_test(lr_balanced, X_test, y_test, w_test, model_name="LR (balanced)")
all_test_results.append(lr_balanced_metrics)

y_prob_lr_bal = lr_balanced.predict_proba(X_test)[:, 1]
calibration_data.append({"model_name": "LR (balanced)", "y_true": y_test, "y_prob": y_prob_lr_bal})
roc_data.append({"model_name": "LR (balanced)", "y_true": y_test, "y_prob": y_prob_lr_bal})

print(f"  ROC-AUC (weighted): {lr_balanced_metrics['roc_auc_weighted']:.4f}")
print(f"  Brier (weighted):   {lr_balanced_metrics['brier_weighted']:.4f}")
print(f"  ECE (weighted):     {lr_balanced_metrics['ece_weighted']:.4f}")

# --- 4.1c  Odds ratios (from default model) ---
print("\n--- Odds ratios (point estimates) ---")
or_df = extract_odds_ratios(lr_default, feature_names)
or_df.to_csv(tables_dir / "odds_ratios_point.csv", index=False)
print("  Top 10 features by |coefficient|:")
for _, row in or_df.head(10).iterrows():
    dir_str = "+" if row["coefficient"] > 0 else "-"
    print(f"    #{row['rank']:.0f}  {row['feature']:<35s}  OR={row['odds_ratio']:.3f}  ({dir_str})")

# --- 4.1d  Bootstrapped CIs for odds ratios ---
print("\n--- Bootstrapping odds ratio CIs (n=1000, may take ~60s) ---")
boot_or = bootstrap_odds_ratios(
    X_train, y_train, w_train,
    C=1.0, class_weight=None, n_bootstrap=1000,
)
boot_or.to_csv(tables_dir / "odds_ratios_bootstrap.csv", index=False)
print("  Top 10 features with 95% CIs:")
for _, row in boot_or.head(10).iterrows():
    sig = "*" if (row["or_ci_lower"] > 1.0 or row["or_ci_upper"] < 1.0) else ""
    print(f"    #{row['rank']:.0f}  {row['feature']:<35s}  "
          f"OR={row['or_mean']:.3f} [{row['or_ci_lower']:.3f}, {row['or_ci_upper']:.3f}]{sig}")

# Forest plot
plot_odds_ratios(boot_or, top_n=20, model_name="LR_default")
print("  -> Odds ratio forest plot saved.")


# =====================================================================
# 4.2  NOVEL: CALIBRATED GRADIENT BOOSTING
# =====================================================================
print("\n" + "=" * 65)
print("4.2  NOVEL: Calibrated Gradient Boosting")
print("=" * 65)

# --- Train/calibration split (70/30) for post-hoc calibration ---
seed = cfg["modeling"]["random_seed"]
X_train_p, X_cal, y_train_p, y_cal, w_train_p, w_cal = train_test_split(
    X_train, y_train, w_train,
    test_size=0.3, random_state=seed, stratify=y_train,
)
print(f"\n  Train/Cal/Test split: {len(y_train_p):,} / {len(y_cal):,} / {len(y_test):,}")

# --- 4.2a  Uncalibrated HistGBM (trained on full training set) ---
print("\n--- HistGradientBoosting (full training set, uncalibrated) ---")
gbm_full = fit_gradient_boosting(
    X_train, y_train, w_train,
    max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=50,
)
gbm_full_metrics = evaluate_on_test(gbm_full, X_test, y_test, w_test, model_name="GBM")
all_test_results.append(gbm_full_metrics)

y_prob_gbm_full = gbm_full.predict_proba(X_test)[:, 1]
calibration_data.append({"model_name": "GBM", "y_true": y_test, "y_prob": y_prob_gbm_full})
roc_data.append({"model_name": "GBM", "y_true": y_test, "y_prob": y_prob_gbm_full})

print(f"  ROC-AUC (weighted): {gbm_full_metrics['roc_auc_weighted']:.4f}")
print(f"  Brier (weighted):   {gbm_full_metrics['brier_weighted']:.4f}")
print(f"  ECE (weighted):     {gbm_full_metrics['ece_weighted']:.4f}")

# --- 4.2b  Proper calibration: train on train_proper, calibrate on cal, eval on test ---
print("\n--- HistGradientBoosting (train_proper) + isotonic calibration (cal set, cv=prefit) ---")
gbm_for_cal = fit_gradient_boosting(
    X_train_p, y_train_p, w_train_p,
    max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=50,
)

# Isotonic calibration on held-out set
gbm_cal = calibrate_model_prefit(gbm_for_cal, X_cal, y_cal, method="isotonic")
gbm_cal_metrics = evaluate_on_test(gbm_cal, X_test, y_test, w_test, model_name="GBM (calibrated)")
all_test_results.append(gbm_cal_metrics)

y_prob_gbm_cal = gbm_cal.predict_proba(X_test)[:, 1]
calibration_data.append({"model_name": "GBM (calibrated)", "y_true": y_test, "y_prob": y_prob_gbm_cal})
roc_data.append({"model_name": "GBM (calibrated)", "y_true": y_test, "y_prob": y_prob_gbm_cal})

# Uncalibrated baseline for fair comparison
gbm_for_cal_metrics = evaluate_on_test(gbm_for_cal, X_test, y_test, w_test, model_name="GBM (pre-cal)")

print(f"  Before calibration: AUC={gbm_for_cal_metrics['roc_auc_weighted']:.4f}  "
      f"Brier={gbm_for_cal_metrics['brier_weighted']:.4f}  ECE={gbm_for_cal_metrics['ece_weighted']:.4f}")
print(f"  After calibration:  AUC={gbm_cal_metrics['roc_auc_weighted']:.4f}  "
      f"Brier={gbm_cal_metrics['brier_weighted']:.4f}  ECE={gbm_cal_metrics['ece_weighted']:.4f}")

ece_before = gbm_for_cal_metrics["ece_weighted"]
ece_after = gbm_cal_metrics["ece_weighted"]
auc_delta = gbm_cal_metrics["roc_auc_weighted"] - gbm_for_cal_metrics["roc_auc_weighted"]
print(f"  AUC delta: {auc_delta:+.4f}")
print(f"  ECE delta: {ece_after - ece_before:+.4f}")
print(f"  Brier delta: {gbm_cal_metrics['brier_weighted'] - gbm_for_cal_metrics['brier_weighted']:+.4f}")


# =====================================================================
# 4.3  CROSS-VALIDATED COMPARISON
# =====================================================================
print("\n" + "=" * 65)
print("4.3  5-Fold Cross-Validation Comparison")
print("=" * 65)

models_for_cv = {
    "LR (default)": LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, solver="lbfgs",
        random_state=cfg["modeling"]["random_seed"],
    ),
    "LR (balanced)": LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced", max_iter=2000,
        solver="lbfgs", random_state=cfg["modeling"]["random_seed"],
    ),
    "GBM": HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=cfg["modeling"]["random_seed"],
    ),
}

cv_results = []
for name, model_template in models_for_cv.items():
    print(f"\n  CV: {name}...")
    cv_r = cv_evaluate(model_template, X, y, weights, n_folds=5, model_name=name)
    cv_results.append(cv_r)
    print(f"    ROC-AUC (weighted): {cv_r['roc_auc_weighted_mean']:.4f} +/- {cv_r['roc_auc_weighted_std']:.4f}")
    print(f"    Brier (weighted):   {cv_r['brier_weighted_mean']:.4f} +/- {cv_r['brier_weighted_std']:.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(tables_dir / "cv_comparison.csv", index=False)
print(f"\n  -> CV comparison saved to {tables_dir / 'cv_comparison.csv'}")


# =====================================================================
# 4.4  TEST-SET COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 65)
print("4.4  Test-Set Model Comparison")
print("=" * 65)

comparison_df = model_comparison_table(all_test_results)
comparison_df.to_csv(tables_dir / "model_comparison.csv", index=False)

print("\n  Model Comparison (test set):\n")
print(f"  {'Model':<25s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Brier':>9s} {'ECE':>9s} {'Bal Acc':>9s}")
print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
for _, row in comparison_df.iterrows():
    print(f"  {row['model']:<25s} "
          f"{row['roc_auc_weighted']:>9.4f} "
          f"{row['pr_auc_weighted']:>9.4f} "
          f"{row['brier_weighted']:>9.4f} "
          f"{row['ece_weighted']:>9.4f} "
          f"{row['balanced_acc_weighted']:>9.4f}")


# =====================================================================
# 4.5  FIGURES
# =====================================================================
print("\n" + "=" * 65)
print("4.5  Generating Figures")
print("=" * 65)

# Calibration comparison
plot_calibration_comparison(calibration_data, save_dir=figures_dir)
print("  -> calibration_comparison.png")

# ROC comparison
plot_roc_comparison(roc_data, save_dir=figures_dir)
print("  -> roc_comparison.png")


# =====================================================================
# 4.6  SAVE MODEL METRICS JSON
# =====================================================================
print("\n--- Saving model metrics JSON ---")

metrics_json = {}
for r in all_test_results:
    name = r["model"]
    metrics_json[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                          for k, v in r.items() if k != "model"}

json_path = get_output_dir("reports") / "model_metrics.json"
with open(json_path, "w") as f:
    json.dump(metrics_json, f, indent=2, default=str)
print(f"  -> {json_path}")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 65)
print("PHASE 4 -- COMPLETE")
print("=" * 65)

print(f"\nOutputs generated:")
print(f"  Tables:")
print(f"    - odds_ratios_point.csv")
print(f"    - odds_ratios_bootstrap.csv")
print(f"    - cv_comparison.csv")
print(f"    - model_comparison.csv")
print(f"  Figures:")
print(f"    - odds_ratios_LR_default.png")
print(f"    - calibration_comparison.png")
print(f"    - roc_comparison.png")
print(f"  Report:")
print(f"    - model_metrics.json")

# Key findings
print(f"\n--- KEY FINDINGS ---")
best_model = comparison_df.loc[comparison_df["roc_auc_weighted"].idxmax()]
print(f"  Best model (ROC-AUC): {best_model['model']} ({best_model['roc_auc_weighted']:.4f})")
print(f"  Calibration improvement (GBM): ECE {ece_before:.4f} -> {ece_after:.4f}")
best_or = boot_or.iloc[0]
print(f"  Strongest predictor (LR): {best_or['feature']} "
      f"(OR={best_or['or_mean']:.3f} [{best_or['or_ci_lower']:.3f}, {best_or['or_ci_upper']:.3f}])")
print()
