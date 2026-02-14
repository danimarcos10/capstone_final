"""
Phase 5+ -- Calibration Upgrade
=================================
Additive upgrade to Phase 5. Compares calibration methods (None / Platt / Isotonic)
on a shared base estimator trained on train_proper (70% of train). Survey weights
are passed to calibration fit. Full-training models included as reference.

Usage:
    python run_calibration_upgrade.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import cfg, set_global_seed, get_output_dir
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import prepare_modeling_data, create_train_test_split
from src.modeling import (
    fit_weighted_logistic,
    fit_gradient_boosting,
    calibrate_model_prefit,
)
from src.evaluation import (
    compute_ece,
    compute_adaptive_ece,
    brier_decomposition,
    calibration_slope_intercept,
    weighted_roc_auc,
    weighted_brier_score,
    weighted_average_precision,
    weighted_balanced_accuracy,
)

set_global_seed()
SEED = cfg["modeling"]["random_seed"]

print("=" * 70)
print("PHASE 5+ -- Calibration Upgrade (apples-to-apples)")
print("=" * 70)

# --- Load + prepare data ----------------------------------------------------
print("\n--- Loading & preparing data ---")
df_raw, meta = load_atp_w119()
df_raw["y_apply"] = create_target_variable(df_raw)

X, y, weights, feat_names = prepare_modeling_data(
    df_raw, feature_set="full",
    missingness_regime="impute_indicator",
    not_sure_treatment="own_category",
    apply_encoding=True, apply_one_hot=True, drop_first=True,
)

X_train, X_test, y_train, y_test, w_train, w_test = create_train_test_split(
    X, y, weights,
)
print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

y_test_arr = np.asarray(y_test)
w_test_arr = np.asarray(w_test)

tables_dir = get_output_dir("tables")

# --- Train_proper / calibration split ---------------------------------------
# 70/30 split so calibrator is fitted on held-out data
print("\n--- Train_proper / calibration split ---")
X_train_p, X_cal, y_train_p, y_cal, w_train_p, w_cal = train_test_split(
    X_train, y_train, w_train,
    test_size=0.3, random_state=SEED, stratify=y_train,
)
print(f"Train-proper: {len(y_train_p):,}  |  Cal: {len(y_cal):,}  |  Test: {len(y_test):,}")


# --- Fit base estimators ----------------------------------------------------
print("\n--- Fitting base estimators (train_proper) ---")

lr_base = fit_weighted_logistic(X_train_p, y_train_p, w_train_p, C=1.0)

gbm_base = fit_gradient_boosting(
    X_train_p, y_train_p, w_train_p,
    max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=50,
)

print("--- Fitting calibrated variants (with survey weights) ---")

lr_platt = calibrate_model_prefit(lr_base, X_cal, y_cal,
                                   method="sigmoid",
                                   sample_weight=np.asarray(w_cal))

lr_isotonic = calibrate_model_prefit(lr_base, X_cal, y_cal,
                                      method="isotonic",
                                      sample_weight=np.asarray(w_cal))

gbm_platt = calibrate_model_prefit(gbm_base, X_cal, y_cal,
                                    method="sigmoid",
                                    sample_weight=np.asarray(w_cal))

gbm_isotonic = calibrate_model_prefit(gbm_base, X_cal, y_cal,
                                       method="isotonic",
                                       sample_weight=np.asarray(w_cal))

print("--- Fitting full-training reference models ---")

lr_full = fit_weighted_logistic(X_train, y_train, w_train, C=1.0)
lr_full_bal = fit_weighted_logistic(X_train, y_train, w_train, C=1.0,
                                     class_weight="balanced")

gbm_full = fit_gradient_boosting(
    X_train, y_train, w_train,
    max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=50,
)

print("  All model variants fitted.")


# --- Evaluate on held-out test set ------------------------------------------
print("\n--- Evaluating on held-out test set ---")


def _eval_row(model, model_name, cal_method, block):
    """Evaluate one model variant and return a dict."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_w   = weighted_roc_auc(y_test_arr, y_prob, w_test_arr)
    prauc_w = weighted_average_precision(y_test_arr, y_prob, w_test_arr)
    brier_w = weighted_brier_score(y_test_arr, y_prob, w_test_arr)
    ece_w   = compute_ece(y_test_arr, y_prob, n_bins=10, weights=w_test_arr)
    aece_w  = compute_adaptive_ece(y_test_arr, y_prob, n_bins=10,
                                    weights=w_test_arr)
    bal_w   = weighted_balanced_accuracy(y_test_arr, y_pred, w_test_arr)
    cal_si  = calibration_slope_intercept(y_test_arr, y_prob,
                                           weights=w_test_arr)
    bd      = brier_decomposition(y_test_arr, y_prob, n_bins=10,
                                   weights=w_test_arr)

    return {
        "Block": block,
        "Model": model_name,
        "Calibration Method": cal_method,
        "ROC-AUC": round(auc_w, 3),
        "PR-AUC": round(prauc_w, 3),
        "Brier": round(brier_w, 3),
        "ECE": round(ece_w, 3),
        "Adaptive ECE": round(aece_w, 3),
        "Cal. Slope": round(cal_si["slope"], 3),
        "Cal. Intercept": round(cal_si["intercept"], 3),
        "Balanced Acc.": round(bal_w, 3),
        "_uncertainty": bd["uncertainty"],
        "_reliability": bd["reliability"],
        "_resolution":  bd["resolution"],
    }


rows = []

# A2A block: same base estimator, different calibrators
rows.append(_eval_row(lr_base,     "LR",  "None",             "A2A"))
rows.append(_eval_row(lr_platt,    "LR",  "Platt (sigmoid)",  "A2A"))
rows.append(_eval_row(lr_isotonic, "LR",  "Isotonic",         "A2A"))
rows.append(_eval_row(gbm_base,    "GBM", "None",             "A2A"))
rows.append(_eval_row(gbm_platt,   "GBM", "Platt (sigmoid)",  "A2A"))
rows.append(_eval_row(gbm_isotonic,"GBM", "Isotonic",         "A2A"))

# Reference block: full-training models (not directly comparable to A2A)
rows.append(_eval_row(lr_full,     "LR (full train)",  "None", "Ref"))
rows.append(_eval_row(lr_full_bal, "LR (balanced)",    "None", "Ref"))
rows.append(_eval_row(gbm_full,    "GBM (full train)", "None", "Ref"))


print(f"\n  {'Block':>5s}  {'Model':<18s}  {'CalMethod':<17s}  "
      f"{'AUC':>5s}  {'Brier':>5s}  {'ECE':>5s}  {'AECE':>5s}  "
      f"{'Slope':>5s}  {'Int':>6s}")
for r in rows:
    print(f"  {r['Block']:>5s}  {r['Model']:<18s}  "
          f"{r['Calibration Method']:<17s}  "
          f"{r['ROC-AUC']:>5.3f}  {r['Brier']:>5.3f}  "
          f"{r['ECE']:>5.3f}  {r['Adaptive ECE']:>5.3f}  "
          f"{r['Cal. Slope']:>5.3f}  {r['Cal. Intercept']:>+6.3f}")


# --- Save calibration_comparison.csv ----------------------------------------
print("\n--- Saving calibration comparison table ---")

cal_df = pd.DataFrame(rows)
display_cols = [
    "Block", "Model", "Calibration Method", "ROC-AUC", "PR-AUC", "Brier",
    "ECE", "Adaptive ECE", "Cal. Slope", "Cal. Intercept", "Balanced Acc.",
]
cal_out = cal_df[display_cols]
cal_path = tables_dir / "calibration_comparison.csv"
cal_out.to_csv(cal_path, index=False)
print(f"  -> {cal_path}")


# --- Save brier_decomposition.csv -------------------------------------------
print("\n--- Saving Brier decomposition table ---")

brier_rows = []
for r in rows:
    brier_rows.append({
        "Block": r["Block"],
        "Model": r["Model"],
        "Calibration Method": r["Calibration Method"],
        "Brier Score": r["Brier"],
        "Uncertainty": round(r["_uncertainty"], 4),
        "Reliability": round(r["_reliability"], 4),
        "Resolution":  round(r["_resolution"], 4),
    })

brier_df = pd.DataFrame(brier_rows)
brier_path = tables_dir / "brier_decomposition.csv"
brier_df.to_csv(brier_path, index=False)
print(f"  -> {brier_path}")

print(f"\n  {'Block':>5s}  {'Model':<18s}  {'CalMethod':<17s}  "
      f"{'Brier':>6s}  {'Uncert':>7s}  {'Reliab':>7s}  {'Resol':>7s}")
for _, r in brier_df.iterrows():
    print(f"  {r['Block']:>5s}  {r['Model']:<18s}  "
          f"{r['Calibration Method']:<17s}  "
          f"{r['Brier Score']:>6.3f}  {r['Uncertainty']:>7.4f}  "
          f"{r['Reliability']:>7.4f}  {r['Resolution']:>7.4f}")


# --- Summary -----------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 5+ -- CALIBRATION UPGRADE COMPLETE")
print("=" * 70)
print(f"\n  A2A: base on train_proper (N={len(y_train_p):,}), only calibrator varies.")
print(f"  Ref: full training set (N={len(y_train):,}), not comparable to A2A.")
print(f"  Survey weights passed to CalibratedClassifierCV.fit().")
print(f"\nNew outputs:")
print(f"  - {cal_path}")
print(f"  - {brier_path}")
print(f"\nExisting Phase 5 outputs are UNTOUCHED.")
print()
