"""
Phase 5 -- Evaluation + Subgroup Diagnostics
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from src.config import cfg, SURVEY, set_global_seed, get_feature_set, get_feature_labels, get_output_dir
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import prepare_modeling_data, create_train_test_split
from src.evaluation import (
    compute_all_metrics, compute_ece, weighted_roc_auc,
    weighted_brier_score, weighted_average_precision,
    weighted_confusion_matrix, weighted_precision_recall_f1,
    weighted_balanced_accuracy,
    find_optimal_threshold_youden, find_optimal_threshold_f1,
)
from src.modeling import fit_weighted_logistic, fit_gradient_boosting

set_global_seed()
SEED = cfg["modeling"]["random_seed"]

print("=" * 70)
print("PHASE 5 -- Evaluation + Subgroup Diagnostics")
print("=" * 70)

# ---- 1. Load + prepare data -----------------------------------------------
print("\n--- Loading & preparing data ---")
df_raw, meta = load_atp_w119()
df_raw["y_apply"] = create_target_variable(df_raw)

X, y, weights, feat_names = prepare_modeling_data(
    df_raw, feature_set="full",
    missingness_regime="impute_indicator",
    not_sure_treatment="own_category",
    apply_encoding=True, apply_one_hot=True, drop_first=True,
)

X_train, X_test, y_train, y_test, w_train, w_test = create_train_test_split(X, y, weights)
print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

figures_dir = get_output_dir("figures")
tables_dir = get_output_dir("tables")
reports_dir = get_output_dir("reports")

# ---- 2. Fit models (same as Phase 4) --------------------------------------
print("\n--- Fitting models ---")
lr = fit_weighted_logistic(X_train, y_train, w_train, C=1.0)
gbm = fit_gradient_boosting(
    X_train, y_train, w_train,
    max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=50,
)

y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_gbm = gbm.predict_proba(X_test)[:, 1]

y_test_arr = np.asarray(y_test)
w_test_arr = np.asarray(w_test)


# =============================================================
# 5.1  COMPREHENSIVE METRICS WITH BOOTSTRAP CIs
# =============================================================
print("\n" + "=" * 70)
print("5.1  Comprehensive Metrics with Bootstrap CIs")
print("=" * 70)


def bootstrap_metrics(y_true, y_prob, w, n_boot=500, seed=42):
    """Bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    results = {"auc": [], "brier": [], "ece": [], "pr_auc": []}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp, ww = y_true[idx], y_prob[idx], w[idx]
        try:
            results["auc"].append(roc_auc_score(yt, yp, sample_weight=ww))
            results["brier"].append(weighted_brier_score(yt, yp, ww))
            results["ece"].append(compute_ece(yt, yp, weights=ww))
            results["pr_auc"].append(weighted_average_precision(yt, yp, ww))
        except ValueError:
            pass

    out = {}
    for k, vals in results.items():
        vals = np.array(vals)
        out[k] = {
            "mean": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }
    return out


print("\n  LR (default):")
lr_boot = bootstrap_metrics(y_test_arr, y_prob_lr, w_test_arr, n_boot=500)
for metric, vals in lr_boot.items():
    print(f"    {metric:>8s}: {vals['mean']:.4f}  [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]")

print("\n  GBM:")
gbm_boot = bootstrap_metrics(y_test_arr, y_prob_gbm, w_test_arr, n_boot=500)
for metric, vals in gbm_boot.items():
    print(f"    {metric:>8s}: {vals['mean']:.4f}  [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]")


# --- ECE sensitivity: multiple bin settings ---
print("\n--- ECE Sensitivity (bin settings) ---")
ece_sensitivity_rows = []
for model_name, y_prob in [("LR", y_prob_lr), ("GBM", y_prob_gbm)]:
    for n_bins in [10, 15, 20]:
        for strategy_name, strat_arg in [("uniform", "uniform"), ("quantile", "quantile")]:
            if strat_arg == "uniform":
                ece_val = compute_ece(y_test_arr, y_prob, n_bins=n_bins, weights=w_test_arr)
            else:
                from sklearn.calibration import calibration_curve as cal_crv
                try:
                    prob_true, prob_pred = cal_crv(y_test_arr, y_prob, n_bins=n_bins, strategy="quantile")
                    ece_val = float(np.mean(np.abs(prob_true - prob_pred)))
                except ValueError:
                    ece_val = np.nan

            ece_sensitivity_rows.append({
                "model": model_name, "n_bins": n_bins,
                "strategy": strategy_name, "ece": round(ece_val, 4),
            })

ece_sens_df = pd.DataFrame(ece_sensitivity_rows)
ece_sens_df.to_csv(tables_dir / "ece_sensitivity.csv", index=False)

print(f"  {'Model':>5s}  {'Bins':>4s}  {'Strategy':>8s}  {'ECE':>7s}")
for _, row in ece_sens_df.iterrows():
    print(f"  {row['model']:>5s}  {row['n_bins']:>4d}  {row['strategy']:>8s}  {row['ece']:>7.4f}")


# =============================================================
# 5.2  THRESHOLD POLICY COMPARISON
# =============================================================
print("\n" + "=" * 70)
print("5.2  Threshold Policy Comparison")
print("=" * 70)

prevalence = y_test_arr.mean()
thresh_youden, _ = find_optimal_threshold_youden(y_test_arr, y_prob_lr)
thresh_f1, _ = find_optimal_threshold_f1(y_test_arr, y_prob_lr)

thresholds = {
    "0.50 (default)": 0.50,
    f"prevalence ({prevalence:.3f})": prevalence,
    f"Youden J ({thresh_youden:.3f})": thresh_youden,
    f"max-F1 ({thresh_f1:.3f})": thresh_f1,
}

print("\n  --- LR (default) ---")
threshold_rows = []
for name, t in thresholds.items():
    y_pred = (y_prob_lr >= t).astype(int)
    m = compute_all_metrics(y_test_arr, y_prob_lr, y_pred, weights=w_test_arr)
    cm = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()

    row = {
        "threshold": name,
        "value": round(t, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision_w": round(m["precision_weighted"], 4),
        "recall_w": round(m["recall_weighted"], 4),
        "f1_w": round(m["f1_weighted"], 4),
        "balanced_acc_w": round(m["balanced_acc_weighted"], 4),
    }
    threshold_rows.append(row)
    print(f"  {name:<28s}  P={row['precision_w']:.3f}  R={row['recall_w']:.3f}  "
          f"F1={row['f1_w']:.3f}  BalAcc={row['balanced_acc_w']:.3f}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")

thresh_df = pd.DataFrame(threshold_rows)
thresh_df.to_csv(tables_dir / "threshold_comparison.csv", index=False)

# Recommend threshold
best_row = thresh_df.loc[thresh_df["balanced_acc_w"].idxmax()]
print(f"\n  Recommended: {best_row['threshold']} "
      f"(highest weighted balanced accuracy: {best_row['balanced_acc_w']:.3f})")


# =============================================================
# 5.3  CALIBRATION
# =============================================================
print("\n" + "=" * 70)
print("5.3  Calibration Analysis")
print("=" * 70)

# --- Reliability plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, y_prob) in zip(axes, [("LR (default)", y_prob_lr), ("GBM", y_prob_gbm)]):
    prob_true, prob_pred = calibration_curve(y_test_arr, y_prob, n_bins=10, strategy="uniform")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(prob_pred, prob_true, "o-", color="#4C72B0", label=name, markersize=6)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability Diagram: {name}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> reliability_diagrams.png")

# --- Calibration slope + intercept: logistic regression of Y ~ logit(p_hat) ---
from scipy.special import logit as sp_logit
from sklearn.linear_model import LogisticRegression as LR_cal

cal_stats = {}
for name, y_prob in [("LR", y_prob_lr), ("GBM", y_prob_gbm)]:
    # Clip to avoid logit(0) or logit(1)
    p_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit_p = sp_logit(p_clipped).reshape(-1, 1)

    cal_lr = LR_cal(penalty=None, max_iter=2000, solver="lbfgs")
    cal_lr.fit(logit_p, y_test_arr, sample_weight=w_test_arr)

    slope = float(cal_lr.coef_[0][0])
    intercept = float(cal_lr.intercept_[0])

    cal_stats[name] = {"slope": round(slope, 4), "intercept": round(intercept, 4)}
    print(f"  {name}: calibration slope = {slope:.4f}, intercept = {intercept:.4f}")
    if abs(slope - 1.0) < 0.15 and abs(intercept) < 0.15:
        print(f"    Well-calibrated (slope near 1, intercept near 0)")
    else:
        print(f"    Calibration imperfect (ideal: slope=1, intercept=0)")


# --- Calibration by subgroup ---
print("\n--- Calibration by subgroup ---")

valid_target = df_raw[df_raw["y_apply"].notna()].index
test_idx = y_test.index  # these are indices into the original df

subgroup_demo_vars = {
    "F_AGECAT": "age_category",
    "F_GENDER": "gender",
    "F_EDUCCAT2": "education",
    "F_RACETHNMOD": "race_ethnicity",
}

cal_by_subgroup_rows = []

for raw_var, readable in subgroup_demo_vars.items():
    subgroup_vals = df_raw.loc[test_idx, raw_var]

    for level in sorted(subgroup_vals.dropna().unique()):
        if level == SURVEY.REFUSED_CODE:
            continue
        mask = (subgroup_vals == level).values
        n = int(mask.sum())
        if n < 50:
            cal_by_subgroup_rows.append({
                "variable": readable, "level": level, "n": n,
                "ece_lr": None, "ece_gbm": None, "note": "N<50, insufficient",
            })
            continue

        ece_lr_sub = compute_ece(y_test_arr[mask], y_prob_lr[mask], weights=w_test_arr[mask])
        ece_gbm_sub = compute_ece(y_test_arr[mask], y_prob_gbm[mask], weights=w_test_arr[mask])

        label = str(meta.variable_value_labels.get(raw_var, {}).get(level, level))[:25]
        cal_by_subgroup_rows.append({
            "variable": readable, "level": level, "label": label, "n": n,
            "ece_lr": round(ece_lr_sub, 4), "ece_gbm": round(ece_gbm_sub, 4), "note": "",
        })

cal_sub_df = pd.DataFrame(cal_by_subgroup_rows)
cal_sub_df.to_csv(tables_dir / "calibration_by_subgroup.csv", index=False)

print("  ECE by subgroup:")
for _, row in cal_sub_df.iterrows():
    lbl = str(row.get("label", row["level"]))[:25]
    if row["note"]:
        print(f"    {row['variable']:>15s} = {lbl:>25s}  N={row['n']:>5d}  {row['note']}")
    else:
        print(f"    {row['variable']:>15s} = {lbl:>25s}  N={row['n']:>5d}  "
              f"ECE_LR={row['ece_lr']:.4f}  ECE_GBM={row['ece_gbm']:.4f}")


# =============================================================
# 5.4  SUBGROUP DIAGNOSTICS
# =============================================================
print("\n" + "=" * 70)
print("5.4  Subgroup Diagnostics (LR default, threshold=0.5)")
print("=" * 70)

MIN_N = 50
y_pred_lr = (y_prob_lr >= 0.5).astype(int)

subgroup_results = []

for raw_var, readable in subgroup_demo_vars.items():
    subgroup_vals = df_raw.loc[test_idx, raw_var].values
    val_labels_map = meta.variable_value_labels.get(raw_var, {})

    for level in sorted(set(subgroup_vals[~np.isnan(subgroup_vals)])):
        if level == SURVEY.REFUSED_CODE:
            continue
        mask = subgroup_vals == level
        n = int(mask.sum())
        label = str(val_labels_map.get(level, level))[:30]

        if n < MIN_N:
            subgroup_results.append({
                "variable": readable, "level": level, "label": label,
                "n": n, "note": f"N<{MIN_N}, insufficient",
            })
            continue

        yt = y_test_arr[mask]
        yp = y_pred_lr[mask]
        yprob = y_prob_lr[mask]
        ww = w_test_arr[mask]

        # Weighted metrics
        cm = weighted_confusion_matrix(yt, yp, ww)
        tpr = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else np.nan
        fpr = cm[0, 1] / cm[0, :].sum() if cm[0, :].sum() > 0 else np.nan
        fnr = 1 - tpr if not np.isnan(tpr) else np.nan
        prec_w, rec_w, f1_w = weighted_precision_recall_f1(yt, yp, ww)
        prev = np.average(yt, weights=ww)

        subgroup_results.append({
            "variable": readable, "level": level, "label": label, "n": n,
            "prevalence": round(prev, 4),
            "tpr": round(tpr, 4) if not np.isnan(tpr) else None,
            "fpr": round(fpr, 4) if not np.isnan(fpr) else None,
            "fnr": round(fnr, 4) if not np.isnan(fnr) else None,
            "precision_w": round(prec_w, 4),
            "f1_w": round(f1_w, 4),
            "note": "",
        })

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv(tables_dir / "subgroup_diagnostics.csv", index=False)

print("\n  Subgroup diagnostics (LR, t=0.5):")
print(f"  {'Variable':>15s}  {'Label':>25s}  {'N':>5s}  {'Prev':>6s}  {'TPR':>6s}  {'FPR':>6s}  {'PPV':>6s}  {'F1':>6s}")
print(f"  {'-'*15}  {'-'*25}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
for _, row in subgroup_df.iterrows():
    if row.get("note"):
        print(f"  {row['variable']:>15s}  {row['label']:>25s}  {row['n']:>5d}  {row['note']}")
    else:
        print(f"  {row['variable']:>15s}  {row['label']:>25s}  {row['n']:>5d}  "
              f"{row['prevalence']:>6.3f}  {row['tpr']:>6.3f}  {row['fpr']:>6.3f}  "
              f"{row['precision_w']:>6.3f}  {row['f1_w']:>6.3f}")


# --- Bootstrap CIs for subgroup metrics ---
print("\n--- Bootstrap CIs for subgroup TPR (n=500) ---")


def bootstrap_subgroup_metric(yt, yp, ww, metric_fn, n_boot=500, seed=42):
    """Bootstrap CI for a single subgroup metric."""
    rng = np.random.RandomState(seed)
    n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            v = metric_fn(yt[idx], yp[idx], ww[idx])
            vals.append(v)
        except (ValueError, ZeroDivisionError):
            pass
    if len(vals) < 10:
        return None, None, None
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tpr_fn(yt, yp, ww):
    cm = weighted_confusion_matrix(yt, yp, ww)
    return cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else np.nan


boot_subgroup_rows = []
for raw_var, readable in subgroup_demo_vars.items():
    subgroup_vals = df_raw.loc[test_idx, raw_var].values
    val_labels_map = meta.variable_value_labels.get(raw_var, {})

    for level in sorted(set(subgroup_vals[~np.isnan(subgroup_vals)])):
        if level == SURVEY.REFUSED_CODE:
            continue
        mask = subgroup_vals == level
        n = int(mask.sum())
        label = str(val_labels_map.get(level, level))[:30]

        if n < MIN_N:
            continue

        yt = y_test_arr[mask]
        yp = y_pred_lr[mask]
        ww = w_test_arr[mask]

        mean_tpr, lo, hi = bootstrap_subgroup_metric(yt, yp, ww, tpr_fn, n_boot=500)
        boot_subgroup_rows.append({
            "variable": readable, "level": level, "label": label, "n": n,
            "tpr_mean": round(mean_tpr, 4) if mean_tpr else None,
            "tpr_ci_lower": round(lo, 4) if lo else None,
            "tpr_ci_upper": round(hi, 4) if hi else None,
        })
        if mean_tpr:
            print(f"    {readable:>15s} = {label:>25s}  TPR = {mean_tpr:.3f} [{lo:.3f}, {hi:.3f}]")

boot_sub_df = pd.DataFrame(boot_subgroup_rows)
boot_sub_df.to_csv(tables_dir / "subgroup_tpr_bootstrap.csv", index=False)


# =============================================================
# 5.5  SUBGROUP DIAGNOSTICS FIGURE
# =============================================================
print("\n--- Generating subgroup diagnostics figure ---")

valid_sub = subgroup_df[subgroup_df["tpr"].notna()].copy()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric, title in zip(axes, ["tpr", "fpr", "fnr"],
                               ["True Positive Rate (Sensitivity)", "False Positive Rate", "False Negative Rate"]):
    labels_plot = [f"{r['label']}\n({r['variable']})" for _, r in valid_sub.iterrows()]
    vals = valid_sub[metric].values

    colors = plt.cm.Set2(np.linspace(0, 1, len(vals)))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels_plot, fontsize=6)
    ax.set_xlabel(metric.upper())
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "subgroup_error_rates.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> subgroup_error_rates.png")


# =============================================================
# GENERATE EVALUATION REPORT
# =============================================================
print("\n--- Generating evaluation report ---")

report_lines = [
    "# Phase 5: Evaluation + Subgroup Diagnostics Report",
    "",
    "---",
    "",
    "## 1. Comprehensive Metrics (with 95% bootstrap CIs, N=500)",
    "",
    "### LR (default)",
    "",
    "| Metric | Estimate | 95% CI |",
    "|--------|----------|--------|",
]
for metric, vals in lr_boot.items():
    report_lines.append(f"| {metric} | {vals['mean']:.4f} | [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}] |")

report_lines += ["", "### GBM", "", "| Metric | Estimate | 95% CI |", "|--------|----------|--------|"]
for metric, vals in gbm_boot.items():
    report_lines.append(f"| {metric} | {vals['mean']:.4f} | [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}] |")

report_lines += [
    "",
    "---",
    "",
    "## 2. Threshold Policy Comparison (LR default)",
    "",
    "| Threshold | Precision | Recall | F1 | Balanced Acc | TP | FP | FN | TN |",
    "|-----------|-----------|--------|----|-----------  -|----|----|----|-----|",
]
for _, row in thresh_df.iterrows():
    report_lines.append(
        f"| {row['threshold']} | {row['precision_w']:.3f} | {row['recall_w']:.3f} | "
        f"{row['f1_w']:.3f} | {row['balanced_acc_w']:.3f} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} |"
    )

report_lines.append(f"\n**Recommended**: {best_row['threshold']} (highest balanced accuracy)")

report_lines += [
    "",
    "---",
    "",
    "## 3. Calibration",
    "",
    "### Calibration Slope + Intercept",
    "",
    "| Model | Slope | Intercept | Assessment |",
    "|-------|-------|-----------|------------|",
]
for name, stats in cal_stats.items():
    ok = "Well-calibrated" if abs(stats["slope"] - 1) < 0.15 and abs(stats["intercept"]) < 0.15 else "Room for improvement"
    report_lines.append(f"| {name} | {stats['slope']:.4f} | {stats['intercept']:.4f} | {ok} |")

report_lines += [
    "",
    "### ECE by Subgroup",
    "",
    "| Variable | Level | N | ECE (LR) | ECE (GBM) |",
    "|----------|-------|---|----------|-----------|",
]
for _, row in cal_sub_df.iterrows():
    if row.get("note"):
        report_lines.append(f"| {row['variable']} | {row.get('label', row['level'])} | {row['n']} | {row['note']} | |")
    else:
        report_lines.append(f"| {row['variable']} | {row.get('label', row['level'])} | {row['n']} | {row['ece_lr']:.4f} | {row['ece_gbm']:.4f} |")

report_lines += [
    "",
    "---",
    "",
    "## 4. Subgroup Diagnostics",
    "",
    f"Minimum-N rule: groups with N < {MIN_N} report 'insufficient data'.",
    "",
    "| Variable | Label | N | Prevalence | TPR | FPR | FNR |",
    "|----------|-------|---|-----------|-----|-----|-----|",
]
for _, row in subgroup_df.iterrows():
    if row.get("note"):
        report_lines.append(f"| {row['variable']} | {row['label']} | {row['n']} | {row['note']} | | | |")
    else:
        report_lines.append(
            f"| {row['variable']} | {row['label']} | {row['n']} | {row['prevalence']:.3f} | "
            f"{row['tpr']:.3f} | {row['fpr']:.3f} | {row['fnr']:.3f} |"
        )

report_lines += [
    "",
    "### Bootstrap CIs for Subgroup TPR",
    "",
    "| Variable | Label | N | TPR | 95% CI |",
    "|----------|-------|---|-----|--------|",
]
for _, row in boot_sub_df.iterrows():
    if row["tpr_mean"]:
        report_lines.append(
            f"| {row['variable']} | {row['label']} | {row['n']} | {row['tpr_mean']:.3f} | "
            f"[{row['tpr_ci_lower']:.3f}, {row['tpr_ci_upper']:.3f}] |"
        )

report_text = "\n".join(report_lines)
eval_report_path = reports_dir / "evaluation_report.md"
eval_report_path.write_text(report_text, encoding="utf-8")
print(f"  -> {eval_report_path}")


# =============================================================
# SUMMARY
# =============================================================
print("\n" + "=" * 70)
print("PHASE 5 -- COMPLETE")
print("=" * 70)
print(f"\nOutputs:")
print(f"  Tables: threshold_comparison.csv, calibration_by_subgroup.csv,")
print(f"          subgroup_diagnostics.csv, subgroup_tpr_bootstrap.csv")
print(f"  Figures: reliability_diagrams.png, subgroup_error_rates.png")
print(f"  Report: evaluation_report.md")
print()
