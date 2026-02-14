"""
Phase 6 -- Robustness Suite + Interpretability Stability
==========================================================
Usage:
    python run_robustness.py
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
from scipy import stats

from src.config import cfg, SURVEY, set_global_seed, get_feature_set, get_output_dir
from src.data_loading import load_atp_w119, create_target_variable, create_alternative_target
from src.preprocessing import prepare_modeling_data, create_train_test_split
from src.evaluation import weighted_roc_auc, weighted_brier_score, compute_ece

set_global_seed()
SEED = cfg["modeling"]["random_seed"]
tables_dir = get_output_dir("tables")
figures_dir = get_output_dir("figures")

print("=" * 70)
print("PHASE 6 -- Robustness Suite + Interpretability Stability")
print("=" * 70)

# --- Load data ---------------------------------------------------------------
print("\n--- Loading raw data ---")
df_raw, meta = load_atp_w119()
df_raw["y_apply"] = create_target_variable(df_raw)


def quick_lr_auc(X_tr, y_tr, w_tr, X_te, y_te, w_te):
    """Fit LR and return weighted test AUC."""
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
    lr.fit(X_tr, y_tr, sample_weight=w_tr)
    p = lr.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, p, sample_weight=w_te), lr


def quick_gbm_auc(X_tr, y_tr, w_tr, X_te, y_te, w_te):
    """Fit GBM and return weighted test AUC."""
    gbm = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=SEED,
    )
    gbm.fit(X_tr, y_tr, sample_weight=w_tr)
    p = gbm.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, p, sample_weight=w_te), gbm


# === 6.1a  Outcome Recoding Sensitivity =====================================
print("\n" + "=" * 70)
print("6.1a  Outcome Recoding: Refused=drop vs Refused=No")
print("=" * 70)

df_alt = df_raw.copy()
df_alt["y_apply_alt"] = create_alternative_target(df_alt)  # Refused -> No

outcome_rows = []
for target_name, target_col in [("Refused=drop", "y_apply"), ("Refused=No", "y_apply_alt")]:
    X, y, w, fn = prepare_modeling_data(
        df_alt, feature_set="full",
        missingness_regime="impute_indicator", not_sure_treatment="own_category",
        apply_encoding=True, apply_one_hot=True, drop_first=True,
    )
    if target_col == "y_apply_alt":
        y = df_alt.loc[X.index, target_col]
        y = y[y.notna()].astype(int)
        common = X.index.intersection(y.index)
        X, y, w = X.loc[common], y.loc[common], w[X.index.isin(common)]

    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y,
    )
    auc_lr, _ = quick_lr_auc(Xtr, ytr, wtr, Xte, yte, wte)
    auc_gbm, _ = quick_gbm_auc(Xtr, ytr, wtr, Xte, yte, wte)
    prev = float(y.mean())
    outcome_rows.append({
        "outcome": target_name, "N": len(y), "prevalence": round(prev, 4),
        "auc_lr": round(auc_lr, 4), "auc_gbm": round(auc_gbm, 4),
    })
    print(f"  {target_name}: N={len(y):,}  prev={prev:.3f}  LR AUC={auc_lr:.4f}  GBM AUC={auc_gbm:.4f}")

outcome_df = pd.DataFrame(outcome_rows)
outcome_df.to_csv(tables_dir / "robustness_outcome.csv", index=False)


# === 6.1b  "Not Sure" Sensitivity ===========================================
print("\n" + "=" * 70)
print("6.1b  'Not Sure' Handling: drop vs own_category vs midpoint")
print("=" * 70)

ns_rows = []
for treatment in ["drop", "own_category", "midpoint"]:
    X, y, w, fn = prepare_modeling_data(
        df_raw, feature_set="full",
        missingness_regime="impute_indicator", not_sure_treatment=treatment,
        apply_encoding=True, apply_one_hot=True, drop_first=True,
    )
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y,
    )
    auc_lr, lr = quick_lr_auc(Xtr, ytr, wtr, Xte, yte, wte)
    auc_gbm, _ = quick_gbm_auc(Xtr, ytr, wtr, Xte, yte, wte)

    coef_df = pd.DataFrame({"feature": fn, "abs_coef": np.abs(lr.coef_[0])})
    top5 = coef_df.nlargest(5, "abs_coef")["feature"].tolist()

    ns_rows.append({
        "treatment": treatment, "N": len(y), "n_features": len(fn),
        "auc_lr": round(auc_lr, 4), "auc_gbm": round(auc_gbm, 4),
        "top5": ", ".join(top5),
    })
    print(f"  {treatment:>14s}: N={len(y):>6,}  feat={len(fn):>3d}  "
          f"LR AUC={auc_lr:.4f}  GBM AUC={auc_gbm:.4f}")

ns_df = pd.DataFrame(ns_rows)
ns_df.to_csv(tables_dir / "robustness_not_sure.csv", index=False)


# === 6.1c  Skip-Pattern Robustness ==========================================
print("\n" + "=" * 70)
print("6.1c  Skip-Pattern Robustness: Full set vs Safe set (A+B+C)")
print("=" * 70)

safe_rows = []
for set_name in ["full", "core_attitudes", "knowledge_ai_orientation", "demographics"]:
    if set_name == "full":
        feat_set = "full"
    else:
        feat_set = set_name

    try:
        X, y, w, fn = prepare_modeling_data(
            df_raw, feature_set=feat_set,
            missingness_regime="impute_indicator", not_sure_treatment="own_category",
            apply_encoding=True, apply_one_hot=True, drop_first=True,
        )
    except Exception as e:
        print(f"  {set_name}: SKIPPED ({e})")
        continue

    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y,
    )
    auc_lr, _ = quick_lr_auc(Xtr, ytr, wtr, Xte, yte, wte)
    auc_gbm, _ = quick_gbm_auc(Xtr, ytr, wtr, Xte, yte, wte)
    safe_rows.append({
        "feature_set": set_name, "n_raw_features": len(get_feature_set(feat_set)),
        "n_encoded": len(fn), "N": len(y),
        "auc_lr": round(auc_lr, 4), "auc_gbm": round(auc_gbm, 4),
    })
    print(f"  {set_name:>25s}: raw={len(get_feature_set(feat_set)):>2d}  "
          f"enc={len(fn):>3d}  N={len(y):>6,}  LR={auc_lr:.4f}  GBM={auc_gbm:.4f}")

safe_df = pd.DataFrame(safe_rows)
safe_df.to_csv(tables_dir / "robustness_feature_sets.csv", index=False)


# === 6.1d  Weight Sensitivity ===============================================
print("\n" + "=" * 70)
print("6.1d  Weight Sensitivity: Weighted vs Unweighted Training")
print("=" * 70)

X, y, w, fn = prepare_modeling_data(
    df_raw, feature_set="full",
    missingness_regime="impute_indicator", not_sure_treatment="own_category",
    apply_encoding=True, apply_one_hot=True, drop_first=True,
)
Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
    X, y, w, test_size=0.2, random_state=SEED, stratify=y,
)

weight_rows = []
for train_label, use_w in [("Weighted train", wtr), ("Unweighted train", np.ones(len(wtr)))]:
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
    lr.fit(Xtr, ytr, sample_weight=use_w)
    p = lr.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p, sample_weight=wte)  # always weighted eval

    gbm = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=SEED,
    )
    gbm.fit(Xtr, ytr, sample_weight=use_w)
    p_g = gbm.predict_proba(Xte)[:, 1]
    auc_g = roc_auc_score(yte, p_g, sample_weight=wte)

    weight_rows.append({
        "training": train_label,
        "auc_lr_weighted_eval": round(auc, 4),
        "auc_gbm_weighted_eval": round(auc_g, 4),
    })
    print(f"  {train_label:>20s}:  LR AUC={auc:.4f}  GBM AUC={auc_g:.4f}  (always weighted eval)")

weight_df = pd.DataFrame(weight_rows)
weight_df.to_csv(tables_dir / "robustness_weights.csv", index=False)


# === 6.1e  Seed Stability ===================================================
print("\n" + "=" * 70)
print("6.1e  Seed Stability: 20 random train/test splits")
print("=" * 70)

seed_results = []
for s in range(20):
    Xtr_s, Xte_s, ytr_s, yte_s, wtr_s, wte_s = train_test_split(
        X, y, w, test_size=0.2, random_state=s, stratify=y,
    )
    auc_lr_s, _ = quick_lr_auc(Xtr_s, ytr_s, wtr_s, Xte_s, yte_s, wte_s)
    auc_gbm_s, _ = quick_gbm_auc(Xtr_s, ytr_s, wtr_s, Xte_s, yte_s, wte_s)
    seed_results.append({"seed": s, "auc_lr": auc_lr_s, "auc_gbm": auc_gbm_s})

seed_df = pd.DataFrame(seed_results)
seed_df.to_csv(tables_dir / "robustness_seed_stability.csv", index=False)

print(f"  LR  AUC: mean={seed_df['auc_lr'].mean():.4f}  "
      f"std={seed_df['auc_lr'].std():.4f}  "
      f"range=[{seed_df['auc_lr'].min():.4f}, {seed_df['auc_lr'].max():.4f}]")
print(f"  GBM AUC: mean={seed_df['auc_gbm'].mean():.4f}  "
      f"std={seed_df['auc_gbm'].std():.4f}  "
      f"range=[{seed_df['auc_gbm'].min():.4f}, {seed_df['auc_gbm'].max():.4f}]")

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(seed_df["seed"], seed_df["auc_lr"], label="LR", marker="o", s=40)
ax.scatter(seed_df["seed"], seed_df["auc_gbm"], label="GBM", marker="s", s=40)
ax.axhline(seed_df["auc_lr"].mean(), color="#4C72B0", linestyle="--", alpha=0.5)
ax.axhline(seed_df["auc_gbm"].mean(), color="#D9534F", linestyle="--", alpha=0.5)
ax.set_xlabel("Random seed")
ax.set_ylabel("Weighted ROC-AUC")
ax.set_title("Seed Stability: AUC across 20 train/test splits", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "seed_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> seed_stability.png")


# === 6.2a  LR Coefficient Rank Stability ====================================
print("\n" + "=" * 70)
print("6.2a  LR Coefficient Rank Stability (200 bootstraps)")
print("=" * 70)

n_boot_rank = 200
rng = np.random.RandomState(SEED)
Xtr_arr = np.asarray(Xtr)
ytr_arr = np.asarray(ytr)
wtr_arr = np.asarray(wtr)
n_train = len(ytr_arr)

rank_counts = {f: [] for f in fn}

for b in range(n_boot_rank):
    idx = rng.choice(n_train, size=n_train, replace=True)
    lr_b = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
    try:
        lr_b.fit(Xtr_arr[idx], ytr_arr[idx], sample_weight=wtr_arr[idx])
        abs_coefs = np.abs(lr_b.coef_[0])
        ranked = np.argsort(-abs_coefs)
        for rank_pos, feat_idx in enumerate(ranked[:10]):
            rank_counts[fn[feat_idx]].append(rank_pos + 1)
    except Exception:
        pass

top10_freq = {f: len(ranks) for f, ranks in rank_counts.items()}
top10_df = (
    pd.DataFrame([
        {"feature": f, "top10_appearances": top10_freq[f],
         "top10_pct": round(top10_freq[f] / n_boot_rank * 100, 1),
         "median_rank": round(np.median(ranks), 1) if ranks else None}
        for f, ranks in rank_counts.items() if ranks
    ])
    .sort_values("top10_appearances", ascending=False)
    .head(20)
)
top10_df.to_csv(tables_dir / "lr_rank_stability.csv", index=False)

print("  Top features by top-10 appearance frequency:")
for _, row in top10_df.head(15).iterrows():
    print(f"    {row['feature']:<35s}  {row['top10_pct']:>5.1f}% of bootstraps  "
          f"median rank: {row['median_rank']}")


# === 6.2b  GBM Permutation Importance Stability =============================
print("\n" + "=" * 70)
print("6.2b  GBM Permutation Importance (5-fold CV)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_arr = np.asarray(X)
y_arr = np.asarray(y)
w_arr = np.asarray(w)

fold_importances = []
for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
    gbm_f = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=SEED,
    )
    gbm_f.fit(X_arr[tr_idx], y_arr[tr_idx], sample_weight=w_arr[tr_idx])
    perm = permutation_importance(gbm_f, X_arr[val_idx], y_arr[val_idx],
                                   n_repeats=10, random_state=SEED, scoring="roc_auc")
    fold_importances.append(perm.importances_mean)

perm_df = pd.DataFrame(np.array(fold_importances), columns=fn)
perm_mean = perm_df.mean().sort_values(ascending=False)
perm_std = perm_df.std()

perm_summary = pd.DataFrame({
    "feature": perm_mean.index,
    "importance_mean": perm_mean.values.round(4),
    "importance_std": perm_std[perm_mean.index].values.round(4),
}).reset_index(drop=True)
perm_summary["rank"] = range(1, len(perm_summary) + 1)
perm_summary.to_csv(tables_dir / "gbm_permutation_importance.csv", index=False)

print("  Top 15 by mean permutation importance:")
for _, row in perm_summary.head(15).iterrows():
    print(f"    #{row['rank']:>2.0f}  {row['feature']:<35s}  "
          f"imp={row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")


# === 6.2c  LR vs GBM Importance Agreement ===================================
print("\n" + "=" * 70)
print("6.2c  Agreement: LR |coef| vs GBM Permutation Importance")
print("=" * 70)

lr_full = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
lr_full.fit(Xtr, ytr, sample_weight=wtr)
lr_abs = pd.Series(np.abs(lr_full.coef_[0]), index=fn, name="lr_abs_coef")

gbm_imp = pd.Series(perm_mean.values, index=perm_mean.index, name="gbm_importance")

agreement = pd.DataFrame({"lr_abs_coef": lr_abs, "gbm_importance": gbm_imp}).dropna()
rho, p_val = stats.spearmanr(agreement["lr_abs_coef"], agreement["gbm_importance"])

print(f"  Spearman rho: {rho:.4f}  (p={p_val:.2e})")
if rho > 0.5:
    print(f"  Strong agreement between LR and GBM importance rankings")
elif rho > 0.3:
    print(f"  Moderate agreement")
else:
    print(f"  Weak agreement -- models may rely on different features")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(agreement["lr_abs_coef"], agreement["gbm_importance"], alpha=0.5, s=20)
ax.set_xlabel("|LR coefficient|")
ax.set_ylabel("GBM permutation importance")
ax.set_title(f"LR vs GBM Importance (Spearman rho={rho:.3f})", fontweight="bold")
ax.grid(alpha=0.3)
top5_lr = agreement.nlargest(5, "lr_abs_coef")
for feat in top5_lr.index:
    ax.annotate(feat, (agreement.loc[feat, "lr_abs_coef"], agreement.loc[feat, "gbm_importance"]),
                fontsize=6, alpha=0.8)
plt.tight_layout()
plt.savefig(figures_dir / "lr_vs_gbm_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> lr_vs_gbm_importance.png")


# === Summary =================================================================
print("\n" + "=" * 70)
print("PHASE 6.1 + 6.2 -- COMPLETE")
print("=" * 70)
print(f"\nOutputs:")
print(f"  Tables: robustness_outcome.csv, robustness_not_sure.csv,")
print(f"          robustness_feature_sets.csv, robustness_weights.csv,")
print(f"          robustness_seed_stability.csv, lr_rank_stability.csv,")
print(f"          gbm_permutation_importance.csv")
print(f"  Figures: seed_stability.png, lr_vs_gbm_importance.png")
print()
