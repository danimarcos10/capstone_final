"""
Phase 7+ -- Latent Psychometric Upgrade
==========================================
Additive upgrade to Phase 7.1. Runs after run_latent_v2.py.

New outputs (does NOT modify any existing files):
  - reports/tables/latent_psychometrics_polychoric.csv
  - reports/tables/latent_factor_comparison.csv
  - reports/figures/latent_scree_plot.png
  - reports/tables/latent_gender_invariance.csv

Usage:
    python run_latent_upgrade.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from src.config import cfg, set_global_seed, get_output_dir, get_feature_set
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import create_train_test_split
from src.latent import (
    prepare_latent_items_v2,
    get_latent_item_columns,
    get_refined_latent_items,
    fit_latent_model,
    validate_latent_model,
    compute_polychoric_matrix,
    _compute_pairwise_spearman,
    _nearest_positive_definite,
    compute_kmo_from_corr,
    compute_bartlett_from_corr,
    cronbach_alpha,
    ordinal_alpha,
    mcdonalds_omega,
)
from factor_analyzer import FactorAnalyzer

set_global_seed()
SEED = cfg["modeling"]["random_seed"]

print("=" * 70)
print("PHASE 7+ -- Latent Psychometric Upgrade")
print("=" * 70)


# --- Load data ---------------------------------------------------------------
print("\n[1/6] Loading data...")
from src.config import PROJECT_ROOT
pickle_path = PROJECT_ROOT / "data_full.pkl"

if pickle_path.exists():
    df = pd.read_pickle(pickle_path)
else:
    df, _ = load_atp_w119()

df["y_apply"] = create_target_variable(df)
df_valid = df[df["y_apply"].notna()].copy()
print(f"  Valid sample: N={len(df_valid):,}")


# --- Prepare latent items (same v2 approach) ---------------------------------
print("\n[2/6] Preparing latent items (v2)...")
items_df, eligible_mask, _ = prepare_latent_items_v2(
    df_valid,
    apply_reverse_coding=True,
    skip_pattern_strategy="eligible_only",
    create_indicators=True,
)

df_eligible = df_valid[eligible_mask].copy()
items_eligible = items_df[eligible_mask].copy()
y_eligible = df_eligible["y_apply"]
w_eligible = df_eligible["WEIGHT_W119"]
print(f"  Eligible: N={len(df_eligible):,}")

# Same train/test split as Phase 7.1
items_train, items_test, y_train, y_test, w_train, w_test = create_train_test_split(
    items_eligible, y_eligible, w_eligible,
    test_size=cfg["modeling"]["test_size"],
    random_state=SEED,
)
df_train = df_eligible.loc[items_train.index]
print(f"  Train: N={len(items_train):,}  |  Test: N={len(items_test):,}")

# Refine items: screen all 9, drop |loading| < 0.3
fitted_screen = fit_latent_model(
    items_train, n_factors=1, method="principal",
    rotation=None, use_pairwise=True,
)
refined_items = get_refined_latent_items(
    loading_threshold=0.3, fitted_model_dict=fitted_screen,
)
items_train_refined = items_train[refined_items]
items_test_refined = items_test[refined_items]
print(f"  Refined items: {len(refined_items)}")

tables_dir = get_output_dir("tables")
figures_dir = get_output_dir("figures")


# --- Polychoric correlations + refit -----------------------------------------
# Polychoric correlations (Olsson, 1979) are appropriate for ordinal items.
print("\n[3/6] Computing polychoric correlations (MLE, pairwise)...")
print("  This may take 1-2 minutes...")

poly_corr = compute_polychoric_matrix(items_train_refined, pairwise=True)
spear_corr = _compute_pairwise_spearman(items_train_refined)

print("  Polychoric correlation matrix computed.")
print(f"  Mean |poly - spear| = {np.abs(poly_corr - spear_corr).mean():.4f}")

poly_corr_pd = _nearest_positive_definite(poly_corr)
n_train_pw = int(items_train_refined.notna().all(axis=1).sum())
avg_pw_n = int(
    (items_train_refined.notna().values.astype(float).T
     @ items_train_refined.notna().values.astype(float)).mean()
)

fa_poly_1f = FactorAnalyzer(
    n_factors=1, rotation=None, method="minres",
    use_smc=True, is_corr_matrix=True,
)
fa_poly_1f.fit(poly_corr_pd)

loadings_poly_1f = fa_poly_1f.loadings_
var_exp_poly_1f = fa_poly_1f.get_factor_variance()

kmo_poly = compute_kmo_from_corr(poly_corr_pd)
bart_p_poly = compute_bartlett_from_corr(poly_corr_pd, avg_pw_n)

items_complete = items_train_refined.dropna()
alpha_poly = cronbach_alpha(items_complete)

# Ordinal alpha using polychoric rather than Spearman
k = poly_corr_pd.shape[0]
total_r_poly = poly_corr_pd.sum()
ord_alpha_poly = (k / (k - 1)) * (1 - k / total_r_poly) if total_r_poly != 0 else np.nan

omega_poly = mcdonalds_omega(loadings_poly_1f, items_train_refined)

print(f"\n  Polychoric 1-factor results:")
print(f"    KMO:             {kmo_poly:.3f}")
print(f"    Bartlett p:      {bart_p_poly:.2e}")
print(f"    Cronbach alpha:  {alpha_poly:.3f}")
print(f"    Ordinal alpha (polychoric): {ord_alpha_poly:.3f}")
print(f"    McDonald's omega: {omega_poly:.3f}")
print(f"    Loadings:")
for name, lam in zip(refined_items, loadings_poly_1f[:, 0]):
    print(f"      {name:30s}: {lam:+.3f}")

psycho_poly = pd.DataFrame({
    "metric": [
        "Correlation method",
        "N items",
        "N (avg pairwise)",
        "N (complete cases)",
        "Cronbach's alpha",
        "Ordinal alpha (polychoric)",
        "McDonald's omega",
        "KMO",
        "Bartlett p-value",
        "Mean |loading|",
        "Min |loading|",
        "Max |loading|",
        "Variance explained (F1)",
    ],
    "value": [
        "Polychoric (MLE, Olsson 1979)",
        str(len(refined_items)),
        f"{avg_pw_n:,}",
        f"{len(items_complete):,}",
        f"{alpha_poly:.3f}",
        f"{ord_alpha_poly:.3f}",
        f"{omega_poly:.3f}",
        f"{kmo_poly:.3f}",
        f"{bart_p_poly:.2e}",
        f"{np.abs(loadings_poly_1f[:, 0]).mean():.3f}",
        f"{np.abs(loadings_poly_1f[:, 0]).min():.3f}",
        f"{np.abs(loadings_poly_1f[:, 0]).max():.3f}",
        f"{var_exp_poly_1f[1][0]:.1%}" if var_exp_poly_1f is not None else "N/A",
    ],
})
psycho_poly_path = tables_dir / "latent_psychometrics_polychoric.csv"
psycho_poly.to_csv(psycho_poly_path, index=False)
print(f"\n  -> {psycho_poly_path}")


# --- 1-Factor vs 2-Factor comparison + scree plot ---------------------------
print("\n[4/6] Comparing 1-factor vs 2-factor solutions...")

eigvals_poly = np.sort(np.linalg.eigvalsh(poly_corr_pd))[::-1]

fa_poly_2f = FactorAnalyzer(
    n_factors=2, rotation="varimax", method="minres",
    use_smc=True, is_corr_matrix=True,
)
fa_poly_2f.fit(poly_corr_pd)

loadings_poly_2f = fa_poly_2f.loadings_
var_exp_poly_2f = fa_poly_2f.get_factor_variance()

spear_corr_pd = _nearest_positive_definite(spear_corr)
eigvals_spear = np.sort(np.linalg.eigvalsh(spear_corr_pd))[::-1]

fa_spear_1f = FactorAnalyzer(
    n_factors=1, rotation=None, method="minres",
    use_smc=True, is_corr_matrix=True,
)
fa_spear_1f.fit(spear_corr_pd)
var_exp_spear_1f = fa_spear_1f.get_factor_variance()

fa_spear_2f = FactorAnalyzer(
    n_factors=2, rotation="varimax", method="minres",
    use_smc=True, is_corr_matrix=True,
)
fa_spear_2f.fit(spear_corr_pd)
var_exp_spear_2f = fa_spear_2f.get_factor_variance()

factor_comp_rows = []
for label, fa_1, fa_2, ve_1, ve_2, eig in [
    ("Polychoric", fa_poly_1f, fa_poly_2f, var_exp_poly_1f, var_exp_poly_2f, eigvals_poly),
    ("Spearman",   fa_spear_1f, fa_spear_2f, var_exp_spear_1f, var_exp_spear_2f, eigvals_spear),
]:
    factor_comp_rows.append({
        "Correlation": label,
        "Factors": 1,
        "Eigenvalue F1": round(eig[0], 3),
        "Eigenvalue F2": round(eig[1], 3),
        "Var Explained F1": f"{ve_1[1][0]:.1%}" if ve_1 is not None else "N/A",
        "Cum Var Explained": f"{ve_1[2][0]:.1%}" if ve_1 is not None else "N/A",
    })
    factor_comp_rows.append({
        "Correlation": label,
        "Factors": 2,
        "Eigenvalue F1": round(eig[0], 3),
        "Eigenvalue F2": round(eig[1], 3),
        "Var Explained F1": f"{ve_2[1][0]:.1%}" if ve_2 is not None else "N/A",
        "Cum Var Explained": f"{ve_2[2][0] + ve_2[2][1] if ve_2[2].shape[0] > 1 else ve_2[2][0]:.1%}" if ve_2 is not None else "N/A",
    })

factor_comp_df = pd.DataFrame(factor_comp_rows)
factor_comp_path = tables_dir / "latent_factor_comparison.csv"
factor_comp_df.to_csv(factor_comp_path, index=False)
print(f"  -> {factor_comp_path}")

print(f"\n  Eigenvalues (polychoric): {', '.join(f'{e:.3f}' for e in eigvals_poly)}")
print(f"  Eigenvalues (Spearman):   {', '.join(f'{e:.3f}' for e in eigvals_spear)}")
print(f"  Kaiser rule (eigenvalue > 1):  "
      f"poly={int((eigvals_poly > 1).sum())} factors, "
      f"spear={int((eigvals_spear > 1).sum())} factors")

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(1, len(eigvals_poly) + 1)
ax.plot(x, eigvals_poly, "o-", color="black", label="Polychoric", markersize=7)
ax.plot(x, eigvals_spear, "s--", color="grey", label="Spearman", markersize=6)
ax.axhline(1.0, color="red", linestyle=":", alpha=0.6, label="Kaiser criterion (=1)")
ax.set_xlabel("Factor number", fontsize=11)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title("Scree Plot: Polychoric vs Spearman Correlations",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
scree_path = figures_dir / "latent_scree_plot.png"
plt.savefig(str(scree_path), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  -> {scree_path}")

print(f"\n  2-Factor loadings (polychoric, varimax):")
print(f"    {'Item':30s}  {'F1':>7s}  {'F2':>7s}")
for name, (l1, l2) in zip(refined_items, loadings_poly_2f):
    print(f"    {name:30s}  {l1:+7.3f}  {l2:+7.3f}")


# --- Gender measurement invariance (configural) -----------------------------
print("\n[5/6] Testing configural invariance across gender...")

gender_col = "F_GENDER"
gender_vals = df_train[gender_col].dropna().unique()

# Pew ATP coding: 1=Man, 2=Woman
gender_map = {}
for v in sorted(gender_vals):
    if v == 99.0:
        continue
    gender_map[v] = f"gender_{int(v)}"

invariance_rows = []
loading_vectors = {}

for gender_code, gender_label in gender_map.items():
    mask = df_train[gender_col] == gender_code
    items_g = items_train_refined.loc[mask]
    n_g = int(mask.sum())

    if n_g < 100:
        print(f"  {gender_label}: N={n_g} < 100, skipping.")
        continue

    poly_corr_g = compute_polychoric_matrix(items_g, pairwise=True)
    poly_corr_g = _nearest_positive_definite(poly_corr_g)

    fa_g = FactorAnalyzer(
        n_factors=1, rotation=None, method="minres",
        use_smc=True, is_corr_matrix=True,
    )
    fa_g.fit(poly_corr_g)
    loadings_g = fa_g.loadings_[:, 0]
    loading_vectors[gender_label] = loadings_g

    kmo_g = compute_kmo_from_corr(poly_corr_g)

    for item_name, lam in zip(refined_items, loadings_g):
        invariance_rows.append({
            "group": gender_label,
            "N": n_g,
            "item": item_name,
            "loading": round(lam, 3),
            "KMO": round(kmo_g, 3),
        })

    print(f"  {gender_label} (N={n_g}): KMO={kmo_g:.3f}")

group_labels = list(loading_vectors.keys())
if len(group_labels) == 2:
    from scipy.stats import spearmanr
    l1 = loading_vectors[group_labels[0]]
    l2 = loading_vectors[group_labels[1]]
    rho_load, p_load = spearmanr(np.abs(l1), np.abs(l2))

    print(f"\n  Loading correlation ({group_labels[0]} vs {group_labels[1]}):")
    print(f"    Spearman rho(|loadings|): {rho_load:.3f}  (p={p_load:.3e})")
    print(f"    Max |loading difference|: {np.abs(l1 - l2).max():.3f}")
    print(f"    Mean |loading difference|: {np.abs(l1 - l2).mean():.3f}")

    invariance_rows.append({
        "group": "COMPARISON",
        "N": "",
        "item": "Spearman rho(|loadings|)",
        "loading": round(rho_load, 3),
        "KMO": "",
    })
    invariance_rows.append({
        "group": "COMPARISON",
        "N": "",
        "item": "Max |loading diff|",
        "loading": round(float(np.abs(l1 - l2).max()), 3),
        "KMO": "",
    })
    invariance_rows.append({
        "group": "COMPARISON",
        "N": "",
        "item": "Mean |loading diff|",
        "loading": round(float(np.abs(l1 - l2).mean()), 3),
        "KMO": "",
    })

invariance_df = pd.DataFrame(invariance_rows)
invariance_path = tables_dir / "latent_gender_invariance.csv"
invariance_df.to_csv(invariance_path, index=False)
print(f"\n  -> {invariance_path}")


# --- Summary -----------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 7+ -- LATENT UPGRADE COMPLETE")
print("=" * 70)
print(f"\nNew outputs:")
print(f"  - {psycho_poly_path}")
print(f"  - {factor_comp_path}")
print(f"  - {scree_path}")
print(f"  - {invariance_path}")
print(f"\nExisting Phase 7.1 outputs are UNTOUCHED.")

print(f"\n--- Spearman vs Polychoric (1-factor) ---")
spear_loadings = fa_spear_1f.loadings_[:, 0]
poly_loadings = loadings_poly_1f[:, 0]
print(f"  {'Item':30s}  {'Spearman':>9s}  {'Polychoric':>10s}  {'Diff':>7s}")
for name, ls, lp in zip(refined_items, spear_loadings, poly_loadings):
    print(f"  {name:30s}  {ls:+9.3f}  {lp:+10.3f}  {lp - ls:+7.3f}")

print(f"\n  Ordinal alpha (Spearman):   {ordinal_alpha(items_train_refined):.3f}")
print(f"  Ordinal alpha (polychoric): {ord_alpha_poly:.3f}")
print(f"  KMO (Spearman):             {compute_kmo_from_corr(spear_corr_pd):.3f}")
print(f"  KMO (polychoric):           {kmo_poly:.3f}")
print()
