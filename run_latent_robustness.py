"""
Phase 7.1 Robustness: Latent Model Stability Checks
====================================================
1. Cross-validation stability of latent scores and loadings (pairwise FA)
2. Loading correlation across folds
3. "Not sure" treatment sensitivity

Usage:
    python run_latent_robustness.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

from src.config import cfg, set_global_seed, get_output_dir
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import create_cv_folds
from src.latent import (
    prepare_latent_items,
    prepare_latent_items_v2,
    get_refined_latent_items,
    fit_latent_model,
    transform_latent,
    validate_latent_model,
    cronbach_alpha,
    ordinal_alpha,
    mcdonalds_omega,
)
from src.modeling import fit_weighted_logistic, fit_gradient_boosting
from src.evaluation import compute_all_metrics


def main():
    """Execute latent model robustness checks."""
    set_global_seed()
    
    print("=" * 70)
    print("  PHASE 7.1 ROBUSTNESS: LATENT MODEL STABILITY")
    print("=" * 70)
    
    # --- Load data -----------------------------------------------------------
    print("\n[1/3] Loading data...")
    from src.config import PROJECT_ROOT
    pickle_path = PROJECT_ROOT / "data_full.pkl"
    
    if pickle_path.exists():
        df = pd.read_pickle(pickle_path)
    else:
        df, _ = load_atp_w119()
    
    df["y_apply"] = create_target_variable(df)
    df_valid = df[df["y_apply"].notna()].copy()
    
    # --- CV stability (pairwise FA, v2) --------------------------------------
    print("\n[2/3] Cross-validation stability (pairwise FA, v2)...")
    
    items_df, eligible_mask, _ = prepare_latent_items_v2(
        df_valid,
        apply_reverse_coding=True,
        skip_pattern_strategy="eligible_only",
        create_indicators=False
    )
    
    df_eligible = df_valid[eligible_mask].copy()
    items_eligible = items_df[eligible_mask].copy()
    y_eligible = df_eligible["y_apply"]
    w_eligible = df_eligible["WEIGHT_W119"]
    
    fitted_screen = fit_latent_model(
        items_eligible, n_factors=1, method="principal", use_pairwise=True
    )
    refined_items = get_refined_latent_items(
        loading_threshold=0.3, fitted_model_dict=fitted_screen
    )
    items_eligible_refined = items_eligible[refined_items]
    print(f"  Using {len(refined_items)} refined items: {refined_items}")
    
    n_folds = cfg["modeling"]["cv_folds"]
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=cfg["modeling"]["random_seed"]
    )
    
    fold_loadings = []
    alphas = []
    ord_alphas = []
    omegas = []
    kmos = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(items_eligible_refined, y_eligible)):
        print(f"  Fold {fold_idx + 1}/{n_folds}...")
        
        items_train = items_eligible_refined.iloc[train_idx]
        
        fitted = fit_latent_model(
            items_train, n_factors=1, method="principal", use_pairwise=True
        )
        validation = validate_latent_model(items_train, fitted)
        
        fold_loadings.append(fitted["loadings"][:, 0])
        alphas.append(validation["cronbach_alpha"])
        ord_alphas.append(validation["ordinal_alpha"])
        omegas.append(validation["omega"])
        kmos.append(fitted["kmo"])
    
    loading_corrs = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            corr = np.corrcoef(fold_loadings[i], fold_loadings[j])[0, 1]
            loading_corrs.append(corr)
    
    mean_loading_corr = np.mean(loading_corrs)
    min_loading_corr = np.min(loading_corrs)
    
    print(f"\n  === Cross-Validation Results ===")
    print(f"  Cronbach's alpha:    {np.mean(alphas):.3f} +/- {np.std(alphas):.3f}")
    print(f"  Ordinal alpha:       {np.mean(ord_alphas):.3f} +/- {np.std(ord_alphas):.3f}")
    print(f"  McDonald's omega:    {np.mean(omegas):.3f} +/- {np.std(omegas):.3f}")
    print(f"  KMO:                 {np.mean(kmos):.3f} +/- {np.std(kmos):.3f}")
    print(f"  Loading correlation: {mean_loading_corr:.3f} (min: {min_loading_corr:.3f})")
    
    tables_dir = get_output_dir("tables")
    stability_data = {
        "fold": list(range(1, n_folds + 1)),
        "cronbach_alpha": alphas,
        "ordinal_alpha": ord_alphas,
        "omega": omegas,
        "kmo": kmos,
    }
    stability_df = pd.DataFrame(stability_data)
    
    summary_row = pd.DataFrame([{
        "fold": "Mean +/- SD",
        "cronbach_alpha": f"{np.mean(alphas):.3f} +/- {np.std(alphas):.3f}",
        "ordinal_alpha": f"{np.mean(ord_alphas):.3f} +/- {np.std(ord_alphas):.3f}",
        "omega": f"{np.mean(omegas):.3f} +/- {np.std(omegas):.3f}",
        "kmo": f"{np.mean(kmos):.3f} +/- {np.std(kmos):.3f}",
    }])
    stability_df = pd.concat([stability_df, summary_row], ignore_index=True)
    
    stability_path = tables_dir / "latent_stability_phase7_1.csv"
    stability_df.to_csv(stability_path, index=False)
    print(f"\n  -> {stability_path}")
    
    # --- "Not sure" treatment sensitivity ------------------------------------
    print("\n[3/3] 'Not sure' treatment sensitivity...")
    
    results_by_treatment = []
    
    for treatment in ["own_category", "drop", "midpoint"]:
        print(f"  Testing treatment: {treatment}...")
        
        items_df_treat, eligible_mask_treat = prepare_latent_items(
            df_valid,
            not_sure_treatment=treatment,
            apply_reverse_coding=True,
            skip_pattern_strategy="eligible_only"
        )
        
        df_eligible_treat = df_valid[eligible_mask_treat].copy()
        items_eligible_treat = items_df_treat[eligible_mask_treat].copy()
        y_eligible_treat = df_eligible_treat["y_apply"]
        w_eligible_treat = df_eligible_treat["WEIGHT_W119"]
        
        from src.preprocessing import create_train_test_split
        items_train, items_test, y_train, y_test, w_train, w_test = create_train_test_split(
            items_eligible_treat,
            y_eligible_treat,
            w_eligible_treat,
            test_size=cfg["modeling"]["test_size"],
            random_state=cfg["modeling"]["random_seed"]
        )
        
        fitted = fit_latent_model(items_train, n_factors=1, method="principal")
        validation = validate_latent_model(items_train, fitted)
        
        latent_train = transform_latent(fitted, items_train, impute_missing=True)
        latent_test = transform_latent(fitted, items_test, impute_missing=True)
        
        X_train = latent_train.copy()
        X_test = latent_test.copy()
        
        common_train = X_train.index.intersection(y_train.index).intersection(w_train.index)
        common_test = X_test.index.intersection(y_test.index).intersection(w_test.index)
        
        X_train = X_train.loc[common_train]
        y_train_aligned = y_train.loc[common_train]
        w_train_aligned = w_train.loc[common_train]
        
        X_test = X_test.loc[common_test]
        y_test_aligned = y_test.loc[common_test]
        w_test_aligned = w_test.loc[common_test]
        
        lr = fit_weighted_logistic(X_train, y_train_aligned, w_train_aligned)
        
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = compute_all_metrics(
            y_test_aligned, y_pred_proba, y_pred, weights=w_test_aligned
        )
        
        results_by_treatment.append({
            "treatment": treatment,
            "n_eligible": len(df_eligible_treat),
            "n_train": len(items_train),
            "n_test": len(items_test),
            "cronbach_alpha": validation["cronbach_alpha"],
            "kmo": fitted["kmo"],
            "roc_auc": metrics["roc_auc_weighted"],
            "brier": metrics["brier_weighted"],
        })
    
    comparison_df = pd.DataFrame(results_by_treatment)
    
    print("\n" + "=" * 70)
    print("NOT SURE TREATMENT SENSITIVITY")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(tables_dir / "latent_not_sure_sensitivity.csv", index=False)
    print(f"\n  -> {tables_dir / 'latent_not_sure_sensitivity.csv'}")
    
    # --- Report --------------------------------------------------------------
    report_lines = [
        "# Phase 7.1 Robustness: Latent Model Stability",
        "",
        "## Cross-Validation Stability (Pairwise FA, v2)",
        "",
        f"- **N folds**: {n_folds}",
        f"- **N items in factor**: {len(refined_items)}",
        f"- **Estimation method**: Pairwise Spearman correlations + MINRES factor analysis",
        "",
        "### Reliability Metrics Across Folds",
        "",
        f"- **Cronbach's alpha**: {np.mean(alphas):.3f} +/- {np.std(alphas):.3f}",
        f"- **Ordinal alpha**: {np.mean(ord_alphas):.3f} +/- {np.std(ord_alphas):.3f}",
        f"- **McDonald's omega**: {np.mean(omegas):.3f} +/- {np.std(omegas):.3f}",
        f"- **KMO**: {np.mean(kmos):.3f} +/- {np.std(kmos):.3f}",
        "",
        "### Loading Stability",
        "",
        f"- **Mean loading correlation across folds**: {mean_loading_corr:.3f}",
        f"- **Min loading correlation**: {min_loading_corr:.3f}",
        f"- **Interpretation**: {'Highly stable' if mean_loading_corr > 0.95 else 'Stable' if mean_loading_corr > 0.90 else 'Moderately stable'} loadings across folds.",
        "",
        "## 'Not Sure' Treatment Sensitivity",
        "",
        "| Treatment | N Eligible | Cronbach's Alpha | KMO | ROC-AUC | Brier |",
        "|-----------|-----------|------------------|-----|---------|-------|",
    ]
    
    for _, row in comparison_df.iterrows():
        report_lines.append(
            f"| {row['treatment']} | {row['n_eligible']:,} | {row['cronbach_alpha']:.3f} | "
            f"{row['kmo']:.3f} | {row['roc_auc']:.4f} | {row['brier']:.4f} |"
        )
    
    best_idx = comparison_df["roc_auc"].argmax()
    best_treatment = comparison_df.iloc[best_idx]["treatment"]
    
    report_lines.extend([
        "",
        f"**Best treatment by ROC-AUC**: {best_treatment}",
        "",
        "## Interpretation",
        "- `own_category`: Treats 'Not sure' as a distinct middle category.",
        "- `drop`: Removes 'Not sure' responses (smaller sample).",
        "- `midpoint`: Replaces 'Not sure' with scale median.",
        "",
        "**Phase 7.1 approach**: Treat 'Not sure' as NaN in factor analysis, "
        "create separate binary indicators for prediction.",
        "",
        "---",
        "*Generated: Phase 7.1 Robustness Checks*",
    ])
    
    reports_dir = get_output_dir("reports")
    (reports_dir / "latent_robustness.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  -> {reports_dir / 'latent_robustness.md'}")
    
    print("\n" + "=" * 70)
    print("  PHASE 7.1 ROBUSTNESS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
