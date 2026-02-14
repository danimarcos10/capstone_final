"""
Phase 7.1: Fair Comparisons and Improved "Not Sure" Handling
=============================================================
Apples-to-apples comparisons (same features baseline vs latent)
and treats "Not sure" as separate indicators instead of ordinal levels.

Usage:
    python run_latent_v2.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

from src.config import cfg, set_global_seed, get_output_dir, get_feature_set
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import (
    prepare_modeling_data,
    create_train_test_split,
    encode_ordinal,
    one_hot_encode,
    handle_refused_as_missing,
)
from src.latent import (
    prepare_latent_items_v2,
    get_latent_item_columns,
    get_refined_latent_items,
    fit_latent_model,
    transform_latent,
    validate_latent_model,
    print_latent_model_summary,
    create_loadings_table,
    create_not_sure_count,
    compute_eligibility_bias,
    ordinal_alpha,
    mcdonalds_omega,
)
from src.modeling import (
    fit_weighted_logistic,
    fit_gradient_boosting,
)
from src.evaluation import (
    compute_all_metrics,
    compute_ece,
)


# --- Helpers -----------------------------------------------------------------

def build_matched_baseline_features(
    df: pd.DataFrame,
    not_sure_indicators: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build features using ONLY the 9 raw attitude items (matched to latent)."""
    from src.preprocessing import encode_ordinal
    
    item_cols = get_latent_item_columns()
    X = df[item_cols].copy()
    X = handle_refused_as_missing(X, item_cols)
    
    ordinal_encoding = cfg["encoding"]["ordinal"]
    for col in item_cols:
        if col in ordinal_encoding:
            X[col] = encode_ordinal(X, col, ordinal_encoding[col])
    
    if not_sure_indicators is not None:
        X = pd.concat([X, not_sure_indicators], axis=1)
    
    return X


def build_latent_plus_extras(
    df: pd.DataFrame,
    latent_scores: pd.DataFrame,
    not_sure_indicators: pd.DataFrame = None,
    supplementary_items: pd.DataFrame = None,
    include_knowledge: bool = True,
    include_demographics: bool = True,
) -> pd.DataFrame:
    """Build features with latent score + key extras (knowledge, demographics)."""
    X = latent_scores.copy()
    
    if not_sure_indicators is not None:
        X = pd.concat([X, not_sure_indicators], axis=1)
    
    # Supplementary raw items dropped from factor but still predictive
    if supplementary_items is not None:
        X = pd.concat([X, supplementary_items], axis=1)
    
    ordinal_encoding = cfg["encoding"]["ordinal"]
    
    if include_knowledge:
        know_cols = get_feature_set("knowledge_ai_orientation")
        know_df = df[know_cols].copy()
        know_df = handle_refused_as_missing(know_df, know_cols)
        
        ordinal_know = ["AI_HEARD_W119", "CNCEXC_W119", "USEAI_W119"]
        for col in ordinal_know:
            if col in know_df.columns and col in ordinal_encoding:
                know_df[col] = encode_ordinal(know_df, col, ordinal_encoding[col])
        
        X = pd.concat([X, know_df], axis=1)
    
    if include_demographics:
        demo_cols = get_feature_set("demographics")
        demo_df = df[demo_cols].copy()
        demo_df = handle_refused_as_missing(demo_df, demo_cols)
        
        ordinal_cols = ["F_AGECAT", "F_EDUCCAT2", "F_INC_TIER2"]
        for col in ordinal_cols:
            if col in demo_df.columns and col in ordinal_encoding:
                demo_df[col] = encode_ordinal(demo_df, col, ordinal_encoding[col])
        
        nominal_cols = ["F_GENDER", "F_RACETHNMOD", "F_PARTY_FINAL"]
        for col in nominal_cols:
            if col in demo_df.columns:
                demo_df = one_hot_encode(demo_df, col, drop_first=True)
        
        X = pd.concat([X, demo_df], axis=1)
    
    return X


def evaluate_model(
    model,
    X_test,
    y_test,
    w_test,
    model_name: str = "Model"
) -> Dict:
    """Evaluate a model and return metrics dictionary."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = compute_all_metrics(
        y_test, y_pred_proba, y_pred, weights=w_test
    )
    
    metrics["ece_weighted"] = compute_ece(
        y_test, y_pred_proba, n_bins=10, weights=w_test
    )
    
    metrics["model_name"] = model_name
    metrics["n_features"] = X_test.shape[1]
    
    return metrics


def plot_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    title: str = "Phase 7.1 Model Comparison"
):
    """Plot model comparison bars."""
    metrics_to_plot = ["roc_auc_weighted", "pr_auc_weighted", "brier_weighted", "ece_weighted"]
    metric_labels = ["ROC-AUC", "PR-AUC", "Brier Score", "ECE"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[i]
        
        ascending = (metric in ["brier_weighted", "ece_weighted"])
        sorted_df = comparison_df.sort_values(metric, ascending=ascending)
        
        bars = ax.barh(sorted_df["model_name"], sorted_df[metric], color="steelblue", alpha=0.7)
        
        best_idx = 0 if ascending else len(bars) - 1
        bars[best_idx].set_color("darkgreen")
        bars[best_idx].set_alpha(0.9)
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f"{label} Comparison", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis='x')
        
        for j, (idx, row) in enumerate(sorted_df.iterrows()):
            val = row[metric]
            ax.text(val, j, f"  {val:.3f}", va='center', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  -> {output_path}")


# --- Main pipeline -----------------------------------------------------------

def main():
    """Execute Phase 7.1 pipeline."""
    set_global_seed()
    
    print("=" * 70)
    print("  PHASE 7.1: FAIR COMPARISONS & NOT SURE INDICATORS")
    print("=" * 70)
    
    # --- Load data -----------------------------------------------------------
    print("\n[1/8] Loading data...")
    from src.config import PROJECT_ROOT
    pickle_path = PROJECT_ROOT / "data_full.pkl"
    
    if pickle_path.exists():
        df = pd.read_pickle(pickle_path)
    else:
        df, _ = load_atp_w119()
    
    df["y_apply"] = create_target_variable(df)
    df_valid = df[df["y_apply"].notna()].copy()
    print(f"  Valid sample: N={len(df_valid):,}")
    
    # --- Prepare latent items (v2: not_sure as indicators) -------------------
    print("\n[2/8] Preparing latent items (v2: not_sure as indicators)...")
    items_df, eligible_mask, not_sure_indicators = prepare_latent_items_v2(
        df_valid,
        apply_reverse_coding=True,
        skip_pattern_strategy="eligible_only",
        create_indicators=True
    )
    
    print(f"  Eligible for latent modeling: N={eligible_mask.sum():,}")
    print(f"  Not sure indicators created: {not_sure_indicators.shape[1]} columns")
    
    df_eligible = df_valid[eligible_mask].copy()
    items_eligible = items_df[eligible_mask].copy()
    not_sure_eligible = not_sure_indicators[eligible_mask].copy()
    y_eligible = df_eligible["y_apply"]
    w_eligible = df_eligible["WEIGHT_W119"]
    
    # --- Train/test split (shared across all models) -------------------------
    print("\n[3/10] Train/test split (shared across all models)...")
    items_train, items_test, y_train, y_test, w_train, w_test = create_train_test_split(
        items_eligible,
        y_eligible,
        w_eligible,
        test_size=cfg["modeling"]["test_size"],
        random_state=cfg["modeling"]["random_seed"]
    )
    
    not_sure_train = not_sure_eligible.loc[items_train.index]
    not_sure_test = not_sure_eligible.loc[items_test.index]
    df_train = df_eligible.loc[items_train.index]
    df_test = df_eligible.loc[items_test.index]
    
    print(f"  Train: N={len(items_train):,}")
    print(f"  Test:  N={len(items_test):,}")
    
    # --- Fit 1-factor latent model -------------------------------------------
    # Pairwise-complete Spearman correlations avoid listwise deletion.
    # Two-stage: (1) screen all 9 items, (2) retain items with |loading| >= 0.3.
    print("\n[4/10] Fitting 1-factor latent model (pairwise correlations)...")
    
    print("  Stage 1: Screening all 9 items...")
    fitted_screen = fit_latent_model(
        items_train,
        n_factors=1,
        method="principal",
        rotation=None,
        use_pairwise=True
    )
    
    # Retain items with |loading| >= 0.3
    refined_items = get_refined_latent_items(
        loading_threshold=0.3, fitted_model_dict=fitted_screen
    )
    dropped_items = [c for c in items_train.columns if c not in refined_items]
    print(f"  Retained items ({len(refined_items)}): {refined_items}")
    if dropped_items:
        print(f"  Dropped items ({len(dropped_items)}, |loading| < 0.3): {dropped_items}")
    
    print(f"  Stage 2: Refitting on {len(refined_items)} strong items...")
    items_train_refined = items_train[refined_items]
    items_test_refined = items_test[refined_items]
    
    fitted_1f = fit_latent_model(
        items_train_refined,
        n_factors=1,
        method="principal",
        rotation=None,
        use_pairwise=True
    )
    
    validation_1f = validate_latent_model(items_train_refined, fitted_1f)
    print_latent_model_summary(fitted_1f, validation_1f)
    
    if "n_pairwise_min" in fitted_1f:
        print(f"\n  Pairwise estimation:")
        print(f"    Average pairwise N: {fitted_1f['n_complete']:,}")
        print(f"    Minimum pairwise N: {fitted_1f['n_pairwise_min']:,}")
        print(f"    Rows with any observed item: {fitted_1f.get('n_any_observed', 'N/A'):,}")
    
    latent_train = transform_latent(fitted_1f, items_train_refined, use_loading_weighted=True)
    latent_test = transform_latent(fitted_1f, items_test_refined, use_loading_weighted=True)
    
    print(f"  Train latent scores: {latent_train.shape}")
    print(f"  Test latent scores:  {latent_test.shape}")
    print(f"  Train scores non-NaN: {latent_train.iloc[:, 0].notna().sum():,}")
    print(f"  Test scores non-NaN:  {latent_test.iloc[:, 0].notna().sum():,}")
    
    not_sure_count_train = create_not_sure_count(df_train)
    not_sure_count_test = create_not_sure_count(df_test)
    
    # Supplementary raw items (dropped from factor)
    supp_train = items_train[dropped_items].copy() if dropped_items else None
    supp_test = items_test[dropped_items].copy() if dropped_items else None
    if supp_train is not None:
        for col in supp_train.columns:
            train_median = supp_train[col].median()
            supp_train[col] = supp_train[col].fillna(train_median)
            supp_test[col] = supp_test[col].fillna(train_median)
    
    # --- Build feature sets for comparison -----------------------------------
    print("\n[5/10] Building feature sets for comparison...")
    
    # Matched baseline: 9 raw items only
    print("  Matched baseline (9 raw items, no indicators)...")
    X_train_matched_raw = build_matched_baseline_features(df_train, not_sure_indicators=None)
    X_test_matched_raw = build_matched_baseline_features(df_test, not_sure_indicators=None)
    
    # Matched baseline + not_sure indicators
    print("  Matched baseline (9 raw items + not_sure indicators)...")
    X_train_matched_ind = build_matched_baseline_features(df_train, not_sure_indicators=not_sure_train)
    X_test_matched_ind = build_matched_baseline_features(df_test, not_sure_indicators=not_sure_test)
    
    # Latent only
    print("  Latent only...")
    X_train_latent_only = latent_train.copy()
    X_test_latent_only = latent_test.copy()
    
    # Latent + not_sure_count (1 count instead of 6 indicators)
    print("  Latent + not_sure_count...")
    X_train_latent_ind = pd.concat([latent_train, not_sure_count_train], axis=1)
    X_test_latent_ind = pd.concat([latent_test, not_sure_count_test], axis=1)
    
    # Latent + knowledge + demographics
    print("  Latent + knowledge + demographics...")
    X_train_latent_full = build_latent_plus_extras(
        df_train, latent_train, not_sure_indicators=None,
        supplementary_items=supp_train,
        include_knowledge=True, include_demographics=True
    )
    X_test_latent_full = build_latent_plus_extras(
        df_test, latent_test, not_sure_indicators=None,
        supplementary_items=supp_test,
        include_knowledge=True, include_demographics=True
    )
    
    # Latent + knowledge + demographics + not_sure_count
    print("  Latent + knowledge + demographics + not_sure_count...")
    X_train_latent_full_ind = build_latent_plus_extras(
        df_train, latent_train, not_sure_indicators=not_sure_count_train,
        supplementary_items=supp_train,
        include_knowledge=True, include_demographics=True
    )
    X_test_latent_full_ind = build_latent_plus_extras(
        df_test, latent_test, not_sure_indicators=not_sure_count_test,
        supplementary_items=supp_test,
        include_knowledge=True, include_demographics=True
    )
    
    # Impute remaining NaNs with train median/mode
    all_feature_sets = [
        (X_train_matched_raw, X_test_matched_raw, "matched_raw"),
        (X_train_matched_ind, X_test_matched_ind, "matched_ind"),
        (X_train_latent_only, X_test_latent_only, "latent_only"),
        (X_train_latent_ind, X_test_latent_ind, "latent_ind"),
        (X_train_latent_full, X_test_latent_full, "latent_full"),
        (X_train_latent_full_ind, X_test_latent_full_ind, "latent_full_ind"),
    ]
    
    for X_tr, X_te, name in all_feature_sets:
        for col in X_tr.columns:
            if X_tr[col].isna().any():
                fill_val = X_tr[col].median() if X_tr[col].dtype in [np.float64, np.int64] else X_tr[col].mode()[0]
                X_tr[col] = X_tr[col].fillna(fill_val)
                X_te[col] = X_te[col].fillna(fill_val)
    
    print(f"    Matched raw: Train {X_train_matched_raw.shape}, Test {X_test_matched_raw.shape}")
    print(f"    Matched+ind: Train {X_train_matched_ind.shape}, Test {X_test_matched_ind.shape}")
    print(f"    Latent only: Train {X_train_latent_only.shape}, Test {X_test_latent_only.shape}")
    print(f"    Latent+count: Train {X_train_latent_ind.shape}, Test {X_test_latent_ind.shape}")
    print(f"    Latent+full: Train {X_train_latent_full.shape}, Test {X_test_latent_full.shape}")
    print(f"    Latent+full+count: Train {X_train_latent_full_ind.shape}, Test {X_test_latent_full_ind.shape}")
    
    # --- Train models on all feature sets ------------------------------------
    print("\n[6/10] Training models (LR + GBM on each feature set)...")
    
    def align_data(X_tr, X_te, y_tr, y_te, w_tr, w_te):
        common_train = X_tr.index.intersection(y_tr.index).intersection(w_tr.index)
        common_test = X_te.index.intersection(y_te.index).intersection(w_te.index)
        return (
            X_tr.loc[common_train], X_te.loc[common_test],
            y_tr.loc[common_train], y_te.loc[common_test],
            w_tr.loc[common_train], w_te.loc[common_test]
        )
    
    models_to_train = [
        (X_train_matched_raw, X_test_matched_raw, "Matched Raw (9 items)"),
        (X_train_matched_ind, X_test_matched_ind, "Matched + NotSure Ind"),
        (X_train_latent_only, X_test_latent_only, "Latent Only"),
        (X_train_latent_ind, X_test_latent_ind, "Latent + NotSure Count"),
        (X_train_latent_full, X_test_latent_full, "Latent + Know + Demo"),
        (X_train_latent_full_ind, X_test_latent_full_ind, "Latent + Know + Demo + NotSure"),
    ]
    
    results = []
    
    for X_tr, X_te, base_name in models_to_train:
        X_tr_a, X_te_a, y_tr_a, y_te_a, w_tr_a, w_te_a = align_data(
            X_tr, X_te, y_train, y_test, w_train, w_test
        )
        
        lr = fit_weighted_logistic(X_tr_a, y_tr_a, w_tr_a)
        metrics_lr = evaluate_model(lr, X_te_a, y_te_a, w_te_a, f"LR: {base_name}")
        results.append(metrics_lr)
        
        gbm = fit_gradient_boosting(X_tr_a, y_tr_a, w_tr_a)
        metrics_gbm = evaluate_model(gbm, X_te_a, y_te_a, w_te_a, f"GBM: {base_name}")
        results.append(metrics_gbm)
    
    # --- Comparison table ----------------------------------------------------
    print("\n[7/10] Creating comparison table...")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[[
        "model_name", "n_features", "roc_auc_weighted", "pr_auc_weighted",
        "brier_weighted", "ece_weighted", "balanced_acc_weighted"
    ]]
    
    print("\n" + "=" * 70)
    print("PHASE 7.1 MODEL COMPARISON (FAIR)")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    
    tables_dir = get_output_dir("tables")
    comparison_path = tables_dir / "phase7_1_model_comparison_matched.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n  -> {comparison_path}")
    
    # --- Psychometric quality table ------------------------------------------
    print("\n[8/10] Saving psychometric quality table...")
    
    psychometrics = {
        "metric": [
            "N items in factor",
            "N (avg pairwise)",
            "N (min pairwise)",
            "N (any observed items)",
            "N (complete cases)",
            "Cronbach's alpha (complete cases)",
            "Ordinal alpha (pairwise Spearman)",
            "McDonald's omega",
            "KMO",
            "Bartlett p-value",
            "Mean |loading|",
            "Min |loading|",
            "Max |loading|",
        ],
        "value": [
            f"{len(fitted_1f['item_names'])}",
            f"{fitted_1f['n_complete']:,}",
            f"{fitted_1f.get('n_pairwise_min', 'N/A'):,}",
            f"{fitted_1f.get('n_any_observed', 'N/A'):,}",
            f"{validation_1f['n_complete_cases']:,}",
            f"{validation_1f['cronbach_alpha']:.3f}",
            f"{validation_1f['ordinal_alpha']:.3f}",
            f"{validation_1f['omega']:.3f}",
            f"{fitted_1f['kmo']:.3f}",
            f"{fitted_1f['bartlett_p']:.2e}",
            f"{validation_1f['avg_loading']:.3f}",
            f"{validation_1f['min_loading']:.3f}",
            f"{validation_1f['max_loading']:.3f}",
        ],
    }
    psychometrics_df = pd.DataFrame(psychometrics)
    psychometrics_path = tables_dir / "latent_psychometrics_phase7_1.csv"
    psychometrics_df.to_csv(psychometrics_path, index=False)
    print(f"  -> {psychometrics_path}")
    print(psychometrics_df.to_string(index=False))
    
    # --- Eligibility bias (impact of Not-sure -> NaN on sample composition) --
    print("\n[9/10] Computing eligibility bias...")
    
    bias_df = compute_eligibility_bias(df_train, items_train)
    bias_path = tables_dir / "latent_eligibility_bias.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"  -> {bias_path}")
    print(bias_df.to_string(index=False))
    
    # --- Visualizations ------------------------------------------------------
    print("\n[10/10] Generating visualizations...")
    
    figures_dir = get_output_dir("figures")
    plot_comparison(
        comparison_df,
        figures_dir / "phase7_1_model_comparison_matched.png",
        title="Phase 7.1: Fair Model Comparisons"
    )
    
    loadings_1f = create_loadings_table(fitted_1f)
    loadings_1f.to_csv(tables_dir / "phase7_1_loadings_v2.csv", index=False)
    print(f"  -> {tables_dir / 'phase7_1_loadings_v2.csv'}")
    
    print("\n" + "=" * 70)
    print("  PHASE 7.1 COMPLETE")
    print("=" * 70)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 70)
    
    matched_best = comparison_df[comparison_df["model_name"].str.contains("Matched")]["roc_auc_weighted"].max()
    latent_best = comparison_df[comparison_df["model_name"].str.contains("Latent")]["roc_auc_weighted"].max()
    
    print(f"Best matched baseline (9 raw items): AUC = {matched_best:.4f}")
    print(f"Best latent model: AUC = {latent_best:.4f}")
    print(f"Difference: {latent_best - matched_best:+.4f}")
    
    latent_no_ind_rows = comparison_df[comparison_df["model_name"] == "GBM: Latent + Know + Demo"]["roc_auc_weighted"].values
    latent_with_ind_rows = comparison_df[comparison_df["model_name"] == "GBM: Latent + Know + Demo + NotSure"]["roc_auc_weighted"].values
    
    if len(latent_no_ind_rows) > 0 and len(latent_with_ind_rows) > 0:
        latent_no_ind = latent_no_ind_rows[0]
        latent_with_ind = latent_with_ind_rows[0]
        
        print(f"\nEffect of Not Sure count indicator:")
        print(f"  Latent + Know + Demo (no indicator):  AUC = {latent_no_ind:.4f}")
        print(f"  Latent + Know + Demo + NotSure count: AUC = {latent_with_ind:.4f}")
        print(f"  Improvement: {latent_with_ind - latent_no_ind:+.4f}")
    
    print(f"\nPsychometric quality (pairwise estimation):")
    print(f"  Avg pairwise N:    {fitted_1f['n_complete']:,}")
    print(f"  Ordinal alpha:     {validation_1f['ordinal_alpha']:.3f}")
    print(f"  McDonald's omega:  {validation_1f['omega']:.3f}")
    print(f"  KMO:               {fitted_1f['kmo']:.3f}")


if __name__ == "__main__":
    main()
