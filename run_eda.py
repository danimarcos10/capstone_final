"""
Phase 3 -- EDA That Motivates Model Choices
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ---- Setup ---------------------------------------------------------------
from src.config import cfg, SURVEY, set_global_seed, get_feature_set, get_feature_labels, get_output_dir
from src.data_loading import load_atp_w119, create_target_variable
from src.preprocessing import (
    handle_not_sure,
    handle_refused_as_missing,
    handle_skip_patterns,
    encode_all_ordinals,
    compute_missingness_rates,
    missingness_model,
    prepare_modeling_data,
)
from src.eda import (
    weighted_outcome_by_subgroup,
    all_subgroup_tables,
    all_cramers_v,
    ordinal_correlations,
    compute_vif,
    lowess_outcome_rate,
    plot_subgroup_outcome_rates,
    plot_cramers_v,
    plot_lowess,
    plot_vif,
    plot_correlation_heatmap,
    generate_eda_report,
)

set_global_seed()
print("=" * 65)
print("PHASE 3 -- EDA That Motivates Model Choices")
print("=" * 65)

# ---- 1. Load data ---------------------------------------------------------
print("\n--- Loading data ---")
df, meta = load_atp_w119()
df["y_apply"] = create_target_variable(df)

n_valid_target = df["y_apply"].notna().sum()
prevalence = df.loc[df["y_apply"].notna(), "y_apply"].mean()
print(f"Valid target N = {n_valid_target:,}  |  Prevalence P(Y=1) = {prevalence:.3f}")

# ---- 2. Pre-process for EDA ------------------------------------------------
df_eda = df.copy()
features_full = get_feature_set("full")
features_safe = (
    get_feature_set("core_attitudes")
    + get_feature_set("knowledge_ai_orientation")
    + get_feature_set("demographics")
)

df_eda = handle_not_sure(df_eda, treatment="own_category")
df_eda = handle_refused_as_missing(df_eda, features_full)
df_eda, skip_notes = handle_skip_patterns(df_eda, features_full)
if skip_notes:
    print("Skip patterns applied:")
    for note in skip_notes:
        print(note)

# ---- 3. Weighted outcome rates by subgroup --------------------------------
print("\n--- 3.1 Weighted outcome rates by subgroup ---")

subgroup_vars = (
    get_feature_set("demographics")
    + ["EMPLSIT_W119", "JOBAPPYR_W119"]
)

subgroup_tables = all_subgroup_tables(df_eda, subgroup_vars=subgroup_vars, meta=meta)
label_map = get_feature_labels()

# Save tables to CSV
tables_dir = get_output_dir("tables")
for var, tbl in subgroup_tables.items():
    readable = label_map.get(var, var)
    tbl.to_csv(tables_dir / f"outcome_by_{readable}.csv", index=False)
    print(f"  {readable}: {len(tbl)} levels, "
          f"range [{tbl['weighted_outcome_pct'].min():.1f}% - {tbl['weighted_outcome_pct'].max():.1f}%]")

# Generate figures
figures_dir = get_output_dir("figures")
plot_subgroup_outcome_rates(subgroup_tables, save_dir=figures_dir)
print(f"  -> Subgroup outcome figures saved to {figures_dir}/")

# ---- 4. Weighted Cramer's V ------------------------------------------------
print("\n--- 3.2 Weighted Cramer's V ---")

cramers_df = all_cramers_v(df_eda, features=features_full, target_col="y_apply")
cramers_df.to_csv(tables_dir / "cramers_v.csv", index=False)

print("  Association ranking:")
for _, row in cramers_df.iterrows():
    v = row["cramers_v"]
    flag = " **" if v >= 0.10 else ""
    print(f"    #{row['rank']:.0f}  {row['readable_name']:<30s}  V = {v:.4f}{flag}")

# Generate figure
plot_cramers_v(cramers_df, save_dir=figures_dir)
print(f"  -> Cramer's V figure saved.")

# ---- 5. Ordinal correlations -----------------------------------------------
print("\n--- 3.2b Ordinal correlations (Spearman) ---")

ordinal_features = list(cfg.get("encoding", {}).get("ordinal", {}).keys())
corr_df = ordinal_correlations(df_eda, features=ordinal_features, target_col="y_apply")
corr_df.to_csv(tables_dir / "ordinal_correlations.csv", index=False)

print("  Spearman correlation ranking:")
for _, row in corr_df.iterrows():
    sig = " *" if row["p_value"] < 0.001 else ""
    print(f"    #{row['rank']:.0f}  {row['readable_name']:<30s}  rho = {row['spearman_rho']:+.4f}{sig}")

# Correlation heatmap
plot_correlation_heatmap(df_eda, features=ordinal_features, target_col="y_apply", save_dir=figures_dir)
print(f"  -> Correlation heatmap saved.")

# ---- 6. VIF multicollinearity check ----------------------------------------
print("\n--- 3.2c VIF multicollinearity check ---")

# drop_first=True required for meaningful VIF
X_vif, y_vif, w_vif, feat_names_vif = prepare_modeling_data(
    df_eda,
    feature_set="full",
    missingness_regime="impute_indicator",
    not_sure_treatment="own_category",
    apply_encoding=True,
    apply_one_hot=True,
    drop_first=True,
)

vif_df = compute_vif(X_vif, threshold=5.0)
vif_df.to_csv(tables_dir / "vif.csv", index=False)

n_flagged = vif_df["flagged"].sum()
print(f"  Total encoded features: {len(vif_df)}")
print(f"  Features with VIF > 5: {n_flagged}")

if n_flagged > 0:
    print("  Flagged features:")
    for _, row in vif_df[vif_df["flagged"]].iterrows():
        print(f"    {row['feature']:<40s}  VIF = {row['vif']:.1f}")

plot_vif(vif_df, save_dir=figures_dir)
print(f"  -> VIF figure saved.")

# ---- 7. LOESS nonlinearity checks ------------------------------------------
print("\n--- 3.3 LOESS nonlinearity checks ---")

lowess_data = {}

# AIKNOW_INDEX (quasi-continuous 0-6)
x, emp, smooth = lowess_outcome_rate(df_eda, "AIKNOW_INDEX_W119", frac=0.6)
lowess_data["ai_knowledge_score"] = (x, emp, smooth)
plot_lowess(x, emp, smooth, "ai_knowledge_score", save_dir=figures_dir)
print(f"  AIKNOW_INDEX: range [{x.min():.0f}, {x.max():.0f}], "
      f"outcome range [{emp.min()*100:.1f}%, {emp.max()*100:.1f}%]")
diffs = np.diff(smooth)
is_mono = np.all(diffs >= -0.01) or np.all(diffs <= 0.01)
print(f"  LOWESS trend: {'approximately monotone' if is_mono else 'NON-MONOTONE'}")

# Income tier (ordinal 1-3)
x2, emp2, smooth2 = lowess_outcome_rate(df_eda, "F_INC_TIER2", frac=0.8)
lowess_data["income_tier"] = (x2, emp2, smooth2)
plot_lowess(x2, emp2, smooth2, "income_tier", save_dir=figures_dir)
print(f"  F_INC_TIER2: range [{x2.min():.0f}, {x2.max():.0f}], "
      f"outcome range [{emp2.min()*100:.1f}%, {emp2.max()*100:.1f}%]")
diffs2 = np.diff(smooth2)
is_mono2 = np.all(diffs2 >= -0.01) or np.all(diffs2 <= 0.01)
print(f"  LOWESS trend: {'approximately monotone' if is_mono2 else 'NON-MONOTONE'}")

# Age category (ordinal 1-4)
x3, emp3, smooth3 = lowess_outcome_rate(df_eda, "F_AGECAT", frac=0.8)
lowess_data["age_category"] = (x3, emp3, smooth3)
plot_lowess(x3, emp3, smooth3, "age_category", save_dir=figures_dir)
print(f"  F_AGECAT: range [{x3.min():.0f}, {x3.max():.0f}], "
      f"outcome range [{emp3.min()*100:.1f}%, {emp3.max()*100:.1f}%]")
diffs3 = np.diff(smooth3)
is_mono3 = np.all(diffs3 >= -0.01) or np.all(diffs3 <= 0.01)
print(f"  LOWESS trend: {'approximately monotone' if is_mono3 else 'NON-MONOTONE'}")

# ---- 8. Get missingness model result (from Phase 2, recompute for report) --
print("\n--- Missingness model (from Phase 2, for report) ---")

miss_result = missingness_model(df_eda, features=features_full)
print(f"  Accuracy: {miss_result.get('accuracy', 'N/A')}")
print(f"  Conclusion: {miss_result.get('conclusion', 'N/A')}")

# ---- 9. Generate EDA summary report ----------------------------------------
print("\n--- 3.4 Generating EDA summary report ---")

report_text = generate_eda_report(
    subgroup_tables=subgroup_tables,
    cramers_df=cramers_df,
    corr_df=corr_df,
    vif_df=vif_df,
    lowess_data=lowess_data,
    missingness_model_result=miss_result,
)

# ---- Summary ---------------------------------------------------------------
print("\n" + "=" * 65)
print("PHASE 3 -- COMPLETE")
print("=" * 65)
print(f"\nOutputs generated:")
print(f"  Tables:  {tables_dir}/")
print(f"    - outcome_by_*.csv (subgroup outcome rates)")
print(f"    - cramers_v.csv")
print(f"    - ordinal_correlations.csv")
print(f"    - vif.csv")
print(f"  Figures: {figures_dir}/")
print(f"    - outcome_by_*.png (subgroup bar charts)")
print(f"    - cramers_v_all.png")
print(f"    - correlation_heatmap.png")
print(f"    - vif_top25.png")
print(f"    - lowess_*.png")
print(f"  Report:  reports/eda_summary.md")

# Key findings summary
print(f"\n--- KEY FINDINGS ---")
top3 = cramers_df.head(3)["readable_name"].tolist()
print(f"  Top 3 predictors (Cramer's V): {', '.join(top3)}")
print(f"  VIF > 5 flagged: {n_flagged}")
print(f"  Missingness: {miss_result.get('conclusion', 'N/A')[:60]}...")
print()
