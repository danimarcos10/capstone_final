"""
EDA Module for ATP W119 Analysis
==================================
Phase 3 deliverables:
  - Weighted outcome rates by subgroup
  - Weighted Cramer's V (association strength)
  - Ordinal correlations
  - VIF (multicollinearity)
  - LOESS nonlinearity checks
  - EDA report generation with modeling recommendations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from src.config import cfg, SURVEY, get_feature_set, get_feature_labels, get_output_dir


# ===================================================================
# 1. WEIGHTED OUTCOME RATES BY SUBGROUP
# ===================================================================

def weighted_outcome_by_subgroup(
    df: pd.DataFrame,
    subgroup_col: str,
    target_col: str = "y_apply",
    weight_col: str | None = None,
    meta=None,
) -> pd.DataFrame:
    """
    Compute weighted P(Y=1) by each level of a subgroup variable.

    Returns DataFrame: subgroup_value, label, n_unweighted,
    weighted_outcome_rate, weighted_outcome_pct.
    """
    if weight_col is None:
        weight_col = SURVEY.WEIGHT_VAR

    # Filter to valid target
    valid = df[df[target_col].notna() & df[subgroup_col].notna()].copy()
    valid = valid[valid[subgroup_col] != SURVEY.REFUSED_CODE]

    val_labels = {}
    if meta is not None:
        val_labels = meta.variable_value_labels.get(subgroup_col, {})

    rows = []
    for val in sorted(valid[subgroup_col].unique()):
        mask = valid[subgroup_col] == val
        subset = valid[mask]
        n = int(mask.sum())
        w_sum_y1 = subset.loc[subset[target_col] == 1, weight_col].sum()
        w_sum_total = subset[weight_col].sum()
        rate = w_sum_y1 / w_sum_total if w_sum_total > 0 else np.nan

        rows.append({
            "value": val,
            "label": val_labels.get(val, str(int(val))),
            "n_unweighted": n,
            "weighted_outcome_rate": round(rate, 4),
            "weighted_outcome_pct": round(rate * 100, 1),
        })

    return pd.DataFrame(rows)


def all_subgroup_tables(
    df: pd.DataFrame,
    subgroup_vars: list[str] | None = None,
    meta=None,
) -> dict[str, pd.DataFrame]:
    """
    Compute outcome rates for a list of subgroup variables.
    Returns dict mapping variable name -> table DataFrame.
    """
    if subgroup_vars is None:
        subgroup_vars = (
            get_feature_set("demographics")
            + ["EMPLSIT_W119", "JOBAPPYR_W119"]
        )

    results = {}
    for var in subgroup_vars:
        if var not in df.columns:
            continue
        results[var] = weighted_outcome_by_subgroup(df, var, meta=meta)

    return results


# ===================================================================
# 2. WEIGHTED CRAMER'S V
# ===================================================================

def weighted_cramers_v(
    x: pd.Series,
    y: pd.Series,
    weights: pd.Series,
) -> float:
    """Compute weighted Cramer's V between two categorical variables."""
    valid = x.notna() & y.notna() & weights.notna()
    x_v, y_v, w_v = x[valid], y[valid], weights[valid]

    cats_x = sorted(x_v.unique())
    cats_y = sorted(y_v.unique())

    ct = np.zeros((len(cats_x), len(cats_y)))
    for i, cx in enumerate(cats_x):
        for j, cy in enumerate(cats_y):
            mask = (x_v == cx) & (y_v == cy)
            ct[i, j] = w_v[mask].sum()

    n = ct.sum()
    if n == 0:
        return 0.0

    row_sums = ct.sum(axis=1, keepdims=True)
    col_sums = ct.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_cells = np.where(expected > 0, (ct - expected) ** 2 / expected, 0)
    chi2 = chi2_cells.sum()

    k = min(len(cats_x), len(cats_y))
    if k <= 1 or n <= 0:
        return 0.0
    v = np.sqrt(chi2 / (n * (k - 1)))
    return round(float(v), 4)


def all_cramers_v(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "y_apply",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute weighted Cramer's V between each feature and the target.
    Returns DataFrame sorted by V descending.
    """
    if weight_col is None:
        weight_col = SURVEY.WEIGHT_VAR

    label_map = get_feature_labels()
    rows = []

    for col in features:
        if col not in df.columns:
            continue

        mask = (df[col] != SURVEY.REFUSED_CODE) & df[col].notna() & df[target_col].notna()
        v = weighted_cramers_v(df.loc[mask, col], df.loc[mask, target_col], df.loc[mask, weight_col])

        rows.append({
            "variable": col,
            "readable_name": label_map.get(col, col),
            "cramers_v": v,
        })

    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    return result


# ===================================================================
# 3. ORDINAL CORRELATIONS
# ===================================================================

def ordinal_correlations(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "y_apply",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Compute Spearman rank correlations between ordinal features and target."""
    if weight_col is None:
        weight_col = SURVEY.WEIGHT_VAR

    label_map = get_feature_labels()
    rows = []

    for col in features:
        if col not in df.columns:
            continue

        mask = df[col].notna() & (df[col] != SURVEY.REFUSED_CODE) & df[target_col].notna()
        x = df.loc[mask, col]
        y = df.loc[mask, target_col]

        if len(x) < 30:
            continue

        rho, p = stats.spearmanr(x, y)
        rows.append({
            "variable": col,
            "readable_name": label_map.get(col, col),
            "spearman_rho": round(rho, 4),
            "p_value": p,
            "abs_rho": round(abs(rho), 4),
            "n": len(x),
        })

    result = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    return result


# ===================================================================
# 4. VIF (MULTICOLLINEARITY)
# ===================================================================

def compute_vif(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Compute VIF for each feature. VIF_j = 1/(1 - R^2_j)."""
    from numpy.linalg import LinAlgError

    X_arr = X.values.astype(float)
    n_features = X_arr.shape[1]
    vifs = []

    for j in range(n_features):
        y_j = X_arr[:, j]
        X_rest = np.delete(X_arr, j, axis=1)

        X_rest_c = np.column_stack([np.ones(len(y_j)), X_rest])

        try:
            beta = np.linalg.lstsq(X_rest_c, y_j, rcond=None)[0]
            y_hat = X_rest_c @ beta
            ss_res = np.sum((y_j - y_hat) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except (LinAlgError, ValueError):
            vif = float("inf")

        vifs.append(vif)

    result = pd.DataFrame({
        "feature": X.columns,
        "vif": [round(v, 2) for v in vifs],
    })
    result["flagged"] = result["vif"] > threshold
    result = result.sort_values("vif", ascending=False)
    return result


# ===================================================================
# 5. LOESS / LOWESS NONLINEARITY CHECK
# ===================================================================

def lowess_outcome_rate(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "y_apply",
    weight_col: str | None = None,
    frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical P(Y=1) at each level of a quasi-continuous feature
    and overlay a LOWESS smooth to check for nonlinearity.

    Returns:
        x_vals: unique feature values
        empirical_rates: weighted P(Y=1) at each level
        lowess_smooth: smoothed rates
    """
    if weight_col is None:
        weight_col = SURVEY.WEIGHT_VAR

    valid = df[[feature_col, target_col, weight_col]].dropna()
    valid = valid[valid[feature_col] != SURVEY.REFUSED_CODE]

    x_vals = sorted(valid[feature_col].unique())
    empirical = []

    for x in x_vals:
        mask = valid[feature_col] == x
        sub = valid[mask]
        w_y1 = sub.loc[sub[target_col] == 1, weight_col].sum()
        w_tot = sub[weight_col].sum()
        empirical.append(w_y1 / w_tot if w_tot > 0 else np.nan)

    x_arr = np.array(x_vals, dtype=float)
    emp_arr = np.array(empirical, dtype=float)

    try:
        import statsmodels.api as sm
        lowess_result = sm.nonparametric.lowess(emp_arr, x_arr, frac=frac)
        smooth = lowess_result[:, 1]
    except ImportError:
        smooth = emp_arr

    return x_arr, emp_arr, smooth


# ===================================================================
# 6. FIGURE GENERATION
# ===================================================================

def plot_subgroup_outcome_rates(
    tables: dict[str, pd.DataFrame],
    save_dir: Path | None = None,
) -> None:
    if save_dir is None:
        save_dir = get_output_dir("figures")

    label_map = get_feature_labels()

    for var, tbl in tables.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = tbl["label"].astype(str).values
        rates = tbl["weighted_outcome_pct"].values
        ns = tbl["n_unweighted"].values

        bars = ax.barh(range(len(labels)), rates, color="#4C72B0", edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(
            [f"{l}  (N={n:,})" for l, n in zip(labels, ns)], fontsize=9
        )
        ax.set_xlabel("Weighted % Would Apply (AIWRKH4=Yes)")
        readable = label_map.get(var, var)
        ax.set_title(f"Willingness to Apply by {readable}", fontweight="bold")
        ax.invert_yaxis()

        for bar, r in zip(bars, rates):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{r:.1f}%", va="center", fontsize=8)

        ax.set_xlim(0, max(rates) * 1.2)
        plt.tight_layout()
        plt.savefig(save_dir / f"outcome_by_{readable}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_cramers_v(cramers_df: pd.DataFrame, save_dir: Path | None = None) -> None:
    if save_dir is None:
        save_dir = get_output_dir("figures")

    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = cramers_df.sort_values("cramers_v", ascending=True)

    colors = ["#D9534F" if v > 0.15 else "#F0AD4E" if v > 0.08 else "#5CB85C"
              for v in df_sorted["cramers_v"]]

    ax.barh(range(len(df_sorted)), df_sorted["cramers_v"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["readable_name"].values, fontsize=8)
    ax.set_xlabel("Weighted Cramer's V")
    ax.set_title("Association Strength with Outcome (AIWRKH4)", fontweight="bold")
    ax.axvline(0.10, color="gray", linestyle="--", alpha=0.5, label="V=0.10 (small)")
    ax.axvline(0.20, color="gray", linestyle=":", alpha=0.5, label="V=0.20 (medium)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / "cramers_v_all.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_lowess(
    x_vals, empirical, smooth,
    feature_name: str,
    save_dir: Path | None = None,
) -> None:
    if save_dir is None:
        save_dir = get_output_dir("figures")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_vals, empirical * 100, s=80, color="#4C72B0", zorder=3, label="Empirical rate")
    ax.plot(x_vals, smooth * 100, color="#D9534F", linewidth=2, label="LOWESS smooth")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Weighted % Would Apply")
    ax.set_title(f"Nonlinearity Check: {feature_name} vs P(Apply)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"lowess_{feature_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_vif(vif_df: pd.DataFrame, save_dir: Path | None = None) -> None:
    if save_dir is None:
        save_dir = get_output_dir("figures")

    df_show = vif_df.head(25).sort_values("vif", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#D9534F" if v > 5 else "#F0AD4E" if v > 3 else "#5CB85C"
              for v in df_show["vif"]]
    ax.barh(range(len(df_show)), df_show["vif"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_show)))
    ax.set_yticklabels(df_show["feature"].values, fontsize=7)
    ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factor (top 25)", fontweight="bold")
    ax.axvline(5, color="red", linestyle="--", alpha=0.7, label="VIF=5 threshold")
    ax.axvline(10, color="darkred", linestyle=":", alpha=0.7, label="VIF=10 severe")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / "vif_top25.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "y_apply",
    save_dir: Path | None = None,
) -> None:
    if save_dir is None:
        save_dir = get_output_dir("figures")

    label_map = get_feature_labels()
    cols = [f for f in features if f in df.columns] + [target_col]
    sub = df[cols].dropna()
    for c in cols:
        sub = sub[sub[c] != SURVEY.REFUSED_CODE]

    corr = sub.corr(method="spearman")

    rename = {c: label_map.get(c, c) for c in corr.columns}
    rename[target_col] = "Y: would_apply"
    corr = corr.rename(index=rename, columns=rename)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-0.5, vmax=0.5, square=True, ax=ax,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
    ax.set_title("Spearman Correlation Matrix (Ordinal Features + Target)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ===================================================================
# 7. EDA REPORT GENERATION
# ===================================================================

def generate_eda_report(
    subgroup_tables: dict[str, pd.DataFrame],
    cramers_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    lowess_data: dict,
    missingness_model_result: dict,
    output_path: str | None = None,
) -> str:
    """
    Generate the updated EDA summary with explicit modeling recommendations.
    """
    label_map = get_feature_labels()

    lines = [
        "# EDA Summary: Model-Choice-Driving Diagnostics",
        "",
        "This report documents the EDA that directly motivates modeling decisions.",
        "Every finding leads to a concrete recommendation for the pipeline.",
        "",
        "---",
        "",
        "## 1. Weighted Outcome Rates by Subgroup",
        "",
    ]

    for var, tbl in subgroup_tables.items():
        readable = label_map.get(var, var)
        lines.append(f"### {readable}")
        lines.append("")
        lines.append("| Value | Label | N (unweighted) | Weighted % Apply |")
        lines.append("|-------|-------|----------------|------------------|")
        for _, row in tbl.iterrows():
            lines.append(
                f"| {row['value']:.0f} | {row['label']} | {row['n_unweighted']:,} | "
                f"{row['weighted_outcome_pct']:.1f}% |"
            )
        lines.append("")

    # Cramer's V
    lines += [
        "---",
        "",
        "## 2. Association Strength (Weighted Cramer's V)",
        "",
        "| Rank | Variable | Cramer's V | Interpretation |",
        "|------|----------|-----------|---------------|",
    ]
    for _, row in cramers_df.iterrows():
        v = row["cramers_v"]
        if v >= 0.20:
            interp = "STRONG"
        elif v >= 0.10:
            interp = "Moderate"
        elif v >= 0.05:
            interp = "Weak"
        else:
            interp = "Negligible"
        lines.append(
            f"| {row['rank']} | {row['readable_name']} | {v:.4f} | {interp} |"
        )

    strong = cramers_df[cramers_df["cramers_v"] >= 0.10]["readable_name"].tolist()
    weak = cramers_df[cramers_df["cramers_v"] < 0.05]["readable_name"].tolist()
    lines += [
        "",
        "**Modeling recommendation**:",
        f"- Strong predictors (V >= 0.10): {', '.join(strong) if strong else 'None'}",
        f"- Negligible predictors (V < 0.05): {', '.join(weak) if weak else 'None'}. Consider dropping for parsimony.",
        "",
    ]

    # Ordinal correlations
    lines += [
        "---",
        "",
        "## 3. Ordinal Correlations (Spearman rho)",
        "",
        "| Rank | Variable | Spearman rho | |rho| | N |",
        "|------|----------|-----------:|------:|----:|",
    ]
    for _, row in corr_df.iterrows():
        lines.append(
            f"| {row['rank']} | {row['readable_name']} | {row['spearman_rho']:.4f} | "
            f"{row['abs_rho']:.4f} | {row['n']:,} |"
        )

    lines.append("")

    n_flagged = vif_df["flagged"].sum()
    lines += [
        "---",
        "",
        "## 4. Multicollinearity (VIF)",
        "",
        f"**Features with VIF > 5**: {n_flagged}",
        "",
    ]
    if n_flagged > 0:
        flagged = vif_df[vif_df["flagged"]]
        lines.append("| Feature | VIF |")
        lines.append("|---------|-----|")
        for _, row in flagged.iterrows():
            lines.append(f"| {row['feature']} | {row['vif']:.1f} |")
        lines += [
            "",
            "**Note**: VIF computed with `drop_first=True` (reference-category encoding).",
            "High VIF for employment_status and INDUSTRYCOMBO_missing is expected:",
            "INDUSTRYCOMBO is only asked to employed respondents, so the missing indicator",
            "is structurally linked to employment status.",
            "",
            "**Modeling recommendation**:",
            "- For logistic regression: keep `drop_first=True`. Consider excluding the",
            "  employment context set (Set D) if collinearity distorts coefficients.",
            "- For tree-based models: VIF is not a concern; trees handle collinearity naturally.",
            "",
        ]
    else:
        lines.append("No features flagged. Multicollinearity is not a concern.")
        lines.append("")

    lines += [
        "---",
        "",
        "## 5. Nonlinearity Checks",
        "",
    ]
    for feat_name, (x_vals, emp, smooth) in lowess_data.items():
        diffs = np.diff(smooth)
        is_monotone = np.all(diffs >= -0.01) or np.all(diffs <= 0.01)
        mono_str = "approximately monotone" if is_monotone else "NON-MONOTONE"

        lines.append(f"### {feat_name}")
        lines.append(f"- Range: [{x_vals.min():.0f}, {x_vals.max():.0f}]")
        lines.append(f"- Outcome range: [{emp.min()*100:.1f}%, {emp.max()*100:.1f}%]")
        lines.append(f"- LOWESS trend: **{mono_str}**")

        if not is_monotone:
            lines.append("- **Recommendation**: Consider GAM spline or binning for this feature.")
        else:
            lines.append("- **Recommendation**: Linear term is adequate.")
        lines.append("")

    # Missingness
    cv_auc = missingness_model_result.get("cv_auc_mean", "N/A")
    cv_std = missingness_model_result.get("cv_auc_std", "N/A")
    train_acc = missingness_model_result.get("train_accuracy", "N/A")
    lines += [
        "---",
        "",
        "## 6. Missingness Diagnostic",
        "",
        f"- 5-fold cross-validated ROC-AUC: **{cv_auc} +/- {cv_std}**",
        f"- Training accuracy (for reference): {train_acc}",
        f"- Conclusion: {missingness_model_result.get('conclusion', 'N/A')}",
        "",
        "Note: This is a diagnostic for whether missingness is associated with",
        "observed demographics. A CV AUC substantially above 0.50 is evidence",
        "against MCAR, supporting imputation + sensitivity analysis.",
        "",
    ]

    lines += [
        "---",
        "",
        "## 7. Overall Modeling Recommendations",
        "",
        "Based on the EDA diagnostics above:",
        "",
        "1. **Missingness regime**: Use impute+indicator (evidence against MCAR; missingness associated with demographics). Listwise deletion loses 56% of data.",
        "2. **'Not sure' treatment**: Use own_category (preserves information and sample size).",
        f"3. **Strongest predictors**: {', '.join(strong[:5])} -- prioritize these in any parsimonious model.",
        f"4. **Negligible predictors**: {', '.join(weak[:3]) if weak else 'None'} -- safe to drop if needed.",
    ]

    # Nonlinearity recommendation
    item_num = 5
    for feat_name, (x_vals, emp, smooth) in lowess_data.items():
        diffs = np.diff(smooth)
        is_monotone = np.all(diffs >= -0.01) or np.all(diffs <= 0.01)
        if not is_monotone:
            lines.append(f"{item_num}. **Nonlinearity ({feat_name})**: NON-MONOTONE. Use GAM spline or binning.")
        else:
            lines.append(f"{item_num}. **Nonlinearity ({feat_name})**: Monotone. Linear/ordinal encoding OK.")
        item_num += 1

    if n_flagged > 0:
        lines.append(f"{item_num}. **Multicollinearity**: {n_flagged} features flagged (VIF>5). Use drop_first for logistic regression. employment_status + INDUSTRYCOMBO_missing are structurally linked (expected).")
    else:
        lines.append(f"{item_num}. **Multicollinearity**: No concern (all VIF < 5).")
    item_num += 1

    lines += [
        f"{item_num}. **Model classes to try**: Logistic regression (interpretable), gradient boosting (flexible), GAM (if nonlinearity found).",
        "",
        "---",
        "",
        "## Figures",
        "",
        "All figures saved to `reports/figures/`:",
        "- `outcome_by_*.png` -- outcome rates by each subgroup",
        "- `cramers_v_all.png` -- association strength bar chart",
        "- `correlation_heatmap.png` -- Spearman correlation matrix",
        "- `vif_top25.png` -- VIF bar chart",
        "- `lowess_*.png` -- nonlinearity checks",
    ]

    report = "\n".join(lines)

    if output_path is None:
        output_path = get_output_dir("reports") / "eda_summary.md"

    Path(output_path).write_text(report, encoding="utf-8")
    print(f"EDA summary report saved -> {output_path}")

    return report
