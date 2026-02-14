"""
run_all.py -- One-command reproducibility pipeline
=====================================================
Regenerates ALL tables, figures, reports, and metadata.

Usage:
    python run_all.py

Phases executed in order:
  1. EDA (diagnostics, figures, eda_summary.md)
  2. Modeling (LR, GBM, odds ratios, comparison)
  3. Evaluation (thresholds, calibration, subgroups)
  4. Robustness (sensitivity suite, interpretability stability)
  5. Metadata + Model Cards
"""

import subprocess
import sys
import json
import hashlib
import platform
import importlib.metadata
from datetime import datetime
from pathlib import Path

from src.config import cfg, SURVEY, get_output_dir, PROJECT_ROOT


def run_script(script_name: str) -> bool:
    """Run a Python script and return True if it succeeded."""
    print(f"\n{'='*70}")
    print(f"  RUNNING: {script_name}")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"\n  [FAIL] {script_name} exited with code {result.returncode}")
        return False
    print(f"\n  [OK] {script_name} completed successfully")
    return True


def generate_metadata():
    """Generate run_metadata.json with reproducibility info."""
    reports_dir = get_output_dir("reports")

    # Package versions
    packages = [
        "pandas", "numpy", "pyreadstat", "scikit-learn", "scipy",
        "statsmodels", "matplotlib", "seaborn", "PyYAML",
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    # Config hash
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config_hash = hashlib.md5(config_path.read_bytes()).hexdigest()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": cfg["modeling"]["random_seed"],
        "config_hash_md5": config_hash,
        "survey": {
            "name": SURVEY.NAME,
            "wave": SURVEY.WAVE,
            "n_total": SURVEY.N_TOTAL,
            "weight_var": SURVEY.WEIGHT_VAR,
        },
        "package_versions": versions,
    }

    meta_path = reports_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> {meta_path}")
    return metadata


def generate_sample_flow():
    """Generate sample-flow table (N at each pipeline stage)."""
    import pandas as pd
    from src.data_loading import load_atp_w119, create_target_variable
    from src.preprocessing import prepare_modeling_data
    from src.config import get_feature_set

    df, _ = load_atp_w119()
    df["y_apply"] = create_target_variable(df)
    features_full = get_feature_set("full")

    rows = [
        {"stage": "Raw loaded", "N": len(df), "note": "Full ATP W119 sample"},
        {"stage": "Valid target", "N": int(df["y_apply"].notna().sum()),
         "note": f"Excluded {int(df['y_apply'].isna().sum())} Refused (code 99)"},
    ]

    # Listwise on full
    X_lw, y_lw, _, _ = prepare_modeling_data(
        df, feature_set="full", missingness_regime="listwise",
        not_sure_treatment="own_category",
        apply_encoding=True, apply_one_hot=True, drop_first=True,
    )
    rows.append({"stage": "Listwise (full, own_category)", "N": len(y_lw),
                 "note": f"{100 - len(y_lw)/int(df['y_apply'].notna().sum())*100:.0f}% data loss"})

    # Impute on full
    X_imp, y_imp, _, _ = prepare_modeling_data(
        df, feature_set="full", missingness_regime="impute_indicator",
        not_sure_treatment="own_category",
        apply_encoding=True, apply_one_hot=True, drop_first=True,
    )
    rows.append({"stage": "Impute+indicator (full, own_category)", "N": len(y_imp),
                 "note": "Near-full retention"})

    # Listwise on full with drop treatment
    X_lw2, y_lw2, _, _ = prepare_modeling_data(
        df, feature_set="full", missingness_regime="listwise",
        not_sure_treatment="drop",
        apply_encoding=True, apply_one_hot=True, drop_first=True,
    )
    rows.append({"stage": "Listwise (full, NS=drop)", "N": len(y_lw2),
                 "note": f"{100 - len(y_lw2)/int(df['y_apply'].notna().sum())*100:.0f}% data loss"})

    flow_df = pd.DataFrame(rows)
    tables_dir = get_output_dir("tables")
    flow_df.to_csv(tables_dir / "sample_flow.csv", index=False)

    # Also write as markdown
    lines = [
        "# Sample Flow Table",
        "",
        "| Stage | N | Note |",
        "|-------|---|------|",
    ]
    for _, row in flow_df.iterrows():
        lines.append(f"| {row['stage']} | {row['N']:,} | {row['note']} |")

    reports_dir = get_output_dir("reports")
    (reports_dir / "sample_flow.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> sample_flow.csv + sample_flow.md")


def generate_model_cards():
    """Generate model card summaries."""
    model_cards_dir = get_output_dir("model_cards")

    # Load metrics
    metrics_path = get_output_dir("reports") / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    cards = {
        "LR_default": {
            "name": "Weighted Logistic Regression (L2, default)",
            "class": "LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')",
            "training": "Weighted by WEIGHT_W119 (survey weights)",
            "features": "58 encoded features (22 raw, impute+indicator, own_category for Not Sure, drop_first)",
            "target": "y_apply: 1=would apply for AI-hiring job, 0=would not",
            "metrics_key": "LR (default)",
            "strengths": [
                "Best-calibrated model (ECE=0.031)",
                "Interpretable: odds ratios with bootstrap CIs",
                "Robust to seed variation (AUC std=0.010 across 20 splits)",
            ],
            "limitations": [
                "Missing indicators dominate top coefficients (small N, unstable)",
                "Gender TPR gap: men=0.70 vs women=0.47",
                "Calibration imperfect for Hispanic (ECE=0.12) and small subgroups",
            ],
        },
        "GBM": {
            "name": "HistGradientBoosting (uncalibrated)",
            "class": "HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05)",
            "training": "Weighted by WEIGHT_W119 (survey weights)",
            "features": "58 encoded features (same as LR)",
            "target": "y_apply: 1=would apply for AI-hiring job, 0=would not",
            "metrics_key": "GBM",
            "strengths": [
                "Best discrimination (AUC=0.867)",
                "Permutation importance stable across CV folds",
                "Top predictor (favor_ai_review_apps) dominates clearly",
            ],
            "limitations": [
                "Less calibrated than LR (ECE=0.039 vs 0.031)",
                "Isotonic calibration did not help (small cal set)",
                "Weak agreement with LR on feature importance (Spearman rho=0.03)",
                "Same gender TPR gap as LR",
            ],
        },
    }

    for card_id, card in cards.items():
        mk = metrics.get(card["metrics_key"], {})

        lines = [
            f"# Model Card: {card['name']}",
            "",
            "## Overview",
            f"- **Model class**: `{card['class']}`",
            f"- **Training**: {card['training']}",
            f"- **Features**: {card['features']}",
            f"- **Target**: {card['target']}",
            f"- **Data**: Pew Research Center ATP W119 (N=11,004, Dec 2022)",
            "",
            "## Metrics (held-out test set, N=2,155)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for metric_name in ["roc_auc_weighted", "pr_auc_weighted", "brier_weighted",
                            "ece_weighted", "balanced_acc_weighted"]:
            val = mk.get(metric_name, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"| {metric_name} | {val} |")

        lines += [
            "",
            "## Missingness Regime",
            "- **Regime**: Impute + missing indicators",
            "- **Justification**: Missingness is associated with demographics (CV AUC=0.772);",
            "  listwise deletion would lose 56% of data and bias toward younger, higher-income respondents.",
            "- **'Not sure' (code 9)**: Treated as own category (preserves information + sample size).",
            "",
            "## Strengths",
        ]
        for s in card["strengths"]:
            lines.append(f"- {s}")

        lines += ["", "## Limitations and Scope"]
        for lim in card["limitations"]:
            lines.append(f"- {lim}")

        lines += [
            "",
            "## Scope",
            "- **Intended use**: Academic analysis of attitudes toward AI in hiring.",
            "- **Not intended for**: Deployment in hiring systems, individual-level prediction.",
            "- **Population**: U.S. adults from Pew's American Trends Panel (non-probability corrections via weighting).",
            "",
            "---",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        ]

        card_path = model_cards_dir / f"{card_id}.md"
        card_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  -> {card_path}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("  W119 CAPSTONE -- FULL REPRODUCIBILITY PIPELINE")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = True
    for script in ["run_eda.py", "run_modeling.py", "run_evaluation.py", "run_calibration_upgrade.py", "run_robustness.py", "run_latent_v2.py", "run_latent_upgrade.py", "run_latent_robustness.py"]:
        if not run_script(script):
            print(f"\n  PIPELINE HALTED at {script}")
            success = False
            break

    if success:
        print(f"\n{'='*70}")
        print("  GENERATING METADATA + MODEL CARDS")
        print(f"{'='*70}")

        generate_metadata()
        generate_sample_flow()
        generate_model_cards()

        print(f"\n{'='*70}")
        print("  PIPELINE COMPLETE")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
    else:
        sys.exit(1)
