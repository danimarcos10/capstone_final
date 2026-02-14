# Model Card: Weighted Logistic Regression (L2, default)

## Overview
- **Model class**: `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')`
- **Training**: Weighted by WEIGHT_W119 (survey weights)
- **Features**: 58 encoded features (22 raw, impute+indicator, own_category for Not Sure, drop_first)
- **Target**: y_apply: 1=would apply for AI-hiring job, 0=would not
- **Data**: Pew Research Center ATP W119 (N=11,004, Dec 2022)

## Metrics (held-out test set, N=2,155)

| Metric | Value |
|--------|-------|
| roc_auc_weighted | 0.8566 |
| pr_auc_weighted | 0.7952 |
| brier_weighted | 0.1449 |
| ece_weighted | 0.0312 |
| balanced_acc_weighted | 0.7454 |

## Missingness Regime
- **Regime**: Impute + missing indicators
- **Justification**: Missingness is associated with demographics (CV AUC=0.772);
  listwise deletion would lose 56% of data and bias toward younger, higher-income respondents.
- **'Not sure' (code 9)**: Treated as own category (preserves information + sample size).

## Strengths
- Best-calibrated model (ECE=0.031)
- Interpretable: odds ratios with bootstrap CIs
- Robust to seed variation (AUC std=0.010 across 20 splits)

## Limitations and Scope
- Missing indicators dominate top coefficients (small N, unstable)
- Gender TPR gap: men=0.70 vs women=0.47
- Calibration imperfect for Hispanic (ECE=0.12) and small subgroups

## Scope
- **Intended use**: Academic analysis of attitudes toward AI in hiring.
- **Not intended for**: Deployment in hiring systems, individual-level prediction.
- **Population**: U.S. adults from Pew's American Trends Panel (non-probability corrections via weighting).

---
*Generated: 2026-02-12 19:04*