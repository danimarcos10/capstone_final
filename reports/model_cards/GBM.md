# Model Card: HistGradientBoosting (uncalibrated)

## Overview
- **Model class**: `HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05)`
- **Training**: Weighted by WEIGHT_W119 (survey weights)
- **Features**: 58 encoded features (same as LR)
- **Target**: y_apply: 1=would apply for AI-hiring job, 0=would not
- **Data**: Pew Research Center ATP W119 (N=11,004, Dec 2022)

## Metrics (held-out test set, N=2,155)

| Metric | Value |
|--------|-------|
| roc_auc_weighted | 0.8669 |
| pr_auc_weighted | 0.7999 |
| brier_weighted | 0.1420 |
| ece_weighted | 0.0387 |
| balanced_acc_weighted | 0.7615 |

## Missingness Regime
- **Regime**: Impute + missing indicators
- **Justification**: Missingness is associated with demographics (CV AUC=0.772);
  listwise deletion would lose 56% of data and bias toward younger, higher-income respondents.
- **'Not sure' (code 9)**: Treated as own category (preserves information + sample size).

## Strengths
- Best discrimination (AUC=0.867)
- Permutation importance stable across CV folds
- Top predictor (favor_ai_review_apps) dominates clearly

## Limitations and Scope
- Less calibrated than LR (ECE=0.039 vs 0.031)
- Isotonic calibration did not help (small cal set)
- Weak agreement with LR on feature importance (Spearman rho=0.03)
- Same gender TPR gap as LR

## Scope
- **Intended use**: Academic analysis of attitudes toward AI in hiring.
- **Not intended for**: Deployment in hiring systems, individual-level prediction.
- **Population**: U.S. adults from Pew's American Trends Panel (non-probability corrections via weighting).

---
*Generated: 2026-02-12 19:04*