# Phase 7.1 Robustness: Latent Model Stability

## Cross-Validation Stability (Pairwise FA, v2)

- **N folds**: 5
- **N items in factor**: 7
- **Estimation method**: Pairwise Spearman correlations + MINRES factor analysis

### Reliability Metrics Across Folds

- **Cronbach's alpha**: 0.643 ± 0.002
- **Ordinal alpha**: 0.668 ± 0.002
- **McDonald's omega**: 0.670 ± 0.002
- **KMO**: 0.719 ± 0.002

### Loading Stability

- **Mean loading correlation across folds**: 0.996
- **Min loading correlation**: 0.992
- **Interpretation**: Highly stable loadings across folds.

## 'Not Sure' Treatment Sensitivity

| Treatment | N Eligible | Cronbach's Alpha | KMO | ROC-AUC | Brier |
|-----------|-----------|------------------|-----|---------|-------|
| own_category | 8,714 | 0.760 | 0.826 | 0.6801 | 0.2156 |
| drop | 8,714 | 0.597 | 0.726 | 0.7847 | 0.1732 |
| midpoint | 8,714 | 0.540 | 0.685 | 0.7843 | 0.1735 |

**Best treatment by ROC-AUC**: drop

## Interpretation
- `own_category`: Treats 'Not sure' as a distinct middle category (preserves information).
- `drop`: Removes 'Not sure' responses (smaller sample, may reduce bias).
- `midpoint`: Replaces 'Not sure' with scale median (imputes uncertainty as neutrality).

**Phase 7.1 approach**: Treat 'Not sure' as NaN in factor analysis, create separate binary indicators for prediction. This preserves measurement rigor while retaining predictive signal.

---
*Generated: Phase 7.1 Robustness Checks*