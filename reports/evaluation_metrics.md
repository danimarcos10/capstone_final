# Evaluation Metrics Report - ATP W119 AI Hiring Analysis

## Overview

This report documents the comprehensive evaluation of two predictive models for willingness to apply to AI-hiring jobs.

**Target**: `y_apply` (1 = Would apply, 0 = Would NOT apply)

---

## Models Evaluated

| Model | Description | Weights |
|-------|-------------|---------|
| Logistic Regression | Linear model with standardized features | Trained with sample weights |
| Gradient Boosting | 100 trees, max_depth=4, learning_rate=0.1 | Trained with sample weights |

---

## Data Split

- **Training**: 80% stratified
- **Test**: 20% stratified
- **Random state**: 42 (reproducible)
- **Positive class rate**: ~33% (would apply)

---

## Evaluation Metrics

### Metric Definitions

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Area under ROC curve (discrimination ability) |
| **PR-AUC** | Area under Precision-Recall curve (useful for imbalanced data) |
| **Accuracy** | Overall correct predictions |
| **Precision** | TP / (TP + FP) for positive class |
| **Recall** | TP / (TP + FN) for positive class |
| **F1** | Harmonic mean of precision and recall |
| **Balanced Accuracy** | Average of recall for each class |
| **Brier Score** | Mean squared error of probabilities (lower is better) |
| **ECE** | Expected Calibration Error (lower is better) |

---

## Results Summary

### At Default Threshold (0.5)

| Metric | LR (Unweighted) | LR (Weighted) | GB (Unweighted) | GB (Weighted) |
|--------|-----------------|---------------|-----------------|---------------|
| ROC-AUC | 0.729 | 0.731 | 0.724 | 0.723 |
| Accuracy | 0.682 | 0.697 | 0.681 | 0.684 |
| Precision | 0.630 | - | 0.598 | - |
| Recall | 0.366 | - | 0.454 | - |
| F1 | 0.463 | - | 0.516 | - |

### At Optimal Threshold (Youden's J)

| Model | Threshold | Accuracy (Unw) | F1 (Unw) |
|-------|-----------|----------------|----------|
| Logistic Regression | 0.350 | 0.658 | 0.610 |
| Gradient Boosting | 0.296 | 0.630 | 0.622 |

**Key Finding**: Using the optimal threshold significantly improves recall and F1 at the cost of precision.

**Note**: Full results exported to `outputs/tables/evaluation_metrics_full.csv`

---

## Threshold Selection

### Methods Compared

1. **Default (0.5)**: Standard classification threshold
2. **Youden's J**: Maximizes (TPR - FPR)
3. **Max F1**: Maximizes F1 score on validation data

### Selected Operating Threshold

We recommend using **Youden's J optimal threshold** for balanced performance:
- Logistic Regression: ~0.35
- Gradient Boosting: ~0.35

---

## Weighted vs. Unweighted Metrics

### Why Both?

- **Unweighted**: Standard ML evaluation, treats all test samples equally
- **Weighted**: Accounts for survey design, generalizes to U.S. population

### Key Differences

- Weighted metrics may differ from unweighted if certain demographic groups are over/under-sampled
- For policy conclusions, weighted metrics are preferred
- For model comparison, unweighted metrics are standard

---

## Confusion Matrix (Gradient Boosting, Optimal Threshold)

### Unweighted Counts

|  | Pred: No | Pred: Yes |
|--|----------|-----------|
| **Actual: No** | TN | FP |
| **Actual: Yes** | FN | TP |

### Weighted Sums

|  | Pred: No | Pred: Yes |
|--|----------|-----------|
| **Actual: No** | Weighted TN | Weighted FP |
| **Actual: Yes** | Weighted FN | Weighted TP |

---

## Cross-Validation Results

5-fold stratified cross-validation on training set:

| Model | CV ROC-AUC (mean ± std) |
|-------|-------------------------|
| Logistic Regression | See notebook |
| Gradient Boosting | See notebook |

---

## Files Generated

- `outputs/tables/evaluation_metrics_full.csv` - Complete metrics table
- `outputs/figures/model_comparison.png` - ROC curves and confusion matrix
- `outputs/tables/robustness_checks.csv` - Sensitivity analyses

---

## Reproducibility

All results generated with:
- `random_state=42`
- Stratified splits
- Fixed hyperparameters (no tuning in final evaluation)

To reproduce, run `notebooks/04_robustness_and_weighted_eval.ipynb` from top to bottom.
