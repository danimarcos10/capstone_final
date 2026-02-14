"""
Evaluation Module for ATP W119 Analysis
Comprehensive metrics computation including weighted evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
import warnings


def weighted_accuracy(y_true, y_pred, weights):
    """Compute weighted accuracy."""
    correct = (y_true == y_pred).astype(float)
    return np.average(correct, weights=weights)


def weighted_roc_auc(y_true, y_prob, weights):
    """Compute weighted ROC AUC."""
    return roc_auc_score(y_true, y_prob, sample_weight=weights)


def weighted_average_precision(y_true, y_prob, weights):
    """Compute weighted average precision (PR AUC)."""
    return average_precision_score(y_true, y_prob, sample_weight=weights)


def weighted_brier_score(y_true, y_prob, weights):
    """Compute weighted Brier score."""
    return brier_score_loss(y_true, y_prob, sample_weight=weights)


def weighted_confusion_matrix(y_true, y_pred, weights):
    """
    Compute weighted confusion matrix (sum of weights in each cell).

    Returns:
        2x2 array with weighted counts.
    """
    cm = np.zeros((2, 2))
    
    for i, (yt, yp, w) in enumerate(zip(y_true, y_pred, weights)):
        cm[int(yt), int(yp)] += w
    
    return cm


def weighted_precision_recall_f1(y_true, y_pred, weights, pos_label=1):
    """
    Compute weighted precision, recall, and F1 for positive class.
    """
    cm = weighted_confusion_matrix(y_true, y_pred, weights)
    
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def weighted_balanced_accuracy(y_true, y_pred, weights):
    """Compute weighted balanced accuracy (average of recall per class)."""
    cm = weighted_confusion_matrix(y_true, y_pred, weights)
    
    recall_0 = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
    recall_1 = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
    
    return (recall_0 + recall_1) / 2


def compute_ece(y_true, y_prob, n_bins=10, weights=None):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        weights: Optional sample weights
        
    Returns:
        ECE value
    """
    if weights is None:
        weights = np.ones(len(y_true))
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_weight = weights.sum()
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        
        if bin_mask.sum() > 0:
            bin_weights = weights[bin_mask]
            bin_weight_sum = bin_weights.sum()
            
            mean_conf = np.average(y_prob[bin_mask], weights=bin_weights)
            mean_acc = np.average(y_true[bin_mask], weights=bin_weights)
            
            ece += (bin_weight_sum / total_weight) * abs(mean_acc - mean_conf)
    
    return ece


def find_optimal_threshold_youden(y_true, y_prob):
    """
    Find optimal threshold using Youden's J statistic (TPR - FPR).
    
    Returns:
        optimal_threshold, youden_j_value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    
    return thresholds[optimal_idx], youden_j[optimal_idx]


def find_optimal_threshold_f1(y_true, y_prob):
    """
    Find optimal threshold that maximizes F1 score.
    
    Returns:
        optimal_threshold, max_f1
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Compute F1 for each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Exclude sentinel
    optimal_idx = np.argmax(f1_scores[:-1])
    
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def compute_all_metrics(y_true, y_prob, y_pred, weights=None, threshold_name='default'):
    """
    Compute comprehensive metrics (both unweighted and weighted).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted labels (binary)
        weights: Sample weights (if None, only unweighted metrics)
        threshold_name: Name for the threshold scheme
        
    Returns:
        Dict with all metrics
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    
    metrics = {
        'threshold_scheme': threshold_name,
        'roc_auc_unweighted': roc_auc_score(y_true, y_prob),
        'pr_auc_unweighted': average_precision_score(y_true, y_prob),
        'accuracy_unweighted': accuracy_score(y_true, y_pred),
        'precision_unweighted': precision_score(y_true, y_pred, zero_division=0),
        'recall_unweighted': recall_score(y_true, y_pred, zero_division=0),
        'f1_unweighted': f1_score(y_true, y_pred, zero_division=0),
        'balanced_acc_unweighted': balanced_accuracy_score(y_true, y_pred),
        'brier_unweighted': brier_score_loss(y_true, y_prob),
        'ece_unweighted': compute_ece(y_true, y_prob, n_bins=10),
    }
    
    if weights is not None:
        weights = np.array(weights)
        
        prec_w, rec_w, f1_w = weighted_precision_recall_f1(y_true, y_pred, weights)
        
        metrics.update({
            'roc_auc_weighted': weighted_roc_auc(y_true, y_prob, weights),
            'pr_auc_weighted': weighted_average_precision(y_true, y_prob, weights),
            'accuracy_weighted': weighted_accuracy(y_true, y_pred, weights),
            'precision_weighted': prec_w,
            'recall_weighted': rec_w,
            'f1_weighted': f1_w,
            'balanced_acc_weighted': weighted_balanced_accuracy(y_true, y_pred, weights),
            'brier_weighted': weighted_brier_score(y_true, y_prob, weights),
            'ece_weighted': compute_ece(y_true, y_prob, n_bins=10, weights=weights),
        })
    
    return metrics


def evaluate_model(model, X_test, y_test, weights_test=None, 
                   threshold_default=0.5, threshold_optimal=None,
                   model_name='Model'):
    """
    Full evaluation of a model with multiple thresholds.
    
    Returns:
        DataFrame with all metrics for both thresholds
    """
    # Get predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold if not provided
    if threshold_optimal is None:
        threshold_optimal, _ = find_optimal_threshold_youden(y_test, y_prob)
    
    results = []
    
    # Evaluate at default threshold (0.5)
    y_pred_default = (y_prob >= threshold_default).astype(int)
    metrics_default = compute_all_metrics(
        y_test, y_prob, y_pred_default, weights_test, 
        threshold_name=f'threshold_{threshold_default}'
    )
    metrics_default['model'] = model_name
    metrics_default['threshold_value'] = threshold_default
    results.append(metrics_default)
    
    # Evaluate at optimal threshold
    y_pred_optimal = (y_prob >= threshold_optimal).astype(int)
    metrics_optimal = compute_all_metrics(
        y_test, y_prob, y_pred_optimal, weights_test,
        threshold_name=f'threshold_optimal_{threshold_optimal:.3f}'
    )
    metrics_optimal['model'] = model_name
    metrics_optimal['threshold_value'] = threshold_optimal
    results.append(metrics_optimal)
    
    return pd.DataFrame(results)


def cross_validate_model(model, X, y, weights, cv, model_name='Model'):
    """
    Perform cross-validation with both unweighted and weighted metrics.
    
    Returns:
        Dict with CV results
    """
    from sklearn.base import clone
    
    cv_results = {
        'roc_auc_unweighted': [],
        'roc_auc_weighted': [],
        'accuracy_unweighted': [],
        'accuracy_weighted': [],
        'f1_unweighted': [],
        'f1_weighted': [],
    }
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    w_arr = np.array(weights)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_train_cv, X_val_cv = X_arr[train_idx], X_arr[val_idx]
        y_train_cv, y_val_cv = y_arr[train_idx], y_arr[val_idx]
        w_train_cv, w_val_cv = w_arr[train_idx], w_arr[val_idx]
        
        model_cv = clone(model)
        try:
            model_cv.fit(X_train_cv, y_train_cv, sample_weight=w_train_cv)
        except TypeError:
            model_cv.fit(X_train_cv, y_train_cv)
        
        y_prob_cv = model_cv.predict_proba(X_val_cv)[:, 1]
        y_pred_cv = model_cv.predict(X_val_cv)
        
        cv_results['roc_auc_unweighted'].append(roc_auc_score(y_val_cv, y_prob_cv))
        cv_results['accuracy_unweighted'].append(accuracy_score(y_val_cv, y_pred_cv))
        cv_results['f1_unweighted'].append(f1_score(y_val_cv, y_pred_cv, zero_division=0))

        cv_results['roc_auc_weighted'].append(weighted_roc_auc(y_val_cv, y_prob_cv, w_val_cv))
        cv_results['accuracy_weighted'].append(weighted_accuracy(y_val_cv, y_pred_cv, w_val_cv))
        _, _, f1_w = weighted_precision_recall_f1(y_val_cv, y_pred_cv, w_val_cv)
        cv_results['f1_weighted'].append(f1_w)
    
    summary = {'model': model_name}
    for metric, values in cv_results.items():
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)
    
    return summary


def subgroup_evaluation(y_true, y_pred, y_prob, weights, subgroup_var, subgroup_values,
                        threshold=0.5):
    """
    Compute metrics by subgroup for fairness analysis.
    
    Returns:
        DataFrame with metrics by subgroup
    """
    results = []
    
    for group_val in sorted(subgroup_values.unique()):
        mask = subgroup_values == group_val
        if mask.sum() < 30:
            continue
            
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        y_prob_g = y_prob[mask]
        weights_g = weights[mask]
        
        prec_w, rec_w, f1_w = weighted_precision_recall_f1(y_true_g, y_pred_g, weights_g)
        weighted_pos_rate = np.average(y_pred_g, weights=weights_g)
        cm = weighted_confusion_matrix(y_true_g, y_pred_g, weights_g)
        tpr = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else np.nan
        fpr = cm[0, 1] / cm[0, :].sum() if cm[0, :].sum() > 0 else np.nan
        
        results.append({
            'subgroup': group_val,
            'n_unweighted': mask.sum(),
            'weighted_positive_rate': weighted_pos_rate,
            'weighted_tpr': tpr,
            'weighted_fpr': fpr,
            'weighted_precision': prec_w,
            'weighted_recall': rec_w,
            'weighted_f1': f1_w,
            'mean_prob': np.average(y_prob_g, weights=weights_g),
        })
    
    return pd.DataFrame(results)


# =====================================================================
# CALIBRATION UPGRADE (Phase 5+)
# =====================================================================

def compute_adaptive_ece(y_true, y_prob, n_bins=10, weights=None):
    """
    Compute ECE with quantile-based bins (Nguyen & O'Connor, 2015).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of quantile bins.
        weights: Optional sample weights.

    Returns:
        Adaptive ECE value (float).
    """
    if weights is None:
        weights = np.ones(len(y_true))

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    weights = np.asarray(weights, dtype=float)

    bin_edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0 + 1e-10

    ece = 0.0
    total_weight = weights.sum()

    for i in range(len(bin_edges) - 1):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_w = weights[mask]
        bin_w_sum = bin_w.sum()
        mean_conf = np.average(y_prob[mask], weights=bin_w)
        mean_acc = np.average(y_true[mask], weights=bin_w)
        ece += (bin_w_sum / total_weight) * abs(mean_acc - mean_conf)

    return ece


def brier_decomposition(y_true, y_prob, n_bins=10, weights=None):
    """
    Murphy (1973) Brier decomposition: Uncertainty - Resolution + Reliability.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width bins.
        weights: Optional sample weights.

    Returns:
        Dict with keys: uncertainty, reliability, resolution, brier_decomposed.
    """
    if weights is None:
        weights = np.ones(len(y_true))

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total_weight = weights.sum()
    p_bar = np.average(y_true, weights=weights)

    uncertainty = p_bar * (1.0 - p_bar)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        if mask.sum() == 0:
            continue

        bin_w = weights[mask]
        bin_w_sum = bin_w.sum()
        o_k = np.average(y_true[mask], weights=bin_w)
        p_k = np.average(y_prob[mask], weights=bin_w)

        reliability += (bin_w_sum / total_weight) * (o_k - p_k) ** 2
        resolution += (bin_w_sum / total_weight) * (o_k - p_bar) ** 2

    return {
        "uncertainty": uncertainty,
        "reliability": reliability,
        "resolution": resolution,
        "brier_decomposed": uncertainty - resolution + reliability,
    }


def calibration_slope_intercept(y_true, y_prob, weights=None):
    """
    Calibration slope and intercept via logistic regression on logit(p_hat).
    Perfect calibration: slope=1, intercept=0.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        weights: Optional sample weights.

    Returns:
        Dict with keys: slope, intercept.
    """
    from scipy.special import logit as sp_logit

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    p_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit_p = sp_logit(p_clipped).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression as _LR
    cal_lr = _LR(penalty=None, max_iter=2000, solver="lbfgs")

    if weights is not None:
        cal_lr.fit(logit_p, y_true, sample_weight=np.asarray(weights))
    else:
        cal_lr.fit(logit_p, y_true)

    return {
        "slope": float(cal_lr.coef_[0][0]),
        "intercept": float(cal_lr.intercept_[0]),
    }
