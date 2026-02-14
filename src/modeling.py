"""
Modeling module for ATP W119 analysis.

Weighted logistic regression, calibrated gradient boosting,
bootstrapped odds ratios, and model comparison utilities.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone

from src.config import cfg, SURVEY, get_output_dir
from src.evaluation import (
    compute_all_metrics,
    compute_ece,
    weighted_roc_auc,
    weighted_brier_score,
    find_optimal_threshold_youden,
)


# ===================================================================
# 1. WEIGHTED LOGISTIC REGRESSION
# ===================================================================

def fit_weighted_logistic(
    X_train, y_train, w_train,
    C: float = 1.0,
    class_weight=None,
    max_iter: int = 2000,
    seed: int | None = None,
) -> LogisticRegression:
    """
    Fit L2-regularized logistic regression with survey weights.

    Args:
        X_train: Training features.
        y_train: Training labels.
        w_train: Survey weights.
        C: Inverse regularization strength.
        class_weight: None or 'balanced'.
        max_iter: Maximum iterations.
        seed: Random seed.

    Returns:
        Fitted LogisticRegression model.
    """
    if seed is None:
        seed = cfg["modeling"]["random_seed"]

    model = LogisticRegression(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def extract_odds_ratios(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract odds ratios from a fitted logistic regression.
    Returns DataFrame sorted by absolute coefficient.
    """
    coefs = model.coef_[0]
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    })
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    df["rank"] = range(1, len(df) + 1)
    return df


def bootstrap_odds_ratios(
    X, y, weights,
    C: float = 1.0,
    class_weight=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Weighted bootstrap for logistic regression coefficients and odds ratios.

    Returns:
        DataFrame with coefficient and OR means, stds, and percentile CIs.
    """
    if seed is None:
        seed = cfg["modeling"]["random_seed"]
    rng = np.random.RandomState(seed)

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    w_arr = np.asarray(weights)
    n = len(y_arr)
    n_features = X_arr.shape[1]

    boot_coefs = np.zeros((n_bootstrap, n_features))

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        model = LogisticRegression(
            penalty="l2", C=C, class_weight=class_weight,
            max_iter=2000, random_state=seed, solver="lbfgs",
        )
        try:
            model.fit(X_arr[idx], y_arr[idx], sample_weight=w_arr[idx])
            boot_coefs[b] = model.coef_[0]
        except Exception:
            boot_coefs[b] = np.nan

    valid = ~np.isnan(boot_coefs).any(axis=1)
    boot_coefs = boot_coefs[valid]
    n_valid = valid.sum()

    alpha = 1 - confidence
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    feature_names = list(X.columns) if hasattr(X, "columns") else [f"X{i}" for i in range(n_features)]

    result = pd.DataFrame({
        "feature": feature_names,
        "coef_mean": boot_coefs.mean(axis=0),
        "coef_std": boot_coefs.std(axis=0),
        "coef_ci_lower": np.percentile(boot_coefs, lo, axis=0),
        "coef_ci_upper": np.percentile(boot_coefs, hi, axis=0),
        "or_mean": np.exp(boot_coefs.mean(axis=0)),
        "or_ci_lower": np.exp(np.percentile(boot_coefs, lo, axis=0)),
        "or_ci_upper": np.exp(np.percentile(boot_coefs, hi, axis=0)),
        "n_valid_boots": n_valid,
    })
    result["abs_coef"] = result["coef_mean"].abs()
    result = result.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    result["rank"] = range(1, len(result) + 1)
    return result


# ===================================================================
# 2. CALIBRATED GRADIENT BOOSTING
# ===================================================================

def fit_gradient_boosting(
    X_train, y_train, w_train,
    max_iter: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    min_samples_leaf: int = 50,
    seed: int | None = None,
) -> HistGradientBoostingClassifier:
    """
    Fit HistGradientBoosting with survey weights.
    """
    if seed is None:
        seed = cfg["modeling"]["random_seed"]

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def calibrate_model_prefit(
    fitted_model,
    X_cal, y_cal,
    method: str = "isotonic",
    sample_weight=None,
):
    """
    Post-hoc calibration on held-out set (cv='prefit'). Preserves AUC.

    Args:
        fitted_model: An already-fitted model with predict_proba.
        X_cal: Held-out calibration features.
        y_cal: Held-out calibration labels.
        method: 'isotonic' or 'sigmoid'.
        sample_weight: Optional survey weights for the calibration set.

    Returns:
        Calibrated model wrapper.
    """
    cal_model = CalibratedClassifierCV(
        estimator=fitted_model,
        method=method,
        cv="prefit",
    )
    if sample_weight is not None:
        cal_model.fit(X_cal, y_cal, sample_weight=sample_weight)
    else:
        cal_model.fit(X_cal, y_cal)
    return cal_model


# ===================================================================
# 3. MODEL EVALUATION ON TEST SET
# ===================================================================

def evaluate_on_test(
    model, X_test, y_test, w_test,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a model on the test set with comprehensive metrics.

    Returns dict with all metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_all_metrics(
        np.asarray(y_test), y_prob, y_pred,
        weights=np.asarray(w_test),
        threshold_name=f"threshold_{threshold}",
    )
    metrics["model"] = model_name
    metrics["threshold"] = threshold
    metrics["n_test"] = len(y_test)
    return metrics


def cv_evaluate(
    model_template,
    X, y, weights,
    n_folds: int = 5,
    model_name: str = "Model",
    seed: int | None = None,
    use_sample_weight: bool = True,
) -> dict:
    """
    Stratified K-fold CV evaluation with weighted metrics.

    Returns dict: metric means and stds.
    """
    if seed is None:
        seed = cfg["modeling"]["random_seed"]

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    w_arr = np.asarray(weights)

    fold_metrics = []

    for train_idx, val_idx in cv.split(X_arr, y_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
        w_tr, w_val = w_arr[train_idx], w_arr[val_idx]

        m = clone(model_template)
        if use_sample_weight:
            try:
                m.fit(X_tr, y_tr, sample_weight=w_tr)
            except TypeError:
                m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr, y_tr)

        y_prob = m.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_metrics.append(compute_all_metrics(y_val, y_prob, y_pred, w_val))

    summary = {"model": model_name}
    keys = fold_metrics[0].keys()
    for k in keys:
        if k == "threshold_scheme":
            continue
        vals = [fm[k] for fm in fold_metrics if isinstance(fm[k], (int, float))]
        if vals:
            summary[f"{k}_mean"] = np.mean(vals)
            summary[f"{k}_std"] = np.std(vals)

    return summary


# ===================================================================
# 4. MODEL COMPARISON TABLE
# ===================================================================

def model_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of test-set evaluation dicts.

    Returns tidy DataFrame with one row per model.
    """
    display_cols = [
        "model", "n_test", "threshold",
        "roc_auc_weighted", "pr_auc_weighted", "brier_weighted",
        "balanced_acc_weighted", "ece_weighted",
        "roc_auc_unweighted", "brier_unweighted", "ece_unweighted",
    ]
    rows = []
    for r in results:
        row = {k: r.get(k, None) for k in display_cols}
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == float:
            df[c] = df[c].round(4)
    return df


# ===================================================================
# 5. FIGURES
# ===================================================================

def plot_odds_ratios(
    boot_df: pd.DataFrame,
    top_n: int = 20,
    save_dir: Path | None = None,
    model_name: str = "logistic",
) -> None:
    """
    Forest plot of bootstrapped odds ratios with 95% CIs.
    """
    if save_dir is None:
        save_dir = get_output_dir("figures")

    df = boot_df.head(top_n).sort_values("or_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35)))

    y_pos = range(len(df))
    ax.errorbar(
        df["or_mean"], y_pos,
        xerr=[df["or_mean"] - df["or_ci_lower"], df["or_ci_upper"] - df["or_mean"]],
        fmt="o", color="#4C72B0", ecolor="#999999", capsize=3, markersize=5,
    )
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="OR = 1 (no effect)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values, fontsize=7)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title(f"Odds Ratios — {model_name}", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"odds_ratios_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_comparison(
    calibration_data: list[dict],
    save_dir: Path | None = None,
) -> None:
    """Plot reliability diagrams for multiple models on the same axes."""
    if save_dir is None:
        save_dir = get_output_dir("figures")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    colors = ["#4C72B0", "#D9534F", "#5CB85C", "#F0AD4E", "#9467BD"]
    for i, entry in enumerate(calibration_data):
        name = entry["model_name"]
        y_true = np.asarray(entry["y_true"])
        y_prob = np.asarray(entry["y_prob"])

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", color=colors[i % len(colors)], label=name, markersize=5)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (Reliability Diagram)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_comparison(
    roc_data: list[dict],
    save_dir: Path | None = None,
) -> None:
    """Plot ROC curves for multiple models."""
    from sklearn.metrics import roc_curve, auc

    if save_dir is None:
        save_dir = get_output_dir("figures")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

    colors = ["#4C72B0", "#D9534F", "#5CB85C", "#F0AD4E", "#9467BD"]
    for i, entry in enumerate(roc_data):
        name = entry["model_name"]
        y_true = np.asarray(entry["y_true"])
        y_prob = np.asarray(entry["y_prob"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], label=f"{name} (AUC={auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
