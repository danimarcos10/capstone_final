"""
Interpretability Module for ATP W119 Analysis
SHAP values, coefficients, and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy import stats
import warnings


def get_logistic_coefficients(model, feature_names, scaler=None):
    """
    Args:
        model: Fitted LogisticRegression model
        feature_names: List of feature names
        scaler: Optional StandardScaler used for features
    Returns:
        DataFrame with feature, coefficient, odds_ratio
    """
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'odds_ratio': np.exp(model.coef_[0])
    })
    
    coef_df = coef_df.reindex(
        coef_df['coefficient'].abs().sort_values(ascending=False).index
    )
    
    return coef_df


def bootstrap_logistic_coefficients(X, y, weights, model_class, n_bootstrap=1000, 
                                     random_state=42, confidence=0.95):
    """
    Args:
        X: Feature matrix
        y: Target
        weights: Sample weights
        model_class: Unfitted model class (e.g., LogisticRegression)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        confidence: Confidence level for intervals
        
    Returns:
        DataFrame with coefficient estimates and CIs
    """
    np.random.seed(random_state)
    n_samples = len(y)
    n_features = X.shape[1]
    
    # Store bootstrap coefficients
    boot_coefs = np.zeros((n_bootstrap, n_features))
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    w_arr = np.array(weights)
    
    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_arr[idx]
        y_boot = y_arr[idx]
        w_boot = w_arr[idx]
        
        # Fit model
        model = model_class(max_iter=1000, random_state=random_state)
        try:
            model.fit(X_boot, y_boot, sample_weight=w_boot)
        except:
            model.fit(X_boot, y_boot)
        
        boot_coefs[b, :] = model.coef_[0]
    
    # Compute statistics
    alpha = 1 - confidence
    lower = np.percentile(boot_coefs, alpha/2 * 100, axis=0)
    upper = np.percentile(boot_coefs, (1 - alpha/2) * 100, axis=0)
    mean_coef = np.mean(boot_coefs, axis=0)
    std_coef = np.std(boot_coefs, axis=0)
    
    results = pd.DataFrame({
        'feature': X.columns if hasattr(X, 'columns') else [f'X{i}' for i in range(n_features)],
        'coefficient_mean': mean_coef,
        'coefficient_std': std_coef,
        'ci_lower': lower,
        'ci_upper': upper,
        'odds_ratio': np.exp(mean_coef),
        'or_ci_lower': np.exp(lower),
        'or_ci_upper': np.exp(upper),
    })
    
    return results


def compute_shap_values(model, X, n_samples=500, random_state=42):
    """
    Args:
        model: Fitted model
        X: Feature matrix
        n_samples: Number of samples to use for SHAP
        random_state: Random seed
    Returns:
        shap_values, explainer, X_sample
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None, None, None
    
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=random_state)
    else:
        X_sample = X.copy()
    
    model_type = type(model).__name__
    
    if 'GradientBoosting' in model_type or 'RandomForest' in model_type or 'XGB' in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Use KernelExplainer for other models
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 for binary
    
    return shap_values, explainer, X_sample


def shap_summary_table(shap_values, feature_names):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    summary['rank'] = range(1, len(summary) + 1)
    
    return summary


def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    perm_imp = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=random_state,
        scoring='roc_auc'
    )
    
    importance_df = pd.DataFrame({
        'feature': X.columns if hasattr(X, 'columns') else [f'X{i}' for i in range(X.shape[1])],
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


def compare_importance_methods(shap_importance, perm_importance):
    comparison = shap_importance[['feature', 'rank']].merge(
        perm_importance[['feature', 'rank']],
        on='feature',
        suffixes=('_shap', '_perm')
    )
    
    comparison['rank_diff'] = abs(comparison['rank_shap'] - comparison['rank_perm'])
    
    spearman_corr, p_value = stats.spearmanr(
        comparison['rank_shap'], 
        comparison['rank_perm']
    )
    
    return comparison, spearman_corr, p_value


def plot_shap_summary(shap_values, X_sample, save_path=None, show=True):
    try:
        import shap
    except ImportError:
        print("SHAP not installed")
        return
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP Feature Importance', fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_shap_dependence(shap_values, X_sample, feature_name, save_path=None, show=True):
    try:
        import shap
    except ImportError:
        print("SHAP not installed")
        return
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values, X_sample, show=False)
    plt.title(f'SHAP Dependence: {feature_name}', fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def interpret_coefficient(feature_name, coefficient, odds_ratio, ci_lower=None, ci_upper=None):
    direction = "increases" if coefficient > 0 else "decreases"
    or_text = f"{odds_ratio:.2f}"
    
    if ci_lower is not None and ci_upper is not None:
        ci_text = f" (95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
    else:
        ci_text = ""
    
    interpretation = (
        f"**{feature_name}**: A one-unit increase in this variable {direction} "
        f"the odds of applying by a factor of {or_text}{ci_text}. "
    )
    
    if odds_ratio > 1.5:
        interpretation += "This is a strong positive predictor."
    elif odds_ratio > 1.1:
        interpretation += "This is a moderate positive predictor."
    elif odds_ratio < 0.67:
        interpretation += "This is a strong negative predictor."
    elif odds_ratio < 0.9:
        interpretation += "This is a moderate negative predictor."
    else:
        interpretation += "The effect size is relatively small."
    
    return interpretation


def interpret_shap_feature(feature_name, shap_values, feature_values):
    correlation = np.corrcoef(feature_values, shap_values)[0, 1]
    
    if correlation > 0.3:
        direction = "Higher values of this feature increase"
    elif correlation < -0.3:
        direction = "Higher values of this feature decrease"
    else:
        direction = "This feature has a complex, possibly non-linear relationship with"
    
    mean_abs_shap = np.abs(shap_values).mean()
    
    interpretation = (
        f"**{feature_name}**: {direction} the predicted probability of applying. "
        f"The mean absolute SHAP value is {mean_abs_shap:.3f}. "
    )
    
    if abs(correlation) < 0.5 and mean_abs_shap > 0.05:
        interpretation += (
            "The weak correlation between feature values and SHAP values suggests "
            "possible non-linear effects or interactions."
        )
    
    return interpretation
