"""
Latent Attitude Construct Module
==================================
Ordinal factor analysis for AI-hiring attitudes (Phase 7).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import spearmanr
import warnings

from src.config import cfg, SURVEY, get_feature_set, get_feature_labels, get_not_sure_variables


# ===================================================================
# 1. ITEM SELECTION AND REVERSE CODING
# ===================================================================

def get_latent_item_columns() -> List[str]:
    """Return the 9 core attitude item column names from config."""
    return get_feature_set("core_attitudes")


def get_refined_latent_items(loading_threshold: float = 0.3,
                              fitted_model_dict: Optional[Dict] = None) -> List[str]:
    """
    Return items with |loading| >= threshold on the first factor.

    Args:
        loading_threshold: Minimum absolute loading to retain.
        fitted_model_dict: If provided, select items from this model's loadings.
            If None, returns default strong items.

    Returns:
        List of retained item column names.
    """
    all_items = get_latent_item_columns()
    
    if fitted_model_dict is not None:
        loadings = fitted_model_dict["loadings"][:, 0]
        item_names = fitted_model_dict["item_names"]
        return [name for name, lam in zip(item_names, loadings)
                if abs(lam) >= loading_threshold]
    
    # HIREBIAS1 (general bias concern) and AIWRKH1 (awareness, not attitude)
    # consistently load < 0.12 on the pro-AI factor across all runs.
    weak_items = {"HIREBIAS1_W119", "AIWRKH1_W119"}
    return [item for item in all_items if item not in weak_items]


def get_reverse_coded_items() -> List[str]:
    """
    Items reverse-coded so higher = more pro-AI.

    AIWRKH1: aware->unaware; AIWRKH2_a/b: oppose->favor;
    AIWRKH3_a/b/c/d: worse->better; HIREBIAS1: problem->not;
    HIREBIAS2: worse->better.
    """
    return [
        "AIWRKH1_W119",
        "AIWRKH2_a_W119", 
        "AIWRKH2_b_W119",
        "AIWRKH3_a_W119",
        "AIWRKH3_b_W119", 
        "AIWRKH3_c_W119",
        "AIWRKH3_d_W119",
        "HIREBIAS1_W119",
        "HIREBIAS2_W119",
    ]


def prepare_latent_items(
    df: pd.DataFrame,
    not_sure_treatment: str = "own_category",
    apply_reverse_coding: bool = True,
    skip_pattern_strategy: str = "eligible_only"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the 9 attitude items for latent modeling.

    Args:
        df: Raw survey DataFrame.
        not_sure_treatment: "own_category", "drop", or "midpoint".
        apply_reverse_coding: Reverse-code so higher = more pro-AI.
        skip_pattern_strategy: "eligible_only" or "impute_indicator".

    Returns:
        (items_df, eligible_mask)
    """
    from src.preprocessing import handle_not_sure, handle_refused_as_missing
    
    item_cols = get_latent_item_columns()
    df_prep = df[item_cols].copy()
    
    df_prep = pd.DataFrame(
        handle_not_sure(df_prep, treatment=not_sure_treatment),
        columns=item_cols
    )
    df_prep = handle_refused_as_missing(df_prep, features=item_cols)
    
    if apply_reverse_coding:
        reverse_items = get_reverse_coded_items()
        for col in reverse_items:
            if col in df_prep.columns:
                vals = df_prep[col].dropna().unique()
                if len(vals) > 0:
                    max_val = df_prep[col].max()
                    min_val = df_prep[col].min()
                    df_prep[col] = max_val + min_val - df_prep[col]
    
    # HIREBIAS2 only asked if HIREBIAS1 in {1, 2} (Major/Minor problem)
    if skip_pattern_strategy == "eligible_only":
        hirebias1_orig = df["HIREBIAS1_W119"]
        eligible_mask = (
            hirebias1_orig.isin([1.0, 2.0]) &
            df_prep["HIREBIAS2_W119"].notna()
        )
    else:
        eligible_mask = pd.Series(True, index=df.index)
    
    return df_prep, eligible_mask


def compute_ordinal_correlations(
    items_df: pd.DataFrame,
    method: str = "spearman",
    pairwise: bool = False
) -> np.ndarray:
    """
    Compute correlation matrix for ordinal items.

    Args:
        items_df: DataFrame with ordinal attitude items.
        method: "spearman" or "pearson".
        pairwise: If True, use pairwise-complete observations.

    Returns:
        Correlation matrix (n_items x n_items).
    """
    if pairwise:
        return _compute_pairwise_spearman(items_df)

    items_complete = items_df.dropna()
    
    if len(items_complete) == 0:
        raise ValueError("No complete cases for correlation computation")
    
    if method == "spearman":
        corr_matrix, _ = spearmanr(items_complete, nan_policy='omit')
        return corr_matrix
    elif method == "pearson":
        return items_complete.corr(method='pearson').values
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_pairwise_spearman(items_df: pd.DataFrame) -> np.ndarray:
    """
    Spearman correlations using pairwise-complete observations.

    Args:
        items_df: DataFrame with ordinal items (may contain NaN).

    Returns:
        Correlation matrix (n_items x n_items).
    """
    cols = items_df.columns
    k = len(cols)
    corr_matrix = np.eye(k)
    
    for i in range(k):
        for j in range(i + 1, k):
            mask = items_df[cols[i]].notna() & items_df[cols[j]].notna()
            if mask.sum() < 3:
                # Set to 0; _nearest_positive_definite() corrects the matrix downstream
                warnings.warn(
                    f"Pairwise N < 3 for ({cols[i]}, {cols[j]}); setting r = 0."
                )
                corr_matrix[i, j] = corr_matrix[j, i] = 0.0
            else:
                rho, _ = spearmanr(
                    items_df.loc[mask, cols[i]],
                    items_df.loc[mask, cols[j]]
                )
                corr_matrix[i, j] = corr_matrix[j, i] = rho
    
    return corr_matrix


def _pairwise_n_matrix(items_df: pd.DataFrame) -> np.ndarray:
    """
    Matrix of pairwise-complete observation counts.

    Args:
        items_df: DataFrame with ordinal items (may contain NaN).

    Returns:
        Matrix (n_items x n_items) of observation counts.
    """
    not_na = items_df.notna().values.astype(float)
    return not_na.T @ not_na


def compute_kmo_from_corr(corr_matrix: np.ndarray) -> float:
    """
    KMO sampling adequacy from a correlation matrix.

    KMO = sum(r_ij^2) / (sum(r_ij^2) + sum(q_ij^2)), q = partial correlations.

    Args:
        corr_matrix: Correlation matrix (p x p).

    Returns:
        KMO statistic (0 to 1).
    """
    try:
        R_inv = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(corr_matrix)
    
    d = np.diag(R_inv)
    d_safe = np.where(d > 0, d, 1e-10)
    partial = -R_inv / np.sqrt(np.outer(d_safe, d_safe))
    np.fill_diagonal(partial, 0.0)
    
    r2 = corr_matrix ** 2
    np.fill_diagonal(r2, 0.0)
    q2 = partial ** 2
    np.fill_diagonal(q2, 0.0)
    
    sum_r2 = r2.sum()
    sum_q2 = q2.sum()
    
    if (sum_r2 + sum_q2) == 0:
        return 0.0
    
    return sum_r2 / (sum_r2 + sum_q2)


def compute_bartlett_from_corr(
    corr_matrix: np.ndarray,
    n_obs: int
) -> float:
    """
    Bartlett's test of sphericity p-value from a correlation matrix.

    Args:
        corr_matrix: Correlation matrix (p x p).
        n_obs: Number of observations (average pairwise N for pairwise case).

    Returns:
        p-value.
    """
    from scipy.stats import chi2
    
    p = corr_matrix.shape[0]
    det_R = np.linalg.det(corr_matrix)
    
    if det_R <= 0:
        return 0.0
    
    chi_sq = -(n_obs - 1 - (2 * p + 5) / 6) * np.log(det_R)
    df = p * (p - 1) / 2
    
    return chi2.sf(chi_sq, df)


# ===================================================================
# 2. LATENT MODEL FITTING
# ===================================================================

def fit_latent_model(
    items_df: pd.DataFrame,
    n_factors: int = 1,
    method: str = "principal",
    rotation: Optional[str] = None,
    min_complete_cases: int = 100,
    use_pairwise: bool = False
) -> Dict:
    """
    Fit ordinal factor analysis model.

    Args:
        items_df: DataFrame with reverse-coded ordinal items.
        n_factors: Number of factors to extract.
        method: Extraction method ("principal", "minres", "ml").
        rotation: Rotation method (None, "varimax", "promax").
        min_complete_cases: Minimum N required.
        use_pairwise: Use pairwise-complete Spearman correlations.

    Returns:
        Dict with model, loadings, n_complete, item_names, kmo, bartlett_p,
        variance_explained.
    """
    if use_pairwise:
        return _fit_latent_pairwise(items_df, n_factors, method, rotation,
                                    min_complete_cases)

    items_complete = items_df.dropna()
    n_complete = len(items_complete)
    
    if n_complete < min_complete_cases:
        raise ValueError(
            f"Insufficient complete cases: {n_complete} < {min_complete_cases}"
        )
    
    kmo_all, kmo_model = calculate_kmo(items_complete)
    chi_square, p_value = calculate_bartlett_sphericity(items_complete)
    
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation,
        method=method,
        use_smc=True
    )
    fa.fit(items_complete)
    
    return {
        "model": fa,
        "loadings": fa.loadings_,
        "n_complete": n_complete,
        "item_names": list(items_df.columns),
        "kmo": kmo_model,
        "bartlett_p": p_value,
        "variance_explained": fa.get_factor_variance(),
    }


def _fit_latent_pairwise(
    items_df: pd.DataFrame,
    n_factors: int = 1,
    method: str = "principal",
    rotation: Optional[str] = None,
    min_complete_cases: int = 100
) -> Dict:
    """
    Fit factor analysis on pairwise-complete Spearman correlations.

    Args:
        items_df: DataFrame with ordinal items (may contain NaN).
        n_factors, method, rotation, min_complete_cases: See fit_latent_model.

    Returns:
        Same as fit_latent_model, plus pairwise N info and corr_matrix.
    """
    n_matrix = _pairwise_n_matrix(items_df)
    min_pairwise_n = int(n_matrix.min())
    avg_pairwise_n = int(n_matrix.mean())
    n_any_observed = int(items_df.notna().any(axis=1).sum())
    
    if min_pairwise_n < min_complete_cases:
        warnings.warn(
            f"Minimum pairwise N = {min_pairwise_n}; some pairs have "
            f"fewer than {min_complete_cases} observations."
        )
    
    corr_matrix = _compute_pairwise_spearman(items_df)
    corr_matrix = _nearest_positive_definite(corr_matrix)
    
    kmo_model = compute_kmo_from_corr(corr_matrix)
    bartlett_p = compute_bartlett_from_corr(corr_matrix, avg_pairwise_n)
    
    # "principal" requires raw data; fall back to minres for correlation input
    fa_method = "minres" if method == "principal" else method
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation,
        method=fa_method,
        use_smc=True,
        is_corr_matrix=True
    )
    fa.fit(corr_matrix)
    
    return {
        "model": fa,
        "loadings": fa.loadings_,
        "n_complete": avg_pairwise_n,
        "n_pairwise_min": min_pairwise_n,
        "n_any_observed": n_any_observed,
        "item_names": list(items_df.columns),
        "kmo": kmo_model,
        "bartlett_p": bartlett_p,
        "variance_explained": fa.get_factor_variance(),
        "corr_matrix": corr_matrix,
    }


def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """Nearest PD matrix via eigenvalue clipping. Needed for pairwise corr matrices."""
    eigvals, eigvecs = np.linalg.eigh(A)
    if np.all(eigvals > 0):
        return A
    
    eigvals = np.maximum(eigvals, 1e-8)
    A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    d = np.sqrt(np.diag(A_pd))
    A_pd = A_pd / np.outer(d, d)
    
    return A_pd


def transform_latent(
    fitted_model_dict: Dict,
    items_df: pd.DataFrame,
    impute_missing: bool = True,
    use_loading_weighted: bool = False
) -> pd.DataFrame:
    """
    Compute latent factor scores for new data.

    Args:
        fitted_model_dict: Dict from fit_latent_model().
        items_df: DataFrame with same columns as training data.
        impute_missing: Impute NaN with column medians before scoring.
        use_loading_weighted: Loading-weighted scoring that handles missingness
            without imputation.

    Returns:
        DataFrame with columns latent_factor_1, latent_factor_2, etc.
    """
    model = fitted_model_dict["model"]
    item_names = fitted_model_dict["item_names"]
    loadings = fitted_model_dict["loadings"]
    
    items_subset = items_df[item_names].copy()
    
    if use_loading_weighted:
        return _compute_loading_weighted_scores(items_subset, loadings)
    
    if impute_missing:
        for col in items_subset.columns:
            if items_subset[col].isna().any():
                median_val = items_subset[col].median()
                items_subset[col] = items_subset[col].fillna(median_val)
    
    try:
        scores = model.transform(items_subset)
    except ValueError as e:
        items_complete = items_subset.dropna()
        if len(items_complete) == 0:
            raise ValueError("Cannot transform: all rows have missing values")
        scores_partial = model.transform(items_complete)
        
        n_factors = scores_partial.shape[1]
        scores = np.full((len(items_subset), n_factors), np.nan)
        scores[items_subset.notna().all(axis=1)] = scores_partial
    
    score_df = pd.DataFrame(
        scores,
        index=items_df.index,
        columns=[f"latent_factor_{i+1}" for i in range(scores.shape[1])]
    )
    
    return score_df


def _compute_loading_weighted_scores(
    items_df: pd.DataFrame,
    loadings: np.ndarray
) -> pd.DataFrame:
    """
    Loading-weighted scores using only observed items per respondent.

    Score_i = sum(lambda_j * z_j) / sum(|lambda_j|)  for observed j

    Args:
        items_df: DataFrame with ordinal items (may have NaN).
        loadings: Factor loadings (n_items x n_factors).

    Returns:
        DataFrame with factor score columns.
    """
    n_factors = loadings.shape[1]
    n_rows = len(items_df)
    
    means = items_df.mean()
    stds = items_df.std()
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        warnings.warn(f"Constant item(s): {zero_std_cols}. Zero contribution to scores.")
    stds = stds.replace(0, 1)
    z = (items_df - means) / stds
    
    scores = np.full((n_rows, n_factors), np.nan)
    
    for f in range(n_factors):
        w = loadings[:, f]
        z_vals = z.values
        observed = ~np.isnan(z_vals)
        
        weighted_sum = np.nansum(z_vals * w[np.newaxis, :], axis=1)
        denom = (observed * np.abs(w)[np.newaxis, :]).sum(axis=1)
        
        valid = denom > 0
        scores[valid, f] = weighted_sum[valid] / denom[valid]
    
    return pd.DataFrame(
        scores,
        index=items_df.index,
        columns=[f"latent_factor_{i+1}" for i in range(n_factors)]
    )


# ===================================================================
# 3. FALLBACK: PCA ON ORDINAL ITEMS
# ===================================================================

def fit_pca_fallback(
    items_df: pd.DataFrame,
    n_components: int = 1
) -> Dict:
    """
    Fallback: PCA on ordinal-encoded items.

    Args:
        items_df: DataFrame with ordinal attitude items.
        n_components: Number of principal components.

    Returns:
        Dict with model, scaler, loadings, n_complete, item_names, explained_variance_ratio.
    """
    items_complete = items_df.dropna()
    n_complete = len(items_complete)
    
    scaler = StandardScaler()
    items_scaled = scaler.fit_transform(items_complete)
    
    pca = PCA(n_components=n_components, random_state=cfg["modeling"]["random_seed"])
    pca.fit(items_scaled)
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return {
        "model": pca,
        "scaler": scaler,
        "loadings": loadings,
        "n_complete": n_complete,
        "item_names": list(items_df.columns),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def transform_pca_fallback(
    fitted_model_dict: Dict,
    items_df: pd.DataFrame,
    impute_missing: bool = True
) -> pd.DataFrame:
    """
    Transform new data using fitted PCA model.

    Args:
        fitted_model_dict: Dict from fit_pca_fallback().
        items_df: DataFrame with same columns as training data.
        impute_missing: Impute NaN with column medians.

    Returns:
        DataFrame with PC score columns.
    """
    pca = fitted_model_dict["model"]
    scaler = fitted_model_dict["scaler"]
    item_names = fitted_model_dict["item_names"]
    
    items_subset = items_df[item_names].copy()
    
    if impute_missing:
        for col in items_subset.columns:
            if items_subset[col].isna().any():
                median_val = items_subset[col].median()
                items_subset[col] = items_subset[col].fillna(median_val)
    
    items_scaled = scaler.transform(items_subset)
    scores = pca.transform(items_scaled)
    
    score_df = pd.DataFrame(
        scores,
        index=items_df.index,
        columns=[f"pca_component_{i+1}" for i in range(scores.shape[1])]
    )
    
    return score_df


# ===================================================================
# 4. VALIDATION METRICS
# ===================================================================

def cronbach_alpha(items_df: pd.DataFrame) -> float:
    """
    Cronbach's alpha for internal consistency.

    Args:
        items_df: DataFrame with ordinal items.

    Returns:
        Alpha coefficient.
    """
    items_complete = items_df.dropna()
    if len(items_complete) < 10:
        return np.nan
    
    k = items_complete.shape[1]
    item_vars = items_complete.var(axis=0, ddof=1)
    total_var = items_complete.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    
    return alpha


def ordinal_alpha(items_df: pd.DataFrame) -> float:
    """
    Cronbach's alpha on the Spearman correlation matrix.

    alpha_ordinal = (k/(k-1)) * (1 - k/sum(R))

    Args:
        items_df: DataFrame with ordinal items (pairwise-complete).

    Returns:
        Ordinal alpha coefficient.
    """
    R = _compute_pairwise_spearman(items_df)
    k = R.shape[0]
    
    total_r = R.sum()
    if total_r == 0:
        return np.nan
    
    return (k / (k - 1)) * (1 - k / total_r)


def mcdonalds_omega(loadings: np.ndarray, items_df: pd.DataFrame) -> float:
    """
    McDonald's omega total from first-factor loadings.

    omega = (sum(lambda))^2 / ((sum(lambda))^2 + sum(1 - lambda^2))

    Args:
        loadings: Factor loadings (n_items x n_factors).
        items_df: DataFrame with ordinal items.

    Returns:
        Omega coefficient.
    """
    lam = loadings[:, 0]
    error_var = np.maximum(1.0 - lam ** 2, 0.0)
    
    sum_lam = lam.sum()
    omega = sum_lam ** 2 / (sum_lam ** 2 + error_var.sum())
    
    return omega


def item_total_correlations(items_df: pd.DataFrame) -> pd.Series:
    """
    Corrected item-total correlations (item vs total excluding that item).

    Args:
        items_df: DataFrame with ordinal items.

    Returns:
        Series of correlations per item.
    """
    items_complete = items_df.dropna()
    if len(items_complete) < 10:
        return pd.Series(np.nan, index=items_df.columns)
    
    correlations = {}
    for col in items_complete.columns:
        other_cols = [c for c in items_complete.columns if c != col]
        total_minus_item = items_complete[other_cols].sum(axis=1)
        corr = items_complete[col].corr(total_minus_item)
        correlations[col] = corr
    
    return pd.Series(correlations)


def validate_latent_model(
    items_df: pd.DataFrame,
    fitted_model_dict: Dict
) -> Dict:
    """
    Compute validation metrics for latent model.

    Args:
        items_df: DataFrame with ordinal items (may contain NaN).
        fitted_model_dict: Fitted model dictionary.

    Returns:
        Dict with cronbach_alpha, ordinal_alpha, omega, item_total_corr,
        avg/min/max_loading, n_items, n_complete_cases.
    """
    items_complete = items_df.dropna()
    
    alpha = cronbach_alpha(items_complete)
    ord_alpha = ordinal_alpha(items_df)
    
    loadings = fitted_model_dict["loadings"]
    omega = mcdonalds_omega(loadings, items_df)
    item_corr = item_total_correlations(items_complete)
    loadings_abs = np.abs(loadings[:, 0])
    
    return {
        "cronbach_alpha": alpha,
        "ordinal_alpha": ord_alpha,
        "omega": omega,
        "item_total_corr": item_corr,
        "avg_loading": loadings_abs.mean(),
        "min_loading": loadings_abs.min(),
        "max_loading": loadings_abs.max(),
        "n_items": len(items_df.columns),
        "n_complete_cases": len(items_complete),
    }


# ===================================================================
# 5. REPORTING UTILITIES
# ===================================================================

def print_latent_model_summary(
    fitted_model_dict: Dict,
    validation_dict: Dict
):
    """Print a summary of the fitted latent model."""
    print("=" * 70)
    print("LATENT ATTITUDE MODEL SUMMARY")
    print("=" * 70)
    
    print(f"\nSample:")
    print(f"  N complete cases: {fitted_model_dict['n_complete']:,}")
    
    print(f"\nModel:")
    print(f"  N factors: {fitted_model_dict['loadings'].shape[1]}")
    print(f"  N items: {fitted_model_dict['loadings'].shape[0]}")
    
    print(f"\nFit statistics:")
    print(f"  KMO: {fitted_model_dict['kmo']:.3f}")
    print(f"  Bartlett's test p-value: {fitted_model_dict['bartlett_p']:.2e}")
    
    if "variance_explained" in fitted_model_dict:
        var_explained = fitted_model_dict["variance_explained"]
        print(f"\nVariance explained:")
        if hasattr(var_explained, 'shape') and len(var_explained.shape) == 2:
            for i in range(var_explained.shape[1]):
                print(f"  Factor {i+1}: {var_explained[1, i]:.1%}")
        else:
            print(f"  (not available)")
    
    print(f"\nInternal consistency:")
    print(f"  Cronbach's alpha: {validation_dict['cronbach_alpha']:.3f}")
    if "ordinal_alpha" in validation_dict:
        print(f"  Ordinal alpha (pairwise Spearman): {validation_dict['ordinal_alpha']:.3f}")
    if "omega" in validation_dict:
        print(f"  McDonald's omega: {validation_dict['omega']:.3f}")
    
    print(f"\nFactor loadings (Factor 1):")
    loadings = fitted_model_dict["loadings"][:, 0]
    item_names = fitted_model_dict["item_names"]
    
    sorted_idx = np.argsort(np.abs(loadings))[::-1]
    for idx in sorted_idx:
        print(f"  {item_names[idx]:30s}: {loadings[idx]:6.3f}")
    
    print(f"\nLoading statistics:")
    print(f"  Mean |loading|: {validation_dict['avg_loading']:.3f}")
    print(f"  Min |loading|:  {validation_dict['min_loading']:.3f}")
    print(f"  Max |loading|:  {validation_dict['max_loading']:.3f}")
    
    print("\n" + "=" * 70)


def create_loadings_table(fitted_model_dict: Dict) -> pd.DataFrame:
    """
    Create a DataFrame of factor loadings for export.

    Args:
        fitted_model_dict: Output from fit_latent_model().

    Returns:
        DataFrame with item, readable_name, factor loadings, abs_loading_f1.
    """
    loadings = fitted_model_dict["loadings"]
    item_names = fitted_model_dict["item_names"]
    
    labels = get_feature_labels()
    readable_names = [labels.get(item, item) for item in item_names]
    
    df = pd.DataFrame(
        loadings,
        index=item_names,
        columns=[f"factor_{i+1}" for i in range(loadings.shape[1])]
    )
    df.insert(0, "item", item_names)
    df.insert(1, "readable_name", readable_names)
    df["abs_loading_f1"] = np.abs(df["factor_1"])
    
    df = df.sort_values("abs_loading_f1", ascending=False)
    
    return df


# ===================================================================
# 6. PHASE 7.1: NOT SURE AS SEPARATE INDICATORS
# ===================================================================

def create_not_sure_indicators(
    df: pd.DataFrame,
    item_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Binary indicators for "Not sure" (code 9) responses, excluding them from FA.

    Args:
        df: Raw survey DataFrame.
        item_cols: Columns to process (default: latent item columns).

    Returns:
        DataFrame with is_not_sure_{col} columns.
    """
    if item_cols is None:
        item_cols = get_latent_item_columns()
    
    not_sure_vars = get_not_sure_variables()
    indicators = pd.DataFrame(index=df.index)
    
    for col in item_cols:
        if col in not_sure_vars:
            indicator_col = f"is_not_sure_{col}"
            indicators[indicator_col] = (df[col] == SURVEY.NOT_SURE_CODE).astype(float)
            indicators.loc[df[col].isna() | (df[col] == SURVEY.REFUSED_CODE), indicator_col] = np.nan
    
    return indicators


def prepare_latent_items_v2(
    df: pd.DataFrame,
    apply_reverse_coding: bool = True,
    skip_pattern_strategy: str = "eligible_only",
    create_indicators: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    Prepare items with "Not sure" as NaN + separate binary indicators.

    Args:
        df: Raw survey DataFrame.
        apply_reverse_coding: Reverse-code so higher = more pro-AI.
        skip_pattern_strategy: "eligible_only" or "impute_indicator".
        create_indicators: Create not_sure indicator columns.

    Returns:
        (items_df, eligible_mask, indicators_df)
    """
    from src.preprocessing import handle_refused_as_missing
    
    item_cols = get_latent_item_columns()
    
    indicators_df = None
    if create_indicators:
        indicators_df = create_not_sure_indicators(df, item_cols)
    
    df_prep = df[item_cols].copy()
    
    # Code 9 -> NaN for factor analysis
    not_sure_vars = get_not_sure_variables()
    for col in df_prep.columns:
        if col in not_sure_vars:
            df_prep[col] = df_prep[col].replace(SURVEY.NOT_SURE_CODE, np.nan)
    
    df_prep = handle_refused_as_missing(df_prep, features=item_cols)
    
    if apply_reverse_coding:
        reverse_items = get_reverse_coded_items()
        for col in reverse_items:
            if col in df_prep.columns:
                vals = df_prep[col].dropna().unique()
                if len(vals) > 0:
                    max_val = df_prep[col].max()
                    min_val = df_prep[col].min()
                    df_prep[col] = max_val + min_val - df_prep[col]
    
    if skip_pattern_strategy == "eligible_only":
        hirebias1_orig = df["HIREBIAS1_W119"]
        eligible_mask = (
            hirebias1_orig.isin([1.0, 2.0]) &
            df_prep["HIREBIAS2_W119"].notna()
        )
    else:
        eligible_mask = pd.Series(True, index=df.index)
    
    return df_prep, eligible_mask, indicators_df


def create_not_sure_count(
    df: pd.DataFrame,
    item_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Count of "Not sure" responses across latent items per respondent.

    Args:
        df: Raw survey DataFrame.
        item_cols: Columns to check (default: latent item columns).

    Returns:
        DataFrame with 'not_sure_count' column.
    """
    if item_cols is None:
        item_cols = get_latent_item_columns()
    
    not_sure_vars = get_not_sure_variables()
    count = pd.Series(0.0, index=df.index)
    
    for col in item_cols:
        if col in not_sure_vars and col in df.columns:
            count += (df[col] == SURVEY.NOT_SURE_CODE).astype(float)
    
    return pd.DataFrame({"not_sure_count": count}, index=df.index)


def compute_eligibility_bias(
    df_full: pd.DataFrame,
    items_df: pd.DataFrame,
    weight_col: str = "WEIGHT_W119",
    demo_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Weighted demographic comparison: factor-eligible vs full sample.

    Args:
        df_full: Full training DataFrame with demographics and weights.
        items_df: Items DataFrame used for factor estimation.
        weight_col: Survey weight column.
        demo_cols: Demographics to check (default: age, gender, education).

    Returns:
        DataFrame with weighted proportions for full, any-observed, and complete-case samples.
    """
    if demo_cols is None:
        demo_cols = ["F_AGECAT", "F_GENDER", "F_EDUCCAT2"]
    
    has_any = items_df.notna().any(axis=1)
    has_all = items_df.notna().all(axis=1)
    
    rows = []
    for col in demo_cols:
        if col not in df_full.columns:
            continue
        vals = df_full[col].dropna().unique()
        for v in sorted(vals):
            mask_v = df_full[col] == v
            
            w_full = df_full.loc[mask_v, weight_col].sum() / df_full[weight_col].sum()
            
            w_any = (
                df_full.loc[mask_v & has_any, weight_col].sum() /
                df_full.loc[has_any, weight_col].sum()
            ) if has_any.sum() > 0 else np.nan
            
            w_complete = (
                df_full.loc[mask_v & has_all, weight_col].sum() /
                df_full.loc[has_all, weight_col].sum()
            ) if has_all.sum() > 0 else np.nan
            
            rows.append({
                "variable": col,
                "value": v,
                "full_sample_pct": round(w_full * 100, 1),
                "any_observed_pct": round(w_any * 100, 1),
                "complete_case_pct": round(w_complete * 100, 1),
                "diff_complete_vs_full": round((w_complete - w_full) * 100, 1),
            })
    
    return pd.DataFrame(rows)


# ===================================================================
# 7. POLYCHORIC CORRELATIONS
# ===================================================================

def _polychoric_pair(x: np.ndarray, y: np.ndarray) -> float:
    """
    Polychoric correlation for two ordinal variables via two-step MLE (Olsson, 1979).
    Falls back to Spearman if MLE fails or N < 10.

    Args:
        x, y: 1-D arrays of ordinal values (NaN-free).

    Returns:
        Estimated polychoric correlation.
    """
    from scipy.stats import norm, mvn as _mvn
    from scipy.optimize import minimize_scalar

    n = len(x)
    if n < 10:
        if n < 3:
            return 0.0
        rho, _ = spearmanr(x, y)
        return float(rho)

    cats_x = np.sort(np.unique(x))
    cats_y = np.sort(np.unique(y))

    if len(cats_x) < 2 or len(cats_y) < 2:
        return 0.0

    thresh_x = [float(norm.ppf(np.mean(x <= c))) for c in cats_x[:-1]]
    thresh_y = [float(norm.ppf(np.mean(y <= c))) for c in cats_y[:-1]]
    thresh_x = [-8.0] + thresh_x + [8.0]
    thresh_y = [-8.0] + thresh_y + [8.0]

    obs = np.zeros((len(cats_x), len(cats_y)))
    for i, cx in enumerate(cats_x):
        for j, cy in enumerate(cats_y):
            obs[i, j] = np.sum((x == cx) & (y == cy))

    mean = np.array([0.0, 0.0])

    def _neg_ll(rho):
        if abs(rho) >= 0.999:
            return 1e10
        cov = np.array([[1.0, rho], [rho, 1.0]])
        ll = 0.0
        for i in range(len(cats_x)):
            for j in range(len(cats_y)):
                if obs[i, j] == 0:
                    continue
                lower = np.array([thresh_x[i], thresh_y[j]])
                upper = np.array([thresh_x[i + 1], thresh_y[j + 1]])
                p, _ = _mvn.mvnun(lower, upper, mean, cov)
                p = max(p, 1e-15)
                ll += obs[i, j] * np.log(p)
        return -ll

    try:
        result = minimize_scalar(
            _neg_ll, bounds=(-0.999, 0.999),
            method="bounded", options={"xatol": 1e-6},
        )
        return float(result.x)
    except Exception:
        warnings.warn("Polychoric MLE failed; falling back to Spearman.")
        rho, _ = spearmanr(x, y)
        return float(rho)


def compute_polychoric_matrix(
    items_df: pd.DataFrame,
    pairwise: bool = True
) -> np.ndarray:
    """
    Polychoric correlation matrix, ensured positive-definite.

    Args:
        items_df: DataFrame with ordinal items (may contain NaN).
        pairwise: Use pairwise-complete observations.

    Returns:
        Polychoric correlation matrix (n_items x n_items).
    """
    cols = items_df.columns
    k = len(cols)
    corr = np.eye(k)

    for i in range(k):
        for j in range(i + 1, k):
            xi = items_df[cols[i]].values
            xj = items_df[cols[j]].values

            if pairwise:
                mask = ~(np.isnan(xi) | np.isnan(xj))
                if mask.sum() < 3:
                    warnings.warn(
                        f"Polychoric: pairwise N < 3 for ({cols[i]}, {cols[j]}); r=0."
                    )
                    corr[i, j] = corr[j, i] = 0.0
                    continue
                rho = _polychoric_pair(xi[mask], xj[mask])
            else:
                complete = items_df.dropna()
                rho = _polychoric_pair(
                    complete[cols[i]].values, complete[cols[j]].values
                )

            corr[i, j] = corr[j, i] = rho

    corr = _nearest_positive_definite(corr)
    return corr
