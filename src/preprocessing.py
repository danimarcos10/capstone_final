"""
Preprocessing module for ATP W119 analysis.
Feature engineering, missing-data handling, encoding, and skip-pattern-aware data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import cfg, SURVEY, get_feature_set, get_feature_labels, get_not_sure_variables


# === 1. "NOT SURE" (CODE 9) HANDLING ===

def handle_not_sure(df: pd.DataFrame,
                    treatment: str | None = None) -> pd.DataFrame:
    """
    Handle "Not sure" (code 9) in variables that have it.

    Args:
        df: DataFrame.
        treatment: "drop" = replace 9 with NaN.
                   "own_category" = keep 9 as-is (distinct level).
                   "midpoint" = replace 9 with scale midpoint.
                   None = use default from config.

    Returns:
        Modified DataFrame (copy).
    """
    if treatment is None:
        treatment = cfg["not_sure"]["default_treatment"]

    ns_vars = get_not_sure_variables()
    df_out = df.copy()

    for col in ns_vars:
        if col not in df_out.columns:
            continue

        if treatment == "drop":
            df_out[col] = df_out[col].replace(SURVEY.NOT_SURE_CODE, np.nan)

        elif treatment == "own_category":
            pass  # keep 9 as-is

        elif treatment == "midpoint":
            vals = df_out[col].dropna()
            substantive = vals[
                (vals != SURVEY.NOT_SURE_CODE) & (vals != SURVEY.REFUSED_CODE)
            ]
            if len(substantive) > 0:
                midpoint = substantive.median()
                df_out[col] = df_out[col].replace(SURVEY.NOT_SURE_CODE, midpoint)

        else:
            raise ValueError(
                f"Unknown treatment '{treatment}'. Use: drop, own_category, midpoint."
            )

    return df_out


# === 2. REFUSED (CODE 99) HANDLING ===

def handle_refused_as_missing(df: pd.DataFrame,
                              features: list[str]) -> pd.DataFrame:
    """Replace code 99 (Refused) with NaN in feature columns."""
    df_out = df.copy()
    for col in features:
        if col in df_out.columns:
            df_out[col] = df_out[col].replace(SURVEY.REFUSED_CODE, np.nan)
    return df_out


def handle_refused_as_category(df: pd.DataFrame,
                               features: list[str]) -> pd.DataFrame:
    """Keep code 99 as a valid category (robustness check)."""
    return df.copy()


# === 3. MISSING INDICATOR COLUMNS ===

def add_missing_indicators(df: pd.DataFrame,
                           features: list[str]) -> pd.DataFrame:
    """
    Add binary {feature}_missing indicators.
    Call AFTER handling refused/not-sure but BEFORE imputation.
    """
    df_out = df.copy()
    for col in features:
        if col in df_out.columns and df_out[col].isna().any():
            df_out[f"{col}_missing"] = df_out[col].isna().astype(int)
    return df_out


# === 4. SIMPLE IMPUTATION ===

def simple_impute(df: pd.DataFrame,
                  features: list[str]) -> pd.DataFrame:
    """
    Mode for categoricals/ordinals, median for numeric.
    Call AFTER add_missing_indicators.
    """
    df_out = df.copy()
    numeric_vars = cfg.get("encoding", {}).get("numeric", [])

    for col in features:
        if col not in df_out.columns:
            continue
        if not df_out[col].isna().any():
            continue

        if col in numeric_vars:
            fill_val = df_out[col].median()
        else:
            modes = df_out[col].mode()
            fill_val = modes.iloc[0] if len(modes) > 0 else 0
        df_out[col] = df_out[col].fillna(fill_val)

    return df_out


# === 5. MISSINGNESS ANALYSIS ===

def compute_missingness_rates(df: pd.DataFrame,
                              features: list[str],
                              weight_col: str | None = None) -> pd.DataFrame:
    """
    Compute per-variable missingness rates, decomposed into
    "Not sure" (code 9), "Refused" (code 99), and structural NaN.

    Returns DataFrame with columns: variable, n_total, n_not_sure,
    pct_not_sure, n_refused, pct_refused, n_structural_nan, pct_structural_nan,
    n_any_missing, pct_any_missing, weighted_pct_any_missing.
    """
    if weight_col is None:
        weight_col = SURVEY.WEIGHT_VAR

    ns_vars = set(get_not_sure_variables())
    rows = []

    for col in features:
        if col not in df.columns:
            continue

        n_total = len(df)
        vals = df[col]

        # Code 9 only applies to variables that offer "Not sure"
        n_ns = (vals == SURVEY.NOT_SURE_CODE).sum() if col in ns_vars else 0
        n_ref = (vals == SURVEY.REFUSED_CODE).sum()
        n_nan = vals.isna().sum()

        # Any missing = Not sure + Refused + structural NaN
        n_any = n_ns + n_ref + n_nan

        if weight_col in df.columns:
            mask_any = vals.isna() | (vals == SURVEY.REFUSED_CODE)
            if col in ns_vars:
                mask_any = mask_any | (vals == SURVEY.NOT_SURE_CODE)
            w_miss = df.loc[mask_any, weight_col].sum()
            w_total = df[weight_col].sum()
            wpct_any = (w_miss / w_total * 100) if w_total > 0 else 0
        else:
            wpct_any = n_any / n_total * 100

        rows.append({
            "variable": col,
            "n_total": n_total,
            "n_not_sure": n_ns,
            "pct_not_sure": round(n_ns / n_total * 100, 2),
            "n_refused": n_ref,
            "pct_refused": round(n_ref / n_total * 100, 2),
            "n_structural_nan": n_nan,
            "pct_structural_nan": round(n_nan / n_total * 100, 2),
            "n_any_missing": n_any,
            "pct_any_missing": round(n_any / n_total * 100, 2),
            "weighted_pct_any_missing": round(wpct_any, 2),
        })

    return pd.DataFrame(rows).sort_values("pct_any_missing", ascending=False)


def missingness_model(df: pd.DataFrame,
                      features: list[str],
                      demo_features: list[str] | None = None,
                      cv_folds: int = 5) -> dict:
    """
    Diagnose whether missingness is associated with observed demographics
    via logistic regression with stratified K-fold CV (ROC-AUC).

    This is a diagnostic for MCAR vs association with observed covariates,
    not a formal MCAR test or proof of MAR.

    Args:
        df: DataFrame with raw codes already processed.
        features: Features to check for missingness.
        demo_features: Demographic predictors (default from config).
        cv_folds: Number of CV folds.

    Returns:
        Dict with model, cv_auc_mean/std, train_accuracy,
        coefficients, n_missing, n_complete, conclusion.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if demo_features is None:
        demo_features = get_feature_set("demographics")

    available = [f for f in features if f in df.columns]
    is_missing = df[available].isna().any(axis=1).astype(int)

    demo_avail = [d for d in demo_features if d in df.columns]
    X_demo = df[demo_avail].copy()

    for col in demo_avail:
        X_demo[col] = X_demo[col].replace(SURVEY.REFUSED_CODE, np.nan)
    valid_mask = X_demo.notna().all(axis=1)

    X_demo = X_demo[valid_mask]
    y_miss = is_missing[valid_mask]

    n_missing = int(y_miss.sum())
    n_complete = int((y_miss == 0).sum())

    if n_missing < 30 or n_complete < 30:
        return {
            "model": None,
            "cv_auc_mean": None,
            "cv_auc_std": None,
            "train_accuracy": None,
            "coefficients": None,
            "n_missing": n_missing,
            "n_complete": n_complete,
            "conclusion": "Too few missing or complete cases for missingness model.",
        }

    seed = cfg["modeling"]["random_seed"]
    model = LogisticRegression(max_iter=1000, random_state=seed)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    auc_scores = cross_val_score(model, X_demo, y_miss, cv=cv, scoring="roc_auc")
    cv_auc_mean = float(auc_scores.mean())
    cv_auc_std = float(auc_scores.std())

    model.fit(X_demo, y_miss)
    train_acc = model.score(X_demo, y_miss)

    coefs = pd.DataFrame({
        "feature": demo_avail,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("coefficient", key=abs, ascending=False)

    if cv_auc_mean > 0.60:
        conclusion = (
            f"Missingness is associated with observed demographics "
            f"(5-fold CV AUC = {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}). "
            f"This is evidence against MCAR and supports using imputation "
            f"with sensitivity checks rather than listwise deletion."
        )
    else:
        conclusion = (
            f"Demographics are weak predictors of missingness "
            f"(5-fold CV AUC = {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}). "
            f"No strong evidence against MCAR; listwise deletion is defensible, "
            f"though imputation remains a reasonable alternative."
        )

    return {
        "model": model,
        "cv_auc_mean": round(cv_auc_mean, 4),
        "cv_auc_std": round(cv_auc_std, 4),
        "train_accuracy": round(train_acc, 4),
        "coefficients": coefs,
        "n_missing": n_missing,
        "n_complete": n_complete,
        "conclusion": conclusion,
    }


# === 6. ORDINAL / NOMINAL ENCODING ===

def encode_ordinal(df: pd.DataFrame,
                   col: str,
                   order: list[float]) -> pd.Series:
    """
    Encode ordinal variable as integer ranks (0, 1, 2, ...).
    Values not in `order` are left as NaN.

    Args:
        df: DataFrame.
        col: Column name.
        order: Raw codes in ascending ordinal order.

    Returns:
        Series of integer ranks.
    """
    mapping = {code: rank for rank, code in enumerate(order)}
    return df[col].map(mapping)


def encode_all_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordinal encoding to all variables specified in config."""
    ordinal_spec = cfg.get("encoding", {}).get("ordinal", {})
    df_out = df.copy()

    for col, order in ordinal_spec.items():
        if col in df_out.columns:
            df_out[col] = encode_ordinal(df_out, col, [float(x) for x in order])

    return df_out


def one_hot_encode(df: pd.DataFrame,
                   columns: list[str] | None = None,
                   drop_first: bool = False) -> pd.DataFrame:
    """
    One-hot encode nominal categorical variables.

    Args:
        df: DataFrame.
        columns: Columns to encode (default from config nominal list).
        drop_first: Drop first category to avoid multicollinearity.

    Returns:
        DataFrame with one-hot columns replacing originals.
    """
    if columns is None:
        columns = cfg.get("encoding", {}).get("nominal", [])

    cols_present = [c for c in columns if c in df.columns]
    if not cols_present:
        return df.copy()

    label_map = get_feature_labels()

    df_out = df.copy()
    for col in cols_present:
        dummies = pd.get_dummies(
            df_out[col], prefix=label_map.get(col, col), drop_first=drop_first
        )
        dummies = dummies.astype(int)
        df_out = pd.concat([df_out.drop(columns=[col]), dummies], axis=1)

    return df_out


# === 7. SKIP-PATTERN HANDLING ===

def get_skip_pattern_info() -> dict:
    """Return skip pattern definitions from config."""
    return cfg.get("skip_patterns", {})


def handle_skip_patterns(df: pd.DataFrame,
                         features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Set values to NaN for respondents ineligible due to skip patterns.
    Structural NaN (skip) must not be conflated with item non-response.

    Returns:
        df_out: DataFrame with ineligible values NaN'd.
        notes: Log of actions taken.
    """
    skip_info = get_skip_pattern_info()
    df_out = df.copy()
    notes = []

    for var in features:
        if var not in skip_info or var not in df_out.columns:
            continue

        spec = skip_info[var]
        filt_var = spec["filter_var"]
        filt_vals = [float(v) for v in spec["filter_values"]]

        if filt_var not in df_out.columns:
            continue

        # Ineligible respondents were correctly not asked this item
        ineligible = ~df_out[filt_var].isin(filt_vals)
        n_ineligible = ineligible.sum()
        df_out.loc[ineligible, var] = np.nan

        notes.append(
            f"  {var}: {n_ineligible:,} ineligible (filter: {filt_var} in {filt_vals})"
        )

    return df_out, notes


# === 8. FULL PREPROCESSING PIPELINE ===

def prepare_modeling_data(
    df: pd.DataFrame,
    feature_set: str = "full",
    target_col: str | None = None,
    weight_col: str | None = None,
    refused_strategy: str = "drop",
    not_sure_treatment: str | None = None,
    missingness_regime: str = "listwise",
    apply_encoding: bool = True,
    apply_one_hot: bool = True,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, list[str]]:
    """
    End-to-end preparation of modeling data.

    Pipeline: Not sure (9) -> Refused (99) -> skip patterns ->
    drop missing target -> ordinal encoding -> missingness regime ->
    one-hot encoding -> readable names.

    Args:
        df: Raw DataFrame with y_apply column.
        feature_set: Feature set name from config, or "full".
        target_col: Target column (default from config).
        weight_col: Weight column (default from config).
        refused_strategy: "drop" (99->NaN) or "keep" (99 as category).
        not_sure_treatment: Treatment for code 9.
        missingness_regime: "listwise" or "impute_indicator".
        apply_encoding: Apply ordinal encoding.
        apply_one_hot: One-hot encode nominals.
        drop_first: Drop first dummy (for regression).

    Returns:
        X, y, weights, feature_names.
    """
    if target_col is None:
        target_col = SURVEY.TARGET_DERIVED
    if weight_col is None:
        weight_col = SURVEY.WEIGHT_VAR

    raw_features = get_feature_set(feature_set)
    available = [f for f in raw_features if f in df.columns]

    if len(available) < len(raw_features):
        missing_cols = set(raw_features) - set(available)
        print(f"[WARN] {len(missing_cols)} features not in data: {missing_cols}")

    cols_needed = list(set(available + [target_col, weight_col]))
    # Include filter variables needed for skip patterns
    skip_info = get_skip_pattern_info()
    for var in available:
        if var in skip_info:
            fv = skip_info[var]["filter_var"]
            if fv not in cols_needed and fv in df.columns:
                cols_needed.append(fv)

    df_sub = df[[c for c in cols_needed if c in df.columns]].copy()

    df_sub = handle_not_sure(df_sub, treatment=not_sure_treatment)

    if refused_strategy == "drop":
        df_sub = handle_refused_as_missing(df_sub, available)
    elif refused_strategy == "keep":
        df_sub = handle_refused_as_category(df_sub, available)

    df_sub, skip_notes = handle_skip_patterns(df_sub, available)
    if skip_notes:
        print(f"Skip patterns applied:")
        for note in skip_notes:
            print(note)

    df_sub = df_sub[df_sub[target_col].notna()]
    n_after_target = len(df_sub)

    # Ordinal encoding before missingness so NaN stays NaN
    if apply_encoding:
        df_sub = encode_all_ordinals(df_sub)

    if missingness_regime == "listwise":
        df_ready = df_sub.dropna(subset=available)
    elif missingness_regime == "impute_indicator":
        df_ready = add_missing_indicators(df_sub, available)
        df_ready = simple_impute(df_ready, available)
        indicator_cols = [c for c in df_ready.columns if c.endswith("_missing")]
        available = available + indicator_cols
    else:
        raise ValueError(f"Unknown regime '{missingness_regime}'.")

    print(f"Starting rows: {len(df):,}")
    print(f"After target filter: {n_after_target:,}")
    print(f"After missingness ({missingness_regime}): {len(df_ready):,}")

    nominal_vars = cfg.get("encoding", {}).get("nominal", [])
    nominal_in_features = [v for v in nominal_vars if v in available]

    if apply_one_hot and nominal_in_features:
        label_map = get_feature_labels()
        rename_nominal = {c: label_map.get(c, c) for c in nominal_in_features}
        df_ready = df_ready.rename(columns=rename_nominal)
        renamed_nominal = [rename_nominal.get(c, c) for c in nominal_in_features]

        for col in renamed_nominal:
            if col in df_ready.columns:
                dummies = pd.get_dummies(
                    df_ready[col], prefix=col, drop_first=drop_first
                ).astype(int)
                df_ready = pd.concat(
                    [df_ready.drop(columns=[col]), dummies], axis=1
                )
        # Replace original nominals with dummy columns
        non_nominal = [c for c in available if c not in nominal_in_features]
        dummy_cols = [c for c in df_ready.columns
                      if any(c.startswith(rename_nominal.get(n, n))
                             for n in nominal_in_features)]
        available = non_nominal + dummy_cols

    label_map = get_feature_labels()
    rename = {}
    for c in available:
        if c in label_map:
            rename[c] = label_map[c]
    df_ready = df_ready.rename(columns=rename)
    available = [rename.get(c, c) for c in available]

    available = [c for c in available if c in df_ready.columns]

    X = df_ready[available].copy()
    y = df_ready[target_col].astype(int)
    weights = df_ready[weight_col].values
    feature_names = list(X.columns)

    print(f"Final feature count: {len(feature_names)}")

    return X, y, weights, feature_names


# === 9. TRAIN/TEST SPLIT & CV ===

def create_train_test_split(X, y, weights,
                            test_size: float | None = None,
                            random_state: int | None = None):
    """Stratified train/test split respecting config."""
    if test_size is None:
        test_size = cfg["modeling"]["test_size"]
    if random_state is None:
        random_state = cfg["modeling"]["random_seed"]

    return train_test_split(
        X, y, weights,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def create_cv_folds(X, y,
                    n_splits: int | None = None,
                    random_state: int | None = None):
    """Stratified K-fold cross-validation."""
    if n_splits is None:
        n_splits = cfg["modeling"]["cv_folds"]
    if random_state is None:
        random_state = cfg["modeling"]["random_seed"]

    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )


# === 10. WEIGHT UTILITIES ===

def trim_extreme_weights(weights: np.ndarray,
                         percentile: int = 99) -> np.ndarray:
    """Trim extreme weights at given percentile."""
    threshold = np.percentile(weights, percentile)
    trimmed = np.clip(weights, None, threshold)
    n_trimmed = (weights > threshold).sum()
    print(f"Trimmed {n_trimmed} weights at {percentile}th pctile (threshold={threshold:.4f})")
    return trimmed


def scale_features(X_train, X_test):
    """Standardize features using training set statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# === 11. VARIABLE CONSTRUCTION REPORT ===

def generate_variable_construction_report(
    df: pd.DataFrame,
    features: list[str],
    missingness_df: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate variable construction report.
    Returns the report as a string and optionally writes to file.
    """
    from src.config import get_output_dir

    label_map = get_feature_labels()
    ordinal_spec = cfg.get("encoding", {}).get("ordinal", {})
    numeric_vars = cfg.get("encoding", {}).get("numeric", [])
    nominal_vars = cfg.get("encoding", {}).get("nominal", [])
    ns_vars = set(get_not_sure_variables())
    skip_info = get_skip_pattern_info()

    lines = [
        "# Variable Construction Report",
        "",
        "This report documents every variable used in the analysis,",
        "its coding, encoding strategy, and missingness handling.",
        "",
        f"**Total features**: {len(features)}",
        f"**Source**: Pew Research Center ATP W119 (N={SURVEY.N_TOTAL:,})",
        "",
        "---",
        "",
        "## Target Variable",
        "",
        f"| Raw Column | Derived | Coding | Note |",
        f"|------------|---------|--------|------|",
        f"| {SURVEY.TARGET_RAW} | y_apply | 1=Yes (code 1), 0=No (code 2), NaN=Refused (99) | NO 'Not sure' option |",
        "",
        "**Robustness alternatives**:",
        "- Default: Refused (99) = NaN (excluded)",
        "- Conservative: Refused (99) = 0 (No)",
        "",
        "---",
        "",
        "## Feature Variables",
        "",
        "| # | Raw Column | Readable Name | Type | Encoding | Not Sure? | Skip Pattern? |",
        "|---|------------|---------------|------|----------|-----------|---------------|",
    ]

    for i, col in enumerate(features, 1):
        readable = label_map.get(col, col)

        if col in ordinal_spec:
            enc_type = "Ordinal"
            order = ordinal_spec[col]
            enc_detail = f"Ranks 0-{len(order)-1}"
        elif col in numeric_vars:
            enc_type = "Numeric"
            enc_detail = "As-is"
        elif col in nominal_vars:
            enc_type = "Nominal"
            enc_detail = "One-hot"
        else:
            enc_type = "Unknown"
            enc_detail = "?"

        has_ns = "Yes (code 9)" if col in ns_vars else "No"
        if col in skip_info:
            sp = skip_info[col]
            has_skip = f"Yes: {sp['filter_var']} in {sp['filter_values']}"
        else:
            has_skip = "No"

        lines.append(
            f"| {i} | {col} | {readable} | {enc_type} | {enc_detail} | {has_ns} | {has_skip} |"
        )

    if missingness_df is not None:
        lines += [
            "",
            "---",
            "",
            "## Missingness Summary",
            "",
            "| Variable | % Not Sure | % Refused | % Structural NaN | % Any Missing |",
            "|----------|------------|-----------|-------------------|---------------|",
        ]
        for _, row in missingness_df.iterrows():
            lines.append(
                f"| {row['variable']} | {row['pct_not_sure']} | {row['pct_refused']} | "
                f"{row['pct_structural_nan']} | {row['pct_any_missing']} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Encoding Details",
        "",
        "### Ordinal Variables (integer-ranked)",
        "",
    ]

    for col, order in ordinal_spec.items():
        if col in features:
            lines.append(f"- **{col}** ({label_map.get(col, col)}): {order} -> ranks 0-{len(order)-1}")

    lines += [
        "",
        "### Numeric Variables (continuous, as-is)",
        "",
    ]
    for col in numeric_vars:
        if col in features:
            lines.append(f"- **{col}** ({label_map.get(col, col)})")

    lines += [
        "",
        "### Nominal Variables (one-hot encoded)",
        "",
    ]
    for col in nominal_vars:
        if col in features:
            lines.append(f"- **{col}** ({label_map.get(col, col)})")

    lines += [
        "",
        "---",
        "",
        "## Skip Patterns",
        "",
        "Variables with structural missingness due to questionnaire logic:",
        "",
    ]
    for var, spec in skip_info.items():
        if var in features:
            lines.append(
                f"- **{var}**: Only asked if {spec['filter_var']} in {spec['filter_values']} "
                f"(eligible N ~ {spec['eligible_n']:,})"
            )

    lines += [
        "",
        "---",
        "",
        "## 'Not Sure' Handling",
        "",
        "Variables with 'Not sure' (code 9) as a distinct response option:",
        "",
    ]
    for col in sorted(ns_vars):
        if col in features:
            lines.append(f"- {col} ({label_map.get(col, col)})")

    lines += [
        "",
        "Three treatments tested as sensitivity analysis:",
        "1. **drop**: Code 9 -> NaN (excluded via listwise deletion)",
        "2. **own_category**: Code 9 kept as a distinct level",
        "3. **midpoint**: Code 9 replaced with scale midpoint",
        "",
    ]

    report = "\n".join(lines)

    if output_path is None:
        output_path = get_output_dir("reports") / "variable_construction.md"

    from pathlib import Path as P
    P(output_path).write_text(report, encoding="utf-8")
    print(f"Variable construction report saved -> {output_path}")

    return report
