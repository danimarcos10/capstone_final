"""
Data loading module for ATP W119 analysis.
SPSS loading, survey-design validation, topline replication, and target creation.
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

from src.config import cfg, SURVEY, get_feature_set, get_feature_labels


# === 1. LOAD DATA ===

def load_atp_w119(data_path: str | Path | None = None) -> tuple[pd.DataFrame, object]:
    """
    Load ATP W119 SPSS file keeping raw numeric codes.

    Args:
        data_path: Path to .sav file (default: '<project_root>/ATP W119.sav').

    Returns:
        df: DataFrame with raw codes (float64).
        meta: pyreadstat metadata.
    """
    if data_path is None:
        from src.config import PROJECT_ROOT
        data_path = PROJECT_ROOT / "ATP W119.sav"

    # apply_value_formats=False keeps raw numeric codes instead of labels
    df, meta = pyreadstat.read_sav(str(data_path), apply_value_formats=False)

    print(f"Loaded {len(df):,} respondents, {len(df.columns)} variables")
    return df, meta


# === 2. CODEBOOK / VARIABLE INFO ===

def get_variable_info(df: pd.DataFrame, meta) -> pd.DataFrame:
    """Create variable codebook with labels, value maps, and special codes."""
    rows = []
    for var in df.columns:
        var_label = meta.column_names_to_labels.get(var, "")
        value_labels = meta.variable_value_labels.get(var, {})

        missing_codes = []
        not_sure_codes = []
        if value_labels:
            for code, label in value_labels.items():
                lbl = str(label).lower()
                if any(x in lbl for x in ["refused", "no answer"]):
                    missing_codes.append(f"{code}={label}")
                if "not sure" in lbl or "don't know" in lbl:
                    not_sure_codes.append(f"{code}={label}")

        rows.append({
            "variable": var,
            "label": var_label,
            "n_valid": df[var].notna().sum(),
            "n_unique": df[var].nunique(),
            "value_labels": str(value_labels) if value_labels else "",
            "missing_codes": "; ".join(missing_codes) if missing_codes else "None",
            "not_sure_codes": "; ".join(not_sure_codes) if not_sure_codes else "None",
        })

    return pd.DataFrame(rows)


# === 3. SURVEY-DESIGN ASSERTIONS ===

def assert_survey_design(df: pd.DataFrame) -> dict:
    """
    Run non-negotiable survey-design assertions.
    Raises AssertionError on failure; returns summary dict.
    """
    results = {}

    assert len(df) == SURVEY.N_TOTAL, (
        f"Expected {SURVEY.N_TOTAL} rows, got {len(df)}"
    )
    results["n_rows"] = len(df)

    w = SURVEY.WEIGHT_VAR
    assert w in df.columns, f"Weight variable '{w}' not in data"
    weights = df[w]
    assert weights.notna().all(), "Weight variable has NaN values"
    assert (weights > 0).all(), "Weight variable has non-positive values"
    results["weight_min"] = float(weights.min())
    results["weight_max"] = float(weights.max())
    results["weight_mean"] = float(weights.mean())

    f = SURVEY.FORM_SPLIT_VAR
    assert f in df.columns, f"Form-split variable '{f}' not in data"
    form_counts = df[f].value_counts().to_dict()
    results["form_counts"] = {str(k): int(v) for k, v in form_counts.items()}

    t = SURVEY.TARGET_RAW
    assert t in df.columns, f"Target variable '{t}' not in data"
    target_vals = set(df[t].dropna().unique())
    expected = {1.0, 2.0, 99.0}
    assert target_vals.issubset(expected), (
        f"Unexpected values in {t}: {target_vals - expected}"
    )
    results["target_value_counts"] = df[t].value_counts().to_dict()

    print("[OK] All survey-design assertions passed")
    return results


# === 4. FORM-SPLIT HANDLING ===

def identify_form_split_items(df: pd.DataFrame, threshold: float = 0.35) -> dict:
    """
    Identify variables likely administered to only one form by checking
    NaN alignment with the form-split variable.

    Args:
        df: Full DataFrame.
        threshold: Minimum NaN fraction to flag (default 0.35).

    Returns:
        Dict mapping variable -> {nan_rate, likely_form}.
    """
    form_var = SURVEY.FORM_SPLIT_VAR
    results = {}

    for col in df.columns:
        nan_rate = df[col].isna().mean()
        if nan_rate < threshold:
            continue

        form1_nan = df.loc[df[form_var] == 1.0, col].isna().mean()
        form2_nan = df.loc[df[form_var] == 2.0, col].isna().mean()

        likely_form = None
        if form1_nan > 0.90 and form2_nan < 0.10:
            likely_form = 2  # asked only to Form 2
        elif form2_nan > 0.90 and form1_nan < 0.10:
            likely_form = 1  # asked only to Form 1

        results[col] = {
            "nan_rate": round(nan_rate, 4),
            "form1_nan_rate": round(form1_nan, 4),
            "form2_nan_rate": round(form2_nan, 4),
            "likely_form": likely_form,
        }

    return results


def filter_by_form(df: pd.DataFrame, form: int) -> pd.DataFrame:
    """Return subset of respondents assigned to a specific form (1 or 2)."""
    assert form in (1, 2), "form must be 1 or 2"
    return df[df[SURVEY.FORM_SPLIT_VAR] == float(form)].copy()


# === 5. TARGET VARIABLE CREATION ===

def create_target_variable(df: pd.DataFrame,
                           target_col: str | None = None) -> pd.Series:
    """
    Create binary y_apply: 1 = "Yes" (code 1), 0 = "No" (code 2).
    Refused (99) and native NaN map to NaN. AIWRKH4 has no "Not sure" option.
    """
    if target_col is None:
        target_col = SURVEY.TARGET_RAW

    y = df[target_col].map({
        cfg["target"]["positive_code"]: 1,
        cfg["target"]["negative_code"]: 0,
    })

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    n_miss = y.isna().sum()
    print(f"Target '{target_col}' -> y_apply:  1={n_pos:,}  0={n_neg:,}  NaN={n_miss:,}")
    return y


def create_alternative_target(df: pd.DataFrame,
                              target_col: str | None = None) -> pd.Series:
    """
    Robustness target: Refused (99) coded as "No" instead of NaN.
    Conservative coding for sensitivity analysis.
    """
    if target_col is None:
        target_col = SURVEY.TARGET_RAW

    y = df[target_col].map({
        cfg["target"]["positive_code"]: 1,
        cfg["target"]["negative_code"]: 0,
        cfg["target"]["refused_code"]: 0,   # conservative: Refused -> No
    })
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    print(f"Alternative target (Refused=No): 1={n_pos:,}  0={n_neg:,}")
    return y


# === 6. FEATURE SET HELPERS ===

def get_feature_columns(set_name: str = "full") -> dict:
    """
    Get feature columns for a named set from config.

    Args:
        set_name: "core_attitudes", "knowledge_ai_orientation",
                  "demographics", "employment_context", or "full".

    Returns:
        Dict with columns, labels, and exclude_vars (leakage prevention).
    """
    columns = get_feature_set(set_name)
    all_labels = get_feature_labels()
    labels = {c: all_labels.get(c, c) for c in columns}

    exclude_vars = [
        SURVEY.TARGET_RAW,
        SURVEY.TARGET_DERIVED,
        "y_apply",
        SURVEY.WEIGHT_VAR,
        "QKEY",
    ]

    return {
        "columns": columns,
        "labels": labels,
        "exclude_vars": exclude_vars,
    }


# === 7. LEAKAGE CHECK ===

def check_leakage(df: pd.DataFrame,
                  target_col: str = "y_apply",
                  threshold: float = 0.95) -> list[tuple[str, float]]:
    """Flag variables with |correlation| >= threshold with target."""
    leakage_vars = []
    target = df[target_col]

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype in ["float64", "int64"]:
            valid = target.notna() & df[col].notna()
            if valid.sum() > 100:
                corr = target[valid].corr(df[col][valid])
                if abs(corr) >= threshold:
                    leakage_vars.append((col, round(corr, 4)))

    return leakage_vars


# === 8. WEIGHTED FREQUENCIES ===

def weighted_frequency(df: pd.DataFrame,
                       var: str,
                       weight_var: str | None = None,
                       exclude_codes: list[float] | None = None,
                       meta=None) -> pd.DataFrame:
    """
    Compute weighted frequency distribution for a variable.

    Args:
        df: DataFrame.
        var: Variable name.
        weight_var: Weight column (default from config).
        exclude_codes: Codes to exclude (default: [99] = Refused).
        meta: Optional pyreadstat metadata for value labels.

    Returns:
        DataFrame with value, label, n_unweighted, weighted_pct.
    """
    if weight_var is None:
        weight_var = SURVEY.WEIGHT_VAR
    if exclude_codes is None:
        exclude_codes = [SURVEY.REFUSED_CODE]

    mask = df[var].notna()
    for code in exclude_codes:
        mask = mask & (df[var] != code)

    valid_df = df[mask]
    total_weight = valid_df[weight_var].sum()

    val_labels = {}
    if meta is not None:
        val_labels = meta.variable_value_labels.get(var, {})

    rows = []
    for value in sorted(valid_df[var].unique()):
        val_mask = valid_df[var] == value
        weight_sum = valid_df.loc[val_mask, weight_var].sum()
        pct = (weight_sum / total_weight) * 100

        rows.append({
            "value": value,
            "label": val_labels.get(value, ""),
            "n_unweighted": int(val_mask.sum()),
            "weighted_pct": round(pct, 1),
        })

    return pd.DataFrame(rows)


# === 9. TOPLINE VALIDATION ===

# Published topline values from Pew Research Center ATP W119 Topline.pdf (pp. 16-19).
# Tolerance of +/-1.5 pp accounts for rounding in the published PDF.
#
# Skip patterns:
#   HIREBIAS2: only asked if HIREBIAS1=1,2 (N=8,911)
#   EVALBIAS2: only asked if EVALBIAS1=1,2 (N=8,371)
#   INDUSTRYCOMBO: only asked if EMPLSIT=1,2 (N=6,497)

PUBLISHED_TOPLINES = {
    "AIWRKH1_W119": {
        # "How much have you heard or read about AI in hiring?"
        1.0: 7,     # A lot
        2.0: 32,    # A little
        3.0: 61,    # Nothing at all
    },
    "AIWRKH4_W119": {
        # "Would you want to apply for a job using AI in hiring?"
        1.0: 32,    # Yes, I would
        2.0: 66,    # No, I would not
    },
    "AIWRKH2_a_W119": {
        # "Favor/oppose AI reviewing job applications"
        1.0: 28,    # Favor
        2.0: 41,    # Oppose
        9.0: 30,    # Not sure
    },
    "AIWRKH2_b_W119": {
        # "Favor/oppose AI making final hiring decision"
        1.0: 7,     # Favor
        2.0: 71,    # Oppose
        9.0: 22,    # Not sure
    },
    "AIWRKH3_a_W119": {
        # "AI better/worse than humans at identifying qualified?"
        1.0: 27,    # Better
        2.0: 23,    # Worse
        3.0: 26,    # Same
        9.0: 23,    # Not sure
    },
    "HIREBIAS1_W119": {
        # "How much of a problem is bias in hiring?"
        1.0: 37,    # A major problem
        2.0: 42,    # A minor problem
        3.0: 19,    # Not a problem
    },
    "EMPLSIT_W119": {
        # "What is your current work situation?"
        1.0: 48,    # Work full time
        2.0: 12,    # Work part time
        3.0: 11,    # Not currently working
        4.0: 7,     # Unable to work
        5.0: 21,    # Retired
    },
    "JOBAPPYR_W119": {
        # "Applied for a job in past 12 months?"
        1.0: 26,    # Yes
        2.0: 73,    # No
    },
}

# Skip-pattern toplines: validated among eligible subsets only
PUBLISHED_TOPLINES_SKIP = {
    "HIREBIAS2_W119": {
        # Only asked if HIREBIAS1=1,2 (N=8,911):
        # "Would AI make bias in hiring better or worse?"
        "filter_var": "HIREBIAS1_W119",
        "filter_values": [1.0, 2.0],
        "expected": {
            1.0: 10,    # Definitely get better
            2.0: 44,    # Probably get better
            3.0: 32,    # Stay about the same
            4.0: 9,     # Probably get worse
            5.0: 4,     # Definitely get worse
        },
    },
}


def validate_toplines(df: pd.DataFrame,
                      tolerance_pp: float = 2.0,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Replicate published topline weighted percentages and compare.
    Handles both ASK-ALL and skip-pattern items.

    Args:
        df: Full DataFrame.
        tolerance_pp: Maximum acceptable deviation in percentage points.
        verbose: Print results.

    Returns:
        DataFrame with variable, code, published/computed pct, deviation, tolerance flag.
    """
    weight_var = SURVEY.WEIGHT_VAR
    rows = []

    # ASK-ALL items
    for var, expected in PUBLISHED_TOPLINES.items():
        if var not in df.columns:
            print(f"[WARN] Variable '{var}' not in data, skipping")
            continue

        # Refused (99) excluded from denominator
        freq = weighted_frequency(df, var, weight_var, exclude_codes=[99.0])

        for code, pub_pct in expected.items():
            match = freq[freq["value"] == code]
            if len(match) == 0:
                computed = None
                dev = None
                ok = False
            else:
                computed = match["weighted_pct"].values[0]
                dev = abs(computed - pub_pct)
                ok = dev <= tolerance_pp

            rows.append({
                "variable": var,
                "code": code,
                "published_pct": pub_pct,
                "computed_pct": computed,
                "deviation_pp": dev,
                "within_tolerance": ok,
                "note": "ASK ALL",
            })

    # Skip-pattern items
    for var, spec in PUBLISHED_TOPLINES_SKIP.items():
        if var not in df.columns:
            print(f"[WARN] Variable '{var}' not in data, skipping")
            continue

        filt_var = spec["filter_var"]
        filt_vals = spec["filter_values"]
        eligible = df[df[filt_var].isin(filt_vals)]
        n_eligible = len(eligible)

        freq = weighted_frequency(eligible, var, weight_var, exclude_codes=[99.0])

        for code, pub_pct in spec["expected"].items():
            match = freq[freq["value"] == code]
            if len(match) == 0:
                computed = None
                dev = None
                ok = False
            else:
                computed = match["weighted_pct"].values[0]
                dev = abs(computed - pub_pct)
                ok = dev <= tolerance_pp

            rows.append({
                "variable": var,
                "code": code,
                "published_pct": pub_pct,
                "computed_pct": computed,
                "deviation_pp": dev,
                "within_tolerance": ok,
                "note": f"SKIP: {filt_var} in {filt_vals} (N={n_eligible:,})",
            })

    result = pd.DataFrame(rows)

    if verbose and len(result) > 0:
        n_total = len(result)
        n_ok = result["within_tolerance"].sum()
        max_dev = result["deviation_pp"].max()
        print(f"\nTopline validation: {n_ok}/{n_total} within +/-{tolerance_pp} pp")
        print(f"Max deviation: {max_dev:.1f} pp")
        if n_ok < n_total:
            failed = result[~result["within_tolerance"]]
            print(f"\nFailed items:\n{failed.to_string(index=False)}")
        else:
            print("All topline values validated successfully")

    return result


def save_topline_report(validation_df: pd.DataFrame,
                        output_path: str | Path | None = None) -> None:
    """Write topline validation results to markdown report."""
    if output_path is None:
        from src.config import get_output_dir
        output_path = get_output_dir("reports") / "audit_topline_validation.md"

    max_dev = validation_df["deviation_pp"].max()
    n_ok = validation_df["within_tolerance"].sum()
    n_total = len(validation_df)

    lines = [
        "# Topline Validation Report",
        "",
        f"**Pipeline validation**: {n_ok}/{n_total} published topline values "
        f"replicated within tolerance.",
        f"**Max absolute deviation**: {max_dev:.1f} percentage points.",
        "",
        "## Results",
        "",
        "| Variable | Code | Published % | Computed % | Deviation | OK? | Note |",
        "|----------|------|-------------|------------|-----------|-----|------|",
    ]

    for _, row in validation_df.iterrows():
        ok_str = "YES" if row["within_tolerance"] else "NO"
        comp = f"{row['computed_pct']:.1f}" if row["computed_pct"] is not None else "N/A"
        dev = f"{row['deviation_pp']:.1f}" if row["deviation_pp"] is not None else "N/A"
        note = row.get("note", "")
        lines.append(
            f"| {row['variable']} | {row['code']:.0f} | "
            f"{row['published_pct']} | {comp} | {dev} | {ok_str} | {note} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Published values from ATP W119 Topline.pdf (pp. 16-19).",
        "- Weighted using `WEIGHT_W119`; Refused (code 99) excluded from denominator.",
        "- Skip-pattern items validated among eligible respondents only.",
        f"- Tolerance: +/-2.0 percentage points (accounts for rounding in published toplines).",
        f"- Validated against N = {SURVEY.N_TOTAL:,} respondents.",
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved topline validation report -> {output_path}")
