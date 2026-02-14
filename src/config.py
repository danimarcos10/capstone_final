"""
Configuration loader and survey design constants for ATP W119 analysis.

Usage:
    from src.config import cfg, set_global_seed, SURVEY
    set_global_seed()           # call once at start of every script/notebook
    print(SURVEY.N_TOTAL)       # 11004
    print(cfg['target']['raw_column'])
"""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path = _DEFAULT_CONFIG) -> dict:
    """Load YAML config and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()


# ---------------------------------------------------------------------------
# Survey design constants (immutable, used in assertions & methodology text)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SurveyDesign:
    """Immutable survey design facts for ATP W119."""
    NAME: str = "Pew Research Center American Trends Panel Wave 119"
    WAVE: int = 119
    FIELD_START: str = "2022-12-12"
    FIELD_END: str = "2022-12-18"
    N_TOTAL: int = 11_004
    N_REMOVED_QUALITY: int = 8
    MODE: str = "Web (English and Spanish)"
    WEIGHT_VAR: str = "WEIGHT_W119"
    FORM_SPLIT_VAR: str = "FORM_W119"
    TARGET_RAW: str = "AIWRKH4_W119"
    TARGET_DERIVED: str = "y_apply"
    REFUSED_CODE: float = 99.0
    NOT_SURE_CODE: float = 9.0


SURVEY = SurveyDesign()


# ---------------------------------------------------------------------------
# Global seed management
# ---------------------------------------------------------------------------
def set_global_seed(seed: int | None = None) -> int:
    """
    Set global random seed for reproducibility.
    Uses config seed if none provided.
    Returns the seed used.
    """
    if seed is None:
        seed = cfg["modeling"]["random_seed"]
    np.random.seed(seed)
    try:
        import sklearn
        # sklearn respects numpy seed for most operations
    except ImportError:
        pass
    return seed


# ---------------------------------------------------------------------------
# Feature set helpers
# ---------------------------------------------------------------------------
def get_feature_set(name: str) -> list[str]:
    """Return list of raw column names for a named feature set."""
    fs = cfg["feature_sets"]
    if name == "full":
        return (
            fs["core_attitudes"]
            + fs["knowledge_ai_orientation"]
            + fs["demographics"]
            + fs["employment_context"]
        )
    if name not in fs:
        raise ValueError(f"Unknown feature set '{name}'. Choose from: {list(fs.keys())} or 'full'.")
    return fs[name]


def get_feature_labels() -> dict[str, str]:
    """Return raw_col → readable_name mapping."""
    return cfg["feature_labels"]


def get_not_sure_variables() -> list[str]:
    """Return list of variables that have 'Not sure' (code 9)."""
    return cfg["not_sure"]["variables_with_not_sure"]


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------
def get_output_dir(kind: str = "reports") -> Path:
    """Return absolute path for an output directory, creating it if needed."""
    key_map = {
        "reports": "reports_dir",
        "figures": "figures_dir",
        "tables": "tables_dir",
        "model_cards": "model_cards_dir",
    }
    rel = cfg["outputs"].get(key_map.get(kind, kind), kind)
    out = _PROJECT_ROOT / rel
    out.mkdir(parents=True, exist_ok=True)
    return out


# Convenience: project root path
PROJECT_ROOT = _PROJECT_ROOT
