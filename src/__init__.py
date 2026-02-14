"""
ATP W119 Analysis Package
=========================
Modules:
    config          – Configuration loading + survey constants
    data_loading    – SPSS loading, validation, codebook
    preprocessing   – Feature engineering, missing data, train/test
    eda             – Exploratory data analysis / diagnostics
    evaluation      – Weighted/unweighted metrics, calibration
    interpretability – SHAP, coefficients, importance analysis
"""

from src import config
from src import data_loading
from src import preprocessing
from src import eda
from src import modeling
from src import evaluation
from src import interpretability
