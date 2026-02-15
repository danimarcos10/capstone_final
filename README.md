# ATP W119 AI Survey Analysis

## Predicting Willingness to Apply for AI-Hiring Jobs

---

## Overview

This project analyzes Pew Research Center's American Trends Panel Wave 119 survey on Americans' attitudes toward artificial intelligence in hiring. The pipeline builds weighted classification models to predict who would apply for a job that uses AI in hiring, evaluates them rigorously, and tests whether a latent "AI attitude" dimension can match raw-item predictions. The entire pipeline is config-driven, reproducible with a single command, and documented across eight analysis phases.

---

## Project Structure

```
W119_Dec22/
|
|-- configs/
|   +-- default.yaml                  # Single source of truth for all parameters
|
|-- src/                               # Reusable Python modules
|   |-- config.py                      # Config loader, SurveyDesign dataclass, seed management
|   |-- data_loading.py                # SPSS ingestion, topline validation, target creation
|   |-- preprocessing.py               # Encoding, missingness, skip patterns, train/test split
|   |-- eda.py                         # Weighted Cramer's V, VIF, LOESS, outcome rates
|   |-- modeling.py                    # Logistic regression, gradient boosting, calibration
|   |-- evaluation.py                  # Weighted metrics, bootstrap CIs, Brier decomposition
|   |-- interpretability.py            # SHAP, permutation importance, coefficient analysis
|   +-- latent.py                      # Ordinal factor analysis, polychoric correlations, scoring
|
|-- notebooks/                         # Jupyter notebooks for interactive exploration
|   |-- 01_load_and_audit.ipynb
|   |-- 02_eda_hiring_block.ipynb
|   |-- 03_models_baselines_and_interpretability.ipynb
|   |-- 04_robustness_and_weighted_eval.ipynb
|   +-- 05_interpretability_directionality.ipynb
|
|-- reports/                           # Generated reports, figures, and tables
|   |-- figures/                       # PNG visualizations
|   |-- tables/                        # CSV/TEX tables
|   |-- model_cards/                   # One-page model summaries (LR_default.md, GBM.md)
|   |-- eda_summary.md
|   |-- evaluation_report.md
|   |-- robustness_results.md
|   |-- latent_robustness.md
|   |-- sample_flow.md
|   |-- variable_construction.md
|   |-- variable_dictionary.md
|   |-- audit_topline_validation.md
|   |-- codebook.csv
|   |-- model_metrics.json
|   +-- run_metadata.json
|
|-- tools/
|   +-- make_methodology_figures.py    # Pipeline diagram, CONSORT flow, feature table, perf table
|
|-- run_all.py                         # Full reproducibility pipeline (one command)
|-- run_eda.py                         # Phase 3: EDA diagnostics
|-- run_modeling.py                    # Phase 4: Train LR + GBM models
|-- run_evaluation.py                  # Phase 5: Evaluation + subgroup diagnostics
|-- run_calibration_upgrade.py         # Phase 5+: Platt scaling, Brier decomposition, adaptive ECE
|-- run_robustness.py                  # Phase 6: Sensitivity suite
|-- run_latent_v2.py                   # Phase 7: Latent construct + fair comparisons
|-- run_latent_upgrade.py              # Phase 7+: Polychoric correlations, 2-factor, invariance
|-- run_latent_robustness.py           # Phase 7.1: Latent stability checks
|
|-- ATP W119 Questionnaire.pdf         # Survey questions
|-- ATP W119 Methodology.pdf           # Survey methodology
|-- ATP W119 Readme.txt                # Original Pew documentation
|-- ATP W119 Topline.pdf               # Published results for validation
|
|-- requirements.txt                   # Pinned Python dependencies (Python 3.12.6)
+-- README.md
```

---

## Analysis Phases

1. **Reproducible Foundation** -- Load SPSS data, validate against 30 published topline values, set up config-driven infrastructure.
2. **Variable Construction** -- Define 22 features across 4 sets, handle "Not sure" and "Refused" codes, encode for modeling.
3. **Exploratory Data Analysis** -- Weighted outcome rates, Cramer's V, Spearman correlations, VIF, LOESS nonlinearity checks.
4. **Modeling Pipeline** -- Weighted logistic regression baseline and calibrated gradient boosting; compare on ROC-AUC, PR-AUC, Brier, ECE.
5. **Evaluation & Subgroup Diagnostics** -- Bootstrap CIs, threshold policies, calibration by subgroup, fairness diagnostics.
5+. **Calibration Upgrade** -- Platt scaling, adaptive ECE, Brier decomposition, calibration slope/intercept. Apples-to-apples comparison across calibration methods.
6. **Robustness Suite** -- Sensitivity across outcome recoding, "Not sure" treatment, skip patterns, weighting, seed stability; interpretability stability.
7. **Latent Attitude Construct** -- 1-factor ordinal factor analysis on 7 refined items, latent-based models vs. matched raw-item baselines, "Not sure" indicators.
7+. **Latent Psychometric Upgrade** -- Polychoric correlations, 1-factor vs. 2-factor comparison, scree plot, configural invariance across gender.

---

## The Data

### Survey
- **Source**: Pew Research Center American Trends Panel Wave 119
- **Fielded**: December 12--18, 2022
- **Sample Size**: 11,004 U.S. adults
- **Method**: Online survey (English and Spanish)

### Features

| Set | Raw | Encoded | What It Measures |
|-----|-----|---------|------------------|
| **Core Attitudes** | 9 | 18 | Awareness, favor/oppose AI in hiring, AI vs. human comparisons, bias perceptions |
| **Knowledge & AI Orientation** | 4 | 7 | AI knowledge score (0--6), awareness, excitement vs. concern, usage frequency |
| **Demographics** | 6 | 18 | Age, gender, education, race/ethnicity, party, income tier |
| **Employment Context** | 3 | varies | Employment status, recent job-seeking, industry |
| **Full** | 22 | 58 | All sets combined |

**Target**: `AIWRKH4_W119` -- "Would you apply for a job if you knew the employer uses AI in hiring?" (Yes / No / Refused).

### Survey Weights
`WEIGHT_W119` is used in all analyses -- descriptive statistics, model training, and evaluation metrics.

### Skip Patterns
- **HIREBIAS2** -- only asked if HIREBIAS1 = "Major" or "Minor" problem (N = 8,911)
- **INDUSTRYCOMBO** -- only asked if currently working (N = 6,497)

Documented in `configs/default.yaml` and handled automatically.

---

## How to Run

### Prerequisites

- Python 3.12.6 (recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Full Pipeline (Recommended)

```bash
python run_all.py
```

Executes in order:
1. `run_eda.py` -- EDA diagnostics, figures, `eda_summary.md`
2. `run_modeling.py` -- LR + GBM training, odds ratios, comparison tables
3. `run_evaluation.py` -- Thresholds, calibration, subgroup diagnostics
4. `run_calibration_upgrade.py` -- Platt scaling, Brier decomposition, adaptive ECE
5. `run_robustness.py` -- Sensitivity suite + interpretability stability
6. `run_latent_v2.py` -- Latent construct + fair comparisons
7. `run_latent_upgrade.py` -- Polychoric, 2-factor, gender invariance
8. `run_latent_robustness.py` -- Latent stability checks
9. Metadata generation -- `run_metadata.json`, sample flow, model cards

### Individual Phases

```bash
python run_eda.py                  # Phase 3
python run_modeling.py             # Phase 4
python run_evaluation.py           # Phase 5
python run_calibration_upgrade.py  # Phase 5+
python run_robustness.py           # Phase 6
python run_latent_v2.py            # Phase 7
python run_latent_upgrade.py       # Phase 7+
python run_latent_robustness.py    # Phase 7.1
```

### Methodology Figures

```bash
python tools/make_methodology_figures.py
```

Generates pipeline diagram, CONSORT sample flow, feature-set summary table, and model performance summary table.

### Interactive Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|----------|-------------|
| `01_load_and_audit.ipynb` | Load SPSS data, create codebook, validate toplines |
| `02_eda_hiring_block.ipynb` | Distributions, demographic splits, attitude visualizations |
| `03_models_baselines_and_interpretability.ipynb` | Train LR + GBM, SHAP, fairness checks |
| `04_robustness_and_weighted_eval.ipynb` | Sensitivity analyses, weighted evaluation |
| `05_interpretability_directionality.ipynb` | Feature direction checks, importance stability |

---

## Key Findings

### Who Would Apply to AI-Hiring Jobs?

About 32% of Americans say they would apply. Attitudes dominate demographics: the top 8 predictors (Cramer's V > 0.20) are all from the AI attitudes battery. Demographics have V < 0.10.

**Increases willingness**: Believing AI would help (strongest), younger age (18--29: 38.1% vs. 65+: 29.1%), higher AI knowledge, positive views of AI in hiring decisions.

**Decreases willingness**: Opposing AI in hiring (OR = 0.61), perceiving AI as more biased, older age, being a woman (28.6% vs. men 37.6%).

### Model Comparison (Held-Out Test Set, N = 2,155)

| Model | ROC-AUC | PR-AUC | Brier | ECE | Balanced Acc |
|-------|---------|--------|-------|-----|--------------|
| LR (default) | 0.857 | 0.795 | 0.145 | **0.031** | 0.745 |
| LR (balanced) | 0.856 | 0.794 | 0.149 | 0.068 | **0.785** |
| **GBM** | **0.867** | **0.800** | **0.142** | 0.039 | 0.762 |
| GBM (calibrated) | 0.867 | 0.777 | 0.144 | 0.043 | 0.751 |

GBM has the best discrimination (AUC); LR has the best calibration (ECE). Bootstrap CIs overlap -- the advantage is not statistically significant.

### Calibration Upgrade (Apples-to-Apples, same base estimator)

| Model | Calibration | AUC | Brier | ECE | Adaptive ECE | Cal. Slope |
|-------|-------------|-----|-------|-----|-------------|------------|
| LR | None | 0.853 | 0.148 | 0.036 | 0.039 | 0.976 |
| LR | Platt | 0.853 | 0.147 | 0.032 | 0.028 | 0.948 |
| LR | Isotonic | 0.850 | 0.147 | 0.031 | 0.033 | 0.852 |
| GBM | None | 0.868 | 0.144 | 0.042 | 0.050 | 0.946 |
| GBM | Platt | 0.868 | 0.142 | 0.040 | 0.041 | 1.094 |
| GBM | Isotonic | 0.868 | 0.143 | 0.037 | 0.033 | 1.012 |

### Latent Attitude Construct

A 1-factor ordinal factor analysis on 7 refined attitude items produces a single "latent AI attitude" score. With polychoric correlations: ordinal alpha = 0.777, KMO = 0.675. When matched against the same items, the latent model trails by only -0.011 AUC.

**Psychometric upgrade**: Polychoric correlations yield uniformly higher loadings (mean |diff| = 0.10) and better reliability than Spearman. Kaiser rule and scree plot both suggest a 2-factor solution (eigenvalues: 3.01, 1.20), though the 1-factor model is retained for parsimony. Configural invariance across gender holds (loading correlation rho = 0.929, p = 0.003).

---

## Robustness & Sensitivity

| Check | Finding |
|-------|---------|
| **Outcome recoding** (Refused = drop vs. No) | Identical AUC (only 233 Refused) |
| **"Not sure" treatment** (drop / own_category / midpoint) | own_category and drop comparable; midpoint worst |
| **Skip-pattern robustness** | Core attitudes alone (AUC ~0.85) nearly match full set |
| **Weight sensitivity** | Unweighted slightly higher AUC but within seed variability |
| **Seed stability** (20 seeds) | LR AUC mean = 0.853 (std = 0.010), GBM mean = 0.855 |

### Interpretability Stability
- **LR**: `favor_ai_review_apps` in top-10 in 97.5% of 200 bootstraps
- **GBM**: `favor_ai_review_apps` dominates permutation importance (0.071)
- **LR vs. GBM agreement**: Spearman rho = 0.03 (weak -- GBM better recovers substantive predictors)

---

## Subgroup Diagnostics

- **Gender gap**: TPR men = 0.70, women = 0.47
- **Race gap**: TPR Asian NH = 0.76, Hispanic = 0.57
- **Calibration by subgroup**: Worst for Hispanic (ECE = 0.123), best for White NH (ECE = 0.043)

Details in `reports/evaluation_report.md`.

---

## Generated Outputs

### Reports (in `reports/`)

| File | Description |
|------|-------------|
| `eda_summary.md` | EDA findings with modeling recommendations |
| `evaluation_report.md` | Metrics, thresholds, calibration, subgroup diagnostics |
| `robustness_results.md` | Sensitivity analyses across 5 dimensions |
| `latent_robustness.md` | Latent stability and "Not sure" sensitivity |
| `audit_topline_validation.md` | 30/30 published topline values validated |
| `variable_construction.md` | Feature encoding, missingness, skip patterns |
| `variable_dictionary.md` | Full variable dictionary |
| `sample_flow.md` | Sample sizes at each pipeline stage |
| `codebook.csv` | Complete variable codebook |
| `model_metrics.json` | Machine-readable model performance |
| `run_metadata.json` | Reproducibility info (versions, seed, config hash) |

### Model Cards (in `reports/model_cards/`)

| File | Description |
|------|-------------|
| `LR_default.md` | Weighted logistic regression summary |
| `GBM.md` | Gradient boosting classifier summary |

### Key Figures (in `reports/figures/`)

| File | Description |
|------|-------------|
| `outcome_by_*.png` | Weighted outcome rates by demographic |
| `cramers_v_all.png` | Cramer's V ranking of all predictors |
| `correlation_heatmap.png` | Spearman correlation heatmap |
| `vif_top25.png` | Top 25 VIF values |
| `lowess_*.png` | LOESS nonlinearity checks |
| `roc_comparison.png` | ROC curves for all model variants |
| `calibration_comparison.png` | Calibration plots (LR + GBM) |
| `reliability_diagrams.png` | Reliability diagrams |
| `odds_ratios_LR_default.png` | Forest plot of top 20 odds ratios with CIs |
| `subgroup_error_rates.png` | TPR/FPR/FNR by demographic subgroup |
| `seed_stability.png` | AUC stability across 20 seeds |
| `lr_vs_gbm_importance.png` | LR vs. GBM feature importance agreement |
| `phase7_1_model_comparison_matched.png` | Latent vs. matched baseline comparison |
| `latent_scree_plot.png` | Eigenvalue scree plot (polychoric vs. Spearman) |
| `methodology_pipeline.png` | Analysis pipeline diagram |
| `methodology_sample_flow.png` | CONSORT-style sample flow |
| `methodology_feature_sets.png` | Feature-set summary table |
| `model_performance_summary.png` | Publication-quality model performance table |

### Key Tables (in `reports/tables/`)

| File | Description |
|------|-------------|
| `model_comparison.csv` | 4-model comparison on held-out test set |
| `calibration_comparison.csv` | A2A calibration comparison (None/Platt/Isotonic) |
| `brier_decomposition.csv` | Brier decomposition (Uncertainty, Reliability, Resolution) |
| `sample_flow.csv` | Sample sizes at each pipeline stage |
| `phase7_1_model_comparison_matched.csv` | 12-model fair comparison (latent vs. baseline) |
| `phase7_1_loadings_v2.csv` | Factor loadings (refined items) |
| `latent_psychometrics_phase7_1.csv` | Psychometric quality (alpha, KMO, omega) |
| `latent_psychometrics_polychoric.csv` | Polychoric-based psychometrics |
| `latent_factor_comparison.csv` | 1-factor vs. 2-factor comparison |
| `latent_gender_invariance.csv` | Configural invariance across gender |
| `latent_stability_phase7_1.csv` | Cross-validation stability of latent scores |
| `latent_eligibility_bias.csv` | Eligibility bias for skip patterns |
| `methodology_feature_sets.csv` | Feature-set summary |
| `model_performance_summary.tex` | LaTeX-ready model performance table |

---

## Configuration

All parameters are centralized in `configs/default.yaml`:

- **Survey design**: wave, sample size, weight variable, form-split variable
- **Target variable**: column, positive/negative/refused codes
- **Feature sets**: 4 named sets + "full" and "safe" combinations
- **Skip patterns**: filter variables and eligible sizes
- **"Not sure" handling**: code, affected variables, treatment (drop / own_category / midpoint)
- **Encoding**: ordinal order, nominal variables, numeric pass-through
- **Missingness**: regime (listwise / impute_indicator), refused code
- **Modeling**: seed (42), test size, CV folds, threshold, class weight
- **Outputs**: directory paths

---

## Glossary

| Term | Definition |
|------|-----------|
| **Survey Weight** | Makes each response count proportionally to match the U.S. population |
| **Logistic Regression** | Predicts the probability of a binary outcome from a linear combination of features |
| **Gradient Boosting** | Ensemble of sequential decision trees; best discrimination in this pipeline |
| **ROC-AUC** | Area under the ROC curve (0.5 = random, 1.0 = perfect discrimination) |
| **PR-AUC** | Area under the precision-recall curve; informative under class imbalance |
| **Brier Score** | Mean squared error of predicted probabilities (lower = better) |
| **ECE** | Expected Calibration Error -- gap between predicted probabilities and observed rates |
| **Adaptive ECE** | ECE with quantile-based bins (Nguyen & O'Connor, 2015) |
| **Balanced Accuracy** | Average of sensitivity and specificity |
| **Odds Ratio** | Multiplicative change in odds per unit change in a feature |
| **Cramer's V** | Association strength between two categorical variables |
| **VIF** | Variance Inflation Factor -- multicollinearity diagnostic (>5 is concerning) |
| **Factor Analysis** | Finds latent dimensions underlying observed survey items |
| **Polychoric Correlation** | Correlation estimated for ordinal variables assuming an underlying bivariate normal (Olsson, 1979) |
| **Cronbach's Alpha** | Internal consistency of a scale (>0.70 acceptable) |
| **Ordinal Alpha** | Alpha computed on the polychoric matrix; appropriate for ordinal items |
| **KMO** | Kaiser-Meyer-Olkin sampling adequacy (>0.60 adequate, >0.80 excellent) |
| **SHAP** | Shapley Additive Explanations -- per-prediction feature importance |
| **Calibration Slope** | Slope of logit(observed) ~ logit(predicted); 1.0 = perfect calibration |
| **Skip Pattern** | Survey question asked only to respondents who gave a specific prior answer |

---

## Missingness

- **Refused** = code 99; **"Not sure"** = code 9 (six attitude items)
- Missingness is not random -- associated with demographics (evidence against MCAR)
- Two regimes: **listwise deletion** (N = 4,798, loses 56%) vs. **impute + indicators** (N = 10,771, default)
- "Not sure" treatment: drop / own_category (default) / midpoint

---

## Data Source

**Pew Research Center American Trends Panel Wave 119**

- [Public Awareness of Artificial Intelligence in Everyday Activities](https://www.pewresearch.org/science/2023/02/15/public-awareness-of-artificial-intelligence-in-everyday-activities/)
- [AI in Hiring and Evaluating Workers: What Americans Think](https://www.pewresearch.org/internet/2023/04/20/ai-in-hiring-and-evaluating-workers-what-americans-think/)

---

## License and Citation

This analysis uses publicly available data from Pew Research Center.

If you use this code or analysis, please cite:
- Pew Research Center (2022). American Trends Panel Wave 119.
- This repository

---

*Last updated: February 2026*
