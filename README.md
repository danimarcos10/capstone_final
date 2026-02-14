# ATP W119 AI Survey Analysis

## A Beginner-Friendly Guide to Analyzing Pew Research Center Data

---

## 📋 What is This Project?

This project analyzes a survey conducted by the Pew Research Center about Americans' attitudes toward artificial intelligence (AI) in hiring. The survey asked questions like:

- "Would you apply for a job if you knew the employer uses AI in hiring?"
- "Do you think AI would be better or worse than humans at avoiding bias?"
- "Would AI help or hurt your chances of getting hired?"

We use data science techniques to understand patterns in the responses, build models to predict who would be willing to apply for jobs using AI in hiring, rigorously evaluate those models, and test whether a latent "AI attitude" dimension can match raw-item predictions. The entire pipeline is config-driven, reproducible with a single command, and documented across seven analysis phases.

---

## 📁 Project Structure

```
capstone/
│
├── configs/
│   └── default.yaml                  # Single source of truth for all analysis parameters
│
├── src/                               # Reusable Python modules
│   ├── config.py                      # Config loader + SurveyDesign dataclass + seed management
│   ├── data_loading.py                # SPSS ingestion, topline validation, target creation
│   ├── preprocessing.py               # Encoding, missingness, "Not sure" handling, train/test split
│   ├── eda.py                         # Weighted Cramer's V, VIF, LOESS, outcome rates
│   ├── modeling.py                    # Logistic regression, gradient boosting, odds ratios
│   ├── evaluation.py                  # Weighted metrics (ROC-AUC, Brier, ECE), bootstrap CIs
│   ├── interpretability.py            # SHAP, permutation importance, coefficient analysis
│   └── latent.py                      # Ordinal factor analysis, latent scoring, "Not sure" indicators
│
├── notebooks/                         # Jupyter notebooks for interactive exploration
│   ├── 01_load_and_audit.ipynb        # Load data, validate against published toplines
│   ├── 02_eda_hiring_block.ipynb      # Visualize attitudes and demographic splits
│   ├── 03_models_baselines_and_interpretability.ipynb  # Train and interpret models
│   ├── 04_robustness_and_weighted_eval.ipynb           # Sensitivity and robustness checks
│   └── 05_interpretability_directionality.ipynb        # Feature direction and importance stability
│
├── reports/                           # Generated reports, figures, and tables
│   ├── figures/                       # PNG visualizations (EDA, model comparison, latent)
│   ├── tables/                        # CSV tables (outcome rates, metrics, loadings)
│   ├── model_cards/                   # One-page model summaries (LR_default.md, GBM.md)
│   ├── eda_summary.md                 # EDA findings with modeling recommendations
│   ├── evaluation_report.md           # Metrics, thresholds, calibration, subgroup diagnostics
│   ├── robustness_results.md          # Sensitivity analyses across 5 dimensions
│   ├── latent_report.md               # Latent construct methodology, loadings, comparison
│   ├── latent_robustness.md           # Latent model stability and "Not sure" sensitivity
│   ├── sample_flow.md                 # Sample sizes at each pipeline stage
│   ├── variable_construction.md       # Feature encoding, missingness, skip patterns
│   ├── variable_dictionary.md         # Full variable dictionary
│   ├── audit_topline_validation.md    # 30/30 published topline values validated
│   ├── modeling_plan.md               # Modeling strategy
│   ├── codebook.csv                   # Complete variable codebook
│   ├── model_metrics.json             # Machine-readable model performance
│   └── run_metadata.json              # Reproducibility info (versions, seed, config hash)
│
├── run_all.py                         # One-command reproducibility pipeline
├── run_eda.py                         # Phase 3: EDA diagnostics
├── run_modeling.py                    # Phase 4: Train LR + GBM models
├── run_evaluation.py                  # Phase 5: Evaluation + subgroup diagnostics
├── run_robustness.py                  # Phase 6: Sensitivity suite
├── run_latent_v2.py                   # Phase 7/7.1: Latent construct + fair comparisons
├── run_latent_robustness.py           # Phase 7.1: Latent stability checks
├── sanity_checks.py                   # Pre-Phase-5 direction + calibration hygiene checks
├── verify_pipeline.py                 # PASS/FAIL verification of all critical assumptions
│
├── ATP W119 Questionnaire.pdf         # Survey questions
├── ATP W119 Methodology.pdf           # How the survey was conducted
├── ATP W119 Readme.txt                # Original Pew documentation
├── ATP W119 Topline.pdf               # Published results to validate against
│
├── CHECKLIST.md                       # Master technical checklist (7 phases)
├── requirements.txt                   # Pinned Python dependencies (Python 3.12.6)
└── README.md                          # This file!
```

---

## 🎯 Project Goals

The analysis is organized into **seven phases**, each building on the last:

1. **Reproducible Foundation** — Load the SPSS data, validate against 30 published topline values, and set up a config-driven infrastructure.
2. **Variable Construction** — Define 22 features across 4 sets (attitudes, knowledge, demographics, employment), handle "Not sure" and "Refused" codes, and encode variables for modeling.
3. **Exploratory Data Analysis** — Compute weighted outcome rates, Cramer's V, Spearman correlations, VIF, and LOESS nonlinearity checks to motivate model choices.
4. **Modeling Pipeline** — Train a weighted logistic regression baseline and a calibrated gradient boosting model; compare on ROC-AUC, PR-AUC, Brier, ECE, and balanced accuracy.
5. **Evaluation & Subgroup Diagnostics** — Bootstrap confidence intervals, threshold policies, calibration by subgroup, and fairness diagnostics (TPR by age, gender, education, race/ethnicity).
6. **Robustness Suite** — Sensitivity checks across outcome recoding, "Not sure" treatment, skip-pattern inclusion, weighting, and seed stability; plus interpretability stability (bootstrap rank + permutation importance).
7. **Latent Attitude Construct** — Fit a 1-factor ordinal factor analysis on 9 attitude items, compare latent-based models against matched raw-item baselines, and test "Not sure" as separate indicators for fair apples-to-apples evaluation.

See `CHECKLIST.md` for the complete phase-by-phase technical checklist with status.

---

## 📊 The Data

### About the Survey
- **Source**: Pew Research Center American Trends Panel Wave 119
- **When**: December 12–18, 2022
- **Sample Size**: 11,004 U.S. adults
- **Method**: Online survey in English and Spanish

### Key Variables We Analyze

We use **22 raw features** organized into four named sets:

| Set | Variables | What They Measure |
|-----|-----------|-------------------|
| **Core Attitudes** (9) | AIWRKH1, AIWRKH2a/b, AIWRKH3a–d, HIREBIAS1, HIREBIAS2 | Awareness, favor/oppose AI in hiring, AI vs. human comparisons, bias perceptions |
| **Knowledge & AI Orientation** (4) | AIKNOW_INDEX, AI_HEARD, CNCEXC, USEAI | AI knowledge score (0–6), AI awareness, excited vs. concerned, AI usage frequency |
| **Demographics** (6) | F_AGECAT, F_GENDER, F_EDUCCAT2, F_RACETHNMOD, F_PARTY_FINAL, F_INC_TIER2 | Age, gender, education, race/ethnicity, party, income tier |
| **Employment Context** (3) | EMPLSIT, JOBAPPYR, INDUSTRYCOMBO | Employment status, recent job-seeking, industry |

**Target variable**: `AIWRKH4_W119` — "Would you apply for a job if you knew the employer uses AI in hiring?" (Yes / No / Refused).

### Survey Weights
The data includes a weight variable (`WEIGHT_W119`) that makes the sample representative of all U.S. adults. **We use this weight in all analyses** — descriptive statistics, model training, and evaluation metrics.

### Skip Patterns
Some variables were only asked to eligible subsets:
- **HIREBIAS2** — only asked if HIREBIAS1 = "Major" or "Minor" problem (N = 8,911)
- **INDUSTRYCOMBO** — only asked if currently working (N = 6,497)

These skip patterns are documented in `configs/default.yaml` and handled automatically by the pipeline.

---

## 🔧 How to Run the Analysis

### Prerequisites

- **Python 3.12.6** (recommended)
- Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, pyreadstat, scikit-learn, scipy, statsmodels, matplotlib, seaborn, shap, PyYAML, factor-analyzer, jupyter, and ipykernel at tested versions.

### Option 1: Run the Full Pipeline (Recommended)

A single command regenerates all tables, figures, reports, model cards, and metadata:

```bash
python run_all.py
```

This executes in order:
1. `run_eda.py` — EDA diagnostics, figures, and `eda_summary.md`
2. `run_modeling.py` — LR + GBM training, odds ratios, comparison tables
3. `run_evaluation.py` — Thresholds, calibration, subgroup diagnostics
4. `run_robustness.py` — Sensitivity suite + interpretability stability
5. `run_latent_v2.py` — Latent construct + fair comparisons
6. `run_latent_robustness.py` — Latent stability checks
7. Metadata generation — `run_metadata.json`, sample flow, model cards

### Option 2: Run Individual Phases

Each phase can be run independently:

```bash
python run_eda.py            # Phase 3: EDA
python run_modeling.py       # Phase 4: Modeling
python run_evaluation.py     # Phase 5: Evaluation
python run_robustness.py     # Phase 6: Robustness
python run_latent_v2.py      # Phase 7/7.1: Latent construct
```

### Option 3: Interactive Notebooks

Open any notebook in `notebooks/` for step-by-step exploration:

```bash
jupyter notebook notebooks/
```

| Notebook | What It Does |
|----------|-------------|
| `01_load_and_audit.ipynb` | Loads SPSS data, creates codebook, validates against published toplines, saves cleaned data |
| `02_eda_hiring_block.ipynb` | Distributions, demographic splits, attitude visualizations |
| `03_models_baselines_and_interpretability.ipynb` | Trains LR + GBM, feature importance, SHAP values, fairness checks |
| `04_robustness_and_weighted_eval.ipynb` | Sensitivity analyses and weighted evaluation |
| `05_interpretability_directionality.ipynb` | Feature direction checks and importance stability |

### Verification

Run the pipeline verification script to confirm all critical assumptions hold:

```bash
python verify_pipeline.py
```

Expected output: all PASS, zero FAIL.

---

## 📈 Key Findings

### Who Would Apply to AI-Hiring Jobs?

**Overall**: Only about 32% of Americans say they would apply to a job that uses AI in hiring.

### Factors That Increase Willingness to Apply:
1. **Believing AI would help them** (strongest factor — OR = 0.43 for each unit of opposition, meaning opposition halves the odds)
2. **Younger age** (18–29: 38.1% vs. 65+: 29.1%)
3. **Higher AI knowledge** (those who know more are more positive)
4. **Positive view of AI reviewing applications and making hiring decisions**

### Factors That Decrease Willingness:
1. **Opposing AI in hiring decisions** (OR = 0.61 for `favor_ai_final_decision`)
2. **Thinking AI is more biased than humans**
3. **Older age**
4. **Being a woman** (28.6% vs. men 37.6%)

### What Matters Most?

**Attitudes dominate demographics.** The top 8 predictors (Cramer's V > 0.20) are all from the AI attitudes battery. Demographics have V < 0.10. This means what people *think about AI* matters far more than *who they are* for predicting willingness to apply.

---

## 🧮 Understanding the Models

### Model 1: Weighted Logistic Regression (Baseline)

**What it is**: A simple model that estimates the probability of applying based on a weighted combination of features.

**Why we use it**: Easy to interpret! Each feature gets an odds ratio that tells us how much it matters. We report bootstrapped 95% confidence intervals (1,000 resamples).

**Performance**: ROC-AUC = 0.857, ECE = 0.031 (best-calibrated model).

### Model 2: Gradient Boosting (HistGBM)

**What it is**: An advanced model (`HistGradientBoostingClassifier`) that combines many simple decision trees.

**Why we use it**: Best discrimination (ROC-AUC = 0.867). Permutation importance reveals `favor_ai_review_apps` as the dominant predictor.

**Interpreting it**: We use SHAP values and permutation importance to see which features matter most.

### Model Comparison (Held-Out Test Set, N = 2,155)

| Model | ROC-AUC | PR-AUC | Brier | ECE | Balanced Acc |
|-------|---------|--------|-------|-----|--------------|
| LR (default) | 0.857 | 0.795 | 0.145 | **0.031** | 0.745 |
| LR (balanced) | 0.856 | 0.794 | 0.149 | 0.068 | **0.785** |
| **GBM** | **0.867** | **0.800** | **0.142** | 0.039 | 0.762 |
| GBM (calibrated) | 0.867 | 0.777 | 0.144 | 0.043 | 0.751 |

**Nuanced story**: GBM has the best discrimination (AUC), LR has the best calibration (ECE). For survey-based interpretive work, LR is arguably preferred. Bootstrap CIs overlap for all metrics — the advantage is not statistically significant.

### Model 3: Latent Attitude Construct (Novel Contribution)

**What it is**: A 1-factor ordinal factor analysis fitted on 9 attitude items, producing a single "latent AI attitude" score per respondent.

**Why we built it**: To test whether a psychometrically principled dimension can match raw-item predictions.

**Quality**: Cronbach's α = 0.760 (acceptable), KMO = 0.826 (excellent). Strong loadings on AIWRKH3 items (> 0.7).

**Fair comparison**: When matched against the same 9 raw items (apples-to-apples), the latent model trails by only −0.011 AUC (matched baseline GBM = 0.841, best latent LR = 0.830). The original unfair gap of −0.12 was due to comparing 58 features vs. 7–11.

---

## 🔬 Robustness & Sensitivity

We tested the pipeline across five dimensions to make sure results are stable:

| Check | Finding |
|-------|---------|
| **Outcome recoding** (Refused = drop vs. Refused = No) | Identical AUC (only 233 Refused, negligible impact) |
| **"Not sure" treatment** (drop / own_category / midpoint) | own_category and drop comparable; midpoint worst. own_category is the safe default |
| **Skip-pattern robustness** | Core attitudes alone (AUC = 0.848/0.854) nearly match full set. Demographics alone are weak (AUC ≈ 0.60) |
| **Weight sensitivity** | Unweighted training slightly higher AUC but within seed variability |
| **Seed stability** (20 random seeds) | LR AUC mean = 0.853 (std = 0.010), GBM mean = 0.855. Very stable |

### Interpretability Stability

- **LR**: `favor_ai_review_apps` appears in top-10 coefficients in 97.5% of 200 bootstraps. Missing indicators dominate top ranks (artifact of rare groups with large |coefficients|).
- **GBM**: `favor_ai_review_apps` dominates permutation importance (0.071), stable across CV folds. Top 8 are all attitude variables.
- **LR vs. GBM agreement**: Spearman ρ = 0.03 — weak. GBM better recovers substantive predictors because it doesn't overweight rare missing indicators.

---

## 📋 Subgroup Diagnostics & Fairness

The pipeline checks model performance across demographic subgroups:

- **Gender gap**: TPR for men = 0.70 vs. women = 0.47 (model under-detects willingness among women)
- **Race gap**: TPR for Asian NH = 0.76 vs. Hispanic = 0.57 (wide CI for Asian due to small N)
- **Age**: TPR relatively stable across groups (0.58–0.65)
- **Calibration by subgroup**: Worst for Hispanic (ECE = 0.123) and Other race (ECE = 0.174); best for White NH (ECE = 0.043) and Women (ECE = 0.040)

These findings are documented in `reports/evaluation_report.md` with bootstrap CIs.

---

## ⚙️ Configuration

All analysis parameters are centralized in `configs/default.yaml`:

- **Survey design**: wave number, sample size, weight variable, form-split variable
- **Target variable**: raw column, positive/negative/refused codes
- **Feature sets**: 4 named sets (core attitudes, knowledge, demographics, employment) + "full" and "safe" combinations
- **Skip patterns**: filter variables and eligible sample sizes
- **"Not sure" handling**: code, affected variables, treatment option (drop / own_category / midpoint)
- **Encoding**: ordinal order, nominal variables for one-hot encoding, numeric pass-through
- **Missingness**: regime (listwise / impute_indicator), refused code
- **Modeling**: random seed (42), test size, CV folds, threshold, class weight
- **Outputs**: directory paths for reports, figures, tables, model cards

Every script and notebook loads this config via `src.config`, ensuring consistency across the entire pipeline.

---

## 📋 Glossary of Terms

| Term | Simple Explanation |
|------|-------------------|
| **Survey Weight** | A number that makes each response count more or less to match the U.S. population |
| **Binary Variable** | A variable with only two values (yes/no, 1/0) |
| **Logistic Regression** | A model that predicts the probability of a yes/no outcome |
| **Gradient Boosting** | An advanced machine learning technique that combines many simple models |
| **SHAP Values** | A way to explain which features are most important for each prediction |
| **ROC-AUC** | A measure of how well a model separates the two classes (0.5 = random, 1.0 = perfect) |
| **PR-AUC** | Area under the precision-recall curve; useful when classes are imbalanced |
| **Brier Score** | Average squared difference between predicted probability and actual outcome (lower = better) |
| **ECE** | Expected Calibration Error — how well predicted probabilities match observed rates |
| **Balanced Accuracy** | Average of sensitivity and specificity; fairer than accuracy when classes are unequal |
| **Odds Ratio** | How much more (or less) likely an outcome is for a one-unit change in a feature |
| **Cramer's V** | A measure of association strength between two categorical variables |
| **VIF** | Variance Inflation Factor — detects multicollinearity (> 5 is concerning) |
| **Factor Analysis** | A technique that finds a latent (hidden) dimension underlying a set of survey items |
| **Cronbach's Alpha** | Internal consistency of a scale (> 0.70 is acceptable) |
| **KMO** | Kaiser-Meyer-Olkin statistic — measures sampling adequacy for factor analysis (> 0.80 is excellent) |
| **Feature Importance** | How much each variable contributes to the model's predictions |
| **Skip Pattern** | A survey question only asked to respondents who gave a specific earlier answer |

---

## ⚠️ Important Notes

### Treating Missing Data
- **Refused** responses are coded as 99; **"Not sure"** responses are coded as 9 in six attitude items
- Missingness is **not random** — it is associated with demographics (older respondents are more likely to have missing data). This is evidence against MCAR and supports imputation over listwise deletion.
- Two regimes are implemented: **listwise deletion** (N = 4,798, loses 56% of data) and **impute + missing indicators** (N = 10,771, near-full retention). We default to impute + indicator.
- "Not sure" can be treated three ways: drop to NaN, keep as own ordinal category, or collapse to midpoint. The default is `own_category`.

### About Survey Weights
- We use `WEIGHT_W119` to make results representative
- Without weights, results might not reflect the true U.S. population
- All percentages, model training, and evaluation metrics incorporate these weights

### Interpreting Results
- Correlation ≠ causation! We're finding patterns, not proving causes
- Models predict at the group level, not for individuals
- Results are specific to December 2022; attitudes may have changed

---

## 📚 Files Generated by This Analysis

### Data Files
| File | Description |
|------|-------------|
| `data_full.pkl` | Complete dataset saved as Python pickle |
| `data_analysis_ready.pkl` | Subset with key variables for analysis |
| `metadata.pkl` | Variable labels and coding information |

### Report Files
| File | Description |
|------|-------------|
| `reports/eda_summary.md` | EDA findings with explicit modeling recommendations |
| `reports/evaluation_report.md` | Metrics, thresholds, calibration, subgroup diagnostics |
| `reports/robustness_results.md` | Sensitivity analyses across 5 dimensions |
| `reports/latent_report.md` | Latent construct methodology, loadings, fair comparison |
| `reports/latent_robustness.md` | Latent stability and "Not sure" sensitivity |
| `reports/audit_topline_validation.md` | 30/30 published topline values validated (± 2.0 pp) |
| `reports/variable_construction.md` | Feature encoding, missingness rates, skip patterns |
| `reports/variable_dictionary.md` | Full variable dictionary |
| `reports/sample_flow.md` | Sample sizes at each pipeline stage |
| `reports/modeling_plan.md` | Detailed modeling strategy |
| `reports/codebook.csv` | Complete variable codebook |
| `reports/model_metrics.json` | Machine-readable model performance |
| `reports/run_metadata.json` | Reproducibility info (Python version, seed, config hash, package versions) |

### Model Cards
| File | Description |
|------|-------------|
| `reports/model_cards/LR_default.md` | One-page summary: weighted logistic regression |
| `reports/model_cards/GBM.md` | One-page summary: gradient boosting classifier |

### Key Figures (in `reports/figures/`)
| File | Description |
|------|-------------|
| `outcome_by_*.png` | Weighted outcome rates by age, gender, education, race, etc. |
| `cramers_v_all.png` | Cramer's V ranking of all predictors |
| `correlation_heatmap.png` | Spearman correlation heatmap for ordinal variables |
| `vif_top25.png` | Top 25 VIF values (multicollinearity check) |
| `lowess_*.png` | LOESS nonlinearity checks for quasi-continuous variables |
| `roc_comparison.png` | ROC curves for all model variants |
| `calibration_comparison.png` | Calibration plots (LR + GBM) |
| `reliability_diagrams.png` | Reliability diagrams for calibration analysis |
| `odds_ratios_LR_default.png` | Forest plot of top 20 odds ratios with bootstrap CIs |
| `subgroup_error_rates.png` | TPR/FPR/FNR by demographic subgroup |
| `latent_score_distribution.png` | Latent attitude score by outcome group |
| `phase7_1_model_comparison_matched.png` | Fair latent vs. matched baseline comparison |

### Key Tables (in `reports/tables/`)
| File | Description |
|------|-------------|
| `model_comparison.csv` | 4-model comparison on held-out test set |
| `sample_flow.csv` | Sample sizes at each pipeline stage |
| `phase7_1_model_comparison_matched.csv` | 12-model fair comparison (latent vs. baseline) |
| `phase7_1_loadings_v2.csv` | Factor loadings with "Not sure" excluded |
| `latent_psychometrics_phase7_1.csv` | Psychometric quality (alpha, KMO) |
| `latent_stability_phase7_1.csv` | Cross-validation stability of latent scores |
| `latent_eligibility_bias.csv` | Eligibility bias analysis for skip patterns |

---

## 🔗 Original Data Source

**Pew Research Center American Trends Panel Wave 119**

Published reports from this data:
- [Public Awareness of Artificial Intelligence in Everyday Activities](https://www.pewresearch.org/science/2023/02/15/public-awareness-of-artificial-intelligence-in-everyday-activities/)
- [AI in Hiring and Evaluating Workers: What Americans Think](https://www.pewresearch.org/internet/2023/04/20/ai-in-hiring-and-evaluating-workers-what-americans-think/)

---

## ❓ FAQ

**Q: Why do we use weights?**
A: Survey weights make the sample representative of all U.S. adults. Without them, certain groups might be over- or under-represented.

**Q: Why two models (plus a latent construct)?**
A: Logistic regression is simpler and easier to explain (best calibration, ECE = 0.031). Gradient boosting is more accurate (best AUC = 0.867) but harder to interpret. The latent construct tests whether a psychometrically principled single score can match raw-item predictions — it comes close (−0.011 AUC gap in fair comparison).

**Q: What is the "impute + indicator" regime?**
A: Instead of throwing away rows with missing values (which loses 56% of data), we fill in a default value and add a binary flag indicating "this value was missing." This preserves nearly the full sample (N = 10,771) without hiding the missingness.

**Q: What if SHAP isn't installed?**
A: The notebook will automatically use permutation importance instead — it's another way to measure feature importance.

**Q: How do I verify everything is working?**
A: Run `python verify_pipeline.py`. It checks every critical assumption and prints PASS/FAIL for each.

**Q: Can I modify the analysis?**
A: Absolutely! All parameters live in `configs/default.yaml`. Change the "Not sure" treatment, feature sets, random seed, or missingness regime and re-run `python run_all.py`.

---

## 🎓 Learning Resources

If you're new to data science, here are some concepts to learn more about:

1. **Survey methodology** — How surveys are designed and weighted
2. **Logistic regression** — A fundamental classification technique
3. **Gradient boosting** — A powerful machine learning method
4. **SHAP values** — Modern interpretability for ML models
5. **Fairness in ML** — Ensuring models don't discriminate
6. **Factor analysis** — Finding latent dimensions in survey data
7. **Calibration** — Ensuring predicted probabilities are meaningful
8. **Bootstrap inference** — Estimating uncertainty without parametric assumptions

---

## 📝 License and Citation

This analysis uses publicly available data from Pew Research Center. 

If you use this code or analysis, please cite:
- Pew Research Center (2022). American Trends Panel Wave 119.
- This repository

---

## 🤝 Contributing

Found a bug? Have a suggestion? Feel free to open an issue or submit a pull request!

---

*Created as part of a data science capstone project. Last updated: February 2026*
