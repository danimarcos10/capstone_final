# EDA Summary: Model-Choice-Driving Diagnostics

This report documents the EDA that directly motivates modeling decisions.
Every finding leads to a concrete recommendation for the pipeline.

---

## 1. Weighted Outcome Rates by Subgroup

### age_category

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | 18-29 | 916 | 38.1% |
| 2 | 30-49 | 3,477 | 35.7% |
| 3 | 50-64 | 3,107 | 28.9% |
| 4 | 65+ | 3,237 | 29.1% |

### gender

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | A man | 4,796 | 37.6% |
| 2 | A woman | 5,855 | 28.6% |
| 3 | In some other way | 91 | 40.0% |

### education

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | Less than high school | 326 | 30.3% |
| 2 | High school graduate | 1,646 | 28.4% |
| 3 | Some college, no degree | 2,268 | 30.6% |
| 4 | Associate's degree | 1,178 | 33.5% |
| 5 | College graduate/some post grad | 2,862 | 36.7% |
| 6 | Postgraduate | 2,462 | 40.6% |

### race_ethnicity

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | White non-Hispanic | 7,044 | 29.7% |
| 2 | Black non-Hispanic | 1,428 | 36.6% |
| 3 | Hispanic | 1,458 | 38.3% |
| 4 | Other | 363 | 36.4% |
| 5 | Asian non-Hispanic | 371 | 45.8% |

### party_affiliation

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | Republican | 3,169 | 25.5% |
| 2 | Democrat | 3,533 | 38.0% |
| 3 | Independent | 2,929 | 34.9% |
| 4 | Something else | 1,057 | 35.3% |

### income_tier

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | Lower income | 2,241 | 30.4% |
| 2 | Middle income | 5,124 | 31.9% |
| 3 | Upper income | 2,590 | 40.7% |

### employment_status

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | Work full time for pay | 5,207 | 34.0% |
| 2 | Work part time for pay | 1,198 | 35.1% |
| 3 | Not currently working for pay | 871 | 36.6% |
| 4 | Unable to work due to a disability | 572 | 27.2% |
| 5 | Retired | 2,888 | 28.8% |

### applied_job_past_year

| Value | Label | N (unweighted) | Weighted % Apply |
|-------|-------|----------------|------------------|
| 1 | Yes, I have | 2,527 | 37.6% |
| 2 | No, I have not | 8,225 | 31.3% |

---

## 2. Association Strength (Weighted Cramer's V)

| Rank | Variable | Cramer's V | Interpretation |
|------|----------|-----------|---------------|
| 1 | favor_ai_review_apps | 0.4666 | STRONG |
| 2 | ai_vs_human_identify_qualified | 0.3977 | STRONG |
| 3 | ai_vs_human_see_potential | 0.3424 | STRONG |
| 4 | ai_vs_human_coworker_fit | 0.3306 | STRONG |
| 5 | ai_vs_human_treat_same | 0.3239 | STRONG |
| 6 | ai_bias_hiring_change | 0.3183 | STRONG |
| 7 | favor_ai_final_decision | 0.2964 | STRONG |
| 8 | ai_excited_vs_concerned | 0.2708 | STRONG |
| 9 | ai_interaction_frequency | 0.1825 | Moderate |
| 10 | awareness_ai_hiring | 0.1689 | Moderate |
| 11 | heard_about_ai | 0.1362 | Moderate |
| 12 | hiring_bias_problem | 0.1303 | Moderate |
| 13 | ai_knowledge_score | 0.1278 | Moderate |
| 14 | party_affiliation | 0.1064 | Moderate |
| 15 | industry | 0.1035 | Moderate |
| 16 | race_ethnicity | 0.1004 | Moderate |
| 17 | gender | 0.0972 | Weak |
| 18 | education | 0.0928 | Weak |
| 19 | age_category | 0.0821 | Weak |
| 20 | income_tier | 0.0803 | Weak |
| 21 | employment_status | 0.0620 | Weak |
| 22 | applied_job_past_year | 0.0590 | Weak |

**Modeling recommendation**:
- Strong predictors (V >= 0.10): favor_ai_review_apps, ai_vs_human_identify_qualified, ai_vs_human_see_potential, ai_vs_human_coworker_fit, ai_vs_human_treat_same, ai_bias_hiring_change, favor_ai_final_decision, ai_excited_vs_concerned, ai_interaction_frequency, awareness_ai_hiring, heard_about_ai, hiring_bias_problem, ai_knowledge_score, party_affiliation, industry, race_ethnicity
- Negligible predictors (V < 0.05): None. Consider dropping for parsimony.

---

## 3. Ordinal Correlations (Spearman rho)

| Rank | Variable | Spearman rho | |rho| | N |
|------|----------|-----------:|------:|----:|
| 1 | ai_bias_hiring_change | -0.3459 | 0.3459 | 8,714 |
| 2 | favor_ai_review_apps | -0.3108 | 0.3108 | 10,754 |
| 3 | ai_vs_human_treat_same | -0.3028 | 0.3028 | 10,736 |
| 4 | ai_vs_human_identify_qualified | -0.2614 | 0.2614 | 10,735 |
| 5 | ai_vs_human_see_potential | -0.2155 | 0.2155 | 10,736 |
| 6 | ai_interaction_frequency | -0.1632 | 0.1632 | 10,661 |
| 7 | awareness_ai_hiring | -0.1416 | 0.1416 | 10,752 |
| 8 | ai_vs_human_coworker_fit | -0.1386 | 0.1386 | 10,732 |
| 9 | heard_about_ai | -0.1353 | 0.1353 | 10,763 |
| 10 | hiring_bias_problem | -0.1105 | 0.1105 | 10,627 |
| 11 | education | 0.0916 | 0.0916 | 10,742 |
| 12 | favor_ai_final_decision | -0.0717 | 0.0717 | 10,759 |
| 13 | income_tier | 0.0673 | 0.0673 | 9,955 |
| 14 | ai_excited_vs_concerned | -0.0432 | 0.0432 | 10,704 |
| 15 | age_category | -0.0394 | 0.0394 | 10,737 |
| 16 | employment_status | -0.0283 | 0.0283 | 10,736 |

---

## 4. Multicollinearity (VIF)

**Features with VIF > 5**: 3

| Feature | VIF |
|---------|-----|
| INDUSTRYCOMBO_W119_missing | 10.8 |
| employment_status | 10.5 |
| industry_2.0 | 7.5 |

**Note**: VIF computed with `drop_first=True` (reference-category encoding).
High VIF for employment_status and INDUSTRYCOMBO_missing is expected:
INDUSTRYCOMBO is only asked to employed respondents, so the missing indicator
is structurally linked to employment status.

**Modeling recommendation**:
- For logistic regression: keep `drop_first=True`. Consider excluding the
  employment context set (Set D) if collinearity distorts coefficients.
- For tree-based models: VIF is not a concern; trees handle collinearity naturally.

---

## 5. Nonlinearity Checks

### ai_knowledge_score
- Range: [0, 6]
- Outcome range: [20.1%, 38.9%]
- LOWESS trend: **approximately monotone**
- **Recommendation**: Linear term is adequate.

### income_tier
- Range: [1, 3]
- Outcome range: [30.4%, 40.7%]
- LOWESS trend: **approximately monotone**
- **Recommendation**: Linear term is adequate.

### age_category
- Range: [1, 4]
- Outcome range: [28.9%, 38.1%]
- LOWESS trend: **approximately monotone**
- **Recommendation**: Linear term is adequate.

---

## 6. Missingness Diagnostic

- 5-fold cross-validated ROC-AUC: **0.7716 +/- 0.0103**
- Training accuracy (for reference): 0.7043
- Conclusion: Missingness is associated with observed demographics (5-fold CV AUC = 0.772 +/- 0.010). This is evidence against MCAR and supports using imputation with sensitivity checks rather than listwise deletion.

Note: This is a diagnostic for whether missingness is associated with
observed demographics. A CV AUC substantially above 0.50 is evidence
against MCAR, supporting imputation + sensitivity analysis.

---

## 7. Overall Modeling Recommendations

Based on the EDA diagnostics above:

1. **Missingness regime**: Use impute+indicator (evidence against MCAR; missingness associated with demographics). Listwise deletion loses 56% of data.
2. **'Not sure' treatment**: Use own_category (preserves information and sample size).
3. **Strongest predictors**: favor_ai_review_apps, ai_vs_human_identify_qualified, ai_vs_human_see_potential, ai_vs_human_coworker_fit, ai_vs_human_treat_same -- prioritize these in any parsimonious model.
4. **Negligible predictors**: None -- safe to drop if needed.
5. **Nonlinearity (ai_knowledge_score)**: Monotone. Linear/ordinal encoding OK.
6. **Nonlinearity (income_tier)**: Monotone. Linear/ordinal encoding OK.
7. **Nonlinearity (age_category)**: Monotone. Linear/ordinal encoding OK.
8. **Multicollinearity**: 3 features flagged (VIF>5). Use drop_first for logistic regression. employment_status + INDUSTRYCOMBO_missing are structurally linked (expected).
9. **Model classes to try**: Logistic regression (interpretable), gradient boosting (flexible), GAM (if nonlinearity found).

---

## Figures

All figures saved to `reports/figures/`:
- `outcome_by_*.png` -- outcome rates by each subgroup
- `cramers_v_all.png` -- association strength bar chart
- `correlation_heatmap.png` -- Spearman correlation matrix
- `vif_top25.png` -- VIF bar chart
- `lowess_*.png` -- nonlinearity checks