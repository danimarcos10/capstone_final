# Latent Attitude Construct — Item Selection

## Purpose
This document defines the item set used to construct a single latent attitude score toward AI in hiring. The latent score represents an underlying "pro-AI hiring disposition" that is not directly observed but inferred from responses to multiple correlated attitude questions.

## Why a Latent Construct?
- **Theoretical**: Attitudes toward AI in hiring likely reflect a single underlying disposition rather than independent opinions on each question.
- **Methodological**: Reduces dimensionality (9 raw items → 1 latent score), improving parsimony and reducing multicollinearity.
- **Survey-native**: Polychoric factor analysis respects the ordinal nature of Likert-type items (standard practice in psychometrics).
- **Interpretability**: A single "pro-AI attitude" score is easier to communicate than 9 separate coefficients.

---

## Item Set: Core Attitude Battery (9 items)

All items from the ATP W119 AI-hiring questionnaire block, excluding demographics and knowledge.

| Raw Column | Readable Name | Question Text (paraphrased) | Response Scale | Skip Pattern? |
|-----------|---------------|----------------------------|----------------|--------------|
| **AIWRKH1_W119** | awareness_ai_hiring | How much have you heard about AI being used in hiring? | 1=A lot, 2=A little, 3=Nothing at all | No |
| **AIWRKH2_a_W119** | favor_ai_review_apps | Do you favor or oppose AI being used to **review** job applications? | 1=Favor, 9=Not sure, 2=Oppose | No |
| **AIWRKH2_b_W119** | favor_ai_final_decision | Do you favor or oppose AI being used to make **final hiring decisions**? | 1=Favor, 9=Not sure, 2=Oppose | No |
| **AIWRKH3_a_W119** | ai_vs_human_identify_qualified | Would AI be better or worse than humans at **identifying the most qualified** candidates? | 1=Better, 3=About the same, 9=Not sure, 2=Worse | No |
| **AIWRKH3_b_W119** | ai_vs_human_treat_same | Would AI be better or worse than humans at **treating all applicants the same way**? | 1=Better, 3=About the same, 9=Not sure, 2=Worse | No |
| **AIWRKH3_c_W119** | ai_vs_human_see_potential | Would AI be better or worse than humans at **seeing potential in people beyond job description**? | 1=Better, 3=About the same, 9=Not sure, 2=Worse | No |
| **AIWRKH3_d_W119** | ai_vs_human_coworker_fit | Would AI be better or worse than humans at **determining if someone would fit in** with coworkers? | 1=Better, 3=About the same, 9=Not sure, 2=Worse | No |
| **HIREBIAS1_W119** | hiring_bias_problem | How much of a problem is bias in hiring and promotions? | 1=Major problem, 2=Minor problem, 3=Not a problem at all | No |
| **HIREBIAS2_W119** | ai_bias_hiring_change | If AI were used, would bias in hiring get better or worse? | 1=Definitely better, 2=Probably better, 3=Not make much difference, 4=Probably worse, 5=Definitely worse | **Yes** — only asked if HIREBIAS1 ∈ {1, 2} |

---

## Coding Direction

**Goal**: Higher latent score = more positive/favorable toward AI in hiring.

To achieve this, we **reverse-code** items where the original coding is inverted:

| Item | Original Coding | Direction Needed | Action |
|------|----------------|------------------|--------|
| AIWRKH1 | 1=A lot aware → 3=Nothing | **Reverse** | Reverse so higher = more aware |
| AIWRKH2_a | 1=Favor → 2=Oppose | **Keep** | Higher already = more opposition → need to reverse |
| AIWRKH2_b | 1=Favor → 2=Oppose | **Keep** | Same as above |
| AIWRKH3_a | 1=Better → 2=Worse | **Keep** | Higher already = worse |
| AIWRKH3_b | 1=Better → 2=Worse | **Keep** | Same |
| AIWRKH3_c | 1=Better → 2=Worse | **Keep** | Same |
| AIWRKH3_d | 1=Better → 2=Worse | **Keep** | Same |
| HIREBIAS1 | 1=Major problem → 3=Not a problem | **Reverse** | Higher should = less concerned about bias |
| HIREBIAS2 | 1=Definitely better → 5=Definitely worse | **Reverse** | Higher should = thinks AI improves bias |

**Implementation**: In `src/latent.py`, we apply consistent reverse-coding so that for all items:
- **Low values** = negative/opposed to AI
- **High values** = positive/favorable toward AI

---

## Handling "Not Sure" (Code 9)

Six items have "Not sure" as a response option:
- AIWRKH2_a, AIWRKH2_b, AIWRKH3_a, AIWRKH3_b, AIWRKH3_c, AIWRKH3_d

**Treatment**: 
- **Default**: Treat code 9 as **own category** (middle position on ordinal scale).
- **Rationale**: "Not sure" is substantive information (uncertainty) rather than missing data. Keeping it preserves sample size (N~8,625 vs N~4,201 if dropped).
- **Sensitivity check**: Compare latent model performance under `own_category` vs `drop` treatment (Phase 7 robustness).

---

## Skip Pattern: HIREBIAS2

**Eligibility rule**: HIREBIAS2 is only asked if respondent said bias is a "Major" or "Minor" problem (HIREBIAS1 ∈ {1, 2}).
- **Eligible N**: ~8,911 (81% of sample)
- **Ineligible N**: ~2,093 who said bias is "Not a problem"

**Handling**:
1. **For latent model fitting**: Only fit the factor model on respondents who answered HIREBIAS2 (i.e., where HIREBIAS2 is not structurally missing).
2. **For imputation**: Use missing indicator for HIREBIAS2 if needed, but be aware that missingness is **informative** (those who think bias isn't a problem were skipped).
3. **Alternative**: Fit two separate latent models (one with HIREBIAS2, one without) and compare performance.

---

## Excluded Variables

The following variables are **not** included in the latent construct:

### Excluded: Demographics
- F_AGECAT, F_GENDER, F_EDUCCAT2, F_RACETHNMOD, F_PARTY_FINAL, F_INC_TIER2

**Rationale**: Demographics are not attitudes. They are used as predictors alongside the latent score, not as components of it.

### Excluded: Knowledge & General AI Orientation
- AIKNOW_INDEX_W119 (knowledge score)
- AI_HEARD_W119 (heard about AI in general)
- CNCEXC_W119 (excited vs concerned about AI generally)
- USEAI_W119 (frequency of AI use)

**Rationale**: These measure general AI awareness/orientation, not specific attitudes toward AI in hiring. We test them separately as "knowledge" features.

### Excluded: Employment Context
- EMPLSIT_W119, JOBAPPYR_W119, INDUSTRYCOMBO_W119

**Rationale**: These are situational/behavioral, not attitudinal.

---

## Latent Modeling Method

**Primary approach**: **Ordinal Factor Analysis via Polychoric Correlations**

1. **Polychoric correlation matrix**: Compute pairwise correlations that respect ordinal nature (not Pearson).
2. **Factor extraction**: 1-factor solution (extracting the single "pro-AI hiring" dimension).
3. **Factor scores**: Compute standardized scores for each respondent using regression method.
4. **Optional**: Also fit 2-factor solution to test if attitudes split into multiple dimensions.

**Fallback**: If polychoric/factor analysis is unavailable, use **PCA on ordinal-encoded items** as a baseline (clearly labeled as suboptimal).

---

## Expected Validity Evidence

If the latent construct is valid, we expect:
- **High factor loadings** on all 9 items (|loading| > 0.5).
- **Cronbach's alpha** > 0.7 (internal consistency).
- **Predictive validity**: Latent score should have high odds ratio / feature importance in predicting `y_apply`.
- **Convergent validity**: Latent score should correlate strongly with raw attitude items.

---

## Sample Size for Latent Modeling

| Regime | Treatment | N (train) | Note |
|--------|-----------|-----------|------|
| **Full attitude set** | own_category | ~8,625 | All 9 items available |
| **Exclude HIREBIAS2** | own_category | ~10,771 | Only 8 items, but larger sample |
| **Drop "Not sure"** | drop | ~4,201 | 6 items with NS → substantial data loss |

**Recommended**: Fit latent model on **full 9-item set with own_category treatment** (maximizes both item coverage and sample size).

---

## References

- Lord, F. M., & Novick, M. R. (1968). *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Jöreskog, K. G. (1994). "On the estimation of polychoric correlations and their asymptotic covariance matrix." *Psychometrika*, 59(3), 381-389.
- Revelle, W. (2023). *psych: Procedures for Psychological, Psychometric, and Personality Research*. R package.

---

*Document created: 2026-02-10*  
*Last updated: 2026-02-10*  
*Author: Phase 7 Implementation*
