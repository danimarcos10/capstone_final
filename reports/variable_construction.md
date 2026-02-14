# Variable Construction Report

This report documents every variable used in the analysis,
its coding, encoding strategy, and missingness handling.

**Total features**: 22
**Source**: Pew Research Center ATP W119 (N=11,004)

---

## Target Variable

| Raw Column | Derived | Coding | Note |
|------------|---------|--------|------|
| AIWRKH4_W119 | y_apply | 1=Yes (code 1), 0=No (code 2), NaN=Refused (99) | NO 'Not sure' option |

**Robustness alternatives**:
- Default: Refused (99) = NaN (excluded)
- Conservative: Refused (99) = 0 (No)

---

## Feature Variables

| # | Raw Column | Readable Name | Type | Encoding | Not Sure? | Skip Pattern? |
|---|------------|---------------|------|----------|-----------|---------------|
| 1 | AIWRKH1_W119 | awareness_ai_hiring | Ordinal | Ranks 0-2 | No | No |
| 2 | AIWRKH2_a_W119 | favor_ai_review_apps | Ordinal | Ranks 0-2 | Yes (code 9) | No |
| 3 | AIWRKH2_b_W119 | favor_ai_final_decision | Ordinal | Ranks 0-2 | Yes (code 9) | No |
| 4 | AIWRKH3_a_W119 | ai_vs_human_identify_qualified | Ordinal | Ranks 0-3 | Yes (code 9) | No |
| 5 | AIWRKH3_b_W119 | ai_vs_human_treat_same | Ordinal | Ranks 0-3 | Yes (code 9) | No |
| 6 | AIWRKH3_c_W119 | ai_vs_human_see_potential | Ordinal | Ranks 0-3 | Yes (code 9) | No |
| 7 | AIWRKH3_d_W119 | ai_vs_human_coworker_fit | Ordinal | Ranks 0-3 | Yes (code 9) | No |
| 8 | HIREBIAS1_W119 | hiring_bias_problem | Ordinal | Ranks 0-2 | No | No |
| 9 | HIREBIAS2_W119 | ai_bias_hiring_change | Ordinal | Ranks 0-4 | No | Yes: HIREBIAS1_W119 in [1.0, 2.0] |
| 10 | AIKNOW_INDEX_W119 | ai_knowledge_score | Numeric | As-is | No | No |
| 11 | AI_HEARD_W119 | heard_about_ai | Ordinal | Ranks 0-2 | No | No |
| 12 | CNCEXC_W119 | ai_excited_vs_concerned | Ordinal | Ranks 0-2 | No | No |
| 13 | USEAI_W119 | ai_interaction_frequency | Ordinal | Ranks 0-4 | No | No |
| 14 | F_AGECAT | age_category | Ordinal | Ranks 0-3 | No | No |
| 15 | F_GENDER | gender | Nominal | One-hot | No | No |
| 16 | F_EDUCCAT2 | education | Ordinal | Ranks 0-5 | No | No |
| 17 | F_RACETHNMOD | race_ethnicity | Nominal | One-hot | No | No |
| 18 | F_PARTY_FINAL | party_affiliation | Nominal | One-hot | No | No |
| 19 | F_INC_TIER2 | income_tier | Ordinal | Ranks 0-2 | No | No |
| 20 | EMPLSIT_W119 | employment_status | Ordinal | Ranks 0-4 | No | No |
| 21 | JOBAPPYR_W119 | applied_job_past_year | Nominal | One-hot | No | No |
| 22 | INDUSTRYCOMBO_W119 | industry | Nominal | One-hot | No | Yes: EMPLSIT_W119 in [1.0, 2.0] |

---

## Missingness Summary

| Variable | % Not Sure | % Refused | % Structural NaN | % Any Missing |
|----------|------------|-----------|-------------------|---------------|
| INDUSTRYCOMBO_W119 | 0.0 | 0.16 | 40.96 | 41.12 |
| AIWRKH2_a_W119 | 29.66 | 0.34 | 0.0 | 30.0 |
| AIWRKH3_d_W119 | 27.73 | 0.59 | 0.0 | 28.32 |
| AIWRKH3_c_W119 | 24.78 | 0.56 | 0.0 | 25.35 |
| AIWRKH3_a_W119 | 22.56 | 0.56 | 0.0 | 23.12 |
| AIWRKH3_b_W119 | 21.93 | 0.55 | 0.0 | 22.48 |
| AIWRKH2_b_W119 | 20.62 | 0.26 | 0.0 | 20.88 |
| HIREBIAS2_W119 | 0.0 | 0.9 | 19.02 | 19.92 |
| F_INC_TIER2 | 0.0 | 5.71 | 2.14 | 7.84 |
| HIREBIAS1_W119 | 0.0 | 1.97 | 0.0 | 1.97 |
| USEAI_W119 | 0.0 | 1.27 | 0.0 | 1.27 |
| F_RACETHNMOD | 0.0 | 1.05 | 0.0 | 1.05 |
| F_PARTY_FINAL | 0.0 | 0.87 | 0.0 | 0.87 |
| CNCEXC_W119 | 0.0 | 0.84 | 0.0 | 0.84 |
| EMPLSIT_W119 | 0.0 | 0.4 | 0.0 | 0.4 |
| AIWRKH1_W119 | 0.0 | 0.34 | 0.0 | 0.34 |
| F_AGECAT | 0.0 | 0.33 | 0.0 | 0.33 |
| F_GENDER | 0.0 | 0.28 | 0.0 | 0.28 |
| F_EDUCCAT2 | 0.0 | 0.28 | 0.0 | 0.28 |
| JOBAPPYR_W119 | 0.0 | 0.22 | 0.0 | 0.22 |
| AI_HEARD_W119 | 0.0 | 0.07 | 0.0 | 0.07 |
| AIKNOW_INDEX_W119 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## Encoding Details

### Ordinal Variables (integer-ranked)

- **AIWRKH1_W119** (awareness_ai_hiring): [1, 2, 3] -> ranks 0-2
- **AIWRKH2_a_W119** (favor_ai_review_apps): [1, 9, 2] -> ranks 0-2
- **AIWRKH2_b_W119** (favor_ai_final_decision): [1, 9, 2] -> ranks 0-2
- **AIWRKH3_a_W119** (ai_vs_human_identify_qualified): [1, 3, 9, 2] -> ranks 0-3
- **AIWRKH3_b_W119** (ai_vs_human_treat_same): [1, 3, 9, 2] -> ranks 0-3
- **AIWRKH3_c_W119** (ai_vs_human_see_potential): [1, 3, 9, 2] -> ranks 0-3
- **AIWRKH3_d_W119** (ai_vs_human_coworker_fit): [1, 3, 9, 2] -> ranks 0-3
- **HIREBIAS1_W119** (hiring_bias_problem): [1, 2, 3] -> ranks 0-2
- **HIREBIAS2_W119** (ai_bias_hiring_change): [1, 2, 3, 4, 5] -> ranks 0-4
- **AI_HEARD_W119** (heard_about_ai): [1, 2, 3] -> ranks 0-2
- **CNCEXC_W119** (ai_excited_vs_concerned): [1, 3, 2] -> ranks 0-2
- **USEAI_W119** (ai_interaction_frequency): [1, 2, 3, 4, 5] -> ranks 0-4
- **EMPLSIT_W119** (employment_status): [1, 2, 3, 4, 5] -> ranks 0-4
- **F_AGECAT** (age_category): [1, 2, 3, 4] -> ranks 0-3
- **F_EDUCCAT2** (education): [1, 2, 3, 4, 5, 6] -> ranks 0-5
- **F_INC_TIER2** (income_tier): [1, 2, 3] -> ranks 0-2

### Numeric Variables (continuous, as-is)

- **AIKNOW_INDEX_W119** (ai_knowledge_score)

### Nominal Variables (one-hot encoded)

- **F_GENDER** (gender)
- **F_RACETHNMOD** (race_ethnicity)
- **F_PARTY_FINAL** (party_affiliation)
- **JOBAPPYR_W119** (applied_job_past_year)
- **INDUSTRYCOMBO_W119** (industry)

---

## Skip Patterns

Variables with structural missingness due to questionnaire logic:

- **HIREBIAS2_W119**: Only asked if HIREBIAS1_W119 in [1.0, 2.0] (eligible N ~ 8,911)
- **INDUSTRYCOMBO_W119**: Only asked if EMPLSIT_W119 in [1.0, 2.0] (eligible N ~ 6,497)

---

## 'Not Sure' Handling

Variables with 'Not sure' (code 9) as a distinct response option:

- AIWRKH2_a_W119 (favor_ai_review_apps)
- AIWRKH2_b_W119 (favor_ai_final_decision)
- AIWRKH3_a_W119 (ai_vs_human_identify_qualified)
- AIWRKH3_b_W119 (ai_vs_human_treat_same)
- AIWRKH3_c_W119 (ai_vs_human_see_potential)
- AIWRKH3_d_W119 (ai_vs_human_coworker_fit)

Three treatments tested as sensitivity analysis:
1. **drop**: Code 9 -> NaN (excluded via listwise deletion)
2. **own_category**: Code 9 kept as a distinct level
3. **midpoint**: Code 9 replaced with scale midpoint
