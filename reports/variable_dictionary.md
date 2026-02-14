# Variable Dictionary - ATP W119 AI Hiring Survey

## Overview

This document describes all variables used in the analysis of willingness to apply for jobs using AI in hiring.

---

## Target Variable

| Variable | Label | Coding | Role |
|----------|-------|--------|------|
| **AIWRKH4_W119** | Would you want to apply for a job with this employer [if they use AI in hiring]? | 1="Yes, I would", 2="No, I would not", 99=Refused | Original Target |
| **y_apply** | Binary target (derived) | 1=Would apply, 0=Would NOT apply, NaN=Refused | **PRIMARY TARGET** |

### Target Coding Rationale

```
Original Code 1 ("Yes, I would")     → y_apply = 1
Original Code 2 ("No, I would not")  → y_apply = 0
Original Code 99 ("Refused")         → y_apply = NaN (excluded)
```

**Interpretation**: `y=1` means the respondent would be willing to apply to a job using AI in hiring.

---

## AI-Related Predictors

| Variable | Label | Coding | Missing |
|----------|-------|--------|---------|
| AIWRKH1_W119 | Have you heard or read about employers using AI in hiring? | 1=Yes, 2=No | 99=Refused |
| AIWRKH2_a_W119 | How good/bad idea: AI reviewing job applications | 1-4 scale | 99=Refused |
| AIWRKH2_b_W119 | How good/bad idea: AI deciding who to interview | 1-4 scale | 99=Refused |
| AIWRKH3_a_W119 | Opinion: AI making final hiring decisions | Varies | 99=Refused |
| AIWRKH3_b_W119 | Opinion: AI making final hiring decisions (cont.) | Varies | 99=Refused |
| AIWRKH3_c_W119 | Opinion: AI making final hiring decisions (cont.) | Varies | 99=Refused |
| AIWRKH3_d_W119 | Opinion: AI making final hiring decisions (cont.) | Varies | 99=Refused |
| HIREBIAS1_W119 | AI better/worse than humans at avoiding bias? | 1=Better, 2=Same, 3=Worse | 99=Refused |
| HIREBIAS2_W119 | Would AI help/hurt YOU in getting hired? | 1=Help, 2=Neither, 3=Hurt | 99=Refused |
| AIKNOW_INDEX_W119 | AI knowledge score (0-6 questions correct) | 0-6 continuous | None |

---

## Demographic Predictors

| Variable | Label | Categories | Missing |
|----------|-------|------------|---------|
| F_AGECAT | Age category | 1=18-29, 2=30-49, 3=50-64, 4=65+ | None |
| F_GENDER | Gender | 1=Male, 2=Female | None |
| F_EDUCCAT2 | Education (4 categories) | 1=Less than HS, 2=HS, 3=Some college, 4=College+ | None |
| F_RACETHNMOD | Race/Ethnicity (modified) | 1=White, 2=Black, 3=Hispanic, 4=Asian, 5=Other | None |
| F_PARTY_FINAL | Party affiliation | 1=Republican, 2=Democrat, 3=Independent, 4=Other | None |

---

## Weight Variable

| Variable | Description | Notes |
|----------|-------------|-------|
| **WEIGHT_W119** | Survey weight | Use for all population estimates. Mean ≈ 1.0 |

---

## Missing Data Handling

### Strategy 1 (Default)
- Code 99 (Refused/Don't know) treated as missing (NaN)
- Complete case analysis after exclusion

### Strategy 2 (Robustness)
- Code 99 kept as explicit category
- May capture informative missingness

---

## Variables Excluded

### Leakage Prevention
The following variables are excluded from predictors to prevent leakage:
- `AIWRKH4_W119` (target variable)
- `AIWRKH4_BINARY` (derived target)
- `y_apply` (derived target)
- `WEIGHT_W119` (weight, not a predictor)
- `QKEY` (respondent ID)

---

## Data Source

**Pew Research Center American Trends Panel Wave 119**
- Dates: December 12-18, 2022
- Sample: N = 11,004 U.S. adults
- Mode: Web (English and Spanish)
