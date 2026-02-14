# Phase 5: Evaluation + Subgroup Diagnostics Report

---

## 1. Comprehensive Metrics (with 95% bootstrap CIs, N=500)

### LR (default)

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| auc | 0.8574 | [0.8271, 0.8817] |
| brier | 0.1448 | [0.1318, 0.1595] |
| ece | 0.0428 | [0.0246, 0.0639] |
| pr_auc | 0.7954 | [0.7534, 0.8329] |

### GBM

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| auc | 0.8675 | [0.8412, 0.8896] |
| brier | 0.1418 | [0.1291, 0.1562] |
| ece | 0.0485 | [0.0297, 0.0676] |
| pr_auc | 0.8001 | [0.7602, 0.8373] |

---

## 2. Threshold Policy Comparison (LR default)

| Threshold | Precision | Recall | F1 | Balanced Acc | TP | FP | FN | TN |
|-----------|-----------|--------|----|-----------  -|----|----|----|-----|
| 0.50 (default) | 0.759 | 0.600 | 0.670 | 0.745 | 470 | 163 | 281 | 1241 |
| prevalence (0.348) | 0.696 | 0.763 | 0.728 | 0.786 | 576 | 287 | 175 | 1117 |
| Youden J (0.354) | 0.699 | 0.762 | 0.729 | 0.787 | 575 | 282 | 176 | 1122 |
| max-F1 (0.354) | 0.699 | 0.762 | 0.729 | 0.787 | 575 | 282 | 176 | 1122 |

**Recommended**: Youden J (0.354) (highest balanced accuracy)

---

## 3. Calibration

### Calibration Slope + Intercept

| Model | Slope | Intercept | Assessment |
|-------|-------|-----------|------------|
| LR | 0.9741 | 0.1829 | Room for improvement |
| GBM | 0.9464 | 0.1839 | Room for improvement |

### ECE by Subgroup

| Variable | Level | N | ECE (LR) | ECE (GBM) |
|----------|-------|---|----------|-----------|
| age_category | 18-29 | 170 | 0.0947 | 0.1353 |
| age_category | 30-49 | 699 | 0.0667 | 0.0753 |
| age_category | 50-64 | 630 | 0.0485 | 0.0556 |
| age_category | 65+ | 648 | 0.0470 | 0.0301 |
| gender | A man | 943 | 0.0521 | 0.0655 |
| gender | A woman | 1195 | 0.0401 | 0.0306 |
| gender | nan | 12 | N<50, insufficient | |
| education | Less than high school | 69 | 0.1178 | 0.1278 |
| education | High school graduate | 322 | 0.0968 | 0.0767 |
| education | Some college, no degree | 461 | 0.0641 | 0.1079 |
| education | Associate's degree | 244 | 0.1109 | 0.0654 |
| education | College graduate/some pos | 557 | 0.0458 | 0.0541 |
| education | Postgraduate | 498 | 0.0575 | 0.0553 |
| race_ethnicity | White non-Hispanic | 1383 | 0.0429 | 0.0441 |
| race_ethnicity | Black non-Hispanic | 287 | 0.0794 | 0.0811 |
| race_ethnicity | Hispanic | 311 | 0.1232 | 0.1141 |
| race_ethnicity | Other | 81 | 0.1738 | 0.1772 |
| race_ethnicity | Asian non-Hispanic | 72 | 0.1526 | 0.1321 |

---

## 4. Subgroup Diagnostics

Minimum-N rule: groups with N < 50 report 'insufficient data'.

| Variable | Label | N | Prevalence | TPR | FPR | FNR |
|----------|-------|---|-----------|-----|-----|-----|
| age_category | 18-29 | 170 | 0.473 | 0.583 | 0.149 | 0.417 |
| age_category | 30-49 | 699 | 0.385 | 0.586 | 0.165 | 0.414 |
| age_category | 50-64 | 630 | 0.321 | 0.606 | 0.043 | 0.395 |
| age_category | 65+ | 648 | 0.298 | 0.648 | 0.070 | 0.352 |
| gender | A man | 943 | 0.453 | 0.699 | 0.139 | 0.301 |
| gender | A woman | 1195 | 0.289 | 0.466 | 0.091 | 0.534 |
| gender | In some other way | 12 | N<50, insufficient | | | |
| education | Less than high school | 69 | 0.323 | 0.652 | 0.089 | 0.348 |
| education | High school graduate | 322 | 0.354 | 0.522 | 0.056 | 0.478 |
| education | Some college, no degree | 461 | 0.356 | 0.609 | 0.118 | 0.391 |
| education | Associate's degree | 244 | 0.396 | 0.587 | 0.090 | 0.413 |
| education | College graduate/some post gra | 557 | 0.342 | 0.600 | 0.134 | 0.401 |
| education | Postgraduate | 498 | 0.413 | 0.708 | 0.195 | 0.292 |
| race_ethnicity | White non-Hispanic | 1383 | 0.333 | 0.616 | 0.077 | 0.384 |
| race_ethnicity | Black non-Hispanic | 287 | 0.320 | 0.583 | 0.161 | 0.417 |
| race_ethnicity | Hispanic | 311 | 0.476 | 0.565 | 0.217 | 0.435 |
| race_ethnicity | Other | 81 | 0.427 | 0.518 | 0.051 | 0.482 |
| race_ethnicity | Asian non-Hispanic | 72 | 0.432 | 0.756 | 0.145 | 0.244 |

### Bootstrap CIs for Subgroup TPR

| Variable | Label | N | TPR | 95% CI |
|----------|-------|---|-----|--------|
| age_category | 18-29 | 170 | 0.582 | [0.434, 0.730] |
| age_category | 30-49 | 699 | 0.587 | [0.500, 0.676] |
| age_category | 50-64 | 630 | 0.602 | [0.502, 0.695] |
| age_category | 65+ | 648 | 0.647 | [0.567, 0.729] |
| gender | A man | 943 | 0.697 | [0.626, 0.763] |
| gender | A woman | 1195 | 0.464 | [0.391, 0.538] |
| education | Less than high school | 69 | 0.647 | [0.370, 0.859] |
| education | High school graduate | 322 | 0.525 | [0.390, 0.660] |
| education | Some college, no degree | 461 | 0.612 | [0.485, 0.731] |
| education | Associate's degree | 244 | 0.587 | [0.430, 0.744] |
| education | College graduate/some post gra | 557 | 0.600 | [0.498, 0.696] |
| education | Postgraduate | 498 | 0.709 | [0.633, 0.781] |
| race_ethnicity | White non-Hispanic | 1383 | 0.614 | [0.549, 0.681] |
| race_ethnicity | Black non-Hispanic | 287 | 0.573 | [0.425, 0.708] |
| race_ethnicity | Hispanic | 311 | 0.571 | [0.429, 0.706] |
| race_ethnicity | Other | 81 | 0.530 | [0.288, 0.826] |
| race_ethnicity | Asian non-Hispanic | 72 | 0.764 | [0.535, 0.948] |