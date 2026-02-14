# Topline Validation Report

**Pipeline validation**: 30/30 published topline values replicated within tolerance.
**Max absolute deviation**: 1.0 percentage points.

## Results

| Variable | Code | Published % | Computed % | Deviation | OK? | Note |
|----------|------|-------------|------------|-----------|-----|------|
| AIWRKH1_W119 | 1 | 7 | 7.0 | 0.0 | YES | ASK ALL |
| AIWRKH1_W119 | 2 | 32 | 32.1 | 0.1 | YES | ASK ALL |
| AIWRKH1_W119 | 3 | 61 | 60.9 | 0.1 | YES | ASK ALL |
| AIWRKH4_W119 | 1 | 32 | 33.0 | 1.0 | YES | ASK ALL |
| AIWRKH4_W119 | 2 | 66 | 67.0 | 1.0 | YES | ASK ALL |
| AIWRKH2_a_W119 | 1 | 28 | 28.0 | 0.0 | YES | ASK ALL |
| AIWRKH2_a_W119 | 2 | 41 | 41.6 | 0.6 | YES | ASK ALL |
| AIWRKH2_a_W119 | 9 | 30 | 30.4 | 0.4 | YES | ASK ALL |
| AIWRKH2_b_W119 | 1 | 7 | 7.1 | 0.1 | YES | ASK ALL |
| AIWRKH2_b_W119 | 2 | 71 | 71.2 | 0.2 | YES | ASK ALL |
| AIWRKH2_b_W119 | 9 | 22 | 21.7 | 0.3 | YES | ASK ALL |
| AIWRKH3_a_W119 | 1 | 27 | 27.4 | 0.4 | YES | ASK ALL |
| AIWRKH3_a_W119 | 2 | 23 | 23.1 | 0.1 | YES | ASK ALL |
| AIWRKH3_a_W119 | 3 | 26 | 26.0 | 0.0 | YES | ASK ALL |
| AIWRKH3_a_W119 | 9 | 23 | 23.5 | 0.5 | YES | ASK ALL |
| HIREBIAS1_W119 | 1 | 37 | 38.0 | 1.0 | YES | ASK ALL |
| HIREBIAS1_W119 | 2 | 42 | 42.9 | 0.9 | YES | ASK ALL |
| HIREBIAS1_W119 | 3 | 19 | 19.1 | 0.1 | YES | ASK ALL |
| EMPLSIT_W119 | 1 | 48 | 48.3 | 0.3 | YES | ASK ALL |
| EMPLSIT_W119 | 2 | 12 | 12.4 | 0.4 | YES | ASK ALL |
| EMPLSIT_W119 | 3 | 11 | 11.5 | 0.5 | YES | ASK ALL |
| EMPLSIT_W119 | 4 | 7 | 7.1 | 0.1 | YES | ASK ALL |
| EMPLSIT_W119 | 5 | 21 | 20.7 | 0.3 | YES | ASK ALL |
| JOBAPPYR_W119 | 1 | 26 | 26.5 | 0.5 | YES | ASK ALL |
| JOBAPPYR_W119 | 2 | 73 | 73.5 | 0.5 | YES | ASK ALL |
| HIREBIAS2_W119 | 1 | 10 | 10.0 | 0.0 | YES | SKIP: HIREBIAS1_W119 in [1.0, 2.0] (N=8,911) |
| HIREBIAS2_W119 | 2 | 44 | 44.3 | 0.3 | YES | SKIP: HIREBIAS1_W119 in [1.0, 2.0] (N=8,911) |
| HIREBIAS2_W119 | 3 | 32 | 32.5 | 0.5 | YES | SKIP: HIREBIAS1_W119 in [1.0, 2.0] (N=8,911) |
| HIREBIAS2_W119 | 4 | 9 | 9.3 | 0.3 | YES | SKIP: HIREBIAS1_W119 in [1.0, 2.0] (N=8,911) |
| HIREBIAS2_W119 | 5 | 4 | 4.0 | 0.0 | YES | SKIP: HIREBIAS1_W119 in [1.0, 2.0] (N=8,911) |

## Notes

- Published values from ATP W119 Topline.pdf (pp. 16-19).
- Weighted using `WEIGHT_W119`; Refused (code 99) excluded from denominator.
- Skip-pattern items validated among eligible respondents only.
- Tolerance: +/-2.0 percentage points (accounts for rounding in published toplines).
- Validated against N = 11,004 respondents.