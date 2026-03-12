# Data access and non-redistribution

This repository does **not** include NLSY microdata.

- The file `data/raw/manifest.json` documents the expected raw input archives.
- To reproduce the full pipeline, you must obtain the relevant NLSY public-use files through official channels and place
  the raw archives under `data/raw/`.
- Do not add or redistribute restricted data through this repository.

## Derived / added variables (high level)

The pipeline may extract additional non-test variables to support exploratory reporting modules (race/SES moderation and
`g_proxy` associations with outcomes). These are mapped into canonical column names in the processed cohort CSVs.

| Canonical column | Meaning | NLSY79 raw code | NLSY97 raw code | CNLSY raw code |
|---|---|---|---|---|
| `race_ethnicity_raw` | Cohort race/ethnicity coding (cohort-specific) | `R0214700` | `R1482600` | `C0005300` |
| `race_ethnicity_3cat` | Harmonized 3-category race group (derived) | derived | derived | derived |
| `mother_education` | Mother highest grade completed (baseline) | `R0006500` | `R0554500` | `C0053500` |
| `father_education` | Father highest grade completed (baseline) | `R0007900` | `R0554800` | - |
| `parent_education` | Mean of available parent education (derived) | derived | derived | derived |
| `education_years` | Respondent highest grade completed (XRND / ever) | `T9900000` | `Z9083800` | `Y1211300` |
| `household_income` | Household/family income (selected year) | `R7006500` | `T5206900` | - |
| `net_worth` | Net worth / wealth (selected year) | `R6940103` | `Z9121900` | - |
| `annual_earnings` | Annual earnings proxy (selected year) | `R3279401` | - | - |
| `sat_math_2007_bin` | NLSY97 SAT math (binned) | - | `Z9033700` | - |
| `sat_verbal_2007_bin` | NLSY97 SAT verbal (binned) | - | `Z9033900` | - |
| `act_2007_bin` | NLSY97 ACT (binned) | - | `Z9034100` | - |

Notes:
- The raw codes correspond to the public-use release noted in each cohort’s `config/*.yml`.
- Some optional variables may be absent in alternative releases; the pipeline treats some of these as optional for portability.
