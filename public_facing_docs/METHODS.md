# Methods

## Glossary

| Term | Meaning |
|------|---------|
| *g* | General cognitive ability factor (latent), estimated from a measurement model fitted to subtest batteries |
| *g\_proxy* | Observed composite proxy for *g*: unit-weighted mean of age-residualised, z-scored subtests. Used for exploratory modules and when latent estimation is not feasible. Different construct than latent *g* |
| *d\_g* | Latent *g* mean difference (male − female), in Cohen's *d* units |
| *d\_g\_proxy* | Mean difference (male − female) on the observed *g\_proxy* composite |
| IQ\_diff | Convenience transform: `15 × d_g`. 1 SD of *g* or *g\_proxy* ≈ 15 IQ points (conventional IQ scale: mean = 100, SD = 15) |
| VR\_g | Latent *g* variance ratio (male variance / female variance). VR > 1 = males more variable |
| race\_ethnicity\_3cat | Harmonised 3-category race/ethnicity group used for cross-cohort reporting: Black, Hispanic, Non-Black non-Hispanic |
| parent\_education | Mean of available mother/father highest grade completed (rowwise); used as SES proxy for moderation analyses |
| Baseline | Point estimates from a single fitted model (with delta-method SEs when available) |
| Family bootstrap | Resampling procedure that resamples at the family level and refits the full SEM on each replicate to quantify uncertainty |
| SEM-refit | Re-estimating the structural equation model on each bootstrap replicate (rather than reusing a fixed fit) |
| Measurement invariance | Constraints that support comparable measurement across groups (e.g. equal factor loadings and/or intercepts across sex or race/ethnicity groups) |
| Metric invariance | Invariance of factor loadings across groups — supports comparing variances and correlations, but not means |
| Scalar invariance | Invariance of intercepts across groups — required for comparing latent means |
| Partial scalar | Scalar invariance with a subset of intercept constraints relaxed. Subject to the partial-invariance stability check |
| Partial-invariance stability check | A guardrail: if the set of freed intercepts is not stable across cohorts or bootstrap resamples, primary latent *d\_g* is withheld |
| Publication lock | A pinned copy of all result tables, figures, and hash manifests intended for citable reporting. Located at `outputs/tables/publication_results_lock/` |

---

## Model specification

### Subtest batteries

| Cohort | Subtests | Factor structure |
|--------|----------|-----------------|
| **NLSY79** | 10 ASVAB subtests: General Science (GS), Arithmetic Reasoning (AR), Word Knowledge (WK), Paragraph Comprehension (PC), Numerical Operations (NO), Coding Speed (CS), Auto & Shop Information (AS), Math Knowledge (MK), Mechanical Comprehension (MC), Electronics Information (EI) | Hierarchical: speed (NO, CS), math (AR, MK), verbal (WK, PC), technical (GS, AS, MC, EI) → higher-order *g* |
| **NLSY97** | 12 CAT-ASVAB subtests: same domains, split Auto/Shop, plus Assembling Objects; 10 adaptive + 2 speeded | Same hierarchical structure |
| **CNLSY** | PPVT, PIAT Math, PIAT Reading Recognition, PIAT Reading Comprehension, Digit Span | Single-factor *g* |

### Age residualisation

Before SEM fitting, each subtest is residualised against age using quadratic OLS regression (age + age²), then z-scored within the full sample. This removes age-related variance while preserving individual differences.

### SEM estimation

Multi-group CFA models are fitted in R using `lavaan` with:

- **Estimator**: MLR (maximum likelihood with robust standard errors)
- **Missing data**: FIML (full information maximum likelihood)
- **Latent scaling**: `std.lv = TRUE` (latent variances standardised in the reference group)
- **Reference group**: female (latent mean fixed to 0 for females)
- **Convergence**: up to 5 optimiser strategies attempted per model (default nlminb, gradient-check-disabled, bounded, extended nlminb, BFGS) with admissibility checks (no negative latent variances, SEs available)

### Computing *d\_g* and VR\_g

Since female is the reference group with latent mean constrained to 0:

- **d\_g** = male latent mean (directly from the SEM, already in SD units because `std.lv = TRUE`)
- **IQ\_diff** = 15 × d\_g
- **VR\_g** = male latent variance / female latent variance

### Observed *g\_proxy*

*g\_proxy* is the unit-weighted mean of age-residualised, z-scored subtests. It does not require scalar invariance and is used for:

- Cohorts where latent *d\_g* is excluded (invariance gates not met)
- All exploratory extensions (race/ethnicity disaggregation, SES moderation, outcome associations, SAT/ACT validity)
- Sibling fixed-effects and intergenerational transmission models

---

## Why scalar invariance matters for latent means

Latent mean comparisons require that both factor loadings (metric invariance) and intercepts (scalar invariance) are equivalent across groups. If intercepts differ between groups, an apparent "latent mean difference" can partially reflect measurement non-equivalence rather than a true construct difference.

The pipeline tests invariance in three steps — configural, metric, scalar — and applies the following gatekeeping thresholds (change relative to the previous step):

| Gate | ΔCFI ≥ | ΔRMSEA ≤ | ΔSRMR ≤ |
|------|--------|----------|---------|
| **Metric** (equal loadings) | -0.010 | 0.015 | 0.030 |
| **Scalar** (equal intercepts) | -0.010 | 0.015 | 0.015 |

If a cohort fails scalar invariance, the pipeline attempts **partial scalar** models (freeing a subset of intercepts) subject to:

- Maximum 2 freed intercepts per factor, or 20% of indicators (whichever is larger)
- At least 50% of indicators must remain invariant per factor
- **Partial-invariance stability check**: the set of freed intercepts must replicate in ≥70% of 200 bootstrap resamples and show ≥70% overlap with the freed set in comparison cohorts

If any gate fails, latent *d\_g* for that cohort is excluded from confirmatory tables. Currently freed intercepts:

- NLSY79: GS~1, AS~1
- NLSY97: PC~1
- CNLSY: none

Current status:

| Cohort | Sex scalar invariance | Race/ethnicity scalar invariance | Latent *d\_g* reported? |
|--------|----------------------|-------------------------------|------------------------|
| NLSY79 | Passed | Mixed (metric failed ΔCFI, scalar passed) | Yes |
| NLSY97 | Failed (ΔCFI) | Passed | No |
| CNLSY | Failed (ΔCFI + ΔRMSEA) | Failed (ΔRMSEA) | No |

See `outputs/tables/confirmatory_exclusions.csv` and `outputs/tables/analysis_tiers.csv`.

---

## Inference: family bootstrap with SEM-refit

Publication-grade uncertainty is produced by family bootstrap resampling (stage 20):

1. Resample families (not individuals) with replacement — preserves sibling structure
2. Refit the full multi-group SEM on each bootstrap replicate
3. Extract *d\_g* and VR\_g from each refit
4. Compute percentile-based 95% CIs from 499 replicates

This is computationally intensive (~16 CPU hours for 499 replicates across cohorts). Output tables:

- `g_mean_diff_family_bootstrap.csv`
- `g_variance_ratio_family_bootstrap.csv`
- `g_proxy_mean_diff_family_bootstrap.csv`
- `g_proxy_variance_ratio_family_bootstrap.csv`

Baseline (single-fit) tables with delta-method SEs are also provided:

- `g_mean_diff.csv`
- `g_variance_ratio.csv`

---

## Exploratory modules

These use observed *g\_proxy* (not latent *g*) and do not require scalar invariance:

| Module | Description | Key output tables |
|--------|------------|-------------------|
| Race × sex interaction | Sex differences in *g\_proxy* disaggregated by race/ethnicity group | `race_sex_group_estimates.csv`, `race_sex_interaction_summary.csv` |
| SES moderation | Sex differences in *g\_proxy* by parental education tercile | `ses_moderation_group_estimates.csv` |
| Income/wealth associations | OLS regression of *g\_proxy* on earnings, household income, net worth | `g_income_wealth_associations.csv` |
| SAT/ACT validity | *g\_proxy* correlation with SAT/ACT bins (NLSY97 only) | `g_sat_act_validity.csv`, `g_sat_act_validity_by_race.csv`, `g_sat_act_validity_by_ses.csv` |
| Degree attainment | Logistic models for BA+ and graduate degree thresholds | `degree_threshold_outcomes.csv`, `explicit_degree_outcomes.csv` |
| Employment | Logistic models for employment status | `g_employment_outcomes.csv` |
| Labour-market dynamics | Two-wave income trajectories, volatility, employment persistence | `nlsy97_income_earnings_trajectories.csv`, `nlsy97_employment_persistence.csv`, etc. |
| Occupation quality | Management entry, Job Zone complexity, education-job match, pay mismatch | `nlsy79_high_skill_occupation_outcome.csv`, `nlsy79_job_zone_complexity_outcome.csv`, etc. |
| Sibling fixed effects | Within-family *g\_proxy* → outcome associations | `sibling_fe_g_outcome.csv`, `sibling_discordance.csv` |
| Intergenerational | Mother → child *g\_proxy* transmission (CNLSY) | `intergenerational_g_transmission.csv` |
| CNLSY carryover | Childhood *g\_proxy* → adult outcomes | `cnlsy_adult_outcome_associations.csv` |
| Subtest profiles | Subtest-level *d* and log-VR profiles per cohort | `subtest_predictive_validity.csv`, `subtest_profile_tilt.csv` |
| Nonlinearity | Quadratic and threshold models for *g\_proxy* → outcomes | `nonlinear_threshold_outcome_summary.csv` |
| Mediation | Education, employment, and job-complexity as mediators of *g* → earnings | `nlsy79_mediation_summary.csv` |
| Cohort comparison | Age-matched cross-cohort validity contrasts | `age_matched_cross_cohort_contrasts.csv`, `cross_cohort_pattern_stability.csv` |
| Robustness | Sampling, age-adjustment, model-form, inference, and weight variants | `specification_stability_summary.csv` |

---

## Data access

This repository does **not** include NLSY microdata. To reproduce the pipeline:

1. Obtain the NLSY public-use files through official channels:
   - [NLS Investigator](https://www.nlsinfo.org/investigator/) (free, no contract required)
   - [NLSY79 overview](https://www.bls.gov/nls/nlsy79.htm)
   - [NLSY97 overview](https://www.bls.gov/nls/nlsy97.htm)
   - [CNLSY overview](https://www.bls.gov/nls/nlsy79-children.htm)
2. Place raw archives under `data/raw/` according to `data/raw/manifest.json`
3. Do not redistribute restricted or microdata through this repository

### Variable mapping

The pipeline maps raw NLSY codes into canonical column names:

| Canonical column | Meaning | NLSY79 | NLSY97 | CNLSY |
|-----------------|---------|--------|--------|-------|
| `race_ethnicity_raw` | Cohort race/ethnicity coding | `R0214700` | `R1482600` | `C0005300` |
| `race_ethnicity_3cat` | Harmonised 3-category group | derived | derived | derived |
| `mother_education` | Mother highest grade completed | `R0006500` | `R0554500` | `C0053500` |
| `father_education` | Father highest grade completed | `R0007900` | `R0554800` | — |
| `parent_education` | Mean of available parent education | derived | derived | derived |
| `education_years` | Respondent highest grade completed | `T9900000` | `Z9083800` | `Y1211300` |
| `household_income` | Household/family income | `R7006500` | `T5206900` | — |
| `net_worth` | Net worth / wealth | `R6940103` | `Z9121900` | — |
| `annual_earnings` | Annual earnings proxy | `R3279401` | — | — |
| `sat_math_2007_bin` | SAT math (binned) | — | `Z9033700` | — |
| `sat_verbal_2007_bin` | SAT verbal (binned) | — | `Z9033900` | — |
| `act_2007_bin` | ACT (binned) | — | `Z9034100` | — |

Raw codes correspond to the public-use release specified in each cohort's `config/*.yml`.

---

## Reproduction

### Reading results without rerunning

No data or runtime dependencies needed. The locked results bundle is committed at `outputs/tables/publication_results_lock/`. Start with:

- `manuscript_results_lock.md` — headline numbers and quality gates
- Individual CSV files — full tables
- `publication_results_lock.zip` — complete citable bundle with hash manifest

### Full reproduction

Requires Python 3.10+, R 4.x (with `lavaan`), and NLSY public-use archives.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src

# 2. Place NLSY archives in data/raw/ per data/raw/manifest.json

# 3. Run pipeline (stages 00-15)
python scripts/13_run_pipeline.py --all --from-stage 0 --to-stage 15

# 4. Bootstrap inference (499 SEM refits per cohort, ~16 CPU hours)
python scripts/20_run_inference_bootstrap.py \
  --variant-token family_bootstrap --engine sem_refit \
  --n-bootstrap 499 --sem-jobs 16 --sem-threads-per-job 1

# 5. Build and verify publication lock
python scripts/24_build_publication_results_lock.py
python scripts/25_verify_publication_snapshot.py
```

### Verification

```bash
# Test suite
python -m pytest -q -p no:cacheprovider

# Snapshot integrity
python scripts/25_verify_publication_snapshot.py

# Portability check
python scripts/99_portability_smoke_check.py --project-root "$(pwd)"
```

### Output locations

| Path | Contents |
|------|----------|
| `outputs/tables/` | All result tables (CSV) |
| `outputs/figures/` | All figures (PNG) |
| `outputs/tables/publication_results_lock/` | Pinned publication bundle with hash manifests |
| `outputs/tables/publication_results_lock.zip` | Zip of publication bundle |
