# cogdiff

**[Browse the interactive results site](https://smkwray.github.io/cogdiff/)**

cogdiff is a Python and R pipeline that estimates latent general cognitive ability (*g*) and an observed composite score (*g\_proxy*) from the test batteries in three cohorts of the National Longitudinal Survey of Youth (NLSY79, NLSY97, and the Children of the NLSY79, CNLSY), checks how well they predict education, earnings, employment and other life outcomes, and measures how scores and their predictive power differ by sex, race/ethnicity and socioeconomic status (SES). The repository ships code, publication-locked result tables and figures. It does not ship NLSY microdata; the pipeline downloads the free public-use files when you reproduce it.

## Headline findings

- Each standard deviation (SD) of *g\_proxy* is associated with 5.3× (NLSY79) and 4.4× (NLSY97) higher odds of a bachelor's degree, and 1.9× and 2.1× higher odds of employment.
- The observed mean sex difference (*d\_g\_proxy*, male minus female) is 0.26 SD in NLSY79 and 0.06 SD in NLSY97. NLSY79 is the only cohort that passes scalar measurement invariance; its latent estimate is *d\_g* = 0.31 [0.24, 0.37].
- Males are more variable in all three cohorts (observed variance ratio 1.32–1.36).
- The earnings return per SD of cognitive ability roughly tripled between the 1960s and 1980s birth cohorts ($7.4k → $22–27k).
- Among Black respondents in NLSY97 the sex difference reverses (females outscore males by 0.16 SD). In both cohorts, Black respondents accumulate 35–40% less net worth than their cognitive scores predict.
- In the exploratory outcome modules, each SD of *g\_proxy* is associated with lower odds of arrest (odds ratio 0.65) and incarceration (0.48) in NLSY97, lower odds of poor self-rated health (0.44–0.64), and 2.3–2.5× higher odds of homeownership.

Latent comparisons are valid only for NLSY79; the other results use the observed composite *g\_proxy*, and outcome results are regression associations in observational survey data. The site presents the main findings with charts; [RESULTS.md](public_facing_docs/RESULTS.md) has the full write-up of every module, including subgroup, sibling, intergenerational, mediation and robustness results.

## Key terms

| Term | Meaning |
|---|---|
| *g* | Latent general cognitive ability, estimated with structural equation modelling (SEM) from the subtest battery |
| *g\_proxy* | Observed composite: the unweighted mean of age-residualised, z-scored subtests |
| 1 SD | One standard deviation of *g* or *g\_proxy*, about 15 points on the conventional IQ scale |
| *d* | Standardised mean difference, male minus female; positive means males score higher |
| VR | Variance ratio, male variance divided by female variance; above 1 means males are more variable |
| OR | Odds ratio per 1 SD of *g\_proxy*; 2.0 means each SD doubles the odds |

## Method

Each subtest is residualised on age (quadratic) and z-scored. A multi-group confirmatory factor model, fitted in R with `lavaan` (robust maximum likelihood, full-information handling of missing data), estimates *g* from speed, math, verbal and technical factors in the two cohorts tested with the Armed Services Vocational Aptitude Battery (ASVAB) and from a single factor in CNLSY. The pipeline tests configural, metric and scalar invariance between sexes and reports a latent mean difference only for cohorts that pass the scalar gate. Confidence intervals come from a family-level bootstrap that resamples families and refits the SEM 499 times. A specification curve varies age adjustment, sampling, model form, weights and inference. The exploratory modules regress each outcome on *g\_proxy* and age (logistic for binary outcomes, least squares for continuous ones). [METHODS.md](public_facing_docs/METHODS.md) gives the model specification, invariance thresholds, variable mapping and module list.

## Data

The [National Longitudinal Surveys](https://www.bls.gov/nls/) are panel surveys funded by the U.S. Bureau of Labor Statistics. All three cohorts are free through the [NLS Investigator](https://www.nlsinfo.org/investigator/) with no data-use contract.

| Cohort | Birth years | *N* | Cognitive battery | Waves | Years |
|---|---|---|---|---|---|
| NLSY79 | 1957–1964 | 12,686 (11,914 tested, 94%) | ASVAB, 10 subtests | 30 rounds | 1979–2022 |
| NLSY97 | 1980–1984 | 8,984 (7,127 tested, 79%) | Computer-adaptive ASVAB (CAT-ASVAB), 12 subtests | 20 rounds | 1997–2022 |
| CNLSY | Children of NLSY79 mothers | 11,545 children identified | Peabody Individual Achievement Test (PIAT), 3 subtests | 16 rounds | 1986–2014 |

## Reproduce

To read the results without running anything, open `outputs/tables/publication_results_lock/`, starting with `manuscript_results_lock.md`.

A full run needs Python 3.10+, R 4.x with the `lavaan`, `jsonlite` and `NlsyLinks` packages, and network access to nlsinfo.org. Run from the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

Rscript scripts/04_export_links_nlsylinks.R          # sibling links used by the family analyses
python scripts/13_run_pipeline.py --all --from-stage 0 --to-stage 15
python scripts/20_run_inference_bootstrap.py --variant-token family_bootstrap \
  --engine sem_refit --n-bootstrap 499 --sem-jobs 4    # set --sem-jobs to your core count
python scripts/24_build_publication_results_lock.py
python scripts/25_verify_publication_snapshot.py
python -m pytest -q
```

Stage 0 downloads the cohort archives into `data/raw/`. The bootstrap is the slow step (about 16 CPU hours for 499 replicates across cohorts); without `--n-bootstrap` it runs 150.

## Outputs and layout

| Path | Contents |
|---|---|
| `outputs/tables/publication_results_lock/` | Committed result tables and figures with a hash manifest |
| `outputs/figures/` | Committed figures |
| `public_facing_docs/` | [RESULTS](public_facing_docs/RESULTS.md) and [METHODS](public_facing_docs/METHODS.md) |
| `docs/` | Source of the results site |
| `config/` | Cohort, model, path and robustness settings |
| `scripts/`, `src/nls_pipeline/` | Pipeline stages and library code |

A run also writes the full set of tables to `outputs/tables/` and the lock bundle to `outputs/tables/publication_results_lock.zip`.

## Limitations

- Only NLSY79 passes scalar invariance by sex. NLSY97 and CNLSY latent mean differences are excluded; their observed *g\_proxy* estimates measure a different construct.
- CNLSY subgroup cells are small (as low as 31 per cell), so its race/ethnicity and SES estimates have wide intervals.
- Cross-cohort comparisons mix birth cohort with measurement age and survey era: NLSY79 outcomes were measured at ages 36–44 about 20 years before NLSY97 outcomes at similar ages.
- The batteries differ (paper ASVAB, CAT-ASVAB, PIAT). Comparing *g* across cohorts assumes they measure the same construct; the specification curve tests sensitivity to harmonisation choices.
- *g\_proxy* is an unweighted mean that does not correct for measurement error. It ranks people within a cohort but does not support direct cross-cohort level comparisons.
- NLSY79 sibling fixed-effects estimates for income and wealth rest on 7–25 families and are not reliable.
- All data are from the United States and describe these cohorts only.

## License

MIT. See [LICENSE](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button reads it.
