# Results

This repository intentionally avoids embedding microdata in the repo. Results are provided as tables/figures and a
publication lock bundle.

## Reading-First

- `outputs/tables/results_snapshot.md`: short, human-readable summary
- `outputs/tables/publication_results_lock/manuscript_results_lock.md`: manuscript-ready methods/results lock
- `outputs/tables/publication_results_lock.zip`: the complete cite-able bundle (tables, figures, manifests)

## Primary results

| Artifact | What it is |
|---|---|
| `outputs/tables/g_mean_diff.csv` | Baseline primary latent `d_g` (only where eligible) |
| `outputs/tables/g_variance_ratio.csv` | Baseline latent `VR_g` (male/female) |
| `outputs/tables/g_mean_diff_family_bootstrap.csv` | Family bootstrap latent `d_g` + CIs |
| `outputs/tables/g_variance_ratio_family_bootstrap.csv` | Family bootstrap `VR_g` + CIs |
| `outputs/tables/g_proxy_*_family_bootstrap.csv` | Family bootstrap observed `g_proxy` mean/VR estimands |
| `outputs/tables/confirmatory_exclusions.csv` | Reasons a cohort/estimand is withheld from primary reporting |
| `outputs/figures/*forestplot.png` | Forest plots (primary and bootstrap-based, if present) |
| `outputs/tables/publication_results_lock/` | Copied "final" artifacts with hashes |
| `outputs/tables/publication_results_lock/publication_results_lock_manifest.csv` | Hash manifest for bundle contents |

<details>
<summary>Exploratory modules (observed <code>g_proxy</code>)</summary>

These modules use an observed composite `g_proxy` (not latent `d_g`) and should not be interpreted
as identical to latent `d_g`. CNLSY subgroup estimates have limited precision due to small samples.

| Artifact | What it is |
|---|---|
| `outputs/tables/race_sex_group_estimates.csv` | Race/ethnicity-disaggregated sex differences on observed `g_proxy` (per group) |
| `outputs/tables/race_sex_interaction_summary.csv` | Heterogeneity/interaction summary for race/ethnicity-disaggregated `g_proxy` sex differences |
| `outputs/tables/racial_changes_over_time.csv` | Simple trend diagnostic of race/ethnicity-group differences across cohorts |
| `outputs/tables/ses_moderation_group_estimates.csv` | SES-bin (e.g., parent-education terciles) sex differences on observed `g_proxy` |
| `outputs/tables/ses_moderation_summary.csv` | SES-bin heterogeneity/interaction summary |
| `outputs/tables/g_income_wealth_associations.csv` | Associations between `g_proxy` and income/wealth outcomes (when available) |
| `outputs/tables/asvab_life_outcomes_by_sex.csv` | Within-sex associations between `g_proxy` and selected outcomes |
| `outputs/tables/g_sat_act_validity.csv` | NLSY97 `g_proxy` associations with SAT/ACT bins (external validity check) |

</details>

## Note on missing primary `d_g`

Some cohorts may not have primary baseline latent `d_g` reported. When this occurs, consult:
- `outputs/tables/confirmatory_exclusions.csv` (the reason), and
- the family bootstrap tables and/or `g_proxy` tables for sensitivity summaries.
