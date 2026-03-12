# Glossary

| Term | Meaning |
|---|---|
| `g` | General cognitive ability factor (latent) estimated from a measurement model. |
| `d_g` | Latent `g` mean difference (male − female), in Cohen’s *d* units. |
| `g_proxy` | Observed composite proxy for `g` (mean of standardized subtests used for descriptive/exploratory modules). |
| `d_g_proxy` | Mean difference (male − female) on the observed `g_proxy` composite (not latent `d_g`). |
| `IQ_diff` | Convenience transform: `15 * d_g` (IQ-point scale). |
| `VR_g` | Latent `g` variance ratio (male / female). |
| `race_ethnicity_3cat` | Harmonized 3-category race/ethnicity group used for cross-cohort exploratory summaries. |
| `parent_education` | Mean of available mother/father education variables (rowwise; exploratory SES proxy). |
| Baseline | Point estimates from a single fitted model (with delta-method SEs when available). |
| Family bootstrap | Resampling procedure that resamples at the family level and refits models to quantify uncertainty. |
| SEM-refit | Re-estimating the SEM on each bootstrap replicate (rather than reusing a fixed fit). |
| Measurement invariance | Constraints that support comparable measurement across groups (e.g., equal loadings/intercepts). |
| Metric invariance | Invariance of factor loadings (supports comparing variances/relationships more than means). |
| Scalar invariance | Invariance of intercepts (supports comparing latent means). |
| Partial scalar | Scalar invariance with a subset of intercept constraints relaxed (freed). |
| Partial-invariance stability check | A guardrail that withholds primary latent mean differences if the partial-scalar solution is not stable across cohorts or bootstrap resamples. |
| Primary | Treated as eligible for headline claims under pre-specified quality gates. |
| Exploratory / sensitivity | Computed and reported for context and robustness, but not treated as primary evidence when quality gates are not met. |
| Publication lock | A copied set of “final” artifacts (tables/figures/manifests) intended for cite-able reporting. |
