# FAQ

## What is `d_g`?

`d_g` is the **latent `g` mean difference** (male − female) estimated from a multi-group SEM/CFA measurement model.
It is reported in Cohen’s *d* units, and the pipeline also reports `IQ_diff = 15 * d_g` as a familiar scale transform.

## What is `VR_g`?

`VR_g` is the **latent `g` variance ratio** (male / female).

## What is `g_proxy` and why does it exist?

`g_proxy` is an **observed composite proxy** for `g`. It is useful as a sensitivity/alternative estimand because it does
not require scalar invariance for latent mean comparisons. However, it is a **different construct** than latent `d_g`,
and should not be treated as a drop-in replacement for latent mean differences.

## Why might primary `d_g` be missing for a cohort?

Latent mean differences depend on measurement assumptions. If the pipeline’s measurement invariance gates (including the
partial-invariance stability check for partial-scalar solutions) indicate that latent means are not sufficiently comparable,
primary baseline `d_g` is withheld by design.

See `outputs/tables/confirmatory_exclusions.csv` for cohort-level reasons.

## What’s the difference between “baseline” and “family bootstrap” results?

- **Baseline** tables summarize point estimates from a single fitted model (with delta-method SEs when available).
- **Family bootstrap (SEM-refit)** tables summarize uncertainty by resampling families and refitting the SEM repeatedly.
  These are the preferred publication-grade uncertainty summaries in this repository.

See:
- `outputs/tables/g_mean_diff.csv`, `outputs/tables/g_variance_ratio.csv`
- `outputs/tables/g_mean_diff_family_bootstrap.csv`, `outputs/tables/g_variance_ratio_family_bootstrap.csv`

## What should I cite?

If you use the code or the results bundle, cite this repository using `CITATION.cff`. For a stable “as-shipped” bundle,
reference the publication lock zip:

- `outputs/tables/publication_results_lock.zip`

The lock bundle includes a hash manifest:

- `outputs/tables/publication_results_lock/publication_results_lock_manifest.csv`

## Where do I start if I only want to read?

1. `outputs/tables/results_snapshot.md`
2. `outputs/tables/publication_results_lock/manuscript_results_lock.md`
3. `outputs/tables/publication_results_lock.zip`
