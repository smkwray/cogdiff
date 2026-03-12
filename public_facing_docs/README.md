# cogdiff

Reproducible pipeline for extracting latent general cognitive ability (*g*) from ASVAB and PIAT test batteries, validating it against education, income, employment, and wealth outcomes, and examining how cognitive test performance and its predictive power vary across demographic and socioeconomic subgroups. Uses three cohorts of the National Longitudinal Survey of Youth spanning birth years 1957–present.

This repository ships **code, publication-locked result tables, and figures**. It does **not** ship NLSY microdata.

## Start Here (Reading-First)

- `outputs/tables/results_snapshot.md` (short overview)
- `outputs/tables/publication_results_lock/manuscript_results_lock.md` (methods + results lock)
- `outputs/tables/publication_results_lock.zip` (complete tables/figures + manifests)

## Docs

- `METHODS.md`: estimands and quality gating (measurement invariance, partial-invariance stability check)
- `RESULTS.md`: how to interpret the tables/figures without rerunning
- `FAQ.md`: common questions and common confusions
- `GLOSSARY.md`: terminology used across tables and writeups
- `REPRODUCE.md`: setup guide (data required)
- `DATA.md`: data access and non-redistribution note

## What’s Included

- `src/`, `scripts/`, `config/`: pipeline implementation
- `outputs/tables/publication_results_lock/` and `.zip`: cite-able results artifacts
- `LICENSE` and `CITATION.cff`: licensing and citation metadata
- `data/raw/manifest.json`: describes expected raw input archives (microdata archives are not included)

## What’s Not Included

- NLSY microdata archives (anything like `data/raw/*.zip`)
- Derived microdata products (e.g., `data/processed/**`, `data/interim/**`)
- Run-specific logs

## How To Reproduce (High Level)

Reproducing the full pipeline requires:
- Obtaining the NLSY public-use files via official channels.
- Placing the raw archives under `data/raw/` (see `data/raw/manifest.json`).
- Running the pipeline scripts to regenerate tables/figures and rebuild the publication lock bundle.

See `REPRODUCE.md` and `DATA.md`.

## Citing

See `CITATION.cff`.

## License

See `LICENSE`.
