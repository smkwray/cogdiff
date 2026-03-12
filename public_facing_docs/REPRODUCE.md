# Reproducing the pipeline (data required)

This repository is a **data-free export**. To reproduce the full pipeline you must obtain the NLSY public-use files
through official channels and place them under `data/raw/` (see `data/raw/manifest.json`).

## High-level steps

1. **Set up Python**
   - Install dependencies from `requirements.txt`.
   - Ensure `PYTHONPATH=src` for imports.

2. **Acquire data**
   - Obtain the NLSY public-use archives via official channels.
   - Place the archives under `data/raw/` according to `data/raw/manifest.json`.

3. **Run the pipeline**
   - Run the stage orchestrator to generate interim artifacts, fit SEMs, and produce tables/figures.
   - Run bootstrap inference (family bootstrap with SEM refits) for publication-grade uncertainty.

4. **Build the publication lock**
   - Rebuild `outputs/tables/publication_results_lock/` and `outputs/tables/publication_results_lock.zip`.
   - Verify integrity manifests.

## Where outputs appear

- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Publication lock bundle:
  - `outputs/tables/publication_results_lock/`
  - `outputs/tables/publication_results_lock.zip`

## If you only want to read results

You do not need data or runtime dependencies to read the bundled outputs:

- `outputs/tables/results_snapshot.md`
- `outputs/tables/publication_results_lock/manuscript_results_lock.md`
- `outputs/tables/publication_results_lock.zip`
