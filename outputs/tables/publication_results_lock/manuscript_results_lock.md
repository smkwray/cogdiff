# Manuscript Results Lock

Generated UTC: 2026-03-11T22:11:19Z

## Weight Policy
- `weight_pair_policy=replication_unweighted_primary_weighted_sensitivity`
- Weighted/unweighted concordance is primary-eligible only when a primary unweighted baseline exists.
- Any `weight_concordance_reason` beginning with `nonconfirmatory_` is excluded from primary robustness claims.

## Non-Primary Estimands
| cohort | estimand | analysis_tier | blocked_confirmatory | reason |
| --- | --- | --- | --- | --- |
| nlsy97 | d_g | exploratory_sensitivity | True | invariance:scalar_gate:failed_delta_cfi |
| cnlsy | d_g | exploratory_sensitivity | True | invariance:scalar_gate:failed_delta_cfi,delta_rmsea |

## Non-Primary Weight Concordance Cases
| cohort | estimand | weight_concordance_reason | weight_unweighted_status | weight_weighted_status | robust_claim_eligible |
| --- | --- | --- | --- | --- | --- |
| cnlsy | d_g | nonconfirmatory_missing_unweighted_baseline | baseline_missing | not_feasible | False |
| cnlsy | vr_g | nonconfirmatory_weighted_not_feasible:weight_quality_gate:effective_n_total_below_threshold | computed | not_feasible | True |
| nlsy79 | d_g | nonconfirmatory_weighted_not_run_placeholder | computed | not_run_placeholder | False |
| nlsy79 | vr_g | nonconfirmatory_weighted_not_run_placeholder | computed | not_run_placeholder | True |
| nlsy97 | d_g | nonconfirmatory_missing_unweighted_baseline | baseline_missing | not_run_placeholder | False |
| nlsy97 | vr_g | nonconfirmatory_weighted_not_run_placeholder | computed | not_run_placeholder | True |
