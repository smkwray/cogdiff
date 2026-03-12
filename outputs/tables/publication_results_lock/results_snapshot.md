# sexg results snapshot

- generated_utc: `2026-03-11T22:11:18Z`
- project_root: `.`

## Warnings

- g_mean_diff baseline missing cohort(s): cnlsy, nlsy97 (primary d_g excluded: cnlsy: invariance:scalar_gate:failed_delta_cfi,delta_rmsea; nlsy97: invariance:scalar_gate:failed_delta_cfi)

## Primary tables

- baseline: `outputs/tables/g_mean_diff.csv`
- baseline: `outputs/tables/g_variance_ratio.csv`
- bootstrap: `outputs/tables/g_mean_diff_family_bootstrap.csv`
- bootstrap: `outputs/tables/g_variance_ratio_family_bootstrap.csv`

## Gating exclusions

- source: `outputs/tables/confirmatory_exclusions.csv`

| cohort | blocked_confirmatory_d_g | reason_d_g |
| --- | --- | --- |
| cnlsy | True | invariance:scalar_gate:failed_delta_cfi,delta_rmsea |
| nlsy79 | False | - |
| nlsy97 | True | invariance:scalar_gate:failed_delta_cfi |

## Observed g proxy (bootstrap inference)

- mean diff table: `outputs/tables/g_proxy_mean_diff_family_bootstrap.csv`
| cohort | d_g_proxy | SE | CI |
| --- | --- | --- | --- |
| cnlsy | 0.408517 | 0.117233 | [0.0757012, 0.540054] |
| nlsy79 | 0.262916 | 0.0174321 | [0.177603, 0.247385] |
| nlsy97 | 0.0617779 | 0.0263027 | [0.0182004, 0.114477] |


- variance ratio table: `outputs/tables/g_proxy_variance_ratio_family_bootstrap.csv`
| cohort | VR_g_proxy | SE_logVR | CI |
| --- | --- | --- | --- |
| cnlsy | 1.3564 | 0.205729 | [0.89013, 1.96179] |
| nlsy79 | 1.35846 | 0.0263329 | [1.28592, 1.42646] |
| nlsy97 | 1.31722 | 0.0221717 | [1.26254, 1.37581] |

## Exploratory extensions (observed `g_proxy`)

These modules use an observed composite `g_proxy` (not latent `d_g`).

### Race × sex interaction (heterogeneity across race groups)

- summary: `outputs/tables/race_sex_interaction_summary.csv`
| cohort | heterogeneity_p | min_d_g_proxy | max_d_g_proxy |
| --- | --- | --- | --- |
| nlsy79 | 2.85411e-06 | 0.12453 | 0.373993 |
| nlsy97 | 1.80776e-06 | -0.15855 | 0.135998 |
| cnlsy | 0.882485 | 0.228696 | 0.472507 |

### Race/ethnicity-disaggregated `g_proxy` differences

- detail: `outputs/tables/race_sex_group_estimates.csv`
| cohort | race_group | n_total | mean_all | var_all | d_g_proxy | SE | CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | BLACK | 2345 | -0.56648 | 0.417455 | 0.12453 | 0.041341 | [0.0435011, 0.205558] |
| nlsy79 | HISPANIC | 1361 | -0.36998 | 0.638855 | 0.311971 | 0.0545641 | [0.205026, 0.418917] |
| nlsy79 | NON-BLACK, NON-HISPANIC | 5555 | 0.329782 | 0.484741 | 0.373993 | 0.0270685 | [0.320939, 0.427048] |
| nlsy97 | BLACK | 1784 | -0.494628 | 0.502116 | -0.15855 | 0.0474491 | [-0.251551, -0.0655501] |
| nlsy97 | HISPANIC | 1346 | -0.281635 | 0.4717 | 0.0601401 | 0.0545311 | [-0.046741, 0.167021] |
| nlsy97 | NON-BLACK, NON-HISPANIC | 3862 | 0.326644 | 0.498004 | 0.135998 | 0.0322335 | [0.0728199, 0.199175] |
| cnlsy | BLACK | 43 | -0.317677 | 0.6573 | 0.472507 | 0.310163 | [-0.135412, 1.08043] |
| cnlsy | HISPANIC | 31 | -0.112512 | 0.478544 | 0.228696 | 0.376578 | [-0.509396, 0.966789] |
| cnlsy | NON-BLACK, NON-HISPANIC | 109 | 0.271945 | 0.458515 | 0.368864 | 0.193418 | [-0.0102346, 0.747963] |

### SES moderation (parent education bins)

- detail: `outputs/tables/ses_moderation_group_estimates.csv`
| cohort | ses_bin | d_g_proxy | SE | CI |
| --- | --- | --- | --- | --- |
| nlsy79 | high | 0.474108 | 0.0510721 | [0.374007, 0.57421] |
| nlsy79 | low | 0.0882802 | 0.0507603 | [-0.0112099, 0.18777] |
| nlsy79 | mid | 0.314992 | 0.0506348 | [0.215748, 0.414236] |
| nlsy97 | high | 0.0727747 | 0.0480635 | [-0.0214298, 0.166979] |
| nlsy97 | low | 0.108261 | 0.0480094 | [0.014163, 0.20236] |
| nlsy97 | mid | 0.105957 | 0.0480234 | [0.011831, 0.200083] |
| cnlsy | high | 0.0433177 | 0.262125 | [-0.470448, 0.557083] |
| cnlsy | low | 0.538854 | 0.237529 | [0.0732971, 1.00441] |
| cnlsy | mid | 0.50082 | 0.324965 | [-0.136111, 1.13775] |

### `g_proxy` associations with income/wealth

- source: `outputs/tables/g_income_wealth_associations.csv`
| cohort | outcome | n_used | corr_all | beta_g | p_beta_g | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | earnings | 5475 | 0.406264 | 4928.45 | 1.36468e-42 | 0.246158 |
| nlsy79 | household_income | 4736 | 0.417827 | 23146.3 | 7.34639e-55 | 0.215334 |
| nlsy79 | net_worth | 4849 | 0.319254 | 107337 | 1.92105e-31 | 0.122977 |
| nlsy97 | household_income | 5168 | 0.270161 | 21125.2 | 4.1338e-45 | 0.0812512 |
| nlsy97 | net_worth | 4484 | 0.183528 | 28068.8 | 1.21646e-17 | 0.0398098 |

### `g_proxy` associations with employment status

- source: `outputs/tables/g_employment_outcomes.csv`
| cohort | outcome_col | age_col | n_used | n_employed | prevalence | odds_ratio_g | p_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | employment_2000 | age_2000 | 5781 | 4752 | 0.822003 | 1.90029 | 8.79618e-48 | 0.0415233 |
| nlsy97 | employment_2011 | age_2011 | 5889 | 4511 | 0.766004 | 2.0715 | 1.41886e-71 | 0.0536574 |

### `g_proxy` associations by sex (selected outcomes)

- source: `outputs/tables/asvab_life_outcomes_by_sex.csv`
| cohort | outcome | n_used | beta_g_male | beta_g_female | p_delta_beta |
| --- | --- | --- | --- | --- | --- |
| nlsy79 | wages | 7390 | 7241.24 | 6071.53 | 0.00183959 |
| nlsy79 | education | 9258 | 1.65678 | 1.78148 | 0.0244317 |
| nlsy97 | education | 6921 | 1.79922 | 2.30049 | 0.000264141 |
| cnlsy | education | 183 | 1.40101 | 1.86467 | 0.246088 |

### NLSY97 SAT/ACT validity (bins)

- source: `outputs/tables/g_sat_act_validity.csv`
| cohort | outcome | n_used | pearson_all | spearman_all | beta_g | p_beta_g |
| --- | --- | --- | --- | --- | --- | --- |
| nlsy97 | sat_math_2007 | 2013 | 0.530615 | 0.553066 | 0.915687 | 4.49293e-78 |
| nlsy97 | sat_verbal_2007 | 1979 | 0.536141 | 0.563898 | 0.928244 | 1.08115e-86 |
| nlsy97 | act_2007 | 1701 | 0.643034 | 0.670108 | 0.88058 | 3.07134e-120 |

### NLSY97 SAT/ACT validity by race/ethnicity

- source: `outputs/tables/g_sat_act_validity_by_race.csv`
| outcome | group | n_used | pearson | beta_g | r2 |
| --- | --- | --- | --- | --- | --- |
| act_2007 | BLACK | 385 | 0.431769 | 0.574346 | 0.186424 |
| act_2007 | HISPANIC | 154 | 0.369368 | 0.502194 | 0.136433 |
| act_2007 | NON-BLACK, NON-HISPANIC | 1162 | 0.660627 | 0.903984 | 0.436428 |
| sat_math_2007 | BLACK | 417 | 0.30809 | 0.583848 | 0.0949196 |
| sat_math_2007 | HISPANIC | 302 | 0.410257 | 0.715559 | 0.168311 |
| sat_math_2007 | NON-BLACK, NON-HISPANIC | 1294 | 0.550098 | 0.988672 | 0.302608 |
| sat_verbal_2007 | BLACK | 406 | 0.378646 | 0.636897 | 0.143372 |
| sat_verbal_2007 | HISPANIC | 294 | 0.391911 | 0.651372 | 0.153594 |
| sat_verbal_2007 | NON-BLACK, NON-HISPANIC | 1279 | 0.529031 | 0.914553 | 0.279874 |

### NLSY97 SAT/ACT validity by SES bins

- source: `outputs/tables/g_sat_act_validity_by_ses.csv`
| outcome | ses_bin | n_used | pearson | beta_g | r2 |
| --- | --- | --- | --- | --- | --- |
| act_2007 | high | 501 | 0.668799 | 0.936563 | 0.447292 |
| act_2007 | low | 558 | 0.595213 | 0.723176 | 0.354279 |
| act_2007 | mid | 503 | 0.641387 | 0.814941 | 0.411377 |
| sat_math_2007 | high | 592 | 0.578972 | 1.02572 | 0.335208 |
| sat_math_2007 | low | 651 | 0.454576 | 0.786217 | 0.206639 |
| sat_math_2007 | mid | 537 | 0.570213 | 0.980539 | 0.325143 |
| sat_verbal_2007 | high | 588 | 0.53295 | 0.883317 | 0.284036 |
| sat_verbal_2007 | low | 636 | 0.480184 | 0.794329 | 0.230577 |
| sat_verbal_2007 | mid | 529 | 0.573374 | 0.93341 | 0.328758 |

### `g_proxy` outcome validity by SES bins

- source: `outputs/tables/g_outcome_associations_by_ses_summary.csv`
| cohort | outcome | n_bins_used | heterogeneity_p | min_beta_g | max_beta_g |
| --- | --- | --- | --- | --- | --- |
| cnlsy | education | 2 | 0.787485 | 1.39596 | 1.52198 |
| nlsy79 | earnings | 3 | 8.54474e-05 | 5713.81 | 7980.34 |
| nlsy79 | education | 3 | 2.06741e-09 | 1.13855 | 1.62653 |
| nlsy79 | household_income | 3 | 2.43048e-06 | 18925.5 | 34454.9 |
| nlsy79 | net_worth | 3 | 1.54591e-08 | 65186.8 | 169243 |
| nlsy97 | education | 3 | 0.538665 | 1.74619 | 1.94046 |
| nlsy97 | household_income | 3 | 0.272807 | 14903.6 | 19352.7 |
| nlsy97 | net_worth | 3 | 0.125851 | 19596.8 | 31313.1 |

### Strongest subtest/factor predictors by outcome

- source: `outputs/tables/subtest_predictive_validity.csv`
| cohort | outcome | predictor_type | predictor | beta | r2 |
| --- | --- | --- | --- | --- | --- |
| cnlsy | education | subtest | PIAT_RR | 0.807921 | 0.132978 |
| nlsy79 | earnings | factor | technical | 6637.38 | 0.156409 |
| nlsy79 | education | subtest | MK | 1.45887 | 0.332911 |
| nlsy79 | household_income | factor | math | 23743.9 | 0.175428 |
| nlsy79 | net_worth | factor | math | 109757 | 0.105007 |
| nlsy97 | act_2007 | factor | verbal | 0.66701 | 0.395543 |
| nlsy97 | education | subtest | MK | 1.64295 | 0.117743 |
| nlsy97 | household_income | factor | math | 16106.8 | 0.0766936 |
| nlsy97 | net_worth | factor | math | 23505.7 | 0.0280241 |
| nlsy97 | sat_math_2007 | factor | math | 0.77542 | 0.30061 |
| nlsy97 | sat_verbal_2007 | subtest | WK | 0.72108 | 0.3532 |

### CNLSY nonlinear age-pattern checks

- source: `outputs/tables/cnlsy_nonlinear_age_patterns.csv`
| metric | n_bins | linear_r2 | quadratic_r2 | delta_r2 | turning_point_age |
| --- | --- | --- | --- | --- | --- |
| male_mean | 5 | 0.30945 | 0.518268 | 0.208818 | 13.2511 |
| female_mean | 5 | 0.19561 | 0.670961 | 0.475351 | 14.3458 |
| mean_diff | 5 | 0.748314 | 0.781199 | 0.032885 | 19.8468 |
| log_variance_ratio | 4 | 0.345633 | 0.939074 | 0.59344 | 14.2594 |

### Cross-cohort predictive-validity contrasts

- source: `outputs/tables/cross_cohort_predictive_validity_contrasts.csv`
| outcome | cohort_a | cohort_b | diff_b_minus_a | p_value |
| --- | --- | --- | --- | --- |
| education | nlsy79 | nlsy97 | 0.339718 | 4.07493e-06 |
| education | nlsy79 | cnlsy | -0.18506 | 0.363673 |
| education | nlsy97 | cnlsy | -0.524778 | 0.0138088 |
| household_income | nlsy79 | nlsy97 | -9112.02 | 1.13406e-12 |
| net_worth | nlsy79 | nlsy97 | -98486.5 | 3.45669e-64 |

### Cross-cohort pattern stability

- source: `outputs/tables/cross_cohort_pattern_stability.csv`
| estimand | n_cohorts | mean_estimate | sd_estimate | range_estimate | cv_estimate |
| --- | --- | --- | --- | --- | --- |
| d_g | 1 | 0.305101 | 0 | 0 | 0 |
| vr_g | 3 | 1.22555 | 0.0968122 | 0.190403 | 0.078995 |

### Sibling fixed-effects outcome associations

- source: `outputs/tables/sibling_fe_g_outcome.csv`
| cohort | outcome | n_families | n_individuals | beta_between | beta_within | p_within | r2_within |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnlsy | education | 6 | 12 | 0.354011 | 0.416426 | 0.629336 | 0.0241887 |
| nlsy79 | earnings | 18 | 38 | 8524.03 | 8733.58 | 1.3334e-06 | 0.490964 |
| nlsy79 | education | 25 | 54 | 1.63441 | 2.10229 | 3.27447e-08 | 0.462023 |
| nlsy79 | household_income | 7 | 15 | 27939.5 | 5699.48 | 0.85135 | 0.241585 |
| nlsy79 | net_worth | 12 | 24 | -13656.6 | 50447.1 | 0.61044 | 0.0504646 |
| nlsy97 | education | 1261 | 2673 | 1.82701 | 1.42073 | 6.1198e-20 | 0.0308973 |
| nlsy97 | household_income | 793 | 1665 | 16830.1 | 6426.49 | 0.00516179 | 0.0163002 |
| nlsy97 | net_worth | 825 | 1736 | 22598.9 | -8523.05 | 0.0805311 | 0.0105347 |

### Within-family cross-cohort contrasts

- source: `outputs/tables/sibling_fe_cross_cohort_contrasts.csv`
| outcome | cohort_a | cohort_b | diff_b_minus_a | p_diff |
| --- | --- | --- | --- | --- |
| education | nlsy79 | nlsy97 | -0.686501 | 0.0456257 |
| household_income | nlsy79 | nlsy97 | -20622.8 | 0.473274 |
| net_worth | nlsy79 | nlsy97 | -24464.4 | 0.789836 |

### Intergenerational mother-child `g_proxy` transmission

- source: `outputs/tables/intergenerational_g_transmission.csv`
| model | n_pairs | beta_mother_g | beta_parent_ed | p | r2 |
| --- | --- | --- | --- | --- | --- |
| bivariate | 115 | 0.447818 | - | 4.97717e-08 | 0.23218 |
| ses_controlled | 114 | 0.402433 | 0.0435521 | 3.68139e-06 | 0.230205 |

### SES attenuation of mother-child `g_proxy` transmission

- source: `outputs/tables/intergenerational_g_attenuation.csv`
| n_pairs_bivariate | n_pairs_ses_controlled | attenuation_abs | attenuation_pct | delta_r2 | beta_parent_ed |
| --- | --- | --- | --- | --- | --- |
| 115 | 114 | 0.0453857 | 10.1348 | -0.00197523 | 0.0435521 |

### Verbal-quantitative subtest profile tilt

- source: `outputs/tables/subtest_profile_tilt.csv`
| cohort | n_used_education | d_tilt | tilt_g_corr | incremental_r2 | p_tilt |
| --- | --- | --- | --- | --- | --- |
| nlsy79 | 9258 | -0.425296 | -0.00672781 | 0.00239849 | 2.71453e-08 |
| nlsy97 | 6921 | -0.120981 | -0.00545074 | 0.000287123 | 0.135355 |

### Tilt interpretation relative to `g_proxy`

- source: `outputs/tables/subtest_profile_tilt_summary.csv`
| cohort | d_g_proxy | d_tilt | tilt_to_g_ratio_abs | incremental_r2_band | interpretation |
| --- | --- | --- | --- | --- | --- |
| nlsy79 | 0.21128 | -0.425296 | 2.01295 | very_small | small_add_on |
| nlsy97 | 0.0490341 | -0.120981 | 2.46729 | negligible | small_or_null_add_on |

### Degree-threshold proxy outcomes

- source: `outputs/tables/degree_threshold_outcomes.csv`
| cohort | threshold | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | ba_or_more | 5794 | 1381 | 0.23835 | 5.32323 | 1.19625e-206 | 0.212118 |
| nlsy79 | graduate_or_more | 5794 | 488 | 0.0842251 | 4.75294 | 1.6478e-88 | 0.159578 |
| nlsy97 | ba_or_more | 5304 | 2106 | 0.397059 | 4.35688 | 1.70747e-197 | 0.175337 |
| nlsy97 | graduate_or_more | 5304 | 931 | 0.175528 | 3.32739 | 2.00269e-99 | 0.113012 |

### Explicit coded degree outcomes

- source: `outputs/tables/explicit_degree_outcomes.csv`
| cohort | threshold | degree_col | age_col | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | ba_or_more_explicit | highest_degree_ever | - | 8278 | 1989 | 0.240275 | 4.29373 | 1.26001e-226 | 0.148816 |
| nlsy79 | graduate_or_more_explicit | highest_degree_ever | - | 8278 | 628 | 0.0758637 | 3.71432 | 1.37874e-81 | 0.100998 |
| nlsy97 | ba_or_more_explicit | degree_2021 | age_2021 | 5343 | 1910 | 0.357477 | 4.829 | 9.69035e-204 | 0.190187 |
| nlsy97 | graduate_or_more_explicit | degree_2021 | age_2021 | 5343 | 706 | 0.132136 | 3.57142 | 7.4416e-88 | 0.1184 |

### Age-matched cross-cohort outcome validity

- source: `outputs/tables/age_matched_outcome_validity.csv`
| outcome | cohort | age_col | n_used | model_type | beta_g | p_value_beta_g | r2_or_pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| annual_earnings | nlsy79 | age_2000 | 3115 | ols | 7369.51 | 8.6006e-138 | 0.182965 |
| annual_earnings | nlsy79 | age_2000 | 4697 | ols | 7594.08 | 4.61031e-203 | 0.182645 |
| annual_earnings | nlsy97 | age_2019 | 3723 | ols | 22290 | 4.72244e-91 | 0.105787 |
| annual_earnings | nlsy97 | age_2021 | 4237 | ols | 26488.4 | 1.07786e-109 | 0.11038 |
| employment | nlsy79 | age_2000 | 3290 | logit | 0.662112 | 9.67969e-28 | 0.0411906 |
| employment | nlsy79 | age_2000 | 4972 | logit | 0.668671 | 1.69504e-43 | 0.0438905 |
| employment | nlsy97 | age_2019 | 4689 | logit | 0.638735 | 3.72811e-42 | 0.0417568 |
| employment | nlsy97 | age_2021 | 5324 | logit | 0.664784 | 7.79325e-52 | 0.044489 |
| household_income | nlsy79 | age_2000 | 2709 | ols | 29207.6 | 1.03392e-127 | 0.193715 |
| household_income | nlsy79 | age_2000 | 4088 | ols | 28212.1 | 1.51256e-173 | 0.177152 |
| household_income | nlsy97 | age_2019 | 3720 | ols | 38531.4 | 5.20893e-112 | 0.127245 |
| household_income | nlsy97 | age_2021 | 4844 | ols | 51350.6 | 1.40583e-159 | 0.139016 |

### Age-matched cross-cohort contrasts

- source: `outputs/tables/age_matched_cross_cohort_contrasts.csv`
| outcome | age_col_b | overlap_min | overlap_max | beta_a | beta_b | diff_b_minus_a | p_value_diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| annual_earnings | age_2019 | 36 | 40 | 7369.51 | 22290 | 14920.5 | 2.209e-41 |
| annual_earnings | age_2021 | 37 | 42 | 7594.08 | 26488.4 | 18894.3 | 1.01591e-57 |
| employment | age_2019 | 36 | 40 | 0.662112 | 0.638735 | -0.0233768 | 0.760539 |
| employment | age_2021 | 37 | 42 | 0.668671 | 0.664784 | -0.00388744 | 0.952527 |
| household_income | age_2019 | 36 | 40 | 29207.6 | 38531.4 | 9323.77 | 3.7395e-06 |
| household_income | age_2021 | 37 | 42 | 28212.1 | 51350.6 | 23138.4 | 5.49758e-29 |

### NLSY97 two-wave income and earnings trajectories

- source: `outputs/tables/nlsy97_income_earnings_trajectories.csv`
| outcome | model | n_used | mean_annualized_log_change | beta_g | p_value_beta_g | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| annual_earnings | annualized_log_change | 3642 | 0.0533692 | 0.00863831 | 0.557992 | 0.00344896 |
| annual_earnings | followup_conditional_on_baseline | 3642 | 0.0533692 | 0.237792 | 3.43707e-24 | 0.267051 |
| household_income | annualized_log_change | 3713 | 0.0432938 | 0.0273431 | 0.343861 | 0.00133726 |
| household_income | followup_conditional_on_baseline | 3713 | 0.0432938 | 0.582079 | 6.36539e-38 | 0.300649 |

### NLSY97 two-wave income and earnings volatility

- source: `outputs/tables/nlsy97_income_earnings_volatility.csv`
| outcome | model | n_used | mean_abs_annualized_log_change | instability_cutoff | prevalence | beta_g | odds_ratio_g | p_value_beta_g | r2_or_pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| annual_earnings | abs_annualized_log_change | 3642 | 0.250203 | - | - | 0.064563 | - | 1.99623e-08 | 0.341141 |
| annual_earnings | high_instability_top_quartile | 3642 | 0.250203 | 0.228622 | 0.250137 | 0.0356152 | 1.03626 | 0.53418 | 0.103145 |
| household_income | abs_annualized_log_change | 3713 | 0.480726 | - | - | -0.06128 | - | 0.0167868 | 0.216184 |
| household_income | high_instability_top_quartile | 3713 | 0.480726 | 0.321744 | 0.250202 | -0.314235 | 0.730348 | 2.39053e-09 | 0.0537632 |

### NLSY97 labor-force persistence and employment transitions

- source: `outputs/tables/nlsy97_employment_persistence.csv`
| model | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| persistent_employment_2019_2021 | 4757 | 3504 | 0.736599 | 1.62162 | 6.82155e-27 | 0.100475 |
| reentry_2021_given_not_employed_2019 | 910 | 298 | 0.327473 | 1.2548 | 0.0113244 | 0.0225403 |
| retention_2021_given_employed_2019 | 3847 | 3504 | 0.91084 | 1.56453 | 1.25287e-09 | 0.0396367 |

### NLSY97 multi-wave employment instability

- source: `outputs/tables/nlsy97_employment_instability.csv`
| model | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| any_transition_2011_2021 | 4757 | 1434 | 0.30145 | 0.685311 | 2.58535e-21 | 0.0158758 |
| double_transition_2011_2021 | 4757 | 288 | 0.0605424 | 0.756605 | 0.000195311 | 0.00637218 |
| mixed_attachment_2011_2021 | 4757 | 1434 | 0.30145 | 0.685311 | 2.58535e-21 | 0.0158758 |

### NLSY97 unemployment-insurance receipt and intensity

- source: `outputs/tables/nlsy97_unemployment_insurance.csv`
| year | model | n_used | n_positive | prevalence | beta_g | odds_ratio_g | p_value_beta_g | r2_or_pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | any_ui_receipt | 134 | 99 | 0.738806 | 0.414359 | 1.5134 | 0.165511 | 0.013251 |
| 2019 | log1p_ui_amount | 112 | 112 | 1 | 0.181753 | - | 0.272585 | 0.0134719 |
| 2019 | log1p_ui_spells | 134 | 99 | 0.738806 | 0.0599169 | - | 0.139964 | 0.0187556 |
| 2021 | any_ui_receipt | 378 | 150 | 0.396825 | 0.151843 | 1.16398 | 0.287835 | 0.00342013 |
| 2021 | log1p_ui_amount | 343 | 343 | 1 | -0.246646 | - | 0.0106477 | 0.0224717 |
| 2021 | log1p_ui_spells | 378 | 150 | 0.396825 | 0.0235485 | - | 0.346743 | 0.00255398 |

### CNLSY 2014 late-adolescent/adult outcome associations

- source: `outputs/tables/cnlsy_adult_outcome_associations.csv`
| outcome | model_type | n_used | mean_outcome | beta_g | p_value_beta_g | r2_or_pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| education_years_2014 | ols | 171 | 2.07018 | 0.0673191 | 0.0919342 | 0.0877532 |
| family_income_2014 | ols | 69 | 89165.4 | 45420.7 | 0.000119972 | 0.210309 |
| num_current_jobs_2014 | ols | 171 | 0.280702 | -0.00743966 | 0.877537 | 0.000758227 |
| wage_income_2014 | ols | 154 | 1015.01 | 107.056 | 0.583483 | 0.00224283 |

### CNLSY child `g_proxy` carryover net mother SES

- source: `outputs/tables/cnlsy_carryover_net_mother_ses_summary.csv`
| outcome | model_type | n_baseline | n_mother_ses | beta_g_baseline | beta_g_mother_ses | attenuation_pct | delta_r2_or_pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log_wage_income_2014 | ols_log_income | 154 | 153 | 0.581699 | 0.675557 | -16.1353 | 0.00525117 |
| log_family_income_2014 | ols_log_income | 69 | 69 | 0.831248 | 0.815035 | 1.9504 | 0.0177863 |
| num_current_jobs_2014 | ols_count_like | 171 | 170 | -0.00743966 | -0.0101786 | -36.8147 | 0.000162556 |

### Nonlinear and threshold outcome models

- source: `outputs/tables/nonlinear_threshold_outcome_summary.csv`
| cohort | outcome | outcome_type | n_linear | beta_g_sq | p_value_g_sq | delta_fit_linear_to_quadratic | threshold_odds_ratio | threshold_beta | p_value_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | log_annual_earnings | continuous | 5475 | -0.201995 | 1.84119e-06 | 0.00383905 | - | 1.51254 | 1.28028e-32 |
| nlsy79 | log_household_income | continuous | 4736 | -0.127797 | 3.79458e-08 | 0.00553898 | - | 1.05407 | 2.40111e-48 |
| nlsy79 | employment_2000 | binary | 5781 | -0.0704355 | 0.0187797 | 0.00100328 | 1.98978 | 0.688022 | 1.17768e-11 |
| nlsy79 | ba_or_more_explicit | binary | 5794 | 0.0500235 | 0.231355 | 0.000217905 | 7.78391 | 2.05206 | 8.00278e-178 |
| nlsy97 | log_annual_earnings_2021 | continuous | 4237 | -0.0147483 | 0.318985 | 0.000217407 | - | 0.616884 | 2.92101e-36 |
| nlsy97 | log_household_income_2021 | continuous | 4844 | -0.27971 | 2.65561e-20 | 0.0152919 | - | 1.22593 | 6.67798e-35 |
| nlsy97 | employment_2021 | binary | 5324 | -0.0685253 | 0.0177651 | 0.00100631 | 2.09646 | 0.740252 | 1.27525e-13 |
| nlsy97 | ba_or_more_explicit | binary | 5359 | -0.0799295 | 0.0397823 | 0.0006136 | 5.96051 | 1.78516 | 6.66344e-127 |

### Sibling discordance beyond education

- source: `outputs/tables/sibling_discordance.csv`
| cohort | outcome | n_pairs | n_families | mean_abs_g_diff | mean_abs_outcome_diff | corr_abs_diff | beta_abs_g_diff | p_value_abs_g_diff | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nlsy79 | education | 33 | 25 | 1.05872 | 3.12121 | 0.272875 | 0.948558 | 0.124433 | 0.0744607 |
| nlsy97 | earnings | 636 | 551 | 0.55589 | 44048.5 | 0.0616417 | 7961.79 | 0.120432 | 0.0037997 |
| nlsy97 | education | 1559 | 1261 | 0.561606 | 2.78576 | 0.0254746 | 0.315673 | 0.314801 | 0.000648955 |
| nlsy97 | employment | 985 | 814 | 0.5646 | 0.306599 | 0.0422544 | 0.045691 | 0.185154 | 0.00178544 |
| nlsy97 | household_income | 813 | 689 | 0.576229 | 74401.1 | 0.00135643 | 294.874 | 0.969196 | 1.8399e-06 |
| nlsy97 | net_worth | 995 | 825 | 0.566813 | 83688.1 | 0.0394565 | 10628.3 | 0.213676 | 0.00155681 |

### NLSY79 mediation of earnings and income associations

- source: `outputs/tables/nlsy79_mediation_summary.csv`
| outcome | model | n_used | beta_g_baseline | beta_g_model | pct_attenuation_g | delta_r2 | mediators_in_model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log_annual_earnings | baseline | 5475 | 1.25594 | 1.25594 | 0 | 0 | baseline |
| log_annual_earnings | plus_education_years | 5475 | 1.25594 | 1.17949 | 6.08732 | 0.000626698 | education_years |
| log_annual_earnings | plus_employment_2000 | 5464 | 1.25401 | 1.04531 | 16.6428 | 0.0554695 | employment_2000 |
| log_annual_earnings | plus_job_zone | 2940 | 0.94056 | 0.965926 | -2.69689 | 0.000468533 | __job_zone |
| log_annual_earnings | plus_all_mediators | 2936 | 0.941598 | 0.875807 | 6.98714 | 0.0148256 | education_years;employment_2000;__job_zone |
| log_household_income | baseline | 4736 | 0.88625 | 0.88625 | 0 | 0 | baseline |
| log_household_income | plus_education_years | 4736 | 0.88625 | 0.717263 | 19.0676 | 0.0107624 | education_years |
| log_household_income | plus_employment_2000 | 4725 | 0.887017 | 0.753324 | 15.0721 | 0.0730921 | employment_2000 |
| log_household_income | plus_job_zone | 2540 | 0.580007 | 0.53635 | 7.52702 | 0.00780083 | __job_zone |
| log_household_income | plus_all_mediators | 2534 | 0.580811 | 0.411751 | 29.1076 | 0.0402617 | education_years;employment_2000;__job_zone |

### NLSY79 occupation major-group summary

- source: `outputs/tables/nlsy79_occupation_major_group_summary.csv`
| occupation_group | n_used | share_used | mean_g_proxy | mean_education_years | mean_household_income |
| --- | --- | --- | --- | --- | --- |
| Management/professional/related | 2445 | 0.474389 | 0.315809 | 14.7108 | 74749.5 |
| Production/transport/material moving | 848 | 0.164532 | -0.453413 | 12.4505 | 35918.3 |
| Construction/extraction/maintenance | 826 | 0.160264 | -0.443619 | 12.0424 | 36563.5 |
| Service | 743 | 0.14416 | -0.0990746 | 12.9933 | 48326 |
| Sales/office | 168 | 0.032596 | -0.0811501 | 11.9345 | 49087 |
| Farming/fishing/forestry | 124 | 0.024059 | -0.0723746 | 12.3871 | 44774.4 |

### NLSY79 management/professional occupation proxy

- source: `outputs/tables/nlsy79_high_skill_occupation_outcome.csv`
| outcome | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| management_professional_related | 5154 | 2445 | 0.474389 | 2.98423 | 8.79188e-151 | 0.115771 |

### NLSY97 latest-adult occupation major-group summary

- source: `outputs/tables/nlsy97_adult_occupation_major_group_summary.csv`
- coverage: `779` with any adult occupation code out of `6992` total respondents
| occupation_group | n_used | share_used | mean_g_proxy | mean_education_years | top_source_wave |
| --- | --- | --- | --- | --- | --- |
| Management/professional/related | 197 | 0.261968 | 0.362466 | 15.7665 | 2013 |
| Service | 148 | 0.196809 | 0.0702309 | 13.8041 | 2013 |
| Farming/fishing/forestry | 134 | 0.178191 | -0.00689016 | 14.5455 | 2013 |
| Production/transport/material moving | 117 | 0.155585 | -0.0483475 | 13.7913 | 2013 |
| Sales/office | 78 | 0.103723 | -0.07329 | 13.013 | 2015 |
| Construction/extraction/maintenance | 78 | 0.103723 | -0.0661407 | 12.9615 | 2019 |

### NLSY97 bounded occupational mobility

- source: `outputs/tables/nlsy97_occupational_mobility_summary.csv`
| n_with_2plus_occupation_waves | n_with_major_group_start_end | n_changed_major_group | pct_changed_major_group | n_upward_to_management_professional | n_downward_from_management_professional | mean_year_gap | top_wave_pair |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | 72 | 39 | 0.541667 | 4 | 10 | 4.80556 | 2015->2021 |

### NLSY97 occupational mobility associations

- source: `outputs/tables/nlsy97_occupational_mobility_models.csv`
| model | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| any_major_group_change | 72 | 39 | 0.541667 | 1.21963 | 0.566652 | 0.0350653 |
| downward_from_management_professional | 15 | 10 | 0.666667 | 5.10697 | 0.131954 | 0.186476 |

### NLSY97 latest-adult management/professional occupation proxy

- source: `outputs/tables/nlsy97_high_skill_occupation_outcome.csv`
| outcome | n_used | n_positive | prevalence | odds_ratio_g | p_value_beta_g | pseudo_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| management_professional_related | 752 | 197 | 0.261968 | 2.08921 | 2.45427e-09 | 0.0450406 |

### NLSY79 Job Zone mapping quality

- source: `outputs/tables/nlsy79_job_zone_mapping_quality.csv`
| n_occ_nonmissing | n_matched_exact | n_matched_prefix_only | n_matched_any | pct_matched_any | mean_job_zone |
| --- | --- | --- | --- | --- | --- |
| 5233 | 2213 | 900 | 3113 | 0.594879 | 3.13792 |

### NLSY79 Job Zone complexity association

- source: `outputs/tables/nlsy79_job_zone_complexity_outcome.csv`
| outcome | n_used | beta_g | p_value_beta_g | beta_age | p_value_beta_age | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| job_zone | 3113 | 0.33782 | 1.50613e-51 | 0.0103968 | 0.308415 | 0.0683475 |

### NLSY79 job-complexity vs pay mismatch summary

- source: `outputs/tables/nlsy79_job_pay_mismatch_summary.csv`
| group | n_used | share_used | mean_job_zone | mean_annual_earnings | mean_pay_residual_z | mean_g_proxy | mean_education_years |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 2516 | 1 | 3.15753 | 20025.3 | 1.46853e-15 | 0.047024 | 13.595 |
| underpaid_for_complexity | 300 | 0.119237 | 3.15358 | 2871.62 | -2.08681 | -0.275898 | 13.1233 |
| aligned_band | 2029 | 0.806439 | 3.17153 | 19611.1 | 0.189333 | 0.035655 | 13.5638 |
| overpaid_for_complexity | 187 | 0.0743243 | 3.01193 | 52038.9 | 1.29351 | 0.688437 | 14.6898 |

### NLSY79 job-complexity vs pay mismatch associations

- source: `outputs/tables/nlsy79_job_pay_mismatch_models.csv`
| outcome | model_family | n_used | beta_g | p_value_beta_g | odds_ratio_g | r2_or_pseudo_r2 | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pay_residual_z | ols_residual | 2516 | 0.380882 | 6.42548e-56 | - | 0.0898543 | - |
| overpaid_for_complexity | logit_residual_tail | 2516 | 1.33674 | 8.81892e-28 | 3.8066 | 0.110908 | 0.0743243 |
| underpaid_for_complexity | logit_residual_tail | 2516 | -0.595979 | 9.44544e-14 | 0.551023 | 0.031161 | 0.119237 |

### NLSY79 occupation education-requirement mapping quality

- source: `outputs/tables/nlsy79_occupation_education_mapping_quality.csv`
| n_occ_nonmissing | n_matched_exact | n_matched_prefix_only | n_matched_any | pct_matched_any | mean_required_education_years | mean_bachelor_plus_share | modal_required_education_label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5233 | 2172 | 899 | 3071 | 0.586853 | 14.0089 | 0.378083 | High School Diploma - or the equivalent (for example, GED) |

### NLSY79 occupation education-requirement associations

- source: `outputs/tables/nlsy79_occupation_education_requirement_outcome.csv`
| outcome | n_used | beta_g | p_value_beta_g | beta_age | p_value_beta_age | r2 | mean_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| required_education_years | 3071 | 0.743767 | 1.18626e-54 | 0.0112089 | 0.605548 | 0.0733015 | 14.0089 |
| bachelor_plus_share | 3071 | 0.139662 | 2.4778e-64 | 0.0036737 | 0.326836 | 0.0854967 | 0.378083 |

### NLSY79 education-job mismatch summary

- source: `outputs/tables/nlsy79_education_job_mismatch_summary.csv`
| group | n_used | share_used | mean_mismatch_years | mean_g_proxy | mean_education_years | mean_required_education_years | mean_annual_earnings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 3071 | 1 | -0.51497 | -0.018882 | 13.494 | 14.0089 | 17027.3 |
| undereducated | 779 | 0.253663 | -3.89174 | -0.215267 | 11.8755 | 15.7672 | 14554.4 |
| matched_band | 1860 | 0.605666 | -0.0578189 | -0.0510603 | 13.3414 | 13.3992 | 16954.1 |
| overeducated | 432 | 0.140671 | 3.60587 | 0.473792 | 17.0694 | 13.4636 | 21863 |

### NLSY79 education-job mismatch associations

- source: `outputs/tables/nlsy79_education_job_mismatch_models.csv`
| outcome | model_family | n_used | beta_g | p_value_beta_g | odds_ratio_g | beta_age | p_value_beta_age | r2_or_pseudo_r2 | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mismatch_years | ols | 3071 | 0.919192 | 1.70723e-57 | - | 0.0398777 | 0.126978 | 0.0769103 | - |
| abs_mismatch_years | ols | 3071 | 0.0833835 | 0.0425533 | - | -0.0107665 | 0.564353 | 0.00150516 | - |
| overeducated | logit | 3071 | 1.01053 | 1.96155e-40 | 2.74705 | 0.0330423 | 0.283141 | 0.0819671 | 0.140671 |
| undereducated | logit | 3071 | -0.42399 | 2.11556e-15 | 0.65443 | -0.0246843 | 0.303182 | 0.0186308 | 0.253663 |

### Race/ethnicity measurement invariance

- source: `outputs/tables/race_invariance_eligibility.csv`
| cohort | status | n_groups | smallest_group_n | metric_pass | scalar_pass | reason_d_g |
| --- | --- | --- | --- | --- | --- | --- |
| cnlsy | computed | 3 | 118 | True | False | scalar_gate:failed_delta_rmsea |
| nlsy79 | computed | 3 | 1361 | False | True | metric_gate:failed_delta_cfi |
| nlsy97 | computed | 3 | 1346 | True | True | - |

## Bootstrap inference coverage

| cohort | rep_dirs | max_rep_index | has_full_sample | in_bootstrap_tables | manifest_status | manifest_attempted |
| --- | --- | --- | --- | --- | --- | --- |
| cnlsy | 499 | 498 | yes | yes | computed | 499 |
| nlsy79 | 499 | 498 | yes | yes | computed | 499 |
| nlsy97 | 499 | 498 | yes | yes | computed | 499 |

## Latest pipeline run summary

- path: `outputs/logs/pipeline/20260301_071215_99710_pipeline_run_summary.csv`
- mtime_utc: `2026-03-01T12:26:31Z`

## Bootstrap vs baseline deltas

### g_mean_diff

- baseline: `outputs/tables/g_mean_diff.csv`
- bootstrap: `outputs/tables/g_mean_diff_family_bootstrap.csv`

| cohort | baseline | bootstrap | delta | baseline_se | bootstrap_se | baseline_ci | bootstrap_ci | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnlsy | - | 0.122298 | - | - | 0.148897 | - | [-0.254015, 0.276647] | computed |
| nlsy79 | 0.305101 | 0.305104 | 2.81604e-06 | 0.0324285 | 0.189638 | [0.241541, 0.368661] | [-0.106444, 0.363077] | computed |
| nlsy97 | - | 0.0166794 | - | - | 0.0234932 | - | [-0.0478774, 0.0442298] | computed |
### g_variance_ratio

- baseline: `outputs/tables/g_variance_ratio.csv`
- bootstrap: `outputs/tables/g_variance_ratio_family_bootstrap.csv`

| cohort | baseline | bootstrap | delta | baseline_se | bootstrap_se | baseline_ci | bootstrap_ci | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnlsy | 1.12019 | 1.12019 | -3.9968e-14 | - | 0.216994 | - | [0.687495, 1.49767] | computed |
| nlsy79 | 1.3106 | 1.3106 | 8.80185e-13 | - | 0.271037 | - | [0.729762, 1.37046] | computed |
| nlsy97 | 1.24585 | 1.24585 | -2.10222e-07 | 0.0338322 | 0.18081 | [1.16592, 1.33127] | [0.787964, 1.30372] | computed |

## Publication lock artifacts

- `outputs/tables/publication_results_lock`: present (bytes=3072, mtime_utc=2026-03-11T20:36:10Z)
- `outputs/tables/publication_results_lock.zip`: present (bytes=1065721, mtime_utc=2026-03-11T20:34:07Z)
- `outputs/tables/publication_results_lock/manuscript_results_lock.md`: present (bytes=1573, mtime_utc=2026-03-11T20:34:07Z)
- `outputs/tables/publication_results_lock/publication_results_lock_manifest.csv`: present (bytes=21620, mtime_utc=2026-03-11T20:34:07Z)
- `outputs/tables/publication_snapshot_manifest.csv`: present (bytes=512, mtime_utc=2026-03-11T20:34:08Z)
