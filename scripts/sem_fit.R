#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for sem_fit.R. Install it in R with: install.packages('jsonlite').")
  }
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    stop("lavaan package is required for sem_fit.R. Install it in R with: install.packages('lavaan').")
  }
}))

library(jsonlite)
library(lavaan)
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag) {
  idx <- which(args == flag)
  if (length(idx) == 0L || idx[1] == length(args)) {
    return(NULL)
  }
  args[[idx[1] + 1L]]
}

trim_ws <- function(value) {
  gsub("^\\s+|\\s+$", "", as.character(value), perl = TRUE)
}

default_if_missing <- function(value, default_value) {
  if (is.null(value) || length(value) == 0L) {
    return(default_value)
  }
  value <- trim_ws(value)
  if (length(value) == 0L || all(is.na(value)) || all(value == "")) {
    default_value
  } else {
    value
  }
}

to_char_vec <- function(value) {
  if (is.null(value)) {
    return(character(0))
  }
  if (is.list(value)) {
    value <- unlist(value, use.names = FALSE)
  }
  normalized <- trim_ws(as.character(value))
  normalized <- normalized[!is.na(normalized) & normalized != ""]
  unique(normalized)
}

is_absolute_path <- function(path_value) {
  if (is.null(path_value) || length(path_value) == 0L) {
    return(FALSE)
  }
  token <- trim_ws(as.character(path_value[[1L]]))
  if (token == "") {
    return(FALSE)
  }
  grepl("^(/|[A-Za-z]:[/\\\\])", token)
}

resolve_request_path <- function(path_value, request_path) {
  token <- trim_ws(as.character(path_value))
  if (token == "") {
    return(token)
  }
  if (startsWith(token, "~")) {
    return(path.expand(token))
  }
  if (is_absolute_path(token)) {
    return(token)
  }
  request_dir <- dirname(normalizePath(request_path, winslash = "/", mustWork = FALSE))
  file.path(request_dir, token)
}

extract_measure <- function(measures, name) {
  if (name %in% names(measures)) {
    as.numeric(measures[[name]])
  } else {
    NA_real_
  }
}

resolve_group_label <- function(group_numeric, group_levels) {
  idx <- suppressWarnings(as.integer(group_numeric))
  if (!is.na(idx) && idx >= 1L && idx <= length(group_levels)) {
    group_levels[[idx]]
  } else {
    as.character(group_numeric)
  }
}

coerce_numeric <- function(vec, var_name) {
  converted <- suppressWarnings(as.numeric(vec))
  if (all(is.na(converted))) {
    stop(sprintf("Observed column '%s' cannot be coerced to numeric.", var_name))
  }
  converted
}

normalize_se_mode <- function(se_mode) {
  if (is.null(se_mode) || length(se_mode) == 0L) {
    return("standard")
  }
  normalized <- trim_ws(as.character(se_mode))
  if (length(normalized) == 0L || all(is.na(normalized)) || all(trim_ws(normalized) == "")) {
    return("standard")
  }
  normalized <- tolower(normalized)
  normalized <- sub("^\\.+|\\.+$", "", gsub("-", ".", normalized, fixed = FALSE))
  normalized <- sub("_", ".", normalized, fixed = TRUE)
  if (normalized == "robustcluster") {
    normalized <- "robust.cluster"
  }
  normalized
}

as_bool <- function(value, default_value = TRUE) {
  if (is.null(value) || length(value) == 0L) {
    return(isTRUE(default_value))
  }
  if (is.logical(value)) {
    return(isTRUE(value[[1L]]))
  }
  normalized <- tolower(trim_ws(as.character(value[[1L]])))
  if (normalized %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }
  if (normalized %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }
  isTRUE(default_value)
}

normalize_group_token <- function(value) {
  token <- tolower(trim_ws(as.character(value)))
  if (token %in% c("f", "female", "2", "w", "woman", "girl")) {
    return("female")
  }
  if (token %in% c("m", "male", "1", "boy", "man")) {
    return("male")
  }
  token
}

infer_sex_groups <- function(group_levels, reference_group = "female") {
  if (length(group_levels) == 0L) {
    return(list(female = NA_character_, male = NA_character_))
  }
  norm_levels <- vapply(group_levels, normalize_group_token, character(1))
  norm_reference <- normalize_group_token(reference_group)

  female <- NA_character_
  female_hits <- which(norm_levels == norm_reference)
  if (length(female_hits) >= 1L) {
    female <- group_levels[[female_hits[[1L]]]]
  } else {
    token_hits <- which(norm_levels %in% c("female"))
    if (length(token_hits) == 1L) {
      female <- group_levels[[token_hits[[1L]]]]
    } else if (length(group_levels) == 2L) {
      female <- group_levels[[1L]]
    }
  }

  male <- NA_character_
  if (!is.na(female)) {
    male_hits <- which(norm_levels %in% c("male") & group_levels != female)
    if (length(male_hits) >= 1L) {
      male <- group_levels[[male_hits[[1L]]]]
    } else {
      other <- group_levels[group_levels != female]
      if (length(other) >= 1L) {
        male <- other[[1L]]
      }
    }
  } else if (length(group_levels) == 2L) {
    female <- group_levels[[1L]]
    male <- group_levels[[2L]]
  }

  list(female = female, male = male)
}

reorder_group_levels <- function(group_levels, reference_group = "female") {
  if (length(group_levels) <= 1L) {
    return(group_levels)
  }
  groups <- infer_sex_groups(group_levels, reference_group = reference_group)
  female <- groups$female
  male <- groups$male
  if (!is.na(female) && !is.na(male) && female != male) {
    leading <- c(female, male)
    trailing <- group_levels[!group_levels %in% leading]
    return(c(leading, trailing))
  }
  group_levels
}

first_num_or_na <- function(vec) {
  if (length(vec) == 0L) {
    return(NA_real_)
  }
  as.numeric(vec[[1L]])
}

fit_step_with_retries <- function(step, fit_args) {
  attempt_specs <- list(
    list(name = "default", args = list()),
    list(name = "no_check_gradient", args = list(check.gradient = FALSE)),
    list(
      name = "bounded_standard",
      args = list(
        check.gradient = FALSE,
        bounds = "standard"
      )
    ),
    list(
      name = "nlminb_extended",
      args = list(control = list(iter.max = 20000, eval.max = 40000, rel.tol = 1e-08))
    ),
    list(
      name = "bfgs_extended",
      args = list(
        optim.method = "bfgs",
        check.gradient = FALSE,
        control = list(maxit = 20000, reltol = 1e-08)
      )
    )
  )

  fit_is_admissible <- function(fit_obj) {
    reasons <- character(0)
    ptable <- tryCatch(
      parameterEstimates(fit_obj, standardized = FALSE),
      error = function(err) NULL
    )
    if (is.null(ptable)) {
      reasons <- c(reasons, "parameterEstimates unavailable")
      return(list(ok = FALSE, reasons = reasons))
    }

    se_vals <- suppressWarnings(as.numeric(ptable$se))
    if (length(se_vals) == 0L || all(is.na(se_vals))) {
      reasons <- c(reasons, "standard errors unavailable")
    }

    latent_names <- tryCatch(
      to_char_vec(lavNames(fit_obj, "lv")),
      error = function(err) character(0)
    )
    if (length(latent_names) > 0L) {
      lv_var_mask <- ptable$op == "~~" &
        ptable$lhs == ptable$rhs &
        ptable$lhs %in% latent_names
      lv_vars <- suppressWarnings(as.numeric(ptable$est[lv_var_mask]))
      if (length(lv_vars) > 0L && any(is.finite(lv_vars) & lv_vars < 0, na.rm = TRUE)) {
        reasons <- c(reasons, "negative latent variance")
      }
    }

    obs_var_mask <- ptable$op == "~~" &
      ptable$lhs == ptable$rhs &
      !(ptable$lhs %in% latent_names)
    obs_vars <- suppressWarnings(as.numeric(ptable$est[obs_var_mask]))
    if (length(obs_vars) > 0L && any(is.finite(obs_vars) & obs_vars < 0, na.rm = TRUE)) {
      reasons <- c(reasons, "negative residual variance")
    }

    std_table <- tryCatch(
      standardizedSolution(fit_obj),
      error = function(err) NULL
    )
    if (!is.null(std_table)) {
      cor_rows <- std_table[std_table$op == "~~" & std_table$lhs != std_table$rhs, , drop = FALSE]
      cor_vals <- suppressWarnings(as.numeric(cor_rows$est.std))
      if (length(cor_vals) > 0L && any(is.finite(cor_vals) & abs(cor_vals) > 1.0, na.rm = TRUE)) {
        reasons <- c(reasons, "correlation exceeds 1.0 in absolute value")
      }
    }

    list(ok = length(reasons) == 0L, reasons = reasons)
  }

  failures <- character(0)
  fallback_fit <- NULL
  fallback_attempt <- NULL
  fallback_reasons <- character(0)
  fallback_warnings <- character(0)
  for (attempt in attempt_specs) {
    attempt_args <- fit_args
    for (arg_name in names(attempt$args)) {
      attempt_args[[arg_name]] <- attempt$args[[arg_name]]
    }
    attempt_warnings <- character(0)
    fit <- withCallingHandlers(
      tryCatch(
        do.call(cfa, attempt_args),
        error = function(err) err
      ),
      warning = function(w) {
        attempt_warnings <<- c(attempt_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    if (inherits(fit, "error")) {
      failures <- c(
        failures,
        sprintf("%s error: %s", attempt$name, conditionMessage(fit))
      )
      next
    }

    converged <- tryCatch(
      isTRUE(lavInspect(fit, "converged")),
      error = function(err) FALSE
    )
    if (converged) {
      admissibility <- fit_is_admissible(fit)
      if (isTRUE(admissibility$ok)) {
        if (length(attempt_warnings) > 0L) {
          for (msg in unique(attempt_warnings)) {
            warning(msg)
          }
        }
        if (attempt$name != "default") {
          warning(sprintf("Step '%s' converged using retry '%s'.", step, attempt$name))
        }
        return(fit)
      }
      if (is.null(fallback_fit)) {
        fallback_fit <- fit
        fallback_attempt <- attempt$name
        fallback_reasons <- admissibility$reasons
        fallback_warnings <- attempt_warnings
      }
      failures <- c(
        failures,
        sprintf(
          "%s converged but inadmissible: %s",
          attempt$name,
          paste(admissibility$reasons, collapse = ", ")
        )
      )
      next
    }

    failures <- c(failures, sprintf("%s non-converged", attempt$name))
  }

  if (!is.null(fallback_fit)) {
    if (length(fallback_warnings) > 0L) {
      for (msg in unique(fallback_warnings)) {
        warning(msg)
      }
    }
    warning(
      sprintf(
        "Step '%s' has no admissible converged fit; using '%s' with issues: %s",
        step,
        fallback_attempt,
        paste(fallback_reasons, collapse = ", ")
      )
    )
    return(fallback_fit)
  }

  stop(sprintf("Failed to fit step '%s': %s", step, paste(failures, collapse = " | ")))
}

request_path <- arg_value("--request")
outdir <- arg_value("--outdir")
if (is.null(request_path) || is.null(outdir)) {
  stop("Usage: sem_fit.R --request <request.json> --outdir <outdir>")
}

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
request <- fromJSON(request_path)

cohort <- trim_ws(default_if_missing(request$cohort, "unknown"))
data_csv <- trim_ws(default_if_missing(request$data_csv, ""))
if (data_csv == "") {
  stop("Request is missing required field: data_csv")
}
data_csv <- resolve_request_path(data_csv, request_path)

group_col <- trim_ws(default_if_missing(request$group_col, "sex"))
reference_group <- trim_ws(default_if_missing(request$reference_group, "female"))
observed_tests <- to_char_vec(default_if_missing(request$observed_tests, character(0)))
if (length(observed_tests) == 0L) {
  stop("Request is missing required field: observed_tests")
}

estimator <- trim_ws(default_if_missing(request$estimator, "MLR"))
missing_arg <- trim_ws(default_if_missing(request$missing, "fiml"))
std_lv <- as_bool(request$std_lv, TRUE)
se_mode <- normalize_se_mode(request$se_mode)
force_standard_se <- as_bool(request$force_standard_se, FALSE)
cluster_col <- trim_ws(default_if_missing(request$cluster_col, ""))
weight_col <- trim_ws(default_if_missing(request$weight_col, ""))
if (!se_mode %in% c("standard", "robust", "robust.cluster", "weighted")) {
  stop(sprintf("Unsupported se_mode '%s'. Supported values: standard, robust, robust.cluster, weighted.", se_mode))
}

if (se_mode == "weighted" && weight_col == "") {
  stop("se_mode 'weighted' requires weight_col.")
}
if (se_mode == "robust.cluster" && cluster_col == "") {
  stop("robust.cluster se_mode requires cluster_col.")
}

requested_steps <- tolower(to_char_vec(default_if_missing(request$invariance_steps, character(0))))
if (length(requested_steps) == 0L) {
  requested_steps <- c("configural", "metric", "scalar", "strict")
}
supported_steps <- c("configural", "metric", "scalar", "strict")
unsupported <- setdiff(requested_steps, supported_steps)
if (length(unsupported) > 0L) {
  stop(sprintf("Unsupported invariance_steps: %s", paste(unsupported, collapse = ", ")))
}

partial_intercepts <- gsub("\\s+", "", to_char_vec(default_if_missing(request$partial_intercepts, character(0))))
if (length(partial_intercepts) > 0L) {
  partial_intercepts <- unique(partial_intercepts)
}

model_path <- file.path(outdir, "model.lavaan")
if (!file.exists(model_path)) {
  stop(sprintf("Missing model file: %s", model_path))
}
model_syntax <- paste(readLines(model_path, warn = FALSE), collapse = "\n")
if (trim_ws(model_syntax) == "") {
  stop(sprintf("Model file is empty: %s", model_path))
}

step_eq <- list(
  configural = character(0),
  metric = c("loadings"),
  scalar = c("loadings", "intercepts"),
  strict = c("loadings", "intercepts", "residuals")
)

data <- tryCatch(
  {
    read.csv(data_csv, stringsAsFactors = FALSE)
  },
  error = function(err) {
    stop(sprintf("Failed reading data file '%s': %s", data_csv, conditionMessage(err)))
  }
)

required_cols <- c(group_col, observed_tests)
if (cluster_col != "") {
  required_cols <- c(required_cols, cluster_col)
}
if (weight_col != "") {
  required_cols <- c(required_cols, weight_col)
}
required_cols <- unique(required_cols)
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0L) {
  stop(sprintf("Data file is missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

data <- data[, required_cols, drop = FALSE]
group_values <- trim_ws(data[[group_col]])
valid_group <- !is.na(group_values) & group_values != ""
data <- data[valid_group, , drop = FALSE]
group_values <- trim_ws(data[[group_col]])
group_levels <- unique(as.character(group_values))
group_levels <- reorder_group_levels(group_levels, reference_group = reference_group)
if (length(group_levels) < 2L) {
  stop(sprintf("Group column '%s' must contain at least two non-missing groups.", group_col))
}
data[[group_col]] <- factor(group_values, levels = group_levels)

for (name in observed_tests) {
  data[[name]] <- coerce_numeric(data[[name]], name)
}
if (weight_col != "") {
  data[[weight_col]] <- coerce_numeric(data[[weight_col]], weight_col)
}

model_lines <- trim_ws(unlist(strsplit(model_syntax, "\n", fixed = TRUE)))
model_lines <- trim_ws(sub("#.*$", "", model_lines))
model_lines <- model_lines[nzchar(model_lines)]
latent_lines <- model_lines[grepl("=~", model_lines, fixed = TRUE)]
latent_factors <- to_char_vec(sub("\\s*=~.*$", "", latent_lines, perl = TRUE))
if (length(latent_factors) == 0L) {
  stop("Unable to detect latent factors from model syntax.")
}

model_fit_rows <- list()
param_rows <- list()
latent_rows <- list()
mod_rows <- list()
score_rows <- list()
group_audit_rows <- list()
sex_estimand_rows <- list()

for (step in requested_steps) {
  fit_args <- list(
    model = model_syntax,
    data = data,
    group = group_col,
    estimator = estimator,
    missing = missing_arg,
    std.lv = std_lv
  )
  if (se_mode != "standard") {
    fit_args$se <- se_mode
  }
  if (isTRUE(force_standard_se)) {
    fit_args$se <- "standard"
    fit_args$test <- "standard"
  }
  if (se_mode == "robust.cluster") {
    fit_args$cluster <- cluster_col
  }
  if (se_mode == "weighted") {
    fit_args$sampling.weights <- weight_col
  }
  group_equal <- step_eq[[step]]
  if (length(group_equal) > 0L) {
    fit_args$group.equal <- group_equal
  }
  if (length(partial_intercepts) > 0L && any(group_equal %in% c("intercepts", "means"))) {
    fit_args$group.partial <- partial_intercepts
  }

  fit <- fit_step_with_retries(step, fit_args)

  measures <- tryCatch(
    fitMeasures(fit),
    error = function(err) {
      stop(sprintf("Failed to extract fitMeasures for step '%s': %s", step, conditionMessage(err)))
    }
  )
  chisq_scaled <- extract_measure(measures, "chisq.scaled")
  if (is.na(chisq_scaled)) {
    chisq_scaled <- extract_measure(measures, "chisq")
  }

  model_fit_rows[[length(model_fit_rows) + 1L]] <- data.frame(
    cohort = cohort,
    model_step = step,
    cfi = extract_measure(measures, "cfi"),
    tli = extract_measure(measures, "tli"),
    rmsea = extract_measure(measures, "rmsea"),
    srmr = extract_measure(measures, "srmr"),
    chisq_scaled = chisq_scaled,
    df = extract_measure(measures, "df"),
    aic = extract_measure(measures, "aic"),
    bic = extract_measure(measures, "bic"),
    stringsAsFactors = FALSE
  )

  ptable <- tryCatch(
    parameterEstimates(fit, standardized = TRUE),
    error = function(err) {
      stop(sprintf("Failed to extract parameter estimates for step '%s': %s", step, conditionMessage(err)))
    }
  )
  # Prefer numeric group index -> observed group value mapping to avoid
  # lavaan-internal label ordering drift in group.label.
  group_map <- if ("group" %in% names(ptable)) {
    vapply(
      ptable$group,
      resolve_group_label,
      character(1),
      group_levels = group_levels
    )
  } else if ("group.label" %in% names(ptable)) {
    as.character(ptable$group.label)
  } else {
    rep(NA_character_, nrow(ptable))
  }
  if ("group.label" %in% names(ptable)) {
    group_label_values <- as.character(ptable$group.label)
    fill_mask <- is.na(group_map) | trim_ws(group_map) == ""
    group_map[fill_mask] <- group_label_values[fill_mask]
  }
  ptable$group <- group_map

  groups <- infer_sex_groups(group_levels, reference_group = reference_group)
  female_group <- groups$female
  male_group <- groups$male

  lavaan_group_labels <- tryCatch(
    as.character(lavInspect(fit, "group.label")),
    error = function(err) character(0)
  )
  if (length(lavaan_group_labels) == 0L) {
    lavaan_group_labels <- rep(NA_character_, length(group_levels))
  }
  if (length(lavaan_group_labels) < length(group_levels)) {
    lavaan_group_labels <- c(
      lavaan_group_labels,
      rep(NA_character_, length(group_levels) - length(lavaan_group_labels))
    )
  }
  group_equal_str <- if (length(group_equal) > 0L) paste(group_equal, collapse = ";") else NA_character_
  group_partial_str <- if (length(partial_intercepts) > 0L) paste(partial_intercepts, collapse = ";") else NA_character_
  for (g_idx in seq_along(group_levels)) {
    group_audit_rows[[length(group_audit_rows) + 1L]] <- data.frame(
      cohort = cohort,
      model_step = step,
      group_index = as.integer(g_idx),
      group_label = as.character(group_levels[[g_idx]]),
      lavaan_group_label = as.character(lavaan_group_labels[[g_idx]]),
      reference_group = reference_group,
      female_group = female_group,
      male_group = male_group,
      stringsAsFactors = FALSE
    )
  }

  param_rows[[length(param_rows) + 1L]] <- data.frame(
    cohort = cohort,
    model_step = step,
    group = ptable$group,
    lhs = as.character(ptable$lhs),
    op = as.character(ptable$op),
    rhs = as.character(ptable$rhs),
    est = as.numeric(ptable$est),
    se = as.numeric(ptable$se),
    z = as.numeric(ptable$z),
    p = as.numeric(ptable$pvalue),
    std_all = as.numeric(ptable$std.all),
    stringsAsFactors = FALSE
  )

  for (factor_name in latent_factors) {
    mean_rows <- ptable[ptable$lhs == factor_name & ptable$op == "~1", , drop = FALSE]
    var_rows <- ptable[
      ptable$lhs == factor_name & ptable$rhs == factor_name & ptable$op == "~~",
      ,
      drop = FALSE
    ]
    for (group_name in group_levels) {
      mean_val <- mean_rows$est[mean_rows$group == group_name][1L]
      var_val <- var_rows$est[var_rows$group == group_name][1L]
      if (length(mean_val) == 0L) {
        mean_val <- NA_real_
      }
      if (length(var_val) == 0L) {
        var_val <- NA_real_
      }
      latent_rows[[length(latent_rows) + 1L]] <- data.frame(
        cohort = cohort,
        group = group_name,
        factor = factor_name,
        mean = as.numeric(mean_val),
        var = as.numeric(var_val),
        sd = sqrt(as.numeric(var_val)),
        stringsAsFactors = FALSE
      )
    }

    mean_female <- if (!is.na(female_group)) first_num_or_na(mean_rows$est[mean_rows$group == female_group]) else NA_real_
    mean_male <- if (!is.na(male_group)) first_num_or_na(mean_rows$est[mean_rows$group == male_group]) else NA_real_
    var_female <- if (!is.na(female_group)) first_num_or_na(var_rows$est[var_rows$group == female_group]) else NA_real_
    var_male <- if (!is.na(male_group)) first_num_or_na(var_rows$est[var_rows$group == male_group]) else NA_real_
    se_mean_female <- if (!is.na(female_group)) first_num_or_na(mean_rows$se[mean_rows$group == female_group]) else NA_real_
    se_mean_male <- if (!is.na(male_group)) first_num_or_na(mean_rows$se[mean_rows$group == male_group]) else NA_real_
    se_var_female <- if (!is.na(female_group)) first_num_or_na(var_rows$se[var_rows$group == female_group]) else NA_real_
    se_var_male <- if (!is.na(male_group)) first_num_or_na(var_rows$se[var_rows$group == male_group]) else NA_real_

    d_g <- if (is.finite(mean_male) && is.finite(mean_female)) {
      mean_male - mean_female
    } else {
      NA_real_
    }
    vr <- if (is.finite(var_male) && is.finite(var_female) && var_male > 0 && var_female > 0) {
      var_male / var_female
    } else {
      NA_real_
    }
    log_vr <- if (is.finite(vr) && vr > 0) log(vr) else NA_real_
    d_g_standardized <- d_g
    if (!std_lv && is.finite(var_female) && var_female > 0) {
      d_g_standardized <- d_g / sqrt(var_female)
    }

    sex_estimand_rows[[length(sex_estimand_rows) + 1L]] <- data.frame(
      cohort = cohort,
      model_step = step,
      factor = factor_name,
      female_group = female_group,
      male_group = male_group,
      mean_female = mean_female,
      mean_male = mean_male,
      var_female = var_female,
      var_male = var_male,
      se_mean_female = se_mean_female,
      se_mean_male = se_mean_male,
      se_var_female = se_var_female,
      se_var_male = se_var_male,
      d_g = d_g,
      delta_iq = if (is.finite(d_g_standardized)) 15 * d_g_standardized else NA_real_,
      vr = vr,
      log_vr = log_vr,
      reference_group = reference_group,
      group_equal = group_equal_str,
      group_partial = group_partial_str,
      stringsAsFactors = FALSE
    )
  }

  mod <- tryCatch(
    modificationIndices(fit, sort = TRUE),
    error = function(err) {
      warning(sprintf(
        "Modification indices unavailable for step '%s': %s. Writing empty MI rows for this step.",
        step,
        conditionMessage(err)
      ))
      NULL
    }
  )
  if (is.null(mod)) {
    mod <- data.frame()
  } else {
    mod <- as.data.frame(mod)
  }
  if (nrow(mod) == 0L) {
    mod_rows[[length(mod_rows) + 1L]] <- data.frame(
      cohort = character(0),
      model_step = character(0),
      lhs = character(0),
      op = character(0),
      rhs = character(0),
      mi = numeric(0),
      epc = numeric(0),
      stringsAsFactors = FALSE
    )
  } else {
    mod_lhs <- if ("lhs" %in% names(mod)) as.character(mod$lhs) else rep(NA_character_, nrow(mod))
    mod_op <- if ("op" %in% names(mod)) as.character(mod$op) else rep(NA_character_, nrow(mod))
    mod_rhs <- if ("rhs" %in% names(mod)) as.character(mod$rhs) else rep(NA_character_, nrow(mod))
    mod_mi <- if ("mi" %in% names(mod)) as.numeric(mod$mi) else rep(NA_real_, nrow(mod))
    mod_epc <- if ("epc" %in% names(mod)) as.numeric(mod$epc) else rep(NA_real_, nrow(mod))
    mod_rows[[length(mod_rows) + 1L]] <- data.frame(
      cohort = rep(cohort, nrow(mod)),
      model_step = rep(step, nrow(mod)),
      lhs = mod_lhs,
      op = mod_op,
      rhs = mod_rhs,
      mi = mod_mi,
      epc = mod_epc,
      stringsAsFactors = FALSE
    )
  }

  score <- tryCatch(
    lavTestScore(fit),
    error = function(err) {
      warning(sprintf(
        "Score tests unavailable for step '%s': %s. Writing empty score rows for this step.",
        step,
        conditionMessage(err)
      ))
      NULL
    }
  )
  score_uni <- if (!is.null(score) && "uni" %in% names(score)) {
    as.data.frame(score$uni)
  } else {
    data.frame()
  }
  if (nrow(score_uni) == 0L) {
    score_rows[[length(score_rows) + 1L]] <- data.frame(
      cohort = character(0),
      model_step = character(0),
      lhs = character(0),
      op = character(0),
      rhs = character(0),
      x2 = numeric(0),
      df = numeric(0),
      p_value = numeric(0),
      mapped_lhs = character(0),
      mapped_op = character(0),
      mapped_rhs = character(0),
      mapped_group_lhs = numeric(0),
      mapped_group_rhs = numeric(0),
      constraint_type = character(0),
      stringsAsFactors = FALSE
    )
  } else {
    score_lhs <- if ("lhs" %in% names(score_uni)) as.character(score_uni$lhs) else rep(NA_character_, nrow(score_uni))
    score_op <- if ("op" %in% names(score_uni)) as.character(score_uni$op) else rep(NA_character_, nrow(score_uni))
    score_rhs <- if ("rhs" %in% names(score_uni)) as.character(score_uni$rhs) else rep(NA_character_, nrow(score_uni))
    score_x2 <- if ("X2" %in% names(score_uni)) as.numeric(score_uni$X2) else rep(NA_real_, nrow(score_uni))
    score_df <- if ("df" %in% names(score_uni)) as.numeric(score_uni$df) else rep(NA_real_, nrow(score_uni))
    score_p <- if ("p.value" %in% names(score_uni)) {
      as.numeric(score_uni[["p.value"]])
    } else if ("pvalue" %in% names(score_uni)) {
      as.numeric(score_uni[["pvalue"]])
    } else {
      rep(NA_real_, nrow(score_uni))
    }

    ptable_full <- tryCatch(
      as.data.frame(parTable(fit)),
      error = function(err) data.frame()
    )
    ptable_plabel <- if ("plabel" %in% names(ptable_full)) as.character(ptable_full$plabel) else character(0)

    mapped_lhs <- rep(NA_character_, nrow(score_uni))
    mapped_op <- rep(NA_character_, nrow(score_uni))
    mapped_rhs <- rep(NA_character_, nrow(score_uni))
    mapped_group_lhs <- rep(NA_real_, nrow(score_uni))
    mapped_group_rhs <- rep(NA_real_, nrow(score_uni))
    constraint_type <- rep("other", nrow(score_uni))

    for (idx in seq_len(nrow(score_uni))) {
      lhs_token <- score_lhs[[idx]]
      rhs_token <- score_rhs[[idx]]
      if (length(ptable_plabel) == 0L) {
        next
      }
      lhs_match <- which(ptable_plabel == lhs_token)
      rhs_match <- which(ptable_plabel == rhs_token)
      if (length(lhs_match) == 0L || length(rhs_match) == 0L) {
        next
      }
      lhs_row <- ptable_full[lhs_match[[1]], , drop = FALSE]
      rhs_row <- ptable_full[rhs_match[[1]], , drop = FALSE]

      mapped_lhs[[idx]] <- as.character(lhs_row$lhs[[1]])
      mapped_op[[idx]] <- as.character(lhs_row$op[[1]])
      mapped_rhs[[idx]] <- as.character(lhs_row$rhs[[1]])
      mapped_group_lhs[[idx]] <- suppressWarnings(as.numeric(lhs_row$group[[1]]))
      mapped_group_rhs[[idx]] <- suppressWarnings(as.numeric(rhs_row$group[[1]]))

      op_name <- mapped_op[[idx]]
      if (!is.na(op_name) && op_name == "~1") {
        constraint_type[[idx]] <- "intercept"
      } else if (!is.na(op_name) && op_name == "=~") {
        constraint_type[[idx]] <- "loading"
      } else if (!is.na(op_name) && op_name == "~~") {
        constraint_type[[idx]] <- "residual"
      }
    }

    score_rows[[length(score_rows) + 1L]] <- data.frame(
      cohort = rep(cohort, nrow(score_uni)),
      model_step = rep(step, nrow(score_uni)),
      lhs = score_lhs,
      op = score_op,
      rhs = score_rhs,
      x2 = score_x2,
      df = score_df,
      p_value = score_p,
      mapped_lhs = mapped_lhs,
      mapped_op = mapped_op,
      mapped_rhs = mapped_rhs,
      mapped_group_lhs = mapped_group_lhs,
      mapped_group_rhs = mapped_group_rhs,
      constraint_type = constraint_type,
      stringsAsFactors = FALSE
    )
  }
}

fit_indices <- do.call(rbind, model_fit_rows)
params <- do.call(rbind, param_rows)
latent_summary <- do.call(rbind, latent_rows)
modindices <- if (length(mod_rows)) {
  do.call(rbind, mod_rows)
} else {
  data.frame(
    cohort = character(0),
    model_step = character(0),
    lhs = character(0),
    op = character(0),
    rhs = character(0),
    mi = numeric(0),
    epc = numeric(0),
    stringsAsFactors = FALSE
  )
}
lavtestscore <- if (length(score_rows)) {
  do.call(rbind, score_rows)
} else {
  data.frame(
    cohort = character(0),
    model_step = character(0),
    lhs = character(0),
    op = character(0),
    rhs = character(0),
    x2 = numeric(0),
    df = numeric(0),
    p_value = numeric(0),
    mapped_lhs = character(0),
    mapped_op = character(0),
    mapped_rhs = character(0),
    mapped_group_lhs = numeric(0),
    mapped_group_rhs = numeric(0),
    constraint_type = character(0),
    stringsAsFactors = FALSE
  )
}
group_label_audit <- if (length(group_audit_rows)) {
  do.call(rbind, group_audit_rows)
} else {
  data.frame(
    cohort = character(0),
    model_step = character(0),
    group_index = numeric(0),
    group_label = character(0),
    lavaan_group_label = character(0),
    reference_group = character(0),
    female_group = character(0),
    male_group = character(0),
    stringsAsFactors = FALSE
  )
}
sex_group_estimands <- if (length(sex_estimand_rows)) {
  do.call(rbind, sex_estimand_rows)
} else {
  data.frame(
    cohort = character(0),
    model_step = character(0),
    factor = character(0),
    female_group = character(0),
    male_group = character(0),
    mean_female = numeric(0),
    mean_male = numeric(0),
    var_female = numeric(0),
    var_male = numeric(0),
    se_mean_female = numeric(0),
    se_mean_male = numeric(0),
    se_var_female = numeric(0),
    se_var_male = numeric(0),
    d_g = numeric(0),
    delta_iq = numeric(0),
    vr = numeric(0),
    log_vr = numeric(0),
    reference_group = character(0),
    group_equal = character(0),
    group_partial = character(0),
    stringsAsFactors = FALSE
  )
}

write.csv(fit_indices, file.path(outdir, "fit_indices.csv"), row.names = FALSE)
write.csv(params, file.path(outdir, "params.csv"), row.names = FALSE)
write.csv(latent_summary, file.path(outdir, "latent_summary.csv"), row.names = FALSE)
write.csv(modindices, file.path(outdir, "modindices.csv"), row.names = FALSE)
write.csv(lavtestscore, file.path(outdir, "lavtestscore.csv"), row.names = FALSE)
write.csv(group_label_audit, file.path(outdir, "group_label_audit.csv"), row.names = FALSE)
write.csv(sex_group_estimands, file.path(outdir, "sex_group_estimands.csv"), row.names = FALSE)

cat(sprintf("Wrote SEM outputs for %s to %s\n", cohort, outdir))
