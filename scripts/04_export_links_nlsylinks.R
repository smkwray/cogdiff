#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(NlsyLinks)
})

dir.create("data/interim/links", recursive = TRUE, showWarnings = FALSE)

utils::write.csv(Links79PairExpanded, "data/interim/links/links79_pair_expanded.csv", row.names = FALSE)
utils::write.csv(Links97PairExpanded, "data/interim/links/links97_pair_expanded.csv", row.names = FALSE)

cat("Exported NlsyLinks expanded pair files to data/interim/links\n")
