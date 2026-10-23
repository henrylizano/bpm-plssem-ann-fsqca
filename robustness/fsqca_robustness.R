# =============================================================================
# ROBUSTNESS ANALYSIS (3/3) — ROBUSTNESS TEST PROTOCOL FOR fsQCA
# Tri-methodological BPM article (PLS-SEM + ANN + fsQCA) — Appendix C
# Author: Henry Lizano-Mora · MIT License
# =============================================================================
# R port of fsqca_robustness.py on top of the canonical QCA package
# (calibrate, truthTable, minimize). Implements the Oana & Schneider (2024)
# protocol:
#   Test 1 — Sensitivity ranges (anchors, consistency, PRI and frequency
#            cut-offs), one parameter at a time.
#   Test 2 — Fit-oriented robustness: test set (TS) of 1,152 plausible
#            specifications; RF_cons, RF_cov and set coincidence between the
#            initial solution (IS) and TS_min / TS_max.
#   Test 3 — Case-oriented robustness: RCR_typ, RCR_dev and classification of
#            each case (robust / shaky / possible).
#   Robust core.
# The reference implementation of these tests is the SetMethods package
# (rob.calibrange, rob.fit, rob.case); this script recomputes them
# transparently to keep the correspondence with the Python version.
#
# Requirements: install.packages(c("QCA", "jsonlite"))
# Usage: Rscript fsqca_robustness.R   (requires final_dataset_plssem.csv)
# Output: fsqca_robustness_results_R.json
# =============================================================================
suppressPackageStartupMessages({ library(QCA); library(jsonlite) })

# ------------------------------------------------------------------ data
data <- read.csv("final_dataset_plssem.csv"); data <- data[complete.cases(data), ]
SA_I <- c("Ae1", "Ae2", "Ae3"); PEO_I <- c("Ep1", "Ep2", "Ep3", "Ep4", "Ep5")
GOV_I <- c("Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4"); BPO_I <- c("Co7", "Ae10", "Md7")
zscore <- function(x) (x - mean(x)) / sd(x)
cscore <- function(df, items) rowMeans(sapply(df[items], zscore))
RAW <- data.frame(SA = cscore(data, SA_I), PEO = cscore(data, PEO_I),
                  GOVMEAS = cscore(data, GOV_I), BPO = rowMeans(data[, BPO_I]))
COND <- c("SA", "PEO", "GOVMEAS"); N <- nrow(RAW)

# ------------------------------------------------------------ calibration
calib_all <- function(p_in = 95, p_cross = 50, p_ex = 5) {
  f <- function(x) { q <- quantile(x, c(p_ex, p_cross, p_in) / 100, names = FALSE)
                     calibrate(x, type = "fuzzy", thresholds = c(q[1], q[2], q[3])) }
  as.data.frame(lapply(RAW, f))
}
# ----------------------------------------------------- set-theoretic measures
incl <- function(m, y) sum(pmin(m, y)) / sum(m)
cov_ <- function(m, y) sum(pmin(m, y)) / sum(y)
pri  <- function(m, y) { a <- sum(pmin(m, y)); b <- sum(pmin(pmin(m, y), 1 - y)); d <- sum(m) - b
                         if (d > 1e-9) (a - b) / d else 0 }
sc   <- function(a, b) sum(pmin(a, b)) / sum(pmax(a, b))

term_m <- function(term, F) {               # membership of a term "~SA*GOVMEAS"
  m <- rep(1, nrow(F))
  for (lit in strsplit(term, "*", fixed = TRUE)[[1]]) {
    lit <- trimws(lit); neg <- startsWith(lit, "~"); nm <- sub("^~", "", lit)
    if (!(nm %in% names(F))) { neg <- TRUE; nm <- toupper(nm) }   # lower-case notation
    m <- pmin(m, if (neg) 1 - F[[nm]] else F[[nm]])
  }
  m
}
sol_m <- function(terms, F) if (length(terms)) do.call(pmax, lapply(terms, term_m, F = F)) else rep(0, nrow(F))

solve_qca <- function(p_in = 95, p_cross = 50, p_ex = 5, incl_cut = 0.80, pri_cut = 0.70, n_cut = 1) {
  F <- calib_all(p_in, p_cross, p_ex)
  tt <- tryCatch(suppressWarnings(truthTable(F, outcome = "BPO", conditions = COND, incl.cut = incl_cut,
                                             pri.cut = pri_cut, n.cut = n_cut, complete = TRUE)),
                 error = function(e) NULL)
  if (is.null(tt) || !any(tt$tt$OUT == "1")) return(list(expr = "∅", terms = character(0), F = F))
  sol <- tryCatch(suppressMessages(minimize(tt, details = TRUE)), error = function(e) NULL)
  if (is.null(sol)) return(list(expr = "∅", terms = character(0), F = F))
  terms <- sort(sol$solution[[1]])
  list(expr = paste(terms, collapse = " + "), terms = terms, F = F, tt = tt)
}

# ===================================================== SOLUCIÓN INICIAL (IS)
cat(strrep("=", 74), "\nINITIAL SOLUTION (IS) — anchors 95/50/5, incl ≥ 0.80, PRI ≥ 0.70, n ≥ 1\n", strrep("=", 74), "\n", sep = "")
IS <- solve_qca(); F0 <- IS$F; y0 <- F0$BPO
cat("  Necessity (high BPO):\n");  print(pof(F0[, COND], y0, relation = "necessity")$incl.cov)
cat("  Necessity (~BPO):\n");      print(pof(1 - F0[, COND], 1 - y0, relation = "necessity")$incl.cov)
cat("\n  Truth table:\n");     print(IS$tt)
IS_m <- sol_m(IS$terms, F0)
cat(sprintf("\n  IS = %s\n", IS$expr))
for (t in IS$terms) {
  mt <- term_m(t, F0); others <- setdiff(IS$terms, t)
  covU <- if (length(others)) (sum(pmin(IS_m, y0)) - sum(pmin(sol_m(others, F0), y0))) / sum(y0) else cov_(mt, y0)
  cat(sprintf("    %-16s inclS=%.3f PRI=%.3f covS=%.3f covU=%.3f\n", t, incl(mt, y0), pri(mt, y0), cov_(mt, y0), covU))
}
cat(sprintf("  inclS=%.3f  PRI=%.3f  covS=%.3f\n", incl(IS_m, y0), pri(IS_m, y0), cov_(IS_m, y0)))

# ================================================ TEST 1 — SENSITIVITY RANGES
cat("\n", strrep("=", 74), "\nTEST 1 — SENSITIVITY RANGES (one parameter at a time)\n", strrep("=", 74), "\n", sep = "")
BASE  <- list(p_in = 95, p_cross = 50, p_ex = 5, incl_cut = 0.80, pri_cut = 0.70, n_cut = 1)
GRIDS <- list(p_in = seq(80, 99, 0.5), p_cross = seq(30, 70, 0.5), p_ex = seq(0.5, 20, 0.5),
              incl_cut = seq(0.70, 0.99, 0.005), pri_cut = seq(0.50, 0.94, 0.005), n_cut = 1:5)
sens <- list()
for (par in names(GRIDS)) {
  vals <- GRIDS[[par]]; same <- logical(0); alts <- character(0)
  for (v in vals) { kw <- BASE; kw[[par]] <- v
    s <- tryCatch(do.call(solve_qca, kw), error = function(e) NULL)
    if (is.null(s)) { same <- c(same, NA); alts <- c(alts, NA); next }
    same <- c(same, s$expr == IS$expr); alts <- c(alts, s$expr) }
  idx <- which.min(abs(vals - BASE[[par]])); lo <- idx; hi <- idx
  while (lo > 1 && isTRUE(same[lo - 1])) lo <- lo - 1
  while (hi < length(vals) && isTRUE(same[hi + 1])) hi <- hi + 1
  sens[[par]] <- list(range = c(vals[lo], vals[hi]), share_same = mean(same, na.rm = TRUE),
                      grid = range(vals), alternatives = as.list(table(alts[!is.na(same) & !same])))
  cat(sprintf("  %-9s base=%-6s IS holds over [%g, %g] of the grid [%g, %g]  (%.0f%% of the grid)\n",
              par, BASE[[par]], vals[lo], vals[hi], min(vals), max(vals), 100 * mean(same, na.rm = TRUE)))
  alt_tab <- sort(table(alts[!is.na(same) & !same]), decreasing = TRUE)
  for (k in seq_len(min(3, length(alt_tab)))) cat(sprintf("            alternative: %s  (%d values)\n", names(alt_tab)[k], alt_tab[k]))
}

# ===================================== TEST 2 — FIT-ORIENTED ROBUSTNESS
cat("\n", strrep("=", 74), "\nTEST 2 — FIT-ORIENTED ROBUSTNESS (test set TS)\n", strrep("=", 74), "\n", sep = "")
TS_GRID <- expand.grid(p_in = c(90, 92.5, 95, 97.5), p_cross = c(45, 50, 55), p_ex = c(2.5, 5, 7.5, 10),
                       incl_cut = c(0.75, 0.80, 0.85, 0.90), pri_cut = c(0.65, 0.70, 0.75), n_cut = c(1, 2))
TS <- vector("list", nrow(TS_GRID))
for (i in seq_len(nrow(TS_GRID))) TS[[i]] <- do.call(solve_qca, as.list(TS_GRID[i, ]))
exprs <- sapply(TS, `[[`, "expr")
cat(sprintf("  Plausible solutions evaluated: %d\n  Distribution of solutions across TS:\n", length(TS)))
dist <- sort(table(exprs), decreasing = TRUE)
for (k in seq_len(min(8, length(dist)))) cat(sprintf("    %4d (%2.0f%%)  %s\n", dist[k], 100 * dist[k] / length(TS), names(dist)[k]))
all_terms <- unlist(lapply(TS, `[[`, "terms"))
term_freq <- sort(table(all_terms), decreasing = TRUE) / length(TS)
cat("\n  Term frequency across TS:\n"); for (k in seq_along(term_freq)) cat(sprintf("    %-22s %.0f%%\n", names(term_freq)[k], 100 * term_freq[k]))
gov_all <- mean(sapply(TS, function(s) length(s$terms) > 0 && all(grepl("(^|\\*)GOVMEAS", s$terms))))
cat(sprintf("  GOVMEAS present in ALL terms: %.0f%% of TS\n", 100 * gov_all))

# TS_min / TS_max evaluated in the IS calibration so they are comparable
nonempty <- Filter(function(s) length(s$terms) > 0, TS)
Ms <- do.call(rbind, lapply(nonempty, function(s) sol_m(s$terms, F0)))
TSmin <- apply(Ms, 2, min); TSmax <- apply(Ms, 2, max)
cat(sprintf("\n  %-8s%8s%8s%10s\n", "", "inclS", "covS", "SC(IS,·)"))
cat(sprintf("  %-8s%8.3f%8.3f%10s\n", "IS", incl(IS_m, y0), cov_(IS_m, y0), "—"))
cat(sprintf("  %-8s%8.3f%8.3f%10.3f\n", "TS_min", incl(TSmin, y0), cov_(TSmin, y0), sc(IS_m, TSmin)))
cat(sprintf("  %-8s%8.3f%8.3f%10.3f\n", "TS_max", incl(TSmax, y0), cov_(TSmax, y0), sc(IS_m, TSmax)))
cat(sprintf("  RF_cons = inclS(IS) − inclS(TS_min) = %+.3f\n", incl(IS_m, y0) - incl(TSmin, y0)))
cat(sprintf("  RF_cov  = covS(IS) − covS(TS_min)   = %+.3f\n", cov_(IS_m, y0) - cov_(TSmin, y0)))

# ===================================== TEST 3 — CASE-ORIENTED ROBUSTNESS
cat("\n", strrep("=", 74), "\nTEST 3 — CASE-ORIENTED ROBUSTNESS\n", strrep("=", 74), "\n", sep = "")
typ_IS <- IS_m > 0.5 & y0 > 0.5; dev_IS <- IS_m > 0.5 & y0 <= 0.5
typ_min <- TSmin > 0.5 & y0 > 0.5; typ_max <- TSmax > 0.5 & y0 > 0.5; dev_min <- TSmin > 0.5 & y0 <= 0.5
RCR_typ <- sum(typ_IS & typ_min) / sum(typ_IS); RCR_dev <- if (sum(dev_IS)) sum(dev_IS & dev_min) / sum(dev_IS) else NA
cat(sprintf("  Typical cases in IS: %d  → robust %d, shaky %d, possible (only in TS_max) %d\n",
            sum(typ_IS), sum(typ_IS & typ_min), sum(typ_IS & !typ_min), sum(!typ_IS & typ_max)))
cat(sprintf("  Deviant cases (consistency) in IS: %d  → robust %d\n", sum(dev_IS), sum(dev_IS & dev_min)))
cat(sprintf("  RCR_typ = %.3f   RCR_dev = %.3f\n", RCR_typ, RCR_dev))
above <- colMeans(Ms > 0.5)
cat(sprintf("  Solution membership: mean σ across TS = %.3f; cases with membership >0.5 in ≥95%% of TS: %d; in ≤5%%: %d; ambiguous: %d\n",
            mean(apply(Ms, 2, sd)), sum(above >= 0.95), sum(above <= 0.05), sum(above > 0.05 & above < 0.95)))

# ============================================================ ROBUST CORE
cat("\n", strrep("=", 74), "\nROBUST CORE\n", strrep("=", 74), "\n", sep = "")
tf_ne <- table(unlist(lapply(nonempty, `[[`, "terms")))
always <- names(tf_ne)[tf_ne == length(nonempty)]
gov_core <- all(sapply(nonempty, function(s) all(grepl("(^|\\*)GOVMEAS", s$terms))))
cat(sprintf("  Terms present in 100 %% of non-empty TS: %s\n", if (length(always)) paste(always, collapse = ", ") else "none literally"))
cat(sprintf("  GOVMEAS present in every term of every solution: %s\n", gov_core))
cat(sprintf("  Empty solutions in TS: %d of %d\n", length(TS) - length(nonempty), length(TS)))

write_json(list(IS = list(expr = IS$expr, cons = incl(IS_m, y0), pri = pri(IS_m, y0), cov = cov_(IS_m, y0)),
                sensitivity = sens, TS_n = length(TS), TS_dist = as.list(dist), term_freq = as.list(term_freq),
                gov_in_all_terms_share = gov_all,
                fit = list(IS = list(cons = incl(IS_m, y0), cov = cov_(IS_m, y0)),
                           TSmin = list(cons = incl(TSmin, y0), cov = cov_(TSmin, y0), SC = sc(IS_m, TSmin)),
                           TSmax = list(cons = incl(TSmax, y0), cov = cov_(TSmax, y0), SC = sc(IS_m, TSmax))),
                cases = list(typ_IS = sum(typ_IS), robust = sum(typ_IS & typ_min), shaky = sum(typ_IS & !typ_min),
                             possible = sum(!typ_IS & typ_max), dev_IS = sum(dev_IS), RCR_typ = RCR_typ, RCR_dev = RCR_dev,
                             stable_in = sum(above >= 0.95), stable_out = sum(above <= 0.05),
                             ambiguous = sum(above > 0.05 & above < 0.95)),
                robust_core = list(always = always, gov_core = gov_core)),
           "fsqca_robustness_results_R.json", auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("\n→ fsqca_robustness_results_R.json\n")
