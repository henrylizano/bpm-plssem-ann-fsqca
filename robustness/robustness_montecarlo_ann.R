# =============================================================================
# ROBUSTNESS ANALYSIS (2/3) — SPECIFICITY OF THE ROBUST ANN PROTOCOL
# Tri-methodological BPM article (PLS-SEM + ANN + fsQCA) — Appendix C
# Author: Henry Lizano-Mora · MIT License
# =============================================================================
# R port of robustness_montecarlo_ann.py using the article's canonical
# packages (nnet + NeuralNetTools::garson + caret::createFolds).
#
# Question: does the robust protocol (10 networks × 10 folds, 3 hidden
# neurons, decay = 0.1, logistic activation, nnet's BFGS optimiser) return a
# Garson importance of ~33/33/33 when the three dimensions contribute equally,
# and does it recover the observed hierarchy when the DGP is the real one?
#
# Design (n = 56; article preprocessing: stage-1 scores and the mean of the
# BPO items rescaled Min-Max):
#   • CONTROL: robust protocol on the real sample (≈ 29.7 / 28.8 / 41.4).
#   • H0a: equal bivariate correlations with BPO (diagnostic).
#   • H0b: equal conditional contribution (identical β_j, R² preserved) —
#          the correct null hypothesis of equal contribution.
#   • H1 : real DGP (observed means and covariances).
# In every replication the single-run network (10 neurons, no regularisation)
# is compared with the robust protocol.
#
# Note: NeuralNetTools::garson applies Garson's algorithm with per-hidden-unit
# normalisation; the Python version uses the simplified weight-product
# variant. Both converge (±0.5 pp) on the real sample.
#
# Requirements: install.packages(c("MASS", "nnet", "NeuralNetTools", "caret", "jsonlite"))
# Usage: Rscript robustness_montecarlo_ann.R   (requires final_dataset_plssem.csv)
# Output: montecarlo_ann_results_R.json
# =============================================================================
suppressPackageStartupMessages({
  library(MASS); library(nnet); library(NeuralNetTools); library(caret); library(jsonlite)
})
t0 <- Sys.time()

# ------------------------------------------------------------------ data
data <- read.csv("final_dataset_plssem.csv"); data <- data[complete.cases(data), ]
SA_I <- c("Ae1", "Ae2", "Ae3"); PEO_I <- c("Ep1", "Ep2", "Ep3", "Ep4", "Ep5")
GOV_I <- c("Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4"); BPO_I <- c("Co7", "Ae10", "Md7")
zscore <- function(x) (x - mean(x)) / sd(x)
cscore <- function(df, items) rowMeans(sapply(df[items], zscore))
Z4  <- cbind(SA = cscore(data, SA_I), PEO = cscore(data, PEO_I),
             GOVMEAS = cscore(data, GOV_I), BPO = rowMeans(data[, BPO_I]))
MU  <- colMeans(Z4); SIG <- cov(Z4)
D   <- sqrt(diag(SIG)); R <- SIG / outer(D, D)
cat("Observed correlations with BPO:", round(R[1:3, 4], 3), "\n")
cat(sprintf("Correlations among dimensions     : SA-PEO %.3f  SA-GOV %.3f  PEO-GOV %.3f\n",
            R[1, 2], R[1, 3], R[2, 3]))

# ------------------------------------------------------- networks and Garson
mm_norm <- function(x) (x - min(x)) / (max(x) - min(x))
prep <- function(Z) data.frame(SA = mm_norm(Z[, 1]), PEO = mm_norm(Z[, 2]),
                               GOVMEAS = mm_norm(Z[, 3]), BPO = mm_norm(Z[, 4]))
garson_imp <- function(nn) {
  gi <- garson(nn, bar_plot = FALSE)
  imp <- gi$rel_imp[match(c("SA", "PEO", "GOVMEAS"), rownames(gi))]
  imp * 100 / sum(imp)
}
single <- function(d, seed) {                      # single-run network (unregularised)
  set.seed(seed)
  nn <- nnet(BPO ~ SA + PEO + GOVMEAS, data = d, size = 10, linout = TRUE,
             maxit = 2000, trace = FALSE)
  garson_imp(nn)
}
robust <- function(d, base = 100, n_nets = 10) {   # 10 regularised networks × 10 folds
  out <- matrix(NA, n_nets, 3)
  for (k in seq_len(n_nets)) {
    set.seed(base + k); folds <- createFolds(d$BPO, k = 10); fi <- matrix(NA, 10, 3)
    for (f in seq_along(folds)) {
      set.seed(base + k)
      nn <- nnet(BPO ~ SA + PEO + GOVMEAS, data = d[-folds[[f]], ], size = 3,
                 decay = 0.1, linout = TRUE, maxit = 2000, trace = FALSE)
      fi[f, ] <- garson_imp(nn)
    }
    out[k, ] <- colMeans(fi)
  }
  list(mean = colMeans(out), sd = apply(out, 2, sd))
}

# --- CONTROL on the real sample ----------------------------------------------
ctrl <- robust(prep(Z4))
cat(sprintf("\nCONTROL real sample: robust = %s (±%s)\n",
            paste(round(ctrl$mean, 1), collapse = " "), paste(round(ctrl$sd, 2), collapse = " ")))

# --- DGP H0a: equal bivariate correlations with BPO ------------------------
R0 <- R; R0[1:3, 4] <- R0[4, 1:3] <- mean(R[1:3, 4]); S0a <- R0 * outer(D, D)
# --- DGP H0b: equal CONDITIONAL contribution (identical β_j, R² preserved) ---
Rxx <- R[1:3, 1:3]; beta_eq <- c(1, 1, 1)
rho <- as.vector(Rxx %*% beta_eq)
cfac <- sqrt(as.numeric(t(R[1:3, 4]) %*% solve(Rxx, R[1:3, 4]))) /
        sqrt(as.numeric(t(beta_eq) %*% Rxx %*% beta_eq))          # match R²
rho <- rho * cfac
R0b <- R; R0b[1:3, 4] <- R0b[4, 1:3] <- rho; S0b <- R0b * outer(D, D)
pd_fix <- function(S) { ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 0) S <- S + diag(4) * (abs(min(ev)) + 1e-3); S }
S0a <- pd_fix(S0a); S0b <- pd_fix(S0b)

run_dgp <- function(S, label, Rn = 120, seed = 0) {
  set.seed(seed)
  rob <- matrix(NA, Rn, 3); sing <- matrix(NA, Rn, 3)
  for (r in seq_len(Rn)) {
    d <- prep(mvrnorm(56, MU, S))
    rob[r, ]  <- robust(d, base = 100 + r)$mean
    sing[r, ] <- single(d, seed = r)
  }
  summ <- function(name, A) {
    dom <- apply(A, 1, which.max); cnt <- tabulate(dom, 3) / nrow(A)
    gap <- apply(A, 1, max) - apply(A, 1, min)
    cat(sprintf("  %-18s mean=%s  σ=%s  P(dom): SA=%.2f PEO=%.2f GOV=%.2f  mean gap=%.1fpp  P(>10pp)=%.2f\n",
                name, paste(round(colMeans(A), 1), collapse = " "), paste(round(apply(A, 2, sd), 1), collapse = " "),
                cnt[1], cnt[2], cnt[3], mean(gap), mean(gap > 10)))
    list(mean = colMeans(A), sd = apply(A, 2, sd), dom = cnt, gap_mean = mean(gap), p_gap10 = mean(gap > 10))
  }
  cat(sprintf("\n%s (%d replications, n = 56)\n", label, Rn))
  list(single = summ("Single run", sing), robust = summ("Robust protocol", rob))
}
res <- list(control = ctrl$mean, corr_bpo = R[1:3, 4])
res$H0a_equal_bivariate   <- run_dgp(S0a, "H0a — equal bivariate correlations with BPO", seed = 11)
res$H0b_equal_conditional <- run_dgp(S0b, "H0b — equal conditional contribution (equal β_j)", seed = 22)

# --- H1: real DGP ------------------------------------------------------------
set.seed(33); rob <- matrix(NA, 120, 3)
for (r in seq_len(120)) rob[r, ] <- robust(prep(mvrnorm(56, MU, SIG)), base = 300 + r)$mean
gaps <- rob[, 3] - pmax(rob[, 1], rob[, 2])
cat("\nH1 — real DGP (120 replications, n = 56), robust protocol\n")
cat(sprintf("  mean=%s  σ=%s  P(GOV dominant)=%.3f  GOV lead: mean=%.1fpp  P(>5pp)=%.2f  CI95%%=[%.1f, %.1f]\n",
            paste(round(colMeans(rob), 1), collapse = " "), paste(round(apply(rob, 2, sd), 1), collapse = " "),
            mean(apply(rob, 1, which.max) == 3), mean(gaps), mean(gaps > 5),
            quantile(gaps, 0.025), quantile(gaps, 0.975)))
res$H1 <- list(mean = colMeans(rob), sd = apply(rob, 2, sd), p_gov = mean(apply(rob, 1, which.max) == 3),
               gap_mean = mean(gaps), gap_ci = as.numeric(quantile(gaps, c(0.025, 0.975))))
write_json(res, "montecarlo_ann_results_R.json", auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat(sprintf("\nTime: %.1f min → montecarlo_ann_results_R.json\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
