# =============================================================================
# ANÁLISIS DE ROBUSTEZ (1/3) — ESTUDIO MONTE CARLO DE RECUPERACIÓN DE PARÁMETROS
# Artículo tri-metodológico BPM (PLS-SEM + ANN + fsQCA) — Anexo C
# Autor: Henry Lizano-Mora · Licencia MIT
# =============================================================================
# Réplica en R del script robustez_montecarlo_pls.py (implementación de
# referencia, validada). El estimador PLS Modo B está codificado explícitamente
# para que la correspondencia con la versión Python sea línea a línea.
#
# Lógica:
#   • El proceso generador de datos (DGP) se fija en el vector de medias y la
#     matriz de covarianzas de los seis indicadores de la etapa 2 (scores SA,
#     PEO, GOVMEAS e ítems Co7, Ae10, Md7) observados en la muestra (n = 56).
#   • Para n ∈ {30, 40, 56, 80, 100, 150, 200}: sesgo, RMSE y desviación de β;
#     sesgo de los pesos formativos; cobertura del IC bootstrap al 95 %;
#     potencia para β ≥ 0.35; P(GOVMEAS dominante); P(peso PEO significativo).
#   • Escenario adverso: β poblacional fijado en 0.70, 0.50 y 0.35 (n = 56).
#
# Nota: los generadores pseudoaleatorios de R y NumPy difieren, por lo que las
# cifras coinciden con las del Anexo C hasta la variabilidad Monte Carlo
# (segundo o tercer decimal), no bit a bit.
#
# Requisitos: install.packages(c("MASS", "jsonlite"))
# Uso: Rscript robustez_montecarlo_pls.R   (requiere final_dataset_plssem.csv)
# Salida: montecarlo_pls_resultados_R.json
# =============================================================================
suppressPackageStartupMessages({ library(MASS); library(jsonlite) })
t0 <- Sys.time()

# ------------------------------------------------------------------ datos
data <- read.csv("final_dataset_plssem.csv")
data <- data[complete.cases(data), ]
SA_ITEMS  <- c("Ae1", "Ae2", "Ae3")
PEO_ITEMS <- c("Ep1", "Ep2", "Ep3", "Ep4", "Ep5")
GOV_ITEMS <- c("Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4")
BPO_ITEMS <- c("Co7", "Ae10", "Md7")
zscore <- function(x) (x - mean(x)) / sd(x)
cscore <- function(df, items) rowMeans(sapply(df[items], zscore))
BPC <- cbind(cscore(data, SA_ITEMS), cscore(data, PEO_ITEMS), cscore(data, GOV_ITEMS))
BPO <- as.matrix(data[, BPO_ITEMS])
Z6  <- cbind(BPC, BPO)                      # 56 × 6
MU  <- colMeans(Z6); SIGMA <- cov(Z6)

# ----------------------------------------------- estimador PLS Modo B
pls_formative <- function(X_ind, Y_ind, max_iter = 300, tol = 1e-7) {
  wx <- rep(1 / sqrt(ncol(X_ind)), ncol(X_ind))
  wy <- rep(1 / sqrt(ncol(Y_ind)), ncol(Y_ind))
  for (it in seq_len(max_iter)) {
    X <- as.vector(X_ind %*% wx); Y <- as.vector(Y_ind %*% wy)
    X <- (X - mean(X)) / sd(X);   Y <- (Y - mean(Y)) / sd(Y)
    b <- cor(X, Y)
    wx_new <- as.vector(qr.solve(X_ind, b * Y))          # mínimos cuadrados
    wy_new <- as.vector(qr.solve(Y_ind, b * X))
    wx_new <- wx_new / (sqrt(sum((X_ind %*% wx_new)^2)) / sqrt(length(X)))
    wy_new <- wy_new / (sqrt(sum((Y_ind %*% wy_new)^2)) / sqrt(length(Y)))
    done <- sum(abs(wx_new - wx)) + sum(abs(wy_new - wy)) < tol
    wx <- wx_new; wy <- wy_new
    if (done) break
  }
  X <- as.vector(X_ind %*% wx); Y <- as.vector(Y_ind %*% wy)
  X <- (X - mean(X)) / sd(X);   Y <- (Y - mean(Y)) / sd(Y)
  list(wx = wx, wy = wy, beta = cor(X, Y), X = X, Y = Y)
}
zmat <- function(M) scale(M)

fit_pls <- function(Z) {
  # Devuelve β, pesos normalizados BPC, cargas BPC, pesos BPO, scores
  Xz <- zmat(Z[, 1:3]); Yz <- zmat(Z[, 4:6])
  f <- pls_formative(Xz, Yz)
  sgn <- sign(cor(f$X, rowSums(Z[, 1:3]))); if (sgn == 0) sgn <- 1
  wx_n <- f$wx / sqrt(sum(f$wx^2)) * sgn
  wy_n <- f$wy / sqrt(sum(f$wy^2)) * sgn
  load <- sapply(1:3, function(i) cor(Xz[, i], f$X * sgn))
  list(beta = abs(f$beta), w = wx_n, load = load, wy = wy_n,
       Xs = f$X * sgn, Ys = f$Y * sgn)
}

# --- valores "poblacionales" = estimación en la muestra real -----------------
pop <- fit_pls(Z6)
beta_pop <- pop$beta; w_pop <- pop$w
cat(strrep("=", 72), "\nDGP fijado en la muestra real (n = 56)\n", strrep("=", 72), "\n", sep = "")
cat(sprintf("  β_pop = %.4f | pesos BPC = %s | cargas = %s | pesos BPO = %s\n",
            beta_pop, paste(round(pop$w, 3), collapse = " "),
            paste(round(pop$load, 3), collapse = " "), paste(round(pop$wy, 3), collapse = " ")))

boot_ci <- function(Z, B = 250) {
  # IC percentil bootstrap de β y pesos BPC
  n <- nrow(Z); bs_b <- numeric(0); bs_w <- NULL
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    r <- tryCatch(fit_pls(Z[idx, ]), error = function(e) NULL)
    if (is.null(r)) next
    bs_b <- c(bs_b, r$beta); bs_w <- rbind(bs_w, r$w)
  }
  list(b = quantile(bs_b, c(0.025, 0.975), names = FALSE),
       w = apply(bs_w, 2, quantile, probs = c(0.025, 0.975), names = FALSE))
}

# =============================================================================
# Recuperación de parámetros en función de n
# =============================================================================
R_MAIN <- 1000; R_COV <- 150; B_BOOT <- 250
SIZES  <- c(30, 40, 56, 80, 100, 150, 200)
results <- list()
cat("\n", strrep("=", 72), "\n", sprintf("MONTE CARLO DE RECUPERACIÓN DE PARÁMETROS (%d réplicas por n)", R_MAIN),
    "\n", strrep("=", 72), "\n", sep = "")
cat(sprintf("%4s %7s %7s %7s %12s %12s %14s %13s\n", "n", "β̄", "sesgo", "RMSE",
            "IC95% ancho", "P(GOV dom.)", "P(β>0.35 sig)", "P(w_PEO sig)"))
for (n in SIZES) {
  set.seed(1000 + n)
  betas <- numeric(0); ws <- NULL; ls <- NULL
  for (r in seq_len(R_MAIN)) {
    Z <- mvrnorm(n, MU, SIGMA)
    f <- tryCatch(fit_pls(Z), error = function(e) NULL); if (is.null(f)) next
    betas <- c(betas, f$beta); ws <- rbind(ws, f$w); ls <- rbind(ls, f$load)
  }
  gov_dom <- mean(apply(ws, 1, which.max) == 3)
  # cobertura y potencia con bootstrap en subconjunto de réplicas
  set.seed(5000 + n)
  cov_b <- 0; cov_w <- c(0, 0, 0); sig_b <- 0; sig_peo <- 0; widths <- numeric(0)
  for (r in seq_len(R_COV)) {
    Z <- mvrnorm(n, MU, SIGMA)
    ci <- tryCatch(boot_ci(Z, B = B_BOOT), error = function(e) NULL); if (is.null(ci)) next
    cov_b   <- cov_b + (ci$b[1] <= beta_pop && beta_pop <= ci$b[2])
    cov_w   <- cov_w + as.numeric(ci$w[1, ] <= w_pop & w_pop <= ci$w[2, ])
    sig_b   <- sig_b + (ci$b[1] > 0.35)       # β detectable como ≥ 0.35
    sig_peo <- sig_peo + (ci$w[1, 2] > 0)     # peso PEO significativo
    widths  <- c(widths, ci$b[2] - ci$b[1])
  }
  results[[as.character(n)]] <- list(
    beta_mean = mean(betas), beta_sd = sd(betas), bias = mean(betas) - beta_pop,
    rmse = sqrt(mean((betas - beta_pop)^2)),
    w_mean = colMeans(ws), w_sd = apply(ws, 2, sd), w_bias = colMeans(ws) - w_pop,
    load_mean = colMeans(ls), p_gov_dominant = gov_dom,
    coverage_beta = cov_b / R_COV, coverage_w = cov_w / R_COV,
    power_beta_035 = sig_b / R_COV, power_w_peo = sig_peo / R_COV,
    ci_width_mean = mean(widths))
  cat(sprintf("%4d %7.3f %+7.3f %7.3f %12.3f %12.3f %14.3f %13.3f\n", n, mean(betas),
              mean(betas) - beta_pop, results[[as.character(n)]]$rmse, mean(widths),
              gov_dom, sig_b / R_COV, sig_peo / R_COV))
}

cat("\n  Cobertura IC95% bootstrap (nominal 0.95):\n")
cat(sprintf("  %4s %7s %7s %7s %7s\n", "n", "β", "w_SA", "w_PEO", "w_GOV"))
for (n in SIZES) { r <- results[[as.character(n)]]
  cat(sprintf("  %4d %7.3f %7.3f %7.3f %7.3f\n", n, r$coverage_beta, r$coverage_w[1], r$coverage_w[2], r$coverage_w[3])) }

cat("\n  Sesgo de los pesos formativos (media − poblacional):\n")
cat(sprintf("  %4s %8s %8s %8s   σ(w_SA) σ(w_PEO) σ(w_GOV)\n", "n", "w_SA", "w_PEO", "w_GOV"))
for (n in SIZES) { r <- results[[as.character(n)]]
  cat(sprintf("  %4d %+8.3f %+8.3f %+8.3f   %7.3f %7.3f %7.3f\n", n, r$w_bias[1], r$w_bias[2], r$w_bias[3],
              r$w_sd[1], r$w_sd[2], r$w_sd[3])) }

# --- Escenario adverso: efecto poblacional más débil ------------------------
cat("\n  Escenario adverso: β_pop reducido artificialmente (n = 56)\n")
cat(sprintf("  %6s %7s %7s %14s %12s\n", "β_pop", "β̄", "sesgo", "P(IC excl. 0)", "P(GOV dom.)"))
for (target in c(0.934, 0.70, 0.50, 0.35)) {
  scale_f <- target / beta_pop
  S <- SIGMA; S[1:3, 4:6] <- S[1:3, 4:6] * scale_f; S[4:6, 1:3] <- S[4:6, 1:3] * scale_f
  ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values           # definida positiva
  if (min(ev) <= 0) S <- S + diag(6) * (abs(min(ev)) + 1e-3)
  set.seed(7000 + round(target * 100))
  bs <- numeric(0); gd <- logical(0); excl <- 0
  for (r in seq_len(300)) {
    Z <- mvrnorm(56, MU, S); f <- fit_pls(Z)
    bs <- c(bs, f$beta); gd <- c(gd, which.max(f$w) == 3)
    if (r <= 100) { ci <- boot_ci(Z, B = 250); excl <- excl + (ci$b[1] > 0) }
  }
  b_true <- fit_pls(mvrnorm(200000, MU, S))$beta
  cat(sprintf("  %6.3f %7.3f %+7.3f %14.3f %12.3f\n", b_true, mean(bs), mean(bs) - b_true, excl / 100, mean(gd)))
  results[[paste0("adverse_", target)]] <- list(beta_true = b_true, beta_mean = mean(bs),
                                                p_excl0 = excl / 100, p_gov = mean(gd))
}
write_json(results, "montecarlo_pls_resultados_R.json", auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat(sprintf("\nTiempo total: %.1f min → montecarlo_pls_resultados_R.json\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
