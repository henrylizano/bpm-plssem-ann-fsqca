# =============================================================================
# ANÁLISIS DE ROBUSTEZ (3/3) — PROTOCOLO DE ROBUSTEZ PARA fsQCA
# Artículo tri-metodológico BPM (PLS-SEM + ANN + fsQCA) — Anexo C
# Autor: Henry Lizano-Mora · Licencia MIT
# =============================================================================
# Réplica en R del script robustez_fsqca.py sobre el paquete canónico QCA
# (calibrate, truthTable, minimize). Implementa el protocolo de Oana y
# Schneider (2024):
#   Test 1 — Rangos de sensibilidad (anclajes, cortes de consistencia, PRI y
#            frecuencia), un parámetro a la vez.
#   Test 2 — Robustez orientada al ajuste: conjunto de prueba (TS) de 1 152
#            especificaciones plausibles; RF_cons, RF_cov y coincidencia de
#            conjuntos entre la solución inicial (IS) y TS_min / TS_max.
#   Test 3 — Robustez orientada a los casos: RCR_typ, RCR_dev y clasificación
#            de cada caso (robusto / inestable / posible).
#   Núcleo robusto.
# La implementación de referencia de estas pruebas es el paquete SetMethods
# (rob.calibrange, rob.fit, rob.case); este script las recalcula de forma
# transparente para mantener la correspondencia con la versión Python.
#
# Requisitos: install.packages(c("QCA", "jsonlite"))
# Uso: Rscript robustez_fsqca.R   (requiere final_dataset_plssem.csv)
# Salida: fsqca_robustez_resultados_R.json
# =============================================================================
suppressPackageStartupMessages({ library(QCA); library(jsonlite) })

# ------------------------------------------------------------------ datos
data <- read.csv("final_dataset_plssem.csv"); data <- data[complete.cases(data), ]
SA_I <- c("Ae1", "Ae2", "Ae3"); PEO_I <- c("Ep1", "Ep2", "Ep3", "Ep4", "Ep5")
GOV_I <- c("Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4"); BPO_I <- c("Co7", "Ae10", "Md7")
zscore <- function(x) (x - mean(x)) / sd(x)
cscore <- function(df, items) rowMeans(sapply(df[items], zscore))
RAW <- data.frame(SA = cscore(data, SA_I), PEO = cscore(data, PEO_I),
                  GOVMEAS = cscore(data, GOV_I), BPO = rowMeans(data[, BPO_I]))
COND <- c("SA", "PEO", "GOVMEAS"); N <- nrow(RAW)

# ------------------------------------------------------------ calibración
calib_all <- function(p_in = 95, p_cross = 50, p_ex = 5) {
  f <- function(x) { q <- quantile(x, c(p_ex, p_cross, p_in) / 100, names = FALSE)
                     calibrate(x, type = "fuzzy", thresholds = c(q[1], q[2], q[3])) }
  as.data.frame(lapply(RAW, f))
}
# ----------------------------------------------------- medidas de conjunto
incl <- function(m, y) sum(pmin(m, y)) / sum(m)
cov_ <- function(m, y) sum(pmin(m, y)) / sum(y)
pri  <- function(m, y) { a <- sum(pmin(m, y)); b <- sum(pmin(pmin(m, y), 1 - y)); d <- sum(m) - b
                         if (d > 1e-9) (a - b) / d else 0 }
sc   <- function(a, b) sum(pmin(a, b)) / sum(pmax(a, b))

term_m <- function(term, F) {               # membresía de un término "~SA*GOVMEAS"
  m <- rep(1, nrow(F))
  for (lit in strsplit(term, "*", fixed = TRUE)[[1]]) {
    lit <- trimws(lit); neg <- startsWith(lit, "~"); nm <- sub("^~", "", lit)
    if (!(nm %in% names(F))) { neg <- TRUE; nm <- toupper(nm) }   # notación en minúsculas
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
cat(strrep("=", 74), "\nSOLUCIÓN INICIAL (IS) — anclajes 95/50/5, incl ≥ 0.80, PRI ≥ 0.70, n ≥ 1\n", strrep("=", 74), "\n", sep = "")
IS <- solve_qca(); F0 <- IS$F; y0 <- F0$BPO
cat("  Necesidad (BPO alto):\n");  print(pof(F0[, COND], y0, relation = "necessity")$incl.cov)
cat("  Necesidad (~BPO):\n");      print(pof(1 - F0[, COND], 1 - y0, relation = "necessity")$incl.cov)
cat("\n  Tabla de verdad:\n");     print(IS$tt)
IS_m <- sol_m(IS$terms, F0)
cat(sprintf("\n  IS = %s\n", IS$expr))
for (t in IS$terms) {
  mt <- term_m(t, F0); others <- setdiff(IS$terms, t)
  covU <- if (length(others)) (sum(pmin(IS_m, y0)) - sum(pmin(sol_m(others, F0), y0))) / sum(y0) else cov_(mt, y0)
  cat(sprintf("    %-16s inclS=%.3f PRI=%.3f covS=%.3f covU=%.3f\n", t, incl(mt, y0), pri(mt, y0), cov_(mt, y0), covU))
}
cat(sprintf("  inclS=%.3f  PRI=%.3f  covS=%.3f\n", incl(IS_m, y0), pri(IS_m, y0), cov_(IS_m, y0)))

# ================================================ TEST 1 — RANGOS DE SENSIBILIDAD
cat("\n", strrep("=", 74), "\nTEST 1 — RANGOS DE SENSIBILIDAD (un parámetro a la vez)\n", strrep("=", 74), "\n", sep = "")
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
  cat(sprintf("  %-9s base=%-6s IS se mantiene en [%g, %g] de la rejilla [%g, %g]  (%.0f%% de la rejilla)\n",
              par, BASE[[par]], vals[lo], vals[hi], min(vals), max(vals), 100 * mean(same, na.rm = TRUE)))
  alt_tab <- sort(table(alts[!is.na(same) & !same]), decreasing = TRUE)
  for (k in seq_len(min(3, length(alt_tab)))) cat(sprintf("            alternativa: %s  (%d valores)\n", names(alt_tab)[k], alt_tab[k]))
}

# ===================================== TEST 2 — ROBUSTEZ ORIENTADA AL AJUSTE
cat("\n", strrep("=", 74), "\nTEST 2 — ROBUSTEZ ORIENTADA AL AJUSTE (conjunto de prueba TS)\n", strrep("=", 74), "\n", sep = "")
TS_GRID <- expand.grid(p_in = c(90, 92.5, 95, 97.5), p_cross = c(45, 50, 55), p_ex = c(2.5, 5, 7.5, 10),
                       incl_cut = c(0.75, 0.80, 0.85, 0.90), pri_cut = c(0.65, 0.70, 0.75), n_cut = c(1, 2))
TS <- vector("list", nrow(TS_GRID))
for (i in seq_len(nrow(TS_GRID))) TS[[i]] <- do.call(solve_qca, as.list(TS_GRID[i, ]))
exprs <- sapply(TS, `[[`, "expr")
cat(sprintf("  Soluciones plausibles evaluadas: %d\n  Distribución de soluciones en TS:\n", length(TS)))
dist <- sort(table(exprs), decreasing = TRUE)
for (k in seq_len(min(8, length(dist)))) cat(sprintf("    %4d (%2.0f%%)  %s\n", dist[k], 100 * dist[k] / length(TS), names(dist)[k]))
all_terms <- unlist(lapply(TS, `[[`, "terms"))
term_freq <- sort(table(all_terms), decreasing = TRUE) / length(TS)
cat("\n  Frecuencia de términos en TS:\n"); for (k in seq_along(term_freq)) cat(sprintf("    %-22s %.0f%%\n", names(term_freq)[k], 100 * term_freq[k]))
gov_all <- mean(sapply(TS, function(s) length(s$terms) > 0 && all(grepl("(^|\\*)GOVMEAS", s$terms))))
cat(sprintf("  GOVMEAS presente en TODOS los términos: %.0f%% de TS\n", 100 * gov_all))

# TS_min / TS_max evaluados en la calibración de IS, para que sean comparables
nonempty <- Filter(function(s) length(s$terms) > 0, TS)
Ms <- do.call(rbind, lapply(nonempty, function(s) sol_m(s$terms, F0)))
TSmin <- apply(Ms, 2, min); TSmax <- apply(Ms, 2, max)
cat(sprintf("\n  %-8s%8s%8s%10s\n", "", "inclS", "covS", "SC(IS,·)"))
cat(sprintf("  %-8s%8.3f%8.3f%10s\n", "IS", incl(IS_m, y0), cov_(IS_m, y0), "—"))
cat(sprintf("  %-8s%8.3f%8.3f%10.3f\n", "TS_min", incl(TSmin, y0), cov_(TSmin, y0), sc(IS_m, TSmin)))
cat(sprintf("  %-8s%8.3f%8.3f%10.3f\n", "TS_max", incl(TSmax, y0), cov_(TSmax, y0), sc(IS_m, TSmax)))
cat(sprintf("  RF_cons = inclS(IS) − inclS(TS_min) = %+.3f\n", incl(IS_m, y0) - incl(TSmin, y0)))
cat(sprintf("  RF_cov  = covS(IS) − covS(TS_min)   = %+.3f\n", cov_(IS_m, y0) - cov_(TSmin, y0)))

# ===================================== TEST 3 — ROBUSTEZ ORIENTADA A LOS CASOS
cat("\n", strrep("=", 74), "\nTEST 3 — ROBUSTEZ ORIENTADA A LOS CASOS\n", strrep("=", 74), "\n", sep = "")
typ_IS <- IS_m > 0.5 & y0 > 0.5; dev_IS <- IS_m > 0.5 & y0 <= 0.5
typ_min <- TSmin > 0.5 & y0 > 0.5; typ_max <- TSmax > 0.5 & y0 > 0.5; dev_min <- TSmin > 0.5 & y0 <= 0.5
RCR_typ <- sum(typ_IS & typ_min) / sum(typ_IS); RCR_dev <- if (sum(dev_IS)) sum(dev_IS & dev_min) / sum(dev_IS) else NA
cat(sprintf("  Casos típicos en IS: %d  → robustos %d, inestables %d, posibles (solo en TS_max) %d\n",
            sum(typ_IS), sum(typ_IS & typ_min), sum(typ_IS & !typ_min), sum(!typ_IS & typ_max)))
cat(sprintf("  Casos desviados (consistencia) en IS: %d  → robustos %d\n", sum(dev_IS), sum(dev_IS & dev_min)))
cat(sprintf("  RCR_typ = %.3f   RCR_dev = %.3f\n", RCR_typ, RCR_dev))
above <- colMeans(Ms > 0.5)
cat(sprintf("  Membresía en la solución: σ media entre TS = %.3f; casos con pertenencia >0.5 en ≥95%% de TS: %d; en ≤5%%: %d; ambiguos: %d\n",
            mean(apply(Ms, 2, sd)), sum(above >= 0.95), sum(above <= 0.05), sum(above > 0.05 & above < 0.95)))

# ============================================================ NÚCLEO ROBUSTO
cat("\n", strrep("=", 74), "\nNÚCLEO ROBUSTO\n", strrep("=", 74), "\n", sep = "")
tf_ne <- table(unlist(lapply(nonempty, `[[`, "terms")))
always <- names(tf_ne)[tf_ne == length(nonempty)]
gov_core <- all(sapply(nonempty, function(s) all(grepl("(^|\\*)GOVMEAS", s$terms))))
cat(sprintf("  Términos presentes en el 100 %% de TS no vacío: %s\n", if (length(always)) paste(always, collapse = ", ") else "ninguno literal"))
cat(sprintf("  GOVMEAS presente en todos los términos de todas las soluciones: %s\n", gov_core))
cat(sprintf("  Soluciones vacías en TS: %d de %d\n", length(TS) - length(nonempty), length(TS)))

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
           "fsqca_robustez_resultados_R.json", auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("\n→ fsqca_robustez_resultados_R.json\n")
