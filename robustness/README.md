# Robustness analysis / Análisis de robustez (Appendix C · Anexo C)

Three components, each implemented in **Python** (reference, executed) and
**R** (faithful replica, *not executed in the reference environment*), in
English and Spanish. Every script expects `final_dataset_plssem.csv` in the
working directory and writes a JSON file with its results.

Tres componentes, cada uno implementado en **Python** (referencia, ejecutado) y
**R** (réplica fiel, *no ejecutada en el entorno de referencia*), en inglés y
español. Todos los scripts esperan `final_dataset_plssem.csv` en el directorio
de trabajo y escriben un JSON con sus resultados.

| Component / Componente | English | Español | Output |
|---|---|---|---|
| 1 · Monte Carlo parameter recovery of the PLS-SEM (Paxton et al., 2001; Kock & Hadaya, 2018) | `robustness_montecarlo_pls.py` · `.R` | `robustez_montecarlo_pls.py` · `.R` | `montecarlo_pls_results.json` / `montecarlo_pls_resultados.json` |
| 2 · Specificity of the robust ANN protocol (H0a, H0b, H1) | `robustness_montecarlo_ann.py` · `.R` | `robustez_montecarlo_ann.py` · `.R` | `montecarlo_ann_results.json` / `montecarlo_ann_resultados.json` |
| 3 · Robustness test protocol for QCA (Oana & Schneider, 2024) | `fsqca_robustness.py` · `.R` | `robustez_fsqca.py` · `.R` | `fsqca_robustness_results.json` / `fsqca_robustez_resultados.json` |

## Reference run / Ejecución de referencia

`results/` holds the JSON outputs and the console logs of the Python scripts
(English versions; the Spanish ones produce the same numbers, with the same
seeds). The tables of Appendix C (C1–C9) and Table 6 and Figure 9 of the
article are taken from these files. Approximate runtimes: fsQCA < 1 min,
ANN ≈ 3 min, PLS-SEM Monte Carlo ≈ 30 min (1,000 replications × 7 sample sizes,
plus 150 × 250 bootstrap resamples for coverage).

`results/` contiene las salidas JSON y los registros de consola de los scripts
de Python (versiones en inglés; las españolas producen las mismas cifras, con
las mismas semillas). Las tablas del Anexo C (C1–C9), la Tabla 6 y la Figura 9
del artículo proceden de estos archivos.

## Design in one paragraph / Diseño en un párrafo

- **PLS-SEM.** The data-generating process is fixed at the observed mean vector
  and covariance matrix of the six stage-2 indicators; for each
  n ∈ {30, 40, 56, 80, 100, 150, 200} 1,000 multivariate-normal samples are
  drawn and the Mode B model is re-estimated with the same algorithm as the
  reproduction scripts. Reported: bias, RMSE and CI width of β; P(GOVMEAS
  dominant weight); power for β ≥ 0.35; P(PEO weight significant); bootstrap
  coverage of β and of the three weights (150 replications × 250 resamples);
  adverse scenarios with β_pop = 0.70, 0.50, 0.35.
- **ANN.** 120 replications of n = 56 under H0a (equal bivariate correlations
  with BPO — diagnostic only), H0b (equal conditional contribution: identical
  regression coefficients with R² preserved — the correct null of equal
  contribution) and H1 (real DGP). In every replication the single-run network
  (10 neurons, no regularisation) is compared with the robust protocol
  (10 networks × 10 folds, 3 neurons, decay 0.1, L-BFGS, logistic activation).
- **fsQCA.** Test 1: sensitivity ranges for p_in, p_cross, p_ex, the raw
  consistency cut-off, the PRI cut-off and the frequency cut-off. Test 2:
  fit-oriented robustness over a test set of 1,152 plausible specifications
  (p_in ∈ {90, 92.5, 95, 97.5}; p_cross ∈ {45, 50, 55}; p_ex ∈ {2.5, 5, 7.5, 10};
  incl ∈ {0.75, 0.80, 0.85, 0.90}; PRI ∈ {0.65, 0.70, 0.75}; n ∈ {1, 2}), with
  TS_min, TS_max, set coincidence, RF_cons, RF_cov and term frequencies.
  Test 3: case-oriented robustness (RCR_typ, RCR_dev, membership stability).

Simulated data are generated inside each script and are **never** part of the
study dataset nor of the substantive inference (Decruyenaere et al., 2024).

## R replicas / Réplicas en R

Requirements: `install.packages(c("MASS", "nnet", "NeuralNetTools", "caret",
"QCA", "jsonlite"))`. The R scripts reproduce the designs and grids of the
Python ones; the fsQCA component should match exactly (deterministic), the
Monte Carlo components will differ slightly because of the pseudo-random
generators. Outputs are suffixed `_R.json`.
