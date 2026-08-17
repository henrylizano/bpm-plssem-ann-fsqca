# Tri-Methodological BPM: PLS-SEM + ANN + fsQCA

A reproducible, fully open-source implementation of the tri-methodological
analysis of a hierarchical **Business Process Management (BPM)** model: partial
least squares structural equation modeling (**PLS-SEM**), artificial neural
networks (**ANN**) and fuzzy-set qualitative comparative analysis (**fsQCA**).

> 🇪🇸 **Versión en español:** [`LEAME.md`](LEAME.md)

**Author:** Henry Lizano-Mora
**Institutions:** Costa Rica Institute of Technology / University of Seville

---

## 1. Repository contents

| File | Language | Locale | Description |
|---|---|---|---|
| `reproduce_article.py` | Python ≥ 3.9 | English | **Reference implementation.** Codes PLS-SEM, ANN and fsQCA **from scratch** (no specialised PLS/QCA dependencies) across 7 modules, replicating the canonical R configurations. Writes figures to `figs_en/`. |
| `reproducir_articulo.py` | Python ≥ 3.9 | Spanish | Spanish-language version of the same 7-module architecture. Writes figures to `figs/`. Modules 3 and 5 still use the earlier simplified variants (blindfolding Q² and piecewise-linear calibration); see the note at the end of this section. |
| `reproducir_articulo.R` | R ≥ 4.0 | Spanish | Independent replication using the field's canonical packages (`seminr`, `QCA`, `nnet`). Serves as **cross-validation** of the Python implementation. Figures in `figs/`. |
| `reproduce_article_en.R` | R ≥ 4.0 | English | Translation of the above. Figures in `figs_en/`. |
| `requirements.txt` | — | — | Python dependencies with pinned versions, transitive ones included, for bit-for-bit reproducibility (pip route). |
| `environment.yml` | — | — | Equivalent conda environment: Python 3.9.6 and the six pinned scientific packages, from `conda-forge`. |
| `renv.lock` | — | — | R lockfile generated with `renv::snapshot()`: R 4.6.1 and **120 packages** (the 7 direct dependencies plus all recursive ones) pinned to exact versions. |
| `final_dataset_plssem.csv` | — | — | Analytical dataset: **n = 56** valid observations, 20 Likert-scale indicators. No personal identifiers. |
| `references/` | — | — | The article's bibliography (63 entries) in three formats: **BibTeX**, **BibLaTeX** and **Zotero RDF**. See [`references/README.md`](references/README.md). |
| `LICENSE` | — | — | MIT License (see §7). |
| `CITATION.cff` | — | — | Machine-readable citation metadata (GitHub, Zenodo, reference managers). |

The two implementations (Python and R) are **independent**: the Python one codes
the algorithms explicitly, whereas the R one delegates to established packages.
Their substantive convergence is itself evidence of the robustness of the
findings; marginal third-decimal differences may arise from differing seeds and
pseudo-random number generators.

> **Note on `reproducir_articulo.py`.** `reproduce_article.py` incorporates three
> methodological improvements the Spanish version does not yet carry: full
> PLSpredict with a linear-model benchmark (module 3), the L-BFGS optimizer with
> logistic activation in the ANN (module 4), and Ragin's logistic calibration
> with a conservative solution and sensitivity analysis (module 5). To reproduce
> the published results, use the English version.

---

## 2. Running the scripts

All scripts expect `final_dataset_plssem.csv` **in the same directory**; run
them from the repository root.

### Python — pip (bit-exact)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproduce_article.py        # or: python reproducir_articulo.py
```

### Python — conda

```bash
conda env create -f environment.yml
conda activate bpm-plssem-ann-fsqca
python reproduce_article.py
```

### R — renv (recommended, exact versions)

```r
install.packages("renv")
renv::restore()                    # installs the 120 packages in renv.lock
source("reproduce_article_en.R")   # or: source("reproducir_articulo.R")
```

### R — manual installation

```r
install.packages(c("seminr", "QCA", "nnet", "NeuralNetTools",
                   "caret", "ggplot2", "reshape2"))
source("reproduce_article_en.R")
```

Approximate runtime on a current laptop: **2–5 minutes** in Python (dominated by
the 3,000-resample bootstrap, the 100 PLSpredict folds and the 130 neural
networks), somewhat longer in R.

---

## 3. Measurement model and data

The model is a **hierarchical component model (HCM)** estimated in two stages.
The 20 indicators in the CSV are grouped as follows:

| Construct | Role | Indicators |
|---|---|---|
| **SA** — Strategic alignment | Reflective (first-order dimension) | `Ae1`, `Ae2`, `Ae3` |
| **PEO** — Process orientation | Reflective (first-order dimension) | `Ep1`–`Ep5` |
| **GOVMEAS** — Governance and measurement | Reflective (first-order dimension) | `Co1`–`Co5`, `Co8`, `Md1`, `Md3`, `Md4` |
| **BPM_Capability** | **Formative** (second-order construct) | Scores of SA, PEO and GOVMEAS |
| **BPO** — Process performance | Composite (dependent variable) | `Co7`, `Ae10`, `Md7` |

Each reflective dimension's score is the mean of its standardized indicators
(*z*-scores); this is the **two-stage construct-score** approach customary in
HCM.

---

## 4. The three algorithms

Every script is organised into the same seven modules, so the Python ↔ R
correspondence is line by line.

```
MODULE 0  Data loading and score construction (Stage 1)
MODULE 1  Measurement model: reliability, AVE, HTMT
MODULE 2  PLS-SEM Stage 2: formative + structural model + bootstrap
MODULE 3  Predictive assessment (PLSpredict: PLS vs. linear benchmark)
MODULE 4  Artificial neural networks (diagnostic + robust)
MODULE 5  fsQCA: logistic calibration + necessity/RoN + conservative solution
MODULE 6  Figure generation
```

### 4.1 PLS-SEM (Modules 1–3)

**What it answers:** to what extent does BPM capability explain and predict
process performance, and how much does each dimension contribute? This is the
**symmetric, linear** part of the design: it estimates a net average effect.

**Measurement model (Module 1).** For each reflective construct the script
computes:

- **Cronbach's alpha**, in its classical form
  `α = k/(k−1) · (1 − Σ var(xᵢ)/var(Σxᵢ))`.
- **Outer loadings** λᵢ as the correlation of each indicator with the construct
  score; reference threshold λ ≥ 0.708.
- **Composite reliability (CR)** = `(Σλ)² / [(Σλ)² + Σ(1 − λᵢ²)]`.
- **Average variance extracted (AVE)** = `mean(λᵢ²)`; criterion AVE ≥ 0.50.
- **HTMT** (Henseler, Ringle & Sarstedt, 2015): the ratio of the mean
  heterotrait–heteromethod correlations to the geometric mean of each block's
  monotrait–heteromethod correlations. Discriminant validity holds if
  HTMT < 0.90. In Python the matrix is computed explicitly (`htmt_matrix`); in R
  it comes from `seminr::estimate_pls`.

**Structural model (Module 2).** The second-order construct `BPM_Capability` is
specified as a **formative composite (Mode B)** over the three standardized
dimensions, and `BPO` likewise as a composite over its three indicators. The
`pls_formative()` function implements the classical PLS algorithm with the
**path weighting scheme**:

1. Initialise the outer weights `wx`, `wy` uniformly and normalise them.
2. **Outer approximation:** compute the latent scores `X = X_ind·wx`,
   `Y = Y_ind·wy` and standardize them.
3. **Inner approximation:** weight each construct by its neighbour in the
   structural network using the correlation coefficient `b` between `X` and `Y`.
4. **Weight update (Mode B):** OLS regression of each inner construct on its own
   indicators (`np.linalg.lstsq`) — unlike Mode A, which would use simple
   correlations.
5. Rescale the weights for unit score variance and repeat from (2) until the sum
   of absolute changes falls below 10⁻⁷ (max. 300 iterations).

From the converged scores the script obtains the path coefficient β (the
correlation between `X` and `Y`), **R²** = β² and **adjusted R²**. It
additionally reports:

- **Formative loadings**: the correlation of each indicator with the construct
  score, for interpreting absolute contribution.
- **VIF** for the formative indicators (`vif_scores`), computed as `1/(1 − R²ᵢ)`
  from the regression of each indicator on the others; VIF < 5 is sought to rule
  out collinearity, a requirement specific to formative models.
- **Non-parametric bootstrap** with **B = 3,000** resamples with replacement.
  Each replication re-estimates the full model and corrects the **sign
  indeterminacy** typical of PLS (if β < 0, β and the weights are flipped
  together). Standard errors, *t*-statistics, *p*-values and 95% percentile
  confidence intervals are derived from the bootstrap distribution.

**Predictive assessment (Module 3).** This module implements **PLSpredict**
(Shmueli *et al.*, 2016, 2019) as methodologically defined: **10-fold ×
10-repetition** out-of-sample cross-validation. Within each fold:

1. Data are standardized **using the training subset's statistics** (never the
   full sample's), to avoid data leakage.
2. The PLS model is estimated on the training data only, correcting sign
   indeterminacy, and the latent score of the holdout observations is projected.
3. The endogenous indicators are predicted as `β · score · loading` and the
   prediction errors are accumulated.
4. In parallel, the **linear benchmark (LM)** is fitted: an OLS regression of
   each endogenous indicator on the three exogenous indicators.

**RMSE** and **MAE** are reported for both models per indicator, along with
**Q²predict** (which uses the training mean as the naive benchmark). The
criterion of Shmueli *et al.* (2019) classifies predictive power by how many
indicators the PLS model beats the LM on: 3/3 high, 2/3 medium, ≤ 1/3 low. The
script prints that verdict automatically rather than assuming it. The R
implementation uses `seminr::predict_pls` with the same parameters.

### 4.2 Artificial neural networks (Module 4)

**What it answers:** are there **non-linear** relationships between the BPM
dimensions and performance that PLS-SEM, being linear by construction, would
miss? The ANNs are used here as a non-linear complement, not a replacement.

The three scores (SA, PEO, GOVMEAS) and the target variable BPO are min–max
normalised to `[0, 1]`. Each predictor's relative importance is obtained with
**Garson's algorithm (1991)**: the absolute values of the input→hidden and
hidden→output weights are multiplied per hidden neuron, summed per predictor and
expressed as a percentage of the total (`garson_importance`; in R,
`NeuralNetTools::garson`).

The module is deliberately split in two, and that split is itself a
methodological finding of the article:

**(a) Diagnostic — why a single network is not enough.** **30 independent
networks** are trained (`MLPRegressor`, one hidden layer of 10 neurons, `tanh`
activation, `adam`, 2,000 iterations) differing **only in the initialization
seed**. The script tabulates which predictor comes out "dominant" for each seed
and the range (max − min) of importance per construct. The result: dominance
changes with the seed — **a single run is not reliable, it is an initialization
artefact**. Figure 4 documents this instability.

**(b) Robust protocol.** **10 networks × 10-fold cross-validation = 100 models**
are trained, with three key changes relative to the diagnostic:

- Parsimonious architecture: hidden layer reduced to **3 neurons**.
- **L2 regularization**: `alpha = 0.1` in scikit-learn, `decay = 0.1` in `nnet`.
- **Exact replication of the R configuration**: the **L-BFGS** optimizer and
  **logistic** (sigmoid) activation, which is what `nnet` uses internally
  (BFGS + sigmoid), instead of the `adam` + `tanh` of the diagnostic module.
  This guarantees cross-platform numerical equivalence of the finding.

Importances are averaged within each network and then across networks, reporting
mean and standard deviation. Overfitting is controlled by comparing **training
RMSE against test RMSE**: the difference is practically nil (+0.004 in the
reference run). Under this configuration the cross-network standard deviation
drops to **≤ 0.2 percentage points**, against the tens-of-points ranges of the
diagnostic: the importances become **stable and reportable**, and are displayed
with error bars in Figure 5.

### 4.3 fsQCA (Module 5)

**What it answers:** which **combinations** of conditions are sufficient for high
performance? Unlike PLS-SEM, fsQCA is **asymmetric and configurational**: it
admits equifinality (several paths to the same outcome) and asymmetric causality
(the causes of high performance are not the mirror image of those of low
performance).

**Calibration.** The continuous scores are transformed into fuzzy memberships in
`[0, 1]` via **Ragin's direct logistic calibration**, with percentile anchors:
full membership `P95`, crossover point `P50` and full non-membership `P05`. The
`calibrate()` function computes log-odds scaled so that they equal
`±log(0.95/0.05)` at the extreme thresholds, then applies the sigmoid
`1/(1+e^−logodds)`. It is **numerically equivalent** to R's
`QCA::calibrate(type = "fuzzy", logistic = TRUE)`, so both implementations
produce the same memberships.

**Necessity analysis.** For each condition the script computes

- **Consistency (inclN)** `= Σ min(condition, outcome) / Σ outcome`
- **Coverage (covN)** `= Σ min(condition, outcome) / Σ condition`
- **RoN** (*Relevance of Necessity*, Schneider & Wagemann, 2012)
  `= Σ(1 − condition) / Σ(1 − min(condition, outcome))`

RoN is essential: a condition can reach high consistency simply by being
**trivially ubiquitous** (nearly every case belongs to it), and RoN detects that
case by returning a low value. Without it, an irrelevant condition can look
necessary. The analysis is run **twice**, for high performance (BPO) and for its
negation (~BPO = 1 − BPO), precisely to test causal asymmetry.

**Sufficiency analysis.** The complete **truth table** is built by traversing the
2³ = 8 corners of the property space. A case's membership in a corner is the
minimum (fuzzy AND) of its conditions, negated via `1 − x` where applicable. For
each configuration the script computes:

- **Sufficiency consistency** `= Σ min(m, BPO) / Σ m`, threshold ≥ **0.80**.
- **PRI** (*Proportional Reduction in Inconsistency*), which penalises
  configurations simultaneously consistent with the outcome and its negation:
  `PRI = [Σ min(m,Y) − Σ min(min(m,Y), min(m,~Y))] / [Σ m − Σ min(min(m,Y), min(m,~Y))]`,
  threshold ≥ **0.70**. This is the filter that discards spurious solutions.
- **Number of cases** with membership > 0.5. Rows with `n = 0` are explicitly
  flagged as **logical remainders** (`OUT = ?`): configurations without empirical
  evidence.

**Conservative minimization.** The `minimize_conservative()` function applies the
**Quine-McCluskey** algorithm to the **observed** rows with `OUT = 1` only,
iteratively combining terms that differ in a single condition and replacing that
condition with a *don't care*; it then selects the essential prime implicants and
greedily completes the coverage. Logical remainders are **not** incorporated as
simplifying assumptions, which yields the **conservative (complex) solution** —
the most demanding of fsQCA's three solutions, because it assumes nothing about
configurations that were never observed. For each term the script reports
consistency, PRI, **raw coverage** and **unique coverage** (the portion of the
outcome explained by that term alone); for the overall solution, consistency, PRI
and coverage. The R equivalent is `QCA::truthTable` + `QCA::minimize`.

**Sensitivity analysis.** The module repeats the entire procedure with
alternative **90/50/10** anchors instead of 95/50/5 and contrasts the two
solutions. That the solution structure and the centrality of GOVMEAS hold under
both schemes is the evidence that the result is not an artefact of calibration
choices.

### 4.4 Tri-methodological synthesis (Module 6)

The final radar chart overlays the three importance profiles — PLS-SEM formative
weights, ANN Garson importance and fsQCA necessity consistency — each series
normalised by its own maximum so they are comparable in shape rather than in
scale. The convergence of the three methods on the same dimension is the
article's central argument.

On completion, the script prints a summary block with the key indicators of all
three methods: β, R² and *t* from PLS-SEM; the PLS-vs-LM tally from PLSpredict;
the mean importance and maximum cross-network standard deviation from the ANN;
and the expression of the fsQCA conservative solution with its consistency, PRI
and coverage. Every value is interpolated from the current run, none is
hard-coded.

---

## 5. Generated figures

The Spanish scripts write to `figs/` and the English ones to `figs_en/` (both
directories are created automatically and are excluded from version control):

| Figure | Content | Python | R |
|---|---|:---:|:---:|
| `fig2_cargas` / `fig2_loadings` | Outer loadings by construct, with the 0.708 threshold | ✓ | ✓ |
| `fig3_htmt` | HTMT matrix (triangular heatmap) | ✓ | — |
| `fig4_ann_inestable` / `fig4_ann_unstable` | Importance instability across 30 single-run networks | ✓ | — |
| `fig5_ann_robusto` / `fig5_ann_robust` | Robust importance (10 networks × 10 folds) with error bars | ✓ | ✓ |
| `fig7_fsqca_xy` | XY sufficiency plot for the core configuration | ✓ | ✓ |
| `fig8_radar` / `fig9_radar` | Tri-methodological synthesis radar | ✓ | ✓ |

---

## 6. Reproducibility

- Fixed global seed: `np.random.seed(42)` / `set.seed(42)`.
- Deterministic per-network seeds: `0…29` in the diagnostic and `100+net` in the
  robust protocol.
- Bootstrap with a fixed seed (`seed = 42` in `seminr::bootstrap_model`).
- Python dependencies pinned to exact versions in `requirements.txt` (pip) and
  `environment.yml` (conda), transitive ones included in the pip route.
- R dependencies pinned in `renv.lock` (R 4.6.1, 120 packages including all
  recursive dependencies), restorable with `renv::restore()`.

Consequently, every run returns the same numbers. Differences between the Python
and R implementations are confined to the third decimal place and stem from
distinct pseudo-random number generators and optimization routines.

---

## 7. License

This project is distributed under the **MIT License** (see [`LICENSE`](LICENSE)).

**Why MIT rather than GPL or Apache-2.0.** The goal is maximum scientific
dissemination: MIT is the shortest and most widely recognised permissive
licence, it imposes no conditions on those who reuse the code (neither copyleft
nor extensive notice obligations), it is accepted without friction by academic
repositories such as Zenodo and figShare and by the code-availability policies of
the major publishers, and it is **GPL-compatible**, so other researchers can
incorporate these routines into GPL projects as well as into proprietary or
commercial work. The GPL would have closed off that second avenue, and
Apache-2.0 — although it adds an express patent grant — introduces notice
requirements and is incompatible with GPL-2, which would complicate reuse
alongside R packages under that licence.

**Note on dependencies.** The code original to this repository is MIT. The R
packages it calls (`seminr`, `QCA`, `nnet`, `caret`, among others) are
distributed under the GPL and retain their own licences; anyone redistributing a
combined work with them must respect the GPL's terms. The Python dependencies
(NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn) are all permissive
(BSD/MIT).

---

## 8. Citation

If you use this code or the data, please cite the associated article and this
repository. The [`CITATION.cff`](CITATION.cff) file holds the metadata in
machine-readable form; GitHub automatically renders a *Cite this repository*
button from it.

---

## 9. Methodological references

- Garson, G. D. (1991). Interpreting neural-network connection weights.
  *AI Expert*, 6(4), 46–51.
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022).
  *A Primer on Partial Least Squares Structural Equation Modeling (PLS-SEM)*
  (3rd ed.). Sage.
- Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
  assessing discriminant validity in variance-based structural equation
  modeling. *Journal of the Academy of Marketing Science*, 43(1), 115–135.
- Ragin, C. C. (2008). *Redesigning Social Inquiry: Fuzzy Sets and Beyond*.
  University of Chicago Press.
- Schneider, C. Q., & Wagemann, C. (2012). *Set-Theoretic Methods for the Social
  Sciences: A Guide to Qualitative Comparative Analysis*. Cambridge University
  Press.
- Sarstedt, M., Hair, J. F., Cheah, J.-H., Becker, J.-M., & Ringle, C. M. (2019).
  How to specify, estimate, and validate higher-order constructs in PLS-SEM.
  *Australasian Marketing Journal*, 27(3), 197–211.
- Shmueli, G., Ray, S., Velasquez Estrada, J. M., & Chatla, S. B. (2016).
  The elephant in the room: Predictive performance of PLS models.
  *Journal of Business Research*, 69(10), 4552–4564.
- Shmueli, G., Sarstedt, M., Hair, J. F., Cheah, J.-H., Ting, H., Vaithilingam, S.,
  & Ringle, C. M. (2019). Predictive model assessment in PLS-SEM: Guidelines for
  using PLSpredict. *European Journal of Marketing*, 53(11), 2322–2347.
