#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ROBUSTNESS ANALYSIS (1/3) — MONTE CARLO PARAMETER-RECOVERY STUDY
Tri-methodological BPM article (PLS-SEM + ANN + fsQCA) — Appendix C
Author: Henry Lizano-Mora · MIT License
==============================================================================
Logic:
  • The data-generating process (DGP) is not invented: the mean vector and
    covariance matrix of the six stage-2 indicators (SA, PEO, GOVMEAS scores
    and items Co7, Ae10, Md7) are taken exactly as observed in the real sample
    (n = 56). Multivariate-normal replications of size n are drawn from that
    population and the article's PLS Mode B estimator is re-fitted.
  • For n ∈ {30, 40, 56, 80, 100, 150, 200} the script reports: bias, RMSE and
    SD of β; bias of the formative weights; 95 % bootstrap CI coverage; power
    to detect β ≥ 0.35; probability that GOVMEAS emerges as the dominant
    weight; probability that the PEO weight is significant.
  • Adverse scenario: the BPC↔BPO cross-covariance is rescaled to fix the
    population β at 0.70, 0.50 and 0.35, and detection is assessed at n = 56.
Simulated data characterise the estimator; they never enter the substantive
inference (Decruyenaere et al., 2024).
Usage:  python robustness_montecarlo_pls.py   (requires final_dataset_plssem.csv)
Output: montecarlo_pls_results.json
==============================================================================
"""
import numpy as np, pandas as pd, json, time, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

t0 = time.time()

# =============================================================================
# 0) Real sample → stage-1 scores → stage-2 matrix
# =============================================================================
data = pd.read_csv("final_dataset_plssem.csv")
SA_ITEMS  = ["Ae1", "Ae2", "Ae3"]
PEO_ITEMS = ["Ep1", "Ep2", "Ep3", "Ep4", "Ep5"]
GOV_ITEMS = ["Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4"]
BPO_ITEMS = ["Co7", "Ae10", "Md7"]

def zscore(s): return (s - s.mean()) / s.std(ddof=1)
def cscore(df, items): return df[items].apply(zscore).mean(axis=1).values

N = len(data)
BPC = np.column_stack([cscore(data, SA_ITEMS), cscore(data, PEO_ITEMS),
                       cscore(data, GOV_ITEMS)])
BPO = data[BPO_ITEMS].values.astype(float)
Z6 = np.column_stack([BPC, BPO])          # 56 × 6
MU, SIGMA = Z6.mean(0), np.cov(Z6, rowvar=False, ddof=1)

# =============================================================================
# 1) PLS Mode B estimator (identical to the reproduction script)
# =============================================================================
def pls_formative(X_ind, Y_ind, max_iter=300, tol=1e-7):
    wx = np.ones(X_ind.shape[1]) / np.sqrt(X_ind.shape[1])
    wy = np.ones(Y_ind.shape[1]) / np.sqrt(Y_ind.shape[1])
    for _ in range(max_iter):
        X = X_ind @ wx; Y = Y_ind @ wy
        X = (X - X.mean()) / X.std(ddof=1); Y = (Y - Y.mean()) / Y.std(ddof=1)
        b = np.corrcoef(X, Y)[0, 1]
        wx_new = np.linalg.lstsq(X_ind, b * Y, rcond=None)[0]
        wy_new = np.linalg.lstsq(Y_ind, b * X, rcond=None)[0]
        wx_new /= np.linalg.norm(X_ind @ wx_new) / np.sqrt(len(X))
        wy_new /= np.linalg.norm(Y_ind @ wy_new) / np.sqrt(len(Y))
        done = np.abs(wx_new - wx).sum() + np.abs(wy_new - wy).sum() < tol
        wx, wy = wx_new, wy_new
        if done: break
    X = X_ind @ wx; Y = Y_ind @ wy
    X = (X - X.mean()) / X.std(ddof=1); Y = (Y - Y.mean()) / Y.std(ddof=1)
    return wx, wy, np.corrcoef(X, Y)[0, 1], X, Y

def zmat(M): return (M - M.mean(0)) / M.std(0, ddof=1)

def fit_pls(Z):
    """Returns β, normalised BPC weights, BPC loadings, BPO weights, scores."""
    Xz, Yz = zmat(Z[:, :3]), zmat(Z[:, 3:])
    wx, wy, beta, Xs, Ys = pls_formative(Xz, Yz)
    sgn = np.sign(np.corrcoef(Xs, Z[:, :3].sum(1))[0, 1]) or 1.0
    wx_n = wx / np.linalg.norm(wx) * sgn
    wy_n = wy / np.linalg.norm(wy) * sgn
    load = np.array([np.corrcoef(Xz[:, i], Xs * sgn)[0, 1] for i in range(3)])
    return abs(beta), wx_n, load, wy_n, Xs * sgn, Ys * sgn

# --- "population" values = estimates on the real sample -----------------
beta_pop, w_pop, l_pop, wy_pop, Xs_real, Ys_real = fit_pls(Z6)
print("=" * 72)
print("DGP fixed at the real sample (n = 56)")
print("=" * 72)
print(f"  β_pop = {beta_pop:.4f} | BPC weights = {np.round(w_pop,3)} | "
      f"loadings = {np.round(l_pop,3)} | BPO weights = {np.round(wy_pop,3)}")

def boot_ci(Z, B=500, rng=None):
    """Percentile bootstrap CI for β and BPC weights."""
    n = len(Z); bs_b, bs_w = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            b, w, *_ = fit_pls(Z[idx]); bs_b.append(b); bs_w.append(w)
        except Exception:
            continue
    bs_b, bs_w = np.array(bs_b), np.array(bs_w)
    return (np.percentile(bs_b, [2.5, 97.5]),
            np.percentile(bs_w, [2.5, 97.5], axis=0))

# =============================================================================
# 2) Parameter recovery as a function of n
# =============================================================================
R_MAIN = 1000          # replications per size (no bootstrap)
R_COV  = 150           # replications with bootstrap for coverage (costly)
B_BOOT = 250
SIZES  = [30, 40, 56, 80, 100, 150, 200]

results = {}
print("\n" + "=" * 72)
print(f"MONTE CARLO PARAMETER RECOVERY ({R_MAIN} replications per n)")
print("=" * 72)
hdr = (f"{'n':>4} {'β̄':>7} {'bias':>7} {'RMSE':>7} {'CI95% width':>12} "
       f"{'P(GOV dom.)':>12} {'P(β>0.35 sig)':>14} {'P(w_PEO sig)':>13}")
print(hdr)
for n in SIZES:
    rng = np.random.default_rng(1000 + n)
    betas, ws, ls = [], [], []
    for r in range(R_MAIN):
        Z = rng.multivariate_normal(MU, SIGMA, size=n)
        try:
            b, w, l, *_ = fit_pls(Z)
        except Exception:
            continue
        betas.append(b); ws.append(w); ls.append(l)
    betas, ws, ls = np.array(betas), np.array(ws), np.array(ls)
    gov_dom = np.mean(np.argmax(ws, axis=1) == 2)
    # coverage and power with bootstrap on a subset of replications
    cov_b, cov_w, sig_b, sig_peo, widths = 0, np.zeros(3), 0, 0, []
    rngc = np.random.default_rng(5000 + n)
    for r in range(R_COV):
        Z = rng.multivariate_normal(MU, SIGMA, size=n)
        try:
            b, w, *_ = fit_pls(Z)
            cib, ciw = boot_ci(Z, B=B_BOOT, rng=rngc)
        except Exception:
            continue
        cov_b += (cib[0] <= beta_pop <= cib[1])
        cov_w += ((ciw[0] <= w_pop) & (w_pop <= ciw[1]))
        sig_b += (cib[0] > 0.35)            # β detectable as ≥ 0.35
        sig_peo += (ciw[0, 1] > 0)           # PEO weight significant
        widths.append(cib[1] - cib[0])
    results[n] = dict(
        beta_mean=float(betas.mean()), beta_sd=float(betas.std(ddof=1)),
        bias=float(betas.mean() - beta_pop),
        rmse=float(np.sqrt(np.mean((betas - beta_pop) ** 2))),
        w_mean=ws.mean(0).tolist(), w_sd=ws.std(0, ddof=1).tolist(),
        w_bias=(ws.mean(0) - w_pop).tolist(),
        load_mean=ls.mean(0).tolist(),
        p_gov_dominant=float(gov_dom),
        coverage_beta=cov_b / R_COV, coverage_w=(cov_w / R_COV).tolist(),
        power_beta_035=sig_b / R_COV, power_w_peo=sig_peo / R_COV,
        ci_width_mean=float(np.mean(widths)))
    print(f"{n:>4} {betas.mean():>7.3f} {betas.mean()-beta_pop:>+7.3f} "
          f"{results[n]['rmse']:>7.3f} {np.mean(widths):>12.3f} "
          f"{gov_dom:>12.3f} {sig_b/R_COV:>14.3f} {sig_peo/R_COV:>13.3f}", flush=True)
    json.dump(results, open('montecarlo_pls_results.json','w'), indent=2)

print("\n  Bootstrap 95% CI coverage (nominal 0.95):")
print(f"  {'n':>4} {'β':>7} {'w_SA':>7} {'w_PEO':>7} {'w_GOV':>7}")
for n in SIZES:
    r = results[n]
    print(f"  {n:>4} {r['coverage_beta']:>7.3f} "
          + " ".join(f"{c:>7.3f}" for c in r['coverage_w']))

print("\n  Bias of the formative weights (mean − population):")
print(f"  {'n':>4} {'w_SA':>8} {'w_PEO':>8} {'w_GOV':>8}   σ(w_SA) σ(w_PEO) σ(w_GOV)")
for n in SIZES:
    r = results[n]
    print(f"  {n:>4} " + " ".join(f"{b:>+8.3f}" for b in r['w_bias'])
          + "   " + " ".join(f"{s:>7.3f}" for s in r['w_sd']))

# --- Adverse scenario: weaker population effect ------------------------
print("\n  Adverse scenario: artificially reduced β_pop (n = 56)")
print(f"  {'β_pop':>6} {'β̄':>7} {'bias':>7} {'P(CI excl. 0)':>14} {'P(GOV dom.)':>12}")
for target in [0.934, 0.70, 0.50, 0.35]:
    # alternative DGP: rescale the BPC↔BPO cross-covariance
    scale = target / beta_pop
    S = SIGMA.copy()
    S[:3, 3:] *= scale; S[3:, :3] *= scale
    # ensure positive definiteness
    ev = np.linalg.eigvalsh(S)
    if ev.min() <= 0: S += np.eye(6) * (abs(ev.min()) + 1e-3)
    rng = np.random.default_rng(7000 + int(target * 100))
    bs, gd, excl = [], [], 0
    for r in range(300):
        Z = rng.multivariate_normal(MU, S, size=56)
        b, w, *_ = fit_pls(Z); bs.append(b); gd.append(np.argmax(w) == 2)
        if r < 100:
            cib, _ = boot_ci(Z, B=250, rng=rng); excl += cib[0] > 0
    b_true, *_ = fit_pls(rng.multivariate_normal(MU, S, size=200000))
    print(f"  {b_true:>6.3f} {np.mean(bs):>7.3f} {np.mean(bs)-b_true:>+7.3f} "
          f"{excl/100:>14.3f} {np.mean(gd):>12.3f}")
    results[f"adverse_{target}"] = dict(beta_true=float(b_true),
        beta_mean=float(np.mean(bs)), p_excl0=excl/100, p_gov=float(np.mean(gd)))

json.dump(results, open("montecarlo_pls_results.json", "w"), indent=2)
print(f"\nTotal time: {(time.time()-t0)/60:.1f} min → montecarlo_pls_results.json")
