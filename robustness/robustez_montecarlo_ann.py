#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ANÁLISIS DE ROBUSTEZ (2/3) — ESPECIFICIDAD DEL PROTOCOLO ANN ROBUSTO
Artículo tri-metodológico BPM (PLS-SEM + ANN + fsQCA) — Anexo C
Autor: Henry Lizano-Mora · Licencia MIT
==============================================================================
Pregunta: ¿el protocolo robusto (10 redes × 10 pliegues, 3 neuronas ocultas,
decaimiento L2 = 0.1, activación logística, optimizador L-BFGS) devuelve una
importancia Garson de ~33/33/33 cuando las tres dimensiones contribuyen por
igual, y recupera la jerarquía observada cuando el DGP es el real?

Diseño (n = 56 en todas las réplicas; preprocesamiento idéntico al artículo:
scores de etapa 1 y media de ítems BPO reescalados Min-Max):
  • CONTROL: protocolo robusto sobre la muestra real (debe reproducir las
    cifras del Anexo A: 30.0 / 29.0 / 41.0).
  • H0a: correlaciones bivariadas de SA, PEO y GOVMEAS con BPO igualadas a su
    media. Se reporta como diagnóstico: por la mayor colinealidad de GOVMEAS,
    esta hipótesis implica una contribución CONDICIONAL menor de GOVMEAS.
  • H0b: contribución condicional igual (coeficientes de regresión β_j
    idénticos, R² preservado). Es la hipótesis nula correcta de
    equicontribución.
  • H1: DGP real (vector de medias y covarianzas observados).
En cada réplica se compara la red de ejecución única (10 neuronas, sin
regularización) con el protocolo robusto.
Uso:  python robustez_montecarlo_ann.py   (requiere final_dataset_plssem.csv)
Salida: montecarlo_ann_resultados.json
==============================================================================
"""
import numpy as np, pandas as pd, json, time, warnings, functools
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)
t0 = time.time()

# ------------------------------------------------------------------ datos
data = pd.read_csv("final_dataset_plssem.csv")
SA_I = ["Ae1", "Ae2", "Ae3"]; PEO_I = ["Ep1", "Ep2", "Ep3", "Ep4", "Ep5"]
GOV_I = ["Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4"]
BPO_I = ["Co7", "Ae10", "Md7"]
def z(s): return (s - s.mean()) / s.std(ddof=1)
def cs(df, it): return df[it].apply(z).mean(axis=1).values
BPC = np.column_stack([cs(data, SA_I), cs(data, PEO_I), cs(data, GOV_I)])
BPOm = data[BPO_I].mean(axis=1).values
Z4 = np.column_stack([BPC, BPOm])                 # 56 × 4 (SA, PEO, GOV, BPO)
MU, SIG = Z4.mean(0), np.cov(Z4, rowvar=False, ddof=1)
D = np.sqrt(np.diag(SIG)); R = SIG / np.outer(D, D)
print("Correlaciones observadas con BPO:", np.round(R[:3, 3], 3))
print("Correlaciones entre dimensiones  : SA-PEO %.3f  SA-GOV %.3f  PEO-GOV %.3f"
      % (R[0, 1], R[0, 2], R[1, 2]))

# ------------------------------------------------------- redes y Garson
def garson(m):
    """Importancia relativa de Garson (1991) a partir de los pesos de la red."""
    a = np.abs(m.coefs_[0]); b = np.abs(m.coefs_[1]).ravel()
    imp = (a * b[None, :]).sum(1); return imp / imp.sum() * 100

def prep(Z):
    """Preprocesamiento del artículo: Min-Max sobre predictores y criterio."""
    return (MinMaxScaler().fit_transform(Z[:, :3]),
            MinMaxScaler().fit_transform(Z[:, 3:4]).ravel())

def single(Xs, ys, seed):
    """Red de ejecución única (configuración sin regularizar)."""
    m = MLPRegressor(hidden_layer_sizes=(10,), activation="logistic", solver="lbfgs",
                     max_iter=2000, random_state=seed)
    m.fit(Xs, ys); return garson(m)

def robust(Xs, ys, base=100, n_nets=10):
    """Protocolo robusto: 10 redes regularizadas × validación cruzada de 10 pliegues."""
    out = []
    for k in range(n_nets):
        kf = KFold(10, shuffle=True, random_state=base + k); fi = []
        for tr, _ in kf.split(Xs):
            m = MLPRegressor(hidden_layer_sizes=(3,), activation="logistic", solver="lbfgs",
                             alpha=0.1, max_iter=2000, random_state=base + k)
            m.fit(Xs[tr], ys[tr]); fi.append(garson(m))
        out.append(np.mean(fi, 0))
    out = np.array(out); return out.mean(0), out.std(0)

# --- CONTROL sobre la muestra real -------------------------------------------
Xr, yr = prep(Z4); mr, sr = robust(Xr, yr)
print(f"\nCONTROL muestra real: robusto = {np.round(mr, 1)} (±{np.round(sr, 2)})")

# --- DGP H0a: correlaciones bivariadas iguales con BPO ----------------------
R0 = R.copy(); R0[:3, 3] = R0[3, :3] = R[:3, 3].mean()
S0a = R0 * np.outer(D, D)
# --- DGP H0b: contribución CONDICIONAL igual (β_j idénticos, R² preservado) ---
Rxx = R[:3, :3]; beta_eq = np.ones(3)
rho = Rxx @ beta_eq                                   # cov(X_j, Y) estandarizada
c = (np.sqrt(R[:3, 3] @ np.linalg.solve(Rxx, R[:3, 3]))
     / np.sqrt(beta_eq @ Rxx @ beta_eq))              # igualar R² al observado
rho = rho * c
R0b = R.copy(); R0b[:3, 3] = R0b[3, :3] = rho
S0b = R0b * np.outer(D, D)
for S in (S0a, S0b):                                  # garantizar definida positiva
    ev = np.linalg.eigvalsh(S)
    if ev.min() <= 0: S += np.eye(4) * (abs(ev.min()) + 1e-3)

def run(S, label, Rn=120, seed=0):
    rng = np.random.default_rng(seed)
    rob, sing, drob, dsing = [], [], [], []
    for r in range(Rn):
        Z = rng.multivariate_normal(MU, S, size=56); Xs, ys = prep(Z)
        m, _ = robust(Xs, ys, base=100 + r); rob.append(m); drob.append(np.argmax(m))
        s = single(Xs, ys, seed=r); sing.append(s); dsing.append(np.argmax(s))
    rob, sing = np.array(rob), np.array(sing)
    def summ(name, A, d):
        cnt = np.bincount(d, minlength=3) / len(d); gap = A.max(1) - A.min(1)
        print(f"  {name:<18} media={np.round(A.mean(0), 1)}  σ={np.round(A.std(0), 1)}  "
              f"P(dom): SA={cnt[0]:.2f} PEO={cnt[1]:.2f} GOV={cnt[2]:.2f}  "
              f"brecha media={gap.mean():.1f}pp  P(>10pp)={np.mean(gap > 10):.2f}")
        return dict(mean=A.mean(0).tolist(), sd=A.std(0).tolist(), dom=cnt.tolist(),
                    gap_mean=float(gap.mean()), p_gap10=float(np.mean(gap > 10)))
    print(f"\n{label} ({Rn} réplicas, n = 56)")
    return dict(single=summ("Ejecución única", sing, dsing),
                robust=summ("Protocolo robusto", rob, drob))

res = dict(control=mr.tolist(), corr_bpo=R[:3, 3].tolist())
res["H0a_equal_bivariate"]   = run(S0a, "H0a — correlaciones bivariadas iguales con BPO", seed=11)
res["H0b_equal_conditional"] = run(S0b, "H0b — contribución condicional igual (β_j iguales)", seed=22)

# --- H1: DGP real ------------------------------------------------------------
rng = np.random.default_rng(33); rob = []; gaps = []
for r in range(120):
    Z = rng.multivariate_normal(MU, SIG, size=56); Xs, ys = prep(Z)
    m, _ = robust(Xs, ys, base=300 + r); rob.append(m); gaps.append(m[2] - max(m[0], m[1]))
rob = np.array(rob); gaps = np.array(gaps)
print(f"\nH1 — DGP real (120 réplicas, n = 56), protocolo robusto")
print(f"  media={np.round(rob.mean(0), 1)}  σ={np.round(rob.std(0), 1)}  "
      f"P(GOV dominante)={np.mean(np.argmax(rob, 1) == 2):.3f}  "
      f"ventaja GOV: media={gaps.mean():.1f}pp  P(>5pp)={np.mean(gaps > 5):.2f}  "
      f"IC95%=[{np.percentile(gaps, 2.5):.1f}, {np.percentile(gaps, 97.5):.1f}]")
res["H1"] = dict(mean=rob.mean(0).tolist(), sd=rob.std(0).tolist(),
                 p_gov=float(np.mean(np.argmax(rob, 1) == 2)), gap_mean=float(gaps.mean()),
                 gap_ci=np.percentile(gaps, [2.5, 97.5]).tolist())
json.dump(res, open("montecarlo_ann_resultados.json", "w"), indent=2)
print(f"\nTiempo: {(time.time() - t0) / 60:.1f} min → montecarlo_ann_resultados.json")
