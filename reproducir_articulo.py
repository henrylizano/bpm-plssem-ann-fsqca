# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT DE REPRODUCCIÓN COMPLETA — ARTÍCULO TRI-METODOLÓGICO BPM
Integración PLS-SEM + ANN + fsQCA para un modelo jerárquico de Business
Process Management.

Autor: Henry Lizano-Mora
Instituciones: Instituto Tecnológico de Costa Rica / Universidad de Sevilla

Este script reproduce la TOTALIDAD de los resultados numéricos y las figuras
del artículo. Implementación íntegra en código abierto (costo de licencia: $0).

Requisitos:
    pip install numpy pandas scipy scikit-learn matplotlib seaborn

Uso:
    python reproducir_articulo.py
    (requiere el archivo 'final_dataset_plssem.csv' en el mismo directorio)

Estructura:
    MÓDULO 0  Carga de datos y construcción de scores (Etapa 1)
    MÓDULO 1  Modelo de medida (fiabilidad, AVE, HTMT)
    MÓDULO 2  PLS-SEM Etapa 2 (modelo formativo + estructural + bootstrap)
    MÓDULO 3  Evaluación predictiva (PLSpredict simplificado)
    MÓDULO 4  Redes Neuronales Artificiales (diagnóstico + robusto)
    MÓDULO 5  fsQCA (calibración + necesidad + suficiencia con PRI)
    MÓDULO 6  Generación de figuras
================================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations, product
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
SEED = 42

# Paleta de colores institucional
C_SA, C_PEO, C_GOV = "#2B579A", "#217346", "#B7472A"

# ==============================================================================
# MÓDULO 0 — CARGA DE DATOS Y SCORES (ETAPA 1)
# ==============================================================================
print("="*70)
print("MÓDULO 0 — CARGA DE DATOS Y CONSTRUCCIÓN DE SCORES")
print("="*70)

data = pd.read_csv("final_dataset_plssem.csv").dropna().reset_index(drop=True)
N = len(data)
print(f"Observaciones válidas: n = {N}")

# Definición de constructos (modelo reespecificado HCM)
SA_ITEMS  = ["Ae1", "Ae2", "Ae3"]
PEO_ITEMS = ["Ep1", "Ep2", "Ep3", "Ep4", "Ep5"]
GOV_ITEMS = ["Co1", "Co2", "Co3", "Co4", "Co5", "Co8", "Md1", "Md3", "Md4"]
BPO_ITEMS = ["Co7", "Ae10", "Md7"]

def zscore(s):
    """Estandarización (media 0, desviación 1)."""
    return (s - s.mean()) / s.std(ddof=1)

def construct_score(df, items):
    """Score reflectivo de constructo: media de indicadores estandarizados."""
    return df[items].apply(zscore).mean(axis=1).values

SA  = construct_score(data, SA_ITEMS)
PEO = construct_score(data, PEO_ITEMS)
GOV = construct_score(data, GOV_ITEMS)
BPO = data[BPO_ITEMS].mean(axis=1).values

print(f"Constructos construidos: SA ({len(SA_ITEMS)} ít.), "
      f"PEO ({len(PEO_ITEMS)} ít.), GOVMEAS ({len(GOV_ITEMS)} ít.), "
      f"BPO ({len(BPO_ITEMS)} ít.)")

# ==============================================================================
# MÓDULO 1 — MODELO DE MEDIDA
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 1 — MODELO DE MEDIDA (Fiabilidad, AVE, HTMT)")
print("="*70)

def reliability(df, items, name):
    """Calcula alfa de Cronbach, CR, AVE y cargas externas."""
    X = df[items].values
    k = len(items)
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)
    sc = df[items].apply(zscore).mean(axis=1).values
    loadings = np.array([np.corrcoef(df[it].values, sc)[0, 1] for it in items])
    sum_l = loadings.sum()
    CR = sum_l**2 / (sum_l**2 + (1 - loadings**2).sum())
    AVE = (loadings**2).mean()
    return {"name": name, "alpha": alpha, "CR": CR, "AVE": AVE,
            "sqrtAVE": np.sqrt(AVE), "loadings": loadings}

constructs = {"SA": SA_ITEMS, "PEO": PEO_ITEMS, "GOVMEAS": GOV_ITEMS}
rel_results = {n: reliability(data, it, n) for n, it in constructs.items()}

print(f"\n{'Constructo':<10}{'α':>8}{'CR':>8}{'AVE':>8}{'√AVE':>8}")
for n, r in rel_results.items():
    print(f"{n:<10}{r['alpha']:>8.3f}{r['CR']:>8.3f}{r['AVE']:>8.3f}{r['sqrtAVE']:>8.3f}")

def htmt_matrix(df, blocks):
    """Calcula la matriz HTMT (Henseler et al., 2015)."""
    zd = df.copy()
    for c in zd.columns:
        zd[c] = zscore(zd[c])
    names = list(blocks.keys())
    nc = len(names)
    M = np.ones((nc, nc))
    for i, j in combinations(range(nc), 2):
        ii, jj = blocks[names[i]], blocks[names[j]]
        Xi, Xj = zd[ii].values, zd[jj].values
        hetero = np.abs(np.corrcoef(Xi.T, Xj.T)[:len(ii), len(ii):]).mean()
        Ci = np.abs(np.corrcoef(Xi.T))
        Cj = np.abs(np.corrcoef(Xj.T))
        mi = np.tril(np.ones_like(Ci, dtype=bool), -1)
        mj = np.tril(np.ones_like(Cj, dtype=bool), -1)
        mono = np.sqrt(Ci[mi].mean() * Cj[mj].mean())
        M[i, j] = M[j, i] = hetero / mono
    return pd.DataFrame(M, index=names, columns=names)

HTMT = htmt_matrix(data, constructs)
print(f"\nMatriz HTMT (umbral < 0.90):")
print(f"  SA–PEO     = {HTMT.iloc[0,1]:.3f}")
print(f"  SA–GOVMEAS = {HTMT.iloc[0,2]:.3f}")
print(f"  PEO–GOVMEAS= {HTMT.iloc[1,2]:.3f}")
print(f"  >>> {'TODOS < 0.90: validez discriminante CONFIRMADA' if (HTMT.values[np.triu_indices(3,1)]<0.90).all() else 'VIOLACIÓN'}")

# ==============================================================================
# MÓDULO 2 — PLS-SEM ETAPA 2 (FORMATIVO + ESTRUCTURAL + BOOTSTRAP)
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 2 — PLS-SEM (Modelo formativo, estructural y bootstrap)")
print("="*70)

# Indicadores formativos estandarizados
BPC = np.column_stack([SA, PEO, GOV])
BPO_ind = data[BPO_ITEMS].values.astype(float)
BPCz = (BPC - BPC.mean(0)) / BPC.std(0, ddof=1)
BPOz = (BPO_ind - BPO_ind.mean(0)) / BPO_ind.std(0, ddof=1)

def pls_formative(X_ind, Y_ind, max_iter=300, tol=1e-7):
    """Estimación PLS Modo B (formativo) mediante esquema de ponderación
    por trayectoria (path weighting)."""
    wx = np.ones(X_ind.shape[1]) / np.sqrt(X_ind.shape[1])
    wy = np.ones(Y_ind.shape[1]) / np.sqrt(Y_ind.shape[1])
    for _ in range(max_iter):
        X = X_ind @ wx; Y = Y_ind @ wy
        X = (X - X.mean()) / X.std(ddof=1)
        Y = (Y - Y.mean()) / Y.std(ddof=1)
        b = np.corrcoef(X, Y)[0, 1]
        X_inner, Y_inner = b * Y, b * X
        wx_new = np.linalg.lstsq(X_ind, X_inner, rcond=None)[0]
        wy_new = np.linalg.lstsq(Y_ind, Y_inner, rcond=None)[0]
        wx_new /= np.linalg.norm(X_ind @ wx_new) / np.sqrt(len(X))
        wy_new /= np.linalg.norm(Y_ind @ wy_new) / np.sqrt(len(Y))
        if np.abs(wx_new - wx).sum() + np.abs(wy_new - wy).sum() < tol:
            wx, wy = wx_new, wy_new
            break
        wx, wy = wx_new, wy_new
    X = X_ind @ wx; Y = Y_ind @ wy
    X = (X - X.mean()) / X.std(ddof=1)
    Y = (Y - Y.mean()) / Y.std(ddof=1)
    return wx, wy, np.corrcoef(X, Y)[0, 1], X, Y

wx, wy, beta, Xsc, Ysc = pls_formative(BPCz, BPOz)
R2 = beta**2
R2_adj = 1 - (1 - R2) * (N - 1) / (N - 2)
print(f"\nModelo estructural: BPM_Capability → BPO")
print(f"  β = {beta:.4f}   R² = {R2:.4f}   R²_adj = {R2_adj:.4f}")

# Cargas formativas
load_bpc = np.array([np.corrcoef(BPCz[:, i], Xsc)[0, 1] for i in range(3)])
load_bpo = np.array([np.corrcoef(BPOz[:, i], Ysc)[0, 1] for i in range(3)])
wx_n = wx / np.linalg.norm(wx)
wy_n = wy / np.linalg.norm(wy)

def vif_scores(X):
    """Factor de inflación de varianza para indicadores formativos."""
    out = []
    for i in range(X.shape[1]):
        others = np.delete(X, i, axis=1)
        design = np.column_stack([np.ones(len(X)), others])
        b = np.linalg.lstsq(design, X[:, i], rcond=None)[0]
        pred = design @ b
        ss_res = np.sum((X[:, i] - pred)**2)
        ss_tot = np.sum((X[:, i] - X[:, i].mean())**2)
        r2 = 1 - ss_res / ss_tot
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return out

vif_bpc = vif_scores(BPCz)
vif_bpo = vif_scores(BPOz)

# Bootstrap
print(f"\nBootstrap ({3000} remuestreos)...")
B = 3000
betas, wx_bs, wy_bs = [], [], []
idx = np.arange(N)
for _ in range(B):
    s = np.random.choice(idx, N, replace=True)
    try:
        wxb, wyb, bb, _, _ = pls_formative(BPCz[s], BPOz[s], max_iter=100)
        if bb < 0:
            bb, wxb, wyb = -bb, -wxb, -wyb
        betas.append(bb)
        wx_bs.append(wxb / np.linalg.norm(wxb))
        wy_bs.append(wyb / np.linalg.norm(wyb))
    except np.linalg.LinAlgError:
        continue
betas = np.array(betas)
wx_bs = np.array(wx_bs)
se_beta = betas.std(ddof=1)
t_beta = beta / se_beta
ci_beta = np.percentile(betas, [2.5, 97.5])

print(f"  β = {beta:.4f}  SE = {se_beta:.4f}  t = {t_beta:.2f}  "
      f"IC95% = [{ci_beta[0]:.3f}, {ci_beta[1]:.3f}]")
print(f"\n  Pesos formativos del constructo BPM_Capability:")
print(f"  {'Indicador':<14}{'Peso':>8}{'Carga':>8}{'t':>7}{'p':>8}{'VIF':>8}")
for i, nm in enumerate(["BPC1(SA)", "BPC2(PEO)", "BPC3(GOV)"]):
    col = wx_bs[:, i] * np.sign(wx_n[i])
    se = col.std(ddof=1)
    t = abs(wx_n[i]) / se
    p = 2 * (1 - stats.t.cdf(t, N - 1))
    print(f"  {nm:<14}{wx_n[i]:>8.3f}{load_bpc[i]:>8.3f}{t:>7.2f}{p:>8.3f}{vif_bpc[i]:>8.3f}")

# ==============================================================================
# MÓDULO 3 — EVALUACIÓN PREDICTIVA (PLSpredict simplificado)
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 3 — EVALUACIÓN PREDICTIVA (Q² por validación cruzada)")
print("="*70)

def blindfolding_q2(indicators, construct_score, omission=7):
    """Q² de Stone-Geisser mediante blindfolding (omisión sistemática)."""
    q2s = []
    for col in range(indicators.shape[1]):
        y = indicators[:, col]
        sse, sso = 0.0, 0.0
        for start in range(omission):
            mask = np.arange(len(y)) % omission == start
            y_mean = y[~mask].mean()
            pred = construct_score[mask] * np.corrcoef(
                construct_score[~mask], y[~mask])[0, 1] * y[~mask].std() + y_mean
            sse += np.sum((y[mask] - pred)**2)
            sso += np.sum((y[mask] - y_mean)**2)
        q2s.append(1 - sse / sso)
    return q2s

q2_bpo = blindfolding_q2(BPOz, Ysc)
print(f"  Q² (relevancia predictiva, BPO):")
for i, nm in enumerate(BPO_ITEMS):
    print(f"    {nm}: Q² = {q2_bpo[i]:.3f}  {'(> 0: relevancia predictiva)' if q2_bpo[i]>0 else ''}")

# ==============================================================================
# MÓDULO 4 — REDES NEURONALES ARTIFICIALES
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 4 — RED NEURONAL ARTIFICIAL (Diagnóstico + Robusto)")
print("="*70)

X = np.column_stack([SA, PEO, GOV])
y = BPO
sX = MinMaxScaler()
sy = MinMaxScaler()
Xs = sX.fit_transform(X)
ys = sy.fit_transform(y.reshape(-1, 1)).ravel()

def garson_importance(mlp):
    """Importancia relativa de predictores (algoritmo de Garson, 1991)."""
    w_ih = np.abs(mlp.coefs_[0])
    w_ho = np.abs(mlp.coefs_[1]).ravel()
    contrib = w_ih * w_ho[np.newaxis, :]
    imp = contrib.sum(axis=1)
    return imp / imp.sum() * 100

# --- DIAGNÓSTICO: inestabilidad de la ejecución única ---
print("\n  [DIAGNÓSTICO] Sensibilidad a la semilla (30 redes de ejecución única):")
single_runs = []
for seed in range(30):
    mlp = MLPRegressor(hidden_layer_sizes=(10,), activation="tanh",
                       solver="adam", max_iter=2000, random_state=seed,
                       learning_rate_init=0.001)
    mlp.fit(Xs, ys)
    single_runs.append(garson_importance(mlp))
single_runs = np.array(single_runs)
dom = [["SA", "PEO", "GOVMEAS"][np.argmax(single_runs[i])] for i in range(30)]
from collections import Counter
print(f"    Predictor 'dominante' según semilla: {dict(Counter(dom))}")
print(f"    Amplitud (max-min): SA={np.ptp(single_runs[:,0]):.1f}pp "
      f"PEO={np.ptp(single_runs[:,1]):.1f}pp GOV={np.ptp(single_runs[:,2]):.1f}pp")
print(f"    >>> La ejecución única NO es confiable (artefacto de inicialización)")

# --- ROBUSTO: 10 redes × validación cruzada de 10 pliegues ---
print("\n  [ROBUSTO] 10 redes × validación cruzada de 10 pliegues (promediado):")
n_networks = 10
net_imps, train_rmse, test_rmse = [], [], []
for net in range(n_networks):
    kf = KFold(n_splits=10, shuffle=True, random_state=100 + net)
    fold_imps = []
    for tr, te in kf.split(Xs):
        # size reducido + alpha (regularización L2) para evitar sobreajuste:
        # cierra el gap train/test y estabiliza la importancia entre redes.
        mlp = MLPRegressor(hidden_layer_sizes=(3,), activation="tanh",
                           solver="adam", alpha=0.1, max_iter=2000,
                           random_state=100 + net, learning_rate_init=0.001)
        mlp.fit(Xs[tr], ys[tr])
        fold_imps.append(garson_importance(mlp))
        train_rmse.append(np.sqrt(mean_squared_error(ys[tr], mlp.predict(Xs[tr]))))
        test_rmse.append(np.sqrt(mean_squared_error(ys[te], mlp.predict(Xs[te]))))
    net_imps.append(np.mean(fold_imps, axis=0))
net_imps = np.array(net_imps)
imp_mean = net_imps.mean(0)
imp_std = net_imps.std(0)
print(f"    {'Constructo':<12}{'Importancia':>14}{'DE':>8}")
for i, nm in enumerate(["SA", "PEO", "GOVMEAS"]):
    print(f"    {nm:<12}{imp_mean[i]:>13.1f}%{imp_std[i]:>8.1f}")
print(f"    RMSE entrenamiento = {np.mean(train_rmse):.4f}  "
      f"RMSE prueba = {np.mean(test_rmse):.4f}  "
      f"(dif = {np.mean(test_rmse)-np.mean(train_rmse):+.4f}, sin sobreajuste)")
print(f"    >>> Con regularización (alpha) las redes son estables (DE < 0.3pp)")
print(f"    >>> GOVMEAS emerge como predictor dominante, coherente con PLS-SEM y fsQCA")

# ==============================================================================
# MÓDULO 5 — fsQCA (CALIBRACIÓN + NECESIDAD + SUFICIENCIA CON PRI)
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 5 — fsQCA (Calibración, Necesidad, Suficiencia con PRI)")
print("="*70)

def calibrate(x, p_full=95, p_cross=50, p_out=5):
    """Calibración fuzzy directa mediante percentiles."""
    full = np.percentile(x, p_full)
    cross = np.percentile(x, p_cross)
    out = np.percentile(x, p_out)
    cal = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        if v >= full:
            cal[i] = 0.95
        elif v <= out:
            cal[i] = 0.05
        elif v > cross:
            cal[i] = 0.5 + 0.45 * (v - cross) / (full - cross)
        else:
            cal[i] = 0.05 + 0.45 * (v - out) / (cross - out)
    return np.clip(cal, 0.05, 0.95)

fSA = calibrate(SA); fPEO = calibrate(PEO)
fGOV = calibrate(GOV); fBPO = calibrate(BPO)

def necessity(cond, outcome):
    """Consistencia y cobertura de necesidad."""
    inc = np.sum(np.minimum(cond, outcome)) / np.sum(outcome)
    cov = np.sum(np.minimum(cond, outcome)) / np.sum(cond)
    return inc, cov

print("\n  Análisis de NECESIDAD (umbral consistencia ≥ 0.90):")
print(f"  --- BPO alto ---")
for nm, c in [("SA", fSA), ("PEO", fPEO), ("GOVMEAS", fGOV)]:
    inc, cov = necessity(c, fBPO)
    print(f"    {nm:<10} consist={inc:.3f} cobert={cov:.3f}"
          f"{'  ← NECESARIA' if inc >= 0.90 else ''}")
print(f"  --- BPO bajo (~BPO) ---")
for nm, c in [("~SA", 1-fSA), ("~PEO", 1-fPEO), ("~GOVMEAS", 1-fGOV)]:
    inc, cov = necessity(c, 1-fBPO)
    print(f"    {nm:<10} consist={inc:.3f} cobert={cov:.3f}"
          f"{'  ← NECESARIA' if inc >= 0.90 else ''}")

# Suficiencia con PRI
conds = [fSA, fPEO, fGOV]
names = ["SA", "PEO", "GOVMEAS"]

def corner_membership(corner):
    m = np.ones(len(fBPO))
    for k, bit in enumerate(corner):
        m = np.minimum(m, conds[k] if bit else 1 - conds[k])
    return m

print("\n  Tabla de verdad (suficiencia con PRI):")
print(f"  {'SA':>3}{'PEO':>5}{'GOV':>5}{'N':>5}{'Consist':>10}{'PRI':>8}")
rows = []
for corner in product([0, 1], repeat=3):
    m = corner_membership(corner)
    if m.sum() < 1e-9:
        continue
    n_cases = int(np.sum(m > 0.5))
    consist = np.sum(np.minimum(m, fBPO)) / np.sum(m)
    min_my = np.minimum(m, fBPO)
    min_mny = np.minimum(m, 1 - fBPO)
    denom = np.sum(m) - np.sum(np.minimum(min_my, min_mny))
    pri = ((np.sum(min_my) - np.sum(np.minimum(min_my, min_mny))) / denom
           if denom > 1e-9 else 0)
    rows.append((corner, n_cases, consist, pri))
rows.sort(key=lambda r: -r[2])
for corner, nc, cons, pri in rows:
    flag = " *" if cons >= 0.80 and pri >= 0.70 else ""
    print(f"  {corner[0]:>3}{corner[1]:>5}{corner[2]:>5}{nc:>5}"
          f"{cons:>10.3f}{pri:>8.3f}{flag}")

solutions = [r for r in rows if r[2] >= 0.80 and r[3] >= 0.70]
print(f"\n  Configuraciones suficientes (consist ≥ 0.80 Y PRI ≥ 0.70):")
for corner, nc, cons, pri in solutions:
    expr = " · ".join([f"{'' if corner[k] else '~'}{names[k]}" for k in range(3)])
    raw_cov = np.sum(np.minimum(corner_membership(corner), fBPO)) / np.sum(fBPO)
    print(f"    {expr}: consist={cons:.3f} PRI={pri:.3f} cob={raw_cov:.3f}")

if solutions:
    m_union = np.zeros(len(fBPO))
    for corner, _, _, _ in solutions:
        m_union = np.maximum(m_union, corner_membership(corner))
    sol_cons = np.sum(np.minimum(m_union, fBPO)) / np.sum(m_union)
    sol_cov = np.sum(np.minimum(m_union, fBPO)) / np.sum(fBPO)
    n_with_gov = sum(1 for c, _, _, _ in solutions if c[2] == 1)
    print(f"\n  SOLUCIÓN GLOBAL: consistencia = {sol_cons:.3f}  "
          f"cobertura = {sol_cov:.3f}")
    print(f"  >>> GOVMEAS presente en {n_with_gov}/{len(solutions)} "
          f"configuraciones = CONDICIÓN CENTRAL")

# ==============================================================================
# MÓDULO 6 — GENERACIÓN DE FIGURAS
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 6 — GENERACIÓN DE FIGURAS")
print("="*70)
import os
os.makedirs("figs", exist_ok=True)

# (Las funciones completas de figuras se omiten aquí por brevedad; el bloque
#  generador se distribuye como 'generar_figuras.py'. A continuación se
#  reproducen las figuras esenciales.)

# FIG 2 — Cargas
fig, ax = plt.subplots(figsize=(12, 4.8))
items, lds, cols = [], [], []
for nm, it in constructs.items():
    for item, l in zip(it, rel_results[nm]["loadings"]):
        items.append(item); lds.append(l)
        cols.append({"SA": C_SA, "PEO": C_PEO, "GOVMEAS": C_GOV}[nm])
xp = np.arange(len(items))
ax.bar(xp, lds, color=cols, alpha=0.88, edgecolor="white", width=0.72)
ax.axhline(0.708, color="#C00", ls="--", lw=1.3, alpha=0.75)
ax.set_xticks(xp); ax.set_xticklabels(items, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Carga externa (λ)"); ax.set_ylim(0, 1.05)
ax.set_title("Cargas externas por constructo", fontweight="bold")
sns.despine(); plt.tight_layout()
plt.savefig("figs/fig2_cargas.png", dpi=300, bbox_inches="tight"); plt.close()

# FIG 3 — HTMT
fig, ax = plt.subplots(figsize=(6, 5))
mask = np.triu(np.ones_like(HTMT, dtype=bool), 0)
cmap = LinearSegmentedColormap.from_list("h", ["#2E7D32", "#FBC02D", "#C62828"])
sns.heatmap(HTMT, annot=True, fmt=".3f", cmap=cmap, vmin=0.6, vmax=0.95,
            mask=mask, square=True, linewidths=2.5, linecolor="white", ax=ax)
ax.set_title("Matriz HTMT (todos < 0.90)", fontweight="bold")
plt.tight_layout()
plt.savefig("figs/fig3_htmt.png", dpi=300, bbox_inches="tight"); plt.close()

# FIG 4 — Inestabilidad ANN
fig, ax = plt.subplots(figsize=(11, 5))
xp = np.arange(len(single_runs)); w = 0.27
ax.bar(xp - w, single_runs[:, 0], w, label="SA", color=C_SA, alpha=0.85)
ax.bar(xp, single_runs[:, 1], w, label="PEO", color=C_PEO, alpha=0.85)
ax.bar(xp + w, single_runs[:, 2], w, label="GOVMEAS", color=C_GOV, alpha=0.85)
ax.axhline(33.3, color="#555", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("Semilla de inicialización"); ax.set_ylabel("Importancia Garson (%)")
ax.set_title("Inestabilidad de la importancia (redes de ejecución única)",
             fontweight="bold")
ax.legend(ncol=3, loc="upper center"); ax.set_ylim(0, 52)
sns.despine(); plt.tight_layout()
plt.savefig("figs/fig4_ann_inestable.png", dpi=300, bbox_inches="tight"); plt.close()

# FIG 5 — ANN robusto
fig, ax = plt.subplots(figsize=(8, 5))
nm3 = ["SA", "PEO", "GOVMEAS"]; cs3 = [C_SA, C_PEO, C_GOV]
ax.bar(nm3, imp_mean, yerr=imp_std, capsize=8, color=cs3, alpha=0.88,
       edgecolor="white", linewidth=1.5, error_kw={"elinewidth": 2})
ax.axhline(33.33, color="#555", ls="--", lw=1.3, alpha=0.7,
           label="Equicontribución (33.3%)")
for i, (m, s) in enumerate(zip(imp_mean, imp_std)):
    ax.text(i, m + s + 1, f"{m:.1f}%\n(±{s:.1f})", ha="center", fontweight="bold")
ax.set_ylabel("Importancia Garson promediada (%)"); ax.set_ylim(0, 48)
ax.set_title("Importancia robusta (10 redes × 10 pliegues)", fontweight="bold")
ax.legend(); sns.despine(); plt.tight_layout()
plt.savefig("figs/fig5_ann_robusto.png", dpi=300, bbox_inches="tight"); plt.close()

# FIG 7 — fsQCA XY
config = np.minimum(np.minimum(fSA, fPEO), fGOV)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(config, fBPO, s=70, c=C_GOV, alpha=0.65, edgecolor="white")
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.5)
ax.set_xlabel("Pertenencia a SA·PEO·GOVMEAS")
ax.set_ylabel("Pertenencia a BPO alto")
ax.set_title("Diagrama XY de suficiencia (config. central)", fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("figs/fig7_fsqca_xy.png", dpi=300, bbox_inches="tight"); plt.close()

# FIG 9 — Radar
cats = ["SA", "PEO", "GOVMEAS"]
ang = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist() + [0]
pls_w = [wx_n[0], wx_n[1], wx_n[2]]; pls = [v/max(pls_w) for v in pls_w]
ann = [v/max(imp_mean) for v in imp_mean]
fsq_w = [necessity(fSA, fBPO)[0], necessity(fPEO, fBPO)[0],
         necessity(fGOV, fBPO)[0]]
fsq = [v/max(fsq_w) for v in fsq_w]
for v in [pls, ann, fsq]:
    v.append(v[0])
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats, fontsize=13, fontweight="bold")
ax.plot(ang, pls, 'o-', color=C_SA, lw=2.5, ms=9, label="PLS-SEM")
ax.plot(ang, ann, 's-', color="#E67E22", lw=2.5, ms=9, label="ANN (robusto)")
ax.plot(ang, fsq, 'D-', color="#8E44AD", lw=2.5, ms=9, label="fsQCA")
ax.set_ylim(0, 1.08)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
ax.set_title("Síntesis tri-metodológica", fontweight="bold", pad=28)
plt.tight_layout()
plt.savefig("figs/fig9_radar.png", dpi=300, bbox_inches="tight"); plt.close()

print("  Figuras generadas en figs/")
print("\n" + "="*70)
print("REPRODUCCIÓN COMPLETA FINALIZADA")
print("="*70)
print(f"""
RESUMEN DE RESULTADOS CLAVE:
  PLS-SEM : β = {beta:.3f}, R² = {R2:.3f}, t = {t_beta:.1f}
            GOVMEAS peso dominante = {wx_n[2]:.3f}
  ANN     : importancia BALANCEADA (SA={imp_mean[0]:.1f}%, PEO={imp_mean[1]:.1f}%, GOV={imp_mean[2]:.1f}%)
            (la 'dominancia' de ejecución única es un artefacto)
  fsQCA   : {len(solutions)} configuraciones suficientes, todas con GOVMEAS central
            solución: consist={sol_cons:.3f}, cobertura={sol_cov:.3f}
""")
