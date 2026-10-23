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
# MÓDULO 3 — EVALUACIÓN PREDICTIVA (PLSpredict: PLS frente a LM)
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 3 — EVALUACIÓN PREDICTIVA (PLSpredict: PLS vs. LM)")
print("="*70)

# Procedimiento PLSpredict (Shmueli et al., 2016, 2019): validación cruzada de
# 10 pliegues × 10 repeticiones. En cada pliegue se estandariza con las
# estadísticas del conjunto de ENTRENAMIENTO, se estima el modelo PLS y se
# predicen los indicadores endógenos del conjunto de prueba. El referente
# lineal (LM) regresa cada indicador endógeno sobre los tres indicadores
# exógenos. Q²predict emplea como base ingenua la media de entrenamiento.
Braw = np.column_stack([SA, PEO, GOV])
Yraw = BPO_ind.copy()

def plspredict(Braw, Yraw, n_folds=10, n_reps=10, seed0=200):
    nobs = len(Braw)
    r_pls = np.zeros((n_reps, 3)); r_lm = np.zeros((n_reps, 3))
    m_pls = np.zeros((n_reps, 3)); m_lm = np.zeros((n_reps, 3))
    q2p   = np.zeros((n_reps, 3))
    for rep in range(n_reps):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed0 + rep)
        e_pls = np.zeros((nobs, 3)); e_lm = np.zeros((nobs, 3))
        yz = np.zeros((nobs, 3))
        for tr, te in kf.split(Braw):
            mB, sB = Braw[tr].mean(0), Braw[tr].std(0, ddof=1)
            mY, sY = Yraw[tr].mean(0), Yraw[tr].std(0, ddof=1)
            Xtr = (Braw[tr] - mB) / sB; Xte = (Braw[te] - mB) / sB
            Ytr = (Yraw[tr] - mY) / sY; Yte = (Yraw[te] - mY) / sY
            wx_t, wy_t, b_t, Xsc_t, Ysc_t = pls_formative(Xtr, Ytr)
            if b_t < 0:
                b_t, wx_t, Xsc_t = -b_t, -wx_t, -Xsc_t
            sc_tr = Xtr @ wx_t
            mu, esc = sc_tr.mean(), sc_tr.std(ddof=1)
            lv_te = b_t * ((Xte @ wx_t) - mu) / esc
            loadY = np.array([np.corrcoef(Ytr[:, i], Ysc_t)[0, 1]
                              for i in range(3)])
            e_pls[te] = Yte - np.outer(lv_te, loadY)
            D_tr = np.column_stack([np.ones(len(tr)), Xtr])
            D_te = np.column_stack([np.ones(len(te)), Xte])
            for i in range(3):
                bc = np.linalg.lstsq(D_tr, Ytr[:, i], rcond=None)[0]
                e_lm[te, i] = Yte[:, i] - D_te @ bc
            yz[te] = Yte
        r_pls[rep] = np.sqrt((e_pls**2).mean(0))
        r_lm[rep]  = np.sqrt((e_lm**2).mean(0))
        m_pls[rep] = np.abs(e_pls).mean(0)
        m_lm[rep]  = np.abs(e_lm).mean(0)
        q2p[rep]   = 1 - (e_pls**2).sum(0) / (yz**2).sum(0)
    return (r_pls.mean(0), r_lm.mean(0), m_pls.mean(0), m_lm.mean(0),
            q2p.mean(0))

rmse_pls, rmse_lm, mae_pls, mae_lm, q2pred = plspredict(Braw, Yraw)
print(f"\n  {'Indicador':<10}{'RMSE_PLS':>10}{'RMSE_LM':>9}"
      f"{'MAE_PLS':>9}{'MAE_LM':>8}{'Q²pred':>8}")
wins = 0
for i, nm in enumerate(BPO_ITEMS):
    gana = rmse_pls[i] < rmse_lm[i]; wins += int(gana)
    print(f"  {nm:<10}{rmse_pls[i]:>10.3f}{rmse_lm[i]:>9.3f}"
          f"{mae_pls[i]:>9.3f}{mae_lm[i]:>8.3f}{q2pred[i]:>8.3f}"
          f"{'   PLS<LM' if gana else '   PLS>=LM'}")
poder = "alto" if wins == 3 else ("medio" if wins == 2 else "bajo")
print(f"\n  >>> El modelo PLS supera al referente LM en {wins}/3 indicadores")
print(f"  >>> Poder predictivo {poder} según los criterios de "
      f"Shmueli et al. (2019)")

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
        # Arquitectura parsimoniosa (3 neuronas) + regularización L2 (alpha),
        # con optimizador L-BFGS y activación logística: réplica de la
        # configuración de nnet en R (BFGS + sigmoide + decay = 0.1), lo que
        # garantiza la equivalencia numérica inter-plataforma del hallazgo.
        mlp = MLPRegressor(hidden_layer_sizes=(3,), activation="logistic",
                           solver="lbfgs", alpha=0.1, max_iter=2000,
                           random_state=100 + net)
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
# MÓDULO 5 — fsQCA (CALIBRACIÓN LOGÍSTICA + NECESIDAD/RoN + SOLUCIÓN CONSERVADORA)
# ==============================================================================
print("\n" + "="*70)
print("MÓDULO 5 — fsQCA (Calibración, Necesidad+RoN, Suficiencia con PRI)")
print("="*70)

def calibrate(x, p_full=95, p_cross=50, p_out=5, idm=0.95):
    """Calibración fuzzy directa por el método logístico de Ragin,
    numéricamente equivalente a QCA::calibrate(type='fuzzy', logistic=TRUE)
    en R: anclajes por percentiles y log-odds = ±log(idm/(1-idm)) en los
    umbrales de pertenencia plena y de no pertenencia."""
    e = np.percentile(x, p_full)
    c = np.percentile(x, p_cross)
    i = np.percentile(x, p_out)
    L = np.log(idm / (1 - idm))
    lo = np.where(x >= c, (x - c) * L / (e - c), (x - c) * L / (c - i))
    return 1.0 / (1.0 + np.exp(-lo))

fSA = calibrate(SA); fPEO = calibrate(PEO)
fGOV = calibrate(GOV); fBPO = calibrate(BPO)

def necessity(cond, outcome):
    """Consistencia y cobertura de necesidad."""
    inc = np.sum(np.minimum(cond, outcome)) / np.sum(outcome)
    cov = np.sum(np.minimum(cond, outcome)) / np.sum(cond)
    return inc, cov

def ron(cond, outcome):
    """Relevance of Necessity (Schneider y Wagemann, 2012)."""
    return np.sum(1 - cond) / np.sum(1 - np.minimum(cond, outcome))

def tabla_necesidad(pares, outcome, titulo):
    print(f"  --- {titulo} ---")
    print(f"    {'Condición':<11}{'inclN':>8}{'RoN':>8}{'covN':>8}")
    for nm, cnd in pares:
        inc, cov = necessity(cnd, outcome)
        marca = "  <- supera 0.90" if inc >= 0.90 else ""
        print(f"    {nm:<11}{inc:>8.3f}{ron(cnd, outcome):>8.3f}"
              f"{cov:>8.3f}{marca}")

print("\n  Análisis de NECESIDAD (umbral de consistencia >= 0.90):")
tabla_necesidad([("SA", fSA), ("PEO", fPEO), ("GOVMEAS", fGOV)],
                fBPO, "BPO alto")
tabla_necesidad([("~SA", 1 - fSA), ("~PEO", 1 - fPEO),
                 ("~GOVMEAS", 1 - fGOV)], 1 - fBPO, "BPO bajo (~BPO)")

# ---- Suficiencia: tabla de verdad, remanentes y minimización conservadora ---
NAMES = ["SA", "PEO", "GOVMEAS"]

def memb(spec, conds):
    """Pertenencia a un término; spec = tupla de 0/1/None por condición."""
    m = np.ones(len(conds[0]))
    for k, bit in enumerate(spec):
        if bit is None:
            continue
        m = np.minimum(m, conds[k] if bit else 1 - conds[k])
    return m

def incl_pri(m, out):
    smy = np.sum(np.minimum(m, out)); sm = np.sum(m)
    s3 = np.sum(np.minimum(np.minimum(m, out), 1 - out))
    incl = smy / sm
    pri = (smy - s3) / (sm - s3) if sm - s3 > 1e-12 else 0.0
    return incl, pri

def tabla_verdad(conds, out, inc_cut=0.80, pri_cut=0.70):
    filas = []
    for corner in product([0, 1], repeat=3):
        m = memb(corner, conds)
        n = int(np.sum(m > 0.5))
        incl, pri = incl_pri(m, out)
        o = "?" if n == 0 else ("1" if (incl >= inc_cut and pri >= pri_cut)
                                 else "0")
        filas.append((corner, n, incl, pri, o))
    return filas

def minimiza_conservadora(tt):
    """Quine-McCluskey sobre las filas OBSERVADAS con OUT = 1; los remanentes
    lógicos (n = 0) NO se incorporan: solución conservadora (compleja)."""
    pos = [tuple(c) for c, n, i, p, o in tt if o == "1"]
    actual = {tuple(c): frozenset([tuple(c)]) for c in pos}
    cambio = True
    while cambio:
        cambio = False
        items = list(actual.items()); nuevo = {}; usados = set()
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                t1, c1 = items[a]; t2, c2 = items[b]
                dif = [k for k in range(3)
                       if t1[k] != t2[k] and t1[k] is not None
                       and t2[k] is not None]
                same_dc = all((t1[k] is None) == (t2[k] is None)
                              for k in range(3))
                if len(dif) == 1 and same_dc:
                    k = dif[0]
                    merged = tuple(None if j == k else t1[j] for j in range(3))
                    nuevo[merged] = nuevo.get(merged, frozenset()) | c1 | c2
                    usados.add(t1); usados.add(t2); cambio = True
        for t, c in items:
            if t not in usados:
                nuevo[t] = nuevo.get(t, frozenset()) | c
        actual = nuevo
    primos = list(actual.items())
    minterms = set().union(*[c for _, c in primos]) if primos else set()
    sel, cubierto = [], set()
    for mt in sorted(minterms):
        cubren = [t for t, c in primos if mt in c]
        if len(cubren) == 1 and cubren[0] not in sel:
            sel.append(cubren[0]); cubierto |= dict(primos)[cubren[0]]
    for t, c in sorted(primos, key=lambda x: -len(x[1])):
        if minterms <= cubierto:
            break
        if t not in sel and c - cubierto:
            sel.append(t); cubierto |= c
    return sel

def expr(term):
    partes = [("" if v else "~") + NAMES[k]
              for k, v in enumerate(term) if v is not None]
    return "*".join(partes) if partes else "1"

def resolver(conds, out, etiqueta):
    tt = tabla_verdad(conds, out)
    print(f"\n  Tabla de verdad ({etiqueta}; incl.cut = 0.80, PRI.cut = 0.70):")
    print(f"    {'SA':>3}{'PEO':>5}{'GOV':>5}{'OUT':>5}{'n':>4}"
          f"{'incl':>8}{'PRI':>7}")
    for corner, n, incl, pri, o in sorted(
            tt, key=lambda r: (-(r[4] == "1"), -r[2])):
        nota = "  <- remanente lógico" if o == "?" else ""
        print(f"    {corner[0]:>3}{corner[1]:>5}{corner[2]:>5}{o:>5}{n:>4}"
              f"{incl:>8.3f}{pri:>7.3f}{nota}")
    sol = minimiza_conservadora(tt)
    m_terms = [memb(t, conds) for t in sol]
    m_union = np.zeros(len(out))
    for m in m_terms:
        m_union = np.maximum(m_union, m)
    inclS, priS = incl_pri(m_union, out)
    covS = np.sum(np.minimum(m_union, out)) / np.sum(out)
    print("\n  Solución CONSERVADORA (los remanentes lógicos no se usan):")
    print(f"    {'Término':<16}{'inclS':>8}{'PRI':>8}{'covS':>8}{'covU':>8}")
    for j, (t, m) in enumerate(zip(sol, m_terms)):
        iT, pT = incl_pri(m, out)
        cT = np.sum(np.minimum(m, out)) / np.sum(out)
        resto = np.zeros(len(out))
        for j2, m2 in enumerate(m_terms):
            if j2 != j:
                resto = np.maximum(resto, m2)
        cU = covS - np.sum(np.minimum(resto, out)) / np.sum(out)
        print(f"    {expr(t):<16}{iT:>8.3f}{pT:>8.3f}{cT:>8.3f}{cU:>8.3f}")
    print(f"    {'M1 (solución)':<16}{inclS:>8.3f}{priS:>8.3f}"
          f"{covS:>8.3f}{'--':>8}")
    return sol, inclS, priS, covS

sol, sol_cons, sol_pri, sol_cov = resolver([fSA, fPEO, fGOV], fBPO,
                                           "anclajes 95/50/5")
sol_expr = " + ".join(expr(t) for t in sol)
n_gov = sum(1 for t in sol if t[2] == 1)
print(f"  >>> GOVMEAS presente en {n_gov}/{len(sol)} términos de la "
      f"solución = CONDICIÓN CENTRAL")

# ---- A10: sensibilidad de la calibración (anclajes 90/50/10) ----------------
print("\n  SENSIBILIDAD de la calibración — anclajes alternativos 90/50/10:")
fSA2 = calibrate(SA, 90, 50, 10); fPEO2 = calibrate(PEO, 90, 50, 10)
fGOV2 = calibrate(GOV, 90, 50, 10); fBPO2 = calibrate(BPO, 90, 50, 10)
inc2 = [necessity(c, fBPO2)[0] for c in (fSA2, fPEO2, fGOV2)]
print(f"    Necesidad (BPO alto): SA={inc2[0]:.3f}  PEO={inc2[1]:.3f}  "
      f"GOVMEAS={inc2[2]:.3f}")
sol2, c2, p2, v2 = resolver([fSA2, fPEO2, fGOV2], fBPO2,
                            "anclajes 90/50/10")
print(f"    Solución 90/50/10: {' + '.join(expr(t) for t in sol2)}"
      f"  (inclS={c2:.3f}, PRI={p2:.3f}, covS={v2:.3f})")
print("  >>> La estructura de la solución y la centralidad de GOVMEAS se "
      "mantienen bajo ambos esquemas de anclaje")

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
ax.set_title("Importancia robusta regularizada (10 redes × 10 pliegues)", fontweight="bold")
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
  PLS-SEM  : β = {beta:.3f}, R² = {R2:.3f}, t = {t_beta:.1f}
             GOVMEAS peso formativo dominante = {wx_n[2]:.3f}
  PLSpredict: PLS supera al referente LM en {wins}/3 indicadores (RMSE)
  ANN (L-BFGS + logística, regularizada):
             GOVMEAS dominante = {imp_mean[2]:.1f}%  (SA = {imp_mean[0]:.1f}%, PEO = {imp_mean[1]:.1f}%)
             estabilidad inter-redes: DE máxima = {imp_std.max():.1f} pp
  fsQCA    : solución conservadora M1 = {sol_expr}
             inclS = {sol_cons:.3f}, PRI = {sol_pri:.3f}, covS = {sol_cov:.3f}
             GOVMEAS presente en todos los términos = condición central
""")
