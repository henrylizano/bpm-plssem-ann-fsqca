#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ANÁLISIS DE ROBUSTEZ (3/3) — PROTOCOLO DE ROBUSTEZ PARA fsQCA
Artículo tri-metodológico BPM (PLS-SEM + ANN + fsQCA) — Anexo C
Autor: Henry Lizano-Mora · Licencia MIT
==============================================================================
Protocolo de robustez para fsQCA (Oana & Schneider, 2024; paquete SetMethods):
  Test 1 — Rangos de sensibilidad (anclajes de calibración, cortes de
           consistencia bruta, PRI y frecuencia), un parámetro a la vez.
  Test 2 — Robustez orientada al ajuste: conjunto de prueba (TS) de soluciones
           plausibles; RF_cons, RF_cov y coincidencia de conjuntos entre la
           solución inicial (IS) y TS_min / TS_max.
  Test 3 — Robustez orientada a los casos: razón de casos típicos robustos
           (RCR_typ) y de casos desviados robustos (RCR_dev); clasificación de
           cada caso como robusto / inestable / posible.
  Núcleo robusto — parte de la solución que sobrevive a todas las pruebas.
Calibración: método directo logístico (Ragin, 2008) como en QCA::calibrate.
Las medidas de necesidad siguen la convención estándar: inclN = Σmin(X,Y)/ΣY,
covN = Σmin(X,Y)/ΣX, RoN = Σ(1−X)/Σ(1−min(X,Y)) (Schneider & Wagemann, 2012).
Uso:  python robustez_fsqca.py   (requiere final_dataset_plssem.csv)
Salida: fsqca_robustez_resultados.json
==============================================================================
"""
import numpy as np, pandas as pd, json, itertools, functools
from collections import Counter, defaultdict
print = functools.partial(print, flush=True)

# ----------------------------------------------------------------- datos
data = pd.read_csv("final_dataset_plssem.csv")
SA_I = ["Ae1","Ae2","Ae3"]; PEO_I = ["Ep1","Ep2","Ep3","Ep4","Ep5"]
GOV_I = ["Co1","Co2","Co3","Co4","Co5","Co8","Md1","Md3","Md4"]; BPO_I = ["Co7","Ae10","Md7"]
def z(s): return (s - s.mean()) / s.std(ddof=1)
def cs(df, it): return df[it].apply(z).mean(axis=1).values
RAW = dict(SA=cs(data,SA_I), PEO=cs(data,PEO_I), GOVMEAS=cs(data,GOV_I),
           BPO=data[BPO_I].mean(axis=1).values)
COND = ["SA","PEO","GOVMEAS"]; N = len(data)

# ------------------------------------------------------------ calibración
def calibrate(x, p_in=95, p_cross=50, p_ex=5):
    i, c, e = np.percentile(x, [p_in, p_cross, p_ex]); L = np.log(0.95/0.05)
    y = np.where(x >= c, (x - c)/(i - c)*L, (x - c)/(c - e)*L)
    return 1/(1+np.exp(-y))

def calibrate_all(p_in=95, p_cross=50, p_ex=5):
    return {k: calibrate(v, p_in, p_cross, p_ex) for k, v in RAW.items()}

# ----------------------------------------------------- medidas de conjunto
def incl(m, y):  return np.minimum(m, y).sum()/m.sum()
def cov(m, y):   return np.minimum(m, y).sum()/y.sum()
def pri(m, y):
    a = np.minimum(m, y).sum(); b = np.minimum(np.minimum(m, y), 1-y).sum()
    d = m.sum() - b; return (a - b)/d if d > 1e-9 else 0.0
def ron(m, y):   return (1-m).sum()/(1-np.minimum(m, y)).sum()

def corner_m(F, corner):
    m = np.ones(N)
    for k, b in enumerate(corner):
        m = np.minimum(m, F[COND[k]] if b else 1-F[COND[k]])
    return m

def term_m(F, term):          # term: tuple de (None|0|1) por condición
    m = np.ones(N)
    for k, b in enumerate(term):
        if b is None: continue
        m = np.minimum(m, F[COND[k]] if b else 1-F[COND[k]])
    return m

def term_str(term):
    p = [("" if b else "~")+COND[k] for k, b in enumerate(term) if b is not None]
    return "*".join(p) if p else "1"

# ---------------------------------------------------- tabla de verdad
def truth_table(F, y, incl_cut, pri_cut, n_cut):
    rows = []
    for corner in itertools.product([0,1], repeat=3):
        m = corner_m(F, corner); n_cases = int((m > 0.5).sum())
        if n_cases < n_cut or m.sum() < 1e-9:
            rows.append((corner, n_cases, np.nan, np.nan, "?")); continue
        ic, pr = incl(m, y), pri(m, y)
        rows.append((corner, n_cases, ic, pr, "1" if (ic >= incl_cut and pr >= pri_cut) else "0"))
    return rows

# --------------------------------------------- minimización conservadora
def minimize_conservative(rows):
    ones = [r[0] for r in rows if r[4] == "1"]
    if not ones: return []
    ones_set = set(ones)
    # implicantes: todos los términos (con None) que cubren solo filas "1"
    implicants = []
    for term in itertools.product([None,0,1], repeat=3):
        if all(b is None for b in term): continue
        covered = [c for c in itertools.product([0,1], repeat=3)
                   if all(b is None or b == c[k] for k, b in enumerate(term))]
        if all(c in ones_set for c in covered):
            implicants.append((term, set(covered)))
    # primos = no contenidos en otro implicante más general
    primes = [t for t in implicants
              if not any(o != t and o[1] > t[1] for o in implicants)]
    # cobertura mínima (búsqueda exhaustiva, 3 condiciones)
    best = None
    for r in range(1, len(primes)+1):
        for combo in itertools.combinations(primes, r):
            if set().union(*[c[1] for c in combo]) == ones_set:
                best = [c[0] for c in combo]; break
        if best: break
    return sorted(best, key=term_str)

def solve(p_in=95, p_cross=50, p_ex=5, incl_cut=0.80, pri_cut=0.70, n_cut=1):
    F = calibrate_all(p_in, p_cross, p_ex); y = F["BPO"]
    rows = truth_table(F, y, incl_cut, pri_cut, n_cut)
    terms = minimize_conservative(rows)
    if not terms:
        return dict(terms=[], expr="∅", m=np.zeros(N), cons=np.nan, pri=np.nan,
                    cov=np.nan, F=F, rows=rows, term_stats=[])
    ms = [term_m(F, t) for t in terms]; m = np.max(ms, axis=0)
    ts = []
    for t, mt in zip(terms, ms):
        others = [o for o in ms if o is not mt]
        uniq = ((np.minimum(m, y).sum() - np.minimum(np.max(others, axis=0), y).sum())/y.sum()
                if others else cov(mt, y))
        ts.append(dict(term=term_str(t), inclS=incl(mt,y), PRI=pri(mt,y),
                       covS=cov(mt,y), covU=uniq))
    return dict(terms=terms, expr=" + ".join(term_str(t) for t in terms), m=m,
                cons=incl(m,y), pri=pri(m,y), cov=cov(m,y), F=F, rows=rows, term_stats=ts)

# ===================================================== SOLUCIÓN INICIAL (IS)
print("="*74); print("SOLUCIÓN INICIAL (IS) — anclajes 95/50/5, incl ≥ 0.80, PRI ≥ 0.70, n ≥ 1")
print("="*74)
IS = solve()
F0, y0 = IS["F"], IS["F"]["BPO"]
print("  Necesidad (BPO alto):")
for c in COND:
    m = F0[c]; print(f"    {c:<8} inclN={cov(m,y0):.3f}  RoN={ron(m,y0):.3f}  covN={incl(m,y0):.3f}")
print("  Necesidad (~BPO):")
for c in COND:
    m = 1-F0[c]; print(f"    ~{c:<7} inclN={cov(m,1-y0):.3f}  RoN={ron(m,1-y0):.3f}")
print("\n  Tabla de verdad:")
for corner, n, ic, pr, out in sorted(IS["rows"], key=lambda r: (r[4]!="1", -(r[2] if r[2]==r[2] else 0))):
    print(f"    {corner}  n={n:<3} incl={ic:.3f} PRI={pr:.3f}  OUT={out}" if ic==ic
          else f"    {corner}  n={n:<3} (remanente)")
print(f"\n  IS = {IS['expr']}")
for t in IS["term_stats"]:
    print(f"    {t['term']:<16} inclS={t['inclS']:.3f} PRI={t['PRI']:.3f} covS={t['covS']:.3f} covU={t['covU']:.3f}")
print(f"  inclS={IS['cons']:.3f}  PRI={IS['pri']:.3f}  covS={IS['cov']:.3f}")
IS_EXPR = IS["expr"]

# ================================================ TEST 1 — RANGOS DE SENSIBILIDAD
print("\n" + "="*74); print("TEST 1 — RANGOS DE SENSIBILIDAD (un parámetro a la vez)")
print("="*74)
BASE = dict(p_in=95, p_cross=50, p_ex=5, incl_cut=0.80, pri_cut=0.70, n_cut=1)
GRIDS = dict(
    p_in=np.arange(80, 99.5, 0.5), p_cross=np.arange(30, 70.5, 0.5),
    p_ex=np.arange(0.5, 20.5, 0.5), incl_cut=np.arange(0.70, 0.995, 0.005),
    pri_cut=np.arange(0.50, 0.95, 0.005), n_cut=[1,2,3,4,5])
sens = {}
for par, grid in GRIDS.items():
    same = []
    for v in grid:
        kw = dict(BASE); kw[par] = float(v) if par != "n_cut" else int(v)
        try: s = solve(**kw)
        except Exception: continue
        same.append((v, s["expr"] == IS_EXPR, s["expr"]))
    # rango contiguo que contiene el valor base y mantiene IS
    base_v = BASE[par]; vals = [v for v,_,_ in same]
    idx = min(range(len(vals)), key=lambda i: abs(vals[i]-base_v))
    lo = idx
    while lo-1 >= 0 and same[lo-1][1]: lo -= 1
    hi = idx
    while hi+1 < len(same) and same[hi+1][1]: hi += 1
    rng = (vals[lo], vals[hi]); frac = np.mean([k for _,k,_ in same])
    alt = Counter(e for _,k,e in same if not k)
    sens[par] = dict(range=[float(rng[0]), float(rng[1])], share_same=float(frac),
                     grid=[float(vals[0]), float(vals[-1])], alternatives=dict(alt))
    print(f"  {par:<9} base={base_v:<6} IS se mantiene en [{rng[0]:g}, {rng[1]:g}] "
          f"de la rejilla [{vals[0]:g}, {vals[-1]:g}]  ({frac:.0%} de la rejilla)")
    for e, k in alt.most_common(3): print(f"            alternativa: {e}  ({k} valores)")

# ===================================== TEST 2 — ROBUSTEZ ORIENTADA AL AJUSTE
print("\n" + "="*74); print("TEST 2 — ROBUSTEZ ORIENTADA AL AJUSTE (conjunto de prueba TS)")
print("="*74)
TS_GRID = dict(p_in=[90, 92.5, 95, 97.5], p_cross=[45, 50, 55],
               p_ex=[2.5, 5, 7.5, 10], incl_cut=[0.75, 0.80, 0.85, 0.90],
               pri_cut=[0.65, 0.70, 0.75], n_cut=[1, 2])
TS = []
for combo in itertools.product(*TS_GRID.values()):
    kw = dict(zip(TS_GRID.keys(), combo))
    s = solve(**kw); s["kw"] = kw; TS.append(s)
exprs = Counter(s["expr"] for s in TS)
print(f"  Soluciones plausibles evaluadas: {len(TS)}")
print(f"  Distribución de soluciones en TS:")
for e, k in exprs.most_common(8): print(f"    {k:>4} ({k/len(TS):.0%})  {e}")
# frecuencia de cada término y de GOVMEAS como condición presente
term_freq = Counter(); gov_present = 0
for s in TS:
    for t in s["terms"]: term_freq[term_str(t)] += 1
    if s["terms"] and all(t[2] == 1 for t in s["terms"]): gov_present += 1
print("\n  Frecuencia de términos en TS:")
for t, k in term_freq.most_common(): print(f"    {t:<22} {k/len(TS):.0%}")
print(f"  GOVMEAS presente en TODOS los términos: {gov_present/len(TS):.0%} de TS")

# TS_min / TS_max sobre la calibración de IS (membresías de las soluciones
# alternativas se evalúan en el espacio de la calibración base para que sean comparables)
nonempty = [s for s in TS if s["terms"]]
Ms = np.array([np.max([term_m(F0, t) for t in s["terms"]], axis=0) for s in nonempty])
TSmin, TSmax = Ms.min(0), Ms.max(0)
def sc(a, b): return np.minimum(a,b).sum()/np.maximum(a,b).sum()
fit = dict(IS=dict(cons=incl(IS["m"],y0), cov=cov(IS["m"],y0)),
           TSmin=dict(cons=incl(TSmin,y0), cov=cov(TSmin,y0), SC=sc(IS["m"],TSmin)),
           TSmax=dict(cons=incl(TSmax,y0), cov=cov(TSmax,y0), SC=sc(IS["m"],TSmax)))
print(f"\n  {'':<8}{'inclS':>8}{'covS':>8}{'SC(IS,·)':>10}")
print(f"  {'IS':<8}{fit['IS']['cons']:>8.3f}{fit['IS']['cov']:>8.3f}{'—':>10}")
print(f"  {'TS_min':<8}{fit['TSmin']['cons']:>8.3f}{fit['TSmin']['cov']:>8.3f}{fit['TSmin']['SC']:>10.3f}")
print(f"  {'TS_max':<8}{fit['TSmax']['cons']:>8.3f}{fit['TSmax']['cov']:>8.3f}{fit['TSmax']['SC']:>10.3f}")
print(f"  RF_cons = inclS(IS) − inclS(TS_min) = {fit['IS']['cons']-fit['TSmin']['cons']:+.3f}")
print(f"  RF_cov  = covS(IS) − covS(TS_min)   = {fit['IS']['cov']-fit['TSmin']['cov']:+.3f}")

# ===================================== TEST 3 — ROBUSTEZ ORIENTADA A LOS CASOS
print("\n" + "="*74); print("TEST 3 — ROBUSTEZ ORIENTADA A LOS CASOS")
print("="*74)
typ_IS = (IS["m"] > 0.5) & (y0 > 0.5)
dev_IS = (IS["m"] > 0.5) & (y0 <= 0.5)
typ_min = (TSmin > 0.5) & (y0 > 0.5); typ_max = (TSmax > 0.5) & (y0 > 0.5)
dev_min = (TSmin > 0.5) & (y0 <= 0.5); dev_max = (TSmax > 0.5) & (y0 <= 0.5)
RCR_typ = (typ_IS & typ_min).sum()/typ_IS.sum() if typ_IS.sum() else np.nan
RCR_dev = (dev_IS & dev_min).sum()/dev_IS.sum() if dev_IS.sum() else np.nan
robust_t = typ_IS & typ_min; shaky_t = typ_IS & ~typ_min; possible_t = ~typ_IS & typ_max
print(f"  Casos típicos en IS: {typ_IS.sum()}  → robustos {robust_t.sum()}, "
      f"inestables {shaky_t.sum()}, posibles (solo en TS_max) {possible_t.sum()}")
print(f"  Casos desviados (consistencia) en IS: {dev_IS.sum()}  → robustos {(dev_IS&dev_min).sum()}")
print(f"  RCR_typ = {RCR_typ:.3f}   RCR_dev = {RCR_dev if RCR_dev==RCR_dev else float('nan'):.3f}")
# estabilidad de la membresía de cada caso a lo largo del TS
memb_sd = Ms.std(0); above = (Ms > 0.5).mean(0)
print(f"  Membresía en la solución: σ media entre TS = {memb_sd.mean():.3f}; "
      f"casos con pertenencia >0.5 en ≥95% de TS: {(above>=0.95).sum()}; "
      f"en ≤5%: {(above<=0.05).sum()}; ambiguos: {((above>0.05)&(above<0.95)).sum()}")

# ============================================================ NÚCLEO ROBUSTO
print("\n" + "="*74); print("NÚCLEO ROBUSTO")
print("="*74)
core_expr = Counter()
for s in nonempty:
    for t in s["terms"]: core_expr[term_str(t)] += 1
always = [t for t,k in core_expr.items() if k == len(nonempty)]
gov_core = all(all(t[2]==1 for t in s["terms"]) for s in nonempty)
print(f"  Términos presentes en el 100 % de TS no vacío: {always if always else 'ninguno literal'}")
print(f"  GOVMEAS presente en todos los términos de todas las soluciones: {gov_core}")
print(f"  Soluciones vacías en TS: {len(TS)-len(nonempty)} de {len(TS)}")

json.dump(dict(IS=dict(expr=IS_EXPR, cons=IS['cons'], pri=IS['pri'], cov=IS['cov'],
                       terms=IS['term_stats']),
               sensitivity=sens, TS_n=len(TS),
               TS_dist={e:k for e,k in exprs.most_common()},
               term_freq={t:k/len(TS) for t,k in term_freq.items()},
               gov_in_all_terms_share=gov_present/len(TS), fit=fit,
               cases=dict(typ_IS=int(typ_IS.sum()), robust=int(robust_t.sum()),
                          shaky=int(shaky_t.sum()), possible=int(possible_t.sum()),
                          dev_IS=int(dev_IS.sum()), RCR_typ=float(RCR_typ),
                          RCR_dev=float(RCR_dev) if RCR_dev==RCR_dev else None,
                          stable_in=int((above>=0.95).sum()), stable_out=int((above<=0.05).sum()),
                          ambiguous=int(((above>0.05)&(above<0.95)).sum())),
               robust_core=dict(always=always, gov_core=gov_core)),
          open("fsqca_robustez_resultados.json","w"), indent=2, default=float)
print("\n→ fsqca_robustez_resultados.json")
