#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figuras del artículo construidas ad hoc (las restantes las generan los
scripts de reproducción reproduce_article.py / reproducir_articulo.py), en
español (figures/es/) e inglés (figures/en/):
  Fig. 1  Modelo de componente jerárquico reespecificado
  Fig. 5  Importancia robusta regularizada (cifras canónicas de R)
  Fig. 6  Solución conservadora M1 (matriz de condiciones + coberturas)
  Fig. 9  Síntesis del análisis de robustez (Monte Carlo, ANN, fsQCA)
Uso: python make_figures.py [directorio con los JSON de robustez]
     (por defecto ../robustness/results)
-----------------------------------------------------------------------------
Article figures built ad hoc (the remaining ones are produced by the
reproduction scripts), in Spanish (figures/es/) and English (figures/en/).
Usage: python make_figures.py [directory holding the robustness JSON files]
       (default ../robustness/results)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
C_SA, C_PEO, C_GOV = "#2B579A", "#217346", "#B7472A"
C_BPM, C_BPO = "#7B52AB", "#E8912D"
F_SA, F_PEO, F_GOV = "#EAF0FA", "#E7F3EB", "#FBEAE6"
F_BPM, F_BPO = "#EDE4F5", "#FDEBD3"
GREY = "#c8ccd2"

T = {
 "es": dict(lang="es", out="figures/es",
    f1_title="Modelo de componente jerárquico (HCM) reespecificado",
    f1_sub="Capacidad de BPM: constructo formativo de segundo orden",
    f1_heads=["Fundamento\nteórico","Indicadores observados\nde primer orden",
              "Constructos de\nprimer orden\n(reflectivos)","Constructo de\nsegundo orden\n(formativo)",
              "Relación\nestructural","Indicadores observados\nde BPO"],
    f1_theories=["Teoría de la contingencia\n(Fiedler, 1964)","Capacidades dinámicas\n(Teece, 2007)",
                 "Ajuste tarea–tecnología (TTF)\n(Goodhue & Thompson, 1995)"],
    f1_subs=["Alineación\nestratégica","Capacidades\nhumanas","Gobernanza\ny medición"],
    f1_bpm=("CAPACIDAD BPM","Segundo orden","Formativo"), f1_bpo=("BPO","Resultados","Formativo"),
    f1_rel="Relaciones formativas\nde segundo orden", f1_struct="Modelo\nestructural",
    f1_foot="Etapa 1: scores reflectivos estandarizados (Modo A)  ·  Etapa 2: indicadores (Modo B)  ·  Bootstrap: 3 000 remuestreos",
    f5_title="Importancia robusta regularizada (10 redes × 10 pliegues)",
    f5_ylab="Importancia Garson promediada (%)", f5_eq="Equicontribución (33.3%)",
    f6_title="Solución conservadora  M1 = ~SA·GOVMEAS + PEO·GOVMEAS",
    f6_sub="inclS = 0.968      PRI = 0.936      cobertura = 0.862",
    f6_cov="Cobertura bruta (gris) y única (color)", f6_xlab="Cobertura",
    f6_leg=["presencia de la condición","ausencia de la condición","condición indiferente"],
    f9_a="(a) PLS-SEM: recuperación de parámetros\nvs. tamaño muestral",
    f9_a_lines=["P(GOVMEAS peso dominante)","P(peso de PEO significativo)","Cobertura IC95% de β"],
    f9_a_x="Tamaño muestral (n)", f9_a_y="Proporción de réplicas",
    f9_b="(b) ANN: importancia Garson\nbajo H0 y H1 (n = 56)",
    f9_b_groups=["H0\nejecución\núnica","H0\nprotocolo\nrobusto","H1\nprotocolo\nrobusto"],
    f9_b_y="Importancia (%)",
    f9_c="(c) fsQCA: frecuencia de términos en 1 152 especificaciones",
    f9_c_x="Proporción del conjunto de prueba",
    f9_n_obs="n observado",
 ),
 "en": dict(lang="en", out="figures/en",
    f1_title="Respecified Hierarchical Component Model (HCM)",
    f1_sub="BPM capability: second-order formative construct",
    f1_heads=["Theoretical\nfoundation","First-order\nobserved indicators",
              "First-order constructs\n(reflective)","Second-order construct\n(formative)",
              "Structural\nrelation","BPO observed\nindicators"],
    f1_theories=["Contingency theory\n(Fiedler, 1964)","Dynamic capabilities\n(Teece, 2007)",
                 "Task–technology fit (TTF)\n(Goodhue & Thompson, 1995)"],
    f1_subs=["Strategic\nalignment","Human\ncapabilities","Governance &\nmeasurement"],
    f1_bpm=("BPM CAPABILITY","Second order","Formative"), f1_bpo=("BPO","Outcomes","Formative"),
    f1_rel="Second-order\nformative relations", f1_struct="Structural\nmodel",
    f1_foot="Stage 1: standardized reflective scores (Mode A)  ·  Stage 2: indicators (Mode B)  ·  Bootstrap: 3,000 resamples",
    f5_title="Regularised robust importance (10 networks × 10 folds)",
    f5_ylab="Averaged Garson importance (%)", f5_eq="Equal contribution (33.3%)",
    f6_title="Conservative solution  M1 = ~SA·GOVMEAS + PEO·GOVMEAS",
    f6_sub="inclS = 0.968      PRI = 0.936      coverage = 0.862",
    f6_cov="Raw (grey) and unique (colour) coverage", f6_xlab="Coverage",
    f6_leg=["condition present","condition absent","condition irrelevant"],
    f9_a="(a) PLS-SEM: parameter recovery\nvs. sample size",
    f9_a_lines=["P(GOVMEAS dominant weight)","P(PEO weight significant)","95% CI coverage of β"],
    f9_a_x="Sample size (n)", f9_a_y="Proportion of replications",
    f9_b="(b) ANN: Garson importance\nunder H0 and H1 (n = 56)",
    f9_b_groups=["H0\nsingle\nrun","H0\nrobust\nprotocol","H1\nrobust\nprotocol"],
    f9_b_y="Importance (%)",
    f9_c="(c) fsQCA: term frequency across 1,152 specifications",
    f9_c_x="Share of the test set",
    f9_n_obs="observed n",
 )}

# =============================================================================
def fig1(t):
    fig, ax = plt.subplots(figsize=(15.4, 10.9), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 73); ax.axis("off")
    def box(x, y, w, h, text, ec, fc, fs=10.5, weight="bold", tc="#111111"):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.18,rounding_size=0.5",
                     linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=tc,
                zorder=4, linespacing=1.35)
    def arrow(p0, p1, color="#333333", lw=1.3, ms=9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color,
                     linewidth=lw, shrinkA=1, shrinkB=1, zorder=2))
    def bracket(x, y0, y1):
        ax.plot([x, x], [y0, y1], color="#555555", lw=1.4, zorder=2)
        for yy in (y0, y1): ax.plot([x, x+0.9], [yy, yy], color="#555555", lw=1.4, zorder=2)
    for x, txt in zip([9, 26, 43, 60.5, 76, 91], t["f1_heads"]):
        ax.text(x, 64.6, txt, ha="center", va="center", fontsize=10.6, fontweight="bold",
                color="#222222", linespacing=1.3)
    cys = [57.0, 40.0, 17.0]
    for y, txt, ec, fc in zip(cys, t["f1_theories"], [C_SA, C_PEO, C_GOV], [F_SA, F_PEO, F_GOV]):
        box(9.2, y, 16.8, 6.2, txt, ec, fc, fs=8.9, weight="normal", tc=ec)
    groups = [dict(items=["Ae1","Ae2","Ae3"], cy=57.0, ec=C_SA, fc=F_SA, name="SA"),
              dict(items=["Ep1","Ep2","Ep3","Ep4","Ep5"], cy=40.0, ec=C_PEO, fc=F_PEO, name="PEO"),
              dict(items=["Co1","Co2","Co3","Co4","Co5","Co8","Md1","Md3","Md4"], cy=17.0,
                   ec=C_GOV, fc=F_GOV, name="GOV")]
    STEP = 2.30
    for g, sub in zip(groups, t["f1_subs"]):
        n = len(g["items"]); top = g["cy"] + (n-1)*STEP/2
        ys = [top - k*STEP for k in range(n)]
        for it, yy in zip(g["items"], ys):
            box(26, yy, 10.6, 2.05, it, g["ec"], g["fc"], fs=10)
            arrow((37.3, g["cy"]), (31.4, yy), lw=1.1, ms=8)
        bracket(19.6, ys[-1]-1.0, ys[0]+1.0)
        ax.plot([17.7, 19.6], [g["cy"], g["cy"]], ls=(0,(4,3)), color="#666666", lw=1.2, zorder=1)
        ax.add_patch(Ellipse((43.6, g["cy"]), 12.6, 8.4, linewidth=1.9, edgecolor=g["ec"],
                     facecolor=g["fc"], zorder=3))
        ax.text(43.6, g["cy"]+1.9, g["name"], ha="center", va="center", fontsize=15,
                fontweight="bold", color="#111111", zorder=4)
        ax.text(43.6, g["cy"]-1.6, sub, ha="center", va="center", fontsize=9.2, color="#222222", zorder=4, linespacing=1.25)
    ax.add_patch(Ellipse((62.5, 37.0), 15.4, 12.6, linewidth=2.1, edgecolor=C_BPM, facecolor=F_BPM, zorder=3))
    ax.text(62.5, 39.3, t["f1_bpm"][0], ha="center", va="center", fontsize=12.6, fontweight="bold", zorder=4)
    ax.text(62.5, 36.6, t["f1_bpm"][1], ha="center", va="center", fontsize=10, zorder=4)
    ax.text(62.5, 34.6, t["f1_bpm"][2], ha="center", va="center", fontsize=10, zorder=4)
    arrow((49.4, 54.6), (56.8, 41.6), lw=1.9, ms=13)
    arrow((49.9, 39.4), (54.9, 37.9), lw=2.6, ms=15)
    arrow((49.4, 20.4), (56.6, 32.2), lw=2.6, ms=15)
    ax.text(52.5, 46.4, "w = 0.254 (n.s.)", fontsize=9.4, color=C_SA, fontweight="bold", ha="center", rotation=-60, zorder=5)
    ax.text(52.2, 40.6, "w = 0.420*", fontsize=9.6, color=C_PEO, fontweight="bold", ha="center", rotation=-17, zorder=5)
    ax.text(51.6, 27.4, "w = 0.871***", fontsize=9.6, color=C_GOV, fontweight="bold", ha="center", rotation=39, zorder=5)
    ax.text(58.6, 23.4, t["f1_rel"], fontsize=9.4, color="#333333", ha="center", va="center", linespacing=1.35, zorder=5)
    ax.add_patch(Circle((81.0, 37.0), 6.3, linewidth=2.1, edgecolor=C_BPO, facecolor=F_BPO, zorder=3))
    ax.text(81.0, 38.9, t["f1_bpo"][0], ha="center", va="center", fontsize=15, fontweight="bold", zorder=4)
    ax.text(81.0, 36.5, t["f1_bpo"][1], ha="center", va="center", fontsize=9.8, zorder=4)
    ax.text(81.0, 34.7, t["f1_bpo"][2], ha="center", va="center", fontsize=9.8, zorder=4)
    arrow((70.3, 37.0), (74.5, 37.0), color="#111111", lw=2.6, ms=15)
    ax.text(72.4, 46.4, "β = 0.934***", fontsize=10, color=C_BPM, fontweight="bold", ha="center", zorder=5)
    ax.text(72.4, 44.5, "t = 54.2 · R² = 0.873", fontsize=9.2, color=C_BPM, fontweight="bold", ha="center", zorder=5)
    ax.text(72.4, 31.2, t["f1_struct"], fontsize=9.4, color="#333333", ha="center", va="center", linespacing=1.35, zorder=5)
    for it, yy in zip(["Co7","Ae10","Md7"], [47.0, 37.0, 27.0]):
        box(92.5, yy, 10.2, 3.0, it, C_BPO, F_BPO, fs=10.5)
        arrow((87.4, yy), (85.4, 37.0+(yy-37.0)*0.45), lw=1.2, ms=9)
    ax.add_patch(FancyBboxPatch((17.0, 1.2), 66.0, 3.6, boxstyle="round,pad=0.2,rounding_size=0.6",
                 linewidth=1.2, edgecolor="#999999", facecolor="#FAFAFA", zorder=3))
    ax.text(50.0, 3.0, t["f1_foot"], ha="center", va="center", fontsize=9.0, fontweight="bold", color="#222222", zorder=4)
    ax.text(50, 70.8, t["f1_title"], ha="center", va="center", fontsize=19, fontweight="bold", color="#111111")
    ax.text(50, 68.2, t["f1_sub"], ha="center", va="center", fontsize=12.4, color="#222222")
    fig.savefig(f"{t['out']}/fig1_hcm.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

# =============================================================================
def fig5(t):
    IMP, SD = [29.7, 28.8, 41.4], [0.2, 0.2, 0.2]
    fig, ax = plt.subplots(figsize=(7.9, 4.9), dpi=300); x = np.arange(3)
    ax.bar(x, IMP, 0.62, color=[C_SA, C_PEO, C_GOV], yerr=SD, capsize=7,
           error_kw=dict(elinewidth=2, capthick=2, ecolor="#111111"), zorder=3)
    ax.axhline(33.3, ls="--", color="#888888", lw=1.6, zorder=2, label=t["f5_eq"])
    for i, (m, s) in enumerate(zip(IMP, SD)):
        ax.text(i, m+s+1.1, f"{m:.1f}%\n(±{s:.1f})", ha="center", fontweight="bold", fontsize=11, linespacing=1.3)
    ax.set_xticks(x); ax.set_xticklabels(["SA","PEO","GOVMEAS"], fontsize=11.5)
    ax.set_ylabel(t["f5_ylab"], fontsize=11); ax.set_title(t["f5_title"], fontweight="bold", fontsize=12.5, pad=12)
    ax.set_ylim(0, 48); ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#EDEDED", lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.savefig(f"{t['out']}/fig5_ann_robusto.png" if t["lang"]=="es" else f"{t['out']}/fig5_ann_robust.png",
                dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)

# =============================================================================
def fig6(t):
    COL = {"SA": C_SA, "PEO": C_PEO, "GOVMEAS": C_GOV}
    TERMS = [("T1","~SA·GOVMEAS",{"SA":"absent","PEO":"dc","GOVMEAS":"present"},0.977,0.850,0.453,0.047),
             ("T2","PEO·GOVMEAS",{"SA":"dc","PEO":"present","GOVMEAS":"present"},0.971,0.943,0.815,0.409)]
    CONDS = ["SA","PEO","GOVMEAS"]
    fig = plt.figure(figsize=(11.6, 4.9), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], left=0.235, right=0.905, top=0.735, bottom=0.205, wspace=0.34)
    axL, axR = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    axL.set_xlim(-0.55, 2.55); axL.set_ylim(-0.38, 1.58); axL.set_xticks([]); axL.set_yticks([])
    for s in axL.spines.values(): s.set_visible(False)
    YS = {0: 1.05, 1: 0.10}; R = 0.155
    for i, (tid, expr, states, inclS, PRI, covS, covU) in enumerate(TERMS):
        y = YS[i]
        axL.text(-0.80, y+0.155, tid, transform=axL.get_yaxis_transform(), ha="left", va="center", fontsize=13, fontweight="bold", clip_on=False)
        axL.text(-0.80, y-0.105, expr, transform=axL.get_yaxis_transform(), ha="left", va="center", fontsize=12.5, clip_on=False)
        axL.text(-0.80, y-0.375, f"inclS = {inclS:.3f}   PRI = {PRI:.3f}", transform=axL.get_yaxis_transform(),
                 ha="left", va="center", fontsize=10.5, style="italic", color="#555555", clip_on=False)
        for j, c in enumerate(CONDS):
            st = states[c]
            if st == "dc":
                axL.add_line(Line2D([j-0.17, j+0.17],[y,y], lw=3.2, color="#9aa0a6", solid_capstyle="round"))
            else:
                filled = st == "present"
                axL.add_patch(Circle((j,y), R, facecolor=COL[c] if filled else "white", edgecolor=COL[c], lw=2.4, zorder=3))
                if not filled:
                    d = R*0.66
                    for dx, dy in ((d,d),(d,-d)):
                        axL.add_line(Line2D([j-dx,j+dx],[y-dy,y+dy], lw=2.2, color=COL[c], zorder=4))
    for j, c in enumerate(CONDS):
        axL.text(j, 1.50, c, ha="center", va="center", fontsize=12.5, fontweight="bold", color=COL[c])
    axL.set_aspect("equal", adjustable="box")
    for i, (tid, expr, states, inclS, PRI, covS, covU) in enumerate(TERMS):
        y = YS[i]
        axR.barh(y, covS, height=0.34, color=GREY, zorder=2); axR.barh(y, covU, height=0.34, color=C_GOV, zorder=3)
        axR.text(covS+0.035, y, f"covS = {covS:.3f}\ncovU = {covU:.3f}", va="center", ha="left", fontsize=10.5, color="#333333", linespacing=1.45)
    axR.set_xlim(0, 1.0); axR.set_ylim(-0.38, 1.58); axR.set_yticks([]); axR.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    axR.tick_params(axis="x", labelsize=10.5); axR.set_xlabel(t["f6_xlab"], fontsize=11.5, labelpad=6)
    axR.set_title(t["f6_cov"], fontsize=11.5, pad=11); axR.grid(axis="x", color="#e6e8eb", lw=0.8, zorder=0); axR.set_axisbelow(True)
    for sp in ("top","right"): axR.spines[sp].set_visible(False)
    for sp in ("left","bottom"): axR.spines[sp].set_color("#9aa0a6")
    fig.text(0.5, 0.945, t["f6_title"], ha="center", va="center", fontsize=14.5, fontweight="bold", color="#1a1a1a")
    fig.text(0.5, 0.865, t["f6_sub"], ha="center", va="center", fontsize=12, color="#444444")
    handles = [Line2D([],[], marker="o", ls="none", ms=11, mfc="#555555", mec="#555555", label=t["f6_leg"][0]),
               Line2D([],[], marker="x", ls="none", ms=10, mew=2.2, mec="#555555", label=t["f6_leg"][1]),
               Line2D([],[], marker="_", ls="none", ms=13, mew=3.2, mec="#9aa0a6", label=t["f6_leg"][2])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5,-0.012), handletextpad=0.6, columnspacing=2.4)
    fig.savefig(f"{t['out']}/fig6_solucion.png" if t["lang"]=="es" else f"{t['out']}/fig6_solution.png",
                dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

# =============================================================================
def fig9(t, mc, ann, fs):
    sizes = [30, 40, 56, 80, 100, 150, 200]
    pgov = [mc[str(n)]["p_gov_dominant"] for n in sizes]
    ppeo = [mc[str(n)]["power_w_peo"] for n in sizes]
    covb = [mc[str(n)]["coverage_beta"] for n in sizes]
    fig = plt.figure(figsize=(10.2, 8.6), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], hspace=0.50, wspace=0.36)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, :])
    FS_T, FS_L, FS_K = 12.5, 11.5, 10.5
    # (a)
    ax = axa
    ax.plot(sizes, pgov, "o-", color=C_GOV, lw=2.4, ms=7.5, label=t["f9_a_lines"][0])
    ax.plot(sizes, ppeo, "s-", color=C_PEO, lw=2.4, ms=7.5, label=t["f9_a_lines"][1])
    ax.plot(sizes, covb, "^-", color=C_BPM, lw=2.4, ms=7.5, label=t["f9_a_lines"][2])
    ax.axhline(0.95, ls=":", color="#777", lw=1.2); ax.axvline(56, ls="--", color="#999", lw=1.2)
    ax.text(53.5, 0.415, t["f9_n_obs"]+" = 56", fontsize=FS_K, color="#555", rotation=90, va="bottom", ha="right")
    ax.set_xlabel(t["f9_a_x"], fontsize=FS_L); ax.set_ylabel(t["f9_a_y"], fontsize=FS_L)
    ax.set_ylim(0.40, 1.02); ax.set_xticks([30, 56, 80, 100, 150, 200]); ax.tick_params(labelsize=FS_K)
    ax.legend(fontsize=10, loc="lower right", frameon=True); ax.set_title(t["f9_a"], fontsize=FS_T, fontweight="bold", loc="left")
    ax.grid(color="#EDEDED", lw=0.8); ax.set_axisbelow(True)
    # (b)
    ax = axb
    groups = [ann["H0b_equal_conditional"]["single"], ann["H0b_equal_conditional"]["robust"], ann["H1"]]
    x = np.arange(3); w = 0.26
    for k, (col, nm) in enumerate(zip([C_SA, C_PEO, C_GOV], ["SA","PEO","GOVMEAS"])):
        means = [g["mean"][k] for g in groups]; sds = [g["sd"][k] for g in groups]
        ax.bar(x + (k-1)*w, means, w, yerr=sds, capsize=3.5, color=col, label=nm, zorder=3,
               error_kw=dict(elinewidth=1.3, ecolor="#222"))
    ax.axhline(33.3, ls="--", color="#888", lw=1.3, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(t["f9_b_groups"], fontsize=FS_K, linespacing=1.3)
    ax.set_ylabel(t["f9_b_y"], fontsize=FS_L); ax.set_ylim(0, 58); ax.tick_params(axis="y", labelsize=FS_K)
    ax.legend(fontsize=10, ncol=3, loc="upper left", frameon=False)
    ax.set_title(t["f9_b"], fontsize=FS_T, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#EDEDED", lw=0.8); ax.set_axisbelow(True)
    # (c)
    ax = axc
    tf = sorted(fs["term_freq"].items(), key=lambda kv: kv[1])
    labels = [k.replace("*","·") for k, _ in tf]; vals = [v for _, v in tf]
    cols = [C_GOV if "GOVMEAS" in k else "#9aa0a6" for k, _ in tf]
    ax.barh(labels, vals, color=cols, zorder=3, height=0.62)
    for i, v in enumerate(vals): ax.text(v+0.008, i, f"{v:.0%}", va="center", fontsize=FS_K)
    ax.set_xlim(0, 0.60); ax.set_xlabel(t["f9_c_x"], fontsize=FS_L); ax.tick_params(labelsize=FS_K)
    ax.set_title(t["f9_c"], fontsize=FS_T, fontweight="bold", loc="left")
    ax.grid(axis="x", color="#EDEDED", lw=0.8); ax.set_axisbelow(True)
    for a in (axa, axb, axc):
        for sp in ("top","right"): a.spines[sp].set_visible(False)
    fig.savefig(f"{t['out']}/fig9_robustez.png" if t["lang"]=="es" else f"{t['out']}/fig9_robustness.png",
                dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)

# =============================================================================
if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "robustness", "results")
    mc  = json.load(open(os.path.join(base, "montecarlo_pls_results.json")))
    ann = json.load(open(os.path.join(base, "montecarlo_ann_results.json")))
    fs  = json.load(open(os.path.join(base, "fsqca_robustness_results.json")))
    os.chdir(ROOT)
    for lang, t in T.items():
        os.makedirs(t["out"], exist_ok=True)
        fig1(t); fig5(t); fig6(t); fig9(t, mc, ann, fs)
        print(lang, "→", sorted(os.listdir(t["out"])))
