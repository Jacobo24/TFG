"""
GENERACIÓN DE FIGURAS DEL TFG  (F0–F5)
======================================
Lee los CSV producidos por los experimentos en results/data/ y genera las seis
figuras del cuerpo en results/figures/paper/. No ejecuta simulaciones.

Mapa de datos -> figura:
  F0  data/convergencia/convergencia_band.csv            (run_convergence.py)
  F1  data/demostracion_eta_delta/demostracion_summary.csv (main.py)   [heatmap]
  F2  data/demostracion_eta_delta/demostracion_summary.csv (main.py)   [coop vs delta_pi]
  F3  data/demostracion_eta_delta/demostracion_summary.csv (main.py)   [coop vs eta_s]
  F4  data/real_density_rho/density_rho_raw.csv          (run_real_montecarlo density)
  F5  data/real_hetero_spread_rho/hetero_spread_rho_raw.csv
      data/real_density_rho/density_rho_raw.csv
      data/real_social_eta_rho/social_eta_rho_raw.csv    (los tres canales)

Uso (desde la raíz del proyecto):
    python -u -m experiments.make_figures
    python -u -m experiments.make_figures --results_dir results
"""

from pathlib import Path
import sys
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
C = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def load(path):
    if not Path(path).exists():
        print(f"  [aviso] falta {path} -> se omite la figura que lo usa")
        return None
    return pd.read_csv(path)


# ----------------------------------------------------------------------
def fig_F0(data, out):
    b = load(data / "convergencia/convergencia_band.csv")
    if b is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"cooperativo": "cooperativo", "transicion": "transición", "colapso": "colapso"}
    for k, point in enumerate(["cooperativo", "transicion", "colapso"]):
        d = b[b["point"] == point].sort_values("round")
        if d.empty:
            continue
        ax.plot(d["round"], d["mean_cooperation"], color=C[k], lw=2,
                label=labels.get(point, point))
        ax.fill_between(d["round"], d["mean_cooperation"] - d["ci95"],
                        d["mean_cooperation"] + d["ci95"], color=C[k], alpha=0.2)
    ax.axvline(75, color="gray", ls="--", lw=1)
    ax.text(80, ax.get_ylim()[1] * 0.9, "≈ estado estacionario", color="gray", fontsize=9)
    ax.set_xlabel("ronda"); ax.set_ylabel("proporción de cooperadores")
    ax.set_title("Convergencia al estado estacionario")
    ax.set_ylim(0, None); ax.legend()
    fig.tight_layout(); fig.savefig(out / "F0_convergencia.png"); plt.close()
    print("  F0 OK")


def fig_F1_F2_F3(data, out):
    s = load(data / "demostracion_eta_delta/demostracion_summary.csv")
    if s is None:
        return
    # F1 heatmap
    piv = s.pivot(index="eta_s", columns="delta_pi", values="internal_cooperation")
    nrow, ncol = piv.shape
    fig, ax = plt.subplots(figsize=(1.05 * ncol + 3, 0.85 * nrow + 2))
    vmax = max(0.6, np.nanmax(piv.values))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(ncol)); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(nrow)); ax.set_yticklabels([f"{r:g}" for r in piv.index])
    ax.set_xlabel(r"$\delta_\pi$ (peso del payoff en la revisión)")
    ax.set_ylabel(r"$\eta_s$ (adaptación social)")
    ax.set_title("Cooperación interna media (modelo canónico)")
    for i in range(nrow):
        for j in range(ncol):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 * vmax else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="proporción de cooperadores")
    fig.tight_layout(); fig.savefig(out / "F1_heatmap_eta_delta.png"); plt.close()
    print("  F1 OK")

    # F2 coop vs delta_pi
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, eta in enumerate([0.1, 0.3, 0.6, 1.0]):
        d = s[np.isclose(s.eta_s, eta)].sort_values("delta_pi")
        if d.empty:
            continue
        ax.plot(d["delta_pi"], d["internal_cooperation"], "o-", color=C[k], label=f"η_s={eta}")
        ax.fill_between(d["delta_pi"], d["internal_cooperation"] - d["ci95"],
                        d["internal_cooperation"] + d["ci95"], color=C[k], alpha=0.2)
    ax.set_xlabel(r"$\delta_\pi$ (peso del payoff en la revisión)")
    ax.set_ylabel("proporción de cooperadores")
    ax.set_title("La presión de imitación ($\\delta_\\pi$) hunde la cooperación")
    ax.set_ylim(0, None); ax.legend()
    fig.tight_layout(); fig.savefig(out / "F2_coop_vs_deltapi.png"); plt.close()
    print("  F2 OK")

    # F3 coop vs eta_s (la cresta)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, dpi in enumerate([0.0, 0.1, 0.35]):
        d = s[np.isclose(s.delta_pi, dpi)].sort_values("eta_s")
        if d.empty:
            continue
        ax.plot(d["eta_s"], d["internal_cooperation"], "o-", color=C[k], label=f"δ_π={dpi}")
        ax.fill_between(d["eta_s"], d["internal_cooperation"] - d["ci95"],
                        d["internal_cooperation"] + d["ci95"], color=C[k], alpha=0.2)
    d0 = s[np.isclose(s.delta_pi, 0.0)].sort_values("eta_s")
    if not d0.empty:
        imax = d0["internal_cooperation"].idxmax()
        ax.annotate("óptimo intermedio",
                    xy=(d0.loc[imax, "eta_s"], d0.loc[imax, "internal_cooperation"]),
                    xytext=(0.45, 0.40), arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=9, color="gray")
    ax.set_xlabel(r"$\eta_s$ (adaptación social)")
    ax.set_ylabel("proporción de cooperadores")
    ax.set_title("La adaptación social ($\\eta_s$) y su óptimo intermedio")
    ax.set_ylim(0, None); ax.legend()
    fig.tight_layout(); fig.savefig(out / "F3_coop_vs_etas.png"); plt.close()
    print("  F3 OK")


def fig_F4(data, out):
    dr = load(data / "real_density_rho/density_rho_raw.csv")
    if dr is None:
        return
    g = dr[np.isclose(dr.rho, 0.4)].groupby("base_prob").agg(
        deg=("avg_degree", "mean"), stc=("std_climate", "mean"),
        stb=("std_b_eff", "mean")).reset_index().sort_values("deg")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g["deg"], g["stc"], "o-", color=C[0], label="std(clima) entre comunidades")
    ax.plot(g["deg"], g["stb"], "s-", color=C[1], label="std($b_{eff}$) entre comunidades")
    ax.set_xlabel("grado medio de la meta-red")
    ax.set_ylabel("desviación estándar entre comunidades")
    ax.set_title("Una meta-red sparse hace variar el clima (y se propaga)")
    ax.invert_xaxis()
    ax.text(0.02, 0.95, "← más sparse", transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_ylim(0, None); ax.legend()
    fig.tight_layout(); fig.savefig(out / "F4_std_clima_vs_grado.png"); plt.close()
    print("  F4 OK")


def fig_F5(data, out):
    def curva(path, filtcol, filtval, xcol):
        r = load(path)
        if r is None:
            return None
        r = r[np.isclose(r[filtcol], filtval)]
        g = r.groupby(xcol)["corr_climate_cooperation"].agg(["mean", "std", "count"]).reset_index()
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
        return g.sort_values(xcol)

    series = [
        (data / "real_hetero_spread_rho/hetero_spread_rho_raw.csv", "spread", 1.0, "rho",
         "canal b · red densa (heterogénea)", C[0]),
        (data / "real_density_rho/density_rho_raw.csv", "base_prob", 0.02, "rho",
         "canal b · red sparse", C[1]),
        (data / "real_social_eta_rho/social_eta_rho_raw.csv", "eta_s", 0.6, "rho_s",
         "canal social (η_s=0.6)", C[2]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = False
    for path, fc, fv, xc, lab, col in series:
        g = curva(path, fc, fv, xc)
        if g is None or g.empty:
            continue
        ax.plot(g[xc], g["mean"], "o-", color=col, label=lab)
        ax.fill_between(g[xc], g["mean"] - g["ci"], g["mean"] + g["ci"], color=col, alpha=0.2)
        plotted = True
    if not plotted:
        plt.close(); return
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel(r"intensidad del acoplamiento ($\rho$ / $\rho_s$)")
    ax.set_ylabel(r"corr(clima vecinal, cooperación)")
    ax.set_title("El acoplamiento no mueve la cooperación por ningún canal")
    ax.set_ylim(-0.2, 0.2); ax.legend()
    fig.tight_layout(); fig.savefig(out / "F5_sintesis_corr_clima.png"); plt.close()
    print("  F5 OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    data = Path(args.results_dir) / "data"
    out = Path(args.results_dir) / "figures" / "paper"
    out.mkdir(parents=True, exist_ok=True)

    print("GENERANDO FIGURAS F0–F5")
    fig_F0(data, out)
    fig_F1_F2_F3(data, out)
    fig_F4(data, out)
    fig_F5(data, out)
    print(f"figuras en: {out}")


if __name__ == "__main__":
    main()