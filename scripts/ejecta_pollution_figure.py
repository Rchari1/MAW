"""
Pollution diagram: ejecta mass delivered to the companion (SERPENS, perfect
retention) per unit mixing mass vs the measured oxygen excess in the companion.

Analogue of the accreted-mass / convection-zone-mass diagrams for polluted white
dwarfs. Curves show the excess expected for a retention efficiency f_ret and an
ejecta oxygen mass fraction X_O,ej:

    Delta[O] = log10(1 + f_ret X_O,ej x_cap / X_O,sun),   x_cap = M_cap / M_mix

Run after scripts/ejecta_systems.py:
    PYTHONPATH=. .venv/bin/python scripts/ejecta_pollution_figure.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paperstyle import apply_paper_style, column_figsize, PALETTE

X_O_SUN = 5.79e-3
X_O_EJ = 0.45            # centre of the assumed 0.3-0.6 range
FIG = "Paper/BlackHole_Paper/Figures/ejecta_pollution_diagram.png"

if __name__ == "__main__":
    rows = json.load(open("output/ejecta/systems_summary.json"))
    sn = [r for r in rows if r["energy"] == "SN"]
    hn = {r["system"]: r for r in rows if r["energy"] == "HN"}

    apply_paper_style(use_tex=False)
    fig, ax = plt.subplots(figsize=column_figsize(0.85))

    x = np.geomspace(1e-9, 1, 400)
    for f_ret, ls in ((1.0, "-"), (0.1, "--"), (0.01, ":")):
        ax.plot(x, np.log10(1 + f_ret * X_O_EJ * x / X_O_SUN), color="0.35", ls=ls, lw=1,
                label=rf"$f_{{\rm ret}} = {f_ret:g}$")
        ax.fill_between(x, np.log10(1 + f_ret * 0.3 * x / X_O_SUN), np.log10(1 + f_ret * 0.6 * x / X_O_SUN),
                        color="0.85", lw=0, zorder=0) if f_ret == 1.0 else None
    ax.axhline(0.1, color="0.6", lw=0.6)
    ax.text(2e-9, 0.115, "typical abundance precision", fontsize=6, color="0.4")

    colors = {"LMXB": PALETTE["red"], "Gaia": PALETTE["blue"]}
    for r in sn:
        # Marker at the SN-energy value; bar spans SN-HN. Zero captures (no hits) fall back to HN.
        xs = sorted(v for v in (r["x_cap"], hn[r["system"]]["x_cap"]) if v > 0)
        x0 = r["x_cap"] if r["x_cap"] > 0 else xs[0]
        c = colors[r["kind"]]
        if r.get("M_O_required") is not None and r["kind"] == "LMXB":
            y = {"Nova Sco 1994": 0.91, "V404 Cyg": 0.69}[r["system"]]
            yerr = {"Nova Sco 1994": 0.09, "V404 Cyg": 0.24}[r["system"]]
            ax.errorbar(r["x_cap"], y, yerr=yerr, fmt="o", ms=5, color=c, zorder=3)
            ax.plot(xs, [y] * len(xs), color=c, lw=2, alpha=0.5)
            ax.annotate(r["system"], (r["x_cap"], y), xytext=(-4, 7), textcoords="offset points",
                        fontsize=7, ha="right", color=c)
        elif r["kind"] == "Gaia":
            ax.errorbar(x0, 0.1, yerr=0.04, uplims=True, fmt="v", ms=5, color=c, zorder=3)
            ax.plot(xs, [0.1] * len(xs), color=c, lw=2, alpha=0.5)
            ha = {"Gaia BH2": "right", "Gaia BH3": "left"}.get(r["system"], "center")
            ax.annotate(r["system"].replace("Gaia ", ""), (x0, 0.1), xytext=(0, 9),
                        textcoords="offset points", fontsize=7, ha=ha, color=c)
        else:   # LMXBs without an O measurement: mark the predicted x only
            ax.plot([r["x_cap"]], [0.0], marker="|", ms=10, color=c)
            dy = {"A0620-00": 0.30, "XTE J1118+480": 0.20}[r["system"]]
            ax.annotate(r["system"] + " (no O)", (r["x_cap"], 0.0), xytext=(r["x_cap"], dy),
                        fontsize=6, color=c, ha="right",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.5))

    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1)
    ax.set_ylim(-0.05, 1.6)
    ax.set_xlabel(r"Delivered ejecta / mixing mass, $M_{\rm cap}/M_{\rm mix}$")
    ax.set_ylabel(r"Oxygen excess $\Delta[{\rm O/H}]$ [dex]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    print("saved", FIG)
