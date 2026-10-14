"""
Li delivery with published quiescent accretion rates and the outburst duty cycle,
plus a sensitivity sweep over the uncertain inputs.

Replaces the single-mdot estimate of scripts/bh_spallation_systems.py:
  * quiescent mdot from published ADAF spectral fits (A0620-00: Narayan, McClintock & Yi
    1996, mdot = 4e-4; V404 Cyg: Narayan, Barret & McClintock 1997, Mdot = 1.2e-9 Msun/yr),
    or, where no fit exists, from L_X with an ADAF efficiency calibrated on those fits;
  * hard-state phases of the outburst cycle (hot flow below ~2% L_Edd; Maccarone 2003),
    built from each system's observed outburst history.
Li production per unit time scales as Mdot * y(mdot) with y ~ mdot, so the duty cycle
enters through <mdot y(mdot)>.

Run after scripts/bh_spallation_systems.py:
    PYTHONPATH=. .venv/bin/python scripts/bh_spallation_dutycycle.py
"""
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import bh_spallation as bs
from src.bh_spallation import inflow_yields, classify_central, SpallationConfig, ESCAPED
from src.paperstyle import apply_paper_style, column_figsize, PALETTE
from scripts.bh_spallation_systems import (SYSTEMS, MDOT_EDD_SUN_PER_YR, L_EDD_PER_SUN, TAU_DEP,
                                           R0_FID, R0_RANGE, roche_lobe_over_a, R_SUN_AU)

warnings.filterwarnings("ignore")
OUT = "output/spallation"
FIG = "Paper/BlackHole_Paper/Figures/spallation_lithium.png"
DAY_YR = 1 / 365.25

# ADAF efficiency calibrated on the NMY96 fit of A0620-00 (L_X/L_Edd = 8.5e-9 at mdot = 4e-4)
# and joining the radiatively efficient hard state at the 2% L_Edd state transition:
#     L / L_Edd = 0.02 (mdot / 0.02)^P
MDOT_TR, L_TR = 0.02, 0.02
P_ADAF = np.log(8.5e-9 / L_TR) / np.log(4e-4 / MDOT_TR)


def mdot_of_L(l_edd):
    return MDOT_TR * np.minimum(np.asarray(l_edd) / L_TR, 1.0) ** (1 / P_ADAF)


# Quiescent mdot and outburst hard-state light-curve segments: (L_start/L_Edd, plateau [d], e-fold [d]).
# Each segment is a plateau followed by an exponential decay to quiescence (rises are treated the
# same way with their own e-folding time).
DUTY = {
    # Outbursts 1917, 1975 (57.8 yr); 1975 decay e-fold 29 d (Chen, Shrader & Livio 1997).
    "A0620-00": dict(mdot_q=4e-4, T_rec_yr=57.8,
                     segments=[(L_TR, 0.0, 29.0),      # decline through the hard state
                               (L_TR, 0.0, 3.0)]),     # hard-state part of the rise
    # Hard-state-only outbursts 2000 (1.2e36 erg/s, ~months) and 2005 (2e35 erg/s, <20 d):
    # both counted once per ~25 yr of monitoring (Zurita+2006; McClintock+2001).
    "XTE J1118+480": dict(mdot_q=None, T_rec_yr=25.0,
                          segments=[(1.2e36 / (L_EDD_PER_SUN * 7.5), 100.0, 30.0),
                                    (2e35 / (L_EDD_PER_SUN * 7.5), 5.0, 5.0)]),
    # Outbursts 1938, 1956, 1989, 2015 (mean 25.7 yr); generic 30 d decay.
    # Quiescent fit: 1.22e-9 Msun/yr at M = 12, D = 3.5 kpc (NBM97); rescaled to M = 9, D = 2.39 kpc
    # with Mdot ~ L^(1/P) ~ D^(2/P).
    "V404 Cyg": dict(mdot_q=1.22e-9 * (2.39 / 3.5) ** (2 / P_ADAF) / (MDOT_EDD_SUN_PER_YR * 9.0),
                     T_rec_yr=25.7, segments=[(L_TR, 0.0, 30.0), (L_TR, 0.0, 3.0)]),
}


def yield_table(r0, mdots):
    return np.array([inflow_yields(r0, m)["Li_H"] for m in mdots])


def mean_mdot_y(name, r0, grid, table, quiescent_only=False):
    """Time average of mdot * y_Li(mdot) over quiescence plus the hard-state outburst phases."""
    d = DUTY[name]
    s = SYSTEMS[name]
    mdot_q = d["mdot_q"] if d["mdot_q"] is not None else mdot_of_L(s["L_X"] / (L_EDD_PER_SUN * s["M_bh"]))
    y = lambda m: np.exp(np.interp(np.log(m), np.log(grid), np.log(table)))
    avg = mdot_q * y(mdot_q)
    if quiescent_only:
        return avg, mdot_q
    l_q = s["L_X"] / (L_EDD_PER_SUN * s["M_bh"])
    T = d["T_rec_yr"]
    for l0, plateau, efold in d["segments"]:
        t_dec = efold * np.log(l0 / l_q)
        t = np.linspace(0, t_dec, 4000)
        m = mdot_of_L(l0 * np.exp(-t / efold))
        integral = plateau * mdot_of_L(l0) * y(mdot_of_L(l0)) + np.trapz(m * y(m), t)
        avg += integral * DAY_YR / T
    return avg, mdot_q


def predict(name, r0, f_esc, g, grid, table, quiescent_only=False):
    s = SYSTEMS[name]
    theta = roche_lobe_over_a(s["M2"] / s["M_bh"])
    f_int = f_esc * g * theta ** 2 / 4
    avg, mdot_q = mean_mdot_y(name, r0, grid, table, quiescent_only)
    Mdot_edd = MDOT_EDD_SUN_PER_YR * s["M_bh"]
    return [12 + np.log10(Mdot_edd * avg * f_int * tau / s["M_conv"]) for tau in TAU_DEP], mdot_q, f_int


_ESC_CACHE = {}


def escape_and_g(r0, n=20000, seed=3):
    key = (r0, bs.THETA_ION)
    if key in _ESC_CACHE:
        return _ESC_CACHE[key]
    cfg = SpallationConfig(r0=r0, n_particles=n, seed=seed)
    pos, vel = bs._launch(cfg, np.random.default_rng(seed))
    o = classify_central(pos, vel, "schwarzschild")
    e = o == ESCAPED
    s_lat = vel[e, 2] / np.linalg.norm(vel[e], axis=1)
    _ESC_CACHE[key] = (np.mean(e), np.mean(np.abs(s_lat) < 0.3) / 0.3)
    return _ESC_CACHE[key]


def run_case(label, r0s=(R0_RANGE[0], R0_FID, R0_RANGE[1]), quiescent_only=False):
    grid = np.geomspace(1e-7, MDOT_TR, 40)
    out = {}
    for name in DUTY:
        vals = []
        for r0 in r0s:
            f_esc, g = escape_and_g(r0)
            table = yield_table(r0, grid)
            a, mdot_q, f_int = predict(name, r0, f_esc, g, grid, table, quiescent_only)
            vals.append((r0, a, mdot_q, f_int))
        out[name] = vals
    return out


if __name__ == "__main__":
    print(f"ADAF efficiency slope P = {P_ADAF:.2f}  (L/L_Edd = 0.02 (mdot/0.02)^P)")
    results = {}
    for label, q in (("quiescent (ADAF fit)", True), ("quiescent + outbursts", False)):
        res = run_case(label, quiescent_only=q)
        results[label] = res
        print(f"\n== {label}")
        for name, vals in res.items():
            fid = [v for v in vals if v[0] == R0_FID][0]
            lo = min(min(v[1]) for v in vals)
            hi = max(max(v[1]) for v in vals)
            print(f"  {name:14s} mdot_q={fid[2]:.1e}  f_int={fid[3]:.1e}  A(Li) r0=10: {fid[1][0]:.2f}..{fid[1][1]:.2f}"
                  f"  full [{lo:.2f}, {hi:.2f}]   observed {SYSTEMS[name]['ALi']} +/- {SYSTEMS[name]['ALi_err']}")

    # Sensitivity sweep at r0 = 10, quiescent + outbursts, tau = 1e8 yr (geometric middle).
    print("\n== Sensitivity (r0 = 10 r_s, quiescent + outbursts, tau_dep = 1e8 yr): change in A(Li) [dex]")
    base = run_case("base", r0s=(R0_FID,))
    def mid(res, name):
        return 0.5 * sum(res[name][0][1])
    sweeps = {}
    saved = (bs.THETA_ION, bs.SIGMA_LI)
    bs.THETA_ION = 1.0 / 6.0
    sweeps["ion temperature 0.55 -> 1.0 T_vir"] = run_case("Ti", r0s=(R0_FID,))
    bs.THETA_ION = saved[0]
    bs.SIGMA_LI = 2 * saved[1]
    sweeps["alpha+alpha cross section x2"] = run_case("sig", r0s=(R0_FID,))
    bs.SIGMA_LI = saved[1]
    for name in DUTY:
        line = f"  {name:14s} base A(Li) = {mid(base, name):.2f}"
        for k, res in sweeps.items():
            line += f" | {k}: {mid(res, name) - mid(base, name):+.2f}"
        print(line)
    print("  escape x10 (hot-flow winds instead of one thermal attempt): +1.00 by construction")

    json.dump({k: {n: [[v[0], v[1], v[2], v[3]] for v in vals] for n, vals in r.items()}
               for k, r in results.items()}, open(f"{OUT}/dutycycle_summary.json", "w"), indent=1, default=float)

    # Updated lithium figure.
    summ = json.load(open(f"{OUT}/summary.json"))
    gaia = summ["systems"]["Gaia BH1"]["cases"]["quiescent"]
    apply_paper_style(use_tex=False)
    fig, ax = plt.subplots(figsize=column_figsize(0.75))
    names = list(DUTY) + ["Gaia BH1"]
    for i, name in enumerate(names):
        if name == "Gaia BH1":
            ax.annotate(rf"predicted $A({{\rm Li}}) \approx {gaia['ALi_fid'][1]:.0f}$ (off scale)",
                        (-4, i), xytext=(-3.9, i), fontsize=6, color=PALETTE["blue"], va="center")
            continue
        for j, (label, col) in enumerate((("quiescent (ADAF fit)", PALETTE["blue"]),
                                          ("quiescent + outbursts", PALETTE["green"]))):
            vals = results[label][name]
            lo = min(min(v[1]) for v in vals)
            hi = max(max(v[1]) for v in vals)
            ax.plot([max(lo, -4), hi], [i + 0.14 * (j - 0.5)] * 2, color=col, lw=4, solid_capstyle="butt",
                    label=("predicted, quiescent ADAF" if j == 0 else "predicted, + outburst hard states")
                    if i == 0 else None)
        ax.errorbar(SYSTEMS[name]["ALi"], i, xerr=SYSTEMS[name]["ALi_err"], fmt="*", ms=9, color="k",
                    label="observed" if i == 0 else None, zorder=4)
    ax.axvspan(-4, 0.35, color="0.92", zorder=0)
    ax.text(-3.9, -1.05, "depleted field-star level", fontsize=6, color="0.4")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlim(-4, 6)
    ax.set_ylim(len(names) - 0.5, -1.3)
    ax.set_xlabel(r"Companion surface $A({\rm Li}) = 12 + \log({\rm Li/H})$")
    ax.legend(frameon=False, fontsize=6, loc="upper right", ncol=1)
    fig.tight_layout(); fig.savefig(FIG, dpi=300); plt.close(fig)
    print("saved", FIG)
