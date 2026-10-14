"""
Steps 1-4 of the black-hole spallation analogue of Doyle, Desch & Young (2021).

  1-2. Fate of ions evaporated (SERPENS thermal mode) from the hot flow at r0:
       exact Schwarzschild classification vs SERPENS Monte Carlo (Newtonian, 1PN).
  3.   Fast-ion fluence and Be/O, Li/H yields per parcel vs accretion rate.
  4.   Li delivered to the companion and predicted surface A(Li), compared with the
       measured Li in black-hole X-ray binary companions, and the intercepted fraction
       compared with the 1e-6 - 1e-4 required by Yi & Narayan (1997).

Run: PYTHONPATH=. .venv/bin/python scripts/bh_spallation_systems.py
"""
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import bh_spallation as bs
from src.bh_spallation import (SpallationConfig, SpallationResult, run_spallation, classify_central,
                               inflow_yields, inflow_speed, PLUNGE, RETAINED, ESCAPED, O_TO_H, C_CGS)
from src.paperstyle import apply_paper_style, column_figsize, fullpage_figsize, PALETTE

warnings.filterwarnings("ignore")
OUT = "output/spallation"
FIGDIR = "Paper/BlackHole_Paper/Figures"
R_SUN_AU = 0.0046524726
MDOT_EDD_SUN_PER_YR = 2.217e-8          # per Msun of BH, eta = 0.1
L_EDD_PER_SUN = 1.26e38                 # erg/s
TAU_DEP = (1e7, 1e9)                    # Li depletion time in the companion [yr] (Yi & Narayan 1997)
R0_FID, R0_RANGE = 10.0, (5.0, 20.0)

# Systems. a from Kepler's third law; LMXB donors fill their Roche lobes.
SYSTEMS = {
    # Gonzalez Hernandez+2004 (A(Li) = 2.31 +/- 0.21; field stars of same mass <= 0.35);
    # L_X (2-10 keV) = 7.1e30 erg/s (arXiv:1510.07392); Mdot_2 ~ 1e-10 Msun/yr (McClintock+1995).
    "A0620-00": dict(M_bh=6.6, M2=0.40, P_d=0.3230, L_X=7.1e30, Mdot_tr=1e-10, M_conv=0.15,
                     ALi=2.31, ALi_err=0.21),
    # Gonzalez Hernandez+2008: A(Li)_NLTE = 1.78 +/- 0.25, quoted as an UPPER LIMIT (weak line);
    # q = 0.027 -> M2 = 0.2. F_X = 1.2e-14 cgs at 1.7 kpc (McClintock+2003); fully convective M dwarf.
    "XTE J1118+480": dict(M_bh=7.5, M2=0.20, P_d=0.1699, L_X=4.1e30, Mdot_tr=1e-10, M_conv=0.20,
                          ALi=1.78, ALi_err=0.25, ALi_upper=True),
    # A(Li) = 2.70 +/- 0.40 (Martin+1996, as compiled in Casares+2007, Table 5).
    # L_X ~ 1e33 (d/4 kpc)^2 -> 4e32 erg/s at 2.39 kpc (Bradley+2007).
    "V404 Cyg": dict(M_bh=9.0, M2=0.54, P_d=6.4714, L_X=4e32, Mdot_tr=1e-9, M_conv=0.30,
                     ALi=2.70, ALi_err=0.40),
    # El-Badry+2023 (a = 1.40 AU, R2 ~ 1 Rsun, detached); Mdot from the BHL capture of this paper
    # (~1e5 kg/s); L_X < 9.4e28 erg/s (Rodriguez+2024). No Li anomaly reported.
    "Gaia BH1": dict(M_bh=9.62, M2=0.93, a_AU=1.40, R2=1.0, mdot=7e-12, M_conv=0.03,
                     ALi=None),
    # El-Badry+2023b: 8.94 + 1.07 Msun, a = 4.96 AU, e = 0.518, giant with R = 7.77 Rsun;
    # horizon accretion rate ~2e-9 Mdot_Edd (their eq. 18). Li not detected (post first dredge-up).
    "Gaia BH2": dict(M_bh=8.94, M2=1.07, a_AU=4.96, R2=7.77, mdot=2e-9, M_conv=0.8,
                     ALi=None),
    # Gaia Collaboration (Panuzzo+2024): 32.7 + 0.76 Msun, a1 = 16.17 AU -> relative a = 16.5 AU,
    # R = 4.94 Rsun giant. Cappelluti+2024 wind models: 1e-9 < mdot < 1e-7; we take 1e-8.
    "Gaia BH3": dict(M_bh=32.7, M2=0.76, a_AU=16.5, R2=4.94, mdot=1e-8, M_conv=0.5,
                     ALi=None),
}


def roche_lobe_over_a(q):
    q13 = q ** (1 / 3)
    return 0.49 * q13 ** 2 / (0.6 * q13 ** 2 + np.log(1 + q13))


def mdot_from_LX(L_X, M_bh):
    """Invert L = eta(mdot) mdot Mdot_Edd c^2 with eta = 0.1 (mdot / 0.01) (src/bh_observables.py)."""
    return np.sqrt(L_X / (100 * L_EDD_PER_SUN * M_bh))


def steps_1_2():
    """Outcome fractions vs launch radius: exact GR, analytic Newtonian, and SERPENS MC."""
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for r0 in (4, 5, 6, 8, 10, 15, 20, 30, 50):
        cfg = SpallationConfig(r0=r0, n_particles=20000, seed=int(r0))
        pos, vel = bs._launch(cfg, np.random.default_rng(cfg.seed))
        row = {"r0": r0}
        for metric in ("schwarzschild", "newtonian"):
            o = classify_central(pos, vel, metric)
            row[metric] = {n: float(np.mean(o == c)) for c, n in bs.OUTCOME_NAMES.items()}
        # Escaper latitude distribution (for the companion's solid angle) from launch directions.
        o = classify_central(pos, vel, "schwarzschild")
        e = o == ESCAPED
        s_lat = vel[e, 2] / np.linalg.norm(vel[e], axis=1)
        row["g_equatorial"] = float(np.mean(np.abs(s_lat) < 0.3) / 0.3) if e.any() else np.nan
        for gr in (False, True):
            # MC: Newtonian at all sampled radii; 1PN only from 10 r_s outward (it does not
            # converge for v ~ c closer in, which is itself the point of the comparison).
            if r0 not in (4, 6, 10, 20, 50) or (gr and r0 < 10):
                continue
            tag = f"mc_r{r0}_{'1pn' if gr else 'newt'}.npz"
            path = os.path.join(OUT, tag)
            if os.path.exists(path):
                res = SpallationResult.load(path)
            else:
                res = run_spallation(SpallationConfig(r0=r0, gr=gr, n_particles=3000, batch_size=150), verbose=False)
                res.save(path)
            key = "mc_1pn" if gr else "mc_newtonian"
            n = len(res.outcome)
            row[key] = {nm: float(res.fraction(c)) for c, nm in bs.OUTCOME_NAMES.items()}
            row[key + "_esc_err"] = float(np.sqrt(max(np.sum(res.escaped()), 1)) / n)
            if not gr and res.escaped().any():
                row["path_to_inflow_Fp"] = float(np.median(res.F_p_path[res.escaped()]) / res.F_p_inflow)
        rows.append(row)
        print(flush=True)
        print(f"r0={r0:3d}  exact GR esc={row['schwarzschild']['escaped']:.4f} plunge={row['schwarzschild']['plunge']:.3f}"
              f"  Newt esc={row['newtonian']['escaped']:.4f}"
              + (f"  MC-Newt esc={row['mc_newtonian']['escaped']:.4f}  MC-1PN esc={row['mc_1pn']['escaped']:.4f}"
                 if "mc_1pn" in row else "") + f"  g={row['g_equatorial']:.2f}")
    return rows


def step_3():
    mdots = np.geomspace(1e-12, 1e-2, 41)
    rows = {r0: [inflow_yields(r0, m) for m in mdots] for r0 in (R0_RANGE[0], R0_FID, R0_RANGE[1])}
    # Inflow (exposure) time from 1e3 r_s to r0 for a 7 Msun black hole.
    x = np.geomspace(R0_FID, 1e3, 2000)
    t_in = np.trapz(7 * 2.953e5 / (inflow_speed(x) * C_CGS), x)
    return mdots, rows, t_in


def step_4(outcomes):
    by_r0 = {r["r0"]: r for r in outcomes}
    res = {}
    for name, s in SYSTEMS.items():
        if "P_d" in s:
            a_AU = ((s["M_bh"] + s["M2"]) * (s["P_d"] / 365.25) ** 2) ** (1 / 3)
            theta = roche_lobe_over_a(s["M2"] / s["M_bh"])
            mdots = {"quiescent": mdot_from_LX(s["L_X"], s["M_bh"]),
                     "time-averaged": s["Mdot_tr"] / (MDOT_EDD_SUN_PER_YR * s["M_bh"])}
        else:
            a_AU = s["a_AU"]
            theta = s["R2"] * R_SUN_AU / a_AU
            mdots = {"quiescent": s["mdot"]}
        out = {"a_AU": a_AU, "theta": theta, "cases": {}}
        for label, mdot in mdots.items():
            preds, f_ints = [], []
            for r0 in (R0_RANGE[0], R0_FID, R0_RANGE[1]):
                row = by_r0[int(r0)]
                f_esc = row["schwarzschild"]["escaped"]
                f_geo = row["g_equatorial"] * theta ** 2 / 4
                f_int = f_esc * f_geo
                y = inflow_yields(r0, mdot)
                Mdot = mdot * MDOT_EDD_SUN_PER_YR * s["M_bh"]           # Msun/yr
                for tau in TAU_DEP:
                    li_h = Mdot * f_int * y["Li_H"] * tau / s["M_conv"]
                    be_h = Mdot * f_int * y["Be_O0"] * O_TO_H * tau / s["M_conv"]
                    preds.append((r0, tau, 12 + np.log10(li_h), 12 + np.log10(be_h)))
                f_ints.append(f_int)
            ALi = np.array([p[2] for p in preds])
            fid = [p for p in preds if p[0] == R0_FID]
            case = {"mdot": mdot, "f_int": f_ints, "ALi_range": [float(ALi.min()), float(ALi.max())],
                    "ALi_fid": [float(fid[0][2]), float(fid[1][2])],
                    "ABe_fid": [float(fid[0][3]), float(fid[1][3])]}
            if s["ALi"] is not None:
                # Intercepted fraction needed to reach the observed A(Li) (Yi & Narayan's free parameter).
                y = inflow_yields(R0_FID, mdot)
                Mdot = mdot * MDOT_EDD_SUN_PER_YR * s["M_bh"]
                case["f_int_required"] = [float(10 ** (s["ALi"] - 12) * s["M_conv"] / (Mdot * y["Li_H"] * tau))
                                          for tau in TAU_DEP]
            out["cases"][label] = case
        res[name] = out
    return res


def figures(outcomes, mdots, yields, t_in, systems):
    apply_paper_style(use_tex=False)

    # Figure 1: fate of evaporated ions vs launch radius.
    fig, ax = plt.subplots(figsize=column_figsize(0.8))
    r0 = np.array([r["r0"] for r in outcomes])
    ax.plot(r0, [r["schwarzschild"]["escaped"] for r in outcomes], "-", color="k", lw=1.4, label="Exact Schwarzschild")
    ax.plot(r0, [r["newtonian"]["escaped"] for r in outcomes], "--", color="0.5", lw=1, label="Newtonian (analytic)")
    mc = [r for r in outcomes if "mc_newtonian" in r]
    ax.errorbar([r["r0"] for r in mc], [r["mc_newtonian"]["escaped"] for r in mc],
                yerr=[r["mc_newtonian_esc_err"] for r in mc], fmt="o", ms=4, color=PALETTE["blue"],
                label="SERPENS MC, Newtonian")
    mc = [r for r in outcomes if "mc_1pn" in r]
    ax.errorbar([r["r0"] for r in mc], [r["mc_1pn"]["escaped"] for r in mc],
                yerr=[r["mc_1pn_esc_err"] for r in mc], fmt="s", ms=4, color=PALETTE["red"],
                label="SERPENS MC, REBOUNDx 1PN")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Launch radius $r_0 / r_s$")
    ax.set_ylabel("Fraction of evaporated ions escaping")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/spallation_escape.png", dpi=300); plt.close(fig)

    # Figure 2: Doyle-style yields per parcel vs accretion rate.
    fig, ax = plt.subplots(figsize=column_figsize(0.8))
    for r0, ls in ((R0_RANGE[0], ":"), (R0_FID, "-"), (R0_RANGE[1], "--")):
        ax.plot(mdots, [y["Be_O0"] for y in yields[r0]], ls, color=PALETTE["purple"], lw=1.2,
                label=rf"Be/O, $r_0={r0:g}\,r_s$")
        ax.plot(mdots, [y["Li_H"] for y in yields[r0]], ls, color=PALETTE["orange"], lw=1.2,
                label=rf"Li/H, $r_0={r0:g}\,r_s$")
    ax.axhline(1e-5, color="0.4", lw=0.8)
    ax.text(1.5e-12, 1.6e-5, "Be/O needed for GALEX J2339 (Doyle+21)", fontsize=6, color="0.3")
    for name, s in systems.items():
        for label, c in s["cases"].items():
            ax.axvline(c["mdot"], color=PALETTE["blue"] if label == "quiescent" else PALETTE["green"],
                       lw=0.5, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Accretion rate $\dot m = \dot M / \dot M_{\rm Edd}$")
    ax.set_ylabel("Spallation yield per parcel")
    ax.set_ylim(1e-16, 1e-1)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/spallation_yields.png", dpi=300); plt.close(fig)

    # Figure 3: predicted vs observed companion A(Li).
    fig, ax = plt.subplots(figsize=column_figsize(0.75))
    names = list(systems)
    for i, name in enumerate(names):
        s = systems[name]
        for j, (label, col) in enumerate((("quiescent", PALETTE["blue"]), ("time-averaged", PALETTE["green"]))):
            if label not in s["cases"]:
                continue
            lo, hi = s["cases"][label]["ALi_range"]
            ax.plot([max(lo, -6), hi], [i + 0.12 * (j - 0.5)] * 2, color=col, lw=4, solid_capstyle="butt",
                    label=(r"predicted, quiescent $\dot m$" if label == "quiescent" else r"predicted, time-averaged $\dot m$")
                    if i == 0 else None)
        if s["cases"]["quiescent"]["ALi_range"][1] < -6:
            ax.annotate(rf"predicted $A({{\rm Li}}) \approx {s['cases']['quiescent']['ALi_fid'][1]:.0f}$ (off scale)",
                        (-6, i), xytext=(-5.8, i + 0.05), fontsize=6, color=PALETTE["blue"], va="center")
        if SYSTEMS[name]["ALi"] is not None:
            ax.errorbar(SYSTEMS[name]["ALi"], i, xerr=SYSTEMS[name]["ALi_err"], fmt="*", ms=9, color="k",
                        label="observed" if i == 0 else None, zorder=4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlim(-6, 5)
    ax.axvspan(-6, 0.35, color="0.92", zorder=0)
    ax.text(-5.8, len(names) - 0.6, "depleted field-star level", fontsize=6, color="0.4")
    ax.set_xlabel(r"Companion surface $A({\rm Li}) = 12 + \log({\rm Li/H})$")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/spallation_lithium.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    print("== Steps 1-2: fate of evaporated hot-flow ions")
    outcomes = steps_1_2()
    print("\n== Step 3: yields")
    mdots, yields, t_in = step_3()
    print(f"inflow exposure time 1e3 -> 10 r_s (7 Msun): {t_in:.2f} s")
    print("\n== Step 4: systems")
    systems = step_4(outcomes)
    for name, s in systems.items():
        print(f"{name}: a = {s['a_AU']:.3f} AU, theta = R2/a = {s['theta']:.3g}")
        for label, c in s["cases"].items():
            req = c.get("f_int_required")
            print(f"   {label:13s} mdot={c['mdot']:.2e}  f_int={c['f_int'][1]:.2e} [{min(c['f_int']):.1e}-{max(c['f_int']):.1e}]"
                  f"  A(Li)_pred(r0=10; tau=1e7,1e9) = {c['ALi_fid'][0]:.2f}, {c['ALi_fid'][1]:.2f}"
                  f"  full range [{c['ALi_range'][0]:.2f}, {c['ALi_range'][1]:.2f}]"
                  f"  A(Be)_pred = {c['ABe_fid'][0]:.2f}, {c['ABe_fid'][1]:.2f}"
                  + (f"  f_int needed = {req[1]:.1e}-{req[0]:.1e}" if req else ""))
    json.dump({"outcomes": outcomes, "systems": systems, "t_inflow_s": t_in},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    figures(outcomes, mdots, yields, t_in, systems)
    print("figures saved")
