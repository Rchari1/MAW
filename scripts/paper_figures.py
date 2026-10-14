"""
Figures and headline numbers for the spallation paper.

  spallation_trajectories.png  SERPENS ions evaporated from the hot flow (plunge vs escape)
  spallation_escape.png        escape fraction vs launch radius: exact GR, Newtonian, 1PN
  spallation_yields.png        Be/O and Li/H per parcel vs accretion rate (Doyle et al. analogue)
  spallation_vs_observed.png   MAIN: predicted vs observed A(Li); delivered vs required fraction

Run after scripts/bh_spallation_systems.py and scripts/bh_spallation_dutycycle.py:
    PYTHONPATH=. .venv/bin/python scripts/paper_figures.py
"""
import json
import warnings

import numpy as np
import rebound
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src import bh_spallation as bs
from src.bh_spallation import inflow_yields, SpallationConfig
from src.paperstyle import apply_paper_style, column_figsize, fullpage_figsize, PALETTE
from scripts.bh_spallation_systems import SYSTEMS, MDOT_EDD_SUN_PER_YR, TAU_DEP, roche_lobe_over_a, R_SUN_AU
from scripts.bh_spallation_dutycycle import DUTY, mean_mdot_y, escape_and_g, yield_table, MDOT_TR

warnings.filterwarnings("ignore")
OUT = "output/spallation"
FIGDIR = "Paper/BlackHole_Paper/Figures"
R0S = (5.0, 10.0, 20.0)
C_Q, C_OB, C_SIM, C_GREY = PALETTE["blue"], PALETTE["green"], "black", "0.55"
SYS_MARK = {"A0620-00": "o", "XTE J1118+480": "s", "V404 Cyg": "D", "Gaia BH1": "^", "Gaia BH2": "v",
            "Gaia BH3": "<"}
GAIA = ("Gaia BH1", "Gaia BH2", "Gaia BH3")


def style():
    apply_paper_style(use_tex=False)
    matplotlib.rcParams.update({
        "mathtext.fontset": "cm", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.4, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    })


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def compute_numbers():
    grid = np.geomspace(1e-7, MDOT_TR, 40)
    tables = {r0: yield_table(r0, grid) for r0 in R0S}
    esc = {r0: escape_and_g(r0) for r0 in R0S}
    out = {}
    for name in DUTY:
        s = SYSTEMS[name]
        theta = roche_lobe_over_a(s["M2"] / s["M_bh"])
        Mdot_edd = MDOT_EDD_SUN_PER_YR * s["M_bh"]
        rec = {"theta": theta, "cases": {}}
        for case, q in (("quiescent", True), ("outbursts", False)):
            c = {}
            for r0 in R0S:
                f_esc, g = esc[r0]
                f_int = f_esc * g * theta ** 2 / 4
                avg, mdot_q = mean_mdot_y(name, r0, grid, tables[r0], quiescent_only=q)
                ali = [12 + np.log10(Mdot_edd * avg * f_int * tau / s["M_conv"]) for tau in TAU_DEP]
                f_req = [10 ** (s["ALi"] - 12) * s["M_conv"] / (Mdot_edd * avg * tau) for tau in TAU_DEP]
                c[r0] = {"ALi": ali, "f_int": f_int, "f_req": f_req, "mdot_q": mdot_q, "mean_mdot_y": avg}
            rec["cases"][case] = c
        # Beryllium (p + O; Doyle et al. 2021) through the same duty cycle at r0 = 10 r_s.
        be_table = np.array([inflow_yields(10.0, m)["Be_O0"] for m in grid])
        avg_be, _ = mean_mdot_y(name, 10.0, grid, be_table)
        avg_li = rec["cases"]["outbursts"][10.0]["mean_mdot_y"]
        offset = np.log10(avg_be * bs.O_TO_H / avg_li)
        rec["beryllium"] = {"ABe_minus_ALi": offset, "ABe_at_observed_Li": s["ALi"] + offset}
        # Launch-radius sensitivity at tau_dep = 1e8 yr (geometric middle of the range).
        mid = {r0: float(np.mean(rec["cases"]["outbursts"][r0]["ALi"])) for r0 in R0S}
        rec["r0_sensitivity"] = {"5": mid[5.0] - mid[10.0], "20": mid[20.0] - mid[10.0]}
        out[name] = rec
    # Gaia BHs: detached, wind-fed (no outbursts).
    for name in GAIA:
        s = SYSTEMS[name]
        theta = s["R2"] * R_SUN_AU / s["a_AU"]
        f_esc, g = esc[10.0]
        y = inflow_yields(10.0, s["mdot"])
        f_int = f_esc * g * theta ** 2 / 4
        ali = [12 + np.log10(MDOT_EDD_SUN_PER_YR * s["M_bh"] * s["mdot"] * y["Li_H"] * f_int * tau / s["M_conv"])
               for tau in TAU_DEP]
        out[name] = {"theta": theta, "f_int": f_int, "ALi": ali, "mdot": s["mdot"]}
    out["escape"] = {str(r0): {"f_esc": esc[r0][0], "g": esc[r0][1]} for r0 in R0S}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def trajectory_paths(n=1200, r0=10.0, seed=7):
    """
    Integrate a sample of evaporated ions with SERPENS and record their paths.

    Returns paths (n, n_times, 3) in r_s and fate (0 crossed the ISCO, 1 escaped, 2 still bound).
    """
    cfg = SpallationConfig(r0=r0, n_particles=n, seed=seed)
    pos, vel = bs._launch(cfg, np.random.default_rng(seed))
    times = np.concatenate([[0.0], np.geomspace(0.2, 600.0, 400)])
    paths = np.full((n, len(times), 3), np.nan)
    fate = np.zeros(n, int)                          # 0 plunge, 1 escape, 2 other
    for start in range(0, n, 300):
        sim = rebound.Simulation()
        sim.G = 0.5
        sim.integrator = "ias15"
        sim.collision = "line"
        sim.add(m=1.0, r=3.0)
        sim.N_active = 1
        ids = range(start, min(start + 300, n))
        for i in ids:
            sim.add(x=pos[i, 0], y=pos[i, 1], z=pos[i, 2], vx=vel[i, 0], vy=vel[i, 1], vz=vel[i, 2], hash=i + 1)
        sim.collision_resolve = "merge"
        for k, t in enumerate(times):
            sim.integrate(t, exact_finish_time=1)
            for p in sim.particles[1:]:
                paths[p.hash.value - 1, k] = p.xyz
        for p in sim.particles[1:]:
            i = p.hash.value - 1
            r = np.linalg.norm(p.xyz)
            v2 = p.vx ** 2 + p.vy ** 2 + p.vz ** 2
            fate[i] = 1 if 0.5 * v2 - 0.5 / r > 0 else 2
    # Particles absent at the end merged into the BH (crossed the ISCO).
    return paths, fate


def fig_trajectories(n=1200, r0=10.0, seed=7):
    """Figure 1: SERPENS ions evaporated from the hot flow."""
    paths, fate = trajectory_paths(n, r0, seed)
    rng = np.random.default_rng(1)
    plunge = np.where(fate == 0)[0]
    show = rng.choice(plunge, size=min(120, len(plunge)), replace=False)

    fig, ax = plt.subplots(figsize=column_figsize(0.92))
    L = 45
    for i in show:
        ax.plot(paths[i, :, 0], paths[i, :, 1], color=C_GREY, lw=0.35, alpha=0.55, zorder=1)
    for i in np.where(fate == 1)[0]:
        ax.plot(paths[i, :, 0], paths[i, :, 1], color=PALETTE["red"], lw=0.9, zorder=3)
    ax.add_patch(Circle((0, 0), 1.0, color="black", zorder=5))
    ax.add_patch(Circle((0, 0), 3.0, fill=False, ls="--", lw=0.8, ec=PALETTE["blue"], zorder=5))
    ax.add_patch(Circle((0, 0), r0, fill=False, ls=":", lw=0.8, ec="black", zorder=5))
    # Legend proxies (no text placed on the data).
    ax.plot([], [], color=C_GREY, lw=0.8, label=f"plunges past ISCO ({len(plunge)}/{n})")
    ax.plot([], [], color=PALETTE["red"], lw=0.9, label=f"escapes ({np.sum(fate == 1)}/{n})")
    ax.plot([], [], ls="--", color=PALETTE["blue"], lw=0.8, label=r"ISCO, $3\,r_s$")
    ax.plot([], [], ls=":", color="black", lw=0.8, label=rf"launch sphere, ${r0:g}\,r_s$")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.set_xlabel(r"$x\ [r_s]$"); ax.set_ylabel(r"$y\ [r_s]$")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")
    fig.savefig(f"{FIGDIR}/spallation_trajectories.png", dpi=300); plt.close(fig)
    return int(np.sum(fate == 1)), int(len(plunge)), n


def fig_escape():
    summ = json.load(open(f"{OUT}/summary.json"))["outcomes"]
    r0 = np.array([r["r0"] for r in summ])
    fig, ax = plt.subplots(figsize=column_figsize(0.72))
    ax.plot(r0, [100 * r["schwarzschild"]["escaped"] for r in summ], "-", color="black", lw=1.2,
            label="exact Schwarzschild (analytic)")
    ax.plot(r0, [100 * r["newtonian"]["escaped"] for r in summ], "--", color="0.6", lw=1.0,
            label="Newtonian (analytic)")
    mc = [r for r in summ if "mc_newtonian" in r]
    ax.errorbar([r["r0"] for r in mc], [100 * r["mc_newtonian"]["escaped"] for r in mc],
                yerr=[100 * r["mc_newtonian_esc_err"] for r in mc], fmt="o", ms=3.5, color=PALETTE["blue"],
                label="SERPENS, Newtonian")
    mc = [r for r in summ if "mc_1pn" in r]
    ax.errorbar([r["r0"] for r in mc], [100 * r["mc_1pn"]["escaped"] for r in mc],
                yerr=[100 * r["mc_1pn_esc_err"] for r in mc], fmt="s", ms=3.5, color=PALETTE["red"],
                label="SERPENS, REBOUNDx 1PN")
    ax.set_xscale("log")
    ax.set_xticks([4, 5, 10, 20, 50]); ax.set_xticklabels(["4", "5", "10", "20", "50"])
    ax.set_ylim(0, 7.5)
    ax.set_xlabel(r"launch radius $r_0\ [r_s]$")
    ax.set_ylabel("ions escaping [%]")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(f"{FIGDIR}/spallation_escape.png", dpi=300); plt.close(fig)


def fig_yields(nums):
    mdots = np.geomspace(1e-12, 1e-2, 50)
    Y = {r0: [inflow_yields(r0, m) for m in mdots] for r0 in R0S}
    fig, ax = plt.subplots(figsize=column_figsize(1.05))
    fig.subplots_adjust(bottom=0.36, top=0.97, left=0.17, right=0.96)
    for key, col, ls, lab in (("Be_O0", PALETTE["purple"], "-", "Be/O"), ("Li_H", PALETTE["orange"], "--", "Li/H")):
        lo = [y[key] for y in Y[20.0]]
        hi = [y[key] for y in Y[5.0]]
        ax.fill_between(mdots, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(mdots, [y[key] for y in Y[10.0]], ls, color=col, lw=1.3,
                label=rf"{lab} ($r_0 = 10\,r_s$; band 5--20)")
    ax.axhline(1e-5, color="0.35", lw=0.8, ls=":", label="Be/O required for GALEX J2339$-$0424")
    # Quiescent accretion rates of the systems, as markers on the Be/O curve.
    be10 = np.array([y["Be_O0"] for y in Y[10.0]])
    for name in DUTY:
        m = nums[name]["cases"]["quiescent"][10.0]["mdot_q"]
        ax.plot(m, np.exp(np.interp(np.log(m), np.log(mdots), np.log(be10))), SYS_MARK[name], ms=4,
                mfc="white", mec="black", mew=0.8, label=f"{name} (quiescent)", zorder=4)
    for name in GAIA:
        m = nums[name]["mdot"]
        ax.plot(m, np.exp(np.interp(np.log(m), np.log(mdots), np.log(be10))), SYS_MARK[name], ms=4,
                mfc="white", mec="black", mew=0.8, label=name, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-12, 1e-2); ax.set_ylim(1e-16, 1e1)
    ax.set_xlabel(r"accretion rate $\dot m = \dot M/\dot M_{\rm Edd}$")
    ax.set_ylabel("spallation yield per parcel")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), frameon=False, ncol=2, columnspacing=1.0)
    fig.savefig(f"{FIGDIR}/spallation_yields.png", dpi=300); plt.close(fig)


def fig_main(nums):
    """Predicted vs observed A(Li), and delivered vs required intercepted fraction."""
    names = list(DUTY)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=fullpage_figsize(0.47),
                                 gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.3})
    fig.subplots_adjust(bottom=0.27, top=0.97, left=0.08, right=0.98)

    # (a) predicted vs observed
    lim = (1.0, 3.4)
    xx = np.linspace(*lim, 10)
    ax.fill_between(xx, xx - 0.5, xx + 0.5, color="0.9", lw=0, zorder=0)
    ax.plot(xx, xx, color="0.4", lw=0.8, ls="--", zorder=1)
    for name in names:
        s = SYSTEMS[name]
        x0 = s["ALi"]
        for dx, case, col in ((-0.045, "quiescent", C_Q), (0.045, "outbursts", C_OB)):
            c = nums[name]["cases"][case]
            full = [v for r0 in R0S for v in c[r0]["ALi"]]
            fid = c[10.0]["ALi"]
            ax.plot([x0 + dx] * 2, [min(full), max(full)], color=col, lw=0.8, zorder=2)
            ax.plot([x0 + dx] * 2, fid, color=col, lw=4.0, solid_capstyle="butt", zorder=3)
        if s.get("ALi_upper"):
            # Upper limit on the observed lithium: arrow toward lower observed values.
            ax.errorbar(x0, x0, xerr=0.3, xuplims=True, fmt=SYS_MARK[name], ms=4.5, mfc="white", mec="black",
                        mew=0.9, color="black", ecolor="black", elinewidth=0.8, capsize=2, zorder=4)
        else:
            ax.errorbar(x0, x0, xerr=s["ALi_err"], fmt=SYS_MARK[name], ms=4.5, mfc="white", mec="black",
                        mew=0.9, ecolor="black", elinewidth=0.8, capsize=2, zorder=4)
    ax.set_xlim(*lim); ax.set_ylim(-1.6, 5.2)
    ax.set_xlabel(r"observed $A({\rm Li})$")
    ax.set_ylabel(r"SERPENS predicted $A({\rm Li})$")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)

    # (b) intercepted fraction: required by the observed Li vs computed by SERPENS
    allnames = names + list(GAIA)
    xs = np.arange(len(allnames))
    bx.axhspan(1e-6, 1e-4, color="0.9", lw=0, zorder=0)
    for i, name in enumerate(allnames):
        if name in GAIA:
            f = nums[name]["f_int"]
            bx.plot(i, f, SYS_MARK[name], ms=5, color=C_SIM, zorder=4)
            continue
        c = nums[name]["cases"]
        req_q = c["quiescent"][10.0]["f_req"]
        req_o = c["outbursts"][10.0]["f_req"]
        bx.plot([i - 0.12] * 2, sorted(req_q), color=C_Q, lw=4, solid_capstyle="butt", zorder=2)
        bx.plot([i + 0.12] * 2, sorted(req_o), color=C_OB, lw=4, solid_capstyle="butt", zorder=2)
        if SYSTEMS[name].get("ALi_upper"):
            # Observed Li is an upper limit, so the required fraction is too.
            for dx, req, col in ((-0.12, req_q, C_Q), (0.12, req_o, C_OB)):
                bx.annotate("", xy=(i + dx, min(req) / 8), xytext=(i + dx, min(req)),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=6))
        f = [c["outbursts"][r0]["f_int"] for r0 in R0S]
        bx.errorbar(i, f[1], yerr=[[f[1] - min(f)], [max(f) - f[1]]], fmt=SYS_MARK[name], ms=5,
                    color=C_SIM, mfc=C_SIM, capsize=2, zorder=4)
    bx.set_yscale("log")
    bx.set_ylim(1e-9, 1e3)
    bx.axhline(1.0, color="0.4", lw=0.6, ls=":")
    bx.set_xticks(xs); bx.set_xticklabels([n.replace(" ", "\n", 1) for n in allnames], rotation=0)
    bx.set_xlim(-0.6, len(allnames) - 0.4)
    bx.set_ylabel(r"fraction of accreted gas reaching the companion")
    bx.text(0.03, 0.97, "(b)", transform=bx.transAxes, ha="left", va="top", fontsize=7.5)

    # One shared legend below both panels (keeps every label off the data).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=C_Q, lw=4), Line2D([], [], color=C_OB, lw=4),
         Line2D([], [], color="0.3", lw=0.8), Patch(color="0.9"),
         Line2D([], [], color="0.4", lw=0.8, ls="--"), Line2D([], [], color="0.4", lw=0.6, ls=":")]
    l = ["quiescent hot flow", "quiescent + outburst hard states",
         r"(a) full range over $r_0 = 5$--$20\,r_s$", r"(a) 1:1 $\pm$0.5 dex;  (b) Yi & Narayan (1997) range",
         "(a) one-to-one line", "(b) all accreted gas (upper bound)"]
    for name in allnames:
        h.append(Line2D([], [], marker=SYS_MARK[name], ms=4.5, mfc="black" if name in GAIA else "white",
                        mec="black", ls="none"))
        l.append((f"{name}: observed (a)" + (", upper limit" if SYSTEMS[name].get("ALi_upper") else ""))
                 if name not in GAIA else f"{name}: computed (b)")
    h.append(Line2D([], [], marker="o", ms=4.5, color=C_SIM, ls="none"))
    l.append("(b) SERPENS computed (filled)")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 0.0),
               columnspacing=1.4, handletextpad=0.6)
    for lab in bx.get_xticklabels():
        lab.set_fontsize(6.2)
    fig.savefig(f"{FIGDIR}/spallation_vs_observed.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    style()
    nums = compute_numbers()
    n_esc, n_pl, n = fig_trajectories()
    nums["trajectories"] = {"escaped": n_esc, "plunged": n_pl, "n": n}
    fig_escape()
    fig_yields(nums)
    fig_main(nums)
    json.dump(nums, open(f"{OUT}/paper_numbers.json", "w"), indent=1, default=float)
    for name in DUTY:
        for case in ("quiescent", "outbursts"):
            c = nums[name]["cases"][case]
            full = [v for r0 in R0S for v in c[r0]["ALi"]]
            print(f"{name:14s} {case:9s} A(Li) r0=10: {c[10.0]['ALi'][0]:.2f}..{c[10.0]['ALi'][1]:.2f} "
                  f"full {min(full):.2f}..{max(full):.2f}  f_int={c[10.0]['f_int']:.2e}  "
                  f"f_req={c[10.0]['f_req'][1]:.1e}..{c[10.0]['f_req'][0]:.1e}  <mdot y>={c[10.0]['mean_mdot_y']:.2e}")
    for name in DUTY:
        print(name, "Be:", {k: round(v, 2) for k, v in nums[name]["beryllium"].items()},
              "r0 sens:", {k: round(v, 2) for k, v in nums[name]["r0_sensitivity"].items()})
    for name in GAIA:
        print(name, nums[name])
    print("trajectories", nums["trajectories"])
