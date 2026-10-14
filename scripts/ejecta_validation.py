"""
Validation of the SERPENS ejecta mode against analytic limits.

(a) Companion capture fraction for mono-energetic ejecta vs the straight-line
    + gravitational-focusing estimate.
(b) Fallback fraction for broken power-law ejecta vs the isolated-BH ballistic
    estimate f = m(< v_esc(R_launch)).

Run: PYTHONPATH=. .venv/bin/python scripts/ejecta_validation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bh_ejecta import (EjectaConfig, EjectaResult, run_ejecta, M_SUN, R_SUN, G,
                           COMPANION, FALLBACK, BOUND_BH,
                           companion_capture_fraction_analytic, fallback_fraction_isolated)
from src.paperstyle import apply_paper_style, fullpage_figsize, PALETTE

OUT = "output/ejecta"
FIG = "Paper/BlackHole_Paper/Figures/ejecta_validation.png"

BASE = dict(M_bh=7 * M_SUN, M_ej=3 * M_SUN, M2=2.5 * M_SUN, R2=3 * R_SUN, a_pre=12 * R_SUN,
            t_end_orbits=3.0)


def cached(name, cfg):
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        return EjectaResult.load(path)
    res = run_ejecta(cfg)
    res.save(path)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    v_esc_launch = np.sqrt(2 * G * BASE["M_bh"] / R_SUN)

    # (a) mono-energetic sweep
    v_grid = v_esc_launch * np.array([1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0, 8.0])
    f_mc, f_err = [], []
    for i, v in enumerate(v_grid):
        cfg = EjectaConfig(**BASE, velocity_model="fixed", v_fixed=float(v), n_particles=16000, seed=100 + i)
        r = cached(f"fixed_{i}", cfg)
        k = np.sum(r.outcome == COMPANION)
        f_mc.append(k / len(r.outcome)); f_err.append(np.sqrt(max(k, 1)) / len(r.outcome))
        print(f"v/v_esc={v / v_esc_launch:.2f}  f_MC={f_mc[-1]:.4f} +/- {f_err[-1]:.4f}  "
              f"f_an={companion_capture_fraction_analytic(v, BASE['M_bh'], BASE['M2'], BASE['R2'], BASE['a_pre'], R_SUN):.4f}  "
              f"{r.fractions()}")

    # (b) fallback vs explosion energy
    E_grid = np.geomspace(3e42, 3e44, 7)
    fb_mc, fb_err = [], []
    for i, E in enumerate(E_grid):
        cfg = EjectaConfig(**BASE, E_kin=float(E), n_particles=8000, seed=200 + i)
        r = cached(f"energy_{i}", cfg)
        fb = r.mass_fraction(FALLBACK) + r.mass_fraction(BOUND_BH)
        fb_mc.append(fb); fb_err.append(np.sqrt(fb * (1 - fb) / len(r.outcome)))
        print(f"E={E:.1e} J  fallback_MC={fb:.4f}  analytic={fallback_fraction_isolated(E, BASE['M_ej'], BASE['M_bh'], R_SUN):.4f}"
              f"  companion={r.mass_fraction(COMPANION):.4f}")

    apply_paper_style(use_tex=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fullpage_figsize(0.38))
    vv = np.geomspace(v_grid[0] * 0.98, v_grid[-1] * 1.05, 300)
    ax1.plot(vv / v_esc_launch, companion_capture_fraction_analytic(vv, BASE["M_bh"], BASE["M2"], BASE["R2"], BASE["a_pre"], R_SUN),
             color="k", lw=1.2, label="Straight line + focusing")
    geo = 0.5 * (1 - np.sqrt(1 - (BASE["R2"] / BASE["a_pre"]) ** 2))
    ax1.axhline(geo, color="0.5", ls=":", lw=1, label=r"Geometric, $R_2^2/4a^2$")
    ax1.errorbar(v_grid / v_esc_launch, f_mc, yerr=f_err, fmt="o", ms=4, color=PALETTE.get("blue", "C0"),
                 label="SERPENS Monte Carlo")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Launch speed $v / v_{\rm esc}(R_{\rm launch})$")
    ax1.set_ylabel("Fraction hitting the companion")
    ax1.legend(frameon=False, fontsize=7)

    EE = np.geomspace(E_grid[0], E_grid[-1], 200)
    ax2.plot(EE * 1e7 / 1e51, [fallback_fraction_isolated(E, BASE["M_ej"], BASE["M_bh"], R_SUN) for E in EE],
             color="k", lw=1.2, label=r"Isolated BH, $m(<v_{\rm esc})$")
    ax2.errorbar(E_grid * 1e7 / 1e51, fb_mc, yerr=fb_err, fmt="o", ms=4, color=PALETTE.get("vermillion", "C1"),
                 label="SERPENS Monte Carlo")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"Explosion energy $E_{\rm kin}$ [$10^{51}$ erg]")
    ax2.set_ylabel("Ejecta mass fraction falling back")
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    print("saved", FIG)
