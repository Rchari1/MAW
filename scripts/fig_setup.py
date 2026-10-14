"""
Setup figure: the black hole -- companion geometry and the SERPENS region.

Left: A0620-00 to scale in the co-rotating frame (q = M2/M_BH = 0.061). The companion fills
its Roche lobe (contour of the Roche potential through L1); the accretion stream is
integrated ballistically from L1 with the Coriolis force; rays are the asymptotic
directions of escaping ions from the exact-Schwarzschild classification, with those
that strike the companion highlighted.
Right: zoom on the black hole, SERPENS trajectories (as in the trajectory figure) over
the ion temperature of the hot flow.

Run: PYTHONPATH=. .venv/bin/python scripts/fig_setup.py
"""
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle, ConnectionPatch, Arc
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src import bh_spallation as bs
from src.bh_spallation import SpallationConfig, classify_central, ESCAPED, ion_kT
from src.paperstyle import fullpage_figsize, PALETTE
from scripts.paper_figures import style, trajectory_paths
from scripts.bh_spallation_systems import roche_lobe_over_a

warnings.filterwarnings("ignore")
FIG = "Paper/BlackHole_Paper/Figures/spallation_setup.png"
M_BH, M_2 = 6.6, 0.40                       # A0620-00
MU = M_2 / (M_BH + M_2)
CMAP = "PuBu"
C_ESC, C_HIT, C_STAR, C_STREAM = PALETTE["red"], "#8B1A00", "#F2A65A", "#D9822B"


def roche_potential(x, y):
    """Co-rotating Roche potential, a = 1, G(M1+M2) = 1, BH at the origin, companion at (1, 0)."""
    r1 = np.hypot(x, y)
    r2 = np.hypot(x - 1.0, y)
    return -(1 - MU) / r1 - MU / r2 - 0.5 * ((x - MU) ** 2 + y ** 2)


def lagrange_L1():
    dphi = lambda x: (roche_potential(x + 1e-7, 0) - roche_potential(x - 1e-7, 0)) / 2e-7
    return brentq(dphi, 0.3, 0.99)


def stream(x_l1, r_stop):
    """Ballistic gas stream from L1 in the rotating frame (Omega = 1)."""
    def grad(x, y, h=1e-6):
        return ((roche_potential(x + h, y) - roche_potential(x - h, y)) / (2 * h),
                (roche_potential(x, y + h) - roche_potential(x, y - h)) / (2 * h))

    def rhs(t, s):
        x, y, vx, vy = s
        gx, gy = grad(x, y)
        return [vx, vy, -gx + 2 * vy, -gy - 2 * vx]

    hit = lambda t, s: np.hypot(s[0], s[1]) - r_stop
    hit.terminal = True
    sol = solve_ivp(rhs, (0, 5), [x_l1 - 2e-3, 0.0, -1e-3, 0.0], events=hit, max_step=1e-3, rtol=1e-9)
    return sol.y[0], sol.y[1]


def escaper_directions(n=200000, r0=10.0, seed=11):
    cfg = SpallationConfig(r0=r0, n_particles=n, seed=seed)
    pos, vel = bs._launch(cfg, np.random.default_rng(seed))
    out = classify_central(pos, vel, "schwarzschild")
    d = vel[out == ESCAPED]
    return d / np.linalg.norm(d, axis=1)[:, None]


if __name__ == "__main__":
    style()
    theta = roche_lobe_over_a(M_2 / M_BH)                 # R_L / a, as in eq. (10)
    x_l1 = lagrange_L1()
    phi_l1 = roche_potential(x_l1, 0)
    r_disk = 0.7 * roche_lobe_over_a(M_BH / M_2)          # tidally truncated disk
    dirs = escaper_directions()
    cosang = dirs[:, 0]                                    # companion along +x
    hits = np.arccos(np.clip(cosang, -1, 1)) < theta
    paths, fate = trajectory_paths()

    fig = plt.figure(figsize=fullpage_figsize(0.43))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.22,
                          left=0.02, right=0.93, bottom=0.17, top=0.99)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(bx).append_axes("right", size="4%", pad=0.06)

    # ---------------- left: the binary ----------------
    XL, YL = (-0.7, 1.3), (-0.62, 0.62)
    X, Y = np.meshgrid(np.linspace(*XL, 900), np.linspace(*YL, 560))
    PHI = roche_potential(X, Y)
    # Accretion disk: surface brightness ~ r^-3/4 inside the truncation radius.
    R = np.hypot(X, Y)
    disk = np.where(R < r_disk, (np.maximum(R, 0.01) / r_disk) ** -0.75, np.nan)
    ax.imshow(disk, extent=(*XL, *YL), origin="lower", cmap=CMAP, norm=LogNorm(0.5, 12), zorder=1)
    ax.add_patch(Circle((0, 0), r_disk, fill=False, lw=0.5, ec="#3B6E8F", zorder=1.5))
    # Escaping-ion rays (projected on the orbital plane); companion-bound rays highlighted.
    rng = np.random.default_rng(3)
    miss = np.where(~hits)[0]
    for i in rng.choice(miss, size=min(240, len(miss)), replace=False):
        ux, uy = dirs[i, 0], dirs[i, 1]
        norm = np.hypot(ux, uy)
        ax.plot([0, 2.0 * ux / norm], [0, 2.0 * uy / norm], color=C_ESC, lw=0.3, alpha=0.3, zorder=2)
    for i in np.where(hits)[0]:
        ca = dirs[i, 0]
        s_hit = ca - np.sqrt(max(theta ** 2 - (1 - ca ** 2), 0.0))   # ray-sphere intersection
        ax.plot([0, s_hit * dirs[i, 0]], [0, s_hit * dirs[i, 1]], color=C_HIT, lw=0.55, zorder=6)
    # Companion: its Roche lobe, shaded toward the centre.
    near2 = np.hypot(X - 1, Y) < 0.45                      # exclude the outer critical surface
    near1 = np.hypot(X, Y) < 0.85
    lobe = (PHI <= phi_l1) & (X > x_l1) & near2
    shade = np.where(lobe, 1 - 0.55 * np.hypot(X - 1, Y) / theta, np.nan)
    ax.imshow(shade, extent=(*XL, *YL), origin="lower", cmap="YlOrBr", vmin=0, vmax=1.4, zorder=4)
    ax.contour(X, Y, np.where((X > x_l1) & near2, PHI, np.nan), levels=[phi_l1], colors="0.25",
               linewidths=0.7, zorder=5)
    ax.contour(X, Y, np.where((X < x_l1) & near1, PHI, np.nan), levels=[phi_l1], colors="0.45",
               linewidths=0.6, linestyles="--", zorder=3)
    sx, sy = stream(x_l1, r_disk)
    ax.plot(sx, sy, color=C_STREAM, lw=1.6, solid_capstyle="round", zorder=6.5)
    ax.plot(x_l1, 0, "+", color="black", ms=5, mew=0.8, zorder=7)
    ax.add_patch(Circle((0, 0), 0.012, color="black", zorder=7))
    # theta: tangent lines from the BH to the companion and an arc.
    for sgn in (1, -1):
        ax.plot([0, np.cos(theta) * 1.0], [0, sgn * np.sin(theta) * 1.0], color="0.2", lw=0.5, ls=":", zorder=6)
    ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=-np.degrees(theta), theta2=np.degrees(theta), lw=0.7, color="0.1", zorder=7))
    ax.text(0.345, 0.0, r"$\theta$", fontsize=7.5, ha="left", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    # Separation a.
    ax.annotate("", xy=(1.0, -0.55), xytext=(0.0, -0.55),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.15"), zorder=7)
    ax.text(0.5, -0.535, r"$a$", fontsize=7.5, ha="center", va="bottom", zorder=8,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    # Labels in empty regions, with leader lines.
    lab = dict(fontsize=6.5, zorder=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    arr = dict(arrowstyle="-", lw=0.5, color="0.2")
    ax.annotate("companion\n(fills Roche lobe)", xy=(1.08, 0.13), xytext=(1.1, 0.48), ha="center", arrowprops=arr, **lab)
    ax.annotate(r"L$_1$", xy=(x_l1, 0.0), xytext=(x_l1 - 0.05, 0.27), ha="center", arrowprops=arr, **lab)
    k = int(0.55 * len(sx))
    ax.annotate("gas stream", xy=(sx[k], sy[k]), xytext=(0.45, 0.45), ha="center", arrowprops=arr, **lab)
    ax.annotate("outer accretion disk", xy=(-0.25, -0.3), xytext=(-0.42, -0.5), ha="center", arrowprops=arr, **lab)
    ax.annotate("black hole +\nhot inner flow", xy=(-0.03, 0.03), xytext=(-0.22, 0.5), ha="center", arrowprops=arr, **lab)
    # Zoom box (not to scale) and connectors to the right panel.
    zb = 0.035
    ax.add_patch(Rectangle((-zb, -zb), 2 * zb, 2 * zb, fill=False, lw=0.7, ec="black", zorder=8))
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.015, 0.975, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)

    # ---------------- right: zoom on the black hole ----------------
    L = 45
    g = np.linspace(-L, L, 700)
    GX, GY = np.meshgrid(g, g)
    GR = np.hypot(GX, GY)
    kT = np.where(GR > 1.0, ion_kT(np.maximum(GR, 1.0)), np.nan)
    im = bx.imshow(kT, extent=(-L, L, -L, L), origin="lower", cmap=CMAP, norm=LogNorm(1.5, 90), zorder=0)
    show = np.random.default_rng(1).choice(np.where(fate == 0)[0], size=120, replace=False)
    for i in show:
        bx.plot(paths[i, :, 0], paths[i, :, 1], color="0.2", lw=0.3, alpha=0.5, zorder=1)
    for i in np.where(fate == 1)[0]:
        bx.plot(paths[i, :, 0], paths[i, :, 1], color=C_ESC, lw=0.9, zorder=3)
    bx.add_patch(Circle((0, 0), 1.0, color="black", zorder=5))
    bx.add_patch(Circle((0, 0), 3.0, fill=False, ls="--", lw=0.8, ec="white", zorder=5))
    bx.add_patch(Circle((0, 0), 10.0, fill=False, ls=":", lw=0.9, ec="white", zorder=5))
    bx.set_xlim(-L, L); bx.set_ylim(-L, L); bx.set_aspect("equal")
    bx.set_xlabel(r"$x\ [r_s]$"); bx.set_ylabel(r"$y\ [r_s]$", labelpad=1)
    bx.text(0.03, 0.97, "(b)", transform=bx.transAxes, ha="left", va="top", fontsize=7.5, color="black",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(im, cax=cax, ticks=[2, 5, 10, 20, 50])
    cb.ax.set_yticklabels(["2", "5", "10", "20", "50"])
    cb.ax.minorticks_off()
    cb.set_label(r"ion temperature $kT_i$ [MeV]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    for xa, ya in ((zb, zb), (zb, -zb)):
        con = ConnectionPatch(xyA=(xa, ya), coordsA=ax.transData,
                              xyB=(0.0, 1.0 if ya > 0 else 0.0), coordsB=bx.transAxes,
                              color="0.3", lw=0.5, ls="-", zorder=10)
        fig.add_artist(con)

    # Shared legend below.
    h = [Line2D([], [], color=C_ESC, lw=0.9), Line2D([], [], color=C_HIT, lw=1.2),
         Line2D([], [], color="0.2", lw=0.6), Line2D([], [], color=C_STREAM, lw=1.6),
         Line2D([], [], color="0.25", lw=0.7), Line2D([], [], color="0.45", lw=0.6, ls="--"),
         Line2D([], [], color="0.4", lw=0.8, ls="--"), Line2D([], [], color="0.4", lw=0.9, ls=":")]
    l = ["escaping ions", "escaping ions that strike the companion", "ions that cross the ISCO",
         r"gas stream from L$_1$", "companion Roche lobe", "black hole Roche lobe",
         r"(b) ISCO, $3\,r_s$", r"(b) launch sphere, $10\,r_s$"]
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.3, handletextpad=0.6, fontsize=6.3)
    fig.savefig(FIG, dpi=300)
    print(f"theta = {theta:.3f}, L1 = {x_l1:.3f} a, disk = {r_disk:.2f} a, escapers = {len(dirs)}, hits = {hits.sum()}")
