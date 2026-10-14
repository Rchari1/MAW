"""
SERPENS ejecta mode applied to black-hole binaries with measured companion abundances.

For each system we take the pre-explosion configuration adopted in the abundance
papers (LMXBs) or reconstructed from the present orbit assuming instantaneous,
kick-free mass loss (Gaia BHs: explosion at periastron, a_pre = a (1 - e),
Delta M = e (M_BH + M2); Blaauw 1961), and compute the ejecta mass delivered to
the companion. Comparing the delivered oxygen with the oxygen excess measured in
the companion's photosphere gives the retention efficiency the observations
require.

Run: PYTHONPATH=. .venv/bin/python scripts/ejecta_systems.py
"""
import json
import zlib
import os
import numpy as np

from src.bh_ejecta import (EjectaConfig, EjectaResult, run_ejecta, M_SUN, R_SUN,
                           COMPANION, FALLBACK, BOUND_BH, CIRCUMBINARY, ESCAPED)

OUT = "output/ejecta"
AU = 1.495978707e11
X_O_SUN = 5.79e-3          # Asplund+2009 photospheric O mass fraction
X_O_EJ = (0.3, 0.6)        # O mass fraction of He-core ejecta for 25-40 Msun progenitors (assumed range)


def gaia(M_bh, M2, a_au, e):
    """Kick-free pre-SN reconstruction from the present eccentric orbit."""
    return dict(M_bh=M_bh, M_ej=e * (M_bh + M2), M2=M2, a_pre=a_au * (1 - e) * AU / R_SUN)


# Masses in Msun, lengths in Rsun. delta_O: measured O excess over Galactic trend [dex];
# for the Gaia systems no anomaly is reported and we adopt <~0.1 dex as an upper limit.
SYSTEMS = {
    # Gonzalez Hernandez+2008 (A&A 478, 203): He core 8.5, M_BH,i ~ 5, M2 ~ 2, a ~ 10 Rsun,
    # captured matter mixed with the whole star. [O/H] = +0.91 +/- 0.09, [Fe/H] = -0.11.
    "Nova Sco 1994": dict(M_bh=5.0, M_ej=3.5, M2=2.0, R2=3.3, a_pre=10.0, M_mix=2.0,
                          delta_O=0.91, delta_O_err=0.09, kind="LMXB"),
    # Gonzalez Hernandez+2011 (ApJ 738, 95): M_He = 15, M_BH,i = 8, M2,i = 2, R2,i = 3.3,
    # a0 ~ 9 Rsun, mixed with the whole star. [O/Fe] = 0.37 vs -0.32 in comparison stars.
    "V404 Cyg": dict(M_bh=8.0, M_ej=7.0, M2=2.0, R2=3.3, a_pre=9.0, M_mix=2.0,
                     delta_O=0.69, delta_O_err=0.24, kind="LMXB"),
    # Gonzalez Hernandez+2004 (ApJ 609, 988): M_He ~ 14, Delta M ~ 4, M2 ~ 1 (R ~ 1.3),
    # a0 ~ 3.3 Rsun, m_conv = 0.652 Msun. O not measured.
    "A0620-00": dict(M_bh=10.0, M_ej=4.0, M2=1.0, R2=1.3, a_pre=3.3, M_mix=0.652,
                     delta_O=None, kind="LMXB"),
    # Gonzalez Hernandez+2008 (ApJ 679, 732): M2,i ~ 1 (R ~ 1.3), a0 = 3-5 Rsun,
    # M_He 8.5-16, m_conv = 0.652 Msun. O not measured.
    "XTE J1118+480": dict(M_bh=7.0, M_ej=5.0, M2=1.0, R2=1.3, a_pre=4.0, M_mix=0.652,
                          delta_O=None, kind="LMXB"),
    # El-Badry+2023a (MNRAS 518, 1057): 9.62 + 0.93 Msun, a = 1.40 AU, e = 0.451.
    "Gaia BH1": dict(**gaia(9.62, 0.93, 1.40, 0.451), R2=1.3, M_mix=0.93,
                     delta_O=0.1, delta_O_err=None, kind="Gaia"),
    # El-Badry+2023b (MNRAS 521, 4323): 8.94 + 1.07 Msun, a = 4.96 AU, e = 0.518;
    # [alpha/Fe] = +0.26 consistent with the thick-disk trend.
    "Gaia BH2": dict(**gaia(8.94, 1.07, 4.96, 0.518), R2=1.3, M_mix=1.07,
                     delta_O=0.1, delta_O_err=None, kind="Gaia"),
    # Gaia Collaboration, Panuzzo+2024 (A&A 686, L2): 32.7 + 0.76 Msun, a = 16.17 AU, e = 0.729.
    "Gaia BH3": dict(**gaia(32.7, 0.76, 16.17, 0.729), R2=1.3, M_mix=0.76,
                     delta_O=0.1, delta_O_err=None, kind="Gaia"),
}

ENERGIES = {"SN": 1e44, "HN": 3e45}     # 1e51 and 3e52 erg


def config_for(name, sysp, E):
    wide = sysp["kind"] == "Gaia"
    return EjectaConfig(
        M_bh=sysp["M_bh"] * M_SUN, M_ej=sysp["M_ej"] * M_SUN, M2=sysp["M2"] * M_SUN,
        R2=sysp["R2"] * R_SUN, a_pre=sysp["a_pre"] * R_SUN, E_kin=E, R_launch=R_SUN,
        n_particles=8000 if wide else 20000,
        cone_half_angle=4 * sysp["R2"] / sysp["a_pre"] if wide else None,
        t_end_orbits=3.0, seed=zlib.crc32(name.encode()))


def run_all():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, sysp in SYSTEMS.items():
        for label, E in ENERGIES.items():
            tag = name.replace(" ", "_").replace("+", "p") + "_" + label
            path = os.path.join(OUT, tag + ".npz")
            if os.path.exists(path):
                res = EjectaResult.load(path)
            else:
                print(f"{name} [{label}]")
                res = run_ejecta(config_for(name, sysp, E))
                res.save(path)
            hit = res.outcome == COMPANION
            wide = res.config.cone_half_angle is not None
            M_cap = res.weight[hit].sum() / M_SUN
            f_geo = 0.5 * (1 - np.sqrt(1 - (sysp["R2"] / sysp["a_pre"]) ** 2))
            row = dict(system=name, energy=label, kind=sysp["kind"],
                       M_ej=sysp["M_ej"], a_pre=sysp["a_pre"], R2=sysp["R2"], M_mix=sysp["M_mix"],
                       f_companion=res.mass_fraction(COMPANION),
                       f_companion_err=res.mass_fraction(COMPANION) / np.sqrt(max(hit.sum(), 1)),
                       f_geometric=f_geo,
                       # Cone (importance-sampled) runs only follow material aimed at the companion,
                       # so their fallback / circumbinary fractions are not meaningful.
                       f_fallback=np.nan if wide else res.mass_fraction(FALLBACK) + res.mass_fraction(BOUND_BH),
                       f_circumbinary=np.nan if wide else res.mass_fraction(CIRCUMBINARY),
                       M_cap=M_cap, x_cap=M_cap / sysp["M_mix"],
                       mean_mass_coord_captured=float(np.average(res.mass_coord[hit], weights=res.weight[hit])) if hit.any() else np.nan,
                       median_v_impact_kms=float(np.nanmedian(res.v_impact[hit]) / 1e3) if hit.any() else np.nan,
                       binary_after=res.binary_final)
            if sysp.get("delta_O") is not None:
                need = sysp["M_mix"] * X_O_SUN * (10 ** sysp["delta_O"] - 1)
                row["M_O_required"] = need
                row["retention_required"] = [need / (x * M_cap) if M_cap > 0 else np.inf for x in X_O_EJ[::-1]]
                # Maximum O excess deliverable with perfect retention.
                row["delta_O_max"] = [float(np.log10(1 + x * M_cap / (X_O_SUN * sysp["M_mix"]))) for x in X_O_EJ]
            rows.append(row)
    with open(os.path.join(OUT, "systems_summary.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return rows


if __name__ == "__main__":
    rows = run_all()
    print(f"\n{'system':15s} {'E':3s} {'f_comp':>9s} {'f_geo':>9s} {'f_fb':>6s} {'M_cap':>9s} {'M_cap/M_mix':>11s} "
          f"{'<m>_cap':>7s} {'v_imp':>6s}  retention needed / max d[O] (f_ret=1)")
    for r in rows:
        extra = ""
        if "retention_required" in r:
            lo, hi = r["retention_required"]
            extra = (f"f_ret = {lo:.2g}-{hi:.2g}" if r["kind"] == "LMXB"
                     else f"d[O]_max = {r['delta_O_max'][0]:.1e}-{r['delta_O_max'][1]:.1e} dex")
        print(f"{r['system']:15s} {r['energy']:3s} {r['f_companion']:9.2e} {r['f_geometric']:9.2e} "
              f"{r['f_fallback']:6.3f} {r['M_cap']:9.2e} {r['x_cap']:11.2e} {r['mean_mass_coord_captured']:7.2f} "
              f"{r['median_v_impact_kms']:6.0f}  {extra}")
