# SERPENS Black Hole — Handoff Document

*Written 2026-05-23. For a future Claude session picking up this work.*

---

## TL;DR

Working on `SERPENS_BlackHole` branch (fork: `Rchari1/SERPENS_BlackHole`). We have extended the SERPENS test-particle Monte Carlo code (originally for exo-Ios in exoplanet atmospheres) to model trace material near stellar-mass black holes. Proof-of-concept target is **Gaia BH1** (El-Badry+2023): a dormant 9.6 M☉ BH with a 0.93 M☉ G-dwarf companion at 480 pc.

The manuscript at `Paper/BlackHole_Paper/main.tex` compiles to an **11-page PDF** with 10 figures. All Claude-added text is in `\textcolor{red}{...}`. The physics is sound, figures are publication-tier, but the paper is a **methods + forecast** contribution suitable for PASP or MNRAS regular — **not an MNRAS Letter** (see "Publishability" below).

---

## Project Origin & Goal

**Original SERPENS** (Meyer zu Westram et al., in prep; branch `master`) is a 3D weighted Monte Carlo test-particle framework built on REBOUND + REBOUNDx. It tracks trace species (Na, S, O, etc.) launched from an exo-Io into a planet's atmosphere, computes ionization lifetimes from a chemical network, and produces column-density maps and phase curves.

**The extension question:** Can this machinery be repurposed for material near a compact accretor? The scientific analogy is **polluted white dwarfs → polluted black holes**: instead of the WD photosphere retaining accreted metals, the BH's kinematic and column-density observables encode the accretion history. Gaia BH1 gives us a concrete target because its binary parameters are astrometrically pinned down.

---

## Current State

### Branches

```
master                       - Original SERPENS (do not touch)
SERPENS_BlackHole            - Main working branch (all commits here)
SERPENS_BlackHole_test       - Scratch branch; already merged
```

Remote `fork` → `https://github.com/Rchari1/SERPENS_BlackHole.git`. Remote `origin` → `momzw/SERPENS.git` (no push access).

Latest commit on `SERPENS_BlackHole`: `d4bace4` (figure-scaling fix for bh_planar_view).

### Manuscript

**Location:** `Paper/BlackHole_Paper/main.tex` + `bibliography.bib`. Compile:
```bash
cd Paper/BlackHole_Paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

**Convention:** everything added by Claude is wrapped in `\textcolor{red}{...}`. Do not modify existing (non-red) text without discussing with the user — those are Raghav/Apurva/Moritz's original passages.

**Bibliography:** 46 entries. See "Bibliography additions" below.

**Missing sections:** Discussion, Conclusions, and a Validation section. Details in "Recommended Next Steps."

### Code Layout

Key files, all in `src/`:

| File | Purpose |
|---|---|
| `serpens_simulation.py` | Main class `SerpensSimulation(rebound.Simulation)`. Handles GR force attachment, ISCO removal, particle spawning, multi-threaded integration |
| `spawner.py` | Particle velocity distributions: thermal, sputter (Smyth-Combi), wind (shifted Maxwellian). Also `parker_v_inf()`, `wood2005_mdot()`, `companion_wind_parameters()` helpers |
| `species.py` | `Species` class + `species_info` dict. 20 species defined; IDs 12–20 added for BH-environment work |
| `network.py` | Chemical reaction networks per species ID. IDs 12–20 have placeholder finite lifetimes (see "Known Gotchas") |
| `parameters.py` | `GLOBAL_PARAMETERS` singleton. Auto-enables `gr_enabled` when loading a system with `r_schwarzschild` defined |
| `bh_observables.py` | **New module.** BHL capture, observer-frame projection, Doppler line profile, X-ray photoionization lifetime, physical-units column density, FIP abundance helper, ADAF efficiency, conservation diagnostics |
| `paperstyle.py` | **New module.** Unified matplotlib rcParams for revtex4 layout. `apply_paper_style(use_tex=True/False)`, `column_figsize(aspect)`, `fullpage_figsize(aspect)`, `PALETTE` (Wong 2011 color-blind-safe) |
| `serpens_analyzer.py` | Original analyzer with DTFE density estimator. Currently used only for the exomoon path; BH path uses `bh_observables.py` |
| `visualize.py` | Original matplotlib visualization framework (planar / LOS / 3D plotly). Untouched by BH work |

**System definitions:** `resources/objects.json`. Three BH systems added: `StellarBH-10`, `SgrA`, `GaiaBH1`. Each has `r_schwarzschild` and `r_isco` fields. BH `r` (collision radius) is set to `r_isco` so collision-merge boundary aligns with ISCO removal — see "Known Gotchas."

**Notebooks:** `notebooks/gaia_bh1.ipynb` runs the full pipeline end-to-end. `notebooks/test_gr.ipynb` is an older StellarBH-10 sanity check.

### Figure Inventory

All figures live in `Paper/BlackHole_Paper/Figures/`. All are at 300 dpi, black-on-white, Wong 2011 palette, revtex4-sized.

| Filename | Type | Size | Content |
|---|---|---|---|
| `SERPENS.png` | \singlefig | (legacy) | Coordinate system schematic |
| `bh_planar_view.png` | \doublefig | 7.0×3.2 in | StellarBH-10 orbital ring (dark bg) + strong-field zoom with annotations |
| `bh_density_maps.png` | \doublefig | 7.0×3.0 in | Face-on + edge-on density projections |
| `bh_radial_kepler.png` | \doublefig | 7.0×2.9 in | Radial histogram + speed-distance Keplerian check |
| `gaiabh1_planar.png` | \singlefig | 3.4×2.6 in | Wind distribution around Gaia BH1 |
| `gaiabh1_skymap.png` | \doublefig | 6.9×2.9 in | LOS velocity + H column density in cm⁻² |
| `gaiabh1_lineprofiles.png` | \doublefig | 7.0×4.0 in | 2×2 Doppler profiles for H, He, C, O with Poisson error bands |
| `gaiabh1_convergence.png` | \singlefig | 3.4×2.3 in | MC bootstrap convergence of ⟨N_H⟩ |
| `gaiabh1_energy_diagram.png` | \singlefig | 3.4×2.8 in | log₁₀|E| vs log₁₀(r/rₛ); all particles unbound |
| `wind_launch_comparison.png` | \singlefig | 3.4×2.3 in | Smyth-Combi vs Parker PDFs |

Also present but not in the paper: `gaiabh1_capture_phase.png`, `bh_3d_interactive.html`, `mixing_ratios.png`.

**Sizing convention:** the `\singlefig` and `\doublefig` macros in `main.tex` now use `width=\columnwidth` and `width=\textwidth` respectively. **The scale argument (#5) is ignored.** Don't change this back — the original scale-based approach broke because saved PNGs have dpi=300 metadata that pdflatex reads as the "natural" width.

---

## Physics & Design Decisions

### GR treatment
- **1PN post-Newtonian** via REBOUNDx `gr_full` (Newhall+1983 formulation)
- **Auto-enabled** when loading any system with `r_schwarzschild` (see `parameters.py:update_celest`)
- **Integrator auto-switches** to MERCURIUS hybrid (WHFast + IAS15 for close encounters) when GR is on, because WHFast fixed-timestep cannot resolve BH-plunge orbits
- **ISCO removal** at `3 r_s` (Schwarzschild). Particles that plunge inside are marked as accreted and removed
- **BH collision radius** set to `r_isco` (not `r_s`) so the REBOUND merge-collision doesn't silently absorb particles before the ISCO diagnostic fires

### Wind launch (source physics)
- **Shifted Maxwellian:** Maxwell-Boltzmann thermal (T_corona) + deterministic radial bulk v_∞
- **`parker_v_inf(T, M_star, R_star)`:** uses `v_∞ = max(2.5 c_s √ln(r_eval/r_c), 0.25 v_esc)`. Not the strict isothermal Parker (which has log divergence with no asymptote); this is a well-behaved practical fit
- **`wood2005_mdot(R_star, age_Gyr)`:** piecewise, saturated at 100× solar below 0.7 Gyr, else scales as age⁻²·³³ (Skumanich × Wood F_X calibration)
- **`companion_wind_parameters(M, R, T, age)`:** convenience wrapper returning `{Mdot, v_inf, T_corona}`

### Composition
- **FIP effect** applied to solar-abundance mass fractions: `fip_abundances(Mdot_total, fip_factor=3.5)` in `bh_observables.py`. Low-FIP elements (Fe, Si) get boosted ×3.5
- Ionization treated as instantaneous death at τ_ion(r). **This is a limitation** — see "Known Gaps"

### Radiative output
- **ADAF/RIAF suppression** at low Mdot/Mdot_Edd: `radiative_efficiency()` in `bh_observables.py` implements the piecewise η(m) that goes as m/10⁻² below the transition. For Gaia BH1, m ~ 10⁻⁸ so η ~ 10⁻⁶ instead of 0.1 (Narayan-Yi 1995)

### Observables
- **Observer-frame projection** via `project_to_observer(positions, velocities, inclination, omega, Omega)` — accepts Euler angles matching El-Badry+2023 convention
- **Physical column density** in cm⁻² via `column_density_map(sky_xy, weights, extent_au)`. Superparticle weight = `Mdot × t_sim / (n_per_species × n_spawns × m_species)`
- **Doppler line profile** via `doppler_line_profile(los_vel, weights, rest_wavelength, thermal_broadening)`

### Conservation diagnostics
- `system_diagnostics(sim)` returns total mechanical energy, angular momentum vector, COM drift
- **Measured `dE/E = 3.6×10⁻⁹` over 2 Gaia BH1 orbital periods** — excellent symplectic conservation

---

## Known Bugs / Gotchas

1. **`fix_source_circular_orbit`** is set to `False` automatically when GR is enabled (via `parameters.py:update_celest`). The original SERPENS heartbeat zeroes source-body eccentricity every step, which would destroy Gaia BH1's e=0.45 orbit. **Do not re-enable it for BH systems.**

2. **New species (IDs 12–20) have placeholder network lifetimes** of 10⁹–10¹⁰ s. If you need physically correct chemistry for these species, add proper reactions in `network.py`. Currently `network=None` cannot occur (would silently kill particles or crash).

3. **Wind particles in Gaia BH1 are 100% unbound.** This is real physics (Parker v_∞ ≈ 400 km/s exceeds escape velocity from BH at binary separation), not a bug. The energy-diagram figure documents this. Consequence: our "capture rate" is dominated by transient flybys through r_BHL, not committed accretors. Don't panic when you see 0% capture — it's the answer.

4. **PyMuPDF is required** for rendering PDF pages in-conversation: `.venv/bin/pip install PyMuPDF`. Already installed. Preview with:
   ```python
   import fitz
   doc = fitz.open('Paper/BlackHole_Paper/main.pdf')
   for i, page in enumerate(doc):
       page.get_pixmap(dpi=120).save(f'/tmp/p{i+1:02d}.png')
   ```

5. **Multithreaded integrate can deadlock** on repeated builds within the same Python process (leaked semaphores from `multiprocessing.resource_tracker`). Fix: run figure-regeneration scripts as one-shot subprocesses via `.venv/bin/python /tmp/regen.py`, not by re-running cells in a persistent kernel.

6. **Figure sizing:** if you generate a new figure, save with `dpi=300` and explicit `figsize` in inches. The macros will size it correctly. Don't set an explicit scale factor — it will be ignored.

7. **Pre-existing typos in original manuscript text:**
   - `\gtreq` (fixed to `\gtrsim` on line ~80)
   - `\Mdot` (fixed to `\dot{M}` on line 283)
   Both were causing compile failures. Fix them if they resurface.

8. **`.venv/bin/pip` path:** the working Python environment is `.venv/`. Use `.venv/bin/python` and `.venv/bin/pip`. Rebound 4.6.0 + REBOUNDx 4.6.1 are installed and working.

---

## Bibliography additions (all new since methods paper started)

Added to `bibliography.bib`:
`ElBadry_2023`, `Newhall_1983`, `Tamayo_2020`, `Bardeen_1972`, `Will_2014`, `Paczynski_Wiita_1980`, `Reynolds_2021`, `Hopman_Alexander_2006`, `Parker_1958`, `LamersCassinelli_1999`, `Wood_2005`, `CranmerSaar_2011`, `Marsch_2006`, `Edgar_2004`, `NarayanYi_1995`, `YuanNarayan_2014`, `Treves_2000`, `Verner_1996`, `ShakuraSunyaev_1973`, `Asplund2009`, `Wood2014`, `ReinHernandez2017`, `ReinTamayo2019`, `Wong2011`, `Bondi1952`, `Wisdom1991`, `Skumanich1972`, `CranmerWinebarger2019`.

Full list of ~46 entries in `bibliography.bib`.

---

## Open Threads (user has been thinking about these)

### 1. Publishability & venue
**User asked (multiple times):** is this an MNRAS Letter? **My honest answer:** no. Currently a methods + forecast paper. Right venues:
- **PASP** (methods) — high acceptance probability
- **MNRAS regular** (methods + Gaia BH1/2/3 application if we add sister systems)
- **RNAAS** for a quick methodology-only note
- **JOSS** if code is the contribution

To *convert* to a Letter would require a single tight result: e.g., "predicted Lyα equivalent width vs orbital phase for Gaia BH1 crosses the HST/COS detection threshold at conjunction." One figure, 4 pages.

### 2. Project naming
**User asked for an acronym containing "black holes."** I recommended:
- **EREBUS** (Greek primordial darkness) — *Ejecta and Relativistic Evolution around Black-hole Unbound Superparticles*
- **ISCO** — *Inspiraling Superparticles around Compact Objects* (insider-clever)
- **BHWIND** — *Black-Hole Wind Ionization & N-body Dynamics* (most literal)

Nothing wired in yet. If the user picks one, update:
- Repo README (create if missing)
- Manuscript title/header
- One-line intro paragraph: "EREBUS extends SERPENS to..."
- Possibly rename branches

### 3. Validation section
User's next question: "are there observables we can validate against?" I proposed three concrete tests:
1. **Mercury perihelion precession** — 43"/century 1PN test
2. **S2 star around Sgr A*** (system already in `objects.json`) — GRAVITY-measured precession
3. **Analytical Bondi-Hoyle-Lyttleton comparison** — measured MC capture rate vs `π r_BHL² ρ v_rel`

None implemented yet. Would give the paper 1-3 more figures and a "Validation" section. Bar-for-referee-credibility rises significantly.

---

## How to Run

### One-off simulation
```python
from src.serpens_simulation import SerpensSimulation
from src.parameters import GLOBAL_PARAMETERS
from src.species import Species
from src.spawner import companion_wind_parameters
from src import bh_observables as bho

# Gaia BH1 setup — GR auto-enables from objects.json
sim = SerpensSimulation(system='GaiaBH1')
wind = companion_wind_parameters(M_star=1.849e30, R_star=6.557e8, T_corona=1.5e6, age_Gyr=4.5)
GLOBAL_PARAMETERS.set('wind_T_corona', wind['T_corona'])
GLOBAL_PARAMETERS.set('wind_v_inf', wind['v_inf'])

# FIP-corrected wind composition
mdots = bho.fip_abundances(wind['Mdot'], fip_factor=3.5, species=['H','He','C','N','O','Fe'])
species_list = []
for name, m in mdots.items():
    tau = bho.species_lifetime_in_disk_field(name, 1e29, 2.105e11)
    species_list.append(Species(name=name, n_wind=200, mass_per_sec=m, lifetime=tau))
sim.object_to_source('companion', species_list)

sim.advance(orbits=2, spawns=40, orbits_reference='companion')
```

### Regenerate all figures
Two scripts in `/tmp/` are the canonical figure generators (they get re-created each session, so committed permanent versions would help):
- `/tmp/regen_all_v2.py` — all Gaia BH1 figures
- `/tmp/regen_benchmarks.py` — StellarBH-10 benchmarks
- `/tmp/fix_planar.py` — dedicated bh_planar_view generator (latest)

**Action item:** these should really live in `notebooks/regen_figures.py` or similar so they're versioned.

### Compile paper
```bash
cd Paper/BlackHole_Paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Cleanup: `rm -f main.aux main.bbl main.blg main.log main.out`.

---

## Recommended Next Steps

Ordered by (value / effort):

### Tier 1 — cheap, high-impact

1. **Version-control the figure-generation scripts.** Move `/tmp/regen_*.py` into `notebooks/` or `scripts/`. Right now they're ephemeral.
2. **Add a Discussion + Conclusions section** to `main.tex`. The paper currently ends abruptly at "Taken together, Figures..." Needs to: (a) list limitations honestly (no radiative transfer, no spin, no dust, no self-gravity), (b) connect to polluted-WD literature (Doyle+2021), (c) roadmap for follow-up.
3. **Add the Validation section.** Mercury precession + S2/SgrA* + analytical-BHL comparison. Three new figures, one new section. See "Open Threads #3."
4. **Rename to EREBUS** (or whatever the user chose). Textual pass.

### Tier 2 — scientific extensions

5. **Apply pipeline to Gaia BH2 + BH3.** Existing objects.json framework supports it. Adds a systems-population figure.
6. **Multi-stage ionization tracking.** Currently death at τ_ion. Real physics is O → O⁺ → O²⁺ → ... Would fix the "line profiles are conceptual" limitation.
7. **Phase-resolved Lyα equivalent width prediction** for Gaia BH1. This is the Letter-worthy result if you want that venue.

### Tier 3 — bigger moves

8. **TDE source model** — deformable companion at periastron.
9. **Kerr geodesics** — for spinning BH predictions.
10. **HMXB tomography validation** (Cyg X-1 / Vela X-1) — actual observational validation.

---

## Contact / Style Notes

- **User is Raghav Chari (NYU).** Co-authors on the paper: Apurva Oza (JPL), Moritz Meyer zu Westram (Bern).
- **User's stated preference:** honest technical feedback, no flattery. They explicitly asked several times for the frank assessment of publishability.
- **All Claude edits to `main.tex` go in `\textcolor{red}{...}`.** This is a hard convention.
- **User uses VS Code with the Claude Code extension.** File references like `[filename.py:42](src/filename.py#L42)` render as clickable links.
- **Auto mode may be on.** When on, bias toward acting; when off, ask before mildly-destructive actions (rename, branch changes, force pushes).

---

## Quick Sanity Check

Before doing anything substantive, confirm the current state:
```bash
cd /Users/raghavchari/SERPENS
git log --oneline -3
# Should show d4bace4, 753fc06, 8410279 (or newer)
git status
# Should be clean or show only .DS_Store / __pycache__

cd Paper/BlackHole_Paper
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "Output written"
# Should say: Output written on main.pdf (11 pages, ~1.6 MB)
```

If those pass, the codebase is in the state described above.

---

*End of handoff. Latest paper PDF is at `Paper/BlackHole_Paper/main.pdf`. Latest push is `d4bace4` on `fork/SERPENS_BlackHole`.*
