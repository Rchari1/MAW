"""
Black-hole-formation ejecta in a binary: who catches what.

SERPENS ejecta mode. At t = 0 the black-hole progenitor (mass M_BH + M_ej) in a
circular binary with a companion star explodes. The remnant black hole keeps
M_BH (plus an optional natal kick) and the ejecta are launched as weighted test
particles from the progenitor surface on homologous (free-expansion) velocities.
Each superparticle is followed under the gravity of the black hole and the
companion (REBOUND IAS15, optional REBOUNDx 1PN around the BH) until it

  * strikes the companion star            -> COMPANION    (pollution)
  * falls back into the black hole         -> FALLBACK
  * leaves the system unbound              -> ESCAPED
  * or, at t_end, is still bound to the BH (BOUND_BH), to the companion
    (BOUND_COMPANION) or to the binary as a whole (CIRCUMBINARY).

Because the ejecta expand homologously, a superparticle's launch speed fixes its
Lagrangian mass coordinate m(<v)/M_ej, so a layered (onion-shell) composition can
be painted on after the run without re-integrating.

Black-hole capture: a particle entering the absorbing sphere r_check (a fraction
of the launch radius, i.e. deep inside the BH's Roche lobe) while bound to the BH
is counted as fallback; it would circularise and accrete. Its Newtonian
pericentre r_p is stored as well: r_p < 8GM/c^2 (= 4 r_s) is direct relativistic
capture of a marginally bound particle (L < 4GM/c; Shapiro & Teukolsky 1983).

Collision search in REBOUND scales as N^2 even for test particles, so particles
are integrated in independent batches across processes.

References:
  * Chevalier & Soker (1989) ApJ 341, 867; Matzner & McKee (1999) ApJ 510, 379
    -- broken power-law ejecta density profile
  * Blaauw (1961) BAN 15, 265 -- orbital response to instantaneous mass loss
  * Eggleton (1983) ApJ 268, 368 -- Roche-lobe radius
  * Shapiro & Teukolsky (1983) "Black Holes, White Dwarfs and Neutron Stars", ch. 12
  * Rein & Spiegel (2015) MNRAS 446, 1424 -- IAS15
  * Tamayo et al. (2020) MNRAS 491, 2885 -- REBOUNDx
"""

import concurrent.futures
import multiprocessing
from dataclasses import dataclass, field, asdict

import numpy as np
import rebound
import reboundx


G = 6.6743e-11
C_LIGHT = 2.99792458e8
M_SUN = 1.989e30
R_SUN = 6.957e8
DAY = 86400.0

ESCAPED, COMPANION, FALLBACK, BOUND_BH, BOUND_COMPANION, CIRCUMBINARY = range(6)
OUTCOME_NAMES = {
    ESCAPED: "escaped",
    COMPANION: "companion",
    FALLBACK: "fallback",
    BOUND_BH: "bound_bh",
    BOUND_COMPANION: "bound_companion",
    CIRCUMBINARY: "circumbinary",
}


# ---------------------------------------------------------------------------
# Ejecta velocity distributions (mass-weighted, dM/dv)
# ---------------------------------------------------------------------------

def transition_velocity(E_kin, M_ej, delta=1.0, n=10.0):
    """
    Break velocity of the Chevalier & Soker (1989) / Matzner & McKee (1999)
    broken power-law ejecta, rho ~ v^-delta inside v_t and v^-n outside:

        v_t = sqrt[ 2 (5-delta)(n-5) E / ((3-delta)(n-3) M_ej) ]

    Arguments
    ---------
    E_kin : float
        Ejecta kinetic energy [J] (1e51 erg = 1e44 J).
    M_ej : float
        Ejecta mass [kg].
    delta, n : float
        Inner and outer density slopes (n > 5).
    """
    return np.sqrt(2 * (5 - delta) * (n - 5) * E_kin / ((3 - delta) * (n - 3) * M_ej))


def ejecta_cdf(v, E_kin, M_ej, delta=1.0, n=10.0):
    """Mass fraction of the ejecta moving slower than v (= Lagrangian mass coordinate)."""
    v_t = transition_velocity(E_kin, M_ej, delta, n)
    x = np.asarray(v, dtype=float) / v_t
    A = (3 - delta) * (n - 3) / (n - delta)    # normalisation of dM/dx
    inner = A * np.minimum(x, 1.0) ** (3 - delta) / (3 - delta)
    outer = np.where(x > 1.0, A * (1.0 - np.maximum(x, 1.0) ** (3 - n)) / (n - 3), 0.0)
    return inner + outer


def sample_ejecta_speeds(u, E_kin, M_ej, delta=1.0, n=10.0):
    """
    Inverse-CDF sampling of the broken power-law ejecta.

    Arguments
    ---------
    u : array-like in [0, 1)
        Uniform deviates, i.e. the target mass coordinates.
    """
    v_t = transition_velocity(E_kin, M_ej, delta, n)
    u = np.asarray(u, dtype=float)
    A = (3 - delta) * (n - 3) / (n - delta)
    u_t = A / (3 - delta)                       # mass fraction inside v_t
    x_in = (u * (3 - delta) / A) ** (1.0 / (3 - delta))
    x_out = (1.0 - (u - u_t) * (n - 3) / A) ** (1.0 / (3 - n))
    return v_t * np.where(u < u_t, x_in, x_out)


def fallback_fraction_isolated(E_kin, M_ej, M_bh, R_launch, delta=1.0, n=10.0):
    """Analytic ballistic fallback for an isolated BH: mass with v < v_esc(R_launch)."""
    return float(ejecta_cdf(np.sqrt(2 * G * M_bh / R_launch), E_kin, M_ej, delta, n))


def companion_capture_fraction_analytic(v, M_bh, M2, R2, a, R_launch):
    """
    Fraction of isotropic ejecta launched at speed v from the BH that hits the
    companion, treating the companion as static and the flight as a straight
    line with gravitational focusing:

        b^2 = R2^2 (1 + v_esc,2^2 / v_c^2),   f = (1 - sqrt(1 - b^2/a^2)) / 2,

    where v_c is the speed left after climbing out of the BH potential to a.
    Valid for v much larger than the orbital speed; the Monte Carlo is needed
    below that.
    """
    v = np.asarray(v, dtype=float)
    v_c2 = v ** 2 - 2 * G * M_bh * (1 / R_launch - 1 / a)
    with np.errstate(invalid="ignore", divide="ignore"):
        b2 = R2 ** 2 * (1 + 2 * G * M2 / R2 / v_c2)
        f = 0.5 * (1 - np.sqrt(np.clip(1 - b2 / a ** 2, 0, 1)))
    return np.where(v_c2 > 0, f, np.nan)


def roche_lobe_radius(q, a):
    """Eggleton (1983) Roche-lobe radius of a body with mass ratio q = M_self / M_other."""
    q13 = q ** (1 / 3)
    return a * 0.49 * q13 ** 2 / (0.6 * q13 ** 2 + np.log(1 + q13))


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------

@dataclass
class EjectaConfig:
    """
    Pre-explosion binary and explosion parameters (SI units).

    M_bh      : remnant black-hole mass
    M_ej      : ejecta mass (progenitor mass = M_bh + M_ej)
    M2, R2    : companion mass and radius at the time of the explosion
    a_pre     : pre-explosion separation (circular orbit)
    E_kin     : ejecta kinetic energy
    R_launch  : launch radius (progenitor radius)
    v_kick    : natal kick on the BH, vector in the binary frame
                (x: BH -> companion, y: orbital motion, z: orbital normal)
    velocity_model : 'broken_powerlaw' or 'fixed' (all particles at v_fixed)
    t_end_orbits   : integration length in pre-explosion orbital periods
    r_max_factor   : escape boundary in units of a_pre (around the COM)
    r_check_factor : BH absorbing sphere in units of R_launch
    cone_half_angle: if set [rad], launch only into a cone aimed at the
                     companion's position at the particle's arrival time (widened
                     by half the lead angle) and down-weight each superparticle
                     by its cone's solid-angle fraction (importance sampling for
                     wide binaries where R2/a is tiny). Outcome fractions are then
                     fractions of the total ejecta mass, but only cone material is
                     followed.
    """
    M_bh: float
    M_ej: float
    M2: float
    R2: float
    a_pre: float
    E_kin: float = 1e44
    R_launch: float = R_SUN
    v_kick: tuple = (0.0, 0.0, 0.0)
    delta: float = 1.0
    n: float = 10.0
    velocity_model: str = "broken_powerlaw"
    v_fixed: float = None
    n_particles: int = 20000
    batch_size: int = 400
    t_end_orbits: float = 5.0
    r_max_factor: float = 20.0
    r_check_factor: float = 0.5
    gr: bool = True
    cone_half_angle: float = None
    seed: int = 42

    @property
    def P_pre(self):
        return 2 * np.pi * np.sqrt(self.a_pre ** 3 / (G * (self.M_bh + self.M_ej + self.M2)))


@dataclass
class EjectaResult:
    config: EjectaConfig
    v_launch: np.ndarray      # launch speed in the progenitor frame [m/s]
    mass_coord: np.ndarray    # Lagrangian mass coordinate m(<v)/M_ej
    direction: np.ndarray     # (N, 3) unit launch vectors
    weight: np.ndarray        # mass per superparticle [kg]
    outcome: np.ndarray       # outcome code, see OUTCOME_NAMES
    t_outcome: np.ndarray     # time of impact / capture / removal [s]; t_end if still bound
    r_peri_bh: np.ndarray     # Newtonian pericentre wrt BH at fallback [m]; NaN otherwise
    v_impact: np.ndarray      # speed relative to companion at impact [m/s]; NaN otherwise
    binary_final: dict = field(default_factory=dict)

    def mass_fraction(self, outcome):
        """Fraction of the total ejecta mass ending in this outcome."""
        return self.weight[self.outcome == outcome].sum() / self.config.M_ej

    def fractions(self):
        return {name: self.mass_fraction(code) for code, name in OUTCOME_NAMES.items()}

    def save(self, path):
        np.savez(path, config=np.array(asdict(self.config), dtype=object),
                 binary_final=np.array(self.binary_final, dtype=object),
                 **{k: getattr(self, k) for k in
                    ("v_launch", "mass_coord", "direction", "weight", "outcome",
                     "t_outcome", "r_peri_bh", "v_impact")})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(config=EjectaConfig(**d["config"].item()),
                   binary_final=d["binary_final"].item(),
                   **{k: d[k] for k in ("v_launch", "mass_coord", "direction", "weight",
                                        "outcome", "t_outcome", "r_peri_bh", "v_impact")})


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _binary_state(cfg):
    """Pre-explosion COM-frame states of progenitor and companion."""
    sim = rebound.Simulation()
    sim.units = ("m", "s", "kg")
    sim.G = G
    sim.add(m=cfg.M_bh + cfg.M_ej)
    sim.add(m=cfg.M2, a=cfg.a_pre, e=0.0, primary=sim.particles[0])
    sim.move_to_com()
    prog, comp = sim.particles[0], sim.particles[1]
    return (np.array(prog.xyz), np.array(prog.vxyz)), (np.array(comp.xyz), np.array(comp.vxyz))


def _launch_states(cfg, rng):
    """Speeds, mass coordinates and launch directions for all superparticles."""
    N = cfg.n_particles
    # Stratified sampling in mass coordinate keeps the slow, fallback-prone tail resolved.
    u = (np.arange(N) + rng.random(N)) / N
    if cfg.velocity_model == "broken_powerlaw":
        v = sample_ejecta_speeds(u, cfg.E_kin, cfg.M_ej, cfg.delta, cfg.n)
    elif cfg.velocity_model == "fixed":
        v = np.full(N, float(cfg.v_fixed))
    else:
        raise ValueError(f"Unknown velocity_model '{cfg.velocity_model}'.")
    if cfg.cone_half_angle is None:
        d = rng.normal(size=(N, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        frac = np.ones(N)
    else:
        # Aim each particle's cap at where the companion will be when ejecta of
        # that speed arrive: the companion moves by (v_c - v_p) t_flight relative
        # to the progenitor frame the ejecta inherit. A fixed cap around the
        # initial line of centres misses it once v_orb / v exceeds R2 / a.
        (x_p, v_p), (x_c, v_c) = _binary_state(cfg)
        v_inf = np.sqrt(np.maximum(v ** 2 - 2 * G * cfg.M_bh / cfg.R_launch, (0.05 * v) ** 2))
        t_fl = np.linalg.norm(x_c - x_p) / v_inf
        target = (x_c - x_p)[None, :] + (v_c - v_p)[None, :] * t_fl[:, None]
        axis = target / np.linalg.norm(target, axis=1)[:, None]
        lead = np.arccos(np.clip(axis @ (x_c - x_p) / np.linalg.norm(x_c - x_p), -1, 1))
        # Margin of half the lead angle covers the error of the straight-line flight time.
        half = np.minimum(cfg.cone_half_angle + 0.5 * lead, np.pi)
        frac = 0.5 * (1 - np.cos(half))
        cos_t = 1 - rng.random(N) * (1 - np.cos(half))
        sin_t = np.sqrt(1 - cos_t ** 2)
        phi = 2 * np.pi * rng.random(N)
        local = np.column_stack([cos_t, sin_t * np.cos(phi), sin_t * np.sin(phi)])
        d = _rotate_x_to(local, axis)
    perm = rng.permutation(N)
    return v[perm], u[perm], d[perm], frac[perm]


def _rotate_x_to(vecs, axes):
    """Rotate each row of vecs by the rotation taking +x onto the matching row of axes."""
    ex = np.array([1.0, 0.0, 0.0])
    k = np.cross(ex, axes)
    s = np.linalg.norm(k, axis=1)
    c = axes @ ex
    out = vecs.copy()
    m = s > 1e-12
    k = k[m] / s[m, None]
    v = vecs[m]
    # Rodrigues' rotation formula.
    out[m] = (v * c[m, None] + np.cross(k, v) * s[m, None]
              + k * np.sum(k * v, axis=1)[:, None] * (1 - c[m, None]))
    out[~m & (c < 0)] *= -1
    return out


def _integrate_batch(args):
    """Worker: integrate one batch of superparticles; returns per-particle outcomes."""
    cfg_dict, ids, v, d = args
    cfg = EjectaConfig(**cfg_dict)
    (x_p, v_p), (x_c, v_c) = _binary_state(cfg)

    sim = rebound.Simulation()
    sim.units = ("m", "s", "kg")
    sim.G = G
    sim.integrator = "ias15"
    sim.collision = "line"
    r_check = cfg.r_check_factor * cfg.R_launch
    sim.add(m=cfg.M_bh, r=r_check, x=x_p[0], y=x_p[1], z=x_p[2],
            vx=v_p[0] + cfg.v_kick[0], vy=v_p[1] + cfg.v_kick[1], vz=v_p[2] + cfg.v_kick[2])
    sim.add(m=cfg.M2, r=cfg.R2, x=x_c[0], y=x_c[1], z=x_c[2], vx=v_c[0], vy=v_c[1], vz=v_c[2])
    sim.N_active = 2

    for i, vi, di in zip(ids, v, d):
        pos = x_p + cfg.R_launch * di
        vel = v_p + vi * di
        sim.add(x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2], hash=int(i) + 1)

    if cfg.gr:
        rebx = reboundx.Extras(sim)
        # Single-source 1PN (Anderson et al. 1975) around the BH; the companion's
        # field is Newtonian. gr_full (all pairs) is ~200x slower for no change
        # in outcomes at these separations.
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = C_LIGHT
        sim.particles[0].params["gr_source"] = 1

    n = len(ids)
    index = {int(i) + 1: k for k, i in enumerate(ids)}
    outcome = np.full(n, -1, dtype=int)
    t_out = np.full(n, np.nan)
    r_peri = np.full(n, np.nan)
    v_imp = np.full(n, np.nan)

    def resolve(sim_pointer, collision):
        s = sim_pointer.contents
        i, j = sorted((collision.p1, collision.p2))
        if j < 2:                     # BH-companion contact: leave to the caller
            return 0
        body, p = s.particles[i], s.particles[j]
        k = index[p.hash.value]
        dx = np.array(p.xyz) - np.array(body.xyz)
        dv = np.array(p.vxyz) - np.array(body.vxyz)
        if i == 1:
            outcome[k] = COMPANION
            v_imp[k] = np.linalg.norm(dv)
        else:
            mu = G * body.m
            E = 0.5 * dv @ dv - mu / np.linalg.norm(dx)
            if E >= 0:                # unbound flyby through the absorbing sphere
                return 0
            h2 = np.sum(np.cross(dx, dv) ** 2)
            ecc = np.sqrt(max(1 + 2 * E * h2 / mu ** 2, 0.0))
            outcome[k] = FALLBACK
            r_peri[k] = h2 / mu / (1 + ecc)
        t_out[k] = s.t
        return 2 if collision.p2 == j else 1

    sim.collision_resolve = resolve

    # Checkpoints: dense early (fast ejecta cross the orbit in ~ a/v), log-spaced later.
    t_end = cfg.t_end_orbits * cfg.P_pre
    t_first = 0.05 * cfg.a_pre / 1e7
    checkpoints = np.geomspace(t_first, t_end, 60)
    r_max = cfg.r_max_factor * cfg.a_pre
    M_tot = cfg.M_bh + cfg.M2

    for t_next in checkpoints:
        if sim.N <= 2:
            break
        sim.integrate(t_next, exact_finish_time=1)
        # Remove particles that have left the system on unbound trajectories.
        com = sim.com()
        r_com = np.array(com.xyz)
        v_com = np.array(com.vxyz)
        for idx in range(sim.N - 1, 1, -1):
            p = sim.particles[idx]
            x = np.array(p.xyz)
            if np.linalg.norm(x - r_com) < r_max:
                continue
            vv = np.array(p.vxyz) - v_com
            if 0.5 * vv @ vv - G * M_tot / np.linalg.norm(x - r_com) > 0:
                k = index[p.hash.value]
                outcome[k] = ESCAPED
                t_out[k] = sim.t
                sim.remove(index=idx)

    # Classify whatever is left at t_end.
    bh, comp = sim.particles[0], sim.particles[1]
    x_bh, v_bh = np.array(bh.xyz), np.array(bh.vxyz)
    x_c2, v_c2 = np.array(comp.xyz), np.array(comp.vxyz)
    sep = np.linalg.norm(x_bh - x_c2)
    rl_bh = roche_lobe_radius(cfg.M_bh / cfg.M2, sep)
    rl_c = roche_lobe_radius(cfg.M2 / cfg.M_bh, sep)
    com = sim.com()
    for idx in range(2, sim.N):
        p = sim.particles[idx]
        k = index[p.hash.value]
        x, vv = np.array(p.xyz), np.array(p.vxyz)
        t_out[k] = sim.t
        e_cases = []
        for xb, vb, m, rl in ((x_bh, v_bh, cfg.M_bh, rl_bh), (x_c2, v_c2, cfg.M2, rl_c)):
            dx, dv = x - xb, vv - vb
            E = 0.5 * dv @ dv - G * m / np.linalg.norm(dx)
            a_orb = -G * m / (2 * E) if E < 0 else np.inf
            h2 = np.sum(np.cross(dx, dv) ** 2)
            ecc = np.sqrt(max(1 + 2 * E * h2 / (G * m) ** 2, 0.0))
            e_cases.append(E < 0 and a_orb * (1 + ecc) < rl)
        dvc = vv - np.array(com.vxyz)
        E_bin = 0.5 * dvc @ dvc - G * cfg.M_bh / np.linalg.norm(x - x_bh) - G * cfg.M2 / np.linalg.norm(x - x_c2)
        if e_cases[0]:
            outcome[k] = BOUND_BH
        elif e_cases[1]:
            outcome[k] = BOUND_COMPANION
        elif E_bin < 0:
            outcome[k] = CIRCUMBINARY
        else:
            outcome[k] = ESCAPED

    o = bh.orbit(primary=comp) if sim.N >= 2 else None
    binary = {"a": o.a, "e": o.e, "P": o.P} if o is not None else {}
    return ids, outcome, t_out, r_peri, v_imp, binary


def run_ejecta(cfg, processes=None, verbose=True):
    """
    Run the ejecta experiment described by cfg and return an EjectaResult.

    Batches run in separate processes (REBOUND collision search is O(N^2)).
    Call this from a script guarded by ``if __name__ == '__main__':`` on macOS.
    """
    rng = np.random.default_rng(cfg.seed)
    v, u, d, frac = _launch_states(cfg, rng)
    N = cfg.n_particles
    ids = np.arange(N)
    batches = [(asdict(cfg), ids[s:s + cfg.batch_size], v[s:s + cfg.batch_size], d[s:s + cfg.batch_size])
               for s in range(0, N, cfg.batch_size)]

    outcome = np.empty(N, dtype=int)
    t_out = np.empty(N)
    r_peri = np.empty(N)
    v_imp = np.empty(N)
    binary = {}
    processes = processes or multiprocessing.cpu_count()
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as ex:
        for done, (bid, bo, bt, brp, bvi, bbin) in enumerate(ex.map(_integrate_batch, batches), 1):
            outcome[bid], t_out[bid], r_peri[bid], v_imp[bid] = bo, bt, brp, bvi
            binary = binary or bbin
            if verbose:
                print(f"\r  batch {done}/{len(batches)}", end="", flush=True)
    if verbose:
        print()

    return EjectaResult(config=cfg, v_launch=v, mass_coord=u, direction=d,
                        weight=cfg.M_ej * frac / N, outcome=outcome, t_outcome=t_out,
                        r_peri_bh=r_peri, v_impact=v_imp, binary_final=binary)


# ---------------------------------------------------------------------------
# Pollution of the companion
# ---------------------------------------------------------------------------

def captured_mass_by_element(result, layers, retention=1.0):
    """
    Mass of each element delivered to the companion, for a layered ejecta model.

    Arguments
    ---------
    result : EjectaResult
    layers : list of (m_outer, {element: mass_fraction})
        Onion-shell composition ordered from the centre outwards; m_outer is the
        outer Lagrangian mass coordinate of the shell (last one = 1.0).
    retention : float
        Fraction of impacting ejecta that stays bound to the companion
        (hydrodynamic ablation/stripping is not modelled; see Hirai+2018).

    Returns
    -------
    dict element -> captured mass [kg]
    """
    hit = result.outcome == COMPANION
    m = result.mass_coord[hit]
    w = result.weight[hit] * retention
    out = {}
    m_inner = 0.0
    for m_outer, comp in layers:
        sel = (m >= m_inner) & (m < m_outer)
        for el, X in comp.items():
            out[el] = out.get(el, 0.0) + X * w[sel].sum()
        m_inner = m_outer
    return out


def surface_enhancement(captured, M_mix, X_surface):
    """
    Logarithmic surface over-abundance after mixing the captured mass into a
    layer of mass M_mix (convective envelope), the analogue of the white-dwarf
    convection-zone dilution:

        Delta[X] = log10( (X_surf M_mix + M_cap,X) / (X_surf (M_mix + M_cap)) )

    Arguments
    ---------
    captured : dict element -> captured mass [kg]
    M_mix : float
        Mixing mass [kg].
    X_surface : dict element -> pre-pollution surface mass fraction
    """
    M_cap = sum(captured.values())
    return {el: np.log10((X_surface[el] * M_mix + captured.get(el, 0.0)) /
                         (X_surface[el] * (M_mix + M_cap)))
            for el in X_surface}
