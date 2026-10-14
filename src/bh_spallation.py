"""
Spallogenic nuclides from black-hole hot accretion flows: the black-hole analogue of
the polluted-white-dwarf beryllium argument of Doyle, Desch & Young (2021).

Doyle et al. relate a white dwarf's Be excess to the MeV-proton fluence F its icy
parent body received in a giant planet's radiation belt, n_Be/n_O = sigma F. Around
a quiescent black hole the radiation belt is the hot, two-temperature accretion flow
(ADAF), whose ions sit near the virial temperature (tens of MeV within ~10 r_s).
Here SERPENS follows ions evaporated from that flow:

  1. Source  : ions launched from a sphere of radius r0 with the SERPENS thermal
               (outward Maxwellian) mode at the local ion temperature, plus the
               flow's sub-Keplerian rotation.
  2. Dynamics: REBOUND IAS15 + REBOUNDx 1PN ("gr") around the black hole, with the
               SERPENS absorbing boundary at the ISCO. Outcomes: PLUNGE (crossed the
               ISCO), RETAINED (bound to the flow), ESCAPED (unbound past r_out).
  3. Dose    : fast-proton and fast-alpha fluences accumulated along each path
               (plus the pre-launch inflow exposure), converted to Be/O (p + O,
               Doyle et al. 2021) and Li/H (alpha + alpha, Yi & Narayan 1997).
  4. Observables are computed in scripts/bh_spallation_systems.py.

Lengths are in units of the Schwarzschild radius r_s and times in r_s/c, so the
dynamics are mass-independent; fluences scale linearly with the Eddington-scaled
accretion rate mdot and are tabulated at mdot = 1.

References:
  * Doyle, Desch & Young (2021) ApJL 907, L35
  * Yi & Narayan (1997) ApJ 486, 363 -- Li production in quiescent SXT ADAFs
  * Narayan & Yi (1995) ApJ 452, 710; Narayan, Mahadevan & Quataert (1998)
    -- self-similar ADAF: v_r = c1 alpha v_K, T_i ~ 0.55 T_vir, Omega ~ 0.4 Omega_K
  * Read & Viola (1984) ADNDT 31, 359 -- alpha + alpha -> Li cross sections
"""

import concurrent.futures
import multiprocessing
from dataclasses import dataclass, asdict

import numpy as np
import rebound
import reboundx
from scipy.special import gammaincc, gamma

from src.spawner import random_pos

# Physical constants (cgs where fluences are concerned)
C_CGS = 2.99792458e10
M_P_MEV = 938.272
MEV = 1.602176634e-6          # erg

PLUNGE, RETAINED, ESCAPED = 0, 1, 2
OUTCOME_NAMES = {PLUNGE: "plunge", RETAINED: "retained", ESCAPED: "escaped"}

# Hot-flow model (Narayan & Yi 1995)
ALPHA_VISC = 0.1
C1 = 0.5
THETA_ION = 0.55 / 6.0        # kT_i = THETA_ION * m_p c^2 / x  (0.55 T_vir; ~86 MeV / x)
ROTATION = 0.4                # Omega / Omega_K
HE_TO_H = 0.085               # number ratio
O_TO_H = 4.9e-4               # Asplund+2009
CNO_TO_H = 8.3e-4             # C + N + O, Asplund+2009

# Spallation (representative cross sections)
E_TH_P = 10.0                 # MeV, p + O -> Be (Doyle et al. 2021)
E_TH_ALPHA = 35.0             # MeV lab, alpha + alpha -> Li threshold
SIGMA_BE = 1e-26              # cm^2, 16O(p,x)9Be (Doyle et al. 2021)
SIGMA_LI = 5e-26              # cm^2, alpha + alpha -> 6,7Li near peak (Read & Viola 1984)
SIGMA_LI_CNO = 1e-26          # cm^2, p + CNO -> 6,7Li (comparable to Be; Doyle et al. 2021)
SIGMA_DESTROY = 2e-25         # cm^2, generic p-induced destruction of light nuclei


# ---------------------------------------------------------------------------
# Hot-flow (ADAF) proton field
# ---------------------------------------------------------------------------

def ion_kT(x):
    """Ion temperature [MeV] at x = r / r_s."""
    return THETA_ION * M_P_MEV / np.asarray(x, dtype=float)


def inflow_speed(x):
    """Radial inflow speed in units of c: v_r = c1 alpha v_K, v_K = c / sqrt(2x)."""
    return C1 * ALPHA_VISC / np.sqrt(2 * np.asarray(x, dtype=float))


def number_density(x, mdot, M_bh_sun):
    """
    Proton number density [cm^-3] from mass conservation, n = Mdot / (4 pi r^2 v_r m_p),
    with Mdot = mdot * Mdot_Edd (eta = 0.1).
    """
    r_s = 2.953e5 * M_bh_sun                       # cm
    mdot_edd = 1.399e18 * M_bh_sun                 # g/s
    x = np.asarray(x, dtype=float)
    return mdot * mdot_edd / (4 * np.pi * (x * r_s) ** 2 * inflow_speed(x) * C_CGS * 1.6726e-24)


def fast_flux(x, mdot, M_bh_sun, species="p"):
    """
    Omnidirectional flux [cm^-2 s^-1] of ions above the spallation threshold in a
    Maxwellian at the local ion temperature.
    """
    kT = ion_kT(x)
    if species == "p":
        E_th, mass, abund = E_TH_P, M_P_MEV, 1.0
    else:
        E_th, mass, abund = E_TH_ALPHA, 4 * M_P_MEV, HE_TO_H
    y = E_th / kT
    frac = gammaincc(1.5, y)
    E_mean = kT * gammaincc(2.5, y) * gamma(2.5) / np.maximum(gammaincc(1.5, y) * gamma(1.5), 1e-300)
    v = C_CGS * np.sqrt(2 * E_mean / mass)
    return abund * number_density(x, mdot, M_bh_sun) * frac * v


def inflow_fluence(x0, x_max=1e3, n=4000):
    """
    Fluences (p, alpha) [cm^-2] accumulated by a parcel flowing in from x_max to x0
    at mdot = 1. Mass-independent: n ~ 1/M and dt ~ M cancel.
    """
    x = np.geomspace(x0, x_max, n)
    t_per_x = 1.0 / (inflow_speed(x) * C_CGS)      # s per r_s, times r_s below
    out = []
    for sp in ("p", "alpha"):
        # Use M = 1 Msun; dr = r_s dx.
        integrand = fast_flux(x, 1.0, 1.0, sp) * t_per_x * 2.953e5
        out.append(np.trapz(integrand, x))
    return tuple(out)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

@dataclass
class SpallationConfig:
    r0: float = 10.0          # launch radius [r_s]
    gr: bool = True
    n_particles: int = 20000
    batch_size: int = 400
    r_out: float = 1e3        # escape / bookkeeping radius [r_s]
    t_end: float = 2e4        # [r_s / c]
    seed: int = 1


def _launch(cfg, rng):
    """SERPENS thermal mode: uniform positions on the source sphere, outward Maxwellian speed."""
    np.random.seed(int(rng.integers(2**31)))
    pos, lat, lon = random_pos(cfg.r0, lat_dist="uniform", long_dist="uniform", n_samples=cfg.n_particles)
    # Speeds in units of c: sigma^2 = kT / m_p.
    sigma = np.sqrt(ion_kT(cfg.r0) / M_P_MEV)
    speed = sigma * np.sqrt(np.sum(rng.normal(size=(cfg.n_particles, 3)) ** 2, axis=1))
    rhat = pos / cfg.r0
    vel = speed[:, None] * rhat
    # Sub-Keplerian rotation of the flow about z.
    v_k = 1 / np.sqrt(2 * cfg.r0)
    phi_hat = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
    vel += (ROTATION * v_k * np.cos(lat))[:, None] * phi_hat
    return pos, vel


def _run_batch(args):
    cfg_dict, ids, pos, vel = args
    cfg = SpallationConfig(**cfg_dict)
    sim = rebound.Simulation()
    sim.G = 0.5               # GM = c^2 r_s / 2 with c = r_s = 1
    sim.integrator = "ias15"
    # Floor on the adaptive step: 1PN fails to converge for v ~ c near the ISCO and would
    # otherwise drive dt -> 0 indefinitely.
    sim.ri_ias15.min_dt = 1e-3
    sim.collision = "line"
    sim.add(m=1.0, r=3.0)     # black hole; collision radius = Schwarzschild ISCO
    sim.N_active = 1
    for i, p, v in zip(ids, pos, vel):
        sim.add(x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2], hash=int(i) + 1)
    if cfg.gr:
        rebx = reboundx.Extras(sim)
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = 1.0
        sim.particles[0].params["gr_source"] = 1

    n = len(ids)
    index = {int(i) + 1: k for k, i in enumerate(ids)}
    outcome = np.full(n, RETAINED)
    F_p = np.zeros(n)
    F_a = np.zeros(n)
    sin_lat = np.full(n, np.nan)
    v_inf = np.full(n, np.nan)
    alive = np.ones(n, bool)

    def resolve(sim_pointer, collision):
        s = sim_pointer.contents
        j = max(collision.p1, collision.p2)
        k = index[s.particles[j].hash.value]
        outcome[k] = PLUNGE
        alive[k] = False
        return 2 if collision.p2 == j else 1

    sim.collision_resolve = resolve

    # Fluence accumulation per step (mdot = 1, M = 1 Msun; dose is mass-independent).
    t_unit = 2.953e5 / C_CGS                         # seconds per (r_s / c) at 1 Msun
    buf = {}

    def heartbeat(sim_pointer):
        s = sim_pointer.contents
        N = s.N - 1
        if N <= 0:
            return
        if buf.get("N") != N:
            buf["xyz"] = np.zeros((N + 1, 3))
            buf["h"] = np.zeros(N + 1, dtype="uint32")
            buf["N"] = N
        s.serialize_particle_data(xyz=buf["xyz"], hash=buf["h"])
        r = np.linalg.norm(buf["xyz"][1:], axis=1)
        ks = np.fromiter((index[h] for h in buf["h"][1:]), int, N)
        dt = s.dt_last_done * t_unit
        F_p[ks] += fast_flux(r, 1.0, 1.0, "p") * dt
        F_a[ks] += fast_flux(r, 1.0, 1.0, "alpha") * dt

    sim.heartbeat = heartbeat

    for t_next in np.geomspace(10.0, cfg.t_end, 40):
        if sim.N <= 1:
            break
        sim.integrate(t_next, exact_finish_time=1)
        for idx in range(sim.N - 1, 0, -1):
            p = sim.particles[idx]
            r = np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
            if r < cfg.r_out:
                continue
            k = index[p.hash.value]
            v2 = p.vx ** 2 + p.vy ** 2 + p.vz ** 2
            if 0.5 * v2 - sim.G / r > 0:
                outcome[k] = ESCAPED
                sin_lat[k] = p.vz / np.sqrt(v2)
                v_inf[k] = np.sqrt(v2 - 2 * sim.G / r)
            # Bound material beyond r_out falls back into the flow: RETAINED.
            alive[k] = False
            sim.remove(index=idx)

    # Remaining particles: classify by (Newtonian) energy at t_end.
    for idx in range(1, sim.N):
        p = sim.particles[idx]
        k = index[p.hash.value]
        r = np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
        v2 = p.vx ** 2 + p.vy ** 2 + p.vz ** 2
        if 0.5 * v2 - sim.G / r > 0 and r > 10 * cfg.r0:
            outcome[k] = ESCAPED
            sin_lat[k] = p.vz / np.sqrt(v2)
            v_inf[k] = np.sqrt(v2 - 2 * sim.G / r)
    return ids, outcome, F_p, F_a, sin_lat, v_inf


@dataclass
class SpallationResult:
    config: SpallationConfig
    outcome: np.ndarray
    F_p_path: np.ndarray      # fast-proton fluence after launch [cm^-2], mdot = 1
    F_a_path: np.ndarray      # fast-alpha fluence after launch [cm^-2], mdot = 1
    F_p_inflow: float         # pre-launch inflow fluence [cm^-2], mdot = 1
    F_a_inflow: float
    sin_lat: np.ndarray       # asymptotic latitude of escapers (sin)
    v_inf: np.ndarray         # asymptotic speed of escapers [c]

    def fraction(self, code):
        return np.mean(self.outcome == code)

    def escaped(self):
        return self.outcome == ESCAPED

    def be_to_o(self, mdot):
        """Be/O of each parcel (Doyle et al. 2021: n_Be/n_O = sigma F), linear regime."""
        return SIGMA_BE * mdot * (self.F_p_path + self.F_p_inflow)

    def li_to_h(self, mdot):
        """Li/H of each parcel from alpha + alpha (Yi & Narayan 1997) and p + CNO, linear regime."""
        return mdot * (HE_TO_H * SIGMA_LI * (self.F_a_path + self.F_a_inflow)
                       + CNO_TO_H * SIGMA_LI_CNO * (self.F_p_path + self.F_p_inflow))

    def destruction_depth(self, mdot):
        """sigma_destroy * F: linear-regime yields are valid while this is << 1."""
        return SIGMA_DESTROY * mdot * (self.F_p_path + self.F_p_inflow)

    def equatorial_enhancement(self, s=0.3):
        """Escaper density near the orbital plane relative to isotropic, P(|sin lat| < s) / s."""
        e = self.escaped()
        if e.sum() == 0:
            return np.nan
        return np.mean(np.abs(self.sin_lat[e]) < s) / s

    def save(self, path):
        np.savez(path, config=np.array(asdict(self.config), dtype=object), outcome=self.outcome,
                 F_p_path=self.F_p_path, F_a_path=self.F_a_path, F_p_inflow=self.F_p_inflow,
                 F_a_inflow=self.F_a_inflow, sin_lat=self.sin_lat, v_inf=self.v_inf)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(config=SpallationConfig(**d["config"].item()), outcome=d["outcome"],
                   F_p_path=d["F_p_path"], F_a_path=d["F_a_path"], F_p_inflow=float(d["F_p_inflow"]),
                   F_a_inflow=float(d["F_a_inflow"]), sin_lat=d["sin_lat"], v_inf=d["v_inf"])


def run_spallation(cfg, processes=None, verbose=True):
    """Run the hot-flow evaporation experiment. Call from a __main__-guarded script."""
    rng = np.random.default_rng(cfg.seed)
    pos, vel = _launch(cfg, rng)
    N = cfg.n_particles
    ids = np.arange(N)
    batches = [(asdict(cfg), ids[s:s + cfg.batch_size], pos[s:s + cfg.batch_size], vel[s:s + cfg.batch_size])
               for s in range(0, N, cfg.batch_size)]
    outcome = np.empty(N, int)
    F_p, F_a = np.empty(N), np.empty(N)
    sin_lat, v_inf = np.empty(N), np.empty(N)
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes or multiprocessing.cpu_count(),
                                                mp_context=ctx) as ex:
        for done, (b, o, fp, fa, sl, vi) in enumerate(ex.map(_run_batch, batches), 1):
            outcome[b], F_p[b], F_a[b], sin_lat[b], v_inf[b] = o, fp, fa, sl, vi
            if verbose:
                print(f"\r  batch {done}/{len(batches)}", end="", flush=True)
    if verbose:
        print()
    Fp_in, Fa_in = inflow_fluence(cfg.r0)
    return SpallationResult(cfg, outcome, F_p, F_a, Fp_in, Fa_in, sin_lat, v_inf)


# ---------------------------------------------------------------------------
# Analytic outcome classification in a central potential
# ---------------------------------------------------------------------------

def classify_central(pos, vel, metric="schwarzschild", x_isco=3.0):
    """
    Outcome of each launched particle around an isolated black hole from its
    conserved energy and angular momentum (units r_s = c = 1).

    A particle PLUNGEs if it can reach x_isco moving inward (same absorbing boundary
    as SERPENS), ESCAPEs if it can reach infinity, and is RETAINED otherwise.

    metric : 'schwarzschild' -- exact geodesics; launch speeds are taken as measured
                                by a static observer: E = gamma sqrt(1 - 1/x),
                                L = gamma x v_t, V^2 = (1 - 1/x)(1 + L^2/x^2)
             'newtonian'     -- E = v^2/2 - 1/(2x), V = -1/(2x) + L^2/(2x^2)
    """
    x = np.linalg.norm(pos, axis=1)
    rhat = pos / x[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    v_t = np.linalg.norm(vel - v_r[:, None] * rhat, axis=1)
    v2 = np.minimum(np.sum(vel ** 2, axis=1), 1 - 1e-12)

    if metric == "schwarzschild":
        g = 1 / np.sqrt(1 - v2)
        E = (g * np.sqrt(1 - 1 / x)) ** 2         # compare E^2 with V^2
        L = g * x * v_t
        V = lambda y, L: (1 - 1 / y) * (1 + L ** 2 / y ** 2)
        E_inf = 1.0
    elif metric == "newtonian":
        E = 0.5 * v2 - 0.5 / x
        L = x * v_t
        V = lambda y, L: -0.5 / y + 0.5 * L ** 2 / y ** 2
        E_inf = 0.0
    else:
        raise ValueError(metric)

    out = np.empty(len(x), int)
    for k in range(len(x)):
        y_out = np.geomspace(x[k], 1e5, 400)
        y_in = np.linspace(x_isco, x[k], 200)
        barrier_out = max(V(y_out, L[k]).max(), E_inf)
        barrier_in = V(y_in, L[k]).max()
        can_escape = E[k] >= barrier_out
        can_plunge = E[k] >= barrier_in
        if v_r[k] >= 0:
            out[k] = ESCAPED if can_escape else (PLUNGE if can_plunge else RETAINED)
        else:
            out[k] = PLUNGE if can_plunge else (ESCAPED if can_escape else RETAINED)
    return out


def inflow_yields(x0, mdot, x_max=1e3, n=4000):
    """
    Light-element yields of a parcel flowing in from x_max to x0 at accretion rate mdot,
    integrating production and proton-induced destruction:

        d(Li/H)  = He/H sigma_Li dF_alpha + CNO/H sigma_Li,CNO dF_p - sigma_d (Li/H) dF_p
        d(O/O0)  = -sigma_d (O/O0) dF_p
        d(Be/O0) = sigma_Be (O/O0) dF_p - sigma_d (Be/O0) dF_p

    In the low-dose limit Be/O0 -> sigma_Be F_p, the relation of Doyle et al. (2021).

    Returns dict with Li_H, Be_O0 (per initial O), F_p, F_alpha.
    """
    x = np.geomspace(x_max, x0, n)                           # inward
    dt_dx = 2.953e5 / (inflow_speed(x) * C_CGS)              # s per unit x at 1 Msun
    dFp = mdot * fast_flux(x, 1.0, 1.0, "p") * dt_dx
    dFa = mdot * fast_flux(x, 1.0, 1.0, "alpha") * dt_dx
    dx = np.abs(np.diff(x))
    li = be = 0.0
    o = 1.0
    Fp = Fa = 0.0
    for k in range(n - 1):
        fp = 0.5 * (dFp[k] + dFp[k + 1]) * dx[k]
        fa = 0.5 * (dFa[k] + dFa[k + 1]) * dx[k]
        decay = np.exp(-SIGMA_DESTROY * fp)
        li = li * decay + HE_TO_H * SIGMA_LI * fa + CNO_TO_H * o * SIGMA_LI_CNO * fp
        be = be * decay + SIGMA_BE * o * fp
        o *= decay
        Fp += fp
        Fa += fa
    return {"Li_H": li, "Be_O0": be, "F_p": Fp, "F_alpha": Fa}
