import numpy as np
from src.bh_ejecta import *
from src import bh_ejecta as be
if __name__ == "__main__":
    # rotation sanity: cap centred on axis
    ax = np.array([[0.3, -0.8, 0.52]]); ax /= np.linalg.norm(ax)
    loc = np.array([[1.0, 0, 0], [np.cos(.1), np.sin(.1), 0]])
    r = be._rotate_x_to(loc, np.repeat(ax, 2, 0)); print("rot ok:", np.allclose(r[0], ax[0]), np.isclose(np.arccos(r[1] @ ax[0]), .1))
    base = dict(M_bh=8*M_SUN, M_ej=3*M_SUN, M2=1.5*M_SUN, R2=6*R_SUN, a_pre=60*R_SUN, E_kin=1e43, t_end_orbits=1.0)
    for name, kw in (("isotropic", dict(n_particles=40000)), ("aimed cone", dict(n_particles=8000, cone_half_angle=4*6/60))):
        res = run_ejecta(EjectaConfig(**base, seed=11, **kw), verbose=False)
        k = np.sum(res.outcome == COMPANION)
        print(f"{name:10s} f_comp = {res.mass_fraction(COMPANION):.2e} +/- {res.mass_fraction(COMPANION)/np.sqrt(max(k,1)):.1e} (hits={k}); "
              f"geometric = {0.5*(1-np.sqrt(1-0.01)):.2e}")
