import numpy as np, warnings
from src.bh_spallation import *
from src import bh_spallation as bs
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    cfg = SpallationConfig(r0=10, n_particles=2000, batch_size=200)
    pos, vel = bs._launch(cfg, np.random.default_rng(cfg.seed))
    for m in ("newtonian", "schwarzschild"):
        o = classify_central(pos, vel, m)
        print(f"analytic {m:13s} plunge={np.mean(o==PLUNGE):.3f} retained={np.mean(o==RETAINED):.4f} escaped={np.mean(o==ESCAPED):.4f}")
    for gr in (False, True):
        r = run_spallation(SpallationConfig(r0=10, gr=gr, n_particles=2000, batch_size=200), verbose=False)
        print(f"SERPENS MC {'1PN' if gr else 'Newtonian':9s}   plunge={r.fraction(PLUNGE):.3f} retained={r.fraction(RETAINED):.4f} escaped={r.fraction(ESCAPED):.4f}")
        if not gr:
            o = classify_central(pos, vel, "newtonian")
            print("  per-particle agreement MC vs analytic Newtonian:", np.mean(o == r.outcome))
