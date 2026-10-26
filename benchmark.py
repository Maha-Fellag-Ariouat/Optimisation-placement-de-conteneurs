"""benchmark.py : mêmes instances pour tous les algorithmes (section 8 du modèle)."""
import time, sys
import numpy as np
from core import generate, metrics
from baselines import round_robin, first_fit, first_fit_decreasing
from sa import simulated_annealing


def run(n, profile, seed, iters):
    inst = generate(n, profile, seed)
    rows = {}
    for name, f in [("RoundRobin", round_robin), ("FirstFit", first_fit), ("FFD", first_fit_decreasing)]:
        t = time.time(); a = f(inst); rows[name] = (metrics(inst, a), time.time() - t)
    a, dt = simulated_annealing(inst, iters=iters, seed=seed)
    rows["SA"] = (metrics(inst, a), dt)
    return inst, rows


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    print(f"{'n':>4} {'profil':<10} {'LB':>3} | " + " | ".join(f"{a:^17}" for a in ("RoundRobin", "FirstFit", "FFD", "SA")))
    for profile in ("uniform", "mixed"):
        for n in (50, 100, 200):
            ks, Bs, lbs = {}, {}, []
            for seed in range(3):
                inst, rows = run(n, profile, seed, iters); lbs.append(inst.lower_bound())
                for k, (m, _) in rows.items():
                    assert m["feasible"], (k, "infaisable")
                    ks.setdefault(k, []).append(m["k"]); Bs.setdefault(k, []).append(m["B"])
            cells = " | ".join(f"k={np.mean(ks[a]):5.1f} B={np.mean(Bs[a]):.3f}" for a in ("RoundRobin", "FirstFit", "FFD", "SA"))
            print(f"{n:>4} {profile:<10} {np.mean(lbs):>5.1f} | {cells}")