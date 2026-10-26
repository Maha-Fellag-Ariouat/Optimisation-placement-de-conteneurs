"""validate.py : compare FFD et SA à l'optimum exact sur de petites instances."""
import numpy as np
from core import generate, metrics
from baselines import first_fit_decreasing
from sa import simulated_annealing
from exact import solve_exact

print(f"{'n':>3} {'profil':<8} {'seed':>4} {'LB':>3} | {'exact':>5} {'FFD':>4} {'SA':>3} | B_exact  B_SA")
for profile in ("uniform", "mixed"):
    for n in (10, 14, 18):
        for seed in range(2):
            inst = generate(n, profile, seed)
            ae, st = solve_exact(inst)
            me, mf = metrics(inst, ae), metrics(inst, first_fit_decreasing(inst))
            ms = metrics(inst, simulated_annealing(inst, iters=15000, seed=seed)[0])
            assert me["feasible"] and ms["feasible"]
            flag = " <-- SA > optimum" if ms["k"] > me["k"] else ""
            print(f"{n:>3} {profile:<8} {seed:>4} {inst.lower_bound():>3} | {me['k']:>5} {mf['k']:>4} {ms['k']:>3} | {me['B']:.3f}  {ms['B']:.3f}  [{st}]{flag}")
