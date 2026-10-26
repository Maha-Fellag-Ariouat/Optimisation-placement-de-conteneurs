"""exact.py : solveur exact du modèle du document (MILP, HiGHS via SciPy), avec cassure de symétrie.
Variables : x_ij (n*m) puis y_j (m).  Aucune dépendance autre que SciPy >= 1.9."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def solve_exact(inst, time_limit=60, sym_break=True):
    n, m = inst.n, inst.m
    nv = n * m + m
    X = lambda i, j: i * m + j
    Y = lambda j: n * m + j
    c = np.zeros(nv); c[n * m:] = inst.cost                               # Z1 = sum c_j y_j
    rows, lo, hi = [], [], []
    A = lil_matrix((n + m * 3 + n * m + (m - 1 if sym_break else 0), nv))
    k = 0
    for i in range(n):                                                    # C1 : affectation unique
        for j in range(m): A[k, X(i, j)] = 1
        lo.append(1); hi.append(1); k += 1
    for j in range(m):                                                    # C2 : capacités utiles
        for r in range(3):
            for i in range(n): A[k, X(i, j)] = inst.d[i, r]
            A[k, Y(j)] = -inst.K[j, r]
            lo.append(-np.inf); hi.append(0); k += 1
    for i in range(n):                                                    # C3 : x_ij <= y_j
        for j in range(m):
            A[k, X(i, j)] = 1; A[k, Y(j)] = -1
            lo.append(-np.inf); hi.append(0); k += 1
    if sym_break:                                                         # y_j >= y_{j+1}
        for j in range(m - 1):
            A[k, Y(j)] = 1; A[k, Y(j + 1)] = -1
            lo.append(0); hi.append(np.inf); k += 1
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if res.x is None:
        return None, res.message
    xs = res.x[:n * m].reshape(n, m)
    return xs.argmax(1), ("Optimal" if res.status == 0 else res.message)
