"""baselines.py : Round-Robin, First-Fit, First-Fit Decreasing."""
import numpy as np


def _fits(L, K, j, dem):
    return np.all(L[j] + dem <= K[j] + 1e-9)


def round_robin(inst):
    """RR sur le cluster complet de m serveurs ; saute un serveur si le conteneur n'y tient pas."""
    a = -np.ones(inst.n, dtype=int)
    L = np.zeros_like(inst.K)
    ptr = 0
    for i in range(inst.n):
        for t in range(inst.m):
            j = (ptr + t) % inst.m
            if _fits(L, inst.K, j, inst.d[i]):
                a[i] = j; L[j] += inst.d[i]; ptr = (j + 1) % inst.m
                break
    return a


def first_fit(inst, order=None):
    a = -np.ones(inst.n, dtype=int)
    L = np.zeros_like(inst.K)
    for i in (range(inst.n) if order is None else order):
        for j in range(inst.m):
            if _fits(L, inst.K, j, inst.d[i]):
                a[i] = j; L[j] += inst.d[i]
                break
    return a


def first_fit_decreasing(inst):
    """Tri par taille = plus grande demande normalisée (heuristique standard en vector bin packing)."""
    size = (inst.d / inst.K[0]).max(1)
    return first_fit(inst, order=np.argsort(-size, kind="stable"))
