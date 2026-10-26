"""core.py : instances, évaluation, métriques (fidèle à la modélisation : sections 3 à 8)."""
import numpy as np
from dataclasses import dataclass

RES = ("CPU", "RAM", "BW")


@dataclass
class Instance:
    d: np.ndarray   # (n, 3) besoins d_ir
    K: np.ndarray   # (m, 3) capacités UTILES = theta * K_nominal (contrainte C2)
    cost: np.ndarray  # (m,) coûts c_j (ici c_j = 1 : Z1 = k)
    Knom: np.ndarray = None  # (m, 3) capacités nominales, pour u_jr = charge / K_nominal
    theta: float = 1.0

    @property
    def n(self): return self.d.shape[0]
    @property
    def m(self): return self.K.shape[0]

    def lower_bound(self):
        """Borne inf. sur k (serveurs identiques) : max_r ceil(sum_i d_ir / K_r)."""
        return int(np.max(np.ceil(self.d.sum(0) / self.K[0])))


def generate(n, profile="mixed", seed=0, cap=100, slack=1.6, theta=1.0):
    """Génère une instance. profile : uniform | correlated | mixed (microservices cpu/ram/réseau-bound)."""
    rng = np.random.default_rng(seed)
    if profile == "uniform":
        d = rng.integers(5, 31, size=(n, 3))
    elif profile == "correlated":
        base = rng.integers(5, 31, size=(n, 1))
        d = np.clip(base + rng.integers(-3, 4, size=(n, 3)), 1, cap)
    elif profile == "mixed":
        d = rng.integers(3, 12, size=(n, 3))
        dom = rng.integers(0, 3, size=n)                 # ressource dominante
        d[np.arange(n), dom] = rng.integers(15, 41, size=n)
    else:
        raise ValueError(profile)
    lb = int(np.max(np.ceil(d.sum(0) / (theta * cap))))
    m = max(2, int(np.ceil(slack * lb)))                 # taille du cluster disponible
    Knom = np.full((m, 3), cap, dtype=float)
    return Instance(d.astype(float), theta * Knom, np.ones(m), Knom, theta)


def loads(inst, a):
    """Charge par serveur (m,3) pour une affectation a[i] = j."""
    L = np.zeros_like(inst.K)
    np.add.at(L, a, inst.d)
    return L


def is_feasible(inst, a):
    if np.any(a < 0) or np.any(a >= inst.m):
        return False
    return bool(np.all(loads(inst, a) <= inst.K + 1e-9))


def metrics(inst, a):
    """Z1, B, utilisations... (sections 7 et 8). a[i] = -1 signifie 'non placé'."""
    placed = a >= 0
    out = {"placed": int(placed.sum()), "feasible": bool(placed.all() and is_feasible(inst, a))}
    L = np.zeros_like(inst.K)
    np.add.at(L, a[placed], inst.d[placed])
    active = L.sum(1) > 0
    u = L[active] / inst.Knom[active]                   # u_jr in [0, theta]
    out["k"] = int(active.sum())
    out["cost"] = float(inst.cost[active].sum())
    out["sigma"] = u.std(0)                             # sigma_r (population, comme dans le doc)
    out["B"] = float(u.std(0).mean())
    out["util_mean"] = u.mean(0)
    out["util_max"] = u.max(0)
    return out