"""sa.py : Recuit simulé lexicographique à deux phases.

Idée clé (amélioration par rapport à un SA 'naïf') :
  - Z1 = k est un plateau (beaucoup de voisins ont le même k) -> SA aveugle.
    On ajoute un terme de guidage lisse (fitness de Falkenauer) qui récompense les
    serveurs très remplis / très vides, donc pousse à VIDER des serveurs.
  - Énergie  E = k + g(x)  avec g in [0,1)  => ordre lexicographique garanti (k d'abord).
  Phase 1 : g = 1 - F   (concentrer la charge pour libérer des serveurs)
  Phase 2 : g = B       (k figé/atteint, on équilibre la charge : objectif secondaire)
Recherche toujours dans l'espace FAISABLE (relocate/swap rejetés s'ils violent une capacité).
"""
import math, time
import numpy as np
from baselines import first_fit_decreasing


def _energy(L, Kn, phase):
    active = L.sum(1) > 0
    k = int(active.sum())
    u = L[active] / Kn[active]
    if phase == 1:
        fill = u.max(1)                       # remplissage = ressource la plus chargée
        F = float((fill ** 2).sum() / k)
        return k + (1.0 - F) * 0.999
    return k + float(u.std(0).mean()) * 1.99 * 0.5   # B <= 0.5 donc < 1


def simulated_annealing(inst, iters=60000, T0=0.5, alpha=None, phase1_frac=0.7, seed=0, init=None, history=False):
    rng = np.random.default_rng(seed)
    a = (init if init is not None else first_fit_decreasing(inst)).copy()
    assert (a >= 0).all(), "solution initiale incomplète"
    n, K, d = inst.n, inst.K, inst.d
    Kn = inst.Knom
    L = np.zeros_like(K); np.add.at(L, a, d)
    members = [set(np.where(a == j)[0].tolist()) for j in range(inst.m)]
    phase = 1
    E = _energy(L, Kn, phase)
    best_a, best_key = a.copy(), None

    def key(L):  # clé lexicographique (k, B) pour mémoriser le meilleur
        act = L.sum(1) > 0
        u = L[act] / Kn[act]
        return (int(act.sum()), float(u.std(0).mean()))

    best_key = key(L)
    T_end = 1e-4
    alpha = alpha or (T_end / T0) ** (1.0 / iters)
    T, hist = T0, []
    t0 = time.time()
    for it in range(iters):
        if phase == 1 and it >= phase1_frac * iters:
            phase = 2
            a, L = best_a.copy(), np.zeros_like(K); np.add.at(L, a, d)
            members = [set(np.where(a == j)[0].tolist()) for j in range(inst.m)]
            E = _energy(L, Kn, phase); T = T0 * 0.2
            alpha = (T_end / T) ** (1.0 / max(1, iters - it))
        active = np.where(L.sum(1) > 0)[0]
        r = rng.random()
        if r < 0.55:                                             # relocate
            if phase == 1 and rng.random() < 0.3:                # viser le serveur le moins chargé
                src = active[np.argmin((L[active] / K[active]).max(1))]
            else:
                src = rng.choice(active)
            i = rng.choice(list(members[src]))
            dst = rng.choice(active)
            if dst == src or np.any(L[dst] + d[i] > K[dst] + 1e-9):
                T *= alpha; continue
            L[src] -= d[i]; L[dst] += d[i]
            En = _energy(L, Kn, phase)
            if En <= E or rng.random() < math.exp(-(En - E) / T):
                E = En; members[src].remove(i); members[dst].add(i); a[i] = dst
            else:
                L[src] += d[i]; L[dst] -= d[i]
        else:                                                    # swap
            if len(active) < 2: continue
            s1, s2 = rng.choice(active, 2, replace=False)
            i, j = rng.choice(list(members[s1])), rng.choice(list(members[s2]))
            n1, n2 = L[s1] - d[i] + d[j], L[s2] - d[j] + d[i]
            if np.any(n1 > K[s1] + 1e-9) or np.any(n2 > K[s2] + 1e-9):
                T *= alpha; continue
            o1, o2 = L[s1].copy(), L[s2].copy()
            L[s1], L[s2] = n1, n2
            En = _energy(L, Kn, phase)
            if En <= E or rng.random() < math.exp(-(En - E) / T):
                E = En; members[s1].remove(i); members[s2].remove(j)
                members[s1].add(j); members[s2].add(i); a[i], a[j] = s2, s1
            else:
                L[s1], L[s2] = o1, o2
        kk = key(L)
        if kk < best_key:
            best_key, best_a = kk, a.copy()
        if history and it % 500 == 0:
            hist.append((it, kk[0], kk[1]))
        T *= alpha
    return (best_a, time.time() - t0, hist) if history else (best_a, time.time() - t0)