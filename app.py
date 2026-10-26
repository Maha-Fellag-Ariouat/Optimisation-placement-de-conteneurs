import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core import generate, metrics
from baselines import round_robin, first_fit, first_fit_decreasing
from sa import simulated_annealing
from exact import solve_exact

st.set_page_config(page_title="HPC Container Placement", layout="wide")
st.title("Optimisation du placement de conteneurs — HPC")
st.caption("Visualisation des allocations réelles produites par les algorithmes du projet.")

ALL_ALGOS = ["RoundRobin", "FirstFit", "FFD", "SA", "Exact"]
EXACT_MAX_N = 30  # au-delà, le MILP devient trop lent pour une interface interactive

with st.sidebar:
    st.header("Paramètres")
    n = st.select_slider("Nombre de conteneurs", options=[10, 14, 18, 20, 30, 50, 100, 150, 200], value=50)
    profile = st.selectbox("Profil", ["uniform", "correlated", "mixed"], index=2)
    theta = st.slider("θ : taux de remplissage maximal", 0.50, 1.00, 1.00, 0.05,
                      help="Capacité utile = θ × capacité nominale (contrainte C2 du modèle).")
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=0, step=1)
    iters = st.select_slider("Itérations SA", options=[1000, 5000, 10000, 20000, 40000, 60000], value=10000)
    with_exact = st.checkbox(f"Calculer l'optimum exact (n ≤ {EXACT_MAX_N})", value=False,
                             disabled=int(n) > EXACT_MAX_N,
                             help="Solveur MILP : minimise k uniquement (B n'est pas optimisé).")
    selected = st.multiselect("Algorithmes à afficher", ALL_ALGOS,
                              default=["FirstFit", "FFD", "SA"])
    run_btn = st.button("Lancer l'expérience", type="primary", use_container_width=True)


def run_experiment(n, profile, seed, iters, theta, with_exact):
    inst = generate(int(n), profile, int(seed), theta=float(theta))
    results = {}
    for name, func in [("RoundRobin", round_robin), ("FirstFit", first_fit),
                       ("FFD", first_fit_decreasing)]:
        t0 = time.perf_counter()
        alloc = func(inst)
        results[name] = {"allocation": np.asarray(alloc, dtype=int),
                         "metrics": metrics(inst, alloc),
                         "time": time.perf_counter() - t0}
    alloc, elapsed = simulated_annealing(inst, iters=int(iters), seed=int(seed))
    results["SA"] = {"allocation": np.asarray(alloc, dtype=int),
                     "metrics": metrics(inst, alloc), "time": elapsed}
    if with_exact and int(n) <= EXACT_MAX_N:
        t0 = time.perf_counter()
        alloc, status = solve_exact(inst, time_limit=30)
        if alloc is not None:
            results["Exact"] = {"allocation": np.asarray(alloc, dtype=int),
                                "metrics": metrics(inst, alloc),
                                "time": time.perf_counter() - t0, "status": status}
    return inst, results


if "experiment" not in st.session_state:
    st.session_state.experiment = None

if run_btn or st.session_state.experiment is None:
    with st.spinner("Exécution des algorithmes…"):
        try:
            st.session_state.experiment = run_experiment(n, profile, seed, iters, theta, with_exact)
            st.session_state.settings = (n, profile, seed, iters, theta)
        except Exception as exc:
            st.error(f"Erreur pendant l'exécution : {exc}")
            st.info("Vérifie que app.py est dans le même dossier que core.py, baselines.py, sa.py et exact.py.")
            st.stop()

inst, results = st.session_state.experiment
settings = st.session_state.get("settings", (n, profile, seed, iters, theta))
th = float(inst.theta)
LB = inst.lower_bound()
st.info(f"Instance affichée : {settings[0]} conteneurs · profil {settings[1]} · seed {settings[2]} · "
        f"SA {settings[3]} itérations · θ = {th:.2f} · borne inférieure LB = {LB} serveurs · "
        f"cluster de {inst.m} serveurs")

available = [x for x in ALL_ALGOS if x in selected and x in results]
if "Exact" in selected and "Exact" not in results:
    st.caption("L'optimum exact n'a pas été calculé pour cette expérience "
               f"(coche l'option dans la barre latérale, n ≤ {EXACT_MAX_N}, puis relance).")
if not available:
    st.warning("Sélectionne au moins un algorithme disponible.")
    st.stop()

summary = []
for name in available:
    m = results[name]["metrics"]
    k = m.get("k")
    summary.append({"Algorithme": name,
                    "Serveurs actifs (k)": k,
                    "Écart à LB": (k - LB) / LB if k is not None else None,
                    "Écart-type (B)": m.get("B"),
                    "Util. max (CPU/RAM/BW)": " / ".join(f"{x:.0%}" for x in m["util_max"]),
                    "Faisable": m.get("feasible"),
                    "Temps (s)": results[name]["time"]})
df_sum = pd.DataFrame(summary)
st.subheader("1. Comparaison")
st.dataframe(df_sum.round(4), use_container_width=True, hide_index=True)

# Gain relatif par rapport aux baselines classiques (section « Métriques » du modèle)
base = {a: results[a]["metrics"]["k"] for a in ("RoundRobin", "FirstFit") if a in results}
if "SA" in results and base:
    ksa = results["SA"]["metrics"]["k"]
    cols = st.columns(len(base) + 1)
    for c, (a, kb) in zip(cols, base.items()):
        c.metric(f"Gain SA vs {a} (serveurs)", f"{(kb - ksa) / kb:.1%}", f"{kb - ksa:+d} serveurs", delta_color="normal")
    if "FFD" in results:
        bs, bf = results["SA"]["metrics"]["B"], results["FFD"]["metrics"]["B"]
        cols[-1].metric("Réduction de B (SA vs FFD)", f"{(bf - bs) / bf:.0%}" if bf > 0 else "—")
if "Exact" in results:
    st.caption(f"Optimum exact : k = {results['Exact']['metrics']['k']} "
               f"({results['Exact'].get('status', '')}). Le solveur minimise k seulement : "
               "son B n'est pas comparable à celui du SA.")

Kn = np.asarray(inst.Knom)   # capacités nominales : u_jr = charge / K_nominal ∈ [0, θ]
D = np.asarray(inst.d)
resource_names = ["CPU", "RAM", "Bande passante"][:Kn.shape[1]]
tabs = st.tabs(available)

for tab, name in zip(tabs, available):
    with tab:
        result = results[name]
        allocation = result["allocation"]
        if len(allocation) != len(D):
            st.error(f"{name} a retourné {len(allocation)} affectations pour {len(D)} conteneurs.")
            continue
        if np.any(allocation >= len(Kn)):
            st.error(f"{name} a retourné des indices de serveurs invalides.")
            continue
        unplaced = int((allocation < 0).sum())
        if unplaced:
            st.warning(f"{unplaced} conteneur(s) non placé(s) par {name} (cluster saturé) : "
                       "ils sont exclus des charges ci-dessous.")
        placed = allocation >= 0
        loads = np.zeros_like(Kn, dtype=float)
        np.add.at(loads, allocation[placed], D[placed])
        active = np.where(loads.sum(axis=1) > 0)[0]
        st.markdown(f"### Placement avec {name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Serveurs actifs", len(active), f"LB = {LB}", delta_color="off")
        c2.metric("Conteneurs placés", int(placed.sum()))
        c3.metric("Faisable", str(result["metrics"].get("feasible", "à vérifier")))
        c4.metric("B (écart-type moyen)", f"{result['metrics']['B']:.3f}")

        st.markdown("#### Vue du cluster")
        for start in range(0, len(active), 4):
            cols = st.columns(4)
            for col, j in zip(cols, active[start:start + 4]):
                with col:
                    ids = np.where(allocation == j)[0]
                    ratios = loads[j] / Kn[j]
                    st.markdown(f"**Serveur S{j}** · {len(ids)} conteneur(s)")
                    st.progress(float(np.clip(ratios.max(), 0, 1)),
                                text=f"Util. max : {ratios.max():.0%} (limite θ = {th:.0%})")
                    st.caption("CPU / RAM / Réseau : " + " / ".join(f"{x:.0%}" for x in ratios))
                    st.write(", ".join(f"C{i}" for i in ids) if len(ids) else "Vide")

        st.markdown("#### Utilisation des ressources par serveur")
        if len(active):
            ratios = loads[active] / Kn[active]
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(active))
            width = 0.8 / max(1, len(resource_names))
            for r, label in enumerate(resource_names):
                ax.bar(x + (r - (len(resource_names) - 1) / 2) * width, ratios[:, r], width, label=label)
            ax.axhline(th, color="red", linestyle="--", linewidth=1, label=f"θ = {th:.2f}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"S{j}" for j in active], rotation=45)
            ax.set_ylabel("Taux d'utilisation u_jr")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Utilisation des ressources — {name}")
            ax.legend(ncol=4, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        df = pd.DataFrame({"conteneur": [f"C{i}" for i in range(len(allocation))],
                           "serveur": [f"S{j}" if j >= 0 else "non placé" for j in allocation]})
        st.download_button(f"Exporter les affectations {name} (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"placement_{name.lower()}.csv", mime="text/csv",
                           key=f"download_{name}")

st.caption("B = moyenne sur CPU/RAM/BW de l'écart-type des taux u_jr = charge / capacité nominale, "
           "calculé sur les serveurs actifs (core.metrics).")