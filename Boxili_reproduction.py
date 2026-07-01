import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.simulation.mc_engine import run_simulation
from src.simulation.mc_engine import (
    NoCutoffPolicy, FixedCutoffPolicy, AdaptivePolicy,
)
from src.metrics.skr import skr_from_samples
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import suggest_t_max
from src.physical.werner_utils import secret_fraction_nat as _sf


def werner_to_fidelity(w):
    return (3 * w + 1) / 4


# ═══════════════════════════════════════════════════════════
#  Boxi Li's parameters (Table I)
# ═══════════════════════════════════════════════════════════
BOXILI_PARAMS = {
    "p_gen": 0.1,
    "p_swap": 0.5,
    "w0": 0.98,
    "t_coh": 400,
}

N_EPISODES = 200_000
SEED       = 42
# Episodes used for the cutoff-search sweep (same as evaluation — no halving)
N_SWEEP    = 50_000


def t_max_for(t_coh: float, best_n: int = 0) -> int:
    """
    Physically motivated t_max.
    Covers ages where w0^2 * exp(-2*age/t_coh) > w_threshold (SKR > 0).
    Formula: age_max = t_coh/2 * ln(w0^2 / w_threshold)
    We take that value + 20, also ensure best_n + 20, cap at 600.
    """
    import math
    w0          = BOXILI_PARAMS["w0"]
    w_threshold = 0.7476
    if w0 * w0 <= w_threshold:
        raw = 10
    else:
        raw = int(t_coh / 2.0 * math.log(w0 * w0 / w_threshold))
    t = max(raw + 20, best_n + 20, 30)
    return min(t, 600)


# ═══════════════════════════════════════════════════════════
#  BL Figure 1: SKR vs Cutoff (their Fig. 5 equivalent)
# ═══════════════════════════════════════════════════════════

def skr_boxili(delivery_times, w_out_array):
    """
    SKR convention Boxi Li: sf(E[w]) / E[T].
    Used ONLY to reproduce Li's published numbers for direct comparison.
    Do NOT use to compare fixed-cutoff vs adaptive MDP — the two estimators
    are inconsistent and will produce artefact differences.
    """
    mean_t = np.mean(delivery_times)
    mean_w = np.mean(w_out_array)
    return max(0.0, float(_sf(mean_w) / mean_t))


def skr_mc(delivery_times, w_out_array):
    """
    SKR via Monte Carlo average-reward estimator: E[sf(w)] / E[T].
    This is what the MDP is trained to maximise. Must be used consistently
    when comparing adaptive MDP vs fixed cutoff vs no-cutoff policies.
    """
    sf_vals = np.array([max(0.0, float(_sf(w))) for w in w_out_array])
    mean_sf = np.mean(sf_vals)
    mean_t  = np.mean(delivery_times)
    return float(mean_sf / mean_t)

def bl_table_numerical_comparison():
    """
    Compare E[T], E[w], SKR (convention BoxiLi) avec
    les valeurs de référence de l'article BoxiLi et al.
    Paramètres: p_gen=0.1, p_swap=0.5, w0=0.98, t_coh=400
    Tolérance acceptable: 5%
    """
    print("  [BL Table] Numerical comparison vs BoxiLi reference...")

    p_gen  = BOXILI_PARAMS["p_gen"]
    p_swap = BOXILI_PARAMS["p_swap"]
    w0     = BOXILI_PARAMS["w0"]
    t_coh  = BOXILI_PARAMS["t_coh"]

    # ── Valeurs de référence extraites de tutorial.ipynb BoxiLi ──
    # (à ajuster si tu lis des valeurs différentes dans le notebook)
    BL_REF = {
    "no_cutoff": {
        "mean_t": 14.74,   # E[max(T1,T2)] = 2/p - 1/(p(2-p)), p=0.1
        "mean_w": None,    # à lire dans tutorial.ipynb
        "skr":    None,    # à lire dans tutorial.ipynb
    },
    "cutoff_16": {
        "mean_t": None,    # à lire dans tutorial.ipynb
        "mean_w": None,    # à lire dans tutorial.ipynb
        "skr":    None,    # à lire dans tutorial.ipynb
    },
}

    results = {}
    
    # ── No cutoff ──
    mc = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        p_swap=p_swap, n_episodes=N_EPISODES, seed=SEED)
    results["no_cutoff"] = {
        "mean_t": float(np.mean(mc.delivery_times)),
        "mean_w": float(np.mean(mc.w_out_array)),
        "skr":    skr_boxili(mc.delivery_times.astype(float),
                             mc.w_out_array),
    }

    # ── Fixed cutoff n*=16 (BoxiLi reference cutoff) ──
    mc = run_simulation(
        FixedCutoffPolicy(16), p_gen, w0, t_coh,
        p_swap=p_swap, n_episodes=N_EPISODES, seed=SEED)
    results["cutoff_16"] = {
        "mean_t": float(np.mean(mc.delivery_times)),
        "mean_w": float(np.mean(mc.w_out_array)),
        "skr":    skr_boxili(mc.delivery_times.astype(float),
                             mc.w_out_array),
    }

    # ── Affichage table ──
    print(f"\n  {'Metric':<12} {'Case':<12} {'This work':>12}"
          f" {'BoxiLi ref':>12} {'Rel Err':>10}  Status")
    print(f"  {'-'*65}")

    for case in ["no_cutoff", "cutoff_16"]:
        for metric in ["mean_t", "mean_w", "skr"]:
            val = results[case][metric]
            ref = BL_REF[case][metric]
            if ref is not None and ref > 0:
                err = abs(val - ref) / ref
                ok  = err < 0.05
                status = "PASS ✓" if ok else "FAIL ✗"
                print(f"  {metric:<12} {case:<12} {val:>12.6f}"
                      f" {ref:>12.6f} {err:>10.4f}  [{status}]")
            else:
                print(f"  {metric:<12} {case:<12} {val:>12.6f}"
                      f" {'(no ref)':>12}  {'N/A':>10}")

    print()

def bl_fig6_mdp_policy_heatmap():
    """
    Heatmap de la politique MDP optimale.
    Axe X = age lien 2 (a2), Axe Y = age lien 1 (a1)
    Couleur = action : 0 (WAIT/blanc) ou 1 (SWAP/vert)
    Montre la frontière de décision optimale.
    """
    print("  [BL Fig 6] MDP policy heatmap...")

    p_gen  = BOXILI_PARAMS["p_gen"]
    p_swap = BOXILI_PARAMS["p_swap"]
    w0     = BOXILI_PARAMS["w0"]
    t_coh  = BOXILI_PARAMS["t_coh"]

    t_max = suggest_t_max(t_coh)
    t_max = min(t_max, 100)

    mdp = solve_mdp(
        p_gen=p_gen, w0=w0, t_coh=t_coh,
        p_swap=p_swap, t_max=t_max, verbose=False)

    policy = mdp["policy"]  # shape attendu: (t_max+1, t_max+1)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        policy, origin="lower",
        cmap="RdYlGn", aspect="equal",
        vmin=0, vmax=1,
        extent=[0, t_max, 0, t_max])

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["WAIT (0)", "SWAP (1)"], fontsize=11)

    # Ligne diagonale a1=a2 pour référence
    ax.plot([0, t_max], [0, t_max], "w--", lw=1.5,
            alpha=0.6, label="$a_1 = a_2$")

    ax.set_xlabel(r"Âge lien 2 ($a_2$)", fontsize=13)
    ax.set_ylabel(r"Âge lien 1 ($a_1$)", fontsize=13)
    ax.set_title(
        "Politique MDP optimale : action en fonction des âges\n"
        f"$p_{{gen}}={p_gen}$, $p_{{swap}}={p_swap}$, "
        f"$w_0={w0}$, $t_{{coh}}={t_coh}$",
        fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    plt.savefig("results/figures/bl_fig6_mdp_policy.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/bl_fig6_mdp_policy.png",
                dpi=200, bbox_inches="tight")
    print("    Saved bl_fig6_mdp_policy.pdf/png")
    plt.close()

def bl_fig1_skr_vs_cutoff():
    """
    SKR as a function of the memory cutoff time n*.
    Reproduces the key result: optimal cutoff dramatically
    improves SKR over no-cutoff.
    """
    print("  [BL Fig 1] SKR vs cutoff...")

    p_gen = BOXILI_PARAMS["p_gen"]
    p_swap = BOXILI_PARAMS["p_swap"]
    w0 = BOXILI_PARAMS["w0"]
    t_coh = BOXILI_PARAMS["t_coh"]

    cutoffs = list(range(1, 120))
    skr_list = []

    # Sweep fixed cutoffs — skr_mc for consistency with MDP objective
    for n_star in cutoffs:
        mc = run_simulation(
            FixedCutoffPolicy(n_star), p_gen=p_gen, w0=w0, t_coh=t_coh,
            p_swap=p_swap, n_episodes=N_SWEEP, seed=SEED)
        skr = skr_mc(mc.delivery_times.astype(float), mc.w_out_array)
        skr_list.append(skr)

    best_idx = np.argmax(skr_list)
    best_n = cutoffs[best_idx]
    best_skr = skr_list[best_idx]

    # No cutoff
    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen=p_gen, w0=w0, t_coh=t_coh, p_swap=p_swap,
        n_episodes=N_EPISODES, seed=SEED)
    skr_nocut = skr_mc(
        mc_nocut.delivery_times.astype(float), mc_nocut.w_out_array)

    # Adaptive MDP — physically grounded t_max, no arbitrary cap
    t_max = t_max_for(t_coh, best_n)
    mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh, p_swap=p_swap,
                    t_max=t_max, verbose=False)
    mc_adapt = run_simulation(
        AdaptivePolicy(mdp["policy"], t_max), p_gen, w0, t_coh, p_swap=p_swap,
        n_episodes=N_EPISODES, seed=SEED + 300)
    skr_adapt = skr_mc(
        mc_adapt.delivery_times.astype(float), mc_adapt.w_out_array)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(cutoffs, skr_list, "b.-", alpha=0.6, markersize=3,
            label="Fixed cutoff (varying $n^*$)")
    ax.axhline(skr_nocut, color="gray", ls="--", lw=2,
               label=f"No cutoff  [E[sf(w)]/E[T]={skr_nocut:.4e}]")
    ax.axhline(skr_adapt, color="red", ls="-", lw=2.5,
               label=f"Adaptive MDP  [E[sf(w)]/E[T]={skr_adapt:.4e}]")
    ax.axvline(best_n, color="blue", ls=":", lw=1.5, alpha=0.5)
    ax.annotate(f"$n^*_{{opt}}={best_n}$\nSKR={best_skr:.4e}",
                (best_n, best_skr), textcoords="offset points",
                xytext=(15, -10), fontsize=10, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue"))

    ax.set_xlabel("Memory cutoff time $n^*$ (time steps)", fontsize=13)
    ax.set_ylabel("Secret Key Rate (per time step)", fontsize=13)
    ax.set_title(
        "SKR vs Memory Cutoff\n"
        f"(Boxi Li params: $p_{{gen}}={p_gen}$, $p_{{swap}}={p_swap}$, "
        f"$w_0={w0}$, $t_{{coh}}={t_coh}$)",
        fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)

    plt.tight_layout()
    plt.savefig("results/figures/bl_fig1_skr_vs_cutoff.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/bl_fig1_skr_vs_cutoff.png",
                dpi=200, bbox_inches="tight")
    print(f"    Best fixed:  n*={best_n}, E[sf(w)]/E[T]={best_skr:.6e}")
    print(f"    No cutoff:   E[sf(w)]/E[T]={skr_nocut:.6e}")
    print(f"    Adaptive:    E[sf(w)]/E[T]={skr_adapt:.6e}")
    print(f"    Adaptive vs fixed: {100*(skr_adapt - best_skr)/best_skr:+.2f}%")
    # Li's own formula for reference (not used for comparison)
    skr_nocut_bl = skr_boxili(mc_nocut.delivery_times.astype(float), mc_nocut.w_out_array)
    skr_adapt_bl = skr_boxili(mc_adapt.delivery_times.astype(float), mc_adapt.w_out_array)
    print(f"    [Li formula] no-cutoff={skr_nocut_bl:.6e}, adaptive={skr_adapt_bl:.6e}")
    print("    Saved bl_fig1_skr_vs_cutoff.pdf/png")
    plt.close()

    return best_n


# ═══════════════════════════════════════════════════════════
#  BL Figure 2: PMF + Werner (their Fig. 4 equivalent)
# ═══════════════════════════════════════════════════════════

def bl_fig2_pmf_and_werner(best_n):
    """
    2×2 subplot: PMF, CDF, Werner vs time, Fidelity vs time.
    Matching Boxi Li's Fig. 4 layout.
    """
    print("  [BL Fig 2] PMF and Werner parameter...")

    p_gen = BOXILI_PARAMS["p_gen"]
    p_swap = BOXILI_PARAMS["p_swap"]
    w0 = BOXILI_PARAMS["w0"]
    t_coh = BOXILI_PARAMS["t_coh"]

    # Run three strategies
    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,p_swap=p_swap,
        n_episodes=N_EPISODES, seed=SEED)

    mc_fixed = run_simulation(
        FixedCutoffPolicy(best_n), p_gen, w0, t_coh,p_swap=p_swap,
        n_episodes=N_EPISODES, seed=SEED + 200)

    t_max = t_max_for(t_coh, best_n)
    mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh, p_swap=p_swap,
                    t_max=t_max, verbose=False)
    mc_adapt = run_simulation(
        AdaptivePolicy(mdp["policy"], t_max), p_gen, w0, t_coh, p_swap=p_swap,
        n_episodes=N_EPISODES, seed=SEED + 300)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Determine plot range
    max_t = int(np.percentile(mc_nocut.delivery_times, 99.5))
    max_t = min(max_t, 600)

    strategies = [
        (mc_nocut, "No cutoff", "gray", "--"),
        (mc_fixed, f"Fixed $n^*={best_n}$", "#4a90d9", "-"),
        (mc_adapt, "Adaptive MDP", "#d94a4a", "-"),
    ]

    # ── PMF (top-left) ──
    ax = axs[0][0]
    bins = np.arange(1, max_t + 1)
    for mc, label, color, ls in strategies:
        ax.hist(mc.delivery_times, bins=bins, alpha=0.35, density=True,
                color=color, label=label)
    ax.set_xlabel("Delivery time $T$", fontsize=11)
    ax.set_ylabel("PMF", fontsize=11)
    ax.set_title("Waiting Time Distribution (PMF)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── CDF (top-right) ──
    ax = axs[0][1]
    for mc, label, color, ls in strategies:
        sorted_t = np.sort(mc.delivery_times)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax.plot(sorted_t, cdf, color=color, lw=2, ls=ls, label=label)
    ax.set_xlabel("Delivery time $T$", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Cumulative Distribution (CDF)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_t)

    # ── Werner parameter vs delivery time (bottom-left) ──
    ax = axs[1][0]
    for mc, label, color, ls in strategies:
        # Bin by delivery time and compute mean Werner
        t_bins = np.arange(1, max_t + 1)
        w_means = []
        t_centers = []
        for t in t_bins:
            mask = mc.delivery_times == t
            if np.sum(mask) > 5:
                w_means.append(np.mean(mc.w_out_array[mask]))
                t_centers.append(t)
        ax.plot(t_centers, w_means, '.', color=color, alpha=0.4,
                markersize=2)
        # Smoothed line
        if len(t_centers) > 10:
            window = min(15, len(t_centers) // 5)
            if window > 1:
                w_smooth = np.convolve(w_means,
                    np.ones(window)/window, mode='valid')
                t_smooth = t_centers[window//2:window//2 + len(w_smooth)]
                ax.plot(t_smooth, w_smooth, color=color, lw=2,
                        ls=ls, label=label)

    ax.set_xlabel("Delivery time $T$", fontsize=11)
    ax.set_ylabel("Werner parameter $w$", fontsize=11)
    ax.set_title("Werner Parameter vs Delivery Time", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_t)

    # ── Fidelity vs delivery time (bottom-right) ──
    ax = axs[1][1]
    for mc, label, color, ls in strategies:
        t_bins = np.arange(1, max_t + 1)
        f_means = []
        t_centers = []
        for t in t_bins:
            mask = mc.delivery_times == t
            if np.sum(mask) > 5:
                f_means.append(np.mean(
                    werner_to_fidelity(mc.w_out_array[mask])))
                t_centers.append(t)
        ax.plot(t_centers, f_means, '.', color=color, alpha=0.4,
                markersize=2)
        if len(t_centers) > 10:
            window = min(15, len(t_centers) // 5)
            if window > 1:
                f_smooth = np.convolve(f_means,
                    np.ones(window)/window, mode='valid')
                t_smooth = t_centers[window//2:window//2 + len(f_smooth)]
                ax.plot(t_smooth, f_smooth, color=color, lw=2,
                        ls=ls, label=label)

    ax.axhline(0.8107, color="red", ls=":", lw=1, alpha=0.7,
               label="SKR threshold")
    ax.set_xlabel("Delivery time $T$", fontsize=11)
    ax.set_ylabel("Fidelity $F$", fontsize=11)
    ax.set_title("Fidelity vs Delivery Time", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_t)

    plt.suptitle(
        "Boxi Li Reproduction: Waiting Time & Quality Distributions\n"
        f"($p_{{gen}}={p_gen}$, $p_{{swap}}={p_swap}$, "
        f"$w_0={w0}$, $t_{{coh}}={t_coh}$)",
        fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/bl_fig2_pmf_werner.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/bl_fig2_pmf_werner.png",
                dpi=200, bbox_inches="tight")
    print("    Saved bl_fig2_pmf_werner.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  BL Figure 3: SKR vs t_coh (their Fig. 7 equivalent)
# ═══════════════════════════════════════════════════════════

def bl_fig3_skr_vs_tcoh():
    """
    SKR as a function of coherence time.
    Three lines: no cutoff, optimal fixed, adaptive MDP.
    Uses Boxi Li's p_gen=0.1, p_swap=0.5, w0=0.98.
    """
    print("  [BL Fig 3] SKR vs t_coh...")

    p_gen = BOXILI_PARAMS["p_gen"]
    p_swap = BOXILI_PARAMS["p_swap"]
    w0 = BOXILI_PARAMS["w0"]

    t_coh_values = [10, 20, 50, 100, 200, 400, 600, 1000]
    n_ep = N_EPISODES // 2

    skr_nocut = []
    skr_fixed = []
    skr_adapt = []
    best_ns = []

    for t_coh in t_coh_values:
        print(f"    t_coh={t_coh}...", end="", flush=True)

        # No cutoff
        mc = run_simulation(
            NoCutoffPolicy(), p_gen, w0, t_coh,p_swap=p_swap,
            n_episodes=n_ep, seed=SEED)
        skr_nocut.append(skr_mc(
            mc.delivery_times.astype(float), mc.w_out_array))

        # Best fixed — sweep to physically grounded max, skr_mc for consistency
        best_n, best_skr = 1, -np.inf
        sweep_max = t_max_for(t_coh)
        for n_star in range(1, sweep_max):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh, p_swap=p_swap,
                n_episodes=N_SWEEP, seed=SEED + 100)
            skr = skr_mc(mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        mc = run_simulation(
            FixedCutoffPolicy(best_n), p_gen, w0, t_coh, p_swap=p_swap,
            n_episodes=n_ep, seed=SEED + 200)
        skr_fixed.append(skr_mc(
            mc.delivery_times.astype(float), mc.w_out_array))
        best_ns.append(best_n)

        # Adaptive — t_max physically grounded
        t_max = t_max_for(t_coh, best_n)
        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh, p_swap=p_swap,
                t_max=t_max, verbose=False)
        mc = run_simulation(
            AdaptivePolicy(mdp["policy"], t_max), p_gen, w0, t_coh, p_swap=p_swap,
            n_episodes=n_ep, seed=SEED + 300)
        skr_adapt.append(skr_mc(
            mc.delivery_times.astype(float), mc.w_out_array))

        print(f" n*={best_n}, done")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t_coh_values, skr_nocut, "s--", color="gray", lw=2,
             markersize=7, label="No cutoff")
    ax1.plot(t_coh_values, skr_fixed, "o-", color="#4a90d9", lw=2,
             markersize=7, label="Optimal fixed cutoff")
    ax1.plot(t_coh_values, skr_adapt, "^-", color="#d94a4a", lw=2.5,
             markersize=8, label="Adaptive MDP")

    ax1.set_ylabel("Secret Key Rate", fontsize=13)
    ax1.set_title(
        "SKR vs Coherence Time (Boxi Li params)\n"
        f"$p_{{gen}}={p_gen}$, $p_{{swap}}={p_swap}$, $w_0={w0}$",
        fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(t_coh_values)
    ax1.set_xticklabels(t_coh_values, fontsize=9, rotation=45)

    # Improvement bars
    imp = [100 * (a - f) / f if f > 0 else 0
           for a, f in zip(skr_adapt, skr_fixed)]
    bar_width = np.min(np.diff(t_coh_values)) * 0.5
    ax2.bar(t_coh_values, imp, width=bar_width, color="#d94a4a",
            alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(t_coh_values)
    ax2.set_xticklabels(t_coh_values, fontsize=9, rotation=45)
    ax2.set_xlabel(r"Coherence time $t_{\mathrm{coh}}$ (time steps)",
                   fontsize=13)
    ax2.set_ylabel("Adaptive vs\nFixed (%)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/figures/bl_fig3_skr_vs_tcoh.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/bl_fig3_skr_vs_tcoh.png",
                dpi=200, bbox_inches="tight")
    print("    Saved bl_fig3_skr_vs_tcoh.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  BL Figure 4: SKR vs p_gen (their Fig. 6 equivalent)
# ═══════════════════════════════════════════════════════════

def bl_fig4_skr_vs_pgen():
    """
    SKR as a function of generation probability.
    Uses Boxi Li's p_swap=0.5, w0=0.98, t_coh=400.
    """
    print("  [BL Fig 4] SKR vs p_gen...")

    p_swap = BOXILI_PARAMS["p_swap"]
    w0 = BOXILI_PARAMS["w0"]
    t_coh = BOXILI_PARAMS["t_coh"]

    p_gen_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    n_ep = N_EPISODES // 2

    skr_nocut = []
    skr_fixed = []
    skr_adapt = []
    best_ns = []

    for p_gen in p_gen_values:
        print(f"    p_gen={p_gen}...", end="", flush=True)

        # No cutoff
        mc = run_simulation(
            NoCutoffPolicy(), p_gen, w0, t_coh,p_swap=p_swap,
            n_episodes=n_ep, seed=SEED)
        skr_nocut.append(skr_boxili(
            mc.delivery_times.astype(float), mc.w_out_array))

        # Best fixed
        best_n, best_skr = 1, -np.inf
        for n_star in range(1, 100):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh,p_swap=p_swap,
                n_episodes=20_000, seed=SEED + 100)
            skr = skr_boxili(
                mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        mc = run_simulation(
            FixedCutoffPolicy(best_n), p_gen, w0, t_coh,p_swap=p_swap,
            n_episodes=n_ep, seed=SEED + 200)
        skr_fixed.append(skr_boxili(
            mc.delivery_times.astype(float), mc.w_out_array))
        best_ns.append(best_n)

        # Adaptive
        t_max = suggest_t_max(t_coh)
        t_max = min(max(t_max, best_n + 20), 150)
        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,p_swap=p_swap,
                t_max=t_max, verbose=False)
        mc = run_simulation(
            AdaptivePolicy(mdp["policy"], t_max), p_gen, w0, t_coh,p_swap=p_swap,
            n_episodes=n_ep, seed=SEED + 300)
        skr_adapt.append(skr_boxili(
            mc.delivery_times.astype(float), mc.w_out_array))

        print(f" n*={best_n}, done")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(p_gen_values, skr_nocut, "s--", color="gray", lw=2,
             markersize=7, label="No cutoff")
    ax1.plot(p_gen_values, skr_fixed, "o-", color="#4a90d9", lw=2,
             markersize=7, label="Optimal fixed cutoff")
    ax1.plot(p_gen_values, skr_adapt, "^-", color="#d94a4a", lw=2.5,
             markersize=8, label="Adaptive MDP")

    ax1.set_ylabel("Secret Key Rate", fontsize=13)
    ax1.set_title(
        "SKR vs Generation Probability (Boxi Li params)\n"
        f"$p_{{swap}}={p_swap}$, $w_0={w0}$, $t_{{coh}}={t_coh}$",
        fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(p_gen_values)
    ax1.set_xticklabels([f"{p:.2f}" for p in p_gen_values], fontsize=9)

    # Improvement bars
    imp = [100 * (a - f) / f if f > 0 else 0
           for a, f in zip(skr_adapt, skr_fixed)]
    bar_width = np.min(np.diff(p_gen_values)) * 0.5
    ax2.bar(p_gen_values, imp, width=bar_width, color="#d94a4a",
            alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(p_gen_values)
    ax2.set_xticklabels([f"{p:.2f}" for p in p_gen_values], fontsize=9)
    ax2.set_xlabel(r"Generation probability $p_{\mathrm{gen}}$",
                   fontsize=13)
    ax2.set_ylabel("Adaptive vs\nFixed (%)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/figures/bl_fig4_skr_vs_pgen.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/bl_fig4_skr_vs_pgen.png",
                dpi=200, bbox_inches="tight")
    print("    Saved bl_fig4_skr_vs_pgen.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  BL Figure 5: Optimal n* vs t_coh (their optimization)
# ═══════════════════════════════════════════════════════════

def bl_fig5_optimal_cutoff_vs_tcoh():
    """
    How optimal cutoff scales with coherence time.
    Reproduces the optimization result from Boxi Li.
    """
    print("  [BL Fig 5] Optimal cutoff vs t_coh...")

    p_gen = BOXILI_PARAMS["p_gen"]
    p_swap = BOXILI_PARAMS["p_swap"]
    w0 = BOXILI_PARAMS["w0"]

    t_coh_values = [10, 20, 50, 100, 200, 400, 600, 1000]

    best_ns = []
    best_skrs = []
    skr_nocuts = []

    for t_coh in t_coh_values:
        print(f"    t_coh={t_coh}...", end="", flush=True)

        # No cutoff SKR
        mc = run_simulation(
            NoCutoffPolicy(), p_gen, w0, t_coh,p_swap=p_swap,
            n_episodes=50_000, seed=SEED)
        skr_nocuts.append(skr_boxili(
            mc.delivery_times.astype(float), mc.w_out_array))

        # Sweep
        best_n, best_skr = 1, -np.inf
        sweep_max = min(int(5 * t_coh * p_gen) + 20, 200)
        sweep_max = max(sweep_max, 20)
        for n_star in range(1, sweep_max):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh,p_swap=p_swap,
                n_episodes=30_000, seed=SEED + 100)
            skr = skr_boxili(
                mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        best_ns.append(best_n)
        best_skrs.append(best_skr)
        print(f" n*={best_n}")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: n* vs t_coh
    ax1.plot(t_coh_values, best_ns, "o-", color="#4a90d9", lw=2,
             markersize=8)
    # Reference: n* ∝ t_coh
    t_ref = np.array(t_coh_values)
    scale = best_ns[3] / t_coh_values[3]  # match at t_coh=100
    ax1.plot(t_ref, scale * t_ref, "k:", lw=1.5, alpha=0.5,
             label=r"$n^* \propto t_{\mathrm{coh}}$ (reference)")

    ax1.set_xlabel(r"Coherence time $t_{\mathrm{coh}}$", fontsize=13)
    ax1.set_ylabel(r"Optimal cutoff $n^*$", fontsize=13)
    ax1.set_title("Optimal Cutoff vs Coherence Time", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: SKR improvement of cutoff over no-cutoff
    imp = [100 * (f - n) / n if n > 0 else 0
           for f, n in zip(best_skrs, skr_nocuts)]
    ax2.bar(range(len(t_coh_values)), imp, color="#4a90d9", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(t_coh_values)))
    ax2.set_xticklabels(t_coh_values, fontsize=10)
    ax2.set_xlabel(r"Coherence time $t_{\mathrm{coh}}$", fontsize=13)
    ax2.set_ylabel("SKR Improvement\nvs No Cutoff (%)", fontsize=11)
    ax2.set_title("Benefit of Cutoff Optimization", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Boxi Li params: $p_{{gen}}={p_gen}$, $p_{{swap}}={p_swap}$, "
        f"$w_0={w0}$",
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/bl_fig5_optimal_cutoff.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/bl_fig5_optimal_cutoff.png",
                dpi=200, bbox_inches="tight")
    print("    Saved bl_fig5_optimal_cutoff.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs("results/figures", exist_ok=True)

    print("═" * 70)
    print("  Reproducing Boxi Li et al. Results")
    print("  'Efficient Optimization of Cutoffs in Quantum Repeater Chains'")
    print("  Parameters: p_gen=0.1, p_swap=0.5, w0=0.98, t_coh=400")
    print("═" * 70)

    best_n = bl_fig1_skr_vs_cutoff()          # ← retourne seulement best_n
    bl_fig2_pmf_and_werner(best_n)
    bl_fig3_skr_vs_tcoh()
    bl_fig4_skr_vs_pgen()
    bl_fig5_optimal_cutoff_vs_tcoh()
    bl_table_numerical_comparison()            # ← nouveau
    bl_fig6_mdp_policy_heatmap()              # ← nouveau

    print("\n" + "═" * 70)
    print("  All figures saved to results/figures/")
    print("═" * 70)
    print("\n  bl_fig1 — SKR vs cutoff")
    print("  bl_fig2 — PMF + Werner (2x2)")
    print("  bl_fig3 — SKR vs t_coh")
    print("  bl_fig4 — SKR vs p_gen")
    print("  bl_fig5 — Optimal n* vs t_coh")
    print("  bl_fig6 — Heatmap politique MDP")    # ← nouveau
    print("  table   — Comparaison numérique")    # ← nouveau
    print("\n  ✅ Done!")

if __name__ == "__main__":
    main()