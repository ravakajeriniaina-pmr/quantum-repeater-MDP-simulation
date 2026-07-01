import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from src.simulation.mc_engine import (
    NoCutoffPolicy,
    FixedCutoffPolicy,
    AdaptivePolicy,
    run_simulation,
)
from src.metrics.skr import skr_from_samples
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import suggest_t_max


def werner_to_fidelity(w):
    """F = (3w + 1) / 4"""
    return (3 * w + 1) / 4


def fig4_fidelity_vs_age():
    """
    Figure 4: Fidelity vs age for different t_coh values.
    Shows how quickly entanglement degrades.
    """
    ages = np.arange(0, 80)
    t_cohs = [10, 20, 50, 100, 200, 500]
    w0 = 1.0

    fig, ax = plt.subplots(figsize=(8, 5))

    for t_coh in t_cohs:
        w = w0 * np.exp(-ages / t_coh)
        F = werner_to_fidelity(w)
        ax.plot(ages, F, linewidth=2, label=f"$t_{{coh}}={t_coh}$")

    # Threshold: F = 0.8107 (w = 0.7476) → sf = 0
    # From secret fraction: sf(w) > 0 when w > w_threshold
    ax.axhline(0.8107, color="red", ls="--", lw=1.5,
           label="Post-swap SKR threshold ($F_{out} \\approx 0.81$)\n"
                 "(applies to delivered pair, not individual link)")
    ax.axhline(0.5, color="black", ls=":", lw=1,
               label="Classical limit ($F = 0.5$)")

    ax.set_xlabel("Age (time steps)", fontsize=13)
    ax.set_ylabel("Fidelity $F$", fontsize=13)
    ax.set_title("Entanglement Fidelity vs Memory Age", fontsize=14)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/fig4_fidelity_vs_age.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig4_fidelity_vs_age.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig4_fidelity_vs_age.pdf/png")
    plt.close()


def fig5_fidelity_distributions():
    """
    Figure 5: Distribution of delivered fidelity for each strategy.
    Shows that adaptive concentrates mass at higher fidelities.
    """
    n_episodes = 100_000
    seed = 42

    # Two interesting regimes
    regimes = [
        {"p_gen": 0.1, "w0": 1.0, "t_coh": 20.0,
         "label": "Strong decoherence\n($p=0.1$, $t_{coh}=20$)"},
        {"p_gen": 0.1, "w0": 1.0, "t_coh": 50.0,
         "label": "Moderate decoherence\n($p=0.1$, $t_{coh}=50$)"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        p_gen = regime["p_gen"]
        w0 = regime["w0"]
        t_coh = regime["t_coh"]

        # ── No cutoff ────────────────────────────────
        mc_nocut = run_simulation(
            NoCutoffPolicy(), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed)
        F_nocut = werner_to_fidelity(mc_nocut.w_out_array)

        # ── Optimal fixed cutoff (find it) ───────────
        best_n, best_skr = 1, -np.inf
        for n_star in range(1, 40):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
                n_episodes=n_episodes // 2, seed=seed + 100)
            skr = skr_from_samples(
                mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        mc_fixed = run_simulation(
            FixedCutoffPolicy(best_n), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed + 200)
        F_fixed = werner_to_fidelity(mc_fixed.w_out_array)

        # ── Adaptive MDP ─────────────────────────────
        t_max = suggest_t_max(t_coh)
        t_max = max(t_max, best_n + 5)
        t_max = min(t_max, 60)

        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                        t_max=t_max, verbose=False)
        mc_adaptive = run_simulation(
            AdaptivePolicy(mdp["policy"], t_max),
            p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed + 300)
        F_adaptive = werner_to_fidelity(mc_adaptive.w_out_array)

        # ── Plot histograms ──────────────────────────
        bins = np.linspace(0.5, 1.0, 60)

        ax.hist(F_nocut, bins=bins, alpha=0.4, density=True,
                color="gray", label=f"No cutoff (mean={np.mean(F_nocut):.3f})")
        ax.hist(F_fixed, bins=bins, alpha=0.4, density=True,
                color="blue",
                label=f"Fixed n*={best_n} (mean={np.mean(F_fixed):.3f})")
        ax.hist(F_adaptive, bins=bins, alpha=0.4, density=True,
                color="red",
                label=f"Adaptive (mean={np.mean(F_adaptive):.3f})")

        # SKR threshold
        ax.axvline(0.8107, color="red", ls="--", lw=1.5, alpha=0.7,
                   label="SKR threshold")

        ax.set_xlabel("Delivered Fidelity $F$", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=12)
        ax.set_title(regime["label"], fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Distribution of Delivered Fidelity by Strategy",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/fig5_fidelity_dist.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig5_fidelity_dist.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig5_fidelity_dist.pdf/png")
    plt.close()


def fig6_fidelity_skr_tradeoff():
    """
    Figure 6: Mean fidelity vs SKR for different cutoffs.
    Shows the Pareto frontier: adaptive operates at a better tradeoff.
    """
    p_gen = 0.1
    w0 = 1.0
    t_coh = 30.0
    n_episodes = 80_000
    seed = 42

    cutoffs = list(range(1, 35))
    mean_F_list = []
    skr_list = []

    # ── Fixed cutoff sweep ────────────────────────────
    for n_star in cutoffs:
        mc = run_simulation(
            FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed)
        F = werner_to_fidelity(mc.w_out_array)
        skr = skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array)
        mean_F_list.append(np.mean(F))
        skr_list.append(skr)

    # ── No cutoff ─────────────────────────────────────
    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed)
    F_nocut = np.mean(werner_to_fidelity(mc_nocut.w_out_array))
    skr_nocut = skr_from_samples(
        mc_nocut.delivery_times.astype(float), mc_nocut.w_out_array)

    # ── Adaptive MDP ──────────────────────────────────
    t_max = suggest_t_max(t_coh)
    t_max = min(max(t_max, 35), 60)
    mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                    t_max=t_max, verbose=False)
    mc_adaptive = run_simulation(
        AdaptivePolicy(mdp["policy"], t_max),
        p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed + 300)
    F_adaptive = np.mean(werner_to_fidelity(mc_adaptive.w_out_array))
    skr_adaptive = skr_from_samples(
        mc_adaptive.delivery_times.astype(float),
        mc_adaptive.w_out_array)

    # ── Plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Fixed cutoff curve (each point = different n*)
    ax.plot(mean_F_list, skr_list, "b.-", alpha=0.5, markersize=5,
            label="Fixed cutoff (varying $n^*$)")

    # Annotate a few cutoff points
    for n_star in [2, 5, 10, 15, 25]:
        if n_star <= len(cutoffs):
            i = n_star - 1
            ax.annotate(f"$n^*={n_star}$",
                        (mean_F_list[i], skr_list[i]),
                        textcoords="offset points",
                        xytext=(8, 5), fontsize=8, color="blue")

    # No cutoff
    ax.plot(F_nocut, skr_nocut, "ks", markersize=10,
            label=f"No cutoff")

    # Adaptive
    ax.plot(F_adaptive, skr_adaptive, "r*", markersize=15,
            label=f"Adaptive MDP")

    ax.set_xlabel("Mean Delivered Fidelity $\\langle F \\rangle$",
                  fontsize=13)
    ax.set_ylabel("Secret Key Rate (per time step)", fontsize=13)
    ax.set_title("Fidelity–SKR Tradeoff ($p_{gen}=0.1$, $t_{coh}=30$)",
                 fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/fig6_fidelity_skr_tradeoff.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig6_fidelity_skr_tradeoff.png",
                dpi=200, bbox_inches="tight")
    print("  Saved fig6_fidelity_skr_tradeoff.pdf/png")
    plt.close()


def main():
    print("═" * 50)
    print("  Generating fidelity figures")
    print("═" * 50)

    fig4_fidelity_vs_age()
    fig5_fidelity_distributions()
    fig6_fidelity_skr_tradeoff()

    print("\n  All fidelity figures saved to results/figures/")
    print("  ✅ Done!")


if __name__ == "__main__":
    main()