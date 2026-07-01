
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from src.simulation.mc_engine import (
    NoCutoffPolicy,
    FixedCutoffPolicy,
    AdaptivePolicy,
    run_simulation,
)
from src.metrics.skr import skr_from_samples
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import suggest_t_max
from src.mdp.actions import SWAP


def fig7_policy_heatmap():
    """
    Visualise the MDP decision at each state (age1, age2).
    WAIT vs SWAP when both links are ready (s1=1, s2=1).
    """
    regimes = [
        {"p_gen": 0.1, "w0": 1.0, "t_coh": 20.0,
         "title": "$p=0.1$, $t_{coh}=20$ (strong decoh)"},
        {"p_gen": 0.1, "w0": 1.0, "t_coh": 50.0,
         "title": "$p=0.1$, $t_{coh}=50$ (moderate decoh)"},
        {"p_gen": 0.3, "w0": 1.0, "t_coh": 20.0,
         "title": "$p=0.3$, $t_{coh}=20$ (fast gen, strong decoh)"},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        p_gen = regime["p_gen"]
        w0 = regime["w0"]
        t_coh = regime["t_coh"]

        t_max = suggest_t_max(t_coh = t_coh, w0=w0)
        t_max = min(max(t_max, 25), 50)

        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                        t_max=t_max, verbose=False)
        policy = mdp["policy"]

        # Build decision grid for both-links-ready states: (1, a1, 1, a2)
        grid_size = min(t_max, 40)
        decision = np.full((grid_size, grid_size), np.nan)

        # Find the SWAP action value
        # From output: 0 = WAIT, 2 = SWAP
        SWAP_ACTION = SWAP

        for a1 in range(1, grid_size):
            for a2 in range(1, grid_size):
                key = (1, a1, 1, a2)
                if key in policy:
                    # 1 if SWAP, 0 if WAIT
                    decision[a2, a1] = 1 if policy[key] == SWAP_ACTION else 0

        # Custom colormap: blue = WAIT, red = SWAP
        cmap = mcolors.ListedColormap(["#4a90d9", "#d94a4a"])
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(decision, origin="lower", cmap=cmap, norm=norm,
                  aspect="equal", extent=[0, grid_size, 0, grid_size])

        # Overlay optimal fixed cutoff boundary
        best_n, best_skr = 1, -np.inf
        for n_star in range(1, grid_size):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
                n_episodes=20_000, seed=42)
            skr = skr_from_samples(
                mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        # Fixed cutoff = square boundary at n*
        # Links are discarded when age > n*, so SWAP region is age <= n*
        ax.plot([best_n, best_n], [0, best_n], "k--", lw=2.5)
        ax.plot([0, best_n], [best_n, best_n], "k--", lw=2.5)
        ax.plot([best_n, grid_size], [best_n, best_n], "k--", lw=2.5,
                alpha=0.3)
        ax.plot([best_n, best_n], [best_n, grid_size], "k--", lw=2.5,
                alpha=0.3)

        ax.set_xlabel("Age of link A ($a_1$)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Age of link B ($a_2$)", fontsize=12)
        ax.set_title(regime["title"], fontsize=12)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)

        # Legend
        legend_elements = [
            Patch(facecolor="#4a90d9", label="WAIT"),
            Patch(facecolor="#d94a4a", label="SWAP"),
            plt.Line2D([0], [0], color="black", ls="--", lw=2.5,
                       label=f"Fixed $n^*={best_n}$"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.suptitle(
        "MDP Optimal Policy: WAIT vs SWAP Decision Boundary",
        fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/fig7_policy_heatmap.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig7_policy_heatmap.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig7_policy_heatmap.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  Figure 8: Delivery Time Distribution
# ═══════════════════════════════════════════════════════════

def fig8_delivery_time_dist():
    """
    Histogram of delivery times for the three strategies.
    """
    p_gen = 0.1
    w0 = 1.0
    t_coh = 30.0
    n_episodes = 200_000
    seed = 42

    # ── No cutoff ──
    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed)

    # ── Best fixed cutoff ──
    best_n, best_skr = 1, -np.inf
    for n_star in range(1, 50):
        mc = run_simulation(
            FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
            n_episodes=30_000, seed=seed + 100)
        skr = skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array)
        if skr > best_skr:
            best_skr = skr
            best_n = n_star

    mc_fixed = run_simulation(
        FixedCutoffPolicy(best_n), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed + 200)

    # ── Adaptive MDP ──
    t_max = suggest_t_max(t_coh=t_coh, w0=w0)
    t_max = min(max(t_max, best_n + 10), 60)
    mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                    t_max=t_max, verbose=False)
    mc_adaptive = run_simulation(
        AdaptivePolicy(mdp["policy"], t_max),
        p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed + 300)

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full histogram
    max_t = int(np.percentile(mc_nocut.delivery_times, 99))
    bins = np.arange(1, max_t + 1)

    ax1.hist(mc_nocut.delivery_times, bins=bins, alpha=0.4,
             density=True, color="gray",
             label=f"No cutoff (mean={np.mean(mc_nocut.delivery_times):.1f})")
    ax1.hist(mc_fixed.delivery_times, bins=bins, alpha=0.4,
             density=True, color="blue",
             label=f"Fixed n*={best_n} (mean={np.mean(mc_fixed.delivery_times):.1f})")
    ax1.hist(mc_adaptive.delivery_times, bins=bins, alpha=0.4,
             density=True, color="red",
             label=f"Adaptive (mean={np.mean(mc_adaptive.delivery_times):.1f})")

    ax1.set_xlabel("Delivery Time (time steps)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Delivery Time Distribution", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: CDF
    for data, color, label in [
        (mc_nocut.delivery_times, "gray", "No cutoff"),
        (mc_fixed.delivery_times, "blue", f"Fixed n*={best_n}"),
        (mc_adaptive.delivery_times, "red", "Adaptive"),
    ]:
        sorted_t = np.sort(data)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax2.plot(sorted_t, cdf, color=color, lw=2, label=label)

    ax2.set_xlabel("Delivery Time (time steps)", fontsize=12)
    ax2.set_ylabel("CDF", fontsize=12)
    ax2.set_title("Cumulative Distribution", fontsize=13)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_t)

    plt.suptitle(
        f"Delivery Time ($p_{{gen}}={p_gen}$, $t_{{coh}}={t_coh}$)",
        fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/fig8_delivery_time.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig8_delivery_time.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig8_delivery_time.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  Figure 9: SKR vs t_coh (line plot)
# ═══════════════════════════════════════════════════════════

def fig9_skr_vs_tcoh():
    """
    SKR vs t_coh for fixed p_gen. Three lines: no cutoff, fixed, adaptive.
    """
    p_gen = 0.1
    w0 = 1.0
    t_coh_values = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    n_episodes = 100_000
    seed = 42

    skr_nocut = []
    skr_fixed = []
    skr_adapt = []
    best_ns = []

    for i, t_coh in enumerate(t_coh_values):
        print(f"    t_coh={t_coh}...", end="", flush=True)

        # No cutoff
        mc = run_simulation(
            NoCutoffPolicy(), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed)
        skr_nocut.append(skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array))

        # Best fixed cutoff
        best_n, best_skr = 1, -np.inf
        sweep_max = min(int(4 * t_coh * p_gen) + 10, 80)
        sweep_max = max(sweep_max, 15)
        for n_star in range(1, sweep_max):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
                n_episodes=30_000, seed=seed + 100)
            skr = skr_from_samples(
                mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        mc = run_simulation(
            FixedCutoffPolicy(best_n), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed + 200)
        skr_fixed.append(skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array))
        best_ns.append(best_n)

        # Adaptive MDP
        t_max = suggest_t_max(t_coh=t_coh, w0=w0)
        t_max = min(max(t_max, best_n + 10), 80)
        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                        t_max=t_max, verbose=False)
        mc = run_simulation(
            AdaptivePolicy(mdp["policy"], t_max),
            p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed + 300)
        skr_adapt.append(skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array))

        print(f" done (n*={best_n})")

     # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: SKR lines (linear x-axis)
    ax1.plot(t_coh_values, skr_nocut, "s--", color="gray", lw=2,
             markersize=6, label="No cutoff")
    ax1.plot(t_coh_values, skr_fixed, "o-", color="#4a90d9", lw=2,
             markersize=6, label="Optimal fixed cutoff")
    ax1.plot(t_coh_values, skr_adapt, "^-", color="#d94a4a", lw=2,
             markersize=7, label="Adaptive MDP")

    ax1.set_ylabel("Secret Key Rate", fontsize=13)
    ax1.set_title(
        f"SKR vs Coherence Time ($p_{{gen}}={p_gen}$, $w_0={w0}$)",
        fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(t_coh_values)
    ax1.set_xticklabels(t_coh_values, fontsize=9, rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/figures/fig9_skr_vs_tcoh.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig9_skr_vs_tcoh.png",
                dpi=200, bbox_inches="tight")
    print("  Saved fig9_skr_vs_tcoh.pdf/png")
    plt.close()

# ═══════════════════════════════════════════════════════════
#  Figure 10: SKR vs p_gen (line plot)
# ═══════════════════════════════════════════════════════════

def fig10_skr_vs_pgen():
    """
    SKR vs p_gen for fixed t_coh. Three lines.
    """
    t_coh = 50.0
    w0 = 1.0
    p_gen_values = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    n_episodes = 100_000
    seed = 42

    skr_nocut = []
    skr_fixed = []
    skr_adapt = []
    best_ns = []

    for i, p_gen in enumerate(p_gen_values):
        print(f"    p_gen={p_gen}...", end="", flush=True)

        # No cutoff
        mc = run_simulation(
            NoCutoffPolicy(), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed)
        skr_nocut.append(skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array))

        # Best fixed cutoff
        best_n, best_skr = 1, -np.inf
        for n_star in range(1, 60):
            mc = run_simulation(
                FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
                n_episodes=30_000, seed=seed + 100)
            skr = skr_from_samples(
                mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n_star

        mc = run_simulation(
            FixedCutoffPolicy(best_n), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed + 200)
        skr_fixed.append(skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array))
        best_ns.append(best_n)

        # Adaptive MDP
        t_max = suggest_t_max(t_coh=t_coh, w0=w0)   
        t_max = min(max(t_max, best_n + 10), 80)
        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                        t_max=t_max, verbose=False)
        mc = run_simulation(
            AdaptivePolicy(mdp["policy"], t_max),
            p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed + 300)
        skr_adapt.append(skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array))

        print(f" done (n*={best_n})")

        # ── Plot ──
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(p_gen_values, skr_nocut, "s--", color="gray", lw=2,
             markersize=6, label="No cutoff")
    ax1.plot(p_gen_values, skr_fixed, "o-", color="#4a90d9", lw=2,
             markersize=6, label="Optimal fixed cutoff")
    ax1.plot(p_gen_values, skr_adapt, "^-", color="#d94a4a", lw=2,
             markersize=7, label="Adaptive MDP")

    ax1.set_ylabel("Secret Key Rate", fontsize=13)
    ax1.set_title(
        f"SKR vs Generation Probability ($t_{{coh}}={t_coh}$, $w_0={w0}$)",
        fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(p_gen_values)
    ax1.set_xticklabels([f"{p:.2f}" for p in p_gen_values], fontsize=9)

    # Bottom: improvement
    imp = [100 * (a - f) / f if f > 0 else 0
           for a, f in zip(skr_adapt, skr_fixed)]
    bar_width = np.min(np.diff(p_gen_values)) * 0.6
    ax2.bar(p_gen_values, imp, width=bar_width, color="#d94a4a",
            alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(p_gen_values)
    ax2.set_xticklabels([f"{p:.2f}" for p in p_gen_values], fontsize=9)
    ax2.set_xlabel(r"Generation probability $p_{\mathrm{gen}}$", fontsize=13)
    ax2.set_ylabel("Adaptive vs\nFixed (%)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlim(ax1.get_xlim())

    # Bottom: improvement
    imp = [100 * (a - f) / f if f > 0 else 0
           for a, f in zip(skr_adapt, skr_fixed)]
    ax2.bar(range(len(p_gen_values)), imp, color="#d94a4a", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(p_gen_values)))
    ax2.set_xticklabels([f"{p:.2f}" for p in p_gen_values], fontsize=10)
    ax2.set_xlabel(r"Generation probability $p_{\mathrm{gen}}$", fontsize=13)
    ax2.set_ylabel("Adaptive vs\nFixed (%)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/figures/fig10_skr_vs_pgen.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig10_skr_vs_pgen.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig10_skr_vs_pgen.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  Figure 11: Optimal n* vs t_coh for different p_gen
# ═══════════════════════════════════════════════════════════

def fig11_optimal_cutoff():
    """
    How the optimal fixed cutoff n* scales with t_coh.
    Multiple lines for different p_gen.
    """
    t_coh_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    p_gen_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    w0 = 1.0
    seed = 42

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(p_gen_values)))

    for p_idx, p_gen in enumerate(p_gen_values):
        best_ns = []
        print(f"    p_gen={p_gen}...", end="", flush=True)

        for t_coh in t_coh_values:
            best_n, best_skr = 1, -np.inf
            sweep_max = min(int(5 * t_coh * p_gen) + 15, 100)
            sweep_max = max(sweep_max, 15)

            for n_star in range(1, sweep_max):
                mc = run_simulation(
                    FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
                    n_episodes=30_000, seed=seed)
                skr = skr_from_samples(
                    mc.delivery_times.astype(float), mc.w_out_array)
                if skr > best_skr:
                    best_skr = skr
                    best_n = n_star

            best_ns.append(best_n)

        ax.plot(t_coh_values, best_ns, "o-", color=colors[p_idx],
                lw=2, markersize=6, label=f"$p_{{gen}}={p_gen}$")
        print(f" done")

    # Reference line: n* ∝ t_coh
    t_ref = np.array(t_coh_values)
    ax.plot(t_ref, 0.15 * t_ref, "k:", lw=1.5, alpha=0.5,
            label=r"$n^* \propto t_{\mathrm{coh}}$ (reference)")

    ax.set_xlabel(r"Coherence time $t_{\mathrm{coh}}$", fontsize=13)
    ax.set_ylabel(r"Optimal fixed cutoff $n^*$", fontsize=13)
    ax.set_title("Optimal Fixed Cutoff vs Coherence Time", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("results/figures/fig11_optimal_cutoff.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig11_optimal_cutoff.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig11_optimal_cutoff.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Generating advanced thesis figures")
    print("═" * 60)

    print("\n[Fig 7] MDP Policy Heatmap...")
    fig7_policy_heatmap()

    print("\n[Fig 8] Delivery Time Distribution...")
    fig8_delivery_time_dist()

    print("\n[Fig 9] SKR vs t_coh...")
    fig9_skr_vs_tcoh()

    print("\n[Fig 10] SKR vs p_gen...")
    fig10_skr_vs_pgen()

    print("\n[Fig 11] Optimal cutoff n* vs parameters...")
    fig11_optimal_cutoff()

    print("\n" + "═" * 60)
    print("  All advanced figures saved to results/figures/")
    print("═" * 60)
    print("\n  Complete figure list:")
    print("    fig7_policy_heatmap    — MDP WAIT/SWAP decision boundary")
    print("    fig8_delivery_time     — Delivery time distribution + CDF")
    print("    fig9_skr_vs_tcoh       — SKR vs coherence time")
    print("    fig10_skr_vs_pgen      — SKR vs generation probability")
    print("    fig11_optimal_cutoff   — Optimal n* scaling")
    print("\n  ✅ Done!")


if __name__ == "__main__":
    main()