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
from src.mdp.actions import SWAP as SWAP_ACTION




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

        t_max = suggest_t_max(w0, t_coh)
        t_max = min(max(t_max, 25), 50)

        mdp = solve_mdp(p_gen=p_gen, w0=w0, t_coh=t_coh,
                        t_max=t_max, verbose=False)
        policy = mdp["policy"]

        # Build decision grid for both-links-ready states: (1, a1, 1, a2)
        grid_size = min(t_max, 40)
        decision = np.full((grid_size, grid_size), np.nan)

        # Find the SWAP action value
        # From output: 0 = WAIT, 2 = SWAP
        SWAP_ACTION = SWAP_ACTION

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



def main():
    print("═" * 60)
    print("  Generating advanced thesis figures")
    print("═" * 60)

    print("\n[Fig 7] MDP Policy Heatmap...")
    fig7_policy_heatmap()
