"""
Plot Results for Thesis
========================
Generates publication-quality figures from experiment CSVs.

Figures:
    1. Improvement heatmap (exp03)
    2. SKR vs cutoff curves (exp02)
    3. Three-strategy bar chart (exp02)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_exp02():
    """Load exp02 summary CSV."""
    rows = []
    with open("results/exp02_fixed_vs_adaptive.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                if k != "label" and k != "mdp_converged":
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def load_exp02_curves():
    """Load exp02 cutoff curves CSV."""
    curves = {}
    with open("results/exp02_cutoff_curves.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in curves:
                curves[label] = {"cutoff": [], "skr": []}
            curves[label]["cutoff"].append(int(row["cutoff"]))
            curves[label]["skr"].append(float(row["skr"]))
    return curves


def load_exp03_grid():
    """Load exp03 improvement grid CSV."""
    with open("results/exp03_improvement_grid.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        t_coh_values = [int(x) for x in header[1:]]
        p_gen_values = []
        grid = []
        for row in reader:
            p_gen_values.append(float(row[0]))
            grid.append([float(x) for x in row[1:]])
    return np.array(p_gen_values), np.array(t_coh_values), np.array(grid)


def fig1_heatmap():
    """Figure 1: Improvement heatmap from exp03."""
    p_gen, t_coh, grid = load_exp03_grid()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Diverging colormap: red = adaptive worse, blue = adaptive better
    vmax = max(abs(grid.min()), abs(grid.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(grid, cmap="RdYlBu", norm=norm,
                   aspect="auto", origin="lower")

    # Labels
    ax.set_xticks(range(len(t_coh)))
    ax.set_xticklabels(t_coh, fontsize=11)
    ax.set_yticks(range(len(p_gen)))
    ax.set_yticklabels([f"{p:.2f}" for p in p_gen], fontsize=11)
    ax.set_xlabel(r"Coherence time $t_{\mathrm{coh}}$ (time steps)",
                  fontsize=13)
    ax.set_ylabel(r"Generation probability $p_{\mathrm{gen}}$",
                  fontsize=13)
    ax.set_title("SKR Improvement: Adaptive MDP vs Optimal Fixed Cutoff (%)",
                 fontsize=13, pad=12)

    # Annotate cells
    for i in range(len(p_gen)):
        for j in range(len(t_coh)):
            val = grid[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Improvement (%)", fontsize=11)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/fig1_heatmap.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig1_heatmap.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig1_heatmap.pdf/png")
    plt.close()


def fig2_cutoff_curves():
    """Figure 2: SKR vs cutoff for selected regimes."""
    curves = load_exp02_curves()
    exp02 = load_exp02()

    # Pick 3 interesting cases
    selected = [
        "A: Fast gen, short memory",
        "B: Moderate gen, moderate memory",
        "E: Short memory stress test",
    ]

    fig, axes = plt.subplots(1, len(selected), figsize=(15, 4.5),
                             sharey=False)

    for idx, label in enumerate(selected):
        ax = axes[idx]

        if label not in curves:
            continue

        cutoffs = curves[label]["cutoff"]
        skrs = curves[label]["skr"]

        # Find matching exp02 row
        row = [r for r in exp02 if r["label"] == label][0]

        # Plot cutoff curve
        ax.plot(cutoffs, skrs, "b.-", alpha=0.7, markersize=3,
                label="Fixed cutoff")

        # No-cutoff baseline
        ax.axhline(row["skr_nocut"], color="gray", ls="--", lw=1.5,
                   label="No cutoff")

        # Optimal fixed
        ax.axhline(row["skr_fixed"], color="blue", ls=":", lw=1.5,
                   label=f"Best fixed (n*={int(row['best_cutoff'])})")

        # Adaptive MDP
        ax.axhline(row["skr_adaptive"], color="red", ls="-", lw=2,
                   label="Adaptive MDP")

        ax.set_xlabel("Cutoff $n^*$", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Secret Key Rate", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("SKR vs Fixed Cutoff with Adaptive MDP Comparison",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/fig2_cutoff_curves.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig2_cutoff_curves.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig2_cutoff_curves.pdf/png")
    plt.close()


def fig3_bar_chart():
    """Figure 3: Three-strategy comparison bar chart."""
    exp02 = load_exp02()

    labels = [r["label"].split(":")[0] for r in exp02]
    skr_nocut = [r["skr_nocut"] for r in exp02]
    skr_fixed = [r["skr_fixed"] for r in exp02]
    skr_adaptive = [r["skr_adaptive"] for r in exp02]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width, skr_nocut, width, label="No cutoff",
                   color="#bbb", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, skr_fixed, width, label="Optimal fixed",
                   color="#4a90d9", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, skr_adaptive, width, label="Adaptive MDP",
                   color="#d94a4a", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Parameter Regime", fontsize=12)
    ax.set_ylabel("Secret Key Rate (per time step)", fontsize=12)
    ax.set_title("SKR Comparison: No Cutoff vs Fixed vs Adaptive MDP",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add improvement annotations
    for i, r in enumerate(exp02):
        imp = r["imp_adaptive_vs_fixed_pct"]
        y_pos = max(skr_nocut[i], skr_fixed[i], skr_adaptive[i])
        ax.text(i + width, y_pos * 1.02, f"{imp:+.1f}%",
                ha="center", va="bottom", fontsize=8, color="red",
                fontweight="bold")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/fig3_bar_chart.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig("results/figures/fig3_bar_chart.png", dpi=200,
                bbox_inches="tight")
    print("  Saved fig3_bar_chart.pdf/png")
    plt.close()


def main():
    print("═" * 50)
    print("  Generating thesis figures")
    print("═" * 50)

    fig1_heatmap()
    fig2_cutoff_curves()
    fig3_bar_chart()

    print("\n  All figures saved to results/figures/")
    print("  ✅ Done!")


if __name__ == "__main__":
    main()