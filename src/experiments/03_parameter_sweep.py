import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import time
import csv

from src.simulation.mc_engine import (
    NoCutoffPolicy,
    FixedCutoffPolicy,
    AdaptivePolicy,
    run_simulation,
)
from src.metrics.skr import skr_from_samples
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import suggest_t_max


def find_optimal_cutoff_fast(p_gen, w0, t_coh, n_episodes=50_000,
                             max_cutoff=50, seed=42):
    """Quick sweep to find optimal n*."""
    best_n = 1
    best_skr = -np.inf

    for n_star in range(1, max_cutoff + 1):
        mc = run_simulation(
            FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed)
        skr = skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array)
        if skr > best_skr:
            best_skr = skr
            best_n = n_star

    return best_n, best_skr


def run_single_point(p_gen, w0, t_coh, n_episodes=50_000,
                     max_cutoff=50, seed=42):
    """
    Compute all three SKRs for one (p_gen, t_coh) point.

    Returns dict with results.
    """
    result = {"p_gen": p_gen, "w0": w0, "t_coh": t_coh}

    # No cutoff
    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed)
    result["skr_nocut"] = skr_from_samples(
        mc_nocut.delivery_times.astype(float), mc_nocut.w_out_array)

    # Optimal fixed cutoff
    best_n, best_skr = find_optimal_cutoff_fast(
        p_gen, w0, t_coh, n_episodes=n_episodes,
        max_cutoff=max_cutoff, seed=seed + 100)
    result["best_cutoff"] = best_n
    result["skr_fixed"] = best_skr

    # Adaptive MDP
    t_max = suggest_t_max(t_coh)
    t_max = max(t_max, best_n + 5)
    t_max = min(t_max, max(60, best_n * 3))  # cap for speed

    mdp = solve_mdp(
        p_gen=p_gen, w0=w0, t_coh=t_coh,
        t_max=t_max, verbose=False)
    result["mdp_converged"] = mdp["converged"]
    result["mdp_gain"] = mdp["gain"]

    mc_adaptive = run_simulation(
        AdaptivePolicy(mdp["policy"], t_max),
        p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed + 200)
    result["skr_adaptive"] = skr_from_samples(
        mc_adaptive.delivery_times.astype(float),
        mc_adaptive.w_out_array)

    # Improvement
    if result["skr_fixed"] > 0:
        result["imp_pct"] = ((result["skr_adaptive"] - result["skr_fixed"])
                             / result["skr_fixed"] * 100)
    else:
        result["imp_pct"] = 0.0

    return result


def main():
    t_global = time.time()
    w0 = 1.0
    n_episodes = 50_000  # per strategy per point

    # ══════════════════════════════════════════════════════════════
    # Grid definition
    # ══════════════════════════════════════════════════════════════

    p_gen_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    t_coh_values = [10, 20, 50, 100, 200, 500]

    n_total = len(p_gen_values) * len(t_coh_values)

    print("═" * 70)
    print("  Experiment 03: Parameter Sweep")
    print("═" * 70)
    print(f"  w0 = {w0}")
    print(f"  p_gen:  {p_gen_values}")
    print(f"  t_coh:  {t_coh_values}")
    print(f"  Grid:   {len(p_gen_values)} × {len(t_coh_values)} "
          f"= {n_total} points")
    print(f"  Episodes per strategy: {n_episodes:,}")
    print(f"  Estimated time: ~{n_total * 30}s")

    # ══════════════════════════════════════════════════════════════
    # Run sweep
    # ══════════════════════════════════════════════════════════════

    all_results = []
    imp_grid = np.zeros((len(p_gen_values), len(t_coh_values)))
    skr_nocut_grid = np.zeros_like(imp_grid)
    skr_fixed_grid = np.zeros_like(imp_grid)
    skr_adapt_grid = np.zeros_like(imp_grid)
    cutoff_grid = np.zeros_like(imp_grid, dtype=int)

    for i, p_gen in enumerate(p_gen_values):
        for j, t_coh in enumerate(t_coh_values):
            idx = i * len(t_coh_values) + j + 1
            print(f"\n  [{idx:>3d}/{n_total}] p_gen={p_gen:.2f}, "
                  f"t_coh={t_coh}...", end="", flush=True)

            t0 = time.time()
            result = run_single_point(
                p_gen, w0, t_coh, n_episodes=n_episodes,
                seed=42 + idx)
            dt = time.time() - t0

            all_results.append(result)
            imp_grid[i, j] = result["imp_pct"]
            skr_nocut_grid[i, j] = result["skr_nocut"]
            skr_fixed_grid[i, j] = result["skr_fixed"]
            skr_adapt_grid[i, j] = result["skr_adaptive"]
            cutoff_grid[i, j] = result["best_cutoff"]

            conv = "✓" if result["mdp_converged"] else "✗"
            print(f"  n*={result['best_cutoff']:>2d}  "
                  f"Δ={result['imp_pct']:>+6.2f}%  "
                  f"[{conv}]  ({dt:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # Console heatmap: improvement (adaptive vs fixed)
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'═' * 70}")
    print(f"  Improvement Heatmap: Adaptive vs Optimal Fixed (%)")
    print(f"{'═' * 70}")

    # Header
    header = f"  {'p\\t_coh':>8s}"
    for t_coh in t_coh_values:
        header += f"  {t_coh:>7d}"
    print(header)
    print(f"  {'─' * (10 + 9 * len(t_coh_values))}")

    for i, p_gen in enumerate(p_gen_values):
        row = f"  {p_gen:>8.2f}"
        for j in range(len(t_coh_values)):
            val = imp_grid[i, j]
            row += f"  {val:>+6.2f}%"
        print(row)

    # ══════════════════════════════════════════════════════════════
    # Console heatmap: optimal fixed cutoff n*
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'═' * 70}")
    print(f"  Optimal Fixed Cutoff n*")
    print(f"{'═' * 70}")

    header = f"  {'p\\t_coh':>8s}"
    for t_coh in t_coh_values:
        header += f"  {t_coh:>7d}"
    print(header)
    print(f"  {'─' * (10 + 9 * len(t_coh_values))}")

    for i, p_gen in enumerate(p_gen_values):
        row = f"  {p_gen:>8.2f}"
        for j in range(len(t_coh_values)):
            row += f"  {cutoff_grid[i, j]:>7d}"
        print(row)

    # ══════════════════════════════════════════════════════════════
    # Console heatmap: adaptive SKR
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'═' * 70}")
    print(f"  Adaptive MDP SKR")
    print(f"{'═' * 70}")

    header = f"  {'p\\t_coh':>8s}"
    for t_coh in t_coh_values:
        header += f"  {t_coh:>7d}"
    print(header)
    print(f"  {'─' * (10 + 9 * len(t_coh_values))}")

    for i, p_gen in enumerate(p_gen_values):
        row = f"  {p_gen:>8.2f}"
        for j in range(len(t_coh_values)):
            row += f"  {skr_adapt_grid[i, j]:>7.1e}"
        print(row)

    # ══════════════════════════════════════════════════════════════
    # Key findings
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'═' * 70}")
    print(f"  KEY FINDINGS")
    print(f"{'═' * 70}")

    max_idx = np.unravel_index(np.argmax(imp_grid), imp_grid.shape)
    min_idx = np.unravel_index(np.argmin(imp_grid), imp_grid.shape)

    print(f"\n  Largest improvement:")
    print(f"    p_gen={p_gen_values[max_idx[0]]}, "
          f"t_coh={t_coh_values[max_idx[1]]}: "
          f"{imp_grid[max_idx]:+.2f}%")

    print(f"\n  Smallest improvement:")
    print(f"    p_gen={p_gen_values[min_idx[0]]}, "
          f"t_coh={t_coh_values[min_idx[1]]}: "
          f"{imp_grid[min_idx]:+.2f}%")

    print(f"\n  Average improvement: {np.mean(imp_grid):+.2f}%")
    print(f"  Median improvement:  {np.median(imp_grid):+.2f}%")

    n_positive = np.sum(imp_grid > 0)
    print(f"\n  Points where adaptive beats fixed: "
          f"{n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════
    # Save to CSV
    # ══════════════════════════════════════════════════════════════

    os.makedirs("results", exist_ok=True)

    # Full results
    csv_path = "results/exp03_parameter_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p_gen", "w0", "t_coh", "best_cutoff",
            "skr_nocut", "skr_fixed", "skr_adaptive",
            "imp_adaptive_vs_fixed_pct",
            "mdp_converged", "mdp_gain",
        ])
        for r in all_results:
            writer.writerow([
                r["p_gen"], r["w0"], r["t_coh"], r["best_cutoff"],
                r["skr_nocut"], r["skr_fixed"], r["skr_adaptive"],
                r["imp_pct"],
                r["mdp_converged"], r["mdp_gain"],
            ])
    print(f"\n  Full results saved to {csv_path}")

    # Improvement grid
    grid_path = "results/exp03_improvement_grid.csv"
    with open(grid_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p_gen\\t_coh"] + [str(t) for t in t_coh_values])
        for i, p_gen in enumerate(p_gen_values):
            writer.writerow([p_gen] + [f"{imp_grid[i,j]:.4f}"
                                       for j in range(len(t_coh_values))])
    print(f"  Improvement grid saved to {grid_path}")

    elapsed = time.time() - t_global
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_results, imp_grid


if __name__ == "__main__":
    results, grid = main()