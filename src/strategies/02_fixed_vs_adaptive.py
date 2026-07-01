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
from src.metrics.skr import skr_from_samples, skr_with_ci
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import suggest_t_max


def separator(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def find_optimal_fixed_cutoff(p_gen, w0, t_coh, n_episodes=100_000,
                              max_cutoff=60, seed=42):
    """
    Brute-force search for the optimal fixed cutoff n*.

    Returns (best_cutoff, best_skr, all_results).
    """
    best_cutoff = 1
    best_skr = -np.inf
    all_results = []

    for n_star in range(1, max_cutoff + 1):
        mc = run_simulation(
            FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=seed)
        skr = skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array)
        all_results.append((n_star, skr))
        if skr > best_skr:
            best_skr = skr
            best_cutoff = n_star

    return best_cutoff, best_skr, all_results


def run_experiment(p_gen, w0, t_coh, label,
                   n_episodes=100_000, max_cutoff=60, seed=42):
    """
    Run the full comparison for one parameter set.

    Returns dict with results.
    """
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"  p_gen={p_gen}, w0={w0}, t_coh={t_coh}")
    print(f"{'─' * 70}")

    results = {"label": label, "p_gen": p_gen, "w0": w0, "t_coh": t_coh}

    # ── 1. No cutoff ──────────────────────────────────────────────
    print(f"  [1/3] No cutoff...")
    t0 = time.time()
    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed)
    ci_nocut = skr_with_ci(
        mc_nocut.delivery_times.astype(float), mc_nocut.w_out_array)
    results["skr_nocut"] = ci_nocut["skr"]
    results["skr_nocut_ci"] = ci_nocut["skr_hw"]
    results["ET_nocut"] = np.mean(mc_nocut.delivery_times)
    results["EW_nocut"] = np.mean(mc_nocut.w_out_array)
    print(f"        SKR = {ci_nocut['skr']:.6e} ± {ci_nocut['skr_hw']:.1e}"
          f"  ({time.time()-t0:.1f}s)")

    # ── 2. Optimal fixed cutoff ───────────────────────────────────
    print(f"  [2/3] Fixed cutoff sweep (1..{max_cutoff})...")
    t0 = time.time()
    best_n, best_skr_fixed, cutoff_curve = find_optimal_fixed_cutoff(
        p_gen, w0, t_coh, n_episodes=n_episodes,
        max_cutoff=max_cutoff, seed=seed + 100)

    # Re-run best cutoff with CI
    mc_fixed = run_simulation(
        FixedCutoffPolicy(best_n), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed + 200)
    ci_fixed = skr_with_ci(
        mc_fixed.delivery_times.astype(float), mc_fixed.w_out_array)

    results["best_cutoff"] = best_n
    results["skr_fixed"] = ci_fixed["skr"]
    results["skr_fixed_ci"] = ci_fixed["skr_hw"]
    results["ET_fixed"] = np.mean(mc_fixed.delivery_times)
    results["EW_fixed"] = np.mean(mc_fixed.w_out_array)
    results["cutoff_curve"] = cutoff_curve
    print(f"        n* = {best_n}, SKR = {ci_fixed['skr']:.6e} "
          f"± {ci_fixed['skr_hw']:.1e}  ({time.time()-t0:.1f}s)")

    # ── 3. Adaptive MDP ───────────────────────────────────────────
    print(f"  [3/3] Adaptive MDP...")
    t0 = time.time()

    t_max = suggest_t_max(t_coh)
    t_max = max(t_max, best_n + 5)  # at least cover the best cutoff
    t_max = min(t_max, 80)          # cap for tractability

    mdp = solve_mdp(
        p_gen=p_gen, w0=w0, t_coh=t_coh,
        t_max=t_max, verbose=False)

    results["mdp_converged"] = mdp["converged"]
    results["mdp_gain"] = mdp["gain"]
    results["mdp_t_max"] = t_max
    results["mdp_n_iter"] = mdp["n_iter"]
    results["mdp_elapsed"] = mdp["elapsed"]

    mc_adaptive = run_simulation(
        AdaptivePolicy(mdp["policy"], t_max),
        p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=seed + 300)
    ci_adaptive = skr_with_ci(
        mc_adaptive.delivery_times.astype(float),
        mc_adaptive.w_out_array)

    results["skr_adaptive"] = ci_adaptive["skr"]
    results["skr_adaptive_ci"] = ci_adaptive["skr_hw"]
    results["ET_adaptive"] = np.mean(mc_adaptive.delivery_times)
    results["EW_adaptive"] = np.mean(mc_adaptive.w_out_array)
    print(f"        MDP gain = {mdp['gain']:.6e}, "
          f"MC SKR = {ci_adaptive['skr']:.6e} ± {ci_adaptive['skr_hw']:.1e}"
          f"  ({time.time()-t0:.1f}s)")

    # ── Improvements ──────────────────────────────────────────────
    if results["skr_fixed"] > 0:
        imp_over_fixed = ((results["skr_adaptive"] - results["skr_fixed"])
                          / results["skr_fixed"] * 100)
    else:
        imp_over_fixed = float('inf') if results["skr_adaptive"] > 0 else 0.0

    if results["skr_nocut"] > 0:
        imp_fixed_over_nocut = ((results["skr_fixed"] - results["skr_nocut"])
                                / results["skr_nocut"] * 100)
        imp_adaptive_over_nocut = ((results["skr_adaptive"] - results["skr_nocut"])
                                   / results["skr_nocut"] * 100)
    else:
        imp_fixed_over_nocut = float('inf')
        imp_adaptive_over_nocut = float('inf')

    results["imp_adaptive_vs_fixed"] = imp_over_fixed
    results["imp_fixed_vs_nocut"] = imp_fixed_over_nocut
    results["imp_adaptive_vs_nocut"] = imp_adaptive_over_nocut

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  ��  No cutoff:      SKR = {results['skr_nocut']:>12.6e}        │")
    print(f"  │  Fixed (n*={best_n:>2d}):  SKR = {results['skr_fixed']:>12.6e}  "
          f"({imp_fixed_over_nocut:>+6.1f}%) │")
    print(f"  │  Adaptive MDP:   SKR = {results['skr_adaptive']:>12.6e}  "
          f"({imp_adaptive_over_nocut:>+6.1f}%) │")
    print(f"  │                                                 │")
    print(f"  │  Adaptive vs Fixed: {imp_over_fixed:>+6.2f}%"
          f"                       │")
    print(f"  └─────────────────────────────────────────────────┘")

    return results


def main():
    t_global = time.time()
    n_episodes = 100_000

    separator("Experiment 02: Fixed Cutoff vs Adaptive MDP")
    print(f"  Episodes per strategy: {n_episodes:,}")

    # ══════════════════════════════════════════════════════════════
    # Parameter sets — chosen to span different regimes
    # ══════════════════════════════���═══════════════════════════════

    experiments = [
        # Regime 1: Fast generation, short coherence (adaptive should help)
        {"p_gen": 0.3,  "w0": 1.0,  "t_coh": 20.0,
         "label": "A: Fast gen, short memory"},

        # Regime 2: Moderate generation, moderate coherence
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 50.0,
         "label": "B: Moderate gen, moderate memory"},

        # Regime 3: Slow generation, long coherence (cutoff less important)
        {"p_gen": 0.05, "w0": 1.0,  "t_coh": 200.0,
         "label": "C: Slow gen, long memory"},

        # Regime 4: Imperfect source
        {"p_gen": 0.1,  "w0": 0.95, "t_coh": 100.0,
         "label": "D: Imperfect source (w0=0.95)"},

        # Regime 5: Very short coherence (adaptive most valuable)
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 20.0,
         "label": "E: Short memory stress test"},

        # Regime 6: Near-ideal (adaptive ≈ fixed)
        {"p_gen": 0.3,  "w0": 1.0,  "t_coh": 500.0,
         "label": "F: Near-ideal reference"},
    ]

    all_results = []
    for exp in experiments:
        result = run_experiment(
            exp["p_gen"], exp["w0"], exp["t_coh"], exp["label"],
            n_episodes=n_episodes)
        all_results.append(result)

    # ══════════════════════════════════════════════════════════════
    separator("SUMMARY TABLE")
    # ══════════════════════════════════════════════════════════════

    header = (f"  {'Label':<35s} {'p':>5s} {'w0':>5s} {'t_coh':>6s} "
              f"{'n*':>3s} {'SKR_nocut':>11s} {'SKR_fixed':>11s} "
              f"{'SKR_adapt':>11s} {'Δ(A/F)':>8s}")
    print(header)
    print(f"  {'─' * len(header)}")

    for r in all_results:
        print(f"  {r['label']:<35s} "
              f"{r['p_gen']:>5.2f} {r['w0']:>5.2f} {r['t_coh']:>6.0f} "
              f"{r['best_cutoff']:>3d} "
              f"{r['skr_nocut']:>11.4e} "
              f"{r['skr_fixed']:>11.4e} "
              f"{r['skr_adaptive']:>11.4e} "
              f"{r['imp_adaptive_vs_fixed']:>+7.2f}%")

    # ══════════════════════════════════════════════════════════════
    separator("KEY FINDINGS")
    # ══════════════════════════════════════════════════════════════

    # Find where adaptive helps most
    best_imp = max(all_results, key=lambda r: r["imp_adaptive_vs_fixed"])
    worst_imp = min(all_results, key=lambda r: r["imp_adaptive_vs_fixed"])

    print(f"\n  Largest improvement of adaptive over fixed:")
    print(f"    {best_imp['label']}: {best_imp['imp_adaptive_vs_fixed']:+.2f}%")
    print(f"\n  Smallest improvement (or degradation):")
    print(f"    {worst_imp['label']}: {worst_imp['imp_adaptive_vs_fixed']:+.2f}%")

    avg_imp = np.mean([r["imp_adaptive_vs_fixed"] for r in all_results])
    print(f"\n  Average improvement: {avg_imp:+.2f}%")

    # ══════════════════════════════════════════════════════════════
    # Save to CSV
    # ══════════════════════════════════════════════════════════════

    os.makedirs("results", exist_ok=True)
    csv_path = "results/exp02_fixed_vs_adaptive.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "p_gen", "w0", "t_coh",
            "best_cutoff", "mdp_t_max",
            "skr_nocut", "skr_nocut_ci",
            "skr_fixed", "skr_fixed_ci",
            "skr_adaptive", "skr_adaptive_ci",
            "imp_fixed_vs_nocut_pct",
            "imp_adaptive_vs_nocut_pct",
            "imp_adaptive_vs_fixed_pct",
            "ET_nocut", "ET_fixed", "ET_adaptive",
            "EW_nocut", "EW_fixed", "EW_adaptive",
            "mdp_gain", "mdp_converged", "mdp_n_iter",
        ])
        for r in all_results:
            writer.writerow([
                r["label"], r["p_gen"], r["w0"], r["t_coh"],
                r["best_cutoff"], r["mdp_t_max"],
                r["skr_nocut"], r["skr_nocut_ci"],
                r["skr_fixed"], r["skr_fixed_ci"],
                r["skr_adaptive"], r["skr_adaptive_ci"],
                r["imp_fixed_vs_nocut"],
                r["imp_adaptive_vs_nocut"],
                r["imp_adaptive_vs_fixed"],
                r["ET_nocut"], r["ET_fixed"], r["ET_adaptive"],
                r["EW_nocut"], r["EW_fixed"], r["EW_adaptive"],
                r["mdp_gain"], r["mdp_converged"], r["mdp_n_iter"],
            ])

    print(f"\n  Results saved to {csv_path}")

    # Save cutoff curves for plotting
    csv_curves_path = "results/exp02_cutoff_curves.csv"
    with open(csv_curves_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "cutoff", "skr"])
        for r in all_results:
            for n_star, skr_val in r["cutoff_curve"]:
                writer.writerow([r["label"], n_star, skr_val])

    print(f"  Cutoff curves saved to {csv_curves_path}")

    elapsed = time.time() - t_global
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_results


if __name__ == "__main__":
    results = main()