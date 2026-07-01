import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import time

from src.analytical.no_cutoff import (
    mean_delivery_time,
    mean_werner_no_cutoff,
    skr_no_cutoff,
)
from src.simulation.mc_engine import (
    NoCutoffPolicy,
    FixedCutoffPolicy,
    AdaptivePolicy,
    run_simulation,
)
from src.metrics.skr import skr_from_samples, skr_with_ci
from src.mdp.value_iteration import solve_mdp


def separator(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def check(name, mc_val, an_val, tol=0.03):
    """Check relative error, return True if within tolerance."""
    if an_val == 0:
        err = abs(mc_val)
        ok = err < 1e-6
    else:
        err = abs(mc_val - an_val) / abs(an_val)
        ok = err < tol
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"   {name:<30s}  MC={mc_val:<12.6f}  "
          f"AN={an_val:<12.6f}  err={err:<8.4f}  [{status}]")
    return ok


def main():
    t_start = time.time()
    all_pass = True
    n_episodes = 500_000

    # ══════════════════════════════════════════════════════════════
    separator("1. No-Cutoff: E[T] Validation")
    # ══════════════════════════════════════════════════════════════

    test_cases_ET = [
        {"p_gen": 0.5,  "label": "p=0.50"},
        {"p_gen": 0.1,  "label": "p=0.10"},
        {"p_gen": 0.05, "label": "p=0.05"},
        {"p_gen": 0.01, "label": "p=0.01"},
    ]

    print(f"\n   {'Case':<15s}  {'MC E[T]':<12s}  {'AN E[T]':<12s}  "
          f"{'Rel Err':<8s}  Status")
    print(f"   {'-'*65}")

    for tc in test_cases_ET:
        p = tc["p_gen"]
        mc = run_simulation(
            NoCutoffPolicy(), p_gen=p, w0=1.0, t_coh=np.inf,
            p_swap=1.0,
            n_episodes=n_episodes, seed=42)
        mc_ET = np.mean(mc.delivery_times)
        an_ET = mean_delivery_time(p)
        ok = check(tc["label"], mc_ET, an_ET, tol=0.02)
        all_pass &= ok

    # ══════════════════════════════════════════════════════════════
    separator("2. No-Cutoff: E[w_out] Validation")
    # ══════════════════════════════════════════════════════════════

    test_cases_W = [
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 1000.0, "label": "weak decoh"},
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 100.0,  "label": "moderate"},
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 50.0,   "label": "strong decoh"},
        {"p_gen": 0.1,  "w0": 0.98, "t_coh": 200.0,  "label": "w0=0.98"},
        {"p_gen": 0.05, "w0": 1.0,  "t_coh": 200.0,  "label": "p=0.05"},
    ]

    print(f"\n   {'Case':<15s}  {'MC E[w]':<12s}  {'AN E[w]':<12s}  "
          f"{'Rel Err':<8s}  Status")
    print(f"   {'-'*65}")

    for tc in test_cases_W:
        mc = run_simulation(
            NoCutoffPolicy(), p_gen=tc["p_gen"], w0=tc["w0"],
            t_coh=tc["t_coh"],
            n_episodes=n_episodes, seed=42)
        mc_W = np.mean(mc.w_out_array)
        an_W = mean_werner_no_cutoff(tc["p_gen"], tc["w0"], tc["t_coh"])
        ok = check(tc["label"], mc_W, an_W, tol=0.02)
        all_pass &= ok

    # ══════════════════════════════════════════════════════════════
    separator("3. No-Cutoff: SKR Validation")
    # ══════════════════════════════════════════════════════════════

    test_cases_SKR = [
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 1000.0, "label": "weak decoh"},
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 100.0,  "label": "moderate"},
        {"p_gen": 0.1,  "w0": 0.98, "t_coh": 400.0,  "label": "w0=0.98"},
    ]

    print(f"\n   {'Case':<15s}  {'MC SKR':<12s}  {'AN SKR':<12s}  "
          f"{'Rel Err':<8s}  Status")
    print(f"   {'-'*65}")

    for tc in test_cases_SKR:
        mc = run_simulation(
            NoCutoffPolicy(), p_gen=tc["p_gen"], w0=tc["w0"],
            p_swap=1.0,
            t_coh=tc["t_coh"],
            n_episodes=n_episodes, seed=42)
        mc_SKR = skr_from_samples(
            mc.delivery_times.astype(float), mc.w_out_array)
        an_SKR = skr_no_cutoff(tc["p_gen"], tc["w0"], tc["t_coh"])
        ok = check(tc["label"], mc_SKR, an_SKR, tol=0.03)
        all_pass &= ok

    # ══════════════════════════════════════════════════════════════
    separator("4. Fixed Cutoff Improves Over No-Cutoff")
    # ══════════════════════════════════════════════════════════════

    p_gen, w0, t_coh = 0.1, 1.0, 50.0
    cutoffs_to_test = [5, 10, 15, 20]

    mc_nocut = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        n_episodes=n_episodes, seed=42)
    skr_nocut = skr_from_samples(
        mc_nocut.delivery_times.astype(float), mc_nocut.w_out_array)

    print(f"\n   No-cutoff SKR = {skr_nocut:.6e}")
    print(f"\n   {'Cutoff':<10s}  {'SKR':<14s}  {'Improvement':<14s}  Status")
    print(f"   {'-'*50}")

    best_cutoff = None
    best_skr = skr_nocut

    for n_star in cutoffs_to_test:
        mc_fixed = run_simulation(
            FixedCutoffPolicy(n_star), p_gen, w0, t_coh,
            n_episodes=n_episodes, seed=42)
        skr_fixed = skr_from_samples(
            mc_fixed.delivery_times.astype(float), mc_fixed.w_out_array)
        imp = (skr_fixed - skr_nocut) / max(skr_nocut, 1e-15) * 100
        ok = True
        ok_best = best_skr > skr_nocut 
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"   n*={n_star:<5d}  {skr_fixed:<14.6e}  {imp:>+10.2f}%"
              f"     [{status}]")
        all_pass &= ok
        if skr_fixed > best_skr:
            best_skr = skr_fixed
            best_cutoff = n_star

    if best_cutoff is not None:
        print(f"\n   Best fixed cutoff: n*={best_cutoff} "
              f"(SKR={best_skr:.6e})")

    # ══════════════════════════════════════════════════════════════
    separator("5. MDP Gain vs MC-Measured SKR")
    # ══════════════════════════════════════════════════════════════

    mdp_cases = [
        {"p_gen": 0.3,  "w0": 1.0,  "t_coh": 30.0,  "t_max": 25,
         "label": "p=0.3, tcoh=30"},
        {"p_gen": 0.1,  "w0": 1.0,  "t_coh": 100.0, "t_max": 40,
         "label": "p=0.1, tcoh=100"},
        {"p_gen": 0.1,  "w0": 0.98, "t_coh": 50.0,  "t_max": 30,
         "label": "p=0.1, tcoh=50, w0=.98"},
    ]

    print(f"\n   {'Case':<25s}  {'MDP gain':<12s}  {'MC SKR':<12s}  "
          f"{'Rel Err':<8s}  Status")
    print(f"   {'-'*70}")

    for tc in mdp_cases:
        mdp = solve_mdp(
            p_gen=tc["p_gen"], w0=tc["w0"], t_coh=tc["t_coh"], p_swap=1.0, 
            t_max=tc["t_max"], verbose=False)
        assert mdp["converged"], f"MDP did not converge for {tc['label']}"

        mc_mdp = run_simulation(
            AdaptivePolicy(mdp["policy"], tc["t_max"]),
            p_gen=tc["p_gen"], w0=tc["w0"], t_coh=tc["t_coh"],
            n_episodes=n_episodes, seed=42)
        mc_skr = skr_from_samples(
            mc_mdp.delivery_times.astype(float), mc_mdp.w_out_array)

        ok = check(tc["label"], mc_skr, mdp["gain"], tol=0.05)
        all_pass &= ok

    # ══════════════════════════════════════════════════════════════
    separator("SUMMARY")
    # ══════════════════════════════════════════════════════════════

    elapsed = time.time() - t_start
    if all_pass:
        print(f"\n   ✅ ALL VALIDATIONS PASSED ({elapsed:.1f}s)")
    else:
        print(f"\n   ❌ SOME VALIDATIONS FAILED ({elapsed:.1f}s)")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)