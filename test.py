"""
preflight_test.py
-----------------
Quick sanity tests BEFORE running full_study.

Checks:
1) imports
2) one no-cutoff MC run
3) one fixed-cutoff MC run
4) MDP solve + adaptive MC run
5) small cutoff scan
6) optional plotting data shape checks
"""

import traceback
import numpy as np

def ok(msg):
    print(f"✅ {msg}")

def fail(msg):
    print(f"❌ {msg}")
    raise RuntimeError(msg)

def main():
    print("=" * 60)
    print("Preflight Test: Quantum Repeater Pipeline")
    print("=" * 60)

    # ── 1. Imports ───────────────────────────────────────────────
    try:
        from src.simulation.mc_engine import (
            NoCutoffPolicy, FixedCutoffPolicy, AdaptivePolicy, run_simulation
        )
        from src.metrics.skr import skr_from_samples, skr_with_ci
        from src.mdp.value_iteration import solve_mdp
        from src.mdp.state_space import suggest_t_max
        from src.metrics.plob import pgen_to_distance, transmissivity, plob_bound
        from src.physical.werner_utils import werner_to_fidelity, secret_fraction_nat
        ok("Imports")
    except Exception:
        traceback.print_exc()
        fail("Import stage failed")

    # Test params (small for speed)
    p_gen = 0.1
    p_swap = 0.5
    w0 = 0.98
    t_coh = 100.0
    n_small = 3000

    # ── 2. No-cutoff smoke MC ────────────────────────────────────
    try:
        mc_no = run_simulation(
            NoCutoffPolicy(), p_gen=p_gen, p_swap=p_swap, w0=w0, t_coh=t_coh,
            n_episodes=n_small, seed=42
        )
        skr_no = skr_from_samples(mc_no.delivery_times.astype(float), mc_no.w_out_array)
        assert np.isfinite(skr_no) and skr_no >= 0.0
        ok(f"No-cutoff MC (SKR={skr_no:.3e})")
    except Exception:
        traceback.print_exc()
        fail("No-cutoff MC failed")

    # ── 3. Fixed-cutoff smoke MC ─────────────────────────────────
    try:
        mc_fx = run_simulation(
            FixedCutoffPolicy(16), p_gen=p_gen, p_swap=p_swap, w0=w0, t_coh=t_coh,
            n_episodes=n_small, seed=43
        )
        skr_fx = skr_from_samples(mc_fx.delivery_times.astype(float), mc_fx.w_out_array)
        assert np.isfinite(skr_fx) and skr_fx >= 0.0
        ok(f"Fixed-cutoff MC (n*=16, SKR={skr_fx:.3e})")
    except Exception:
        traceback.print_exc()
        fail("Fixed-cutoff MC failed")

    # ── 4. MDP solve + adaptive MC ───────────────────────────────
    try:
        t_max = suggest_t_max(t_coh)
        t_max = min(max(t_max, 30), 80)  # cap for speed

        mdp = solve_mdp(
            p_gen=p_gen, p_swap=p_swap, w0=w0, t_coh=t_coh,
            t_max=t_max, verbose=False
        )
        assert "policy" in mdp and isinstance(mdp["policy"], dict)
        assert len(mdp["policy"]) > 0
        assert "gain" in mdp and np.isfinite(mdp["gain"])

        pol_ad = AdaptivePolicy(mdp["policy"], t_max)
        mc_ad = run_simulation(
            pol_ad, p_gen=p_gen, p_swap=p_swap, w0=w0, t_coh=t_coh,
            n_episodes=n_small, seed=44
        )
        skr_ad = skr_from_samples(mc_ad.delivery_times.astype(float), mc_ad.w_out_array)
        assert np.isfinite(skr_ad) and skr_ad >= 0.0
        ok(f"MDP+Adaptive MC (t_max={t_max}, gain={mdp['gain']:.3e}, SKR={skr_ad:.3e})")
    except Exception:
        traceback.print_exc()
        fail("MDP/adaptive stage failed")

    # ── 5. Tiny cutoff scan ──────────────────────────────────────
    try:
        best_n = None
        best_skr = -np.inf
        for n in range(1, 11):
            mc = run_simulation(
                FixedCutoffPolicy(n), p_gen=p_gen, p_swap=p_swap, w0=w0, t_coh=t_coh,
                n_episodes=1500, seed=100 + n
            )
            skr = skr_from_samples(mc.delivery_times.astype(float), mc.w_out_array)
            if skr > best_skr:
                best_skr = skr
                best_n = n
        assert best_n is not None
        ok(f"Tiny cutoff scan (best n*={best_n}, SKR={best_skr:.3e})")
    except Exception:
        traceback.print_exc()
        fail("Cutoff scan failed")

    # ── 6. PLOB mapping smoke ────────────────────────────────────
    try:
        dist = pgen_to_distance(p_gen, n_segments=2)
        eta = transmissivity(dist)
        kplob = plob_bound(eta)
        assert np.isfinite(dist) and dist >= 0.0
        assert eta >= 0.0
        assert kplob >= 0.0 or np.isinf(kplob)
        ok(f"PLOB mapping (distance={dist:.2f} km, eta={eta:.3e}, PLOB={kplob:.3e})")
    except Exception:
        traceback.print_exc()
        fail("PLOB mapping failed")

    print("\n" + "=" * 60)
    print("✅ PREFLIGHT PASSED — you can run full simulation safely.")
    print("=" * 60)


if __name__ == "__main__":
    main()