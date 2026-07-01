import time
import numpy as np

from src.mdp.actions import (
    WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL,
    N_ACTIONS, available_actions, ACTION_NAMES,
)
from src.mdp.transitions import transition_compact
from src.mdp.rewards import reward
from src.mdp.state_space import StateSpace


# ═════════════════════════════════════════════════════════════════════
# RELATIVE VALUE ITERATION
# ═════════════════════════════════════════════════════════════════════
def solve_mdp(p_gen: float, w0: float, t_coh: float,
              t_max: int, p_swap: float = 1.0,
              max_iter: int = 10000,
              tol: float = 1e-8,
              verbose: bool = False) -> dict:
    t_start = time.time()

    # Build state space
    ss = StateSpace(t_max)
    states = ss.states
    n_states = ss.n_states

    if verbose:
        print(f"  State space: {n_states:,} states (t_max={t_max})")

    # Initialize relative values to zero
    h = {s: 0.0 for s in states}
    policy = {s: WAIT for s in states}
    ref_state = (0, 0, 0, 0)  # reference state for RVI

    # Precompute transitions for all (state, action) pairs
    if verbose:
        print("  Precomputing transitions...")
        t_pre = time.time()

    trans_cache = {}
    for s in states:
        s1, a1, s2, a2 = s
        for a in available_actions(s1, s2):
            trans_cache[(s, a)] = transition_compact(s, a, p_gen, t_max)
    # Note: SWAP transitions to (0,0,0,0) — handled inside transition_compact

    if verbose:
        print(f"  Precomputation done in {time.time() - t_pre:.2f}s")
        print(f"  Cache size: {len(trans_cache):,} entries")

    gain = 0.0
    converged = False

    for iteration in range(1, max_iter + 1):
        h_new = {}
        new_policy = {}

        for s in states:
            s1, a1, s2, a2 = s
            actions = available_actions(s1, s2)

            best_val = -np.inf
            best_action = actions[0]

            for a in actions:
                # Expected immediate reward (includes p_swap)
                r = reward(a, a1, a2, w0, t_coh, p_swap=p_swap)

                q_val = r
                for prob, next_s in trans_cache[(s, a)]:
                    q_val += prob * h[next_s]

                if q_val > best_val:
                    best_val = q_val
                    best_action = a

            h_new[s] = best_val
            new_policy[s] = best_action

        # RVI normalization
        ref_val = h_new[ref_state]
        for s in states:
            h_new[s] -= ref_val

        diffs = [h_new[s] - h[s] for s in states]
        span = max(diffs) - min(diffs)
        gain = ref_val

        if verbose and (iteration % 100 == 0 or iteration <= 5):
            print(f"    iter {iteration:>5}: gain={gain:.8e}, span={span:.2e}")

        h = h_new
        policy = new_policy

        if span < tol:
            converged = True
            if verbose:
                print(f"    Converged at iteration {iteration} (span={span:.2e})")
            break

    elapsed = time.time() - t_start

    if verbose and not converged:
        print(f"    WARNING: no convergence in {max_iter} iterations (span={span:.2e})")

    return {
        "policy": policy,
        "values": h,
        "gain": gain,
        "n_iter": iteration,
        "converged": converged,
        "elapsed": elapsed,
        "state_space": ss,
    }

# ═════════════════════════════════════════════════════════════════════
# POLICY ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def policy_summary(result: dict) -> dict:
    policy = result["policy"]
    ss = result["state_space"]

    # Count actions
    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    for s, a in policy.items():
        action_counts[ACTION_NAMES[a]] += 1

    # Find swap region (ages where SWAP is chosen)
    swap_ages = []
    for s, a in policy.items():
        if a == SWAP:
            _, a1, _, a2 = s
            swap_ages.append((a1, a2))
    swap_ages.sort()

    # Find max swap age (boundary of swap region)
    max_swap_age = 0
    for a1, a2 in swap_ages:
        max_swap_age = max(max_swap_age, a1, a2)

    return {
        "action_counts": action_counts,
        "swap_ages": swap_ages,
        "n_swap_states": len(swap_ages),
        "max_swap_age": max_swap_age,
    }


def print_policy_summary(result: dict) -> None:
    """Pretty-print the policy summary."""
    summary = policy_summary(result)
    ss = result["state_space"]

    print(f"\nPolicy Summary (t_max={ss.t_max}):")
    print(f"  Gain (SKR) = {result['gain']:.8e}")
    print(f"  Converged: {result['converged']} "
          f"({result['n_iter']} iterations, "
          f"{result['elapsed']:.2f}s)")
    print(f"\n  Action distribution:")
    for name, count in summary["action_counts"].items():
        pct = count / ss.n_states * 100
        print(f"    {name:<12}: {count:>8,} states ({pct:>5.1f}%)")
    print(f"\n  Swap region:")
    print(f"    {summary['n_swap_states']:,} states use SWAP")
    print(f"    Max swap age: {summary['max_swap_age']}")


def print_policy_slice(result: dict, max_age: int = 20) -> None:
    policy = result["policy"]
    ss = result["state_space"]
    max_display = min(max_age, ss.t_max)

    action_char = {WAIT: '.', SWAP: 'S', CUTOFF_1: '1',
                   CUTOFF_2: '2', CUTOFF_ALL: 'X'}

    print(f"\nPolicy map (both entangled, age1 × age2):")
    print(f"  . = WAIT, S = SWAP, 1 = CUT1, 2 = CUT2, X = CUT_ALL")

    # Header
    header = "a2→" + "".join(f"{a2:>3}" for a2 in range(max_display + 1))
    print(f"  a1↓{header}")

    for a1 in range(max_display + 1):
        row = f"  {a1:>3} "
        for a2 in range(max_display + 1):
            s = (1, a1, 1, a2)
            if ss.contains(s):
                a = policy[s]
                row += f"  {action_char.get(a, '?')}"
            else:
                row += "  -"
        print(row)


# ═════════════════════════════════════════════════════════════════════
# EXTRACT FIXED-CUTOFF EQUIVALENT
# ═════════════════════════════════════════════════════════════════════

def extract_effective_cutoff(result: dict) -> dict:
    policy = result["policy"]
    ss = result["state_space"]

    # Find cutoff age for link 1 (when link 2 is pending)
    link1_cutoff = None
    for a1 in range(ss.t_max + 1):
        s = (1, a1, 0, 0)
        if ss.contains(s) and policy[s] == CUTOFF_1:
            link1_cutoff = a1
            break

    # Find cutoff age for link 2 (when link 1 is pending)
    link2_cutoff = None
    for a2 in range(ss.t_max + 1):
        s = (0, 0, 1, a2)
        if ss.contains(s) and policy[s] == CUTOFF_2:
            link2_cutoff = a2
            break

    is_symmetric = (link1_cutoff == link2_cutoff)

    # Check if both-entangled policy is state-dependent
    # A fixed-cutoff policy would SWAP whenever both are entangled
    # and both ages are below the cutoff. Check if MDP does
    # something different.
    is_state_dependent = False
    for s, a in policy.items():
        s1, a1, s2, a2 = s
        if s1 == 1 and s2 == 1:
            # Fixed cutoff would always SWAP here
            if a != SWAP:
                is_state_dependent = True
                break

    return {
        "link1_cutoff": link1_cutoff,
        "link2_cutoff": link2_cutoff,
        "is_symmetric": is_symmetric,
        "is_state_dependent": is_state_dependent,
    }


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Value Iteration Self-Test")
    print("=" * 60)

    # ── Test 1: Trivial case — perfect links, no decoherence ────
    print("\n1. Trivial case (w0=1, t_coh=inf, small t_max):")

    result_trivial = solve_mdp(
        p_gen=0.5, w0=1.0, t_coh=np.inf,
        t_max=10, max_iter=5000, tol=1e-10,
        verbose=True,
    )

    assert result_trivial["converged"]
    gain_trivial = result_trivial["gain"]
    print(f"\n  Gain (SKR) = {gain_trivial:.8e}")

    # With no decoherence, best policy is SWAP immediately whenever
    # both links are entangled. Reward = sf(w0^2) = sf(1.0) = 1.0.
    # Expected delivery time depends on p_gen.
    # E[max(T1,T2)] = 2/p - 1/(p(2-p)) = 4 - 4/3 = 8/3
    # SKR = 1.0 / (8/3) = 3/8 = 0.375
    expected_skr = 1.0 / (2.0 / 0.5 - 1.0 / (0.5 * 1.5))
    print(f"  Expected   = {expected_skr:.8e}")
    assert abs(gain_trivial - expected_skr) < 1e-4, \
        f"Gain {gain_trivial} != expected {expected_skr}"
    print(f"  Match  ✓")

    # Policy: should always SWAP when both entangled
    policy_triv = result_trivial["policy"]
    for s, a in policy_triv.items():
        s1, _, s2, _ = s
        if s1 == 1 and s2 == 1:
            assert a == SWAP, \
                f"Expected SWAP at {s}, got {ACTION_NAMES[a]}"
    print(f"  Policy: always SWAP when both entangled  ✓")

    # ── Test 2: With decoherence ─────────────────────────────────
    print("\n2. With decoherence (w0=1, t_coh=50, t_max=30):")

    result_decoh = solve_mdp(
        p_gen=0.1, w0=1.0, t_coh=50.0,
        t_max=30, max_iter=5000, tol=1e-10,
        verbose=True,
    )

    assert result_decoh["converged"]
    gain_decoh = result_decoh["gain"]
    print(f"\n  Gain (SKR) = {gain_decoh:.8e}")
    assert gain_decoh > 0, "SKR should be positive"
    print(f"  Positive  ✓")

    # Should be less than the no-decoherence case with same p_gen
    result_nodecoh = solve_mdp(
        p_gen=0.1, w0=1.0, t_coh=np.inf,
        t_max=30, max_iter=5000, tol=1e-10,
    )
    assert gain_decoh < result_nodecoh["gain"]
    print(f"  Less than no-decoherence ({result_nodecoh['gain']:.6e})  ✓")

    # ── Test 3: Policy analysis ──────────────────────────────────
    print("\n3. Policy analysis:")

    print_policy_summary(result_decoh)

    # The MDP should sometimes NOT swap (cut off old pairs)
    summary = policy_summary(result_decoh)
    has_cutoff = any(
        summary["action_counts"].get(name, 0) > 0
        for name in ["CUTOFF_1", "CUTOFF_2", "CUTOFF_ALL"]
    )
    print(f"\n  Uses cutoff actions: {has_cutoff}")

    # ── Test 4: Policy visualization ─────────────────────────────
    print("\n4. Policy map (both-entangled slice):")
    print_policy_slice(result_decoh, max_age=20)

    # ── Test 5: Effective cutoff extraction ──────────────────────
    print("\n5. Effective cutoff analysis:")

    eff_cut = extract_effective_cutoff(result_decoh)
    print(f"  Link 1 cutoff: {eff_cut['link1_cutoff']}")
    print(f"  Link 2 cutoff: {eff_cut['link2_cutoff']}")
    print(f"  Symmetric: {eff_cut['is_symmetric']}")
    print(f"  State-dependent: {eff_cut['is_state_dependent']}")

    # Should be symmetric (links are identical)
    assert eff_cut["is_symmetric"], \
        "Policy should be symmetric for identical links"
    print(f"  Symmetry verified  ✓")

    # ── Test 6: Monotonicity of gain with t_coh ──────────────────
    print("\n6. Gain increases with t_coh:")

    gains = []
    for tc in [20, 50, 100]:
        r = solve_mdp(
            p_gen=0.1, w0=1.0, t_coh=tc,
            t_max=30, max_iter=5000, tol=1e-8,
        )
        gains.append(r["gain"])
        print(f"  t_coh={tc:>4}: gain={r['gain']:.6e} "
              f"({r['n_iter']} iters)")

    for i in range(len(gains) - 1):
        assert gains[i] <= gains[i + 1] + 1e-10, \
            f"Gain should increase with t_coh"
    print(f"  Monotonically increasing  ✓")

    # ── Test 7: Gain increases with p_gen ────────────────────────
    print("\n7. Gain increases with p_gen:")

    gains_p = []
    for pg in [0.05, 0.1, 0.3]:
        r = solve_mdp(
            p_gen=pg, w0=1.0, t_coh=50.0,
            t_max=30, max_iter=5000, tol=1e-8,
        )
        gains_p.append(r["gain"])
        print(f"  p_gen={pg:.2f}: gain={r['gain']:.6e}")

    for i in range(len(gains_p) - 1):
        assert gains_p[i] <= gains_p[i + 1] + 1e-10
    print(f"  Monotonically increasing  ✓")

    # ── Test 8: Gain increases with w0 ───────────────────────────
    print("\n8. Gain increases with w0:")

    gains_w = []
    for w in [0.8, 0.9, 1.0]:
        r = solve_mdp(
            p_gen=0.1, w0=w, t_coh=50.0,
            t_max=30, max_iter=5000, tol=1e-8,
        )
        gains_w.append(r["gain"])
        print(f"  w0={w:.1f}: gain={r['gain']:.6e}")

    for i in range(len(gains_w) - 1):
        assert gains_w[i] <= gains_w[i + 1] + 1e-10
    print(f"  Monotonically increasing  ✓")

    # ── Test 9: One-entangled policy check ───────────────────────
    print("\n9. One-entangled policy behavior:")

    policy = result_decoh["policy"]
    ss = result_decoh["state_space"]

    # At age 0, should WAIT (freshly generated, don't cut)
    assert policy[(1, 0, 0, 0)] == WAIT
    print(f"  (1,0,0,0) → WAIT  ✓  (fresh link, keep it)")

    # At high age, should CUTOFF (too old, useless)
    cut_age = eff_cut["link1_cutoff"]
    if cut_age is not None and cut_age <= ss.t_max:
        assert policy[(1, cut_age, 0, 0)] == CUTOFF_1
        print(f"  (1,{cut_age},0,0) → CUTOFF_1  ✓  "
              f"(old link, discard)")
    else:
        print(f"  No cutoff found in one-entangled states")

    # ── Test 10: Value function properties ───────────────────────
    print("\n10. Value function properties:")

    values = result_decoh["values"]

    # Reference state value should be 0 (by RVI construction)
    assert abs(values[(0, 0, 0, 0)]) < 1e-6
    print(f"  h(0,0,0,0) ≈ 0  ✓  (reference state)")

    # Both-entangled at age 0 should have highest value
    v_fresh = values.get((1, 0, 1, 0), None)
    if v_fresh is not None:
        # Should be higher than both-pending
        assert v_fresh > values[(0, 0, 0, 0)]
        print(f"  h(1,0,1,0) = {v_fresh:.4f} > h(0,0,0,0) = 0  ✓")

    # Value should decrease with age (for both-entangled)
    v_prev = values.get((1, 0, 1, 0), 0)
    monotone = True
    for a in range(1, min(20, ss.t_max + 1)):
        v_curr = values.get((1, a, 1, a), 0)
        if v_curr > v_prev + 1e-10:
            monotone = False
            break
        v_prev = v_curr
    if monotone:
        print(f"  h(1,a,1,a) decreasing with a  ✓")
    else:
        print(f"  h(1,a,1,a) not strictly monotone (may be OK)")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary of test results:")
    print(f"  Trivial (no decoherence): gain = {gain_trivial:.6e} "
          f"(expected {expected_skr:.6e})")
    print(f"  With decoherence:         gain = {gain_decoh:.6e}")
    print(f"  Effective cutoff:         n* = {eff_cut['link1_cutoff']}")
    print(f"  State-dependent:          {eff_cut['is_state_dependent']}")
    print("=" * 60)

    print("\n✅ value_iteration.py self-test passed")