import numpy as np


# Action constants (avoid circular import with actions.py)
_WAIT = 0
_SWAP = 1
_CUTOFF_1 = 2
_CUTOFF_2 = 3
_CUTOFF_ALL = 4


# ═════════════════════════════════════════════════════════════════════
# SINGLE-LINK TRANSITIONS
# ═════════════════════════════════════════════════════════════════════

def _link_wait_transitions(status: int, age: int,
                           p_gen: float) -> list:
    if status == 1:
        # Entangled link ages deterministically
        return [(1.0, 1, age + 1)]
    else:
        # Pending link attempts generation
        outcomes = []
        if p_gen > 0:
            outcomes.append((p_gen, 1, 0))           # success
        if p_gen < 1:
            outcomes.append((1.0 - p_gen, 0, age + 1))  # failure
        return outcomes


# ═════════════════════════════════════════════════════════════════════
# FULL TWO-LINK TRANSITIONS
# ═════════════════════════════════════════════════════════════════════

def transition(state: tuple, action: int,
               p_gen: float) -> list:
    s1, a1, s2, a2 = state

    if action == _SWAP:
        if s1 != 1 or s2 != 1:
            raise ValueError(
                f"SWAP requires both links entangled, got status=({s1},{s2})")
    # In average-reward MDP, after SWAP the system restarts.
    # The reward is handled separately in rewards.py.
    # Transition: restart from (0,0,0,0) with probability 1.
        return [(1.0, (0, 0, 0, 0))]

    # For cutoff actions, reset the appropriate link(s) first
    if action == _CUTOFF_1:
        if s1 != 1:
            raise ValueError(
                f"CUTOFF_1 requires link 1 entangled, got status_1={s1}")
        s1, a1 = 0, 0
    elif action == _CUTOFF_2:
        if s2 != 1:
            raise ValueError(
                f"CUTOFF_2 requires link 2 entangled, got status_2={s2}")
        s2, a2 = 0, 0
    elif action == _CUTOFF_ALL:
        if s1 != 1 or s2 != 1:
            raise ValueError(
                f"CUTOFF_ALL requires both links entangled, "
                f"got status=({s1},{s2})")
        s1, a1 = 0, 0
        s2, a2 = 0, 0
    elif action != _WAIT:
        raise ValueError(f"Unknown action {action}")

    # Now apply WAIT dynamics to the (possibly reset) state
    link1_outcomes = _link_wait_transitions(s1, a1, p_gen)
    link2_outcomes = _link_wait_transitions(s2, a2, p_gen)

    # Combine independent link outcomes
    successors = []
    for p1, ns1, na1 in link1_outcomes:
        for p2, ns2, na2 in link2_outcomes:
            prob = p1 * p2
            next_state = (ns1, na1, ns2, na2)
            successors.append((prob, next_state))

    return successors


def transition_compact(state: tuple, action: int,
                       p_gen: float, t_max: int) -> list:
    raw = transition(state, action, p_gen)

    # Compact and merge
    merged = {}
    for prob, (ns1, na1, ns2, na2) in raw:
        # Collapse pending ages
        if ns1 == 0:
            na1 = 0
        else:
            na1 = min(na1, t_max)
        if ns2 == 0:
            na2 = 0
        else:
            na2 = min(na2, t_max)

        compact = (ns1, na1, ns2, na2)
        merged[compact] = merged.get(compact, 0.0) + prob

    return [(p, s) for s, p in merged.items()]


# ═══════��═════════════════════════════════════════════════════════════
# TRANSITION MATRIX BUILDER (for value iteration)
# ═════════════════════════════════════════════════════════════════════

def build_transition_dict(state_list: list, action: int,
                          p_gen: float, t_max: int) -> dict:
    from mdp.actions import is_valid_action
    
    result = {}
    for state in state_list:
        s1, a1, s2, a2 = state
        if is_valid_action(action, s1, s2):
            result[state] = transition_compact(
                state, action, p_gen, t_max)
    return result


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MDP Transitions Self-Test")
    print("=" * 60)

    p = 0.1  # generation probability

    # ── Test WAIT from both pending ──────────────────────────────
    print("\n1. WAIT from (0,0,0,0):")

    successors = transition((0, 0, 0, 0), _WAIT, p)
    print(f"   {len(successors)} successor states:")

    total_prob = 0.0
    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")
        total_prob += prob

    assert abs(total_prob - 1.0) < 1e-10
    print(f"   Total probability: {total_prob}  ✓")

    # Check by looking up specific states rather than assuming order
    succ_dict = {s: pr for pr, s in successors}

    assert abs(succ_dict[(0, 1, 0, 1)] - (1 - p) ** 2) < 1e-10
    print(f"   Both fail:    p={(1-p)**2:.4f} → (0,1,0,1)  ✓")

    assert abs(succ_dict[(1, 0, 1, 0)] - p ** 2) < 1e-10
    print(f"   Both succeed: p={p**2:.4f} → (1,0,1,0)  ✓")

    assert abs(succ_dict[(1, 0, 0, 1)] - p * (1 - p)) < 1e-10
    print(f"   L1 only:      p={p*(1-p):.4f} → (1,0,0,1)  ✓")

    assert abs(succ_dict[(0, 1, 1, 0)] - (1 - p) * p) < 1e-10
    print(f"   L2 only:      p={(1-p)*p:.4f} → (0,1,1,0)  ✓")


        # ── Test WAIT from one entangled ─────────────────────────────
    print("\n2. WAIT from (1,5,0,3):")

    successors = transition((1, 5, 0, 3), _WAIT, p)
    total_prob = sum(pr for pr, _ in successors)

    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")

    assert abs(total_prob - 1.0) < 1e-10
    print(f"   Total probability: {total_prob}  ✓")

    # Link 1 ages deterministically: age 5→6
    # Link 2 attempts generation
    assert len(successors) == 2

    succ_dict = {s: pr for pr, s in successors}

    # Failure: link 2 stays pending
    assert abs(succ_dict[(1, 6, 0, 4)] - (1 - p)) < 1e-10
    print(f"   Link 1 ages to 6, link 2 fails: (1,6,0,4)  ✓")

    # Success: link 2 generates
    assert abs(succ_dict[(1, 6, 1, 0)] - p) < 1e-10
    print(f"   Link 1 ages to 6, link 2 succeeds: (1,6,1,0)  ✓")

    # ── Test WAIT from both entangled ────────────────────────────
    print("\n3. WAIT from (1,10,1,20):")

    successors = transition((1, 10, 1, 20), _WAIT, p)
    assert len(successors) == 1  # both age deterministically
    assert successors[0] == (1.0, (1, 11, 1, 21))
    print(f"   Both age: (1,11,1,21) with p=1.0  ✓")

    # ── Test SWAP ────────────────────────────────────────────────
    print("\n4. SWAP from (1,10,1,20):")

    successors = transition((1, 10, 1, 20), _SWAP, p)
    assert successors == []
    print(f"   Terminal: no successors  ✓")

    # SWAP from wrong state
    try:
        transition((0, 0, 1, 5), _SWAP, p)
        print(f"   SWAP from (0,_,1,_): should have raised  ✗")
    except ValueError:
        print(f"   SWAP from (0,_,1,_): ValueError raised  ✓")

    # ── Test CUTOFF_1 ────────────────────────────────────────────
    print("\n5. CUTOFF_1 from (1,50,0,3):")

    successors = transition((1, 50, 0, 3), _CUTOFF_1, p)
    total_prob = sum(pr for pr, _ in successors)

    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")

    assert abs(total_prob - 1.0) < 1e-10
    print(f"   Total probability: {total_prob}  ✓")

    # After cutoff_1: link 1 resets to pending, then WAIT
    # Both links are now pending → same as WAIT from (0,0,0,3)
    # 4 outcomes: both fail, 1 succeeds, 2 succeeds, both succeed
    assert len(successors) == 4
    print(f"   4 successors (both pending → 4 outcomes)  ✓")

    # Check that link 1 was reset (age not 50 or 51)
    for _, (ns1, na1, ns2, na2) in successors:
        assert na1 <= 1  # either 0 (generated) or 1 (failed)
    print(f"   Link 1 ages are 0 or 1 (was reset from 50)  ✓")

    # ── Test CUTOFF_2 ────────────────────────────────────────────
    print("\n6. CUTOFF_2 from (0,7,1,100):")

    successors = transition((0, 7, 1, 100), _CUTOFF_2, p)
    total_prob = sum(pr for pr, _ in successors)

    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")

    assert abs(total_prob - 1.0) < 1e-10

    # Link 2 was reset → both pending → 4 outcomes
    assert len(successors) == 4
    for _, (ns1, na1, ns2, na2) in successors:
        assert na2 <= 1  # link 2 was reset
    print(f"   Link 2 reset from age 100  ✓")

    # ── Test CUTOFF_ALL from both entangled ──────────────────────
    print("\n7. CUTOFF_ALL from (1,50,1,80):")

    successors = transition((1, 50, 1, 80), _CUTOFF_ALL, p)
    total_prob = sum(pr for pr, _ in successors)

    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")

    assert abs(total_prob - 1.0) < 1e-10
    assert len(successors) == 4  # both reset → both pending → 4 outcomes

    # Should be identical to WAIT from (0,0,0,0)
    wait_from_fresh = transition((0, 0, 0, 0), _WAIT, p)
    for (pa, sa), (pb, sb) in zip(
            sorted(successors), sorted(wait_from_fresh)):
        assert abs(pa - pb) < 1e-10
        assert sa == sb
    print(f"   Same as WAIT from (0,0,0,0)  ✓")

    # ── Test CUTOFF_1 from both entangled ────────────────────────
    print("\n8. CUTOFF_1 from (1,30,1,10):")

    successors = transition((1, 30, 1, 10), _CUTOFF_1, p)
    total_prob = sum(pr for pr, _ in successors)

    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")

    assert abs(total_prob - 1.0) < 1e-10

    # Link 1 was reset (pending), link 2 ages (entangled)
    # 2 outcomes: link 1 gen success/failure × link 2 ages
    assert len(successors) == 2
    for _, (ns1, na1, ns2, na2) in successors:
        assert na2 == 11  # link 2 aged from 10 to 11
    print(f"   Link 2 aged to 11, link 1 reset  ✓")

    # ── Test error handling ──────────────────────────────────────
    print("\n9. Error handling:")

    try:
        transition((0, 0, 0, 0), _CUTOFF_1, p)
        print(f"   CUTOFF_1 from pending: should have raised  ✗")
    except ValueError:
        print(f"   CUTOFF_1 from pending: ValueError  ✓")

    try:
        transition((0, 0, 0, 0), _CUTOFF_ALL, p)
        print(f"   CUTOFF_ALL from pending: should have raised  ✗")
    except ValueError:
        print(f"   CUTOFF_ALL from pending: ValueError  ✓")

    try:
        transition((1, 5, 0, 0), _CUTOFF_2, p)
        print(f"   CUTOFF_2 link 2 pending: should have raised  ✗")
    except ValueError:
        print(f"   CUTOFF_2 link 2 pending: ValueError  ✓")

    try:
        transition((0, 0, 0, 0), 99, p)
        print(f"   Unknown action 99: should have raised  ✗")
    except ValueError:
        print(f"   Unknown action 99: ValueError  ✓")

    # ── Test compact transitions ─────────────────────────────────
    print("\n10. Compact transitions:")

    t_max = 100

    # Normal case: no clamping needed
    raw = transition((1, 50, 0, 0), _WAIT, p)
    compact = transition_compact((1, 50, 0, 0), _WAIT, p, t_max)

    # Should be same (ages within t_max)
    for (pr, sr), (pc, sc) in zip(sorted(raw), sorted(compact)):
        assert abs(pr - pc) < 1e-10
    print(f"   Normal case matches raw  ✓")

    # Edge case: age at t_max
    compact_edge = transition_compact(
        (1, t_max, 1, t_max), _WAIT, p, t_max)
    assert len(compact_edge) == 1
    assert compact_edge[0][1] == (1, t_max, 1, t_max)
    assert abs(compact_edge[0][0] - 1.0) < 1e-10
    print(f"   Age clamped at t_max={t_max}  ✓")

    # Pending age collapse
    compact_pend = transition_compact(
        (0, 0, 0, 0), _WAIT, p, t_max)
    for _, (ns1, na1, ns2, na2) in compact_pend:
        if ns1 == 0:
            assert na1 == 0, f"Pending age should be 0, got {na1}"
        if ns2 == 0:
            assert na2 == 0, f"Pending age should be 0, got {na2}"
    print(f"   Pending ages collapsed to 0  ✓")

    # ── Test probability conservation ���───────────────────────────
    print("\n11. Probability conservation (exhaustive):")

    test_states = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 50, 0, 0),
        (0, 0, 1, 50),
        (1, 50, 1, 30),
        (1, 99, 1, 99),
    ]
    test_actions_map = {
        (0, 0): [_WAIT],
        (1, 0): [_WAIT, _CUTOFF_1],
        (0, 1): [_WAIT, _CUTOFF_2],
        (1, 1): [_WAIT, _CUTOFF_1, _CUTOFF_2, _CUTOFF_ALL],
        # SWAP is terminal, not tested for prob conservation
    }

    for state in test_states:
        s1, _, s2, _ = state
        for action in test_actions_map[(s1, s2)]:
            succ = transition(state, action, p)
            total = sum(pr for pr, _ in succ)
            assert abs(total - 1.0) < 1e-10, \
                f"Prob not 1.0 for state={state}, action={action}: {total}"
    print(f"   All {len(test_states)} states × valid actions: "
          f"sum(p) = 1.0  ✓")

    # ── Test symmetry ────────────────────────────────────────────
    print("\n12. Link symmetry tests:")

    # WAIT from (1,5,0,3) vs (0,3,1,5) should be "mirror" images
    succ_a = transition((1, 5, 0, 3), _WAIT, p)
    succ_b = transition((0, 3, 1, 5), _WAIT, p)

    # Mirror: swap link 1 ↔ link 2 in successors
    def mirror(state):
        return (state[2], state[3], state[0], state[1])

    succ_a_mirrored = sorted(
        [(pr, mirror(s)) for pr, s in succ_a])
    succ_b_sorted = sorted(succ_b)

    for (pa, sa), (pb, sb) in zip(succ_a_mirrored, succ_b_sorted):
        assert abs(pa - pb) < 1e-10
        assert sa == sb
    print(f"   WAIT: (1,5,0,3) mirrors (0,3,1,5)  ✓")

    # CUTOFF_1 from (1,50,1,10) mirrors CUTOFF_2 from (1,10,1,50)
    succ_c1 = transition((1, 50, 1, 10), _CUTOFF_1, p)
    succ_c2 = transition((1, 10, 1, 50), _CUTOFF_2, p)

    succ_c1_mirrored = sorted(
        [(pr, mirror(s)) for pr, s in succ_c1])
    succ_c2_sorted = sorted(succ_c2)

    for (pa, sa), (pb, sb) in zip(succ_c1_mirrored, succ_c2_sorted):
        assert abs(pa - pb) < 1e-10
        assert sa == sb
    print(f"   CUTOFF_1 on (1,50,1,10) mirrors CUTOFF_2 on (1,10,1,50)  ✓")

       # ── Test WAIT from one entangled ─────────────────────────────
    print("\n13. WAIT from (1,5,0,3):")

    successors = transition((1, 5, 0, 3), _WAIT, p)
    total_prob = sum(pr for pr, _ in successors)

    for prob, next_s in successors:
        print(f"     p={prob:.4f} → {next_s}")

    assert abs(total_prob - 1.0) < 1e-10
    print(f"   Total probability: {total_prob}  ✓")

    # Link 1 ages deterministically: age 5→6
    # Link 2 attempts generation
    assert len(successors) == 2

    succ_dict = {s: pr for pr, s in successors}

    # Failure: link 2 stays pending
    assert abs(succ_dict[(1, 6, 0, 4)] - (1 - p)) < 1e-10
    print(f"   Link 1 ages to 6, link 2 fails: (1,6,0,4)  ✓")

    # Success: link 2 generates
    assert abs(succ_dict[(1, 6, 1, 0)] - p) < 1e-10
    print(f"   Link 1 ages to 6, link 2 succeeds: (1,6,1,0)  ✓")

    # ── Test cutoff + immediate generation ───────────────────────
    print("\n14. Cutoff + immediate generation (same time step):")

    # CUTOFF_1 from (1,50,1,10): link 1 resets and attempts gen
    succ = transition((1, 50, 1, 10), _CUTOFF_1, p)
    print(f"   CUTOFF_1 from (1,50,1,10):")
    for prob, next_s in succ:
        print(f"     p={prob:.2f} → {next_s}")

    # Check: link 1 can generate immediately (age=0)
    has_immediate_gen = any(ns1 == 1 and na1 == 0
                           for _, (ns1, na1, _, _) in succ)
    assert has_immediate_gen
    print(f"   Link 1 can generate immediately after cutoff  ✓")

    # Check: link 2 always ages by 1
    for _, (_, _, ns2, na2) in succ:
        assert ns2 == 1 and na2 == 11
    print(f"   Link 2 ages from 10→11 during same step  ✓")

    # ── Transition count summary ─────────────────────────────────
    print("\n15. Transition count summary:")
    print(f"   {'State category':<20} {'Action':<12} {'# successors':<12}")
    print(f"   {'─'*20} {'─'*12} {'─'*12}")

    cases = [
        ("Both pending", (0, 0, 0, 0), _WAIT),
        ("One entangled", (1, 5, 0, 0), _WAIT),
        ("One entangled", (1, 5, 0, 0), _CUTOFF_1),
        ("Both entangled", (1, 5, 1, 3), _WAIT),
        ("Both entangled", (1, 5, 1, 3), _CUTOFF_1),
        ("Both entangled", (1, 5, 1, 3), _CUTOFF_2),
        ("Both entangled", (1, 5, 1, 3), _CUTOFF_ALL),
    ]
    action_names = {0: "WAIT", 1: "SWAP", 2: "CUTOFF_1",
                    3: "CUTOFF_2", 4: "CUTOFF_ALL"}
    for cat, state, action in cases:
        succ = transition(state, action, p)
        print(f"   {cat:<20} {action_names[action]:<12} {len(succ):<12}")

    print("\n✅ transitions.py self-test passed")