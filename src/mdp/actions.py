
# ═════════════════════════════════════════════════════════════════════
# ACTION CONSTANTS
# ═════════════════════════════════════════════════════════════════════

WAIT = 0
SWAP = 1
CUTOFF_1 = 2   # discard link 1 (left)
CUTOFF_2 = 3   # discard link 2 (right)
CUTOFF_ALL = 4  # discard both links

# Number of actions in the full action space
N_ACTIONS = 5

# Human-readable names
ACTION_NAMES = {
    WAIT: "WAIT",
    SWAP: "SWAP",
    CUTOFF_1: "CUTOFF_1",
    CUTOFF_2: "CUTOFF_2",
    CUTOFF_ALL: "CUTOFF_ALL",
}

# Short names for compact display
ACTION_SHORT = {
    WAIT: "W",
    SWAP: "S",
    CUTOFF_1: "C1",
    CUTOFF_2: "C2",
    CUTOFF_ALL: "CA",
}


# ═════════════════════════════════════════════════════════════════════
# AVAILABLE ACTIONS PER STATE
# ═════════════════════════════════════════════════════════════════════

def available_actions(status_1: int, status_2: int) -> list:
    if status_1 == 0 and status_2 == 0:
        # Both pending: can only wait
        return [WAIT]

    elif status_1 == 1 and status_2 == 0:
        # Link 1 entangled, link 2 pending
        return [WAIT, CUTOFF_1]

    elif status_1 == 0 and status_2 == 1:
        # Link 1 pending, link 2 entangled
        return [WAIT, CUTOFF_2]

    elif status_1 == 1 and status_2 == 1:
        # Both entangled: full action set
        return [WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL]

    else:
        raise ValueError(
            f"Invalid status: ({status_1}, {status_2}). "
            f"Must be 0 or 1.")


def available_actions_from_tuple(state: tuple) -> list:
    return available_actions(state[0], state[2])


def n_available_actions(status_1: int, status_2: int) -> int:
    return len(available_actions(status_1, status_2))


# ═════════════════════════════════════════════════════════════════════
# ACTION VALIDATION
# ═════════════════════════════════════════════════════════════════════

def is_valid_action(action: int, status_1: int, status_2: int) -> bool:
    return action in available_actions(status_1, status_2)


def validate_action(action: int, status_1: int, status_2: int) -> None:
    if action not in range(N_ACTIONS):
        raise ValueError(
            f"Unknown action {action}. Must be in [0, {N_ACTIONS-1}].")

    if not is_valid_action(action, status_1, status_2):
        valid = available_actions(status_1, status_2)
        valid_names = [ACTION_NAMES[a] for a in valid]
        raise ValueError(
            f"Action {ACTION_NAMES[action]} is not valid in state "
            f"(status=({status_1},{status_2})). "
            f"Valid actions: {valid_names}")


# ═════════════════════════════════════════════════════════════════════
# ACTION PROPERTIES
# ═════════════════════════════════════════════════════════════════════

def is_delivery_action(action: int) -> bool:
    """
    Returns True if the action delivers an entangled pair (SWAP).
    Note: in the average-reward MDP, SWAP is NOT terminal —
    the system restarts from (0,0,0,0) after each delivery.
    """
    return action == SWAP

is_terminal_action = is_delivery_action  # alias

def is_cutoff_action(action: int) -> bool:
    return action in (CUTOFF_1, CUTOFF_2, CUTOFF_ALL)


def action_name(action: int) -> str:
    return ACTION_NAMES.get(action, f"UNKNOWN({action})")


def action_short_name(action: int) -> str:
    return ACTION_SHORT.get(action, f"?{action}")


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MDP Actions Self-Test")
    print("=" * 60)

    # ── Test action constants ────────────────────────────────────
    print("\n1. Action constants:")

    assert WAIT == 0
    assert SWAP == 1
    assert CUTOFF_1 == 2
    assert CUTOFF_2 == 3
    assert CUTOFF_ALL == 4
    assert N_ACTIONS == 5
    print(f"   WAIT={WAIT}, SWAP={SWAP}, CUTOFF_1={CUTOFF_1}, "
          f"CUTOFF_2={CUTOFF_2}, CUTOFF_ALL={CUTOFF_ALL}")
    print(f"   N_ACTIONS={N_ACTIONS}  ✓")

    # ── Test available_actions ───────────────────────────────────
    print("\n2. Available actions per state:")

    # Both pending
    acts = available_actions(0, 0)
    assert acts == [WAIT]
    print(f"   (0,_,0,_): {[ACTION_NAMES[a] for a in acts]}  ✓")

    # Link 1 entangled
    acts = available_actions(1, 0)
    assert acts == [WAIT, CUTOFF_1]
    print(f"   (1,_,0,_): {[ACTION_NAMES[a] for a in acts]}  ✓")

    # Link 2 entangled
    acts = available_actions(0, 1)
    assert acts == [WAIT, CUTOFF_2]
    print(f"   (0,_,1,_): {[ACTION_NAMES[a] for a in acts]}  ✓")

    # Both entangled
    acts = available_actions(1, 1)
    assert acts == [WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL]
    print(f"   (1,_,1,_): {[ACTION_NAMES[a] for a in acts]}  ✓")

    # ── Test available_actions_from_tuple ─────────────────────────
    print("\n3. Available actions from tuple:")

    assert available_actions_from_tuple((0, 0, 0, 0)) == [WAIT]
    assert available_actions_from_tuple((1, 50, 0, 3)) == [WAIT, CUTOFF_1]
    assert available_actions_from_tuple((0, 7, 1, 20)) == [WAIT, CUTOFF_2]
    assert available_actions_from_tuple((1, 10, 1, 20)) == \
        [WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL]
    print(f"   All tuple tests passed  ✓")

    # ── Test n_available_actions ──────────────────────────────────
    print("\n4. Number of available actions:")

    assert n_available_actions(0, 0) == 1
    assert n_available_actions(1, 0) == 2
    assert n_available_actions(0, 1) == 2
    assert n_available_actions(1, 1) == 5
    print(f"   (0,0)→1, (1,0)→2, (0,1)→2, (1,1)→5  ✓")

    # ── Test is_valid_action ─────────────────────────────────────
    print("\n5. Action validation:")

    # WAIT is always valid
    for s1 in [0, 1]:
        for s2 in [0, 1]:
            assert is_valid_action(WAIT, s1, s2)
    print(f"   WAIT always valid  ✓")

    # SWAP only when both entangled
    assert is_valid_action(SWAP, 1, 1) == True
    assert is_valid_action(SWAP, 0, 0) == False
    assert is_valid_action(SWAP, 1, 0) == False
    assert is_valid_action(SWAP, 0, 1) == False
    print(f"   SWAP only when both entangled  ✓")

    # CUTOFF_1 only when link 1 entangled
    assert is_valid_action(CUTOFF_1, 1, 0) == True
    assert is_valid_action(CUTOFF_1, 1, 1) == True
    assert is_valid_action(CUTOFF_1, 0, 0) == False
    assert is_valid_action(CUTOFF_1, 0, 1) == False
    print(f"   CUTOFF_1 only when link 1 entangled  ✓")

    # CUTOFF_2 only when link 2 entangled
    assert is_valid_action(CUTOFF_2, 0, 1) == True
    assert is_valid_action(CUTOFF_2, 1, 1) == True
    assert is_valid_action(CUTOFF_2, 0, 0) == False
    assert is_valid_action(CUTOFF_2, 1, 0) == False
    print(f"   CUTOFF_2 only when link 2 entangled  ✓")

    # CUTOFF_ALL only when both entangled
    assert is_valid_action(CUTOFF_ALL, 1, 1) == True
    assert is_valid_action(CUTOFF_ALL, 0, 0) == False
    assert is_valid_action(CUTOFF_ALL, 1, 0) == False
    assert is_valid_action(CUTOFF_ALL, 0, 1) == False
    print(f"   CUTOFF_ALL only when both entangled  ✓")

    # ── Test validate_action raises errors ───────────────────────
    print("\n6. Validate action error handling:")

    # Valid action → no error
    try:
        validate_action(WAIT, 0, 0)
        print(f"   WAIT in (0,0): no error  ✓")
    except ValueError:
        print(f"   WAIT in (0,0): unexpected error  ✗")

    # Invalid action → error
    try:
        validate_action(SWAP, 0, 0)
        print(f"   SWAP in (0,0): should have raised  ✗")
    except ValueError as e:
        print(f"   SWAP in (0,0): ValueError raised  ✓")

    # Unknown action → error
    try:
        validate_action(99, 0, 0)
        print(f"   Action 99: should have raised  ✗")
    except ValueError as e:
        print(f"   Action 99: ValueError raised  ���")

    # Invalid status → error
    try:
        available_actions(2, 0)
        print(f"   Status 2: should have raised  ✗")
    except ValueError as e:
        print(f"   Status 2: ValueError raised  ✓")

    # ── Test terminal / cutoff queries ───────────────────────────
    print("\n7. Action properties:")

    assert is_terminal_action(SWAP) == True
    assert is_terminal_action(WAIT) == False
    assert is_terminal_action(CUTOFF_1) == False
    assert is_terminal_action(CUTOFF_2) == False
    assert is_terminal_action(CUTOFF_ALL) == False
    print(f"   Only SWAP is terminal  ✓")

    assert is_cutoff_action(CUTOFF_1) == True
    assert is_cutoff_action(CUTOFF_2) == True
    assert is_cutoff_action(CUTOFF_ALL) == True
    assert is_cutoff_action(WAIT) == False
    assert is_cutoff_action(SWAP) == False
    print(f"   CUTOFF_1/2/ALL are cutoff actions  ✓")

    # ── Test action names ────────────────────────────────────────
    print("\n8. Action names:")

    for a in range(N_ACTIONS):
        name = action_name(a)
        short = action_short_name(a)
        print(f"   {a}: {name} ({short})")
        assert len(name) > 0
        assert len(short) > 0

    assert action_name(99) == "UNKNOWN(99)"
    assert action_short_name(99) == "?99"
    print(f"   Unknown action handled  ✓")

    # ── Test symmetry table ──────────────────────────────────────
    print("\n9. Action availability table:")
    print(f"   {'State':>12} {'N_acts':>7} {'Actions':>30}")
    print(f"   {'─'*12} {'─'*7} {'─'*30}")

    for s1 in [0, 1]:
        for s2 in [0, 1]:
            acts = available_actions(s1, s2)
            n = n_available_actions(s1, s2)
            names = ", ".join(ACTION_NAMES[a] for a in acts)
            print(f"   ({s1},_,{s2},_) {n:>7} {names:>30}")

    # ── Test total action space size ─────────────────────────────
    print("\n10. Action space summary:")
    print(f"    Total actions defined: {N_ACTIONS}")
    print(f"    Max actions per state: {n_available_actions(1, 1)}")
    print(f"    Min actions per state: {n_available_actions(0, 0)}")

    # In the MDP, the policy must choose one action per state.
    # When both pending, the only choice is WAIT (no decision).
    # The interesting decisions happen when at least one link is stored.
    total_states = 4  # (0,0), (1,0), (0,1), (1,1)
    total_action_combos = 1
    for s1 in [0, 1]:
        for s2 in [0, 1]:
            total_action_combos *= n_available_actions(s1, s2)
    print(f"    Total action combinations (ignoring age): "
          f"{total_action_combos}")
    print(f"    = 1 × 2 × 2 × 5 = 20")

    print("\n✅ actions.py self-test passed")