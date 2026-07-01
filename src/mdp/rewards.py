"""
MDP reward function for the quantum repeater.

The reward is ONLY received when the agent performs a SWAP action.
All other actions (WAIT, CUTOFF) yield zero reward.

Upon SWAP, the expected reward equals p_swap times the secret key
fraction of the output Werner parameter:

    R(state, SWAP) = p_swap * secret_fraction(w_out)

where:
    w_out = w0^2 * exp(-2 * max(age1, age2) / t_coh)

With p_swap < 1, the swap may fail (prob 1 - p_swap): both links are
consumed and no key is produced. The expected reward accounts for this.

This is the "quality" of the delivered pair, weighted by success probability.
The MDP maximizes the long-run average reward:

    SKR = lim (1/T) * sum_{t=0}^{T} r_t
        = p_swap * E[secret_fraction(w_out)] / E[delivery_time]

By maximizing this ratio, the MDP balances:
    - Swapping early → shorter delivery time, higher quality
    - Swapping late → longer delivery time, lower quality (decoherence)
    - Cutting off → restart penalty, but avoid locking in bad quality

The reward uses NATURAL LOG in the entropy function, matching Boxi Li.

NOTE ON REWARD DESIGN:
    We do NOT penalize WAIT or CUTOFF with negative rewards.
    The "cost" of waiting is implicit: every time step spent
    waiting is a time step NOT earning reward. This is naturally
    captured by the average-reward MDP formulation.
"""


import numpy as np
from src.physical.werner_utils import (
    secret_fraction_nat as _secret_fraction,
    binary_entropy_nat as _binary_entropy_nat,
    swap_output_werner
)

# ═════════════════════════════════════════════════════════════════════
# SWAP OUTPUT WERNER PARAMETER
# ═════════════════════════════════════════════════════════════════════

def swap_werner(age_1: int, age_2: int,
                w0: float, t_coh: float) -> float:
    """
    Output Werner parameter after entanglement swap.
    Delegates to werner_utils.swap_output_werner — single source of truth.
    w_out = w0^2 * exp(-(age_1 + age_2) / t_coh)
    """
    return swap_output_werner(age_1, age_2, w0, t_coh)


# ═════════════════════════════════════════════════════════════════════
# REWARD FUNCTION
# ═════════════════════════════════════════════════════════════════════

# Action constants (duplicated here to avoid circular import)
_WAIT = 0
_SWAP = 1
_CUTOFF_1 = 2
_CUTOFF_2 = 3
_CUTOFF_ALL = 4


def reward(action: int, age_1: int, age_2: int,
           w0: float, t_coh: float,
           p_swap: float = 1.0) -> float:
    """
    Reward for taking an action.

    Only SWAP gives nonzero reward.
    With probabilistic swap (p_swap < 1), the expected reward is:
        R(SWAP) = p_swap * secret_fraction(w_out)

    When the swap fails (prob 1 - p_swap), both links are consumed
    and no key is produced, so the expected contribution is zero
    for the failure branch. The full expected reward is therefore:
        R(SWAP) = p_swap * secret_fraction(w_out)

    Parameters
    ----------
    action : int
        Action taken.
    age_1, age_2 : int
        Ages of the two links.
    w0 : float
        Initial Werner parameter.
    t_coh : float
        Memory coherence time.
    p_swap : float
        Swap success probability (default 1.0).

    Returns
    -------
    float
        Expected reward.
    """
    if action != 1:  # not SWAP
        return 0.0
    w_out = swap_werner(age_1, age_2, w0, t_coh)
    return p_swap * _secret_fraction(w_out)


def reward_from_state(action: int, state: tuple,
                      w0: float, t_coh: float,
                      p_swap: float = 1.0) -> float:
    """
    Compute reward from a state tuple.

    Parameters
    ----------
    action : int
        Action taken.
    state : tuple
        (status_1, age_1, status_2, age_2).
    w0 : float
        Initial Werner parameter.
    t_coh : float
        Memory coherence time.
    p_swap : float
        Swap success probability (default 1.0).

    Returns
    -------
    float

    Examples
    --------
    >>> reward_from_state(_SWAP, (1, 0, 1, 0), 1.0, 1000)
    1.0
    >>> reward_from_state(_WAIT, (0, 0, 0, 0), 1.0, 1000)
    0.0
    """
    return reward(action, state[1], state[3], w0, t_coh, p_swap)


# ═════════════════════════════════════════════════════════════════════
# REWARD ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def swap_reward(age1: int, age2: int, w0: float, t_coh: float) -> float:
    """
    Reward specifically for SWAP action (convenience function).

    Equivalent to reward(_SWAP, age1, age2, w0, t_coh).

    Parameters
    ----------
    age1, age2 : int
        Ages of the two links.
    w0 : float
        Initial Werner parameter.
    t_coh : float
        Memory coherence time.

    Returns
    -------
    float
        Secret fraction of the swap output.
    """
    w_out = swap_werner(age1, age2, w0, t_coh)
    return _secret_fraction(w_out)


def max_age_for_positive_reward(w0: float, t_coh: float) -> int:
    """
    Maximum symmetric age a such that SWAP reward is still positive
    at (a, a), i.e. secret_fraction(w0^2 * exp(-2*a/t_coh)) > 0.

    Returns
    -------
    int
        Largest integer a with strictly positive swap reward at (a, a).
    """
    if np.isinf(t_coh):
        return 10**9 if _secret_fraction(w0 * w0) > 0 else 0

    # If even fresh swap has no positive key fraction
    if _secret_fraction(w0 * w0) <= 0:
        return 0

    # Find an upper bound where reward becomes zero
    hi = 1
    while True:
        w_hi = w0 * w0 * np.exp(-2.0 * hi / t_coh)
        if _secret_fraction(w_hi) <= 0:
            break
        hi *= 2
        if hi > 10**9:
            return hi

    # Binary search last age with positive reward
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        w_mid = w0 * w0 * np.exp(-2.0 * mid / t_coh)
        if _secret_fraction(w_mid) > 0:
            lo = mid
        else:
            hi = mid - 1

    return lo

def reward_table(w0: float, t_coh: float, max_age: int) -> np.ndarray:
    """
    Precompute reward for all (age1, age2) pairs.

    table[a1, a2] = secret_fraction(w0^2 * exp(-2*max(a1,a2)/t_coh))

    Parameters
    ----------
    w0 : float
        Initial Werner parameter.
    t_coh : float
        Memory coherence time.
    max_age : int
        Maximum age (table is (max_age+1) x (max_age+1)).

    Returns
    -------
    np.ndarray
        2D array of rewards. Symmetric.
    """
    size = max_age + 1
    table = np.zeros((size, size), dtype=np.float64)

    for a1 in range(size):
        for a2 in range(size):
            table[a1, a2] = swap_reward(a1, a2, w0, t_coh)

    return table


def reward_table_vectorized(t_max, w0, t_coh):
    ages = np.arange(t_max + 1)
    a1 = ages[:, None]
    a2 = ages[None, :]
    sum_ages = a1 + a2                     # ← CORRECT
    if np.isinf(t_coh):
        w_out = np.full((t_max + 1, t_max + 1), w0 * w0)
    else:
        w_out = w0 * w0 * np.exp(-sum_ages / t_coh)  # ← CORRECT
    return np.vectorize(_secret_fraction)(w_out)
# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════���═════

if __name__ == "__main__":
    print("=" * 60)
    print("MDP Rewards Self-Test")
    print("=" * 60)

    # ── Test secret fraction ─────────────────────────────────────
    print("\n1. Secret fraction tests:")

    assert abs(_secret_fraction(1.0) - 1.0) < 1e-10
    print(f"   sf(1.0) = {_secret_fraction(1.0)}  ✓")

    assert _secret_fraction(0.0) == 0.0
    print(f"   sf(0.0) = {_secret_fraction(0.0)}  ✓")

    assert _secret_fraction(0.5) == 0.0
    print(f"   sf(0.5) = {_secret_fraction(0.5)}  ✓  (below threshold)")

    sf_09 = _secret_fraction(0.9)
    assert sf_09 > 0
    print(f"   sf(0.9) = {sf_09:.6f}  ✓  (above threshold)")

    # ── Test swap_werner ─────────────────────────────────────────
    print("\n2. swap_werner tests:")

    assert swap_werner(0, 0, 1.0, 1000) == 1.0
    print(f"   ages=(0,0): w_out = 1.0  ✓")

    w_sym = swap_werner(100, 100, 1.0, 1000)
    expected = np.exp(-200 / 1000)
    assert abs(w_sym - expected) < 1e-10
    print(f"   ages=(100,100): w_out = {w_sym:.6f}  ✓")

    # Depends on max(age1, age2)
    w_a = swap_werner(0, 200, 1.0, 1000)
    w_b = swap_werner(200, 0, 1.0, 1000)
    assert abs(w_a - w_b) < 1e-10
    print(f"   ages=(0,200) == ages=(200,0)  ✓  (symmetric)")

    w_inf = swap_werner(500, 500, 1.0, np.inf)
    assert w_inf == 1.0
    print(f"   t_coh=inf: w_out = 1.0  ✓")

    # ── Test reward function ─────────────────────────────────────
    print("\n3. Reward function tests:")

    # Non-SWAP actions give 0
    assert reward(_WAIT, 10, 20, 1.0, 1000) == 0.0
    assert reward(_CUTOFF_1, 50, 0, 1.0, 1000) == 0.0
    assert reward(_CUTOFF_2, 0, 50, 1.0, 1000) == 0.0
    assert reward(_CUTOFF_ALL, 50, 50, 1.0, 1000) == 0.0
    print(f"   WAIT, CUTOFF_1/2/ALL → reward = 0  ✓")

    # SWAP with fresh pairs → reward = sf(w0^2)
    r_fresh = reward(_SWAP, 0, 0, 1.0, 1000)
    assert abs(r_fresh - 1.0) < 1e-10
    print(f"   SWAP(0,0), w0=1: reward = {r_fresh}  ✓")

    r_imperfect = reward(_SWAP, 0, 0, 0.9, 1000)
    expected_r = _secret_fraction(0.81)  # 0.9^2
    assert abs(r_imperfect - expected_r) < 1e-10
    print(f"   SWAP(0,0), w0=0.9: reward = {r_imperfect:.6f}  ✓")

    # SWAP with old pairs → reward decreases
    r_old = reward(_SWAP, 200, 200, 1.0, 1000)
    assert r_old < r_fresh
    print(f"   SWAP(200,200): reward = {r_old:.6f} < {r_fresh:.6f}  ✓")

    # Very old pairs → reward = 0
    r_ancient = reward(_SWAP, 10000, 10000, 1.0, 1000)
    assert r_ancient == 0.0
    print(f"   SWAP(10000,10000): reward = 0  ✓  (fully decohered)")

    # ── Test reward_from_state ───────────────────────────────────
    print("\n4. reward_from_state tests:")

    r1 = reward_from_state(_SWAP, (1, 0, 1, 0), 1.0, 1000)
    assert abs(r1 - 1.0) < 1e-10
    print(f"   SWAP at state (1,0,1,0): reward = {r1}  ✓")

    r2 = reward_from_state(_WAIT, (0, 0, 0, 0), 1.0, 1000)
    assert r2 == 0.0
    print(f"   WAIT at state (0,0,0,0): reward = {r2}  ✓")

    r3 = reward_from_state(_SWAP, (1, 50, 1, 100), 1.0, 1000)
    r3_direct = reward(_SWAP, 50, 100, 1.0, 1000)
    assert abs(r3 - r3_direct) < 1e-10
    print(f"   Consistency with reward(): {r3:.6f}  ✓")

    # ── Test swap_reward ─────────────────────────────────────────
    print("\n5. swap_reward convenience function:")

    for a1, a2 in [(0, 0), (50, 0), (0, 100), (50, 100), (100, 100)]:
        sr = swap_reward(a1, a2, 1.0, 1000)
        r_check = reward(_SWAP, a1, a2, 1.0, 1000)
        assert abs(sr - r_check) < 1e-10
    print(f"   All match reward()  ✓")

       # ── Test 6: max_age_for_positive_reward ──────────────────────
    # ── Test 6: max_age_for_positive_reward ──────────────────────
    print("\n6. max_age_for_positive_reward:")

    max_a = max_age_for_positive_reward(1.0, 1000.0)

    # Consistent check with symmetric age definition
    r_at = reward(_SWAP, max_a, max_a, 1.0, 1000.0)
    r_above = reward(_SWAP, max_a + 1, max_a + 1, 1.0, 1000.0)

    status = "✓" if r_at > 0 and r_above == 0 else "✗"
    print(f"   w0=1.0, t_coh=1000: max_sym_age={max_a}, "
      f"r({max_a},{max_a})={r_at:.6f}, "
      f"r({max_a+1},{max_a+1})={r_above:.6f}  ({status})")

    assert r_at > 0, f"Reward at max_age should be positive, got {r_at}"
    assert r_above == 0.0, f"Reward above max_age should be zero, got {r_above}"
    print("   ✓")

    # ── Test reward_table ────────────────────────────────────────
    print("\n7. Reward table tests:")

    table = reward_table(1.0, 100, 20)
    assert table.shape == (21, 21)
    print(f"   Shape: {table.shape}  ✓")

    # Symmetric
    assert np.allclose(table, table.T)
    print(f"   Symmetric  ✓")

    # Non-negative
    assert np.all(table >= 0)
    print(f"   Non-negative  ✓")

    # Maximum at (0,0)
    assert table[0, 0] == np.max(table)
    print(f"   Maximum at (0,0): {table[0,0]:.6f}  ✓")

    # Decreasing along rows and columns
    for i in range(20):
        assert table[0, i] >= table[0, i+1]
    print(f"   Monotonically decreasing  ✓")

    # Matches scalar function
    for a1 in range(21):
        for a2 in range(21):
            scalar = swap_reward(a1, a2, 1.0, 100)
            assert abs(table[a1, a2] - scalar) < 1e-10
    print(f"   All entries match scalar function  ✓")

        # ── Test 8: Vectorized reward table ──────────────────────────
    print("\n8. Vectorized reward table:")

    t_max_test = 20
    w0_test = 1.0
    t_coh_test = 100.0

    # Scalar loop
    table = np.zeros((t_max_test + 1, t_max_test + 1))
    for a1 in range(t_max_test + 1):
        for a2 in range(t_max_test + 1):
            table[a1, a2] = reward(_SWAP, a1, a2, w0_test, t_coh_test)

    # Vectorized
    table_vec = reward_table_vectorized(t_max_test, w0_test, t_coh_test)

    assert table.shape == table_vec.shape, \
        f"Shape mismatch: {table.shape} vs {table_vec.shape}"
    assert np.allclose(table, table_vec, atol=1e-10)
    print(f"   Shape: {table_vec.shape}  ✓")
    print(f"   Max diff: {np.max(np.abs(table - table_vec)):.2e}  ✓")

    # ── Test reward landscape ────────────────────────────────────
    print("\n9. Reward landscape (w0=1.0, t_coh=1000):")
    print(f"   {'age1':>6} {'age2':>6} {'w_out':>8} {'reward':>8}")
    print(f"   {'─'*6} {'─'*6} {'─'*8} {'─'*8}")

    for a1, a2 in [(0, 0), (0, 50), (0, 100), (0, 150),
                   (50, 50), (100, 100), (50, 150), (145, 145)]:
        w = swap_werner(a1, a2, 1.0, 1000)
        r = swap_reward(a1, a2, 1.0, 1000)
        print(f"   {a1:>6} {a2:>6} {w:>8.4f} {r:>8.4f}")

    # ── Test reward monotonicity ─────────────────────────────────
    print("\n10. Reward monotonicity properties:")

    # Reward decreases with max age
    prev_r = swap_reward(0, 0, 1.0, 1000)
    for a in range(1, 200):
        r = swap_reward(a, a, 1.0, 1000)
        assert r <= prev_r, f"Not monotonically decreasing at age={a}"
        prev_r = r
    print(f"   Decreasing with symmetric age  ✓")

    # For fixed age1, reward decreases with age2
    for a2 in range(1, 200):
        r1 = swap_reward(0, a2 - 1, 1.0, 1000)
        r2 = swap_reward(0, a2, 1.0, 1000)
        assert r2 <= r1
    print(f"   Decreasing with age2 (fixed age1=0)  ✓")

    # Symmetric: reward(a1, a2) == reward(a2, a1)
    for a1 in range(0, 50, 5):
        for a2 in range(0, 50, 5):
            r1 = swap_reward(a1, a2, 1.0, 1000)
            r2 = swap_reward(a2, a1, 1.0, 1000)
            assert abs(r1 - r2) < 1e-10
    print(f"   Symmetric in (age1, age2)  ✓")

    # ── Test the MDP insight ─────────────────────────────────────
    print("\n11. MDP insight — why cutoff can help:")

    # Scenario: link 1 has age 0 (fresh), link 2 has age 140
    r_swap_now = swap_reward(0, 140, 1.0, 1000)
    # Alternative: cut link 2, regenerate both, swap fresh
    # Expected wait: ~2/p_gen = 20 steps for both to regenerate
    r_swap_fresh = swap_reward(0, 0, 1.0, 1000)
    print(f"   Swap now  (0,140): reward = {r_swap_now:.6f}")
    print(f"   Swap fresh (0,0):  reward = {r_swap_fresh:.6f}")
    print(f"   Quality gain from cutting: {r_swap_fresh - r_swap_now:.6f}")
    print(f"   But costs ~E[T_regen] = {1/0.1:.0f} more time steps")
    print(f"   → Trade-off: MDP must decide if gain > cost")

    print("\n✅ rewards.py self-test passed")