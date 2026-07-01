import numpy as np


# ═════════════════════════════════════════════════════════════════════
# CORE SWAP FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def swap_werner(w1: float, w2: float, decay: float) -> float:
    return w1 * w2 * decay


def swap_fidelity(w1: float, w2: float, decay: float) -> float:
    w_out = swap_werner(w1, w2, decay)
    return (1.0 + 3.0 * w_out) / 4.0


# ═════════════════════════════════════════════════════════════════════
# SWAP WITH AGES (combines decoherence + swap in one call)
# ═════════════════════════════════════════════════════════════════════

def swap_werner_from_ages(age1: int, age2: int, w0: float,
                          t_coh: float) -> float:
    if np.isinf(t_coh):
        return w0 * w0

    # w1 = w0 * exp(-age1 / t_coh)
    # w2 = w0 * exp(-age2 / t_coh)
    # decay = exp(-|age1 - age2| / t_coh)
    # w_out = w1 * w2 * decay
    #       = w0^2 * exp(-(age1 + age2 + |age1-age2|) / t_coh)
    #       = w0^2 * exp(-2 * max(age1, age2) / t_coh)
    max_age = max(age1, age2)
    return w0 * w0 * np.exp(-2.0 * max_age / t_coh)


def swap_fidelity_from_ages(age1: int, age2: int, w0: float,
                            t_coh: float) -> float:
    w_out = swap_werner_from_ages(age1, age2, w0, t_coh)
    return (1.0 + 3.0 * w_out) / 4.0


# ═════════════════════════════════════════════════════════════════════
# MULTI-LEVEL CHAIN SWAP
# ═════════════════════════════════════════════════════════════════════

def chain_swap_werner(w_list: list, decay_list: list) -> float:
    result = 1.0
    for w in w_list:
        result *= w
    for d in decay_list:
        result *= d
    return result


def chain_swap_fidelity(w_list: list, decay_list: list) -> float:
    w_end = chain_swap_werner(w_list, decay_list)
    return (1.0 + 3.0 * w_end) / 4.0


# ═════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═════════════════════════════════════════════════════════════════════

def max_useful_age(w0: float, t_coh: float) -> int:
    # w_out = w0^2 * exp(-2*age/t_coh) >= w_threshold
    # exp(-2*age/t_coh) >= w_threshold / w0^2
    # -2*age/t_coh >= ln(w_threshold / w0^2)
    # age <= -t_coh/2 * ln(w_threshold / w0^2)

    # w_threshold for SKR > 0 (using natural log, matching Boxi Li)
    # Numerically: w ≈ 0.7476 gives secret_fraction(w) = 0
    w_threshold = 0.7476

    if w0 == 0:
        return 0

    ratio = w_threshold / (w0 * w0)
    if ratio >= 1.0:
        return 0  # even fresh pairs can't produce SKR
    if ratio <= 0:
        return int(1e9)

    if np.isinf(t_coh):
        if w0 * w0 > w_threshold:
            return int(1e9)  # never expires
        else:
            return 0

    age = -t_coh / 2.0 * np.log(ratio)
    return int(np.floor(age))


def swap_werner_table(w0: float, t_coh: float,
                      max_age: int) -> np.ndarray:
    size = max_age + 1
    table = np.zeros((size, size), dtype=np.float64)

    if np.isinf(t_coh):
        table[:, :] = w0 * w0
        return table

    for a1 in range(size):
        for a2 in range(size):
            max_a = max(a1, a2)
            table[a1, a2] = w0 * w0 * np.exp(-2.0 * max_a / t_coh)

    return table


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Entanglement Swapping Self-Test")
    print("=" * 60)

    # ── Test swap_werner ────────────────��────────────────────────
    print("\n1. swap_werner tests:")

    # Perfect inputs, no decay
    assert swap_werner(1.0, 1.0, 1.0) == 1.0
    print(f"   swap(1.0, 1.0, 1.0) = {swap_werner(1.0, 1.0, 1.0)}  ✓")

    # Multiplication property
    assert swap_werner(0.5, 0.5, 1.0) == 0.25
    print(f"   swap(0.5, 0.5, 1.0) = {swap_werner(0.5, 0.5, 1.0)}  ✓")

    # With decay
    result = swap_werner(0.8, 0.9, 0.95)
    expected = 0.8 * 0.9 * 0.95
    assert abs(result - expected) < 1e-10
    print(f"   swap(0.8, 0.9, 0.95) = {result:.6f}  "
          f"(expected {expected:.6f})  ✓")

    # Zero Werner
    assert swap_werner(0.0, 1.0, 1.0) == 0.0
    print(f"   swap(0.0, 1.0, 1.0) = {swap_werner(0.0, 1.0, 1.0)}  ✓")

    # Zero decay
    assert swap_werner(0.9, 0.9, 0.0) == 0.0
    print(f"   swap(0.9, 0.9, 0.0) = {swap_werner(0.9, 0.9, 0.0)}  ✓")

    # Commutative in w1, w2
    assert swap_werner(0.3, 0.7, 0.5) == swap_werner(0.7, 0.3, 0.5)
    print(f"   swap(0.3,0.7,0.5) == swap(0.7,0.3,0.5)  ✓  (commutative)")

    # ── Test swap_fidelity ───────────────────────────────────────
    print("\n2. swap_fidelity tests:")

    assert swap_fidelity(1.0, 1.0, 1.0) == 1.0
    print(f"   F(1.0, 1.0, 1.0) = {swap_fidelity(1.0, 1.0, 1.0)}  ✓")

    assert swap_fidelity(0.0, 0.0, 1.0) == 0.25
    print(f"   F(0.0, 0.0, 1.0) = {swap_fidelity(0.0, 0.0, 1.0)}  ✓  "
          f"(maximally mixed)")

    f_05 = swap_fidelity(0.5, 0.5, 1.0)
    expected_f = (1 + 3 * 0.25) / 4
    assert abs(f_05 - expected_f) < 1e-10
    print(f"   F(0.5, 0.5, 1.0) = {f_05:.4f}  "
          f"(expected {expected_f:.4f})  ✓")

    # ── Test swap_werner_from_ages ────────────────────────────────
    print("\n3. swap_werner_from_ages tests:")

    # Both fresh
    w_fresh = swap_werner_from_ages(0, 0, 1.0, 1000)
    assert w_fresh == 1.0
    print(f"   ages=(0,0): w_out = {w_fresh}  ✓")

    # Symmetric ages
    w_sym = swap_werner_from_ages(100, 100, 1.0, 1000)
    expected_sym = np.exp(-200 / 1000)
    assert abs(w_sym - expected_sym) < 1e-10
    print(f"   ages=(100,100): w_out = {w_sym:.6f}  "
          f"(expected {expected_sym:.6f})  ✓")

    # Asymmetric ages: result depends on max(age1, age2)
    w_asym1 = swap_werner_from_ages(0, 200, 1.0, 1000)
    w_asym2 = swap_werner_from_ages(200, 0, 1.0, 1000)
    assert abs(w_asym1 - w_asym2) < 1e-10, "Must be symmetric in age1, age2"
    print(f"   ages=(0,200) = {w_asym1:.6f}  ==  "
          f"ages=(200,0) = {w_asym2:.6f}  ✓  (symmetric)")

    # Verify: ages=(0,200) == ages=(100,100)
    # Both give exp(-2*200/1000) = exp(-0.4)... wait:
    # ages=(0,200): max=200, w_out = exp(-400/1000) = exp(-0.4)
    # ages=(100,100): max=100, w_out = exp(-200/1000) = exp(-0.2)
    # These are NOT equal. The symmetric case is actually better.
    assert w_sym > w_asym1, \
        "Symmetric ages should give better output than asymmetric"
    print(f"   ages=(100,100)={w_sym:.4f} > ages=(0,200)={w_asym1:.4f}  ✓  "
          f"(balanced is better)")

    # Verify the formula: w_out = w0^2 * exp(-2*max(age1,age2)/t_coh)
    for a1, a2 in [(0, 0), (50, 100), (100, 50), (200, 200), (0, 500)]:
        w_func = swap_werner_from_ages(a1, a2, 0.95, 500)
        w_manual = 0.95**2 * np.exp(-2 * max(a1, a2) / 500)
        assert abs(w_func - w_manual) < 1e-10, \
            f"Mismatch at ages=({a1},{a2})"
    print(f"   Formula verification for 5 age pairs  ✓")

    # Verify equivalence with step-by-step computation
    # w1 = w0*exp(-age1/t_coh), w2 = w0*exp(-age2/t_coh)
    # decay = exp(-|age1-age2|/t_coh)
    # w_out_manual = w1 * w2 * decay
    print("\n4. Equivalence with step-by-step computation:")
    for a1, a2, w0, tc in [(50, 150, 1.0, 1000),
                            (0, 300, 0.9, 500),
                            (100, 100, 0.95, 2000)]:
        # Step by step
        w1 = w0 * np.exp(-a1 / tc)
        w2 = w0 * np.exp(-a2 / tc)
        decay = np.exp(-abs(a1 - a2) / tc)
        w_manual = w1 * w2 * decay

        # Our function
        w_func = swap_werner_from_ages(a1, a2, w0, tc)

        assert abs(w_func - w_manual) < 1e-10
        print(f"   ages=({a1},{a2}), w0={w0}, t_coh={tc}: "
              f"func={w_func:.6f} == manual={w_manual:.6f}  ✓")

    # ── Test infinite coherence ──────────────────────────────────
    print("\n5. Infinite coherence time (no decoherence):")

    w_inf = swap_werner_from_ages(999, 999, 1.0, np.inf)
    assert w_inf == 1.0
    print(f"   ages=(999,999), t_coh=inf: w_out = {w_inf}  ✓")

    w_inf2 = swap_werner_from_ages(0, 1000, 0.8, np.inf)
    assert abs(w_inf2 - 0.64) < 1e-10  # 0.8^2
    print(f"   ages=(0,1000), w0=0.8, t_coh=inf: w_out = {w_inf2}  ✓  "
          f"(just w0^2)")

    # ── Test chain_swap_werner ───────────────────────────────────
    print("\n6. chain_swap_werner tests:")

    # All perfect
    w_chain = chain_swap_werner([1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0])
    assert w_chain == 1.0
    print(f"   All perfect: {w_chain}  ✓")

    # All same Werner, no decay
    w_chain2 = chain_swap_werner([0.9, 0.9, 0.9, 0.9],
                                 [1.0, 1.0, 1.0])
    expected_chain2 = 0.9**4
    assert abs(w_chain2 - expected_chain2) < 1e-10
    print(f"   4 links w=0.9, no decay: {w_chain2:.6f}  "
          f"(expected 0.9^4 = {expected_chain2:.6f})  ✓")

    # Two links with decay
    w_chain3 = chain_swap_werner([0.8, 0.8], [0.9])
    expected_chain3 = 0.8 * 0.8 * 0.9
    assert abs(w_chain3 - expected_chain3) < 1e-10
    print(f"   2 links w=0.8, decay=0.9: {w_chain3:.4f}  "
          f"(expected {expected_chain3:.4f})  ✓")

    # Fidelity version
    f_chain = chain_swap_fidelity([0.9, 0.9, 0.9, 0.9],
                                  [1.0, 1.0, 1.0])
    expected_f_chain = (1 + 3 * 0.9**4) / 4
    assert abs(f_chain - expected_f_chain) < 1e-10
    print(f"   Chain fidelity: {f_chain:.4f}  "
          f"(expected {expected_f_chain:.4f})  ✓")

    # ── Test max_useful_age ──────────────────────────────────────
    print("\n7. max_useful_age tests:")

    for w0, tc in [(1.0, 1000), (1.0, 500), (0.95, 1000), (0.9, 1000)]:
        max_age = max_useful_age(w0, tc)
        # Verify: at max_age, swap output should be near threshold
        if max_age > 0 and max_age < 1e8:
            w_at_max = swap_werner_from_ages(max_age, max_age, w0, tc)
            w_above = swap_werner_from_ages(max_age + 1, max_age + 1, w0, tc)
            print(f"   w0={w0}, t_coh={tc}: max_age={max_age}, "
                  f"w_at_max={w_at_max:.4f}, w_at_max+1={w_above:.4f}")
        else:
            print(f"   w0={w0}, t_coh={tc}: max_age={max_age}")

    assert max_useful_age(1.0, np.inf) > 1e6
    print(f"   t_coh=inf: max_age > 1e6  ✓  (never expires)")

    assert max_useful_age(0.0, 1000) == 0
    print(f"   w0=0: max_age = 0  ✓  (always useless)")

    # ── Test swap_werner_table ───────────────────────────────────
    print("\n8. swap_werner_table tests:")

    table = swap_werner_table(1.0, 100, 10)
    assert table.shape == (11, 11)
    print(f"   Shape: {table.shape}  ✓")

    assert table[0, 0] == 1.0
    print(f"   table[0,0] = {table[0,0]}  ✓")

    # Symmetric
    is_symmetric = np.allclose(table, table.T)
    assert is_symmetric
    print(f"   Symmetric: {is_symmetric}  ✓")

    # Decreasing along rows and columns
    for i in range(10):
        assert table[i, 0] >= table[i+1, 0], \
            f"Not decreasing at row {i}"
        assert table[0, i] >= table[0, i+1], \
            f"Not decreasing at col {i}"
    print(f"   Monotonically decreasing  ✓")

    # Matches scalar function
    all_match = True
    for a1 in range(11):
        for a2 in range(11):
            scalar = swap_werner_from_ages(a1, a2, 1.0, 100)
            if abs(table[a1, a2] - scalar) > 1e-12:
                all_match = False
    assert all_match
    print(f"   All table entries match scalar function  ✓")

    # ── Numerical example for thesis ─────────────────────────────
    print("\n9. Numerical example (for thesis Table):")
    print(f"   {'age1':>6} {'age2':>6} {'w_out':>8} {'F_out':>8} {'SKR>0?':>7}")
    print(f"   {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*7}")
    w0, tc = 1.0, 1000
    for a1, a2 in [(0, 0), (0, 50), (0, 100), (0, 200),
                   (50, 50), (100, 100), (50, 150), (200, 200)]:
        w = swap_werner_from_ages(a1, a2, w0, tc)
        f = swap_fidelity_from_ages(a1, a2, w0, tc)
        skr_ok = "yes" if w > 0.7476 else "no"
        print(f"   {a1:>6} {a2:>6} {w:>8.4f} {f:>8.4f} {skr_ok:>7}")

    print("\n✅ swapping.py self-test passed")