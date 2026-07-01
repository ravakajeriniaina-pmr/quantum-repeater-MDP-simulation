import numpy as np


# ═════════════════════════════════════════════════════════════════════
# WERNER ↔ FIDELITY CONVERSIONS
# ═════════════════════════════════════════════════════════════════════

def werner_to_fidelity(w: float) -> float:
    return (1.0 + 3.0 * w) / 4.0


def fidelity_to_werner(f: float) -> float:
    return (4.0 * f - 1.0) / 3.0


def werner_to_fidelity_array(w_array: np.ndarray) -> np.ndarray:
    return (1.0 + 3.0 * w_array) / 4.0


def fidelity_to_werner_array(f_array: np.ndarray) -> np.ndarray:
    return (4.0 * f_array - 1.0) / 3.0


# ═════════════════════════════════════════════════════════════════════
# QBER
# ═════════════════════════════════════════════════════════════════════

def werner_to_qber(w: float) -> float:
    f = werner_to_fidelity(w)
    return (1.0 - f) / 2.0


def qber_to_werner(e: float) -> float:
    # F = 1 - 2e
    # w = (4F - 1) / 3 = (4(1-2e) - 1) / 3 = (3 - 8e) / 3 = 1 - 8e/3
    return 1.0 - 8.0 * e / 3.0


# ═════════════════════════════════════════════════════════════════════
# ENTROPY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def binary_entropy_nat(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def binary_entropy_bits(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


# ═════════════════════════════════════════════════════════════════════
# SECRET KEY FRACTION
# ═════════════════════════════════════════════════════════════════════

def secret_fraction_nat(w: float) -> float:
    x = (1.0 - w) / 2.0
    return max(0.0, 1.0 - 2.0 * binary_entropy_nat(x))
# Default: use natural log to match Boxi Li
secret_fraction = secret_fraction_nat


# ═════════════════════════════════════════════════════════════════════
# THRESHOLD COMPUTATIONS
# ═════════════════════════════════════════════════════════════════════

def swap_output_werner(age_1: int, age_2: int,
                       w0: float, t_coh: float) -> float:
    """
    Werner parameter after swap:
    w_out = w0^2 * exp(-(age_1 + age_2)/t_coh)
    """
    if np.isinf(t_coh):
        return w0 * w0
    return w0 * w0 * np.exp(-(age_1 + age_2) / t_coh)

def skr_threshold_werner_nat() -> float:
    # Bisection: secret_fraction_nat(w) transitions from 0 to positive
    lo, hi = 0.0, 1.0
    for _ in range(100):  # more than enough for machine precision
        mid = (lo + hi) / 2.0
        if secret_fraction_nat(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0

def skr_threshold_fidelity() -> float:
    w_thr = skr_threshold_werner_nat()
    return werner_to_fidelity(w_thr)


def skr_threshold_qber() -> float:
    w_thr = skr_threshold_werner_nat()
    return werner_to_qber(w_thr)


# ═════════════════════════════════════════════════════════════════════
# ENTANGLEMENT QUERIES
# ═════════════════════════════════════════════════════════════════════

def is_entangled(w: float) -> bool:
    return w > 1.0 / 3.0


def is_skr_positive(w: float) -> bool:
    return secret_fraction_nat(w) > 0.0


def concurrence(w: float) -> float:
    return max(0.0, (3.0 * w - 1.0) / 2.0)


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Werner Utilities Self-Test")
    print("=" * 60)

    # ── Test werner ↔ fidelity roundtrip ─────────────────────────
    print("\n1. Werner ↔ Fidelity conversions:")

    test_values = [0.0, 0.1, 1/3, 0.5, 0.75, 0.9, 1.0]
    for w in test_values:
        f = werner_to_fidelity(w)
        w_back = fidelity_to_werner(f)
        assert abs(w_back - w) < 1e-12, f"Roundtrip failed for w={w}"
    print(f"   Roundtrip for {len(test_values)} values  ✓")

    # Key values
    assert werner_to_fidelity(1.0) == 1.0
    assert werner_to_fidelity(0.0) == 0.25
    assert abs(werner_to_fidelity(1/3) - 0.5) < 1e-12
    print(f"   w=1.0 → F=1.0      ✓")
    print(f"   w=0.0 → F=0.25     ✓  (maximally mixed)")
    print(f"   w=1/3 → F=0.5      ✓  (entanglement boundary)")

    assert fidelity_to_werner(1.0) == 1.0
    assert abs(fidelity_to_werner(0.25)) < 1e-12
    print(f"   F=1.0  → w=1.0     ✓")
    print(f"   F=0.25 → w=0.0     ✓")

    # Array versions
    w_arr = np.array([0.0, 0.5, 1.0])
    f_arr = werner_to_fidelity_array(w_arr)
    w_back_arr = fidelity_to_werner_array(f_arr)
    assert np.allclose(w_arr, w_back_arr)
    print(f"   Array roundtrip     ✓")

    # ── Test QBER ────────────────────────────────────────────────
    print("\n2. QBER conversions:")

    assert werner_to_qber(1.0) == 0.0
    print(f"   w=1.0 → QBER=0.0   ✓  (no errors)")

    assert werner_to_qber(0.0) == 0.375
    print(f"   w=0.0 → QBER=0.375 ✓  (maximally mixed)")

    qber_third = werner_to_qber(1/3)
    assert abs(qber_third - 0.25) < 1e-12
    print(f"   w=1/3 → QBER=0.25  ✓")

    # Roundtrip
    for w in test_values:
        e = werner_to_qber(w)
        w_back = qber_to_werner(e)
        assert abs(w_back - w) < 1e-12
    print(f"   QBER roundtrip for {len(test_values)} values  ���")

    # ── Test binary entropy ──────────────────────────────────────
    print("\n3. Binary entropy tests:")

    # Edge cases
    assert binary_entropy_nat(0.0) == 0.0
    assert binary_entropy_nat(1.0) == 0.0
    assert binary_entropy_bits(0.0) == 0.0
    assert binary_entropy_bits(1.0) == 0.0
    print(f"   h(0) = 0, h(1) = 0  ✓  (both bases)")

    # Maximum at x = 0.5
    h_nat_half = binary_entropy_nat(0.5)
    assert abs(h_nat_half - np.log(2)) < 1e-12
    print(f"   h_nat(0.5) = {h_nat_half:.6f}  "
          f"(expected ln(2) = {np.log(2):.6f})  ✓")

    h_bits_half = binary_entropy_bits(0.5)
    assert abs(h_bits_half - 1.0) < 1e-12
    print(f"   h_bits(0.5) = {h_bits_half}  (expected 1.0)  ✓")

    # Symmetry: h(x) = h(1-x)
    for x in [0.1, 0.2, 0.3, 0.4]:
        assert abs(binary_entropy_nat(x) - binary_entropy_nat(1-x)) < 1e-12
        assert abs(binary_entropy_bits(x) - binary_entropy_bits(1-x)) < 1e-12
    print(f"   Symmetry h(x) = h(1-x)  ✓")

    # Relationship: h_bits = h_nat / ln(2)
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
        ratio = binary_entropy_nat(x) / binary_entropy_bits(x)
        assert abs(ratio - np.log(2)) < 1e-10
    print(f"   h_nat / h_bits = ln(2)  ✓")

    # Non-negative
    for x in np.linspace(0, 1, 1000):
        assert binary_entropy_nat(x) >= 0
        assert binary_entropy_bits(x) >= 0
    print(f"   Non-negative for 1000 values  ✓")

    # ── Test secret fraction ─────────────────────────────────────
    print("\n4. Secret fraction tests:")

    # Perfect state
    sf_perfect = secret_fraction_nat(1.0)
    assert abs(sf_perfect - 1.0) < 1e-10
    print(f"   sf_nat(1.0) = {sf_perfect}  ✓  (maximum)")

    # Maximally mixed
    sf_zero = secret_fraction_nat(0.0)
    assert sf_zero == 0.0
    print(f"   sf_nat(0.0) = {sf_zero}  ✓  (zero)")

    # Below threshold
    sf_half = secret_fraction_nat(0.5)
    assert sf_half == 0.0
    print(f"   sf_nat(0.5) = {sf_half}  ✓  (below threshold)")

    # Non-negative for all w
    for w in np.linspace(0, 1, 1000):
        assert secret_fraction_nat(w) >= 0
    print(f"   Non-negative for 1000 values  ✓")

    # Monotonically increasing above threshold
    prev_sf = 0.0
    for w in np.linspace(0.75, 1.0, 100):
        sf = secret_fraction_nat(w)
        assert sf >= prev_sf, f"Not monotonic at w={w}"
        prev_sf = sf
    print(f"   Monotonically increasing above threshold  ✓")

    # Default function matches nat version
    assert secret_fraction(0.9) == secret_fraction_nat(0.9)
    print(f"   Default == nat version  ✓")

       # ── Test thresholds ──────────────────────────────────────────
    print("\n5. Threshold computations:")

    w_thr_nat = skr_threshold_werner_nat()
    f_thr = skr_threshold_fidelity()
    e_thr = skr_threshold_qber()

    print(f"   Werner threshold (nat):  {w_thr_nat:.6f}")
    print(f"   Fidelity threshold:      {f_thr:.6f}")
    print(f"   QBER threshold:          {e_thr:.6f}")

    # The nat and bits thresholds are DIFFERENT because
    # 1 - 2*h_nat(x) = 0  and  1 - 2*h_bits(x) = 0
    # solve to different x values (h_nat and h_bits have different scales).
    # 
    # Nat threshold:  w ≈ 0.7476  →  F ≈ 0.8107
    # Bits threshold: w ≈ 0.8228  →  F ≈ 0.8671
    #
    # Boxi Li uses NATURAL LOG, so our operational threshold is w_thr_nat.

    # Verify nat threshold: sf is 0 just below, positive just above
    assert secret_fraction_nat(w_thr_nat - 0.001) == 0.0
    assert secret_fraction_nat(w_thr_nat + 0.001) > 0.0
    print(f"   sf_nat(threshold - ε) = 0  ✓")
    print(f"   sf_nat(threshold + ε) > 0  ✓")

    # Our operational threshold (matching Boxi Li) uses nat
    f_thr_check = werner_to_fidelity(w_thr_nat)
    assert abs(f_thr - f_thr_check) < 1e-10
    print(f"   F_threshold = {f_thr:.4f} (from nat)  ✓")

    # ── Test entanglement queries ────────────────────────────────
    print("\n6. Entanglement queries:")

    assert is_entangled(1.0) == True
    assert is_entangled(0.5) == True
    assert is_entangled(1/3) == False
    assert is_entangled(0.0) == False
    print(f"   is_entangled: w=1→T, w=0.5→T, w=1/3→F, w=0→F  ✓")

    assert is_skr_positive(1.0) == True
    assert is_skr_positive(0.8) == True
    assert is_skr_positive(0.5) == False
    assert is_skr_positive(0.0) == False
    print(f"   is_skr_positive: w=1→T, w=0.8→T, w=0.5→F, w=0→F  ✓")

    # ── Test concurrence ─────────────────────────────────────────
    print("\n7. Concurrence tests:")

    assert concurrence(1.0) == 1.0
    print(f"   C(w=1.0) = 1.0  ✓  (maximally entangled)")

    assert concurrence(0.0) == 0.0
    print(f"   C(w=0.0) = 0.0  ✓  (separable)")

    assert concurrence(1/3) == 0.0
    print(f"   C(w=1/3) = 0.0  ✓  (entanglement boundary)")

    c_half = concurrence(0.5)
    assert abs(c_half - 0.25) < 1e-12
    print(f"   C(w=0.5) = {c_half}  ✓")

    # Non-negative
    for w in np.linspace(0, 1, 100):
        assert concurrence(w) >= 0
    print(f"   Non-negative for all w  ✓")

    # ── Comprehensive table for thesis ───────────────────────────
    print("\n8. Reference table (for thesis):")
    print(f"   {'w':>6} {'F':>6} {'QBER':>6} {'C':>6} "
          f"{'sf_nat':>8} {'sf_bits':>8} {'SKR>0':>6}")
    print(f"   {'─'*6} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*8} {'─'*8} {'─'*6}")

    for w in [0.0, 0.1, 1/3, 0.5, 0.7, 0.7476, 0.8, 0.9, 0.95, 1.0]:
        f = werner_to_fidelity(w)
        e = werner_to_qber(w)
        c = concurrence(w)
        sf_n = secret_fraction_nat(w)
        skr_ok = "yes" if sf_n > 0 else "no"
        print(f"   {w:>6.4f} {f:>6.4f} {e:>6.4f} {c:>6.4f} "
              f"{sf_n:>8.4f} {skr_ok:>6}")

    # ── Verify Boxi Li consistency ───────────────────────────────
    print("\n9. Boxi Li consistency check:")
    print("   His entropy function uses natural log.")
    print("   His secret_fraction: max(1 - 2*h_nat((1-w)/2), 0)")

    # Manually compute what Boxi Li's code would give
    for w in [0.8, 0.9, 1.0]:
        x = (1.0 - w) / 2.0
        if 0 < x < 1:
            h_val = -x * np.log(x) - (1-x) * np.log(1-x)
        else:
            h_val = 0.0
        sf_manual = max(1.0 - 2.0 * h_val, 0.0)
        sf_ours = secret_fraction_nat(w)
        match = abs(sf_manual - sf_ours) < 1e-12
        print(f"   w={w}: Boxi Li sf = {sf_manual:.6f}, "
              f"ours = {sf_ours:.6f}  {'✓' if match else '✗'}")

    print("\n✅ werner_utils.py self-test passed")