import numpy as np


# ═════════════════════════════════════════════════════════════════════
# PLOB BOUND
# ═════════════════════════════════════════════════════════════════════

def plob_bound(eta: float) -> float:
    if eta <= 0.0:
        return 0.0
    if eta >= 1.0:
        return np.inf
    return -np.log2(1.0 - eta)


def plob_bound_approx(eta: float) -> float:
    return eta / np.log(2.0)


def plob_bound_array(eta_array: np.ndarray) -> np.ndarray:
    result = np.zeros_like(eta_array, dtype=np.float64)
    valid = (eta_array > 0) & (eta_array < 1)
    result[valid] = -np.log2(1.0 - eta_array[valid])
    result[eta_array >= 1.0] = np.inf
    return result


# ═════════════════════════════════════════════════════════════════════
# CHANNEL TRANSMISSIVITY
# ═════════════════════════════════════════════════════════════════════

def transmissivity(length_km: float, alpha_db_per_km: float = 0.2) -> float:
    if length_km <= 0.0:
        return 1.0
    return 10.0 ** (-alpha_db_per_km * length_km / 10.0)


def transmissivity_array(lengths_km: np.ndarray,
                         alpha_db_per_km: float = 0.2) -> np.ndarray:
    return 10.0 ** (-alpha_db_per_km * lengths_km / 10.0)


def loss_db(length_km: float, alpha_db_per_km: float = 0.2) -> float:
    return alpha_db_per_km * length_km


# ═════════════════════════════════════════════════════════════════════
# DISTANCE ↔ p_gen MAPPING
# ═══════════════════════════════════════════════════════════════���═════

def distance_to_pgen(total_distance_km: float, n_segments: int,
                     eta_coupling: float = 0.5,
                     eta_detector: float = 0.9,
                     alpha_db_per_km: float = 0.2) -> float:
    if total_distance_km <= 0.0:
        return eta_coupling * eta_detector

    segment_length = total_distance_km / n_segments
    # Each photon travels half the segment to the midpoint
    photon_distance = segment_length / 2.0
    eta_fiber = transmissivity(photon_distance, alpha_db_per_km)

    return eta_fiber * eta_coupling * eta_detector


def pgen_to_distance(p_gen: float, n_segments: int,
                     eta_coupling: float = 0.5,
                     eta_detector: float = 0.9,
                     alpha_db_per_km: float = 0.2) -> float:
    eta_reduced = p_gen / (eta_coupling * eta_detector)

    if eta_reduced >= 1.0:
        return 0.0
    if eta_reduced <= 0.0:
        return np.inf

    # η_fiber = 10^(-α * d / 10)  where d = photon distance = L0/2
    # log10(η_fiber) = -α * d / 10
    # d = -10 * log10(η_fiber) / α
    photon_distance = -10.0 * np.log10(eta_reduced) / alpha_db_per_km

    # L_total = N * L0 = N * 2 * photon_distance
    return n_segments * 2.0 * photon_distance


def pgen_array_to_distance(p_gen_array: np.ndarray,
                           n_segments: int,
                           eta_coupling: float = 0.5,
                           eta_detector: float = 0.9,
                           alpha_db_per_km: float = 0.2) -> np.ndarray:
    return np.array([
        pgen_to_distance(p, n_segments, eta_coupling, eta_detector,
                         alpha_db_per_km)
        for p in p_gen_array
    ])


# ═════════════════════════════════════════════════════════════════════
# PLOB BOUND VS DISTANCE (for plotting)
# ═════════════════════════════════════════════════════════════════════

def plob_vs_distance(distances_km: np.ndarray,
                     alpha_db_per_km: float = 0.2) -> np.ndarray:
    eta = transmissivity_array(distances_km, alpha_db_per_km)
    return plob_bound_array(eta)


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PLOB Bound & Distance Mapping Self-Test")
    print("=" * 60)

    # ── Test transmissivity ──────────────────────────────────────
    print("\n1. Transmissivity tests:")

    assert transmissivity(0.0) == 1.0
    print(f"   η(0 km)   = {transmissivity(0.0)}  ✓  (no loss)")

    eta_50 = transmissivity(50.0)
    expected_50 = 10 ** (-0.2 * 50 / 10)  # 10^(-1) = 0.1... wait
    # α=0.2 dB/km, L=50 km → loss = 10 dB → η = 10^(-1) = 0.1
    # Actually: 10^(-0.2*50/10) = 10^(-1) = 0.1
    # Hmm, let me recalculate: 0.2 * 50 = 10 dB, 10^(-10/10) = 0.1
    # But above says 0.316228...
    # 10^(-0.2*50/10) = 10^(-10/10) = 10^(-1) = 0.1
    expected_50_correct = 10.0 ** (-0.2 * 50.0 / 10.0)
    assert abs(eta_50 - expected_50_correct) < 1e-10
    print(f"   η(50 km)  = {eta_50:.6f}  "
          f"(10 dB loss)  ✓")

    eta_100 = transmissivity(100.0)
    assert abs(eta_100 - 0.01) < 1e-10
    print(f"   η(100 km) = {eta_100:.6f}  "
          f"(20 dB loss)  ✓")

    eta_200 = transmissivity(200.0)
    assert abs(eta_200 - 0.0001) < 1e-10
    print(f"   η(200 km) = {eta_200:.8f}  "
          f"(40 dB loss)  ✓")

    # Monotonically decreasing
    distances = np.linspace(0, 500, 100)
    etas = transmissivity_array(distances)
    assert all(etas[i] >= etas[i+1] for i in range(len(etas)-1))
    print(f"   Monotonically decreasing  ✓")

    # ── Test loss_db ─────────────────────────────────────────────
    print("\n2. Loss tests:")

    assert loss_db(50.0) == 10.0
    print(f"   loss(50 km)  = {loss_db(50.0)} dB  ✓")

    assert loss_db(100.0) == 20.0
    print(f"   loss(100 km) = {loss_db(100.0)} dB  ✓")

    assert loss_db(0.0) == 0.0
    print(f"   loss(0 km)   = {loss_db(0.0)} dB  ✓")

    # ── Test PLOB bound ──────────────────────────────────────────
    print("\n3. PLOB bound tests:")

    assert plob_bound(0.0) == 0.0
    print(f"   K(η=0) = {plob_bound(0.0)}  ✓")

    k_half = plob_bound(0.5)
    assert abs(k_half - 1.0) < 1e-10
    print(f"   K(η=0.5) = {k_half}  ✓  (-log2(0.5) = 1)")

    assert plob_bound(1.0) == np.inf
    print(f"   K(η=1) = inf  ✓")

    # Small η approximation
    for eta in [0.001, 0.01, 0.05]:
        exact = plob_bound(eta)
        approx = plob_bound_approx(eta)
        rel_err = abs(exact - approx) / exact
        print(f"   η={eta}: exact={exact:.6f}, "
              f"approx={approx:.6f}, err={rel_err:.4f}  "
              f"({'✓' if rel_err < 0.1 else '✗'})")

    # Monotonically increasing in η
    etas_test = np.linspace(0.001, 0.999, 100)
    plobs = plob_bound_array(etas_test)
    assert all(plobs[i] <= plobs[i+1] for i in range(len(plobs)-1))
    print(f"   Monotonically increasing in η  ✓")

    # ── Test distance ↔ p_gen roundtrip ──────────────────────────
    print("\n4. Distance ↔ p_gen roundtrip:")

    for n_seg in [4, 8]:
        for p_original in [0.1, 0.01, 0.001]:
            dist = pgen_to_distance(p_original, n_seg)
            p_back = distance_to_pgen(dist, n_seg)
            rel_err = abs(p_back - p_original) / p_original
            assert rel_err < 1e-10, \
                f"Roundtrip failed: p={p_original}, N={n_seg}"
            print(f"   N={n_seg}, p_gen={p_original} → "
                  f"{dist:.1f} km → p_gen={p_back:.6f}  ✓")

    # Edge cases
    assert pgen_to_distance(0.45, 4) == 0.0  # η_c*η_d = 0.45
    print(f"   p_gen=0.45 (>= η_c*η_d) → 0 km  ✓")

    dist_zero = distance_to_pgen(0.0, 4)
    assert abs(dist_zero - 0.45) < 1e-10  # η_c * η_d = 0.5 * 0.9
    print(f"   distance=0 → p_gen={dist_zero}  ✓  (= η_c × η_d)")

    # ── Test PLOB vs distance ────────────────────────────────────
    print("\n5. PLOB bound vs distance:")

    plob_dists = np.array([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    plob_vals = plob_vs_distance(plob_dists)
    print(f"   {'Distance':>10} {'η':>10} {'Loss(dB)':>10} {'PLOB':>12}")
    print(f"   {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
    for d, k in zip(plob_dists, plob_vals):
        eta = transmissivity(d)
        db = loss_db(d)
        print(f"   {d:>10.0f} {eta:>10.6f} {db:>10.1f} {k:>12.6f}")

    # ── Test p_gen ↔ distance reference table ────────────────────
    print("\n6. Reference table: p_gen to distance (N=4 segments):")
    print(f"   {'p_gen':>10} {'Distance':>12} {'L0':>10} "
          f"{'η_fiber':>10} {'Loss/seg':>10}")
    print(f"   {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

    for p in [0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]:
        dist = pgen_to_distance(p, 4)
        if dist < 1e10:
            l0 = dist / 4
            eta_f = p / (0.5 * 0.9)
            loss_seg = loss_db(l0)
            print(f"   {p:>10.4f} {dist:>10.1f} km {l0:>10.1f} "
                  f"{eta_f:>10.6f} {loss_seg:>8.1f} dB")
        else:
            print(f"   {p:>10.4f} {'inf':>12}")

    # ── Test array functions ─────────────────────────────────────
    print("\n7. Array function consistency:")

    dists = np.array([50.0, 100.0, 200.0])
    etas_arr = transmissivity_array(dists)
    etas_scalar = np.array([transmissivity(d) for d in dists])
    assert np.allclose(etas_arr, etas_scalar)
    print(f"   transmissivity_array matches scalar  ✓")

    plobs_arr = plob_bound_array(etas_arr)
    plobs_scalar = np.array([plob_bound(e) for e in etas_arr])
    assert np.allclose(plobs_arr, plobs_scalar)
    print(f"   plob_bound_array matches scalar  ✓")

    pgen_arr = np.array([0.1, 0.01, 0.001])
    dists_arr = pgen_array_to_distance(pgen_arr, 4)
    dists_scalar = np.array([pgen_to_distance(p, 4) for p in pgen_arr])
    assert np.allclose(dists_arr, dists_scalar)
    print(f"   pgen_array_to_distance matches scalar  ✓")

    plob_dists_arr = plob_vs_distance(dists)
    for i, d in enumerate(dists):
        eta = transmissivity(d)
        expected = plob_bound(eta)
        assert abs(plob_dists_arr[i] - expected) < 1e-12
    print(f"   plob_vs_distance matches manual  ✓")

    print("\n✅ plob.py self-test passed")