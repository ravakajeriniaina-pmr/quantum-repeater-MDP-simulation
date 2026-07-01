import numpy as np


# ═════════════════════════════════════════════════════════════════════
# PMF (Probability Mass Function)
# ═════════════════════════════════════════════════════════════════════

def generation_pmf(p_gen: float, t_trunc: int) -> np.ndarray:
    """
    PMF of single-link generation time.
    pmf[t] = P(link generated at step t)
    pmf[0] = 0  (no generation before step 1)
    pmf[1] = p_gen
    pmf[t] = p_gen * (1-p_gen)^(t-1) for t >= 1

    Convention: age at generation = 0 (not 1).
    This matches mc_engine.py where age resets to 0 on success.
    """
    pmf = np.zeros(t_trunc, dtype=np.float64)
    if t_trunc <= 1:
        raise ValueError(
        f"t_trunc must be >= 2 to contain at least pmf[1]=p_gen, "
        f"got t_trunc={t_trunc}"
    )

    t_list = np.arange(1, t_trunc, dtype=np.float64)
    pmf[1:] = p_gen * (1.0 - p_gen) ** (t_list - 1.0)

    return pmf


def generation_cdf(p_gen: float, t_trunc: int) -> np.ndarray:
    cdf = np.zeros(t_trunc, dtype=np.float64)
    if t_trunc <= 1:
        return cdf

    t_list = np.arange(1, t_trunc, dtype=np.float64)
    cdf[1:] = 1.0 - (1.0 - p_gen) ** t_list

    return cdf




# ═════════════════════════════════════════════════════════════════════
# WERNER FUNCTION AT GENERATION
# ═════════════════════════════════════════════════════════════════════

def initial_werner_function(w0: float, t_trunc: int) -> np.ndarray:
    return np.full(t_trunc, w0, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════
# STATISTICS
# ═════════════════════════════════════════════════════════════════════

def mean_generation_time(p_gen: float) -> float:
    return 1.0 / p_gen


def variance_generation_time(p_gen: float) -> float:
    return (1.0 - p_gen) / (p_gen ** 2)


def std_generation_time(p_gen: float) -> float:
    return np.sqrt(variance_generation_time(p_gen))


def pmf_coverage(pmf: np.ndarray) -> float:
    return float(np.sum(pmf))


def suggest_t_trunc(p_gen: float, coverage: float = 0.999) -> int:
    if p_gen >= 1.0:
        return 2  # instant generation, just need t=0 and t=1

    t = np.log(1.0 - coverage) / np.log(1.0 - p_gen)
    return int(np.ceil(t)) + 1  # +1 because array index starts at 0


# ═════════════════════════════════════════════════════════════════════
# SAMPLING (for Monte Carlo)
# ═════════════════════════════════════════════════════════════════════

def sample_generation_time(p_gen: float, rng: np.random.Generator) -> int:
    # NumPy geometric: number of failures before first success
    # We want: number of trials including the success = failures + 1
    return rng.geometric(p_gen)


def sample_generation_times(p_gen: float, n_samples: int,
                            rng: np.random.Generator) -> np.ndarray:
    return rng.geometric(p_gen, size=n_samples)


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Elementary Link Self-Test")
    print("=" * 60)

    # ── Test generation_pmf ──────────────────────────────────────
    print("\n1. generation_pmf tests:")

    pmf = generation_pmf(0.1, 100)
    assert pmf[0] == 0.0, "pmf[0] must be 0 (cannot generate at t=0)"
    print(f"   pmf[0] = {pmf[0]}  ✓  (no generation at t=0)")

    assert abs(pmf[1] - 0.1) < 1e-10, "pmf[1] must equal p_gen"
    print(f"   pmf[1] = {pmf[1]}  ✓  (equals p_gen)")

    expected_pmf2 = 0.1 * 0.9
    assert abs(pmf[2] - expected_pmf2) < 1e-10
    print(f"   pmf[2] = {pmf[2]}  ✓  (expected {expected_pmf2})")

    expected_pmf3 = 0.1 * 0.9**2
    assert abs(pmf[3] - expected_pmf3) < 1e-10
    print(f"   pmf[3] = {pmf[3]:.6f}  ✓  (expected {expected_pmf3:.6f})")

    # PMF must be non-negative and monotonically decreasing
    assert np.all(pmf >= 0), "PMF values must be non-negative"
    assert all(pmf[i] >= pmf[i+1] for i in range(1, len(pmf)-1)), \
        "PMF must be monotonically decreasing for t >= 1"
    print(f"   All values non-negative and decreasing  ✓")

    # ── Test coverage ────────────────────────────────────────────
    print("\n2. PMF coverage tests:")

    for p in [0.1, 0.01, 0.001]:
        t_tr = suggest_t_trunc(p, 0.999)
        pmf_test = generation_pmf(p, t_tr)
        cov = pmf_coverage(pmf_test)
        print(f"   p_gen={p}, t_trunc={t_tr}: coverage = {cov:.6f}  "
              f"({'✓' if cov >= 0.999 else '✗'})")
        assert cov >= 0.999, f"Coverage too low: {cov}"

    # ── Test mean from PMF matches analytical ────────────────────
    print("\n3. Mean generation time tests:")

    for p in [0.1, 0.01, 0.001]:
        analytical_mean = mean_generation_time(p)
        t_tr = suggest_t_trunc(p, 0.9999)
        pmf_test = generation_pmf(p, t_tr)
        t_values = np.arange(t_tr, dtype=np.float64)
        numerical_mean = np.sum(t_values * pmf_test) / np.sum(pmf_test)
        rel_error = abs(numerical_mean - analytical_mean) / analytical_mean
        print(f"   p_gen={p}: analytical={analytical_mean:.1f}, "
              f"numerical={numerical_mean:.1f}, "
              f"rel_error={rel_error:.6f}  "
              f"({'✓' if rel_error < 0.001 else '✗'})")
        assert rel_error < 0.001

    # ── Test initial_werner_function ─────────────────────────────
    print("\n4. initial_werner_function tests:")

    wf = initial_werner_function(1.0, 100)
    assert len(wf) == 100
    assert np.all(wf == 1.0), "All values must equal w0"
    print(f"   w0=1.0, length=100: all values = 1.0  ✓")

    wf2 = initial_werner_function(0.9, 50)
    assert np.all(wf2 == 0.9)
    print(f"   w0=0.9, length=50: all values = 0.9  ✓")

    wf3 = initial_werner_function(0.95, 1)
    assert len(wf3) == 1 and wf3[0] == 0.95
    print(f"   w0=0.95, length=1: correct  ✓")

    # ── Test CDF ─────────────────────────────────────────────────
    print("\n5. generation_cdf tests:")

    cdf = generation_cdf(0.1, 50)
    assert cdf[0] == 0.0
    print(f"   cdf[0] = {cdf[0]}  ✓")

    assert abs(cdf[1] - 0.1) < 1e-10
    print(f"   cdf[1] = {cdf[1]}  ✓  (equals p_gen)")

    expected_cdf10 = 1.0 - 0.9**10
    assert abs(cdf[10] - expected_cdf10) < 1e-10
    print(f"   cdf[10] = {cdf[10]:.6f}  ✓  (expected {expected_cdf10:.6f})")

    # CDF must be monotonically increasing
    assert all(cdf[i] <= cdf[i+1] for i in range(len(cdf)-1))
    print(f"   Monotonically increasing  ✓")

    # CDF = cumsum of PMF
    pmf_check = generation_pmf(0.1, 50)
    cdf_from_pmf = np.cumsum(pmf_check)
    assert np.allclose(cdf, cdf_from_pmf, atol=1e-12)
    print(f"   CDF == cumsum(PMF)  ✓")

    # ── Test variance and std ────────────────────────────────────
    print("\n6. Variance and std tests:")

    for p in [0.1, 0.01]:
        var = variance_generation_time(p)
        std = std_generation_time(p)
        expected_var = (1 - p) / p**2
        assert abs(var - expected_var) < 1e-10
        assert abs(std - np.sqrt(expected_var)) < 1e-10
        print(f"   p_gen={p}: var={var:.1f}, std={std:.2f}  ✓")

    # ── Test sampling ────────────────────────────────────────────
    print("\n7. Sampling tests:")

    rng = np.random.default_rng(42)
    p = 0.1
    n = 100_000
    samples = sample_generation_times(p, n, rng)

    assert np.all(samples >= 1), "All samples must be >= 1"
    print(f"   All {n} samples >= 1  ✓")

    sample_mean = np.mean(samples)
    expected_mean = mean_generation_time(p)
    rel_error = abs(sample_mean - expected_mean) / expected_mean
    print(f"   Sample mean = {sample_mean:.2f}, "
          f"expected = {expected_mean:.1f}, "
          f"rel_error = {rel_error:.4f}  "
          f"({'✓' if rel_error < 0.02 else '✗'})")
    assert rel_error < 0.02

    sample_var = np.var(samples)
    expected_var = variance_generation_time(p)
    rel_error_var = abs(sample_var - expected_var) / expected_var
    print(f"   Sample var = {sample_var:.2f}, "
          f"expected = {expected_var:.1f}, "
          f"rel_error = {rel_error_var:.4f}  "
          f"({'✓' if rel_error_var < 0.05 else '✗'})")
    assert rel_error_var < 0.05

    # ── Test suggest_t_trunc ─────────────────────────────────────
    print("\n8. suggest_t_trunc tests:")

    for p in [0.1, 0.01, 0.001, 0.0001]:
        suggested = suggest_t_trunc(p, 0.999)
        mean_t = mean_generation_time(p)
        ratio = suggested / mean_t
        print(f"   p_gen={p}: suggest={suggested}, "
              f"mean={mean_t:.0f}, ratio={ratio:.1f}x")

    assert suggest_t_trunc(1.0, 0.999) == 2
    print(f"   p_gen=1.0: suggest=2  ✓  (instant generation)")

    # ── Test edge case: p_gen = 1.0 ──────────────────────────────
    print("\n9. Edge case p_gen=1.0:")

    pmf_instant = generation_pmf(1.0, 5)
    assert pmf_instant[0] == 0.0
    assert pmf_instant[1] == 1.0
    assert np.all(pmf_instant[2:] == 0.0)
    print(f"   pmf = {pmf_instant}  ✓  (all probability at t=1)")

    cdf_instant = generation_cdf(1.0, 5)
    assert cdf_instant[0] == 0.0
    assert np.all(cdf_instant[1:] == 1.0)
    print(f"   cdf = {cdf_instant}  ✓")

    assert mean_generation_time(1.0) == 1.0
    print(f"   mean = 1.0  ✓")

    # ── Test edge case: very small p_gen ─────────────────────────
    print("\n10. Edge case small p_gen:")

    pmf_small = generation_pmf(1e-4, 100)
    assert pmf_small[0] == 0.0
    assert abs(pmf_small[1] - 1e-4) < 1e-15
    print(f"   p_gen=1e-4: pmf[1] = {pmf_small[1]}  ✓")
    print(f"   coverage in 100 steps = {pmf_coverage(pmf_small):.6f}  "
          f"(expected ~0.01)")

    suggested_small = suggest_t_trunc(1e-4, 0.999)
    print(f"   suggest_t_trunc = {suggested_small}")

    print("\n✅ elementary_link.py self-test passed")