import numpy as np


# ═════════════════════════════════════════════════════════════════════
# DELIVERY TIME (no cutoff)
# ═════════════════════════════════════════════════════════════════════

def mean_delivery_time(p_gen: float, p_swap: float = 1.0) -> float:
    p = p_gen
    # E[max] = 2/p - 1/(p*(2-p))
    e_max = 2.0 / p - 1.0 / (p * (2.0 - p))
    return e_max / p_swap


def mean_max_geom(p_gen: float) -> float:
    p = p_gen
    return 2.0 / p - 1.0 / (p * (2.0 - p))


def var_max_geom(p_gen: float) -> float:
    # Compute numerically from the PMF
    # Truncate at a safe upper bound
    p = p_gen
    t_max = int(50 / p)
    t_vals = np.arange(1, t_max + 1)

    pmf = pmf_max_geom(p, t_max)

    e_max = np.sum(t_vals * pmf)
    e_max2 = np.sum(t_vals ** 2 * pmf)

    return e_max2 - e_max ** 2


# ═════════════════════════════════════════════════════════════════════
# PMF OF max(T1, T2)
# ═════════════════════════════════════════════════════════════════════

def pmf_max_geom(p_gen: float, t_max: int) -> np.ndarray:
    p = p_gen
    q = 1.0 - p

    t_vals = np.arange(1, t_max + 1)
    cdf = 1.0 - q ** t_vals
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])

    pmf = cdf ** 2 - cdf_prev ** 2
    return pmf


def cdf_max_geom(p_gen: float, t_max: int) -> np.ndarray:
    p = p_gen
    q = 1.0 - p

    t_vals = np.arange(1, t_max + 1)
    cdf_geom = 1.0 - q ** t_vals
    return cdf_geom ** 2


# ═════════════════════════════════════════════════════════════════════
# OUTPUT WERNER PARAMETER (no cutoff)
# ═════════════════════════════════════════════════════════════════════

def mean_werner_no_cutoff(p_gen: float, w0: float,
                          t_coh: float, t_max: int = None) -> float:
    if np.isinf(t_coh):
        return w0 * w0

    if t_max is None:
        t_max = int(50 / p_gen)

    p = p_gen
    q = 1.0 - p

    t_vals = np.arange(1, t_max + 1)
    pmf = p * q ** (t_vals - 1)  # shape (t_max,)

    sum_ages = t_vals[:, None] + t_vals[None, :]
    decay = np.exp(-sum_ages / t_coh)       # ← CORRECT
        # Explication : age_1 = t1, age_2 = t2 au moment du swap
        # w_out = w0^2 * exp(-(t1 + t2) / t_coh)

    # Outer product of PMFs
    joint = pmf[:, None] * pmf[None, :]  # (t_max, t_max)

    return float(w0 * w0 * np.sum(joint * decay))


def mean_fidelity_no_cutoff(p_gen: float, w0: float,
                            t_coh: float, t_max: int = None) -> float:
    mean_w = mean_werner_no_cutoff(p_gen, w0, t_coh, t_max)
    return (1.0 + 3.0 * mean_w) / 4.0


# ═════════════════════════════════════════════════════════════════════
# SECRET KEY RATE (no cutoff)
# ══════════════════��══════════════════════════════════════════════════

def _binary_entropy_nat(x: float) -> float:
    """Binary entropy with natural log."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def _secret_fraction(w: float) -> float:
    """Secret fraction matching Boxi Li."""
    x = (1.0 - w) / 2.0
    return max(0.0, 1.0 - 2.0 * _binary_entropy_nat(x))


def skr_no_cutoff(p_gen: float, w0: float, t_coh: float,
                  p_swap: float = 1.0, t_max: int = None) -> float:
    mean_t = mean_delivery_time(p_gen, p_swap)
    mean_w = mean_werner_no_cutoff(p_gen, w0, t_coh, t_max)

    sf = _secret_fraction(mean_w)
    if sf <= 0 or mean_t <= 0:
        return 0.0

    return sf / mean_t


# ═════════════════════════════════════════════════════════════════════
# COMPLETE ANALYTICAL BASELINE
# ═════════════════════════════════════════════════════════════════════

def no_cutoff_baseline(p_gen: float, w0: float, t_coh: float,
                       p_swap: float = 1.0,
                       t_max: int = None) -> dict:
    if t_max is None:
        t_max = int(50 / p_gen)

    mean_t = mean_delivery_time(p_gen, p_swap)
    mean_w = mean_werner_no_cutoff(p_gen, w0, t_coh, t_max)
    mean_f = (1.0 + 3.0 * mean_w) / 4.0
    sf = _secret_fraction(mean_w)
    skr = sf / mean_t if mean_t > 0 else 0.0
    skr = max(0.0, skr)
    pmf = pmf_max_geom(p_gen, t_max)

    return {
        "mean_t": float(mean_t),
        "mean_w": float(mean_w),
        "mean_f": float(mean_f),
        "skr": float(skr),
        "sf": float(sf),
        "pmf": pmf,
    }


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("No-Cutoff Analytical Baseline Self-Test")
    print("=" * 60)

    # ── Test mean_max_geom ───────────────────────────────────────
    print("\n1. E[max(T1, T2)] tests:")

    # p=1: both always succeed at t=1, max=1
    assert abs(mean_max_geom(1.0) - 1.0) < 1e-10
    print(f"   p=1.0: E[max] = {mean_max_geom(1.0)}  ✓")

    # p=0.5: E[max] = 2/0.5 - 1/(0.5*1.5) = 4 - 4/3 = 8/3
    expected = 8.0 / 3.0
    assert abs(mean_max_geom(0.5) - expected) < 1e-10
    print(f"   p=0.5: E[max] = {mean_max_geom(0.5):.4f}  "
          f"(expected {expected:.4f})  ✓")

    # p=0.1: E[max] = 20 - 1/(0.1*1.9) = 20 - 100/19 ≈ 14.7368
    expected_01 = 2.0 / 0.1 - 1.0 / (0.1 * 1.9)
    assert abs(mean_max_geom(0.1) - expected_01) < 1e-10
    print(f"   p=0.1: E[max] = {mean_max_geom(0.1):.4f}  "
          f"(expected {expected_01:.4f})  ✓")

    # Verify against PMF numerical sum
    for p in [0.05, 0.1, 0.2, 0.5]:
        t_max = int(200 / p)
        pmf = pmf_max_geom(p, t_max)
        t_vals = np.arange(1, t_max + 1)
        numerical_mean = np.sum(t_vals * pmf)
        analytical_mean = mean_max_geom(p)
        rel_err = abs(numerical_mean - analytical_mean) / analytical_mean
        assert rel_err < 1e-6, \
            f"Mismatch at p={p}: numerical={numerical_mean:.6f}, " \
            f"analytical={analytical_mean:.6f}"
    print(f"   Analytical matches PMF numerical mean for 4 values  ✓")

    # ── Test mean_delivery_time ──────────────────────────────────
    print("\n2. Mean delivery time tests:")

    # With p_swap=1, same as mean_max_geom
    assert abs(mean_delivery_time(0.1, 1.0) - mean_max_geom(0.1)) < 1e-10
    print(f"   p_swap=1: E[T] = E[max]  ✓")

    # With p_swap=0.5, doubles
    assert abs(mean_delivery_time(0.1, 0.5) - 2 * mean_max_geom(0.1)) < 1e-10
    print(f"   p_swap=0.5: E[T] = 2*E[max]  ✓")

    # ── Test PMF ─────────────────────────────────────────────────
    print("\n3. PMF tests:")

    for p in [0.05, 0.1, 0.5]:
        t_max = int(100 / p)
        pmf = pmf_max_geom(p, t_max)

        # Non-negative
        assert np.all(pmf >= 0), f"Negative PMF at p={p}"

        # Sums to ~1
        coverage = np.sum(pmf)
        assert coverage > 0.9999, f"Coverage too low at p={p}: {coverage}"

        # Matches CDF
        cdf = cdf_max_geom(p, t_max)
        cdf_from_pmf = np.cumsum(pmf)
        assert np.allclose(cdf, cdf_from_pmf, atol=1e-10)

    print(f"   Non-negative, sums to 1, matches CDF  ✓")

    # pmf[0] = P(max=1) = P(T1=1)*P(T2=1) = p^2
    for p in [0.1, 0.5]:
        pmf = pmf_max_geom(p, 100)
        assert abs(pmf[0] - p * p) < 1e-10
    print(f"   pmf[0] = p^2  ✓")

    # ── Test MC validation of delivery time ──────────────────────
    print("\n4. Monte Carlo validation of E[max]:")

    rng = np.random.default_rng(42)
    for p in [0.05, 0.1, 0.5]:
        n_samples = 500_000
        t1 = rng.geometric(p, size=n_samples)
        t2 = rng.geometric(p, size=n_samples)
        max_t = np.maximum(t1, t2)
        mc_mean = np.mean(max_t)
        analytical = mean_max_geom(p)
        rel_err = abs(mc_mean - analytical) / analytical
        status = "✓" if rel_err < 0.01 else "✗"
        print(f"   p={p}: MC={mc_mean:.4f}, analytical={analytical:.4f}, "
              f"err={rel_err:.4f}  {status}")
        assert rel_err < 0.01

    # ── Test variance ────────────────────────────────────────────
    print("\n5. Variance of max(T1, T2):")

    for p in [0.1, 0.5]:
        var_analytical = var_max_geom(p)
        # MC check
        n_samples = 500_000
        t1 = rng.geometric(p, size=n_samples)
        t2 = rng.geometric(p, size=n_samples)
        max_t = np.maximum(t1, t2)
        var_mc = np.var(max_t, ddof=1)
        rel_err = abs(var_mc - var_analytical) / var_analytical
        status = "✓" if rel_err < 0.02 else "✗"
        print(f"   p={p}: Var_MC={var_mc:.2f}, "
              f"Var_analytical={var_analytical:.2f}, "
              f"err={rel_err:.4f}  {status}")
        assert rel_err < 0.02

    # ── Test mean Werner ─────────────────────────────────────────
    print("\n6. Mean Werner parameter:")

    # Infinite coherence: E[w] = w0^2
    w_inf = mean_werner_no_cutoff(0.1, 0.98, np.inf)
    assert abs(w_inf - 0.98 ** 2) < 1e-10
    print(f"   t_coh=inf: E[w] = {w_inf:.6f} "
          f"(expected {0.98**2:.6f})  ✓")

    # Finite coherence: E[w] < w0^2
    w_fin = mean_werner_no_cutoff(0.1, 0.98, 1000)
    assert w_fin < 0.98 ** 2
    assert w_fin > 0
    print(f"   t_coh=1000: E[w] = {w_fin:.6f} < {0.98**2:.6f}  ✓")

        # MC validation
    n_mc = 500_000
    t1 = rng.geometric(0.1, size=n_mc)
    t2 = rng.geometric(0.1, size=n_mc)
    age1 = np.maximum(t1, t2) - t1
    age2 = np.maximum(t1, t2) - t2
    w_mc = 0.98 ** 2 * np.exp(-(age1 + age2) / 1000.0)
    w_mc_mean = np.mean(w_mc)
    rel_err = abs(w_mc_mean - w_fin) / w_fin
    print(f"   MC check: E[w]_MC={w_mc_mean:.6f}, "
          f"analytical={w_fin:.6f}, err={rel_err:.4f}  "
          f"{'✓' if rel_err < 0.01 else '✗'}")
    assert rel_err < 0.01

    # Higher p_gen → less waiting → higher Werner
    w_fast = mean_werner_no_cutoff(0.5, 0.98, 1000)
    w_slow = mean_werner_no_cutoff(0.01, 0.98, 1000)
    assert w_fast > w_slow
    print(f"   Higher p_gen → higher E[w]: "
          f"{w_fast:.4f} > {w_slow:.4f}  ✓")

    # ── Test fidelity ────────────────────────────────────────────
    print("\n7. Mean fidelity:")

    f_inf = mean_fidelity_no_cutoff(0.1, 1.0, np.inf)
    assert abs(f_inf - 1.0) < 1e-10
    print(f"   w0=1, t_coh=inf: E[F] = {f_inf}  ✓")

    f_fin = mean_fidelity_no_cutoff(0.1, 0.98, 1000)
    w_check = mean_werner_no_cutoff(0.1, 0.98, 1000)
    assert abs(f_fin - (1 + 3 * w_check) / 4) < 1e-10
    print(f"   Consistent with E[w]: E[F] = {f_fin:.6f}  ✓")

    # ── Test SKR ─────────────────────────────────────────────────
    print("\n8. Secret key rate:")

    # High p_gen, high coherence → positive SKR
    skr_good = skr_no_cutoff(0.1, 1.0, 1000)
    assert skr_good > 0
    print(f"   p=0.1, w0=1, t_coh=1000: SKR = {skr_good:.6e}  ✓")

    # Very low coherence → zero SKR
    skr_bad = skr_no_cutoff(0.1, 1.0, 10)
    assert skr_bad == 0.0
    print(f"   p=0.1, w0=1, t_coh=10: SKR = {skr_bad}  ✓  "
          f"(fully decohered)")

    # Infinite coherence, perfect source
    skr_perf = skr_no_cutoff(0.1, 1.0, np.inf)
    expected_skr = _secret_fraction(1.0) / mean_max_geom(0.1)
    assert abs(skr_perf - expected_skr) < 1e-10
    print(f"   t_coh=inf, w0=1: SKR = {skr_perf:.6e}  "
          f"(= 1/E[max] = {expected_skr:.6e})  ✓")

    # p_swap < 1 → lower SKR
    skr_pswap = skr_no_cutoff(0.1, 1.0, 1000, p_swap=0.5)
    assert skr_pswap < skr_good
    print(f"   p_swap=0.5: SKR = {skr_pswap:.6e} < {skr_good:.6e}  ✓")

    # ── Test no_cutoff_baseline ──────────────────────────────────
    print("\n9. Complete baseline:")

    bl = no_cutoff_baseline(0.1, 0.98, 400)
    print(f"   p=0.1, w0=0.98, t_coh=400:")
    print(f"     E[T]  = {bl['mean_t']:.4f}")
    print(f"     E[w]  = {bl['mean_w']:.6f}")
    print(f"     E[F]  = {bl['mean_f']:.6f}")
    print(f"     sf    = {bl['sf']:.6f}")
    print(f"     SKR   = {bl['skr']:.6e}")
    print(f"     PMF len = {len(bl['pmf'])}")

    assert bl['mean_t'] > 0
    assert 0 < bl['mean_w'] <= 1
    assert bl['skr'] >= 0
    assert len(bl['pmf']) > 0
    print(f"   ✓ All fields valid")

    # ── Reference table ──────────────────────────────────────────
    print("\n10. Reference table (no cutoff):")
    print(f"   {'p_gen':>6} {'t_coh':>6} {'E[T]':>10} {'E[w]':>8} "
          f"{'E[F]':>8} {'SKR':>12}")
    print(f"   {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*12}")

    for p in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for tc in [100, 400, 1000]:
            bl = no_cutoff_baseline(p, 0.98, tc)
            print(f"   {p:>6.2f} {tc:>6} {bl['mean_t']:>10.2f} "
                  f"{bl['mean_w']:>8.4f} {bl['mean_f']:>8.4f} "
                  f"{bl['skr']:>12.4e}")

    # ── Why cutoff helps ─────────────────────────────────────────
    print("\n11. Why cutoff helps (p=0.01, w0=0.98, t_coh=400):")

    bl_slow = no_cutoff_baseline(0.01, 0.98, 400)
    print(f"   No cutoff:")
    print(f"     E[T] = {bl_slow['mean_t']:.2f}")
    print(f"     E[w] = {bl_slow['mean_w']:.6f}")
    print(f"     E[F] = {bl_slow['mean_f']:.6f}")
    print(f"     SKR  = {bl_slow['skr']:.6e}")

    if bl_slow['skr'] == 0:
        print(f"     → SKR is ZERO! Pairs decohere before delivery.")
        print(f"     → Cutoff can help by discarding old pairs and retrying.")
    else:
        print(f"     → SKR is positive but low.")
        print(f"     → Cutoff can improve by trading time for quality.")

    print("\n✅ no_cutoff.py self-test passed")