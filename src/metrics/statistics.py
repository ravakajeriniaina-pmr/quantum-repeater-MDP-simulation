import numpy as np
from scipy import stats


# ═════════════════════════════════════════════════════════════════════
# CONFIDENCE INTERVALS
# ═════════════════════════════════════════════════════════════════════

def confidence_interval(data: np.ndarray,
                        confidence: float = 0.95) -> tuple:
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # z-score for the given confidence level
    # For 95%: z = 1.96, for 99%: z = 2.576
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    half_width = z * std / np.sqrt(n)
    lower = mean - half_width
    upper = mean + half_width

    return (mean, lower, upper, half_width)


def confidence_interval_99(data: np.ndarray) -> tuple:
    """Shorthand for 99% confidence interval."""
    return confidence_interval(data, confidence=0.99)


def confidence_interval_999(data: np.ndarray) -> tuple:
    """Shorthand for 99.9% confidence interval."""
    return confidence_interval(data, confidence=0.999)


# ═════════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTING
# ═════════════════════════════════════════════════════════════════════

def welch_t_test(data1: np.ndarray, data2: np.ndarray) -> dict:
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_001": p_value < 0.001,
        "significant_01": p_value < 0.01,
        "significant_05": p_value < 0.05,
        "mean_diff": float(np.mean(data2) - np.mean(data1)),
        "mean1": float(np.mean(data1)),
        "mean2": float(np.mean(data2)),
    }


def paired_t_test(data1: np.ndarray, data2: np.ndarray) -> dict:
    assert len(data1) == len(data2), \
        "Paired test requires equal-length arrays (same runs)"

    diff = data2 - data1
    t_stat, p_value = stats.ttest_1samp(diff, 0.0)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_001": p_value < 0.001,
        "significant_01": p_value < 0.01,
        "significant_05": p_value < 0.05,
        "mean_diff": float(np.mean(diff)),
        "mean1": float(np.mean(data1)),
        "mean2": float(np.mean(data2)),
        "mean_paired_diff": float(np.mean(diff)),
        "std_paired_diff": float(np.std(diff, ddof=1)),
    }


# ═════════════════════════════════════════════════════════════════════
# RELATIVE IMPROVEMENT
# ═════════════════════════════════════════════════════════════════════

def relative_improvement(baseline: float, improved: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (improved - baseline) / baseline * 100.0


def relative_error(reference: float, measured: float) -> float:
    if reference == 0.0:
        return 0.0
    return abs(measured - reference) / abs(reference)


# ═════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════

def summary_stats(data: np.ndarray, name: str = "data") -> dict:
    mean, ci_lo, ci_hi, ci_hw = confidence_interval(data, 0.95)

    return {
        "name": name,
        "n": len(data),
        "mean": float(mean),
        "std": float(np.std(data, ddof=1)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "ci95_lower": float(ci_lo),
        "ci95_upper": float(ci_hi),
        "ci95_hw": float(ci_hw),
    }


def print_summary(stats_dict: dict) -> None:
    s = stats_dict
    print(f"  {s['name']}:")
    print(f"    N       = {s['n']:,}")
    print(f"    Mean    = {s['mean']:.6f} ± {s['ci95_hw']:.6f} (95% CI)")
    print(f"    Std     = {s['std']:.6f}")
    print(f"    Median  = {s['median']:.6f}")
    print(f"    Min     = {s['min']:.6f}")
    print(f"    Max     = {s['max']:.6f}")
    print(f"    Q25     = {s['q25']:.6f}")
    print(f"    Q75     = {s['q75']:.6f}")
    print(f"    95% CI  = [{s['ci95_lower']:.6f}, {s['ci95_upper']:.6f}]")


def print_comparison(name1: str, data1: np.ndarray,
                     name2: str, data2: np.ndarray,
                     paired: bool = False) -> dict:
    s1 = summary_stats(data1, name1)
    s2 = summary_stats(data2, name2)

    print(f"\n{'='*60}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*60}")

    print_summary(s1)
    print()
    print_summary(s2)

    # Improvement
    imp = relative_improvement(s1["mean"], s2["mean"])
    print(f"\n  Relative improvement: {imp:+.2f}%")

    # Statistical test
    if paired:
        test = paired_t_test(data1, data2)
        test_name = "Paired t-test"
    else:
        test = welch_t_test(data1, data2)
        test_name = "Welch's t-test"

    print(f"\n  {test_name}:")
    print(f"    t-statistic = {test['t_statistic']:.4f}")
    print(f"    p-value     = {test['p_value']:.2e}")
    print(f"    Significant at α=0.05:  {test['significant_05']}")
    print(f"    Significant at α=0.01:  {test['significant_01']}")
    print(f"    Significant at α=0.001: {test['significant_001']}")
    print(f"{'='*60}")

    return test


# ═════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═════════════════════════════════════════════════════════════════════

def validate_mc_against_analytical(mc_value: float, analytical_value: float,
                                   mc_ci_hw: float,
                                   tolerance: float = 0.02,
                                   label: str = "") -> bool:
    rel_err = relative_error(analytical_value, mc_value)
    within_ci = abs(mc_value - analytical_value) <= 2 * mc_ci_hw
    passes = rel_err < tolerance

    status = "✓ PASS" if passes else "✗ FAIL"
    print(f"  {label}")
    print(f"    Analytical: {analytical_value:.6e}")
    print(f"    MC:         {mc_value:.6e} ± {mc_ci_hw:.6e}")
    print(f"    Rel error:  {rel_err:.4f} ({rel_err*100:.2f}%)")
    print(f"    Within CI:  {within_ci}")
    print(f"    {status} (tolerance = {tolerance*100:.1f}%)")

    return passes


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Statistics Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ── Test confidence interval ───────────────────────────────��─
    print("\n1. Confidence interval tests:")

    # Known distribution: N(10, 2)
    data = rng.normal(10.0, 2.0, size=100_000)
    mean, lo, hi, hw = confidence_interval(data, 0.95)

    assert abs(mean - 10.0) < 0.05
    assert lo < mean < hi
    assert lo < 10.0 < hi
    print(f"   N(10,2), n=100000:")
    print(f"   Mean = {mean:.4f}, CI = [{lo:.4f}, {hi:.4f}], "
          f"hw = {hw:.4f}  ✓")

    # Constant data → zero width
    const_data = np.ones(1000) * 5.0
    mean_c, lo_c, hi_c, hw_c = confidence_interval(const_data)
    assert mean_c == 5.0
    assert hw_c == 0.0
    assert lo_c == hi_c == 5.0
    print(f"   Constant data: hw = {hw_c}  ✓  (zero width)")

    # 99% CI should be wider than 95%
    _, _, _, hw_95 = confidence_interval(data, 0.95)
    _, _, _, hw_99 = confidence_interval(data, 0.99)
    assert hw_99 > hw_95
    print(f"   99% CI ({hw_99:.4f}) > 95% CI ({hw_95:.4f})  ✓")

    # Larger sample → narrower CI
    small = rng.normal(0, 1, size=100)
    large = rng.normal(0, 1, size=10000)
    _, _, _, hw_small = confidence_interval(small)
    _, _, _, hw_large = confidence_interval(large)
    assert hw_small > hw_large
    print(f"   n=100 hw ({hw_small:.4f}) > n=10000 hw ({hw_large:.4f})  ✓")

    # ── Test Welch's t-test ──────────────────────────────────────
    print("\n2. Welch's t-test:")

    # Same distribution → not significant
    d1 = rng.normal(10.0, 1.0, size=10000)
    d2 = rng.normal(10.0, 1.0, size=10000)
    result_same = welch_t_test(d1, d2)
    assert not result_same["significant_05"], \
        "Same distributions should not be significant"
    print(f"   Same distributions: p = {result_same['p_value']:.4f}, "
          f"sig_05 = {result_same['significant_05']}  ✓")

    # Different means → significant
    d3 = rng.normal(10.0, 1.0, size=10000)
    d4 = rng.normal(10.5, 1.0, size=10000)
    result_diff = welch_t_test(d3, d4)
    assert result_diff["significant_001"], \
        "Different means should be highly significant"
    assert result_diff["mean_diff"] > 0
    print(f"   Different means (Δ=0.5): p = {result_diff['p_value']:.2e}, "
          f"sig_001 = {result_diff['significant_001']}  ✓")

    # ── Test paired t-test ───────────────────────────────────────
    print("\n3. Paired t-test (CRN simulation):")

    # Simulate CRN: shared base + small independent noise
    base = rng.normal(100.0, 50.0, size=10000)
    noise1 = rng.normal(0, 0.5, size=10000)
    noise2 = rng.normal(0, 0.5, size=10000)
    crn_d1 = base + noise1
    crn_d2 = base + 0.3 + noise2  # small systematic improvement

    # Welch test (ignores pairing) — may or may not detect
    welch_result = welch_t_test(crn_d1, crn_d2)

    # Paired test (exploits CRN) — should detect easily
    paired_result = paired_t_test(crn_d1, crn_d2)

    print(f"   Shared variance = 50.0, true diff = 0.3")
    print(f"   Welch:  p = {welch_result['p_value']:.4e}, "
          f"sig_001 = {welch_result['significant_001']}")
    print(f"   Paired: p = {paired_result['p_value']:.4e}, "
          f"sig_001 = {paired_result['significant_001']}")

    # Paired test should have smaller p-value
    assert paired_result["p_value"] <= welch_result["p_value"], \
        "Paired test should be more powerful with CRN data"
    print(f"   Paired p-value <= Welch p-value  ✓")
    print(f"   Paired std of diff = {paired_result['std_paired_diff']:.4f}  "
          f"(vs raw std ≈ 50)")

    # ── Test relative improvement ────────────────────────────────
    print("\n4. Relative improvement tests:")

    assert relative_improvement(100, 120) == 20.0
    print(f"   100 → 120: {relative_improvement(100, 120):+.1f}%  ✓")

    assert relative_improvement(100, 80) == -20.0
    print(f"   100 → 80:  {relative_improvement(100, 80):+.1f}%  ✓")

    assert relative_improvement(100, 100) == 0.0
    print(f"   100 → 100: {relative_improvement(100, 100):+.1f}%  ✓")

    assert relative_improvement(0, 50) == 0.0
    print(f"   0 → 50:    {relative_improvement(0, 50):.1f}%  ✓  "
          f"(zero baseline)")

    # ── Test relative error ──────────────────────────────────────
    print("\n5. Relative error tests:")

    assert abs(relative_error(1.0, 1.02) - 0.02) < 1e-10
    print(f"   ref=1.0, meas=1.02: err={relative_error(1.0, 1.02):.6f}  ✓")

    assert abs(relative_error(1.0, 0.98) - 0.02) < 1e-10
    print(f"   ref=1.0, meas=0.98: err={relative_error(1.0, 0.98):.6f}  ✓")

    assert relative_error(0.0, 0.001) == 0.0
    print(f"   ref=0.0, meas=0.001: err={relative_error(0.0, 0.001)}  ✓  "
          f"(zero reference)")

    assert abs(relative_error(0.005, 0.00495) - 0.01) < 1e-10
    print(f"   ref=0.005, meas=0.00495: "
          f"err={relative_error(0.005, 0.00495):.6f}  ✓")

    # ── Test summary stats ───────────────────────────────────────
    print("\n6. Summary statistics:")

    test_data = rng.exponential(100.0, size=50000)
    s = summary_stats(test_data, "Exponential(λ=100)")
    print_summary(s)

    assert s["n"] == 50000
    assert abs(s["mean"] - 100.0) < 2.0  # within 2 of true mean
    assert s["min"] >= 0.0  # exponential is non-negative
    assert s["q25"] < s["median"] < s["q75"]
    print(f"   All checks passed  ✓")

    # ── Test validation helper ───────────────────────────────────
    print("\n7. MC validation helper:")

    # Good match
    print()
    pass1 = validate_mc_against_analytical(
        mc_value=1.234e-4,
        analytical_value=1.230e-4,
        mc_ci_hw=0.008e-4,
        tolerance=0.02,
        label="Test 1: Good match"
    )
    assert pass1

    # Bad match
    print()
    pass2 = validate_mc_against_analytical(
        mc_value=1.300e-4,
        analytical_value=1.230e-4,
        mc_ci_hw=0.008e-4,
        tolerance=0.02,
        label="Test 2: Bad match"
    )
    assert not pass2

    # ── Test print_comparison ────────────────────────────────────
    print("\n8. Full comparison demo:")

    strategy_a = rng.exponential(100.0, size=10000)
    strategy_b = rng.exponential(90.0, size=10000)  # faster delivery

    test_result = print_comparison(
        "FIXED", strategy_a,
        "ADAPTIVE", strategy_b,
        paired=False
    )

    # ── Test paired comparison ───────────────────────────────────
    print("\n9. Paired comparison demo (CRN):")

    base_times = rng.exponential(100.0, size=10000)
    fixed_times = base_times + rng.normal(0, 5, size=10000)
    adaptive_times = base_times * 0.95 + rng.normal(0, 5, size=10000)

    test_paired = print_comparison(
        "FIXED", fixed_times,
        "ADAPTIVE", adaptive_times,
        paired=True
    )

    print("\n✅ statistics.py self-test passed") 