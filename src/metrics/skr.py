"""
Secret Key Rate (SKR) from Monte Carlo samples.
"""

import numpy as np
from src.physical.werner_utils import (
    secret_fraction_nat as _secret_fraction,
    binary_entropy_nat as _binary_entropy_nat,
)
from src.physical.werner_utils import secret_fraction_nat as _secret_fraction_nat

def secret_fraction_array(w_array: np.ndarray) -> np.ndarray:
    """Vectorized secret fraction. Delegates to werner_utils for consistency."""
    return np.vectorize(_secret_fraction_nat)(w_array)

def skr_from_samples(delivery_times: np.ndarray, w_out_array: np.ndarray) -> float:
    if len(delivery_times) == 0:
        return 0.0
    mean_t = np.mean(delivery_times)
    if mean_t <= 0 or np.isinf(mean_t):
        return 0.0
    mean_sf = np.mean(secret_fraction_array(w_out_array))
    return max(0.0, float(mean_sf / mean_t))


def skr_from_rewards(total_reward: float, total_time: int) -> float:
    if total_time <= 0:
        return 0.0
    return max(0.0, total_reward / total_time)


def _empty_skr_result() -> dict:
    return {
        "skr": 0.0,
        "skr_lower": 0.0,
        "skr_upper": 0.0,
        "skr_hw": 0.0,
        "mean_t": 0.0,
        "mean_sf": 0.0,
        "mean_w": 0.0,
        "mean_f": 0.0,
        "n_episodes": 0,
    }


def skr_with_ci(delivery_times: np.ndarray,
                w_out_array: np.ndarray,
                confidence: float = 0.95) -> dict:
    from scipy import stats

    n = len(delivery_times)
    if n == 0:
        return _empty_skr_result()

    sf_array = secret_fraction_array(w_out_array)
    mean_t = np.mean(delivery_times)
    mean_sf = np.mean(sf_array)
    mean_w = np.mean(w_out_array)
    mean_f = (1.0 + 3.0 * mean_w) / 4.0

    if mean_t <= 0:
        return _empty_skr_result()

    skr = mean_sf / mean_t

    var_sf = np.var(sf_array, ddof=1)
    var_t = np.var(delivery_times, ddof=1)
    cov_sf_t = np.cov(sf_array, delivery_times, ddof=1)[0, 1]

    var_skr = (
        (1.0 / mean_t) ** 2 * var_sf / n
        + (mean_sf / mean_t ** 2) ** 2 * var_t / n
        - 2.0 * (mean_sf / mean_t ** 3) * cov_sf_t / n
    )
    var_skr = max(0.0, var_skr)
    std_skr = np.sqrt(var_skr)

    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    hw = z * std_skr

    return {
        "skr": float(skr),
        "skr_lower": float(max(0.0, skr - hw)),
        "skr_upper": float(skr + hw),
        "skr_hw": float(hw),
        "mean_t": float(mean_t),
        "mean_sf": float(mean_sf),
        "mean_w": float(mean_w),
        "mean_f": float(mean_f),
        "n_episodes": n,
    }


def skr_improvement(skr_baseline: float, skr_improved: float) -> float:
    if skr_baseline <= 0:
        return 0.0
    return (skr_improved - skr_baseline) / skr_baseline * 100.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    t = rng.geometric(0.1, size=10000).astype(float)
    w = np.clip(np.exp(-t / 200.0), 0, 1)
    r = skr_with_ci(t, w)
    assert r["skr"] >= 0
    sf = secret_fraction_array(np.array([0.0, 0.5, 1.0]))
    assert sf[0] == 0 and sf[1] == 0 and abs(sf[2] - 1.0) < 1e-12
    print("✅ skr.py self-test passed")