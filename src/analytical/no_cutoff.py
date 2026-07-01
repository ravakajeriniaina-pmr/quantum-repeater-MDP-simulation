import numpy as np
from src.physical.werner_utils import secret_fraction_nat as _secret_fraction


# ═════════════════════════════════════════════════════════════════════
# DELIVERY TIME (no cutoff)
# ═════════════════════════════════════════════════════════════════════

def mean_max_geom(p_gen: float) -> float:
    """
    E[max(T1,T2)] where T1,T2 ~ Geometric(p_gen) on {1,2,...}.
    """
    p = p_gen
    return 2.0 / p - 1.0 / (p * (2.0 - p))


def mean_delivery_time(p_gen: float, p_swap: float = 1.0) -> float:
    """
    Mean delivery time with swap success probability p_swap.
    """
    if not (0.0 < p_swap <= 1.0):
        raise ValueError(f"p_swap must be in (0,1], got {p_swap}")
    return mean_max_geom(p_gen) / p_swap


def pmf_max_geom(p_gen: float, t_max: int) -> np.ndarray:
    """
    PMF of M=max(T1,T2), for T1,T2 i.i.d geometric(p_gen), support t>=1.
    Returned array has length t_max, index i corresponds to t=i+1.
    """
    p = p_gen
    q = 1.0 - p
    t_vals = np.arange(1, t_max + 1)
    cdf = 1.0 - q ** t_vals
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])
    return cdf**2 - cdf_prev**2


def cdf_max_geom(p_gen: float, t_max: int) -> np.ndarray:
    p = p_gen
    q = 1.0 - p
    t_vals = np.arange(1, t_max + 1)
    return (1.0 - q ** t_vals) ** 2


def _default_tmax(p_gen: float, t_coh: float) -> int:
    """
    Conservative truncation for accurate E[w] and E[sf(w)] sums.
    """
    if np.isinf(t_coh):
        return int(max(2000 / p_gen, 20000))
    return int(max(2000 / p_gen, 100 * t_coh, 50000))


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
    pmf = p * q ** (t_vals - 1)

    # CORRECTION : facteur 2 dans l'exposant
    # age1 = max(t1,t2) - t1,  age2 = max(t1,t2) - t2
    # max(age1, age2) = |t1 - t2|
    # w_out = w0² * exp(-2 * |t1-t2| / t_coh)
    diff = np.abs(t_vals[:, None] - t_vals[None, :])
    decay = np.exp(-2.0 * diff / t_coh)   # ← ajouter le facteur 2

    joint = pmf[:, None] * pmf[None, :]

    return float(w0 * w0 * np.sum(joint * decay))

def mean_fidelity_no_cutoff(p_gen: float, w0: float,
                            t_coh: float, t_max: int = None) -> float:
    mean_w = mean_werner_no_cutoff(p_gen, w0, t_coh, t_max)
    return (1.0 + 3.0 * mean_w) / 4.0


# ═════════════════════════════════════════════════════════════════════
# SECRET FRACTION + SKR (no cutoff)
# ═════════════════════════════════════════════════════════════════════

def mean_secret_fraction_no_cutoff(p_gen: float, w0: float,
                                   t_coh: float, t_max: int = None) -> float:
    if np.isinf(t_coh):
        return float(_secret_fraction(w0 * w0))

    if t_max is None:
        t_max = int(50 / p_gen)

    p = p_gen
    q = 1.0 - p

    t_vals = np.arange(1, t_max + 1)
    pmf = p * q ** (t_vals - 1)

    diff = np.abs(t_vals[:, None] - t_vals[None, :])
    w_mat = w0 * w0 * np.exp(-2.0 * diff / t_coh)  # même convention
    sf_mat = np.vectorize(_secret_fraction)(w_mat)

    joint = pmf[:, None] * pmf[None, :]
    return float(np.sum(joint * sf_mat))

def skr_no_cutoff(p_gen: float, w0: float, t_coh: float,
                  p_swap: float = 1.0, t_max: int = None) -> float:
    """
    SKR = E[sf(w_out)] / E[T_delivery]
    """
    mean_t = mean_delivery_time(p_gen, p_swap)
    if mean_t <= 0 or np.isinf(mean_t):
        return 0.0

    mean_sf = mean_secret_fraction_no_cutoff(p_gen, w0, t_coh, t_max)
    return max(0.0, float(mean_sf / mean_t))


# ═════════════════════════════════════════════════════════════════════
# COMPLETE ANALYTICAL BASELINE
# ═════════════════════════════════════════════════════════════════════

def no_cutoff_baseline(p_gen: float, w0: float, t_coh: float,
                       p_swap: float = 1.0,
                       t_max: int = None) -> dict:
    if t_max is None:
        t_max = _default_tmax(p_gen, t_coh)

    mean_t = mean_delivery_time(p_gen, p_swap)
    mean_w = mean_werner_no_cutoff(p_gen, w0, t_coh, t_max)
    mean_f = (1.0 + 3.0 * mean_w) / 4.0
    mean_sf = mean_secret_fraction_no_cutoff(p_gen, w0, t_coh, t_max)
    skr = mean_sf / mean_t if mean_t > 0 else 0.0
    skr = max(0.0, skr)
    pmf = pmf_max_geom(p_gen, t_max)

    return {
        "mean_t": float(mean_t),
        "mean_w": float(mean_w),
        "mean_f": float(mean_f),
        "sf": float(mean_sf),
        "skr": float(skr),
        "pmf": pmf,
    }


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) E[max] closed form sanity
    assert abs(mean_max_geom(1.0) - 1.0) < 1e-12
    assert abs(mean_max_geom(0.5) - (8.0 / 3.0)) < 1e-12

    # 2) mean_delivery_time with p_swap
    m = mean_max_geom(0.1)
    assert abs(mean_delivery_time(0.1, 1.0) - m) < 1e-12
    assert abs(mean_delivery_time(0.1, 0.5) - 2.0 * m) < 1e-12

    # 3) PMF sanity
    pmf = pmf_max_geom(0.1, 5000)
    assert np.all(pmf >= 0)
    assert np.sum(pmf) > 0.999

    # 4) MC consistency for E[w]
    p_gen, w0, t_coh = 0.1, 0.98, 100.0
    n = 400_000
    t1 = rng.geometric(p_gen, size=n)
    t2 = rng.geometric(p_gen, size=n)
    mmax = np.maximum(t1, t2)

    age_eff = np.maximum(mmax - 1, 0)
    w_mc = w0 * w0 * np.exp(-age_eff / t_coh)
    w_mc_mean = float(np.mean(w_mc))

    w_an = mean_werner_no_cutoff(p_gen, w0, t_coh)
    rel_w = abs(w_mc_mean - w_an) / max(w_an, 1e-12)
    assert rel_w < 0.02, (w_mc_mean, w_an, rel_w)

    # 5) SKR sanity
    skr1 = skr_no_cutoff(0.1, 1.0, 1000.0, p_swap=1.0)
    skr2 = skr_no_cutoff(0.1, 1.0, 1000.0, p_swap=0.5)
    assert skr2 < skr1

    print("✅ no_cutoff.py self-test passed")