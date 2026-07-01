import os
import sys
import warnings
import logging
from copy import deepcopy

import numpy as np


# ═════════════════════════════════════════════════════════════════════
# SETUP: Add Boxi Li's code to the import path
# ═════════════════════════════════════════════════════════════════════

def _find_boxili_path() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "external", "boxili"),
        os.path.join(os.path.dirname(__file__), "..", "external", "boxili"),
        os.path.join(os.getcwd(), "external", "boxili"),
        os.environ.get("BOXILI_PATH", ""),
    ]

    for path in candidates:
        if path and os.path.isdir(path):
            abs_path = os.path.abspath(path)
            # Verify key files exist
            if os.path.isfile(os.path.join(abs_path, "repeater_algorithm.py")):
                return abs_path

    raise FileNotFoundError(
        "Cannot find Boxi Li's code. Please clone it:\n"
        "  git clone https://github.com/BoxiLi/repeater-cut-off-optimization.git "
        "external/boxili\n"
        "Or set BOXILI_PATH environment variable."
    )


def _patch_numpy_compat():
    """
    Patch NumPy compatibility issues in Boxi Li's code.

    His code uses np.iinfo(np.int).max which was deprecated in
    NumPy 1.20 and removed in NumPy 1.24. We restore np.int
    as an alias for Python int.
    """
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'complex'):
        np.complex = complex
    if not hasattr(np, 'bool'):
        np.bool = bool


# Apply patches before importing
_patch_numpy_compat()

# Track whether imports succeeded
_boxili_available = False
_import_error = None

try:
    _boxili_path = _find_boxili_path()
    if _boxili_path not in sys.path:
        sys.path.insert(0, _boxili_path)

    # Suppress numba and matplotlib warnings during import
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Suppress Boxi Li's logging setup
        logging.disable(logging.CRITICAL)

        from external.boxili.repeater_algorithm import repeater_sim as _repeater_sim
        from external.boxili.utility_functions import (
            secret_key_rate as _secret_key_rate,
            get_mean_werner as _get_mean_werner,
            get_mean_waiting_time as _get_mean_waiting_time,
            secret_fraction as _secret_fraction,
            werner_to_fid as _werner_to_fid,
        )
        logging.disable(logging.NOTSET)

    _boxili_available = True

except (FileNotFoundError, ImportError, ModuleNotFoundError) as e:
    _import_error = str(e)
    _boxili_available = False


def is_available() -> bool:
    """Check if Boxi Li's code is available."""
    return _boxili_available


def require_available():
    """Raise ImportError if Boxi Li's code is not available."""
    if not _boxili_available:
        raise ImportError(
            f"Boxi Li's code is not available: {_import_error}\n"
            "Please clone it:\n"
            "  git clone https://github.com/BoxiLi/"
            "repeater-cut-off-optimization.git external/boxili"
        )


# ═════════════════════════════════════════════════════════════════════
# PARAMETER CONVERSION
# ═════════════════════════════════════════════════════════════════════

def _make_params(p_gen: float, w0: float, p_swap: float, t_coh: float,
                 protocol: tuple, t_trunc: int, cutoff,
                 cut_type: str = "memory_time") -> dict:
    params = {
        "protocol": tuple(protocol),
        "p_gen": p_gen,
        "p_swap": p_swap,
        "w0": w0,
        "t_coh": t_coh,
        "t_trunc": t_trunc,
        "cut_type": cut_type,
    }

    # Handle cutoff: can be int (uniform) or tuple (per-level)
    n_levels = len(protocol)
    if isinstance(cutoff, (int, float)):
        cutoff_int = int(cutoff)
        params["mt_cut"] = (cutoff_int,) * n_levels
    else:
        params["mt_cut"] = tuple(int(c) for c in cutoff)

    return params


def _make_params_from_config(config, cutoff) -> dict:
    return _make_params(
        p_gen=config.p_gen,
        w0=config.w0,
        p_swap=config.p_swap,
        t_coh=config.t_coh,
        protocol=tuple(config.protocol),
        t_trunc=config.t_trunc,
        cutoff=cutoff,
    )


# ═════════════════════════════════════════════════════════════════════
# CORE WRAPPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def compute_pmf_werner(p_gen: float, w0: float, p_swap: float,
                       t_coh: float, protocol: tuple, t_trunc: int,
                       cutoff) -> tuple:
    require_available()
    params = _make_params(p_gen, w0, p_swap, t_coh, protocol, t_trunc, cutoff)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pmf, w_func = _repeater_sim(params)

    return pmf, w_func


def compute_analytical_baseline(p_gen: float, w0: float, p_swap: float,
                                t_coh: float, protocol: tuple,
                                t_trunc: int, cutoff) -> dict:
    require_available()
    pmf, w_func = compute_pmf_werner(
        p_gen, w0, p_swap, t_coh, protocol, t_trunc, cutoff)

    # Use Boxi Li's exact functions for consistency
    skr = _secret_key_rate(pmf, w_func)
    mean_w = _get_mean_werner(pmf, w_func)
    mean_t = _get_mean_waiting_time(pmf)
    mean_f = _werner_to_fid(mean_w)
    coverage = float(np.sum(pmf))

    return {
        "pmf": pmf,
        "w_func": w_func,
        "skr": float(skr),
        "mean_t": float(mean_t),
        "mean_w": float(mean_w),
        "mean_f": float(mean_f),
        "coverage": coverage,
        "cutoff": cutoff,
    }


def compute_analytical_from_config(config, cutoff) -> dict:
    return compute_analytical_baseline(
        p_gen=config.p_gen,
        w0=config.w0,
        p_swap=config.p_swap,
        t_coh=config.t_coh,
        protocol=tuple(config.protocol),
        t_trunc=config.t_trunc,
        cutoff=cutoff,
    )


# ═════════════════════════════════════════════════════════════════════
# CUTOFF OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════

def optimize_fixed_cutoff(p_gen: float, w0: float, p_swap: float,
                          t_coh: float, protocol: tuple, t_trunc: int,
                          cutoff_range: range = None,
                          verbose: bool = False) -> tuple:
    require_available()

    if cutoff_range is None:
        max_cut = min(int(5 * t_coh), t_trunc - 1, 10000)
        cutoff_range = range(1, max(max_cut, 2))

    best_skr = 0.0
    n_star = 1
    best_results = None

    for n in cutoff_range:
        try:
            result = compute_analytical_baseline(
                p_gen, w0, p_swap, t_coh, protocol, t_trunc, cutoff=n)
            skr = result["skr"]

            if verbose and n % 50 == 0:
                print(f"  cutoff={n}: SKR={skr:.6e}")

            if skr > best_skr:
                best_skr = skr
                n_star = n
                best_results = result

        except Exception as e:
            if verbose:
                print(f"  cutoff={n}: ERROR - {e}")
            continue

    if best_results is None:
        # No valid cutoff found — run with cutoff=1 to return something
        best_results = compute_analytical_baseline(
            p_gen, w0, p_swap, t_coh, protocol, t_trunc, cutoff=1)
        n_star = 1
        best_skr = best_results["skr"]

    if verbose:
        print(f"\n  Optimal cutoff: n* = {n_star}")
        print(f"  Best SKR: {best_skr:.6e}")
        print(f"  Mean delivery time: {best_results['mean_t']:.2f}")
        print(f"  Mean fidelity: {best_results['mean_f']:.6f}")

    return n_star, best_skr, best_results


def optimize_fixed_cutoff_from_config(config, cutoff_range: range = None,
                                      verbose: bool = False) -> tuple:
    return optimize_fixed_cutoff(
        p_gen=config.p_gen,
        w0=config.w0,
        p_swap=config.p_swap,
        t_coh=config.t_coh,
        protocol=tuple(config.protocol),
        t_trunc=config.t_trunc,
        cutoff_range=cutoff_range,
        verbose=verbose,
    )


# ═════════════════════════════════════════════════════════════════════
# SECRET FRACTION (direct access to Boxi Li's function)
# ═════════════════════════════════════════════════════════════════════

def boxili_secret_fraction(w: float) -> float:
    require_available()
    return _secret_fraction(w)


def boxili_secret_key_rate(pmf: np.ndarray, w_func: np.ndarray) -> float:
    require_available()
    return _secret_key_rate(pmf, w_func)


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Boxi Li Wrapper Self-Test")
    print("=" * 60)

    # ── Check availability ───────────────────────────────────────
    print(f"\n1. Boxi Li code available: {is_available()}")

    if not is_available():
        print(f"   Error: {_import_error}")
        print("\n   To fix, run:")
        print("   git clone https://github.com/BoxiLi/"
              "repeater-cut-off-optimization.git external/boxili")
        print("\n⚠️  Cannot run tests without Boxi Li's code.")
        sys.exit(1)

    print(f"   Path: {_boxili_path}")

    # ── Test basic PMF computation ───────────────────────────────
    print("\n2. Basic PMF computation (1-level swap, no cutoff):")

    # Use a very large cutoff = effectively no cutoff
    pmf, w_func = compute_pmf_werner(
        p_gen=0.1, w0=0.98, p_swap=0.5,
        t_coh=400, protocol=(0,),
        t_trunc=3000, cutoff=10_000_000
    )

    coverage = np.sum(pmf)
    print(f"   PMF shape: {pmf.shape}")
    print(f"   W_func shape: {w_func.shape}")
    print(f"   Coverage: {coverage:.6f}")
    print(f"   pmf[0] = {pmf[0]:.6e}")
    print(f"   pmf[1] = {pmf[1]:.6e}")
    print(f"   pmf[2] = {pmf[2]:.6e}")

    # pmf[0] should be ~0 (cannot deliver at t=0)
    # pmf[1] = p_gen^2 * p_swap = 0.01 * 0.5 = 0.005 (both gen at t=1, swap succeeds)
    assert pmf[0] < 1e-10, f"pmf[0] should be ~0, got {pmf[0]}"
    assert abs(pmf[1] - 0.005) < 1e-6, f"pmf[1] should be ~0.005, got {pmf[1]}"
    assert coverage > 0.99, f"Coverage too low: {coverage}"
    print(f"   ✓ Basic computation works")
    
    # ── Test compute_analytical_baseline ─────────────────────────
    print("\n3. Analytical baseline with cutoff:")

    result = compute_analytical_baseline(
        p_gen=0.1, w0=0.98, p_swap=0.5,
        t_coh=400, protocol=(0,),
        t_trunc=3000, cutoff=50
    )

    print(f"   SKR:       {result['skr']:.6e}")
    print(f"   Mean T:    {result['mean_t']:.2f}")
    print(f"   Mean W:    {result['mean_w']:.6f}")
    print(f"   Mean F:    {result['mean_f']:.6f}")
    print(f"   Coverage:  {result['coverage']:.6f}")

    assert result['skr'] > 0, "SKR should be positive"
    assert result['mean_t'] > 0, "Mean T should be positive"
    assert 0 < result['mean_w'] <= 1, "Mean W should be in (0, 1]"
    assert result['coverage'] > 0.99, "Coverage should be > 0.99"
    print(f"   ✓ Analytical baseline works")

    # ── Test cutoff optimization ─────────────────────────────────
    print("\n4. Cutoff optimization (1-level swap):")

    n_star, best_skr, best_result = optimize_fixed_cutoff(
        p_gen=0.1, w0=0.98, p_swap=0.5,
        t_coh=400, protocol=(0,),
        t_trunc=3000,
        cutoff_range=range(1, 200),
        verbose=False
    )

    print(f"   Optimal cutoff: n* = {n_star}")
    print(f"   Best SKR: {best_skr:.6e}")
    print(f"   Mean T at n*: {best_result['mean_t']:.2f}")
    print(f"   Mean F at n*: {best_result['mean_f']:.6f}")

    assert n_star > 0, "Optimal cutoff should be positive"
    assert best_skr > 0, "Best SKR should be positive"
    print(f"   ✓ Cutoff optimization works")

    # ── Test Boxi Li's examples.py parameters ────────────────────
    print("\n5. Boxi Li paper example (3-level swap):")

    result_paper = compute_analytical_baseline(
        p_gen=0.1, w0=0.98, p_swap=0.5,
        t_coh=400, protocol=(0, 0, 0),
        t_trunc=3000, cutoff=(16, 31, 55)
    )

    print(f"   SKR:      {result_paper['skr']:.6e}")
    print(f"   Mean T:   {result_paper['mean_t']:.2f}")
    print(f"   Mean F:   {result_paper['mean_f']:.6f}")
    print(f"   Coverage: {result_paper['coverage']:.6f}")

    assert result_paper['skr'] > 0, "Paper example SKR should be positive"
    print(f"   ✓ Paper example works")

    # ── Test secret fraction consistency ─────────────────────────
    print("\n6. Secret fraction consistency:")

    # Import our own version for comparison
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from physical.werner_utils import secret_fraction_nat

        for w in [0.8, 0.9, 0.95, 1.0]:
            sf_boxili = boxili_secret_fraction(w)
            sf_ours = secret_fraction_nat(w)
            match = abs(sf_boxili - sf_ours) < 1e-10
            print(f"   w={w}: Boxi Li={sf_boxili:.6f}, "
                  f"ours={sf_ours:.6f}  {'✓' if match else '✗'}")
            assert match, f"Secret fraction mismatch at w={w}"
    except ImportError:
        print("   (skipping — werner_utils not importable from here)")

    # ── Test effect of cutoff on SKR ─────────────────────────────
    print("\n7. Cutoff scan (SKR vs cutoff, 1-level swap):")
    print(f"   {'cutoff':>8} {'SKR':>12} {'Mean T':>10} {'Mean F':>10}")
    print(f"   {'─'*8} {'─'*12} {'─'*10} {'─'*10}")

    for n in [5, 10, 20, 50, 100, 200, 500]:
        r = compute_analytical_baseline(
            p_gen=0.1, w0=0.98, p_swap=0.5,
            t_coh=400, protocol=(0,),
            t_trunc=3000, cutoff=n
        )
        print(f"   {n:>8} {r['skr']:>12.6e} {r['mean_t']:>10.2f} "
              f"{r['mean_f']:>10.6f}")

    # ── Test no-cutoff vs cutoff ─────────────────────────────────
    print("\n8. No cutoff vs optimal cutoff:")

    result_no_cut = compute_analytical_baseline(
        p_gen=0.1, w0=0.98, p_swap=0.5,
        t_coh=400, protocol=(0,),
        t_trunc=3000, cutoff=10_000_000
    )

    print(f"   No cutoff: SKR={result_no_cut['skr']:.6e}, "
          f"Mean F={result_no_cut['mean_f']:.6f}")
    print(f"   n*={n_star}:   SKR={best_skr:.6e}, "
          f"Mean F={best_result['mean_f']:.6f}")

    if best_skr > result_no_cut['skr']:
        improvement = (best_skr - result_no_cut['skr']) / result_no_cut['skr'] * 100
        print(f"   Cutoff improves SKR by {improvement:.1f}%")
    elif result_no_cut['skr'] == 0 and best_skr > 0:
        print(f"   Cutoff ENABLES positive SKR (no cutoff gives 0)")
    print(f"   ✓ Cutoff comparison works")

    # ── Test with RepeaterConfig ─────────────────────────────────
    print("\n9. Test with RepeaterConfig object:")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import fast_test_config

        config = fast_test_config()
        result_config = compute_analytical_from_config(config, cutoff=30)

        print(f"   Config: {config}")
        print(f"   SKR: {result_config['skr']:.6e}")
        print(f"   Mean T: {result_config['mean_t']:.2f}")
        print(f"   ✓ Config interface works")

        n_star_cfg, skr_cfg, _ = optimize_fixed_cutoff_from_config(
            config, cutoff_range=range(1, 100))
        print(f"   Optimal cutoff: n* = {n_star_cfg}, SKR = {skr_cfg:.6e}")
        print(f"   ✓ Config optimization works")
    except ImportError:
        print("   (skipping — config not importable from here)")

    print("\n✅ boxili_wrapper.py self-test passed")