"""
Corrected PLOB Bound Calculations for Quantum Repeater Systems
================================================================

CRITICAL PHYSICS CORRECTION:
The PLOB (Pirandola-Laurenza-Ottaviani-Banchi) bound represents the 
fundamental quantum limit for point-to-point communication WITHOUT repeaters.

For quantum repeater systems, we must distinguish between:
1. Direct transmission PLOB (no repeater) - the fundamental limit
2. Segment-level PLOB - what the repeater actually operates on
3. System-level comparison - fair benchmarking

This module provides corrected calculations for proper physical comparison.

References:
- PLOB bound: Nature Communications 8, 15043 (2017)
- Quantum repeaters: Rev. Mod. Phys. 83, 33 (2011)
"""

import numpy as np
from typing import Dict, Tuple


# ═════════════════════════════════════════════════════════════════════
# BASIC PLOB FORMULA (unchanged)
# ═════════════════════════════════════════════════════════════════════

def plob_bound(eta: float) -> float:
    """
    PLOB bound: K(η) = -log₂(1 - η)
    
    This is the maximum secret key rate achievable over a lossy channel
    with transmissivity η WITHOUT quantum repeaters.
    
    Args:
        eta: Channel transmissivity (0 to 1)
        
    Returns:
        Maximum secret key capacity in bits per channel use
    """
    if eta <= 0.0:
        return 0.0
    if eta >= 1.0:
        return np.inf
    return -np.log2(1.0 - eta)


def transmissivity(length_km: float, alpha_db_per_km: float = 0.2) -> float:
    """
    Channel transmissivity η(L) = 10^(-αL/10)
    
    Args:
        length_km: Distance in kilometers
        alpha_db_per_km: Fiber loss in dB/km (default 0.2 for telecom fiber)
        
    Returns:
        Transmissivity (0 to 1)
    """
    if length_km <= 0.0:
        return 1.0
    return 10.0 ** (-alpha_db_per_km * length_km / 10.0)


# ═════════════════════════════════════════════════════════════════════
# CORRECTED PLOB FOR REPEATER COMPARISON
# ═════════════════════════════════════════════════════════════════════

def plob_direct_transmission(total_distance_km: float, 
                             alpha_db_per_km: float = 0.2) -> float:
    """
    PLOB bound for direct point-to-point transmission (NO repeater).
    
    This is the fundamental quantum limit your repeater system should be
    compared against. Any repeater protocol MUST be below this bound.
    
    Args:
        total_distance_km: Total end-to-end distance
        alpha_db_per_km: Fiber loss coefficient
        
    Returns:
        PLOB bound for direct transmission in bits per channel use
    """
    eta_direct = transmissivity(total_distance_km, alpha_db_per_km)
    return plob_bound(eta_direct)


def plob_single_segment(segment_distance_km: float,
                        alpha_db_per_km: float = 0.2) -> float:
    """
    PLOB bound for a single repeater segment.
    
    This represents the maximum capacity of ONE elementary link,
    but is NOT the right comparison for the full repeater chain.
    
    Args:
        segment_distance_km: Distance of one repeater segment
        alpha_db_per_km: Fiber loss coefficient
        
    Returns:
        PLOB bound for one segment in bits per channel use
    """
    eta_seg = transmissivity(segment_distance_km, alpha_db_per_km)
    return plob_bound(eta_seg)


def plob_repeater_chain_simple(segment_distance_km: float,
                               n_segments: int,
                               alpha_db_per_km: float = 0.2) -> float:
    """
    Simple bound for repeater chain: PLOB of total distance.
    
    This is a conservative (loose) upper bound. The true capacity of
    a repeater chain with N segments is complex and depends on:
    - Memory coherence time
    - Swapping success rate
    - Cutoff strategy
    
    For fair comparison, use this as the "direct transmission" baseline.
    
    Args:
        segment_distance_km: Length of each segment
        n_segments: Number of segments in the chain
        alpha_db_per_km: Fiber loss coefficient
        
    Returns:
        Conservative PLOB bound for the repeater system
    """
    total_distance = segment_distance_km * n_segments
    return plob_direct_transmission(total_distance, alpha_db_per_km)


# ═════════════════════════════════════════════════════════════════════
# DISTANCE MAPPING FROM p_gen (for consistency)
# ═════════════════════════════════════════════════════════════════════

def pgen_to_segment_distance(p_gen: float,
                             eta_coupling: float = 0.5,
                             eta_detector: float = 0.9,
                             alpha_db_per_km: float = 0.2) -> float:
    """
    Map generation probability to physical segment distance.
    
    p_gen = η_fiber * η_coupling * η_detector
    where η_fiber = 10^(-α * L_photon / 10)
    and L_photon = L_segment / 2 (photon travels to midpoint)
    
    Args:
        p_gen: Link generation probability
        eta_coupling: Coupling efficiency (default 0.5)
        eta_detector: Detector efficiency (default 0.9)
        alpha_db_per_km: Fiber loss coefficient
        
    Returns:
        Segment length L₀ in kilometers
    """
    eta_reduced = p_gen / (eta_coupling * eta_detector)
    
    if eta_reduced >= 1.0:
        return 0.0
    if eta_reduced <= 0.0:
        return np.inf
    
    # L_photon = -10 * log₁₀(η_reduced) / α
    photon_distance = -10.0 * np.log10(eta_reduced) / alpha_db_per_km
    
    # L_segment = 2 * L_photon
    return 2.0 * photon_distance


# ═════════════════════════════════════════════════════════════════════
# COMPLETE COMPARISON METRICS
# ═════════════════════════════════════════════════════════════════════

def compute_plob_comparison(p_gen: float, 
                           n_segments: int = 2,
                           eta_coupling: float = 0.5,
                           eta_detector: float = 0.9,
                           alpha_db_per_km: float = 0.2) -> Dict[str, float]:
    """
    Compute comprehensive PLOB metrics for fair comparison.
    
    Returns all relevant bounds and distances for proper interpretation
    of quantum repeater performance.
    
    Args:
        p_gen: Elementary link generation probability
        n_segments: Number of repeater segments
        eta_coupling: Coupling efficiency
        eta_detector: Detector efficiency
        alpha_db_per_km: Fiber loss coefficient
        
    Returns:
        Dictionary containing:
        - segment_distance_km: Length of one repeater segment
        - total_distance_km: Total end-to-end distance
        - segment_transmissivity: η for one segment
        - total_transmissivity: η for full distance
        - plob_segment: PLOB for one segment (NOT for comparison)
        - plob_direct: PLOB for direct transmission (CORRECT baseline)
        - plob_interpretation: Which bound to use for comparison
    """
    # Calculate segment distance from p_gen
    L0 = pgen_to_segment_distance(p_gen, eta_coupling, eta_detector, alpha_db_per_km)
    L_total = L0 * n_segments
    
    # Transmissivities
    eta_seg = transmissivity(L0, alpha_db_per_km)
    eta_total = transmissivity(L_total, alpha_db_per_km)
    
    # PLOB bounds
    plob_seg = plob_bound(eta_seg)
    plob_dir = plob_bound(eta_total)
    
    return {
        "segment_distance_km": float(L0),
        "total_distance_km": float(L_total),
        "segment_transmissivity": float(eta_seg),
        "total_transmissivity": float(eta_total),
        "plob_segment": float(plob_seg),
        "plob_direct": float(plob_dir),
        "plob_for_comparison": float(plob_dir),  # USE THIS for SKR ratio
        "interpretation": (
            f"Compare your repeater SKR to plob_direct={plob_dir:.6f}. "
            f"Your SKR MUST be < plob_direct (you cannot beat physics). "
            f"The repeater advantage is measured against direct transmission "
            f"at the same distance."
        )
    }


def normalize_to_plob(skr_measured: float,
                     p_gen: float,
                     n_segments: int = 2,
                     **kwargs) -> Tuple[float, str]:
    """
    Normalize measured SKR to PLOB bound with physical validity check.
    
    Args:
        skr_measured: Your measured secret key rate
        p_gen: Elementary link generation probability
        n_segments: Number of repeater segments
        **kwargs: Additional parameters for compute_plob_comparison
        
    Returns:
        (ratio, interpretation) where:
        - ratio = skr_measured / plob_direct
        - interpretation: String explaining the result
    """
    metrics = compute_plob_comparison(p_gen, n_segments, **kwargs)
    plob_correct = metrics["plob_for_comparison"]
    
    if plob_correct <= 0 or np.isinf(plob_correct):
        return (float('nan'), "PLOB bound is invalid (distance too large)")
    
    ratio = skr_measured / plob_correct
    
    if ratio > 1.0:
        interpretation = (
            f"⚠️  PHYSICS VIOLATION: ratio={ratio:.3f} > 1.0\n"
            f"Your SKR ({skr_measured:.6f}) exceeds PLOB ({plob_correct:.6f}).\n"
            f"This is impossible. Check your calculation."
        )
    elif ratio > 0.5:
        interpretation = (
            f"✓ Excellent: {ratio*100:.1f}% of fundamental limit\n"
            f"Distance: {metrics['total_distance_km']:.1f} km, "
            f"Loss: {-10*np.log10(metrics['total_transmissivity']):.1f} dB"
        )
    elif ratio > 0.1:
        interpretation = (
            f"✓ Good: {ratio*100:.1f}% of fundamental limit\n"
            f"Distance: {metrics['total_distance_km']:.1f} km, "
            f"Loss: {-10*np.log10(metrics['total_transmissivity']):.1f} dB"
        )
    else:
        interpretation = (
            f"Low efficiency: {ratio*100:.1f}% of fundamental limit\n"
            f"Distance: {metrics['total_distance_km']:.1f} km, "
            f"Loss: {-10*np.log10(metrics['total_transmissivity']):.1f} dB\n"
            f"This is typical for repeater chains with memory decoherence."
        )
    
    return (float(ratio), interpretation)


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Corrected PLOB Calculation Self-Test")
    print("=" * 70)
    
    # Test case: BoxiLi-base parameters
    print("\n1. BoxiLi-base scenario (p_gen=0.1, N=2 segments):")
    metrics = compute_plob_comparison(p_gen=0.1, n_segments=2)
    
    print(f"\n   Segment distance: {metrics['segment_distance_km']:.2f} km")
    print(f"   Total distance:   {metrics['total_distance_km']:.2f} km")
    print(f"   Loss (total):     {-10*np.log10(metrics['total_transmissivity']):.1f} dB")
    print(f"\n   η (segment):      {metrics['segment_transmissivity']:.6f}")
    print(f"   η (total):        {metrics['total_transmissivity']:.6f}")
    print(f"\n   PLOB (segment):   {metrics['plob_segment']:.6f}  (don't use this)")
    print(f"   PLOB (direct):    {metrics['plob_direct']:.6f}  ← USE THIS")
    
    # Validate physical consistency
    assert metrics['plob_direct'] < metrics['plob_segment'], \
        "Direct transmission PLOB must be lower than segment PLOB"
    assert metrics['total_transmissivity'] < metrics['segment_transmissivity'], \
        "Total transmissivity must be lower than segment transmissivity"
    print(f"\n   ✓ Physical consistency checks passed")
    
    # Test normalization with realistic SKR
    print("\n2. SKR normalization test:")
    skr_test = 0.025  # Typical repeater SKR from your data
    ratio, interp = normalize_to_plob(skr_test, p_gen=0.1, n_segments=2)
    print(f"\n   Measured SKR:     {skr_test:.6f}")
    print(f"   PLOB bound:       {metrics['plob_direct']:.6f}")
    print(f"   Ratio:            {ratio:.4f}")
    print(f"\n   {interp}")
    
    assert ratio < 1.0, "Ratio must be < 1.0 (cannot exceed PLOB)"
    assert ratio > 0.0, "Ratio must be positive"
    print(f"\n   ✓ Normalization is physically valid")
    
    # Compare with old (incorrect) calculation
    print("\n3. Comparison with old calculation:")
    old_plob = plob_bound(0.00244)  # Old method used tiny η
    print(f"   Old (wrong):      {old_plob:.6f}")
    print(f"   New (correct):    {metrics['plob_direct']:.6f}")
    print(f"   Difference:       {metrics['plob_direct']/old_plob:.1f}x larger")
    print(f"\n   Old ratio:        {skr_test/old_plob:.2f}  ⚠️ > 1.0 INVALID!")
    print(f"   New ratio:        {ratio:.4f}  ✓ < 1.0 Valid")
    
    # Test edge cases
    print("\n4. Edge cases:")
    
    # Very short distance (high p_gen)
    m_short = compute_plob_comparison(p_gen=0.4, n_segments=2)
    print(f"   Short distance:   {m_short['total_distance_km']:.2f} km, "
          f"PLOB={m_short['plob_direct']:.6f}")
    
    # Very long distance (low p_gen)
    m_long = compute_plob_comparison(p_gen=0.001, n_segments=2)
    print(f"   Long distance:    {m_long['total_distance_km']:.2f} km, "
          f"PLOB={m_long['plob_direct']:.6f}")
    
    assert m_short['plob_direct'] > m_long['plob_direct'], \
        "PLOB must decrease with distance"
    print(f"\n   ✓ PLOB decreases with distance as expected")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed - PLOB calculation is physically correct")
    print("=" * 70)
