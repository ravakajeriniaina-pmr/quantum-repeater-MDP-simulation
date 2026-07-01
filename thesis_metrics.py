"""
thesis_metrics.py
=================
Complete metrics framework for a Master thesis on
Quantum Repeater MDP optimization.

Context:
  - QKD network: metric = Secret Key Rate (SKR)
  - Baseline: Boxi Li fixed-cutoff protocol
  - Improvement: MDP / Reinforcement Learning policy
  - Goal: show MDP >= fixed cutoff, characterize when and why

Metric families:
  M1 — Core QKD performance
  M2 — Comparative gains
  M3 — Statistical validity
  M4 — Physical quality
  M5 — MDP convergence & policy structure
  M6 — Robustness across parameters
  M7 — Distance to theoretical limits (PLOB)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats

from src.physical.werner_utils import (
    secret_fraction_nat, werner_to_fidelity,
    skr_threshold_werner_nat,
)
from src.metrics.skr import skr_from_samples, skr_with_ci


# ══════════════════════════════════════════════════════════════════════
# M1 — CORE QKD PERFORMANCE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CoreMetrics:
    """
    M1: Fundamental QKD performance indicators.
    These are the primary results for the thesis.
    """
    # SKR — two conventions (must report both + explain difference)
    skr_mc:      float   # E[sf(w)] / E[T]  ← MDP objective
    skr_boxili:  float   # sf(E[w]) / E[T]  ← Boxi Li convention
    skr_ci_low:  float   # 95% CI lower
    skr_ci_high: float   # 95% CI upper

    # Delivery time
    mean_T:  float
    std_T:   float
    p50_T:   float       # median
    p90_T:   float       # 90th percentile (latency bound)
    p99_T:   float       # 99th percentile (tail latency)

    # Output quality
    mean_w:  float       # mean Werner parameter
    mean_F:  float       # mean fidelity F = (1+3w)/4
    std_w:   float
    std_F:   float

    # Fraction of episodes with positive SKR
    frac_positive_key: float   # P(sf(w_out) > 0)


def compute_core_metrics(delivery_times: np.ndarray,
                         w_out: np.ndarray) -> CoreMetrics:
    t  = delivery_times.astype(float)
    w  = w_out.astype(float)
    f  = werner_to_fidelity(w)
    sf = np.vectorize(secret_fraction_nat)(w)

    ci = skr_with_ci(t, w)

    return CoreMetrics(
        skr_mc      = float(np.mean(sf) / np.mean(t)),
        skr_boxili  = float(secret_fraction_nat(float(np.mean(w))) / float(np.mean(t))),
        skr_ci_low  = ci["skr_lower"],
        skr_ci_high = ci["skr_upper"],

        mean_T = float(np.mean(t)),
        std_T  = float(np.std(t,  ddof=1)),
        p50_T  = float(np.percentile(t, 50)),
        p90_T  = float(np.percentile(t, 90)),
        p99_T  = float(np.percentile(t, 99)),

        mean_w = float(np.mean(w)),
        mean_F = float(np.mean(f)),
        std_w  = float(np.std(w, ddof=1)),
        std_F  = float(np.std(f, ddof=1)),

        frac_positive_key = float(np.mean(sf > 0)),
    )


# ══════════════════════════════════════════════════════════════════════
# M2 — COMPARATIVE GAINS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ComparativeMetrics:
    """
    M2: Relative improvements between strategies.
    Central thesis result: 'MDP improves over fixed cutoff by X%'
    """
    # Fixed cutoff vs no-cutoff
    gain_fixed_vs_nocut_pct:   float
    # Adaptive MDP vs fixed cutoff (key thesis claim)
    gain_adapt_vs_fixed_pct:   float
    # Adaptive MDP vs no-cutoff
    gain_adapt_vs_nocut_pct:   float

    # Absolute SKR differences
    delta_skr_fixed_vs_nocut:  float
    delta_skr_adapt_vs_fixed:  float

    # Delivery time reduction (%)
    dt_reduction_fixed_pct:    float   # fixed vs no-cutoff
    dt_reduction_adapt_pct:    float   # adaptive vs no-cutoff

    # Quality improvement
    dw_adapt_vs_fixed:         float   # Δ mean_w
    dF_adapt_vs_fixed:         float   # Δ mean_F

    # Optimal fixed cutoff found
    best_n_star: int


def compute_comparative(nocut: CoreMetrics,
                        fixed: CoreMetrics,
                        adapt: CoreMetrics,
                        best_n: int) -> ComparativeMetrics:

    def pct(a, b): return 100*(a - b)/b if b > 0 else float("nan")
    def delta(a, b): return a - b

    return ComparativeMetrics(
        gain_fixed_vs_nocut_pct  = pct(fixed.skr_mc,  nocut.skr_mc),
        gain_adapt_vs_fixed_pct  = pct(adapt.skr_mc,  fixed.skr_mc),
        gain_adapt_vs_nocut_pct  = pct(adapt.skr_mc,  nocut.skr_mc),

        delta_skr_fixed_vs_nocut = delta(fixed.skr_mc, nocut.skr_mc),
        delta_skr_adapt_vs_fixed = delta(adapt.skr_mc, fixed.skr_mc),

        dt_reduction_fixed_pct = pct(nocut.mean_T, fixed.mean_T),
        dt_reduction_adapt_pct = pct(nocut.mean_T, adapt.mean_T),

        dw_adapt_vs_fixed = adapt.mean_w - fixed.mean_w,
        dF_adapt_vs_fixed = adapt.mean_F - fixed.mean_F,

        best_n_star = best_n,
    )


# ══════════════════════════════════════════════════════════════════════
# M3 — STATISTICAL VALIDITY
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StatisticalMetrics:
    """
    M3: Statistical significance of improvements.
    Required for thesis credibility.
    """
    # Welch t-test: is SKR difference significant?
    ttest_adapt_vs_fixed_pvalue:  float
    ttest_adapt_vs_fixed_sig001:  bool

    # Effect size (Cohen's d for SKR)
    cohens_d_adapt_vs_fixed: float

    # Confidence interval on the improvement itself
    improvement_ci_low_pct:  float
    improvement_ci_high_pct: float

    # Number of episodes used
    n_episodes: int

    # Monte Carlo convergence: std(running_mean_SKR) in last 10%
    skr_convergence_std: float


def compute_statistical(t_fixed:  np.ndarray,
                        w_fixed:  np.ndarray,
                        t_adapt:  np.ndarray,
                        w_adapt:  np.ndarray) -> StatisticalMetrics:
    sf_fx = np.vectorize(secret_fraction_nat)(w_fixed)
    sf_ad = np.vectorize(secret_fraction_nat)(w_adapt)

    # Per-episode contribution to SKR numerator
    t_stat, p_val = stats.ttest_ind(sf_fx / t_fixed.astype(float),
                                     sf_ad / t_adapt.astype(float),
                                     equal_var=False)
    sig = bool(p_val < 0.01)

    # Cohen's d
    m1, m2 = np.mean(sf_fx), np.mean(sf_ad)
    s1, s2 = np.std(sf_fx, ddof=1), np.std(sf_ad, ddof=1)
    s_pool = np.sqrt((s1**2 + s2**2) / 2)
    d = float((m2 - m1) / s_pool) if s_pool > 0 else 0.0

    # Bootstrap CI on improvement
    n = min(len(sf_fx), len(sf_ad), 5000)
    rng = np.random.default_rng(99)
    boot_imp = []
    for _ in range(1000):
        i_fx = rng.integers(0, len(sf_fx), n)
        i_ad = rng.integers(0, len(sf_ad), n)
        skr_fx_b = np.mean(sf_fx[i_fx]) / np.mean(t_fixed[i_fx].astype(float))
        skr_ad_b = np.mean(sf_ad[i_ad]) / np.mean(t_adapt[i_ad].astype(float))
        if skr_fx_b > 0:
            boot_imp.append(100*(skr_ad_b - skr_fx_b)/skr_fx_b)
    boot_imp = np.array(boot_imp)

    # Convergence of running SKR in last 10%
    run_skr = np.cumsum(sf_ad) / np.cumsum(t_adapt.astype(float))
    tail = run_skr[int(0.9*len(run_skr)):]

    return StatisticalMetrics(
        ttest_adapt_vs_fixed_pvalue  = float(p_val),
        ttest_adapt_vs_fixed_sig001  = sig,
        cohens_d_adapt_vs_fixed      = d,
        improvement_ci_low_pct       = float(np.percentile(boot_imp, 2.5)),
        improvement_ci_high_pct      = float(np.percentile(boot_imp, 97.5)),
        n_episodes                   = len(t_adapt),
        skr_convergence_std          = float(np.std(tail)),
    )


# ══════════════════════════════════════════════════════════════════════
# M4 — PHYSICAL QUALITY
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PhysicalMetrics:
    """
    M4: Physical quality of entanglement delivered.
    Connects simulation to quantum information theory.
    """
    # Werner threshold
    w_threshold: float          # ≈ 0.7476 for positive SKR
    frac_above_threshold: float # fraction of episodes usable for QKD

    # Mean age at swap (measures how well the protocol manages aging)
    mean_age_link1: float
    mean_age_link2: float
    mean_sum_ages:  float       # relevant for Werner: w = w0²·exp(-(a1+a2)/t_coh)
    mean_age_asymmetry: float   # |a1 - a2| average

    # Entanglement fidelity distribution
    frac_F_above_075: float     # F > 0.75 (entangled)
    frac_F_above_081: float     # F > 0.8107 (positive SKR)
    frac_F_above_090: float     # F > 0.90 (high quality)
    frac_F_above_095: float     # F > 0.95 (near-perfect)

    # Concurrence (entanglement measure)
    mean_concurrence: float

    # SKR rate-delay product
    skr_times_mean_T: float     # SKR × E[T] = E[sf(w)] (normalized quality)


def compute_physical(delivery_times: np.ndarray,
                     w_out: np.ndarray,
                     age1_array: Optional[np.ndarray] = None,
                     age2_array: Optional[np.ndarray] = None) -> PhysicalMetrics:
    w   = w_out.astype(float)
    f   = werner_to_fidelity(w)
    sf  = np.vectorize(secret_fraction_nat)(w)
    w_t = skr_threshold_werner_nat()

    # Concurrence: C(w) = max(0, (3w-1)/2)
    conc = np.maximum(0.0, (3*w - 1) / 2)

    skr_mc = float(np.mean(sf)) / float(np.mean(delivery_times))

    # Age stats (if provided)
    if age1_array is not None and age2_array is not None:
        ma1  = float(np.mean(age1_array))
        ma2  = float(np.mean(age2_array))
        msum = float(np.mean(age1_array + age2_array))
        masym= float(np.mean(np.abs(age1_array - age2_array)))
    else:
        ma1 = ma2 = msum = masym = float("nan")

    return PhysicalMetrics(
        w_threshold          = float(w_t),
        frac_above_threshold = float(np.mean(w > w_t)),

        mean_age_link1    = ma1,
        mean_age_link2    = ma2,
        mean_sum_ages     = msum,
        mean_age_asymmetry= masym,

        frac_F_above_075 = float(np.mean(f > 0.75)),
        frac_F_above_081 = float(np.mean(f > 0.8107)),
        frac_F_above_090 = float(np.mean(f > 0.90)),
        frac_F_above_095 = float(np.mean(f > 0.95)),

        mean_concurrence  = float(np.mean(conc)),
        skr_times_mean_T  = float(skr_mc * np.mean(delivery_times)),
    )


# ══════════════════════════════════════════════════════════════════════
# M5 — MDP CONVERGENCE & POLICY STRUCTURE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MDPMetrics:
    """
    M5: MDP solver diagnostics.
    Validates that the learned policy is meaningful.
    """
    gain:          float    # RVI gain = theoretical SKR
    n_iterations:  int      # iterations to convergence
    converged:     bool
    elapsed_sec:   float

    # Policy structure (for both-entangled states)
    n_swap_states:   int    # states where action = SWAP
    n_wait_states:   int    # states where action = WAIT
    n_cut1_states:   int    # states where action = CUTOFF_1
    n_cut2_states:   int    # states where action = CUTOFF_2
    n_cutall_states: int    # states where action = CUTOFF_ALL

    # Effective cutoff from policy
    effective_cutoff_link1: Optional[int]   # age threshold for link 1
    effective_cutoff_link2: Optional[int]   # age threshold for link 2
    policy_is_symmetric:    bool            # link 1 == link 2 threshold

    # MDP gain vs MC SKR discrepancy (should be < 5%)
    gain_vs_mc_discrepancy_pct: float


def compute_mdp_metrics(mdp: dict, mc_skr: float) -> MDPMetrics:
    from src.mdp.actions import WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL

    policy = mdp["policy"]
    ss     = mdp["state_space"]

    counts = {WAIT:0, SWAP:0, CUTOFF_1:0, CUTOFF_2:0, CUTOFF_ALL:0}
    for s, a in policy.items():
        if a in counts:
            counts[a] += 1

    # Effective cutoff: first age where CUTOFF is chosen (one-entangled state)
    cut1, cut2 = None, None
    for a1 in range(ss.t_max + 1):
        s = (1, a1, 0, 0)
        if ss.contains(s) and policy[s] == CUTOFF_1:
            cut1 = a1
            break
    for a2 in range(ss.t_max + 1):
        s = (0, 0, 1, a2)
        if ss.contains(s) and policy[s] == CUTOFF_2:
            cut2 = a2
            break

    gain = mdp["gain"]
    disc = float(abs(gain - mc_skr) / gain * 100) if gain > 0 else float("nan")

    return MDPMetrics(
        gain          = float(gain),
        n_iterations  = int(mdp["n_iter"]),
        converged     = bool(mdp["converged"]),
        elapsed_sec   = float(mdp["elapsed"]),

        n_swap_states   = counts[SWAP],
        n_wait_states   = counts[WAIT],
        n_cut1_states   = counts[CUTOFF_1],
        n_cut2_states   = counts[CUTOFF_2],
        n_cutall_states = counts[CUTOFF_ALL],

        effective_cutoff_link1    = cut1,
        effective_cutoff_link2    = cut2,
        policy_is_symmetric       = (cut1 == cut2),
        gain_vs_mc_discrepancy_pct = disc,
    )


# ══════════════════════════════════════════════════════════════════════
# M6 — ROBUSTNESS SWEEP
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessPoint:
    """One point in the parameter sweep."""
    p_gen:  float
    t_coh:  float
    skr_no: float
    skr_fx: float
    skr_ad: float
    best_n: int
    gain_adapt_vs_fixed_pct: float


def build_robustness_table(
        p_gen_values, t_coh_values,
        w0: float = 0.98, p_swap: float = 0.5,
        n_episodes: int = 30_000, seed: int = 42
) -> list:
    """
    Sweep over (p_gen, t_coh) grid.
    Returns list of RobustnessPoint — one per grid cell.
    """
    from src.simulation.mc_engine import (
        NoCutoffPolicy, FixedCutoffPolicy, AdaptivePolicy, run_simulation,
    )
    from src.mdp.value_iteration import solve_mdp
    from src.mdp.state_space import suggest_t_max

    results = []
    for p_gen in p_gen_values:
        for t_coh in t_coh_values:
            p = dict(p_gen=p_gen, p_swap=p_swap, w0=w0, t_coh=t_coh)

            mc_no = run_simulation(NoCutoffPolicy(), **p, n_episodes=n_episodes, seed=seed)
            skr_no = skr_from_samples(mc_no.delivery_times.astype(float), mc_no.w_out_array)

            best_n, best_skr = 1, -np.inf
            for n in range(1, min(int(5*t_coh*p_gen)+15, 80)+1):
                mc = run_simulation(FixedCutoffPolicy(n), **p, n_episodes=n_episodes//2, seed=seed+1)
                s  = skr_from_samples(mc.delivery_times.astype(float), mc.w_out_array)
                if s > best_skr:
                    best_skr, best_n = s, n

            t_max = min(max(suggest_t_max(t_coh), best_n+10), 100)
            mdp   = solve_mdp(**p, t_max=t_max, verbose=False)
            mc_ad = run_simulation(AdaptivePolicy(mdp["policy"], t_max), **p, n_episodes=n_episodes, seed=seed+2)
            skr_ad = skr_from_samples(mc_ad.delivery_times.astype(float), mc_ad.w_out_array)

            imp = 100*(skr_ad - best_skr)/best_skr if best_skr > 0 else float("nan")
            results.append(RobustnessPoint(
                p_gen=p_gen, t_coh=t_coh,
                skr_no=skr_no, skr_fx=best_skr, skr_ad=skr_ad,
                best_n=best_n, gain_adapt_vs_fixed_pct=imp,
            ))
    return results


# ══════════════════════════════════════════════════════════════════════
# M7 — DISTANCE TO PLOB
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PLOBMetrics:
    """
    M7: How far is the protocol from the theoretical limit?
    Contextualizes results for a thesis.
    """
    plob_bound:         float    # -log2(1-η)
    distance_km:        float
    eta_fiber:          float

    ratio_nocut_plob:   float   # SKR_no / PLOB
    ratio_fixed_plob:   float   # SKR_fixed / PLOB
    ratio_adapt_plob:   float   # SKR_adapt / PLOB

    # Gap to close (in %)
    gap_adapt_to_plob_pct: float

    # Number of "PLOB halvings" away
    # (how many times would we need to double SKR to reach PLOB)
    halvings_to_plob: float


def compute_plob_metrics(skr_no: float, skr_fx: float, skr_ad: float,
                         p_gen: float, n_segments: int = 2) -> PLOBMetrics:
    from src.metrics.plob import pgen_to_distance, transmissivity, plob_bound

    dist = pgen_to_distance(p_gen, n_segments=n_segments)
    eta  = transmissivity(dist)
    kp   = plob_bound(eta)

    def ratio(s): return s / kp if (np.isfinite(kp) and kp > 0) else float("nan")
    def gap(s):   return 100*(kp - s)/kp if (np.isfinite(kp) and kp > 0) else float("nan")
    def halvings(s):
        if kp <= 0 or s <= 0 or not np.isfinite(kp):
            return float("nan")
        return float(np.log2(kp / s))

    return PLOBMetrics(
        plob_bound    = float(kp),
        distance_km   = float(dist),
        eta_fiber     = float(eta),
        ratio_nocut_plob = ratio(skr_no),
        ratio_fixed_plob = ratio(skr_fx),
        ratio_adapt_plob = ratio(skr_ad),
        gap_adapt_to_plob_pct = gap(skr_ad),
        halvings_to_plob      = halvings(skr_ad),
    )


# ══════════════════════════════════════════════════════════════════════
# FULL REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════

def print_full_report(label: str,
                      core_no:  CoreMetrics,
                      core_fx:  CoreMetrics,
                      core_ad:  CoreMetrics,
                      comp:     ComparativeMetrics,
                      stat:     StatisticalMetrics,
                      phys_ad:  PhysicalMetrics,
                      mdp_m:    MDPMetrics,
                      plob_m:   PLOBMetrics) -> None:

    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  FULL METRICS REPORT — {label}")
    print(sep)

    print("\n── M1: Core QKD Performance ──────────────────────────────")
    print(f"  {'':30s} {'No cut':>12} {'Fixed':>12} {'Adaptive':>12}")
    print(f"  {'─'*66}")
    print(f"  {'SKR (MC convention)':30s} "
          f"{core_no.skr_mc:>12.4e} {core_fx.skr_mc:>12.4e} {core_ad.skr_mc:>12.4e}")
    print(f"  {'SKR (Boxi Li conv.)':30s} "
          f"{core_no.skr_boxili:>12.4e} {core_fx.skr_boxili:>12.4e} {core_ad.skr_boxili:>12.4e}")
    print(f"  {'95% CI (adaptive)':30s} "
          f"  [{core_ad.skr_ci_low:.4e}, {core_ad.skr_ci_high:.4e}]")
    print(f"  {'E[T]':30s} "
          f"{core_no.mean_T:>12.2f} {core_fx.mean_T:>12.2f} {core_ad.mean_T:>12.2f}")
    print(f"  {'P90[T]':30s} "
          f"{core_no.p90_T:>12.2f} {core_fx.p90_T:>12.2f} {core_ad.p90_T:>12.2f}")
    print(f"  {'E[F]':30s} "
          f"{core_no.mean_F:>12.4f} {core_fx.mean_F:>12.4f} {core_ad.mean_F:>12.4f}")
    print(f"  {'Frac. positive key':30s} "
          f"{core_no.frac_positive_key:>12.3f} "
          f"{core_fx.frac_positive_key:>12.3f} "
          f"{core_ad.frac_positive_key:>12.3f}")

    print("\n── M2: Comparative Gains ──────────────────────────────────")
    print(f"  Fixed vs No-cut:          {comp.gain_fixed_vs_nocut_pct:>+8.2f}%")
    print(f"  Adaptive vs Fixed:        {comp.gain_adapt_vs_fixed_pct:>+8.2f}%   ← KEY RESULT")
    print(f"  Adaptive vs No-cut:       {comp.gain_adapt_vs_nocut_pct:>+8.2f}%")
    print(f"  Optimal fixed cutoff n*:  {comp.best_n_star}")
    print(f"  Δ mean_F (adapt-fixed):   {comp.dF_adapt_vs_fixed:>+8.5f}")

    print("\n── M3: Statistical Validity ───────────────────────────────")
    sig = "✅ significant" if stat.ttest_adapt_vs_fixed_sig001 else "⚠️  NOT significant"
    print(f"  t-test adapt vs fixed:    p = {stat.ttest_adapt_vs_fixed_pvalue:.4e}  {sig}")
    print(f"  Cohen's d:                {stat.cohens_d_adapt_vs_fixed:.3f}")
    print(f"  Bootstrap 95% CI on Δ:   [{stat.improvement_ci_low_pct:+.2f}%, "
          f"{stat.improvement_ci_high_pct:+.2f}%]")
    print(f"  n_episodes:               {stat.n_episodes:,}")
    print(f"  SKR convergence std:      {stat.skr_convergence_std:.4e}")

    print("\n── M4: Physical Quality ───────────────────────────────────")
    print(f"  Werner threshold (w*):    {phys_ad.w_threshold:.4f}")
    print(f"  Frac episodes above w*:   {phys_ad.frac_above_threshold:.3f}")
    print(f"  F > 0.8107 (SKR > 0):     {phys_ad.frac_F_above_081:.3f}")
    print(f"  F > 0.90:                 {phys_ad.frac_F_above_090:.3f}")
    print(f"  F > 0.95:                 {phys_ad.frac_F_above_095:.3f}")
    print(f"  Mean concurrence:         {phys_ad.mean_concurrence:.4f}")
    print(f"  Mean sum of ages:         {phys_ad.mean_sum_ages:.2f}")
    print(f"  Mean age asymmetry |Δa|:  {phys_ad.mean_age_asymmetry:.2f}")

    print("\n── M5: MDP Policy ─────────────────────────────────────────")
    print(f"  Converged:                {mdp_m.converged}  "
          f"({mdp_m.n_iterations} iters, {mdp_m.elapsed_sec:.1f}s)")
    print(f"  Theoretical gain:         {mdp_m.gain:.4e}")
    print(f"  Gain vs MC discrepancy:   {mdp_m.gain_vs_mc_discrepancy_pct:.2f}%")
    print(f"  Effective cutoff link 1:  {mdp_m.effective_cutoff_link1}")
    print(f"  Effective cutoff link 2:  {mdp_m.effective_cutoff_link2}")
    print(f"  Policy symmetric:         {mdp_m.policy_is_symmetric}")
    print(f"  SWAP / WAIT / CUT states: "
          f"{mdp_m.n_swap_states} / {mdp_m.n_wait_states} / "
          f"{mdp_m.n_cut1_states + mdp_m.n_cut2_states + mdp_m.n_cutall_states}")

    print("\n── M7: Distance to PLOB ───────────────────────────────────")
    print(f"  Equivalent distance:      {plob_m.distance_km:.2f} km")
    print(f"  Channel transmissivity:   {plob_m.eta_fiber:.4e}")
    print(f"  PLOB bound:               {plob_m.plob_bound:.4e}")
    print(f"  Adaptive / PLOB:          {plob_m.ratio_adapt_plob:.4f}  "
          f"({plob_m.ratio_adapt_plob*100:.2f}% of PLOB)")
    print(f"  Gap to PLOB:              {plob_m.gap_adapt_to_plob_pct:.2f}%")
    print(f"  Halvings to reach PLOB:   {plob_m.halvings_to_plob:.2f}")

    print(f"\n{sep}\n")