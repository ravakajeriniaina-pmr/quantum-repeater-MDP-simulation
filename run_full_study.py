"""
run_full_study.py  —  patched (label bug fixed)
"""

import os
import csv
import numpy as np
from dataclasses import dataclass

from src.simulation.mc_engine import (
    NoCutoffPolicy, FixedCutoffPolicy, AdaptivePolicy, run_simulation,
)
from src.metrics.skr import skr_from_samples, skr_with_ci
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import suggest_t_max
from src.metrics.plob import pgen_to_distance, transmissivity, plob_bound
from src.physical.werner_utils import secret_fraction_nat, werner_to_fidelity

OUT_DIR = "results/full_study"
os.makedirs(OUT_DIR, exist_ok=True)

SEED    = 42
N_SWEEP = 40_000
N_EVAL  = 120_000
N_TOMO  = 5_000

BOXILI_CASE = dict(label="BoxiLi-base", p_gen=0.1, p_swap=0.5, w0=0.98, t_coh=400.0)

CASES = [
    BOXILI_CASE,
    dict(label="short-memory",  p_gen=0.1,  p_swap=0.5, w0=0.98, t_coh=50.0),
    dict(label="fast-gen",      p_gen=0.3,  p_swap=0.5, w0=0.98, t_coh=200.0),
    dict(label="slow-gen",      p_gen=0.05, p_swap=0.5, w0=0.98, t_coh=400.0),
    dict(label="imperfect-src", p_gen=0.1,  p_swap=0.5, w0=0.95, t_coh=400.0),
]


# ── KEY FIX: helper that strips "label" ──────────────────────────────
def phys(case: dict) -> dict:
    """Physical parameters only — no 'label' key."""
    return {k: case[k] for k in ("p_gen", "p_swap", "w0", "t_coh")}


@dataclass
class EvalResult:
    skr: float; skr_hw: float
    mean_t: float; mean_w: float; mean_f: float
    p50_t: float; p90_t: float; p99_t: float


def eval_policy(policy, *, p_gen, p_swap, w0, t_coh,
                n_episodes, seed) -> EvalResult:
    mc = run_simulation(policy, p_gen=p_gen, w0=w0, t_coh=t_coh,
                        p_swap=p_swap, n_episodes=n_episodes, seed=seed)
    ci = skr_with_ci(mc.delivery_times.astype(float), mc.w_out_array)
    t  = mc.delivery_times.astype(float)
    w  = mc.w_out_array
    f  = werner_to_fidelity(w)
    return EvalResult(
        skr=ci["skr"], skr_hw=ci["skr_hw"],
        mean_t=float(np.mean(t)), mean_w=float(np.mean(w)), mean_f=float(np.mean(f)),
        p50_t=float(np.percentile(t, 50)),
        p90_t=float(np.percentile(t, 90)),
        p99_t=float(np.percentile(t, 99)),
    )


def skr_boxili_convention(delivery_times, w_out_array) -> float:
    mean_t = float(np.mean(delivery_times))
    mean_w = float(np.mean(w_out_array))
    return float(max(0.0, secret_fraction_nat(mean_w) / mean_t)) if mean_t > 0 else 0.0


def find_best_fixed_cutoff(case: dict, max_cut: int = 150):
    p = phys(case)
    best_n, best_skr, curve = 1, -np.inf, []
    for n in range(1, max_cut + 1):
        mc  = run_simulation(FixedCutoffPolicy(n), **p, n_episodes=N_SWEEP, seed=SEED+100)
        skr = skr_from_samples(mc.delivery_times.astype(float), mc.w_out_array)
        curve.append((n, skr))
        if skr > best_skr:
            best_skr, best_n = skr, n
    return best_n, best_skr, curve


def build_mdp(case: dict, best_n: int):
    p     = phys(case)
    t_max = min(max(suggest_t_max(p["t_coh"]), best_n + 10), 120)
    mdp   = solve_mdp(**p, t_max=t_max, verbose=False)
    return mdp, t_max


def plob_for_case(case: dict, n_segments: int = 2):
    dist = pgen_to_distance(case["p_gen"], n_segments=n_segments)
    eta  = transmissivity(dist)
    k    = plob_bound(eta)
    return float(dist), float(eta), float(k)


def tomography_series(case: dict, policy, n_episodes=N_TOMO, seed=123) -> dict:
    p  = phys(case)
    mc = run_simulation(policy, **p, n_episodes=n_episodes, seed=seed)
    t  = mc.delivery_times.astype(float)
    w  = mc.w_out_array.astype(float)
    f  = werner_to_fidelity(w)
    sf = np.vectorize(secret_fraction_nat)(w)
    ep = np.arange(1, n_episodes + 1, dtype=float)
    return dict(t=t, w=w, f=f,
                running_mean_w =np.cumsum(w)  / ep,
                running_mean_f =np.cumsum(f)  / ep,
                running_skr    =np.cumsum(sf) / np.cumsum(t))


def main():
    summary_rows, cutoff_rows, per_case_rows = [], [], []

    for case in CASES:
        label = case["label"]
        p     = phys(case)           # ← only physical keys
        print(f"\n{'='*55}\n=== {label} ===")

        # 1. Cutoff scan
        print("  [1/4] cutoff scan …")
        best_n, _, curve = find_best_fixed_cutoff(case)
        cutoff_rows.extend([[label, n, skr] for n, skr in curve])
        print(f"        best n* = {best_n}")

        # 2. No-cutoff
        print("  [2/4] no-cutoff …")
        no_cut = eval_policy(NoCutoffPolicy(), **p, n_episodes=N_EVAL, seed=SEED)

        # 3. Fixed cutoff
        print("  [3/4] fixed cutoff …")
        fixed = eval_policy(FixedCutoffPolicy(best_n), **p, n_episodes=N_EVAL, seed=SEED+1)

        # 4. MDP + adaptive
        print("  [4/4] MDP + adaptive …")
        mdp, t_max = build_mdp(case, best_n)
        adapt = eval_policy(
            AdaptivePolicy(mdp["policy"], t_max), **p,
            n_episodes=N_EVAL, seed=SEED+2,
        )
        print(f"        gain={mdp['gain']:.4e}  MC-SKR={adapt.skr:.4e}")

        # Boxi Li convention
        mc_bl_no = run_simulation(NoCutoffPolicy(),          **p, n_episodes=40_000, seed=SEED+20)
        mc_bl_fx = run_simulation(FixedCutoffPolicy(best_n), **p, n_episodes=40_000, seed=SEED+21)
        bl_no = skr_boxili_convention(mc_bl_no.delivery_times.astype(float), mc_bl_no.w_out_array)
        bl_fx = skr_boxili_convention(mc_bl_fx.delivery_times.astype(float), mc_bl_fx.w_out_array)

        # PLOB
        dist_km, eta, kplob = plob_for_case(case)
        r_fx = fixed.skr / kplob if (np.isfinite(kplob) and kplob > 0) else float("nan")
        r_ad = adapt.skr / kplob if (np.isfinite(kplob) and kplob > 0) else float("nan")

        # Improvements
        def imp(a, b): return 100*(a - b)/b if b > 0 else float("nan")

        summary_rows.append([
            label, p["p_gen"], p["p_swap"], p["w0"], p["t_coh"],
            best_n, t_max, mdp["gain"], mdp["converged"], mdp["n_iter"],
            no_cut.skr, fixed.skr,  adapt.skr,
            no_cut.skr_hw, fixed.skr_hw, adapt.skr_hw,
            imp(fixed.skr, no_cut.skr),
            imp(adapt.skr, fixed.skr),
            imp(adapt.skr, no_cut.skr),
            no_cut.mean_t, fixed.mean_t, adapt.mean_t,
            no_cut.mean_w, fixed.mean_w, adapt.mean_w,
            no_cut.mean_f, fixed.mean_f, adapt.mean_f,
            bl_no, bl_fx,
            dist_km, eta, kplob, r_fx, r_ad,
        ])

        for strat, res in [("no_cutoff", no_cut), ("fixed", fixed), ("adaptive", adapt)]:
            per_case_rows.append([label, strat, res.p50_t, res.p90_t, res.p99_t])

        # Tomography (BoxiLi-base only)
        if label == "BoxiLi-base":
            print("  [+] tomography …")
            tno = tomography_series(case, NoCutoffPolicy(),                     seed=700)
            tfx = tomography_series(case, FixedCutoffPolicy(best_n),            seed=701)
            tad = tomography_series(case, AdaptivePolicy(mdp["policy"], t_max), seed=702)
            np.savez(os.path.join(OUT_DIR, "tomography_series.npz"),
                     t_no=tno["t"],  w_no=tno["w"],  f_no=tno["f"],
                     rw_no=tno["running_mean_w"],  rf_no=tno["running_mean_f"],  rskr_no=tno["running_skr"],
                     t_fx=tfx["t"],  w_fx=tfx["w"],  f_fx=tfx["f"],
                     rw_fx=tfx["running_mean_w"],  rf_fx=tfx["running_mean_f"],  rskr_fx=tfx["running_skr"],
                     t_ad=tad["t"],  w_ad=tad["w"],  f_ad=tad["f"],
                     rw_ad=tad["running_mean_w"],  rf_ad=tad["running_mean_f"],  rskr_ad=tad["running_skr"])

    # ── Write CSVs ──────────────────────────────────────────────────
    _hdr = [
        "label","p_gen","p_swap","w0","t_coh",
        "best_fixed_n","mdp_t_max","mdp_gain","mdp_converged","mdp_n_iter",
        "skr_no","skr_fixed","skr_adapt","ci_no","ci_fixed","ci_adapt",
        "imp_fixed_vs_no_pct","imp_adapt_vs_fixed_pct","imp_adapt_vs_no_pct",
        "mean_t_no","mean_t_fixed","mean_t_adapt",
        "mean_w_no","mean_w_fixed","mean_w_adapt",
        "mean_f_no","mean_f_fixed","mean_f_adapt",
        "skr_boxili_no","skr_boxili_fixed",
        "distance_km","eta","plob_bound","fixed_over_plob","adapt_over_plob",
    ]
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(_hdr); w.writerows(summary_rows)
    with open(os.path.join(OUT_DIR, "cutoff_scan.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","cutoff_n","skr"]); w.writerows(cutoff_rows)
    with open(os.path.join(OUT_DIR, "per_case_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","strategy","p50_delivery","p90_delivery","p99_delivery"])
        w.writerows(per_case_rows)

    print(f"\n{'='*55}")
    print(f"✅  Done.  Outputs in:  {OUT_DIR}")


if __name__ == "__main__":
    main()