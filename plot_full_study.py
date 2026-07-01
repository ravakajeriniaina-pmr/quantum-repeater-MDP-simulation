"""
plot_full_study.py
==================
Plots for run_full_study.py outputs.
Reads:
- results/full_study/summary.csv
- results/full_study/cutoff_scan.csv
- results/full_study/tomography_series.npz
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "results/full_study"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def as_float(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def plot_cutoff_scans(summary, cutoff_rows):
    labels = sorted(set(r["label"] for r in cutoff_rows))
    for label in labels:
        rows = [r for r in cutoff_rows if r["label"] == label]
        n = np.array([int(r["cutoff_n"]) for r in rows])
        skr = np.array([float(r["skr"]) for r in rows])

        srow = [r for r in summary if r["label"] == label][0]
        best_n = int(float(srow["best_fixed_n"]))
        skr_no = float(srow["skr_no"])
        skr_ad = float(srow["skr_adapt"])

        plt.figure(figsize=(8, 4.8))
        plt.plot(n, skr, "b.-", label="Fixed cutoff scan")
        plt.axvline(best_n, color="b", ls=":", lw=1.5, label=f"Best fixed n*={best_n}")
        plt.axhline(skr_no, color="gray", ls="--", label=f"No cutoff ({skr_no:.3e})")
        plt.axhline(skr_ad, color="r", ls="-", label=f"Adaptive MDP ({skr_ad:.3e})")
        plt.xlabel("Cutoff n*")
        plt.ylabel("SKR")
        plt.title(f"SKR vs Fixed Cutoff — {label}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"cutoff_scan_{label}.png"), dpi=200)
        plt.close()


def plot_strategy_comparison(summary):
    labels = [r["label"] for r in summary]
    x = np.arange(len(labels))
    w = 0.25

    skr_no = as_float(summary, "skr_no")
    skr_fx = as_float(summary, "skr_fixed")
    skr_ad = as_float(summary, "skr_adapt")

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, skr_no, width=w, color="gray", label="No cutoff")
    plt.bar(x,     skr_fx, width=w, color="#4a90d9", label="Fixed")
    plt.bar(x + w, skr_ad, width=w, color="#d94a4a", label="Adaptive")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("SKR")
    plt.title("Strategy Comparison Across Regimes")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "strategy_comparison.png"), dpi=200)
    plt.close()


def plot_plob_comparison(summary):
    labels = [r["label"] for r in summary]
    x = np.arange(len(labels))

    fixed_over_plob = as_float(summary, "fixed_over_plob")
    adapt_over_plob = as_float(summary, "adapt_over_plob")

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, fixed_over_plob, "o-", label="Fixed / PLOB")
    plt.plot(x, adapt_over_plob, "s-", label="Adaptive / PLOB")
    plt.axhline(1.0, color="k", ls="--", lw=1, label="PLOB limit")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Normalized rate")
    plt.title("Comparison to PLOB Bound")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "plob_comparison.png"), dpi=200)
    plt.close()


def plot_boxili_convention(summary):
    row = [r for r in summary if r["label"] == "BoxiLi-base"]
    if not row:
        return
    r = row[0]
    vals = [
        float(r["skr_boxili_no"]),
        float(r["skr_boxili_fixed"]),
        float(r["skr_no"]),
        float(r["skr_fixed"]),
        float(r["skr_adapt"]),
    ]
    labels = ["BoxiLi no", "BoxiLi fixed", "MC no", "MC fixed", "MC adaptive"]

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(np.arange(len(vals)), vals, color=["#999", "#4a90d9", "#777", "#2b6cb0", "#d94a4a"])
    plt.xticks(np.arange(len(vals)), labels, rotation=15)
    plt.ylabel("SKR")
    plt.title("BoxiLi Convention vs MC Objective (BoxiLi-base)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxili_vs_mc_convention.png"), dpi=200)
    plt.close()


def plot_tomography():
    npz_path = os.path.join(OUT_DIR, "tomography_series.npz")
    if not os.path.exists(npz_path):
        return
    d = np.load(npz_path)

    # Running mean Werner
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(d["rw_no"], label="No cutoff")
    plt.plot(d["rw_fx"], label="Fixed")
    plt.plot(d["rw_ad"], label="Adaptive")
    plt.xlabel("Episode index")
    plt.ylabel("Running mean Werner")
    plt.title("Tomography-like Tracking: Running Werner Mean")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tomo_running_werner.png"), dpi=200)
    plt.close()

    # Running mean Fidelity
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(d["rf_no"], label="No cutoff")
    plt.plot(d["rf_fx"], label="Fixed")
    plt.plot(d["rf_ad"], label="Adaptive")
    plt.xlabel("Episode index")
    plt.ylabel("Running mean Fidelity")
    plt.title("Tomography-like Tracking: Running Fidelity Mean")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tomo_running_fidelity.png"), dpi=200)
    plt.close()

    # Running SKR
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(d["rskr_no"], label="No cutoff")
    plt.plot(d["rskr_fx"], label="Fixed")
    plt.plot(d["rskr_ad"], label="Adaptive")
    plt.xlabel("Episode index")
    plt.ylabel("Running SKR")
    plt.title("Tomography-like Tracking: Running SKR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tomo_running_skr.png"), dpi=200)
    plt.close()

    # Fidelity distributions
    plt.figure(figsize=(8.5, 4.8))
    bins = np.linspace(0.5, 1.0, 60)
    plt.hist(d["f_no"], bins=bins, density=True, alpha=0.4, label="No cutoff")
    plt.hist(d["f_fx"], bins=bins, density=True, alpha=0.4, label="Fixed")
    plt.hist(d["f_ad"], bins=bins, density=True, alpha=0.4, label="Adaptive")
    plt.xlabel("Delivered Fidelity")
    plt.ylabel("Density")
    plt.title("Delivered Fidelity Distribution (Tomography-like)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tomo_fidelity_hist.png"), dpi=200)
    plt.close()


def plot_delivery_quantiles(per_case):
    labels = sorted(set(r["label"] for r in per_case))
    strategies = ["no_cutoff", "fixed", "adaptive"]
    q = ["p50_delivery", "p90_delivery", "p99_delivery"]

    for label in labels:
        rows = [r for r in per_case if r["label"] == label]
        rows = {r["strategy"]: r for r in rows}

        vals = np.array([[float(rows[s][qq]) for qq in q] for s in strategies])
        x = np.arange(len(q))
        w = 0.25

        plt.figure(figsize=(7.5, 4.6))
        plt.bar(x - w, vals[0], width=w, label="No cutoff")
        plt.bar(x,     vals[1], width=w, label="Fixed")
        plt.bar(x + w, vals[2], width=w, label="Adaptive")
        plt.xticks(x, ["P50", "P90", "P99"])
        plt.ylabel("Delivery time")
        plt.title(f"Delivery Time Quantiles — {label}")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"quantiles_{label}.png"), dpi=200)
        plt.close()


def main():
    summary = read_csv(os.path.join(OUT_DIR, "summary.csv"))
    cutoff_rows = read_csv(os.path.join(OUT_DIR, "cutoff_scan.csv"))
    per_case = read_csv(os.path.join(OUT_DIR, "per_case_metrics.csv"))

    plot_cutoff_scans(summary, cutoff_rows)
    plot_strategy_comparison(summary)
    plot_plob_comparison(summary)
    plot_boxili_convention(summary)
    plot_tomography()
    plot_delivery_quantiles(per_case)

    print(f"✅ Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()