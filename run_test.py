"""
run_tests.py
Racine du projet MDP_simulation-V5/
Usage : python run_tests.py
"""

import subprocess
import sys
import time
import os

# ── S'assurer que la racine est dans sys.path ──
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ══════════════════════════════════════════════════════════
# MODULES À TESTER — dans l'ordre de dépendance
# ══════════════════════════════════════════════════════════

MODULES = [
    # ── Couche physique (aucune dépendance interne) ──
    ("src.physical.werner_utils",
     "Physical — Werner utilities"),

    # ── MDP core (dépend de werner_utils) ──
    ("src.mdp.actions",
     "MDP — Actions"),
    ("src.mdp.state_space",
     "MDP — State space"),
    ("src.mdp.rewards",
     "MDP — Rewards"),
    ("src.mdp.transitions",
     "MDP — Transitions"),
    ("src.mdp.value_iteration",
     "MDP — Value iteration"),

    # ── Métriques (dépend de werner_utils) ──
    ("src.metrics.skr",
     "Metrics — SKR"),
    ("src.metrics.statistics",
     "Metrics — Statistics"),

    # ── Simulation MC (dépend de tout le MDP) ──
    ("src.simulation.mc_engine",
     "Simulation — MC engine"),

    # ── Config (autonome) ──
    ("config",
     "Configuration"),
]

# ══════════════════════════════════════════════════════════
# MODULES OPTIONNELS (peuvent échouer sans bloquer)
# ══════════════════════════════════════════════════════════

OPTIONAL_MODULES = [
    ("src.analytical.no_cutoff",
     "Analytical — No-cutoff baseline"),
    ("src.analytical.boxili_wrapper",
     "Analytical — Boxi Li wrapper (needs external/boxili)"),
    ("src.metrics.plob",
     "Metrics — PLOB bound"),
    ("src.noise.decoherence",
     "Noise — Decoherence model"),
]


def run_module(module: str, description: str,
               optional: bool = False) -> tuple:
    """
    Lance le self-test d'un module.
    Retourne (ok: bool, elapsed: float, stdout: str, stderr: str)
    """
    print(f"\n{'─'*65}")
    tag = "[OPTIONAL]" if optional else "[REQUIRED]"
    print(f"  {tag} {description}")
    print(f"  Module : {module}")
    print(f"{'─'*65}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start
    ok = result.returncode == 0

    status = "✅ PASS" if ok else ("⚠️  SKIP" if optional else "❌ FAIL")
    print(f"\n  {status}  ({elapsed:.2f}s)")

    return ok, elapsed


def run_all():
    print("=" * 65)
    print("  QUANTUM REPEATER MDP — Full Test Suite")
    print("  Project: MDP_simulation-V5")
    print("=" * 65)

    results = []
    total_start = time.time()
    fatal_error = False

    # ── Tests obligatoires ──
    print("\n\n  ══ REQUIRED MODULES ══")
    for module, description in MODULES:
        ok, elapsed = run_module(module, description, optional=False)
        results.append((module, description, ok, elapsed, False))

        if not ok:
            print(f"\n  ⛔ FATAL: {module} failed.")
            print(f"  Fix this module before continuing.\n")
            fatal_error = True
            break

    # ── Tests optionnels (seulement si les obligatoires passent) ──
    if not fatal_error:
        print("\n\n  ══ OPTIONAL MODULES ══")
        for module, description in OPTIONAL_MODULES:
            ok, elapsed = run_module(module, description, optional=True)
            results.append((module, description, ok, elapsed, True))
            # Les optionnels ne bloquent pas

    # ── Résumé ──
    total_elapsed = time.time() - total_start

    print(f"\n\n{'=' * 65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 65}")

    required_all_pass = True
    for module, desc, ok, elapsed, optional in results:
        if optional:
            status = "✅" if ok else "⚠️ "
            tag = "(opt)"
        else:
            status = "✅" if ok else "❌"
            tag = "     "
            if not ok:
                required_all_pass = False

        short_desc = desc[:45]
        print(f"  {status} {tag} {short_desc:<45} {elapsed:>6.2f}s")

    print(f"{'─' * 65}")
    print(f"  Total: {total_elapsed:.2f}s")
    print(f"{'=' * 65}")

    if required_all_pass and not fatal_error:
        print("\n  ✅ ALL REQUIRED TESTS PASSED")
        print("  → You can now run: python Boxili_reproduction.py\n")
    else:
        print("\n  ❌ SOME REQUIRED TESTS FAILED")
        print("  → Check the output above and fix errors first.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_all()