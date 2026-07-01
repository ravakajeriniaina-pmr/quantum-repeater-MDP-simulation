import numpy as np
from src.physical.werner_utils import skr_threshold_fidelity as _f_thr

# ═════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE DE DÉCOHÉRENCE
# ═════════════════════════════════════════════════════════════════════

def decay_factor(delta_t: float, t_coh: float) -> float:
    # calcule la décohérence  (facteur de désintégration) pour un temps d'attente donné
    if delta_t == 0:
        return 1.0
    if np.isinf(t_coh):
        return 1.0
    return np.exp(-delta_t / t_coh)


def decay_factor_array(max_time: int, t_coh: float) -> np.ndarray:
    # génère un tableau de facteurs de décohérence pour tous les temps d'attente de 0 à max_time-1
    if np.isinf(t_coh):
        return np.ones(max_time)
    return np.exp(-np.arange(max_time, dtype=np.float64) / t_coh)


# ═════════════════════════════════════════════════════════════════════
# PARAMÈTRE DE WERNER APRÈS STOCKAGE
# ═════════════════════════════════════════════════════════════════════

def werner_after_storage(w0: float, storage_time: float,
                         t_coh: float) -> float:
    return w0 * decay_factor(storage_time, t_coh)


def werner_after_storage_array(w0: float, max_time: int,
                               t_coh: float) -> np.ndarray:
    return w0 * decay_factor_array(max_time, t_coh)


# ═════════════════════════════════════════════════════════════════════
# FIDELITÉ APRÈS STOCKAGE
# ═════════════════════════════════════════════════════════════════════

def fidelity_after_storage(w0: float, storage_time: float,
                           t_coh: float) -> float:
    w = werner_after_storage(w0, storage_time, t_coh)
    return (1.0 + 3.0 * w) / 4.0


def fidelity_after_storage_array(w0: float, max_time: int,
                                 t_coh: float) -> np.ndarray:
    w_arr = werner_after_storage_array(w0, max_time, t_coh)
    return (1.0 + 3.0 * w_arr) / 4.0


# ═════════════════════════════════════════════════════════════════════
# FONCTION QUI CALCULE LE TEMPS POUR ATTEINDRE UNE FIDÉLITÉ CIBLE
# ═════════════════════════════════════════════════════════════════════

def time_to_reach_fidelity(w0: float, target_fidelity: float,
                           t_coh: float) -> float:
    # F(t) = 1/4 + (3/4)*w0*exp(-t/t_coh)
    # Fidelité minimale possible est 0.25 (quand w=0), donc si la cible est en dessous, c'est impossible
    if target_fidelity <= 0.25:
        return np.inf

    # Si w0 = 0, on part déjà de 0.25 et on ne peut jamais descendre en dessous
    if w0 == 0:
        return np.inf

    # Si aucune décohérence, la fidélité ne diminue jamais
    if np.isinf(t_coh):
        return np.inf

    initial_fidelity = (1.0 + 3.0 * w0) / 4.0

    # Si la fidélité initiale est déjà en dessous de la cible, elle ne peut que diminuer
    # et donc on ne peut jamais atteindre la cible
    if initial_fidelity < target_fidelity:
        return np.inf

    # SI la fidélité initiale est déjà égale à la cible, le temps pour l'atteindre est 0
    if initial_fidelity == target_fidelity:
        return 0.0

    # Résolution: (1 + 3*w0*exp(-t/t_coh))/4 = target_fidelity
    # => exp(-t/t_coh) = (target_fidelity - 0.25) / (0.75 * w0)
    # => t = -t_coh * ln(...)
    ratio = (target_fidelity - 0.25) / (0.75 * w0)

    return -t_coh * np.log(ratio)


def time_to_skr_threshold(w0: float, t_coh: float) -> float:
    """
    Time for a single link to reach the SKR threshold fidelity.
    Uses skr_threshold_fidelity() from werner_utils for consistency.
    """
    return time_to_reach_fidelity(w0, _f_thr(), t_coh)


# AJOUTER à la fin du fichier :

if __name__ == "__main__":
    print("=" * 60)
    print("Decoherence Self-Test")
    print("=" * 60)

    # 1. decay_factor
    print("\n1. decay_factor:")
    assert decay_factor(0, 100) == 1.0
    assert decay_factor(100, 100) == np.exp(-1.0)
    assert decay_factor(0, np.inf) == 1.0
    assert decay_factor(500, np.inf) == 1.0
    print("   All basic tests passed  ✓")

    # 2. decay_factor_array
    print("\n2. decay_factor_array:")
    arr = decay_factor_array(5, 100.0)
    assert arr[0] == 1.0
    assert abs(arr[1] - np.exp(-1/100)) < 1e-12
    assert len(arr) == 5
    print(f"   arr[0..4] = {arr[:5].round(4)}  ✓")

    # 3. werner_after_storage
    print("\n3. werner_after_storage:")
    w = werner_after_storage(1.0, 0, 100)
    assert w == 1.0
    w = werner_after_storage(1.0, 100, 100)
    assert abs(w - np.exp(-1.0)) < 1e-12
    print("   Basic tests passed  ✓")

    # 4. time_to_reach_fidelity
    print("\n4. time_to_reach_fidelity:")

    # Roundtrip: compute time, then verify fidelity at that time
    w0, t_coh = 1.0, 100.0
    target_F = 0.85
    t_star = time_to_reach_fidelity(w0, target_F, t_coh)
    F_at_tstar = fidelity_after_storage(w0, t_star, t_coh)
    assert abs(F_at_tstar - target_F) < 1e-10
    print(f"   Roundtrip: target={target_F}, "
          f"t*={t_star:.2f}, F(t*)={F_at_tstar:.6f}  ✓")

    # Edge cases
    assert time_to_reach_fidelity(1.0, 0.2, 100) == np.inf
    assert time_to_reach_fidelity(1.0, 0.85, np.inf) == np.inf
    assert time_to_reach_fidelity(1.0, 1.0, 100) == 0.0
    print("   Edge cases passed  ✓")

    # 5. time_to_skr_threshold
    print("\n5. time_to_skr_threshold:")
    t_skr = time_to_skr_threshold(1.0, 100.0)
    F_at_threshold = fidelity_after_storage(1.0, t_skr, 100.0)
    assert abs(F_at_threshold - _f_thr()) < 1e-10   # roundtrip exact
    print(f"   t_skr={t_skr:.2f}, F(t_skr)={F_at_threshold:.6f} "
      f"(SKR threshold = {_f_thr():.6f})  ✓")
    print(f"   t_skr={t_skr:.2f}, "
          f"F(t_skr)={F_at_threshold:.4f} ≈ 0.8107  ✓")

    print("\n✅ decoherence.py self-test passed")