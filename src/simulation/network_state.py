import numpy as np
from src.physical.werner_utils import swap_output_werner as _swap_output_werner


class NetworkState:
    def __init__(self):
        """Initialize to all-pending state: (0, 0, 0, 0)."""
        self.status = [0, 0]
        self.age = [0, 0]

    # ═════════════════════════════════════════════════════════════
    # STATE ACCESS
    # ═════════════════════════════════════════════════════════════

    def to_tuple(self) -> tuple:
        return (self.status[0], self.age[0],
                self.status[1], self.age[1])

    def copy(self) -> "NetworkState":
        new = NetworkState()
        new.status = self.status.copy()
        new.age = self.age.copy()
        return new

    # ═════════════════════════════════════════════════════════════
    # STATE QUERIES
    # ═════════════════════════════════════════════════════════════

    def both_entangled(self) -> bool:
        return self.status[0] == 1 and self.status[1] == 1

    def any_entangled(self) -> bool:
        return self.status[0] == 1 or self.status[1] == 1

    def both_pending(self) -> bool:
        return self.status[0] == 0 and self.status[1] == 0

    def is_link_entangled(self, link: int) -> bool:
        return self.status[link] == 1

    def is_link_pending(self, link: int) -> bool:
        return self.status[link] == 0

    def get_link_age(self, link: int) -> int:
        return self.age[link]

    def get_max_age(self) -> int:
        max_a = 0
        for i in range(2):
            if self.status[i] == 1:
                max_a = max(max_a, self.age[i])
        return max_a

    def get_age_difference(self) -> int:
        
        return abs(self.age[0] - self.age[1])
    def get_sum_age(self) -> int:
        """Sum of ages of entangled links. Relevant for swap Werner."""
        total = 0
        for i in range(2):
            if self.status[i] == 1:
                total += self.age[i]
        return total

    # ═════════════════════════════════════════════════════════════
    # STATE MUTATIONS
    # ═════════════════════════════════════════════════════════════

    def reset(self) -> None:
        self.status = [0, 0]
        self.age = [0, 0]

    def generation_success(self, link: int) -> None:
        self.status[link] = 1
        self.age[link] = 0

    def generation_failure(self, link: int) -> None:
        self.age[link] += 1

    def age_link(self, link: int) -> None:
        self.age[link] += 1

    def cutoff_link(self, link: int) -> None:
        self.status[link] = 0
        self.age[link] = 0

    def cutoff_all(self) -> None:
        self.status = [0, 0]
        self.age = [0, 0]

    # ═════════════════════════════════════════════════════════════
    # PHYSICS COMPUTATIONS
    # ═════════════════════════════════════════════════════════════

    def get_werner(self, link: int, w0: float, t_coh: float) -> float:
        if self.status[link] == 0:
            return 0.0
        if np.isinf(t_coh):
            return w0
        return w0 * np.exp(-self.age[link] / t_coh)

    def get_fidelity(self, link: int, w0: float, t_coh: float) -> float:
        w = self.get_werner(link, w0, t_coh)
        return (1.0 + 3.0 * w) / 4.0

    def get_swap_werner(self, w0: float, t_coh: float) -> float:
        """
        Werner after swap: w_out = w0^2 * exp(-(a1 + a2) / t_coh)
        Both links must be entangled.
        """
        if np.isinf(t_coh):
            return w0 * w0
        return _swap_output_werner(self.age[0], self.age[1], w0, t_coh)

    def get_swap_fidelity(self, w0: float, t_coh: float) -> float:
        w_out = self.get_swap_werner(w0, t_coh)
        return (1.0 + 3.0 * w_out) / 4.0

    # ═════════════════════════════════════════════════════════════
    # DISPLAY
    # ═════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        s1 = "ENT" if self.status[0] == 1 else "PND"
        s2 = "ENT" if self.status[1] == 1 else "PND"
        return (f"NetworkState(L1={s1} age={self.age[0]}, "
                f"L2={s2} age={self.age[1]})")

    def __eq__(self, other) -> bool:
        if not isinstance(other, NetworkState):
            return False
        return self.to_tuple() == other.to_tuple()


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NetworkState Self-Test")
    print("=" * 60)

    # ── Test initial state ───────────────────────────────────────
    print("\n1. Initial state:")

    s = NetworkState()
    assert s.to_tuple() == (0, 0, 0, 0)
    assert s.both_pending()
    assert not s.any_entangled()
    assert not s.both_entangled()
    print(f"   {s}  ✓")
    print(f"   to_tuple() = {s.to_tuple()}  ✓")
    print(f"   both_pending = {s.both_pending()}  ✓")

    # ── Test generation success ──────────────────────────────────
    print("\n2. Generation success:")

    s = NetworkState()
    s.generation_success(0)
    assert s.to_tuple() == (1, 0, 0, 0)
    assert s.is_link_entangled(0)
    assert s.is_link_pending(1)
    assert s.any_entangled()
    assert not s.both_entangled()
    print(f"   Link 0 success: {s}  ✓")

    s.generation_success(1)
    assert s.to_tuple() == (1, 0, 1, 0)
    assert s.both_entangled()
    assert not s.both_pending()
    print(f"   Both success:   {s}  ✓")

    # ── Test generation failure ──────────────────────────────────
    print("\n3. Generation failure:")

    s = NetworkState()
    s.generation_failure(0)
    assert s.to_tuple() == (0, 1, 0, 0)
    print(f"   1 failure: {s}  ✓")

    s.generation_failure(0)
    s.generation_failure(0)
    assert s.to_tuple() == (0, 3, 0, 0)
    print(f"   3 failures: {s}  ✓")

    # ── Test aging ───────────────────────────────────────────────
    print("\n4. Link aging:")

    s = NetworkState()
    s.generation_success(0)
    assert s.age[0] == 0
    s.age_link(0)
    assert s.age[0] == 1
    s.age_link(0)
    s.age_link(0)
    assert s.age[0] == 3
    assert s.to_tuple() == (1, 3, 0, 0)
    print(f"   3 aging steps: {s}  ✓")

    # ── Test cutoff_link ─────────────────────────────────────────
    print("\n5. Cutoff single link:")

    s = NetworkState()
    s.status = [1, 1]
    s.age = [100, 20]
    print(f"   Before: {s}")

    s.cutoff_link(0)
    assert s.to_tuple() == (0, 0, 1, 20)
    assert s.is_link_pending(0)
    assert s.is_link_entangled(1)
    assert s.age[1] == 20  # link 2 unchanged
    print(f"   Cut link 0: {s}  ✓")

    s.cutoff_link(1)
    assert s.to_tuple() == (0, 0, 0, 0)
    print(f"   Cut link 1: {s}  ✓")

    # ── Test cutoff_all ──────────────────────────────────────────
    print("\n6. Cutoff all:")

    s = NetworkState()
    s.status = [1, 1]
    s.age = [150, 80]
    print(f"   Before:    {s}")

    s.cutoff_all()
    assert s.to_tuple() == (0, 0, 0, 0)
    print(f"   After:     {s}  ✓")

    # ── Test reset ───────────────────────────────────────────────
    print("\n7. Reset:")

    s = NetworkState()
    s.status = [1, 0]
    s.age = [42, 7]
    s.reset()
    assert s.to_tuple() == (0, 0, 0, 0)
    print(f"   Reset: {s}  ✓")

    # ── Test age queries ─────────────────────────────────────────
    print("\n8. Age queries:")

    s = NetworkState()
    s.status = [1, 1]
    s.age = [30, 80]

    assert s.get_link_age(0) == 30
    assert s.get_link_age(1) == 80
    assert s.get_max_age() == 80
    assert s.get_age_difference() == 50
    print(f"   ages=(30,80): max={s.get_max_age()}, "
          f"diff={s.get_age_difference()}  ✓")

    # Max age when no links entangled
    s.reset()
    assert s.get_max_age() == 0
    print(f"   No entangled: max_age=0  ✓")

    # ── Test Werner parameter ────────────────────────────────────
    print("\n9. Werner parameter:")

    s = NetworkState()

    # Pending link → w = 0
    assert s.get_werner(0, 1.0, 1000) == 0.0
    print(f"   Pending link: w = {s.get_werner(0, 1.0, 1000)}  ✓")

    # Fresh entangled → w = w0
    s.generation_success(0)
    assert s.get_werner(0, 1.0, 1000) == 1.0
    assert s.get_werner(0, 0.9, 1000) == 0.9
    print(f"   Fresh (w0=1.0): w = {s.get_werner(0, 1.0, 1000)}  ✓")
    print(f"   Fresh (w0=0.9): w = {s.get_werner(0, 0.9, 1000)}  ✓")

    # After aging
    for _ in range(100):
        s.age_link(0)
    w_aged = s.get_werner(0, 1.0, 1000)
    expected = np.exp(-100.0 / 1000.0)
    assert abs(w_aged - expected) < 1e-10
    print(f"   age=100, t_coh=1000: w = {w_aged:.6f} "
          f"(expected {expected:.6f})  ✓")

    # Infinite coherence
    assert s.get_werner(0, 1.0, np.inf) == 1.0
    print(f"   t_coh=inf: w = {s.get_werner(0, 1.0, np.inf)}  ✓")

    # ── Test fidelity ────────────────────────────────────────────
    print("\n10. Fidelity:")

    s = NetworkState()
    # Pending → F = 0.25
    assert s.get_fidelity(0, 1.0, 1000) == 0.25
    print(f"   Pending: F = {s.get_fidelity(0, 1.0, 1000)}  ✓")

    # Fresh → F = 1.0
    s.generation_success(0)
    assert s.get_fidelity(0, 1.0, 1000) == 1.0
    print(f"   Fresh: F = {s.get_fidelity(0, 1.0, 1000)}  ✓")

    # ── Test swap Werner ─────────────────────────────────────────
    print("\n11. Swap Werner parameter:")

    # Both fresh
    s = NetworkState()
    s.generation_success(0)
    s.generation_success(1)
    assert s.get_swap_werner(1.0, 1000) == 1.0
    print(f"   Both fresh: w_out = {s.get_swap_werner(1.0, 1000)}  ✓")

    # Asymmetric ages
    s.age = [0, 200]
    w_out = s.get_swap_werner(1.0, 1000)
    expected = np.exp(-(0 + 200) / 1000)   # ← CORRECT = exp(-0.2)
    assert abs(w_out - expected) < 1e-10
    assert abs(w_out - expected) < 1e-10
    print(f"   ages=(0,200): w_out = {w_out:.6f} "
          f"(expected {expected:.6f})  ✓")

    # Symmetric ages: should be better than asymmetric
    s.age = [100, 100]
    w_sym = s.get_swap_werner(1.0, 1000)
    s.age = [0, 200]
    w_asym = s.get_swap_werner(1.0, 1000)
    assert abs(w_sym - w_asym) < 1e-10
    print(f"   Symmetric (100,100)={w_sym:.4f} > "
          f"Asymmetric (0,200)={w_asym:.4f}  ✓")

    # Infinite coherence
    s.age = [500, 500]
    assert s.get_swap_werner(1.0, np.inf) == 1.0
    print(f"   t_coh=inf: w_out = {s.get_swap_werner(1.0, np.inf)}  ✓")

    # w0 < 1
    s.age = [0, 0]
    w_imperfect = s.get_swap_werner(0.9, 1000)
    assert abs(w_imperfect - 0.81) < 1e-10  # 0.9^2
    print(f"   w0=0.9, both fresh: w_out = {w_imperfect}  ✓  (= 0.9²)")

    # ── Test swap fidelity ───────────────────────────────────────
    print("\n12. Swap fidelity:")

    s = NetworkState()
    s.generation_success(0)
    s.generation_success(1)
    assert s.get_swap_fidelity(1.0, 1000) == 1.0
    print(f"   Both fresh: F_out = {s.get_swap_fidelity(1.0, 1000)}  ✓")

    s.age = [0, 200]
    f_out = s.get_swap_fidelity(1.0, 1000)
    w_out = s.get_swap_werner(1.0, 1000)
    expected_f = (1.0 + 3.0 * w_out) / 4.0
    assert abs(f_out - expected_f) < 1e-10
    print(f"   ages=(0,200): F_out = {f_out:.6f}  ✓")

    # ── Test copy independence ───────────────────────────────────
    print("\n13. Copy independence:")

    s = NetworkState()
    s.status = [1, 0]
    s.age = [50, 10]
    s_copy = s.copy()

    assert s == s_copy
    assert s is not s_copy
    print(f"   Copy equals original  ✓")

    s_copy.cutoff_all()
    assert s.to_tuple() == (1, 50, 0, 10)  # original unchanged
    assert s_copy.to_tuple() == (0, 0, 0, 0)
    print(f"   Modifying copy doesn't affect original  ✓")

    # ── Test __eq__ ──────────────────────────────────────────────
    print("\n14. Equality:")

    s1 = NetworkState()
    s2 = NetworkState()
    assert s1 == s2
    print(f"   Two fresh states are equal  ✓")

    s1.generation_success(0)
    assert s1 != s2
    print(f"   After mutation, not equal  ✓")

    assert s1 != "not a state"
    print(f"   Not equal to non-NetworkState  ✓")

    # ── Test repr ──────────��─────────────────────────────────────
    print("\n15. String representation:")

    s = NetworkState()
    print(f"   Fresh:        {s}")
    s.generation_success(0)
    print(f"   L1 entangled: {s}")
    s.generation_success(1)
    s.age = [42, 7]
    print(f"   Both, aged:   {s}")

    # ── Simulate a mini episode ──────────────────────────────────
    print("\n16. Mini episode simulation:")

    s = NetworkState()
    rng = np.random.default_rng(42)
    p_gen = 0.1
    w0 = 1.0
    t_coh = 100.0
    t = 0

    print(f"   t={t}: {s}")
    while not s.both_entangled():
        for link in range(2):
            if s.is_link_pending(link):
                if rng.random() < p_gen:
                    s.generation_success(link)
                else:
                    s.generation_failure(link)
            else:
                s.age_link(link)
        t += 1
        if t <= 5 or s.both_entangled():
            w_info = ""
            for link in range(2):
                if s.is_link_entangled(link):
                    w = s.get_werner(link, w0, t_coh)
                    w_info += f" w{link+1}={w:.3f}"
            print(f"   t={t}: {s}{w_info}")

    if t > 5:
        print(f"   ... (skipped t=6 to t={t-1})")

    w_out = s.get_swap_werner(w0, t_coh)
    f_out = s.get_swap_fidelity(w0, t_coh)
    print(f"   SWAP: w_out={w_out:.6f}, F_out={f_out:.6f}")
    print(f"   Delivery time: {t} steps")

    print("\n✅ network_state.py self-test passed")