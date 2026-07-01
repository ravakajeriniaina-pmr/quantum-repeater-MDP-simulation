import numpy as np


# ═════════════════════════════════════════════════════════════════════
# STATE SPACE CONSTRUCTION
# ═══════════════════════��═════════════════════════════════════════════

class StateSpace:
    def __init__(self, t_max: int):
        if t_max < 1:
            raise ValueError(f"t_max must be >= 1, got {t_max}")

        self.t_max = t_max
        self.states = []
        self.state_to_index = {}

        self._build()

    def _build(self):
        """Enumerate all valid compact states."""
        states = []

        # Category 1: Both pending → (0, 0, 0, 0)
        # Only ONE state (pending ages collapsed to 0)
        states.append((0, 0, 0, 0))

        # Category 2: Link 1 entangled, link 2 pending
        # (1, a1, 0, 0) for a1 = 0, 1, ..., t_max
        for a1 in range(self.t_max + 1):
            states.append((1, a1, 0, 0))

        # Category 3: Link 1 pending, link 2 entangled
        # (0, 0, 1, a2) for a2 = 0, 1, ..., t_max
        for a2 in range(self.t_max + 1):
            states.append((0, 0, 1, a2))

        # Category 4: Both entangled
        # (1, a1, 1, a2) for a1, a2 = 0, ..., t_max
        for a1 in range(self.t_max + 1):
            for a2 in range(self.t_max + 1):
                states.append((1, a1, 1, a2))

        self.states = states
        self.n_states = len(states)
        self.state_to_index = {s: i for i, s in enumerate(states)}

    # ═════════════════════════════════════════════════════════════
    # LOOKUP
    # ═════════════════════════════════════════════════════════════

    def index(self, state: tuple) -> int:
        return self.state_to_index[state]

    def state(self, idx: int) -> tuple:
        return self.states[idx]

    def contains(self, state: tuple) -> bool:
        return state in self.state_to_index

    # ═════════════════════════════════════════════════════════════
    # COMPACT CONVERSION
    # ═════════════════════════════════════════════════════════════

    def to_compact(self, state: tuple) -> tuple:
        s1, a1, s2, a2 = state

        # Collapse pending ages to 0
        if s1 == 0:
            a1 = 0
        else:
            a1 = min(a1, self.t_max)

        if s2 == 0:
            a2 = 0
        else:
            a2 = min(a2, self.t_max)

        return (s1, a1, s2, a2)

    def to_compact_index(self, state: tuple) -> int:
        return self.index(self.to_compact(state))

    # ═════════════════════════════════════════════════════════════
    # STATE CATEGORIES
    # ═════════════════════════════════════════════════════════════

    def is_both_pending(self, state: tuple) -> bool:
        """Check if both links are pending."""
        return state[0] == 0 and state[2] == 0

    def is_both_entangled(self, state: tuple) -> bool:
        """Check if both links are entangled."""
        return state[0] == 1 and state[2] == 1

    def is_one_entangled(self, state: tuple) -> bool:
        """Check if exactly one link is entangled."""
        return (state[0] + state[2]) == 1

    def get_category(self, state: tuple) -> str:
        if self.is_both_pending(state):
            return "BOTH_PENDING"
        elif self.is_both_entangled(state):
            return "BOTH_ENTANGLED"
        else:
            return "ONE_ENTANGLED"

    # ═════════════════════════════════════════════════════════════
    # CATEGORY ITERATION
    # ═════════════════════════════════════════════════════════════

    def both_pending_states(self) -> list:
        return [(0, 0, 0, 0)]

    def one_entangled_states(self) -> list:
        result = []
        for a in range(self.t_max + 1):
            result.append((1, a, 0, 0))
        for a in range(self.t_max + 1):
            result.append((0, 0, 1, a))
        return result

    def both_entangled_states(self) -> list:
        result = []
        for a1 in range(self.t_max + 1):
            for a2 in range(self.t_max + 1):
                result.append((1, a1, 1, a2))
        return result

    # ═════════════════════════════════════════════════════════════
    # SIZE ANALYTICS
    # ═════════════════════════════════════════════════════════════

    def count_by_category(self) -> dict:
        n_bp = 1
        n_oe = 2 * (self.t_max + 1)
        n_be = (self.t_max + 1) ** 2
        total = n_bp + n_oe + n_be

        return {
            "BOTH_PENDING": n_bp,
            "ONE_ENTANGLED": n_oe,
            "BOTH_ENTANGLED": n_be,
            "TOTAL": total,
        }

    def memory_estimate_mb(self) -> float:
        value_bytes = self.n_states * 8  # float64
        policy_bytes = self.n_states * 4  # int32
        return (value_bytes + policy_bytes) / 1e6

    # ═════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════���══════════════

    def __repr__(self) -> str:
        return (f"StateSpace(t_max={self.t_max}, "
                f"n_states={self.n_states})")

    def print_summary(self) -> None:
        """Print a summary of the state space."""
        counts = self.count_by_category()
        print(f"StateSpace Summary:")
        print(f"  t_max = {self.t_max}")
        print(f"  Total states = {self.n_states}")
        print(f"    BOTH_PENDING:   {counts['BOTH_PENDING']:>8}")
        print(f"    ONE_ENTANGLED:  {counts['ONE_ENTANGLED']:>8}")
        print(f"    BOTH_ENTANGLED: {counts['BOTH_ENTANGLED']:>8}")
        print(f"  Memory estimate: {self.memory_estimate_mb():.3f} MB")


# ═════════════════════════════════════════════════════════════════════
# HELPER: compute t_max from physical parameters
# ═════════════════════════════════════════════════════════════════════

def suggest_t_max(t_coh: float, w0: float = 1.0,
                  safety_factor: float = 3.0) -> int:
    if np.isinf(t_coh):
        return 10000

    return max(int(np.ceil(safety_factor * t_coh)), 10)


def state_space_size(t_max: int) -> int:
    m = t_max + 1
    return 1 + 2 * m + m * m


# ═════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("StateSpace Self-Test")
    print("=" * 60)

    # ── Test construction ────────────────────────────────────────
    print("\n1. Construction tests:")

    ss = StateSpace(t_max=3)
    print(f"   {ss}")

    # Size formula: 1 + 2*(t_max+1) + (t_max+1)^2
    # t_max=3: 1 + 2*4 + 16 = 25
    expected_n = 1 + 2 * 4 + 4 * 4
    assert ss.n_states == expected_n, \
        f"Expected {expected_n} states, got {ss.n_states}"
    print(f"   n_states = {ss.n_states}  (expected {expected_n})  ✓")

    # Verify against formula
    assert ss.n_states == state_space_size(3)
    print(f"   Matches state_space_size()  ✓")

    # ── Test state enumeration ───────────────────────────────────
    print("\n2. State enumeration:")

    # First state is always (0,0,0,0)
    assert ss.states[0] == (0, 0, 0, 0)
    print(f"   states[0] = {ss.states[0]}  ✓")

    # All states should be unique
    assert len(set(ss.states)) == ss.n_states
    print(f"   All states unique  ✓")

    # Index roundtrip
    for i, s in enumerate(ss.states):
        assert ss.index(s) == i
        assert ss.state(i) == s
    print(f"   Index ↔ state roundtrip for all {ss.n_states} states  ✓")

    # ── Test contains ────────────────────────────────────────────
    print("\n3. Contains tests:")

    assert ss.contains((0, 0, 0, 0))
    assert ss.contains((1, 0, 0, 0))
    assert ss.contains((1, 3, 0, 0))  # t_max = 3
    assert ss.contains((1, 3, 1, 3))
    print(f"   Valid states found  ✓")

    assert not ss.contains((1, 4, 0, 0))  # age > t_max
    assert not ss.contains((0, 1, 0, 0))  # pending with nonzero age
    assert not ss.contains((2, 0, 0, 0))  # invalid status
    print(f"   Invalid states rejected  ✓")

    # ── Test compact conversion ──────────────────────────────────
    print("\n4. Compact conversion:")

    ss100 = StateSpace(t_max=100)

    # Pending ages collapse to 0
    assert ss100.to_compact((0, 57, 0, 23)) == (0, 0, 0, 0)
    print(f"   (0,57,0,23) → (0,0,0,0)  ✓  (pending ages collapse)")

    # Entangled ages pass through
    assert ss100.to_compact((1, 50, 0, 99)) == (1, 50, 0, 0)
    print(f"   (1,50,0,99) → (1,50,0,0)  ✓  (pending age collapses)")

    # Ages clamp to t_max
    assert ss100.to_compact((1, 999, 1, 30)) == (1, 100, 1, 30)
    print(f"   (1,999,1,30) → (1,100,1,30)  ✓  (age clamped to t_max)")

    # Already compact → unchanged
    assert ss100.to_compact((1, 50, 1, 30)) == (1, 50, 1, 30)
    print(f"   (1,50,1,30) → unchanged  ✓")

    # Compact index roundtrip
    idx = ss100.to_compact_index((0, 42, 1, 50))
    assert ss100.state(idx) == (0, 0, 1, 50)
    print(f"   to_compact_index roundtrip  ✓")

    # ── Test categories ──────────────────────────────────────────
    print("\n5. State categories:")

    assert ss.get_category((0, 0, 0, 0)) == "BOTH_PENDING"
    assert ss.get_category((1, 2, 0, 0)) == "ONE_ENTANGLED"
    assert ss.get_category((0, 0, 1, 1)) == "ONE_ENTANGLED"
    assert ss.get_category((1, 1, 1, 2)) == "BOTH_ENTANGLED"
    print(f"   Categories correct  ✓")

    assert ss.is_both_pending((0, 0, 0, 0))
    assert not ss.is_both_pending((1, 0, 0, 0))
    assert ss.is_one_entangled((1, 0, 0, 0))
    assert ss.is_one_entangled((0, 0, 1, 0))
    assert not ss.is_one_entangled((1, 0, 1, 0))
    assert ss.is_both_entangled((1, 0, 1, 0))
    assert not ss.is_both_entangled((0, 0, 1, 0))
    print(f"   Category queries correct  ✓")

    # ── Test category counts ─────────────────────────────────────
    print("\n6. Category counts:")

    counts = ss.count_by_category()
    print(f"   t_max = {ss.t_max}:")
    print(f"     BOTH_PENDING:   {counts['BOTH_PENDING']}")
    print(f"     ONE_ENTANGLED:  {counts['ONE_ENTANGLED']}")
    print(f"     BOTH_ENTANGLED: {counts['BOTH_ENTANGLED']}")
    print(f"     TOTAL:          {counts['TOTAL']}")

    assert counts['BOTH_PENDING'] == 1
    assert counts['ONE_ENTANGLED'] == 2 * (ss.t_max + 1)
    assert counts['BOTH_ENTANGLED'] == (ss.t_max + 1) ** 2
    assert counts['TOTAL'] == ss.n_states
    print(f"   Counts match formula  ✓")

    # Verify by iterating
    bp_list = ss.both_pending_states()
    oe_list = ss.one_entangled_states()
    be_list = ss.both_entangled_states()
    assert len(bp_list) == counts['BOTH_PENDING']
    assert len(oe_list) == counts['ONE_ENTANGLED']
    assert len(be_list) == counts['BOTH_ENTANGLED']
    assert len(bp_list) + len(oe_list) + len(be_list) == ss.n_states
    print(f"   Category iteration matches counts  ✓")

    # ── Test category lists content ──────────────────────────────
    print("\n7. Category list content:")

    # All both_pending states
    for s in bp_list:
        assert ss.is_both_pending(s)
    print(f"   Both pending list correct  ✓")

    # All one_entangled states
    for s in oe_list:
        assert ss.is_one_entangled(s)
    print(f"   One entangled list correct  ✓")

    # All both_entangled states
    for s in be_list:
        assert ss.is_both_entangled(s)
    print(f"   Both entangled list correct  ✓")

    # No overlaps
    all_sets = set(bp_list) | set(oe_list) | set(be_list)
    assert len(all_sets) == ss.n_states
    print(f"   No overlaps between categories  ✓")

    # ── Test state_space_size formula ────────────────────────────
    print("\n8. State space size formula:")

    for t in [1, 3, 10, 50, 100, 500, 1000]:
        formula = state_space_size(t)
        actual = StateSpace(t).n_states if t <= 100 else formula
        if t <= 100:
            assert formula == actual
        m = t + 1
        expected = 1 + 2 * m + m * m
        assert formula == expected
        print(f"   t_max={t:>5}: n_states = {formula:>10,}")

    # ── Test suggest_t_max ───────────────────────────────────────
    print("\n9. suggest_t_max tests:")

    assert suggest_t_max(100) == 300
    print(f"   t_coh=100 → t_max={suggest_t_max(100)}  ✓")

    assert suggest_t_max(1000) == 3000
    print(f"   t_coh=1000 → t_max={suggest_t_max(1000)}  ✓")

    assert suggest_t_max(np.inf) == 10000
    print(f"   t_coh=inf → t_max={suggest_t_max(np.inf)}  ✓")

    assert suggest_t_max(1) == 10  # minimum
    print(f"   t_coh=1 → t_max={suggest_t_max(1)}  ✓  (minimum 10)")

    # ── Test error handling ──────────────────────────────────────
    print("\n10. Error handling:")

    try:
        StateSpace(t_max=0)
        print(f"   t_max=0: should have raised  ✗")
    except ValueError:
        print(f"   t_max=0: ValueError raised  ✓")

    try:
        ss.index((9, 9, 9, 9))
        print(f"   Invalid state index: should have raised  ✗")
    except KeyError:
        print(f"   Invalid state index: KeyError raised  ✓")

    # ── Test memory estimate ─────────────────────────────────────
    print("\n11. Memory estimates:")

    for t in [100, 500, 1000, 3000, 10000]:
        ss_test = StateSpace(t) if t <= 1000 else None
        n = state_space_size(t)
        mem_mb = n * 12 / 1e6  # 8 bytes value + 4 bytes policy
        print(f"   t_max={t:>5}: {n:>12,} states, "
              f"~{mem_mb:>8.3f} MB")

    # ── Print summary ────────────────────────────────────────────
    print("\n12. Summary for typical parameters:")

    for t_coh in [100, 400, 1000]:
        t_max = suggest_t_max(t_coh)
        n = state_space_size(t_max)
        print(f"\n   t_coh = {t_coh}:")
        print(f"     t_max = {t_max}")
        print(f"     n_states = {n:,}")
        print(f"     Both-entangled states = {(t_max+1)**2:,} "
              f"({(t_max+1)**2/n*100:.1f}%)")

    # ── Test compactness matters ─────────────────────────────────
    print("\n13. Why compact representation matters:")

    t_max = 100
    # Full (naive): pending ages 0..t_max too
    # n_full = (2 * (t_max+1))^2 = 4 * (t_max+1)^2
    n_full = (2 * (t_max + 1)) ** 2
    n_compact = state_space_size(t_max)
    ratio = n_full / n_compact
    print(f"   t_max = {t_max}:")
    print(f"     Full (naive):  {n_full:>10,} states")
    print(f"     Compact:       {n_compact:>10,} states")
    print(f"     Reduction:     {ratio:.1f}x")

    print("\n✅ state_space.py self-test passed")