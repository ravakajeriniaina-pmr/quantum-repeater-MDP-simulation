import numpy as np
from dataclasses import dataclass
from typing import Callable

from yaml import warnings

from src.mdp.actions import (
    WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL,
)
from src.physical.werner_utils import swap_output_werner as swap_werner


# ═════════════════════════════════════════════════════════════════════
# POLICIES
# ═════════════════════════════════════════════════════════════════════

class NoCutoffPolicy:
    def __call__(self, state: tuple) -> int:
        s1, a1, s2, a2 = state
        if s1 == 1 and s2 == 1:
            return SWAP
        return WAIT

    def __repr__(self):
        return "NoCutoffPolicy()"


class FixedCutoffPolicy:
    def __init__(self, cutoff: int):
        if cutoff < 1:
            raise ValueError(f"cutoff must be >= 1, got {cutoff}")
        self.cutoff = cutoff

    def __call__(self, state: tuple) -> int:
        s1, a1, s2, a2 = state

        if s1 == 1 and s2 == 1:
            both_ok = (a1 <= self.cutoff) and (a2 <= self.cutoff)
            if both_ok:
                return SWAP

            l1_old = a1 > self.cutoff
            l2_old = a2 > self.cutoff

            if l1_old and l2_old:
                return CUTOFF_ALL
            elif l1_old:
                return CUTOFF_1
            else:
                return CUTOFF_2

        elif s1 == 1 and s2 == 0:
            return CUTOFF_1 if a1 > self.cutoff else WAIT

        elif s1 == 0 and s2 == 1:
            return CUTOFF_2 if a2 > self.cutoff else WAIT

        return WAIT

    def __repr__(self):
        return f"FixedCutoffPolicy(cutoff={self.cutoff})"


class AdaptivePolicy:
    def __init__(self, policy_dict: dict, t_max: int):
        self.policy_dict = policy_dict
        self.t_max = t_max

    def __call__(self, state: tuple) -> int:
        s1, a1, s2, a2 = state

        if s1 == 0:
            a1 = 0
        else:
            a1 = min(a1, self.t_max)

        if s2 == 0:
            a2 = 0
        else:
            a2 = min(a2, self.t_max)

        compact = (s1, a1, s2, a2)

        if compact in self.policy_dict:
            return self.policy_dict[compact]

        #        Should never reach here if t_max is correctly set
            import warnings
            warnings.warn(
        f"AdaptivePolicy: state {compact} not in policy_dict "
        f"(t_max={self.t_max}). Falling back to default. "
        f"Consider increasing t_max.",
        RuntimeWarning, stacklevel=2
)
            if s1 == 1 and s2 == 1:
                return SWAP
        return WAIT

    def __repr__(self):
        return f"AdaptivePolicy(t_max={self.t_max})"


# ═════════════════════════════════════════════════════════════════════
# SINGLE EPISODE
# ═════════════════════════════════════════════════════════════════════

def run_episode(policy: Callable, p_gen: float, w0: float,
                t_coh: float, rng: np.random.Generator,
                p_swap: float = 1.0,
                max_steps: int = 10_000_000) -> tuple:
    """
    One episode until successful end-to-end pair delivery or timeout.

    Timing model (correct):
      1) Observe current state
      2) Choose action
      3) If SWAP: success with p_swap, else consume links and continue
      4) If not SWAP: apply WAIT/CUTOFF dynamics over one time step
    """
    s1, a1, s2, a2 = 0, 0, 0, 0
    t = 0

    for _ in range(max_steps):
        state = (s1, a1, s2, a2)
        action = policy(state)

        if action == SWAP:
            # SWAP on current ages (no extra aging before swap)
            if rng.random() < p_swap:
                w_out = swap_werner(a1, a2, w0, t_coh)
                return t, w_out
            # failed swap: both links consumed, restart
            s1, a1, s2, a2 = 0, 0, 0, 0
            continue

        # Apply cutoffs first
        if action == CUTOFF_1:
            s1, a1 = 0, 0
        elif action == CUTOFF_2:
            s2, a2 = 0, 0
        elif action == CUTOFF_ALL:
            s1, a1 = 0, 0
            s2, a2 = 0, 0
        elif action != WAIT:
            raise ValueError(f"Unknown action {action} at state {state}")

        # One time-step WAIT dynamics
        # link 1
        if s1 == 0:
            if rng.random() < p_gen:
                s1, a1 = 1, 0
            else:
                a1 += 1
        else:
            a1 += 1

        # link 2
        if s2 == 0:
            if rng.random() < p_gen:
                s2, a2 = 1, 0
            else:
                a2 += 1
        else:
            a2 += 1

        t += 1

    return max_steps, 0.0


# ═════════════════════════════════════════════════════════════════════
# BATCH SIMULATION
# ═════════════════════════════════════════════════════════════════════

@dataclass
class MCResult:
    delivery_times: np.ndarray
    w_out_array: np.ndarray
    n_episodes: int
    n_timeouts: int
    policy_name: str
    max_steps: int

    @property
    def mean_delivery_time(self) -> float:
        """
        Mean delivery time over ALL episodes, including timeouts.
        Timeouts count as max_steps, consistent with skr_from_samples().
        """
        return float(np.mean(self.delivery_times))

    @property
    def mean_delivery_time_no_timeout(self) -> float:
        """
        Mean delivery time excluding timed-out episodes.
        Use only for diagnostics, not for SKR computation.
        """
        valid = self.delivery_times[self.delivery_times < self.max_steps]
        if len(valid) == 0:
            return np.inf
        return float(np.mean(valid))

    @property
    def mean_werner(self) -> float:
        valid = self.w_out_array[self.delivery_times < self.max_steps]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(valid))

    @property
    def mean_fidelity(self) -> float:
        return (1.0 + 3.0 * self.mean_werner) / 4.0


def run_simulation(policy: Callable, p_gen: float, w0: float,
                   t_coh: float, n_episodes: int,
                   p_swap: float = 1.0,
                   max_steps: int = 10_000_000,
                   seed: int = 42,
                   verbose: bool = False) -> MCResult:
    rng = np.random.default_rng(seed)

    delivery_times = np.zeros(n_episodes, dtype=np.int64)
    w_out_array = np.zeros(n_episodes, dtype=np.float64)
    n_timeouts = 0

    policy_name = repr(policy)

    for i in range(n_episodes):
        dt, w_out = run_episode(
            policy, p_gen, w0, t_coh, rng,
            p_swap=p_swap, max_steps=max_steps
        )
        delivery_times[i] = dt
        w_out_array[i] = w_out
        if dt >= max_steps:
            n_timeouts += 1

    return MCResult(
        delivery_times=delivery_times,
        w_out_array=w_out_array,
        n_episodes=n_episodes,
        n_timeouts=n_timeouts,
        policy_name=policy_name,
        max_steps=max_steps,
    )


def compare_strategies(p_gen: float, w0: float, t_coh: float,
                       cutoff: int,
                       mdp_policy: dict = None,
                       mdp_t_max: int = None,
                       n_episodes: int = 50_000,
                       p_swap: float = 1.0,
                       max_steps: int = 10_000_000,
                       seed: int = 42,
                       verbose: bool = False) -> dict:
    results = {}
    results["no_cutoff"] = run_simulation(
        NoCutoffPolicy(), p_gen, w0, t_coh,
        n_episodes=n_episodes, p_swap=p_swap,
        max_steps=max_steps, seed=seed, verbose=verbose
    )
    results["fixed"] = run_simulation(
        FixedCutoffPolicy(cutoff), p_gen, w0, t_coh,
        n_episodes=n_episodes, p_swap=p_swap,
        max_steps=max_steps, seed=seed + 1, verbose=verbose
    )

    if mdp_policy is not None and mdp_t_max is not None:
        results["adaptive"] = run_simulation(
            AdaptivePolicy(mdp_policy, mdp_t_max), p_gen, w0, t_coh,
            n_episodes=n_episodes, p_swap=p_swap,
            max_steps=max_steps, seed=seed + 2, verbose=verbose
        )
    return results


if __name__ == "__main__":
    # smoke tests
    rng = np.random.default_rng(42)
    dt, w = run_episode(NoCutoffPolicy(), 1.0, 1.0, np.inf, rng, p_swap=1.0)
    assert dt == 1
    assert abs(w - 1.0) < 1e-12

    res1 = run_simulation(NoCutoffPolicy(), 0.1, 1.0, 50.0, 5000, p_swap=1.0, seed=1)
    res2 = run_simulation(NoCutoffPolicy(), 0.1, 1.0, 50.0, 5000, p_swap=0.5, seed=1)

    from src.metrics.skr import skr_from_samples
    skr1 = skr_from_samples(res1.delivery_times.astype(float), res1.w_out_array)
    skr2 = skr_from_samples(res2.delivery_times.astype(float), res2.w_out_array)
    assert skr2 <= skr1 + 1e-8
    print("✅ mc_engine.py self-test passed")