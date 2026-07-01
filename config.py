"""
Central parameter configuration for the quantum repeater MDP simulation.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class RepeaterConfig:
    p_gen: float = 0.01
    w0: float = 1.0
    p_swap: float = 1.0
    t_coh: float = 1000.0
    protocol: List[int] = field(default_factory=lambda: [0, 0])

    t_trunc: int = 10000
    gamma: float = 0.99
    a_max: int = 200
    epsilon: float = 1e-6

    n_runs: int = 100_000
    mc_max_steps: int = 10_000_000
    seed: int = 42

    @property
    def mean_gen_time(self) -> float:
        return 1.0 / self.p_gen

    @property
    def n_segments(self) -> int:
        n_swaps = sum(1 for op in self.protocol if op == 0)
        return 2 ** n_swaps

    @property
    def n_swap_levels(self) -> int:
        return sum(1 for op in self.protocol if op == 0)

    @property
    def state_space_size(self) -> int:
        return (2 * (self.a_max + 1)) ** 2

    def suggest_t_trunc(self) -> int:
        return max(int(10 / self.p_gen), int(10 * self.t_coh), 1000)

    def suggest_a_max(self) -> int:
        return min(int(5 * self.t_coh), self.t_trunc, 500)

    def validate(self) -> None:
        if not (0.0 < self.p_gen <= 1.0):
            raise ValueError(f"p_gen must be in (0,1], got {self.p_gen}")
        if not (0.0 < self.w0 <= 1.0):
            raise ValueError(f"w0 must be in (0,1], got {self.w0}")
        if not (0.0 < self.p_swap <= 1.0):
            raise ValueError(f"p_swap must be in (0,1], got {self.p_swap}")
        if not (self.t_coh > 0):
            raise ValueError(f"t_coh must be positive, got {self.t_coh}")
        if not all(op in (0, 1) for op in self.protocol):
            raise ValueError(f"protocol must contain only 0/1, got {self.protocol}")
        if len(self.protocol) == 0:
            raise ValueError("protocol must be non-empty")
        if not (self.t_trunc > 0):
            raise ValueError(f"t_trunc must be positive, got {self.t_trunc}")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f"gamma must be in (0,1), got {self.gamma}")
        if not (self.a_max > 0):
            raise ValueError(f"a_max must be positive, got {self.a_max}")
        if not (self.epsilon > 0):
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if not (self.n_runs > 0):
            raise ValueError(f"n_runs must be positive, got {self.n_runs}")

    def to_boxili_params(self, cutoff) -> dict:
        n_levels = len(self.protocol)
        if isinstance(cutoff, (int, float)):
            mt_cut = (int(cutoff),) * n_levels
        else:
            mt_cut = tuple(int(c) for c in cutoff)

        return {
            "p_gen": self.p_gen,
            "w0": self.w0,
            "p_swap": self.p_swap,
            "t_coh": self.t_coh,
            "protocol": tuple(self.protocol),
            "t_trunc": self.t_trunc,
            "mt_cut": mt_cut,
            "cut_type": "memory_time",
        }

    def __repr__(self) -> str:
        return (
            f"RepeaterConfig(p_gen={self.p_gen}, w0={self.w0}, p_swap={self.p_swap}, "
            f"t_coh={self.t_coh}, protocol={self.protocol})"
        )


def default_config() -> RepeaterConfig:
    c = RepeaterConfig()
    c.validate()
    return c


def fast_test_config() -> RepeaterConfig:
    c = RepeaterConfig(
        p_gen=0.1, w0=1.0, p_swap=1.0, t_coh=100.0,
        protocol=[0], t_trunc=500, a_max=50, n_runs=1000, mc_max_steps=100_000
    )
    c.validate()
    return c


if __name__ == "__main__":
    c = default_config()
    assert c.p_swap == 1.0
    p = c.to_boxili_params(10)
    assert "p_swap" in p
    assert p["mt_cut"] == (10,) * len(c.protocol)
    print("✅ config.py self-test passed")