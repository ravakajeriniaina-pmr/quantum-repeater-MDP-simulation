"""
Simulation module for the quantum repeater.

Provides Monte Carlo simulation engine and policy classes.

Quick start:
    from src.simulation import run_simulation, FixedCutoffPolicy

    result = run_simulation(
        FixedCutoffPolicy(cutoff=20),
        p_gen=0.1, w0=1.0, t_coh=100.0,
        n_episodes=50_000)
"""

from src.simulation.mc_engine import (
    NoCutoffPolicy,
    FixedCutoffPolicy,
    AdaptivePolicy,
    run_episode,
    run_simulation,
    compare_strategies,
    MCResult,
)