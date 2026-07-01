"""
src/mdp — MDP modules for the two-link quantum repeater.

Public API
----------
Actions
~~~~~~~
    WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL
        Integer action constants.
    N_ACTIONS
        Total number of actions (5).
    available_actions(status_1, status_2) -> list
        Return the list of valid actions for a given link-status pair.
    available_actions_from_tuple(state) -> list
        Same as above but accepts a full state tuple.
    is_valid_action(action, status_1, status_2) -> bool
    validate_action(action, status_1, status_2) -> None
        Raises ValueError if the action is not valid.
    action_name(action) -> str
    action_short_name(action) -> str
    is_terminal_action(action) -> bool   # True only for SWAP
    is_cutoff_action(action) -> bool     # True for CUTOFF_1/2/ALL

Transitions
~~~~~~~~~~~
    transition(state, action, p_gen, t_max, *, p_swap=1.0) -> list[tuple]
        Return [(prob, next_state), ...] for all successor states.
    transition_compact(state, action, p_gen, t_max, *, p_swap=1.0)
        Same as transition() but merges duplicate successors.
    build_transition_dict(state_list, action, p_gen, t_max) -> dict
        Pre-compute transitions for a full list of states.

Rewards
~~~~~~~
    reward(action, age_1, age_2, w0, t_coh, p_swap=1.0) -> float
        Expected reward for *action* (non-zero only for SWAP).
    reward_from_state(action, state, w0, t_coh, p_swap=1.0) -> float
        Convenience wrapper that unpacks the state tuple.
    swap_werner(age_1, age_2, w0, t_coh) -> float
        Output Werner parameter after a BSM swap.
    swap_reward(age_1, age_2, w0, t_coh) -> float
        secret_fraction(swap_werner(...)) — reward ignoring p_swap.
    max_age_for_positive_reward(w0, t_coh) -> int
        Largest symmetric age still yielding positive key fraction.
    reward_table(w0, t_coh, max_age) -> np.ndarray
        Pre-computed (max_age+1)x(max_age+1) reward table.
    reward_table_vectorized(t_max, w0, t_coh) -> np.ndarray
        Same as reward_table, but computed via NumPy broadcasting.

Solver
~~~~~~
    solve_mdp(p_gen, w0, t_coh, t_max, p_swap=1.0, ...) -> dict
        Run Relative Value Iteration and return a dict with keys:
        'gain', 'policy', 'h', 'iterations', 'converged', 'elapsed'.
    policy_summary(result) -> dict
        Summarise the optimal policy by state category.
    print_policy_summary(result) -> None
        Print a human-readable policy summary.
    print_policy_slice(result, max_age=20) -> None
        Print a 2-D policy slice (age_1 x age_2) for both-entangled states.
    extract_effective_cutoff(result) -> dict
        Extract the effective age threshold used by the cutoff actions.

Notes
-----
* States are 4-tuples (status_1, age_1, status_2, age_2) where
  status is 0 (pending) or 1 (entangled), and age is a non-negative
  integer capped at t_max.
* The MDP uses the average-reward (gain) criterion; the solver
  implements Relative Value Iteration (RVI).

Example
-------
>>> from src.mdp import solve_mdp
>>> result = solve_mdp(p_gen=0.1, w0=0.99, t_coh=1000, t_max=200)
>>> print(f"SKR gain = {result['gain']:.6f}")
"""

# ── Actions ───────────────────────────────────────────────────────────
# src/mdp/__init__.py
# garder seulement les constantes/actions sûres
from src.mdp.actions import WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL

# ── Transitions ───────────────────────────────────────────────────────
from src.mdp.transitions import (
    transition,
    transition_compact,
    build_transition_dict,
)

# ── Rewards ───────────────────────────────────────────────────────────
from src.mdp.rewards import (
    swap_werner,
    reward,
    reward_from_state,
    swap_reward,
    max_age_for_positive_reward,
    reward_table,
    reward_table_vectorized,
)

# ── Solver ────────────────────────────────────────────────────────────
from src.mdp.value_iteration import (
    solve_mdp,
    policy_summary,
    print_policy_summary,
    print_policy_slice,
    extract_effective_cutoff,
)


__all__ = [
    # actions
    "WAIT", "SWAP", "CUTOFF_1", "CUTOFF_2", "CUTOFF_ALL",
    "N_ACTIONS", "ACTION_NAMES", "ACTION_SHORT",
    "available_actions", "available_actions_from_tuple",
    "n_available_actions",
    "is_valid_action", "validate_action",
    "is_terminal_action", "is_cutoff_action",
    "action_name", "action_short_name",
    # transitions
    "transition", "transition_compact", "build_transition_dict",
    # rewards
    "swap_werner", "reward", "reward_from_state",
    "swap_reward", "max_age_for_positive_reward",
    "reward_table", "reward_table_vectorized",
    # solver
    "solve_mdp", "policy_summary",
    "print_policy_summary", "print_policy_slice",
    "extract_effective_cutoff",
]
