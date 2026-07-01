# MDP Quantum Repeater Simulation - Module Index

## 📑 Complete Module Reference

### Root Level Files

| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `config.py` | Central configuration for all simulations | `RepeaterConfig` |
| `Boxili_reproduction.py` | Reproduces published quantum repeater results | Main script |
| `no_cutoff.py` | Baseline no-cutoff policy analysis | Main script |
| `correctif.txt` | Project notes and corrections | Documentation |

---

## 🎯 Core MDP Module (`src/mdp/`)

| Module | File | Purpose | Key Exports |
|--------|------|---------|-------------|
| Actions | `actions.py` | Action space definition | `WAIT`, `SWAP`, `CUTOFF_1`, `CUTOFF_2`, `CUTOFF_ALL`, `N_ACTIONS` |
| State Space | `state_space.py` | State representation and utilities | `State`, `state_to_tuple()`, `suggest_t_max()` |
| Transitions | `transitions.py` | Compute next state probabilities | `transition()`, `transition_compact()` |
| Rewards | `rewards.py` | Reward function definitions | `reward_function()`, `fidelity_reward()` |
| Solver | `value_iteration.py` | MDP Bellman solver | `solve_mdp()`, `policy_from_values()` |

### MDP Actions Hierarchy
```
WAIT (0)           → No operation, ages advance
SWAP (1)           → Terminal action, creates end-to-end link
CUTOFF_1 (2)       → Discard link 1
CUTOFF_2 (3)       → Discard link 2
CUTOFF_ALL (4)     → Discard both links
```

---

## ⚙️ Simulation Module (`src/simulation/`)

| Module | File | Purpose | Key Class |
|--------|------|---------|-----------|
| MC Engine | `mc_engine.py` | Policy simulation executor | `run_simulation()`, Policy classes |
| Network State | `network_state.py` | Quantum network state tracking | `QuantumNetworkState`, `LinkState` |

### Policy Classes (in `mc_engine.py`)
```python
NoCutoffPolicy()              # WAIT until both ready, then SWAP
FixedCutoffPolicy(cutoff)     # Cut links at fixed age T
AdaptivePolicy(v_func, pi)    # Use MDP optimal policy
```

---

## 🔬 Physical Layer (`src/physical/`)

| Module | File | Purpose | Key Functions |
|--------|------|---------|---------------|
| Elementary Link | `elementary_link.py` | Generate quantum pairs | `generate_pair()`, `werner_state()` |
| Swapping | `swapping.py` | Bell state operations | `swap_fidelity()`, `perform_swap()` |
| Werner Utils | `werner_utils.py` | Fidelity calculations | `werner_fidelity()`, `secret_fraction_nat()` |
| Decoherence | `noise/decoherence.py` | Decoherence model | `decoherence_rate()`, `fidelity_decay()` |

---

## 📊 Metrics Module (`src/metrics/`)

| Module | File | Purpose | Key Functions |
|--------|------|---------|---------------|
| SKR | `skr.py` | Secret Key Rate calculation | `skr_from_samples()`, `compute_skr()` |
| PLOB | `plob.py` | PLOB bound calculation | `plob_bound()`, `capacity_at_distance()` |
| Statistics | `statistics.py` | Statistical analysis | `mean_std()`, `confidence_interval()` |

### Key Metrics Computed
- **Secret Key Rate (SKR)**: Primary performance metric [bits/time]
- **PLOB Bound**: Theoretical maximum [bits/time]
- **Fidelity**: End-to-end Werner parameter [0.5-1.0]
- **Success Rate**: Fraction of successful swaps [0-1]

---

## 🧪 Experiments Module (`src/experiments/`)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_validate_analytical.py` | Validate MDP solutions against analytical | Config | Validation report |
| `02_fixed_vs_adaptive.py` | Compare three policy types | Config | `exp02_*.csv` |
| `03_parameter_sweep.py` | Explore parameter space | Config matrix | `exp03_*.csv` |
| `plots.py` | General plotting utilities | Data | Matplotlib figures |
| `plot_fidelity.py` | Fidelity-specific plots | Simulation results | Plots |
| `plot_heatmap.py` | Parameter sweep heatmaps | `exp03_*.csv` | Heatmap figures |
| `plot_advanced.py` | Advanced visualizations | Results | Publication-quality plots |

---

## 📁 Analytical Module (`src/analytical/`)

| Module | File | Purpose | Usage |
|--------|------|---------|-------|
| Boxi Li Wrapper | `boxili_wrapper.py` | Interface to reference implementation | Validation benchmarks |
| No-Cutoff | `no_cutoff.py` | Analytical solution (no cutoff) | Baseline comparison |

---

## 🔗 Dependency Graph (Import Hierarchy)

```
config.py
  ├─→ src/mdp/
  │    ├─→ actions.py
  │    ├─→ state_space.py
  │    ├─→ transitions.py (uses state_space.py)
  │    ├─→ rewards.py
  │    └─→ value_iteration.py (uses all above)
  │
  ├─→ src/physical/
  │    ├─→ elementary_link.py
  │    ├─→ swapping.py
  │    ├─→ werner_utils.py
  │    └─→ noise/decoherence.py
  │
  └─→ src/simulation/
       ├─→ mc_engine.py (uses mdp/ + physical/)
       └─→ network_state.py (uses physical/)

src/metrics/
  ├─→ skr.py (quantum theory)
  ├─→ plob.py (bounds)
  └─→ statistics.py (numpy)

src/experiments/
  ├─→ (All modules above)
  ├─→ plots.py (matplotlib)
  └─→ plot_*.py (specialized plots)
```

---

## 🎨 Data Flow Through Modules

```
1. CONFIG LAYER
   config.py
   └─ RepeaterConfig(p_gen, w0, t_coh, ...)

2. MODELING LAYER
   src/mdp/
   ├─ state_space.py: Define states
   ├─ actions.py: Define actions
   ├─ transitions.py: P(s'|s,a)
   ├─ rewards.py: R(s,a)
   └─ value_iteration.py: Solve V*(s), π*(s)

3. POLICY LAYER
   src/simulation/mc_engine.py
   ├─ NoCutoffPolicy
   ├─ FixedCutoffPolicy
   └─ AdaptivePolicy(from MDP solution)

4. PHYSICS LAYER
   src/physical/ + src/noise/
   ├─ Generate quantum pairs
   ├─ Apply decoherence
   ├─ Perform swaps
   └─ Track fidelity

5. SIMULATION LAYER
   mc_engine.py: Execute policies
   network_state.py: Track state

6. ANALYSIS LAYER
   src/metrics/
   ├─ skr.py: Compute SKR
   ├─ plob.py: Compute bounds
   └─ statistics.py: Aggregate

7. EXPERIMENT LAYER
   src/experiments/*.py
   ├─ Run workflows
   ├─ Aggregate results
   └─ Generate plots

8. OUTPUT LAYER
   results/
   ├─ *.csv (data)
   └─ figures/ (plots)
```

---

## 📈 Value Iteration Algorithm (in `value_iteration.py`)

```python
def solve_mdp(mdp_dict, config, verbose=False):
    """
    Inputs:
      - mdp_dict: {(s,a): [(prob, next_s, reward), ...], ...}
      - config: RepeaterConfig
      - verbose: Print convergence info
    
    Process:
      1. Initialize V(s) = 0 for all states
      2. Repeat until convergence:
         For each state s:
            V(s) ← max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
      3. Extract π*(s) = arg max_a [...]
    
    Outputs:
      - V_func: dict of state → value
      - policy: dict of state → action
    """
```

---

## 🔍 State Space Exploration

```
Example state: (s1=1, a1=3, s2=1, a2=7)
Meaning: 
  - Link 1: Ready (1), age 3 time units
  - Link 2: Ready (1), age 7 time units
  - Action: SWAP → Creates end-to-end link

Typical transitions:
  WAIT   → (s1', a1', s2', a2') where ages increase
  SWAP   → Terminal; score SKR
  CUTOFF → Discards and regenerates links
```

---

## 🚀 Quick Module Usage

### Running MDP Solver
```python
from src.mdp.value_iteration import solve_mdp
from src.mdp.state_space import build_state_list
from config import RepeaterConfig

config = RepeaterConfig()
states = build_state_list(config)
mdp_dict = build_mdp(states, config)  # From transitions.py
V, pi = solve_mdp(mdp_dict, config)
```

### Running Simulation
```python
from src.simulation.mc_engine import run_simulation, AdaptivePolicy
from config import RepeaterConfig

config = RepeaterConfig()
policy = AdaptivePolicy(V, pi)  # Use MDP solution
results = run_simulation(policy, config, n_runs=100000)
```

### Computing Metrics
```python
from src.metrics.skr import skr_from_samples
from src.metrics.statistics import mean_std

skr_values = skr_from_samples(results)
mean, std = mean_std(skr_values)
```

---

## 📊 CSV Output Format

### `exp02_fixed_vs_adaptive.csv`
```
cutoff_age,strategy,n_runs,skr_mean,skr_std,skr_ci_lower,skr_ci_upper,
fidelity_mean,fidelity_std,success_rate,runtime_sec
0,NoCutoff,100000,0.00893,0.00234,0.00850,0.00936,0.9512,0.0234,0.891,45.2
5,Fixed,100000,0.01098,0.00267,0.01050,0.01146,0.8734,0.0456,0.987,47.1
...
```

### `exp03_parameter_sweep.csv`
```
p_gen,w0,t_coh,optimal_cutoff,skr_no_cutoff,skr_fixed,skr_adaptive,
improvement_factor,improvement_percent
0.001,0.95,500,12,0.00451,0.00612,0.00892,1.98,98.2
0.001,0.95,1000,25,0.00513,0.00708,0.01034,2.02,101.6
...
```

---

## 🔧 Configuration Class Properties

### `RepeaterConfig` Class
```python
# Physical parameters
p_gen: float          # Link generation probability
w0: float             # Initial Werner parameter (fidelity)
p_swap: float         # Swapping success probability
t_coh: float          # Coherence time

# MDP parameters
gamma: float          # Discount factor
a_max: int            # Maximum age threshold
t_trunc: int          # Finite horizon
epsilon: float        # Convergence tolerance

# Simulation parameters
n_runs: int           # Number of MC runs
mc_max_steps: int     # Max steps per run
seed: int             # Random seed

# Computed properties
@property mean_gen_time()      # 1/p_gen
@property n_segments()          # 2^(# of swaps)
@property n_swap_levels()       # Number of swap operations
@property state_space_size()    # (2(a_max+1))^2
```

---

## 📝 Typical Result Interpretation

For a given configuration with fixed parameters:

| Policy | SKR | Fidelity | Interpretation |
|--------|-----|----------|-----------------|
| NoCutoff | Low | High | Conservative: waits too long |
| FixedCutoff(10) | Medium | Medium | Early cutoff: loses fidelity |
| FixedCutoff(20) | Higher | Lower | Later cutoff: better SKR |
| Adaptive (MDP) | **Highest** | **Balanced** | Optimal trade-off |

**The MDP's job**: Find cutoff policy that maximizes expected SKR considering decoherence.

---

*This index serves as a comprehensive guide to all modules, their functions, and how they interact in the MDP quantum repeater simulation system.*
