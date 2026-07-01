# MDP Quantum Repeater Simulation - Project Structure

## 📋 Project Overview

This project implements a **Markov Decision Process (MDP)** framework for optimizing quantum repeater protocols. It combines Monte Carlo simulations, MDP value iteration, and physical modeling of quantum systems to find optimal strategies for quantum state distribution through repeater chains.

**Key Focus**: Optimizing cutoff policies for quantum repeater systems using reinforcement learning principles.

---

## 🏗️ Directory Structure

```
MDP_simulation-V5/
├── 📄 Root Configuration & Scripts
│   ├── config.py                    # Central parameter configuration (RepeaterConfig class)
│   ├── Boxili_reproduction.py       # Reproduces Boxi Li's quantum repeater results
│   ├── no_cutoff.py                 # Baseline: no cutoff policy analysis
│   └── correctif.txt                # Notes/corrections log
│
├── 📁 src/                          # Main source code
│   ├── 🔧 mdp/                      # Markov Decision Process core
│   │   ├── __init__.py              # Public API exports
│   │   ├── actions.py               # Action definitions (WAIT, SWAP, CUTOFF_*)
│   │   ├── rewards.py               # Reward function definitions
│   │   ├── state_space.py           # State space representation & utilities
│   │   ├── transitions.py           # Transition probability calculations
│   │   └── value_iteration.py       # MDP solver (Bellman equation iteration)
│   │
│   ├── ⚙️ simulation/               # Monte Carlo simulation engine
│   │   ├── __init__.py
│   │   ├── mc_engine.py             # Main simulation engine with policies
│   │   │                             # - NoCutoffPolicy
│   │   │                             # - FixedCutoffPolicy
│   │   │                             # - AdaptivePolicy
│   │   └── network_state.py         # Quantum network state tracking
│   │
│   ├── 🔬 physical/                 # Quantum physics models
│   │   ├── __init__.py
│   │   ├── elementary_link.py       # Single quantum link generation
│   │   ├── swapping.py              # Bell state swapping operations
│   │   ├── werner_utils.py          # Werner state utilities & calculations
│   │   └── [utilities]              # Helper functions for quantum operations
│   │
│   ├── 📊 metrics/                  # Key performance indicators & calculations
│   │   ├── __init__.py
│   │   ├── plob.py                  # PLOB bound calculations
│   │   ├── skr.py                   # Secret key rate metrics
│   │   ├── statistics.py            # Statistical analysis tools
│   │   └── [utilities]              # Helper calculations
│   │
│   ├── 🌫️ noise/                    # Noise and decoherence models
│   │   └── decoherence.py           # Quantum decoherence effects
│   │
│   ├── 🧪 experiments/              # Experimental workflows
│   │   ├── 01_validate_analytical.py     # Validate analytical solutions
│   │   ├── 02_fixed_vs_adaptive.py       # Compare policy types
│   │   ├── 03_parameter_sweep.py         # Explore parameter space
│   │   ├── plot_advanced.py              # Advanced visualization
│   │   ├── plot_fidelity.py              # Fidelity analysis plots
│   │   ├── plot_heatmap.py               # Heatmap visualizations
│   │   └── plots.py                      # General plotting utilities
│   │
│   └── 📈 analytical/               # Analytical solutions & wrappers
│       ├── boxili_wrapper.py        # Wrapper for Boxi Li reference implementation
│       ├── no_cutoff.py             # Analytical no-cutoff case
│       └── [utilities]              # Analytical calculations
│
├── 📁 external/                     # External dependencies & references
│   └── boxili/                      # Reference implementation (Boxi Li)
│       ├── environment.yml          # Conda environment
│       ├── examples.py
│       ├── logging_utilities.py
│       ├── optimize_cutoff.py
│       ├── plot_paper.py / plot_paper_new.py
│       ├── protocol_units.py / protocol_units_efficient.py
│       ├── repeater_algorithm.py
│       ├── repeater_mc.py
│       ├── utility_functions.py
│       ├── test_optimization.py
│       ├── test_protocol.py
│       ├── tutorial.ipynb
│       └── data/
│           └── figures/             # Pre-generated reference figures
│               └── logging/
│
├── 📁 results/                      # Generated experimental results
│   ├── exp02_cutoff_curves.csv      # Cutoff policy curves
│   ├── exp02_fixed_vs_adaptive.csv  # Policy comparison data
│   ├── exp03_improvement_grid.csv   # Parameter sweep improvements
│   ├── exp03_parameter_sweep.csv    # Full parameter exploration
│   └── figures/
│       └── FigV1/                   # Generated visualization outputs
│
└── 📁 strategies/                   # Strategy implementations
    ├── 01_validate_analytical.py
    ├── 02_fixed_vs_adaptive.py
    └── 03_parameter_sweep.py
```

---

## 🔄 Core Workflow

### 1. **Problem Definition** (`src/mdp/`)
- **State Space**: `(s1, a1, s2, a2)` where:
  - `s1, s2` = quantum link status (0=failed, 1=ready)
  - `a1, a2` = link age (time since generation)
- **Actions**: WAIT, SWAP, CUTOFF_1, CUTOFF_2, CUTOFF_ALL
- **Reward**: Typically based on successful swaps or secret key rate

### 2. **Physics Modeling** (`src/physical/`)
- Generate quantum states (Werner states, entangled pairs)
- Simulate Bell swapping operations
- Track quantum link fidelity degradation
- Account for decoherence and noise

### 3. **MDP Solver** (`src/mdp/value_iteration.py`)
- Bellman equation iteration to find optimal policy
- Handles finite horizon (`t_trunc`) with cutoff states
- Returns: Value function V(s) and optimal actions π(s)

### 4. **Monte Carlo Simulation** (`src/simulation/mc_engine.py`)
- Evaluates policies in realistic scenarios
- Supports three policy types:
  - **NoCutoffPolicy**: Wait for both links ready, then swap
  - **FixedCutoffPolicy**: Cutoff links after fixed time
  - **AdaptivePolicy**: Use MDP-optimal decisions
- Generates performance metrics

### 5. **Metrics & Analysis** (`src/metrics/`)
- **SKR (Secret Key Rate)**: Primary performance metric
- **PLOB Bound**: Theoretical upper limit
- Statistics: Mean, variance, confidence intervals

### 6. **Experiments** (`src/experiments/`)
- **Exp01**: Validate analytical solutions vs simulations
- **Exp02**: Compare fixed vs adaptive policies
- **Exp03**: Parameter sweep (explore `p_gen`, `t_coh`, `w0`, etc.)
- Output: CSV results + visualizations

---

## 🔑 Key Components

### Configuration (`config.py`)
```python
RepeaterConfig:
  - p_gen: Link generation probability
  - p_swap: Swapping success probability  
  - w0: Initial fidelity (Werner parameter)
  - t_coh: Coherence time (decoherence timescale)
  - a_max: Maximum age for consideration
  - gamma: MDP discount factor
```

### Main Policies
- **NoCutoffPolicy**: Baseline (WAIT until both ready)
- **FixedCutoffPolicy(cutoff_age)**: Cut after age T
- **AdaptivePolicy(mdp_solution)**: Uses optimal MDP policy

### Metrics
- **SKR**: Secret key rate (bits/time)
- **PLOB**: PLOB bound (theoretical max)
- **Fidelity**: End-to-end Werner parameter
- **Success Rate**: Fraction of successful swaps

---

## 📊 Result Files

| File | Purpose |
|------|---------|
| `exp02_cutoff_curves.csv` | Cutoff time vs performance |
| `exp02_fixed_vs_adaptive.csv` | Policy comparison results |
| `exp03_parameter_sweep.csv` | Grid of parameter combinations |
| `exp03_improvement_grid.csv` | MDP improvement over baselines |

---

## 🚀 Typical Workflow

```
1. Configure parameters (config.py)
2. Run MDP solver (mdp/value_iteration.py)
3. Extract optimal policy
4. Create AdaptivePolicy from solution
5. Run MC simulations with all three policies
6. Collect metrics (SKR, fidelity, etc.)
7. Generate comparison plots
8. Save results to CSV
```

---

## 📚 External Reference

- **Boxi Li's Original Work**: `external/boxili/`
  - Reference implementation for validation
  - Published protocol benchmarks
  - Paper reproduction scripts

---

## 🎯 Key Metrics & Objectives

- **Primary**: Maximize Secret Key Rate (SKR)
- **Secondary**: Maximize end-to-end fidelity
- **Constraint**: Minimize resource consumption (swaps, cutoffs)
- **Goal**: Find optimal cutoff age as function of network parameters

---

## 🔧 Usage Examples

### Run a full experiment:
```python
python src/experiments/02_fixed_vs_adaptive.py
```

### Reproduce Boxi Li results:
```python
python Boxili_reproduction.py
```

### Parameter sweep:
```python
python src/experiments/03_parameter_sweep.py
```

---

## 📝 Notes

- Uses conda environment (see `external/boxili/environment.yml`)
- All simulations use seeded random number generation for reproducibility
- Results aggregated over `n_runs=100,000` simulations per configuration
- MDP solver typically completes in seconds; MC simulations take minutes to hours
