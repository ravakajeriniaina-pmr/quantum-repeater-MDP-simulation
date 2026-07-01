# MDP Quantum Repeater Simulation - Quick Reference Guide

## 🎯 Project Goal

Find **optimal quantum repeater cutoff policies** using Markov Decision Process (MDP) optimization to maximize **Secret Key Rate (SKR)** in quantum networks.

---

## 📖 Key Concepts & Terminology

### Quantum Concepts
| Term | Definition |
|------|-----------|
| **Werner State** | Mixed entangled state with tuneable fidelity parameter `w0` (0=separable, 1=maximally entangled) |
| **Fidelity** | Measure of quantum state quality; higher = closer to pure entangled Bell state |
| **Decoherence** | Loss of quantum coherence over time; exponential decay with timescale `t_coh` |
| **Swapping** | Bell measurement connecting two links into a longer-distance link |
| **Entanglement Swapping** | Create end-to-end entanglement using intermediate repeaters |

### MDP Concepts
| Term | Definition |
|------|-----------|
| **State** | `(s1, a1, s2, a2)` = link status + ages on two segments |
| **s1, s2** | Link status: 1=ready (entangled), 0=failed |
| **a1, a2** | Link age: time elapsed since pair generation |
| **Action** | Decision: WAIT, SWAP, or CUTOFF |
| **Transition** | Probability of reaching next state given current state+action |
| **Reward** | Quality measure (SKR, fidelity) for action outcome |
| **Policy** | Mapping: State → Action (decision rule) |
| **Value Function** | Expected cumulative reward from each state following optimal policy |

### Network Parameters
| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Generation probability | `p_gen` | Prob. of successfully generating entangled pair per time unit |
| Swap success probability | `p_swap` | Prob. of successful Bell measurement (typically 1.0 in simulations) |
| Initial fidelity | `w0` | Werner parameter of generated pairs (0.5-1.0) |
| Coherence time | `t_coh` | Characteristic decoherence timescale |
| Discount factor | `gamma` | MDP discount (typically 0.99) |
| Max age | `a_max` | Maximum link age considered (prevents state explosion) |
| Time horizon | `t_trunc` | Simulation time limit for finite-horizon MDP |

### Policy Types
| Policy | Strategy | Pros | Cons |
|--------|----------|------|------|
| **NoCutoff** | Wait until both links ready → swap | Simple, no premature loss | Long wait times, high decoherence |
| **FixedCutoff(T)** | Force swap after age T | Prevents excessive aging | Fixed T may be suboptimal |
| **Adaptive** | Use MDP-computed optimal decisions | Theoretically optimal | Requires solving MDP first |

### Performance Metrics
| Metric | Symbol | Formula | Range |
|--------|--------|---------|-------|
| **Secret Key Rate** | SKR | Bits successfully transmitted per unit time | [0, ∞) |
| **PLOB Bound** | - | Upper limit on SKR for quantum repeaters | ≤ PLOB |
| **End-to-End Fidelity** | F | Final Werner parameter after all operations | [0.5, 1.0] |
| **Success Rate** | P_succ | Fraction of attempted swaps that succeed | [0, 1] |

---

## 🔧 Main Configuration Parameters

```python
# src/config.py - RepeaterConfig class

# Physical parameters
p_gen = 0.01          # Link generation rate (1% per time unit)
w0 = 1.0              # Initial fidelity (1.0 = perfect)
p_swap = 1.0          # Swapping success (1.0 = deterministic)
t_coh = 1000.0        # Coherence time (time units)

# MDP parameters
gamma = 0.99          # Discount factor
a_max = 200           # Max age considered
t_trunc = 10000       # Finite horizon length
epsilon = 1e-6        # Convergence tolerance for value iteration

# Simulation parameters
n_runs = 100_000      # Simulations per configuration
mc_max_steps = 10_000_000  # Max steps per simulation
seed = 42             # Random seed for reproducibility
```

---

## 🔄 Typical Analysis Workflow

```
1. CONFIGURE
   └─ Set parameters in config.py
   └─ Choose network scenario

2. SOLVE MDP
   └─ Run value_iteration.py
   └─ Get optimal policy π*(s)
   └─ Get value function V*(s)

3. CREATE POLICIES
   └─ NoCutoffPolicy() [baseline]
   └─ FixedCutoffPolicy(cutoff_age) [varying T]
   └─ AdaptivePolicy(mdp_solution) [optimal]

4. SIMULATE
   └─ Run Monte Carlo with each policy
   └─ Generate n_runs=100,000 samples
   └─ Track SKR, fidelity, success rates

5. ANALYZE
   └─ Compute statistics (mean, CI)
   └─ Compare policy performance
   └─ Identify optimal cutoff age

6. VISUALIZE
   └─ Plot SKR vs cutoff age
   └─ Plot fidelity improvements
   └─ Generate heatmaps for parameter space

7. EXPORT
   └─ Save results to CSV
   └─ Save plots to figures/
```

---

## 📁 File Organization by Purpose

### Core Simulation Pipeline
```
config.py                          ← Start here
src/mdp/value_iteration.py        ← Solve MDP
src/simulation/mc_engine.py       ← Run simulations
src/metrics/skr.py                ← Compute metrics
src/experiments/02_fixed_vs_adaptive.py  ← Full workflow
```

### Physics & Models
```
src/physical/werner_utils.py      ← State fidelity
src/physical/swapping.py          ← Swapping operations
src/noise/decoherence.py          ← Decoherence effects
```

### Analysis & Visualization
```
src/metrics/statistics.py         ← Statistical tools
src/experiments/plot_*.py         ← Visualization scripts
results/                          ← Output storage
```

### Reference & Validation
```
external/boxili/                  ← Boxi Li's code
Boxili_reproduction.py            ← Reproduce published results
src/experiments/01_validate_analytical.py  ← Validate vs analytical
```

---

## 🚀 Running Experiments

### Quick Test
```bash
cd /Volumes/Data/PHYSIQUE\ S10/SIMULATION/MDP_simulation-V5
python -c "from config import RepeaterConfig; print(RepeaterConfig())"
```

### Reproduce Boxi Li Results
```bash
python Boxili_reproduction.py
# Generates plots matching published paper
```

### Validate Analytical Solution
```bash
python src/experiments/01_validate_analytical.py
# Compares MDP solution against analytical formulas
```

### Compare Policies
```bash
python src/experiments/02_fixed_vs_adaptive.py
# Tests: NoCutoff vs FixedCutoff vs Adaptive
# Output: results/exp02_*.csv
```

### Parameter Space Exploration
```bash
python src/experiments/03_parameter_sweep.py
# Sweeps over p_gen, w0, t_coh, etc.
# Output: results/exp03_parameter_sweep.csv
```

---

## 📊 Result Files Structure

### Experiment 02: Fixed vs Adaptive
```
exp02_fixed_vs_adaptive.csv:
  cutoff_age | strategy | n_runs | skr_mean | skr_std | fidelity | ...
  0          | NoCutoff | 100k   | 0.0089   | 0.002   | 0.95    | ...
  5          | Fixed    | 100k   | 0.0127   | 0.003   | 0.87    | ...
  ...
  optimal    | Adaptive | 100k   | 0.0156   | 0.002   | 0.88    | ...
```

### Experiment 03: Parameter Sweep
```
exp03_parameter_sweep.csv:
  p_gen | w0   | t_coh | optimal_cutoff | skr_no_cutoff | skr_adaptive | improvement
  0.001 | 0.95 | 500   | 12             | 0.0045        | 0.0089       | 1.98x
  0.001 | 0.95 | 1000  | 25             | 0.0051        | 0.0103       | 2.02x
  ...
```

---

## 🔍 Key State Space Properties

```
State: (s1, a1, s2, a2)
  - s1, s2 ∈ {0, 1}              (2 values each)
  - a1, a2 ∈ {0, 1, ..., a_max}  (a_max+1 values each)

Total states: 2 × (a_max+1) × 2 × (a_max+1) = 4(a_max+1)²

With a_max=200: ~160,000 states
With a_max=500: ~1,000,000 states

Terminal states: (1, *, 1, *) where s1=s2=1 (ready for swap)
```

---

## 🧮 MDP Formulation

### Bellman Equation (Value Iteration)
$$V_{t+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_t(s') \right]$$

Where:
- $V_t(s)$ = Value of state $s$ at iteration $t$
- $R(s,a)$ = Immediate reward for action $a$ in state $s$
- $P(s'|s,a)$ = Transition probability to state $s'$
- $\gamma$ = Discount factor (typically 0.99)

### Optimal Policy
$$\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]$$

---

## 📈 Expected Results

For typical parameters (p_gen=0.01, w0=0.98, t_coh=1000):

| Policy | SKR (bits/time) | Fidelity | Improvement over NoCutoff |
|--------|-----------------|----------|--------------------------|
| NoCutoff | ~0.0089 | 0.95 | Baseline |
| Fixed (T=10) | ~0.0098 | 0.88 | +10% |
| Fixed (T=20) | ~0.0121 | 0.85 | +36% |
| Adaptive (MDP) | ~0.0156 | 0.83 | **+75%** |

Adaptive policy achieves **1.5-2x improvement** by learning optimal cutoff age!

---

## ⚠️ Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| MDP solver slow | State space too large | Reduce `a_max` or increase `epsilon` |
| Simulations crash | Out of memory | Reduce `n_runs` or `mc_max_steps` |
| Results don't match Boxi Li | Different parameters | Check `Boxili_reproduction.py` for exact config |
| Low SKR values | Poor initial fidelity or coherence | Increase `w0` or `t_coh` in config |
| Adaptive worse than Fixed | MDP misconfigured | Verify reward function in `rewards.py` |

---

## 📚 References

- **Primary**: Boxi Li et al., Quantum repeater optimization paper
- **Code**: `external/boxili/` directory with reference implementation
- **Theory**: Standard MDP/Bellman equation formulations
- **Metrics**: SKR based on quantum information theory

---

## 🤝 Project Structure Summary

```
Configuration (config.py)
    ↓
Physics Models (src/physical/)
    ↓
MDP Formulation (src/mdp/)
    ↓
Policy Extraction → Policies (NoCutoff, Fixed, Adaptive)
    ↓
Monte Carlo Simulation (src/simulation/)
    ↓
Metrics & Analysis (src/metrics/)
    ↓
Experiments (src/experiments/)
    ↓
Results (results/ + CSV files)
```

---

*Last Updated: June 2026*
*Project: MDP Quantum Repeater Simulation V5*
