# Project File Tree & Quick Reference

## 📂 Complete Project Directory Tree

```
MDP_simulation-V5/
│
├── 📋 DOCUMENTATION FILES (Start here!)
│   ├── README_DOCUMENTATION.md         ✨ Navigation guide for all docs
│   ├── GETTING_STARTED.md              ⭐ Quick start & common tasks
│   ├── PROJECT_STRUCTURE.md            Overview & architecture diagrams
│   ├── QUICK_REFERENCE.md              Definitions & concepts
│   └── MODULE_INDEX.md                 Technical deep-dive reference
│
├── 🔧 MAIN ENTRY POINTS
│   ├── config.py                       [RepeaterConfig] - All parameters here!
│   ├── Boxili_reproduction.py          [SCRIPT] Run to reproduce published results
│   ├── no_cutoff.py                    [SCRIPT] Baseline no-cutoff analysis
│   └── correctif.txt                   Project notes & corrections
│
├── 🎯 SOURCE CODE - src/
│   │
│   ├── 📐 mdp/                         ← MDP Core (State, Actions, Solver)
│   │   ├── __init__.py                 [PUBLIC API]
│   │   ├── actions.py                  [WAIT, SWAP, CUTOFF_*] Action definitions
│   │   ├── state_space.py              [State, utilities] State representation
│   │   ├── transitions.py              [transition()] Probability calculations
│   │   ├── rewards.py                  [reward_function()] Reward definitions
│   │   └── value_iteration.py          [solve_mdp()] 🔑 MDP solver (Bellman)
│   │
│   ├── ⚙️  simulation/                 ← Monte Carlo Engine
│   │   ├── __init__.py
│   │   ├── mc_engine.py                [NoCutoffPolicy, FixedCutoffPolicy, ...]
│   │   │                                [run_simulation()] Main simulator
│   │   └── network_state.py            [QuantumNetworkState] State tracking
│   │
│   ├── 🔬 physical/                    ← Quantum Physics Models
│   │   ├── __init__.py
│   │   ├── elementary_link.py          [generate_pair()] Generate quantum pairs
│   │   ├── swapping.py                 [perform_swap()] Bell state operations
│   │   └── werner_utils.py             [werner_fidelity()] Fidelity calculations
│   │
│   ├── 🌫️  noise/                      ← Decoherence & Noise
│   │   └── decoherence.py              [decoherence_rate()] Quantum decay
│   │
│   ├── 📊 metrics/                     ← Performance Metrics
│   │   ├── __init__.py
│   │   ├── skr.py                      [skr_from_samples()] Secret Key Rate
│   │   ├── plob.py                     [plob_bound()] PLOB upper limit
│   │   └── statistics.py               [mean_std(), CI()] Statistical tools
│   │
│   ├── 🧪 experiments/                 ← Experimental Workflows
│   │   ├── 01_validate_analytical.py   [SCRIPT] Validate vs analytical solutions
│   │   ├── 02_fixed_vs_adaptive.py     [SCRIPT] 🔑 Compare three policies
│   │   ├── 03_parameter_sweep.py       [SCRIPT] 🔑 Explore parameter space
│   │   ├── plot_advanced.py            Advanced visualization utilities
│   │   ├── plot_fidelity.py            Fidelity-specific plots
│   │   ├── plot_heatmap.py             Parameter heatmaps
│   │   └── plots.py                    General plotting utilities
│   │
│   └── 📈 analytical/                  ← Analytical Solutions
│       ├── boxili_wrapper.py           [WRAPPER] Reference implementation
│       └── no_cutoff.py                [ANALYTICAL] No-cutoff solution
│
├── 📚 EXTERNAL - external/             ← Reference Code
│   └── boxili/                         Boxi Li's published implementation
│       ├── environment.yml             Conda environment spec
│       ├── README.md
│       ├── repeater_algorithm.py       Reference algorithm
│       ├── repeater_mc.py              Reference MC simulation
│       ├── protocol_units.py
│       ├── protocol_units_efficient.py
│       ├── optimize_cutoff.py
│       ├── utility_functions.py
│       ├── logging_utilities.py
│       ├── examples.py
│       ├── test_protocol.py
│       ├── test_optimization.py
│       ├── tutorial.ipynb
│       ├── plot_paper.py
│       ├── plot_paper_new.py
│       ├── data/
│       ├── figures/
│       │   ├── swap_with_cutoff.npy
│       │   ├── swap_with_cutoff copy.npy
│       │   └── trade_off.npy
│       └── logging/
│           └── logging_record.json
│
├── 📁 RESULTS - results/               ← Experiment Outputs
│   ├── exp02_cutoff_curves.csv         Data: Fixed cutoff analysis
│   ├── exp02_fixed_vs_adaptive.csv     Data: 🔑 Policy comparison results
│   ├── exp03_improvement_grid.csv      Data: Performance improvements
│   ├── exp03_parameter_sweep.csv       Data: 🔑 Full parameter exploration
│   └── figures/
│       └── FigV1/                      Generated visualizations
│           ├── policy_comparison.png
│           ├── parameter_heatmap.png
│           ├── fidelity_analysis.png
│           └── ... (more plots)
│
└── 📋 STRATEGY - strategies/          ← Alternative Implementations
    ├── 01_validate_analytical.py
    ├── 02_fixed_vs_adaptive.py
    └── 03_parameter_sweep.py
```

---

## 🗂️ Quick File Lookup

### "I need to..."

#### Change parameters
→ Edit: `config.py` (RepeaterConfig class)

#### Run a quick test
→ Run: `python Boxili_reproduction.py`
→ Or: `python no_cutoff.py`

#### Find optimal cutoff
→ Run: `python src/experiments/02_fixed_vs_adaptive.py`
→ Check: `results/exp02_fixed_vs_adaptive.csv`

#### Explore parameter space
→ Run: `python src/experiments/03_parameter_sweep.py`
→ Check: `results/exp03_parameter_sweep.csv`

#### Validate against theory
→ Run: `python src/experiments/01_validate_analytical.py`

#### Understand MDP solver
→ Read: `src/mdp/value_iteration.py`

#### Modify physics model
→ Edit: `src/physical/`

#### Add new metric
→ Create: `src/metrics/new_metric.py`

#### Debug simulation
→ Read: `src/simulation/mc_engine.py`

#### See reference code
→ Check: `external/boxili/`

#### Analyze results
→ Use: Python + pandas + matplotlib
→ Use plots in: `src/experiments/plot_*.py`

---

## 📊 File Purposes Summary

| Type | Count | Purpose |
|------|-------|---------|
| Documentation | 5 | Learning & reference |
| Config | 1 | Parameters for all runs |
| Entry Scripts | 3 | Quick experiments |
| MDP Modules | 6 | Decision optimization |
| Physics Modules | 4 | Quantum modeling |
| Simulation Modules | 2 | MC simulation |
| Metrics | 3 | Performance measurement |
| Experiments | 7 | Workflows & analysis |
| Analytical | 2 | Theory comparison |
| External | Many | Reference code |
| Results | Many | Output data & plots |

---

## 🎯 Key Files to Know

### MUST READ (10 min)
- `README_DOCUMENTATION.md` — Navigation (this repo)
- `GETTING_STARTED.md` — Quick start
- `config.py` — All parameters

### SHOULD READ (30 min)
- `PROJECT_STRUCTURE.md` — Architecture overview
- `QUICK_REFERENCE.md` — Concepts & definitions
- `src/mdp/value_iteration.py` — MDP solver

### REFERENCE AS NEEDED
- `MODULE_INDEX.md` — Technical deep-dive
- `src/simulation/mc_engine.py` — Simulation logic
- `src/physical/werner_utils.py` — Physics calculations

### FOR ANALYSIS
- `results/*.csv` — All data
- `src/experiments/02_*` — Policy comparison
- `src/experiments/03_*` — Parameter sweep
- `src/experiments/plot_*.py` — Visualization

---

## 🚀 Typical File Usage by Task

### Task: Compare Policies
```python
# 1. Modify parameters
vim config.py

# 2. Run experiment
python src/experiments/02_fixed_vs_adaptive.py

# 3. Analyze results
results/exp02_fixed_vs_adaptive.csv

# 4. Visualize
python src/experiments/plot_heatmap.py
```

### Task: Parameter Sweep
```python
# 1. Define sweep
vim src/experiments/03_parameter_sweep.py

# 2. Run (will take hours)
python src/experiments/03_parameter_sweep.py

# 3. Analyze
results/exp03_parameter_sweep.csv

# 4. Generate heatmaps
python src/experiments/plot_advanced.py
```

### Task: Understand MDP
```python
# 1. Read the solver
cat src/mdp/value_iteration.py

# 2. Read concepts
cat QUICK_REFERENCE.md

# 3. Check state space
cat src/mdp/state_space.py

# 4. Look at transitions
cat src/mdp/transitions.py
```

---

## 📈 Output Files Created by Experiments

### After running `02_fixed_vs_adaptive.py`:
```
results/
├── exp02_cutoff_curves.csv
├── exp02_fixed_vs_adaptive.csv          ← Main results
└── figures/
    └── FigV1/
        ├── policy_comparison.png
        └── ... (more plots)
```

### After running `03_parameter_sweep.py`:
```
results/
├── exp03_improvement_grid.csv
├── exp03_parameter_sweep.csv            ← Main results
└── figures/
    └── FigV1/
        ├── parameter_heatmap.png
        ├── fidelity_heatmap.png
        └── ... (more plots)
```

---

## 🔍 Finding Code by Functionality

### Want to understand: States
→ `src/mdp/state_space.py`
→ `QUICK_REFERENCE.md` (State section)

### Want to understand: Actions
→ `src/mdp/actions.py`
→ `QUICK_REFERENCE.md` (Actions section)

### Want to understand: Transitions
→ `src/mdp/transitions.py`
→ `MODULE_INDEX.md` (Transitions section)

### Want to understand: Rewards
→ `src/mdp/rewards.py`
→ `QUICK_REFERENCE.md` (Rewards section)

### Want to understand: Solver
→ `src/mdp/value_iteration.py` 🔑
→ `QUICK_REFERENCE.md` (MDP Formulation)
→ `MODULE_INDEX.md` (Algorithm section)

### Want to understand: Simulation
→ `src/simulation/mc_engine.py` 🔑
→ `src/simulation/network_state.py`
→ `GETTING_STARTED.md` (Workflow section)

### Want to understand: Physics
→ `src/physical/werner_utils.py` 🔑
→ `src/physical/swapping.py`
→ `src/noise/decoherence.py`
→ `QUICK_REFERENCE.md` (Quantum Concepts)

### Want to understand: Metrics
→ `src/metrics/skr.py` 🔑
→ `src/metrics/plob.py`
→ `QUICK_REFERENCE.md` (Metrics section)

### Want to understand: Experiments
→ `src/experiments/02_fixed_vs_adaptive.py`
→ `src/experiments/03_parameter_sweep.py`
→ `GETTING_STARTED.md` (Workflow section)

---

## 📂 File Organization by Layer

### Configuration Layer
```
config.py ─→ ALL other modules
```

### Core Algorithm Layer
```
src/mdp/
  ├─ state_space.py
  ├─ actions.py
  ├─ transitions.py
  ├─ rewards.py
  └─ value_iteration.py [produces Policy & Value function]
```

### Physics Layer
```
src/physical/
  ├─ werner_utils.py
  ├─ elementary_link.py
  ├─ swapping.py
  └─ src/noise/decoherence.py
```

### Simulation Layer
```
src/simulation/
  ├─ mc_engine.py [Policies + run_simulation]
  └─ network_state.py [State management]
```

### Metrics Layer
```
src/metrics/
  ├─ skr.py
  ├─ plob.py
  └─ statistics.py
```

### Experiment Layer
```
src/experiments/
  ├─ 01_validate_analytical.py
  ├─ 02_fixed_vs_adaptive.py
  ├─ 03_parameter_sweep.py
  ├─ plot_*.py [Visualization]
  └─ [generates results/]
```

---

## ✨ File Icons Legend

| Icon | Meaning |
|------|---------|
| 📄 | Documentation file |
| 🔧 | Configuration/setup |
| 🔑 | KEY - Important file to understand |
| ⭐ | START HERE |
| 🎯 | Main objective/core file |
| 📊 | Data/results |
| 🧪 | Experimental/test code |
| ⚙️  | Core algorithm |

---

## 🎓 Reading Path by File Type

### For Documentation (5-60 min)
1. README_DOCUMENTATION.md (10 min)
2. GETTING_STARTED.md (15 min)
3. PROJECT_STRUCTURE.md (15 min)
4. QUICK_REFERENCE.md (15 min)
5. MODULE_INDEX.md (Reference as needed)

### For Code (varies)
1. `config.py` (understand parameters)
2. `src/mdp/value_iteration.py` (understand solver)
3. `src/simulation/mc_engine.py` (understand simulation)
4. `src/experiments/` (understand workflows)
5. Other modules as needed

### For Results (10 min)
1. `results/` directory listing
2. CSV files (pandas)
3. Plots (matplotlib)
4. Interpretation guide in MODULE_INDEX.md

---

## 🔗 Cross-References

### Documents Reference Each Other:
- README_DOCUMENTATION.md → Navigation to all files
- GETTING_STARTED.md → Quick tasks & references to other docs
- PROJECT_STRUCTURE.md → Architecture & module descriptions
- QUICK_REFERENCE.md → Concepts linked to modules
- MODULE_INDEX.md → Deep technical reference

### Code Links Back to Docs:
- `config.py` → See QUICK_REFERENCE.md for parameter meanings
- `src/mdp/` → See QUICK_REFERENCE.md for MDP concepts
- `src/physical/` → See QUICK_REFERENCE.md for quantum concepts
- `src/experiments/` → See GETTING_STARTED.md for workflow

---

## 📋 Quick Checklist

- [ ] Read this file (FILE_TREE.md)
- [ ] Read GETTING_STARTED.md (Quick Start)
- [ ] Open `config.py` and understand parameters
- [ ] Run `python Boxili_reproduction.py`
- [ ] Check `results/` directory
- [ ] Read PROJECT_STRUCTURE.md
- [ ] Run `python src/experiments/02_fixed_vs_adaptive.py`
- [ ] Analyze `results/exp02_fixed_vs_adaptive.csv`
- [ ] Reference MODULE_INDEX.md when needed

---

**This file tree serves as a map of the entire project.**
*Use this as your reference when navigating the codebase.*

*Last Updated: June 2026*
