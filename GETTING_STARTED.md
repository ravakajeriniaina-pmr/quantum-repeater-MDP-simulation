# MDP Quantum Repeater Simulation - Getting Started Guide

## 🎯 Quick Start (5 minutes)

### 1. Verify Project Structure
```bash
cd /Volumes/Data/PHYSIQUE\ S10/SIMULATION/MDP_simulation-V5
ls -la                    # See all files
python -c "from config import RepeaterConfig; print(RepeaterConfig())"
```

### 2. Run a Quick Test
```bash
# Test baseline (no cutoff policy)
python no_cutoff.py

# Or reproduce Boxi Li's published results
python Boxili_reproduction.py
```

### 3. Explore the Code
- Read: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Overview
- Reference: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — Concepts & commands
- Details: [MODULE_INDEX.md](MODULE_INDEX.md) — All modules explained

---

## 📋 Common Tasks & How-To's

### Task 1: Change Network Parameters

**File**: `config.py`

```python
# Modify the RepeaterConfig class
p_gen = 0.05          # Increase link generation (faster)
w0 = 0.90             # Lower fidelity (noisier)
t_coh = 2000.0        # Longer coherence time
```

**Then run**: `python src/experiments/02_fixed_vs_adaptive.py`

---

### Task 2: Find Optimal Cutoff Age

**Workflow**:
1. **Run experiment**:
   ```bash
   python src/experiments/02_fixed_vs_adaptive.py
   ```

2. **Check results**:
   ```bash
   cat results/exp02_fixed_vs_adaptive.csv
   # Look for cutoff_age with highest SKR
   ```

3. **Visualize**:
   ```bash
   python src/experiments/plot_heatmap.py
   # Shows SKR vs cutoff age
   ```

**Output**: 
- CSV with all tested cutoff ages
- Plots showing optimal point
- MDP solution beats fixed policies by 50-100%

---

### Task 3: Compare Three Policies

**What's compared**:
1. **NoCutoff**: Wait until both links ready
2. **FixedCutoff(T)**: Cut after time T
3. **Adaptive**: Use MDP-optimal decisions

**Run**:
```bash
python src/experiments/02_fixed_vs_adaptive.py
```

**Results**:
- `results/exp02_fixed_vs_adaptive.csv`
- Columns: cutoff_age, strategy, skr_mean, fidelity_mean, etc.
- Plots: `results/figures/FigV1/policy_comparison.*`

---

### Task 4: Explore Parameter Dependencies

**Question**: How does SKR depend on p_gen, w0, t_coh?

**Solution**:
```bash
python src/experiments/03_parameter_sweep.py
```

**Configuration** (in experiment file):
```python
p_gen_values = [0.001, 0.005, 0.01, 0.05]
w0_values = [0.90, 0.95, 0.98, 1.0]
t_coh_values = [500, 1000, 2000, 5000]
```

**Results**:
- `results/exp03_parameter_sweep.csv` — All combinations
- Heatmaps showing trends
- Optimal cutoff for each point

---

### Task 5: Reproduce Published Results

**Goal**: Verify against Boxi Li's paper

```bash
python Boxili_reproduction.py
```

**What it does**:
- Uses Boxi Li's exact parameters (Table I)
- Runs with reference implementation
- Generates paper figures
- Compares MDP solution vs published

**Output**: `results/figures/FigV1/boxili_comparison.*`

---

### Task 6: Debug/Validate Your Configuration

**Check if parameters are reasonable**:
```python
from config import RepeaterConfig

config = RepeaterConfig(
    p_gen=0.01,
    t_coh=1000,
    w0=0.98
)

# Validate
config.validate()

# Get suggestions
print(f"Suggested t_trunc: {config.suggest_t_trunc()}")
print(f"Suggested a_max: {config.suggest_a_max()}")

# Check state space size
print(f"Total states: {config.state_space_size}")
```

---

### Task 7: Run Custom Analysis

**Template**:
```python
from config import RepeaterConfig
from src.mdp.value_iteration import solve_mdp
from src.simulation.mc_engine import run_simulation, AdaptivePolicy
from src.metrics.skr import skr_from_samples
from src.metrics.statistics import mean_std

# 1. Configure
config = RepeaterConfig(p_gen=0.01, w0=0.95, t_coh=1000)

# 2. Solve MDP
mdp_dict = build_mdp(config)  # See value_iteration.py
V, policy = solve_mdp(mdp_dict, config)

# 3. Simulate
policy_obj = AdaptivePolicy(V, policy)
results = run_simulation(policy_obj, config, n_runs=100000)

# 4. Analyze
skr_values = skr_from_samples(results)
mean_skr, std_skr = mean_std(skr_values)
print(f"SKR: {mean_skr:.6f} ± {std_skr:.6f}")
```

---

## 🔗 Module Dependency Chain

For different tasks, you'll use different parts:

### For MDP Solution Only
```
config.py 
  → src/mdp/state_space.py
  → src/mdp/transitions.py
  → src/mdp/value_iteration.py
```
**Time**: ~1 second | **Output**: Optimal policy dictionary

### For Monte Carlo Simulation
```
config.py
  → src/physical/ (quantum models)
  → src/simulation/mc_engine.py
  → src/metrics/
```
**Time**: Minutes-hours (depends on n_runs) | **Output**: Simulation samples

### For Complete Experiment
```
All of above
  → src/experiments/
  → Plotting functions
  → CSV export
  → Visualization
```
**Time**: Hours for large sweeps | **Output**: Results + figures

---

## 📊 Understanding Your Results

### SKR (Secret Key Rate)

**What is it**: Bits of secret key transmitted per time unit

**Typical values**:
- No cutoff: 0.004-0.010 bits/time
- Fixed cutoff: 0.008-0.015 bits/time
- Adaptive (MDP): 0.012-0.020 bits/time

**Better = Higher SKR**

### Fidelity

**What is it**: Quality of quantum states (0.5 = separable, 1.0 = perfect)

**Typical values**: 0.80-0.98

**Trade-off**: Higher fidelity → Lower SKR (must wait)

### Improvement Factor

**Formula**: `skr_adaptive / skr_no_cutoff`

**Typical**: 1.5x - 2.5x improvement

**Interpretation**: MDP strategy is 50-150% better!

---

## 🧪 Experiment Workflow Decisions

### Which experiment should I run?

| Question | Experiment | Time |
|----------|-----------|------|
| What's the optimal cutoff age? | 02 | 5-10 min |
| How do parameters affect performance? | 03 | 1-2 hours |
| Does MDP outperform baselines? | 02 | 5-10 min |
| Does my code match theory? | 01 | 5 min |
| What was Boxi Li's exact result? | Boxili_repro | 2-5 min |

---

## 🛠️ Troubleshooting

### Problem: MDP solver is slow
**Cause**: Too many states (a_max too large)
**Fix**: Reduce `a_max` in config.py or increase `epsilon`

### Problem: Simulations won't finish
**Cause**: Too many runs or steps requested
**Fix**: Reduce `n_runs` or `mc_max_steps` in config.py

### Problem: Results don't match Boxi Li
**Cause**: Different parameters or seeds
**Fix**: Use exact config from `Boxili_reproduction.py`

### Problem: Adaptive policy is worse than NoCutoff
**Cause**: Bug in MDP or reward function
**Fix**: Run validation experiment 01 first

### Problem: Memory error during sweep
**Cause**: Parameter grid too large
**Fix**: Reduce number of parameter values in experiment

---

## 📈 Typical Timeline

```
1. Initial exploration (30 min)
   ├─ Read: PROJECT_STRUCTURE.md
   ├─ Run: Boxili_reproduction.py
   └─ Review: Results plots

2. Quick analysis (1-2 hours)
   ├─ Modify config.py
   ├─ Run experiment 02 (policy comparison)
   └─ Generate plots

3. Detailed study (4-8 hours)
   ├─ Run experiment 03 (parameter sweep)
   ├─ Create heatmaps
   ├─ Analyze trends
   └─ Export results

4. Paper/publication (8-24 hours)
   ├─ Run multiple sweeps
   ├─ Generate publication-quality plots
   ├─ Run validation experiments
   └─ Prepare results for presentation
```

---

## 💾 Saving & Organizing Results

### File Organization
```
results/
├── exp02_fixed_vs_adaptive_v1.csv    ← Result from run 1
├── exp02_fixed_vs_adaptive_v2.csv    ← After parameter change
├── exp03_parameter_sweep_full.csv    ← Complete sweep
└── figures/
    ├── FigV1_policy_comparison.png
    ├── FigV2_parameter_heatmap.png
    ├── FigV3_convergence.png
    └── ...
```

### Backup Strategy
```bash
# Archive results after important run
cp -r results results_backup_v1.tar.gz

# Keep experiment configs
cp config.py config_v1_bak.py
```

---

## 🎓 Learning Path

**Week 1**: Foundation
- Read PROJECT_STRUCTURE.md
- Run Boxili_reproduction.py
- Explore MODULE_INDEX.md
- Modify simple parameters

**Week 2**: Deep dive
- Understand MDP formulation
- Study transitions.py, value_iteration.py
- Run experiments 02 and 03
- Analyze results

**Week 3**: Advanced
- Implement custom reward functions
- Add new policy types
- Conduct parameter sweeps
- Generate publication plots

---

## 🔍 Key Files to Start With

1. **config.py** — Start here to understand parameters
2. **Boxili_reproduction.py** — Run first to see workflow
3. **src/experiments/02_fixed_vs_adaptive.py** — Main experiment workflow
4. **src/mdp/value_iteration.py** — Core algorithm
5. **src/simulation/mc_engine.py** — Simulation engine

---

## 📞 Common Commands Reference

```bash
# View results
head results/exp02_fixed_vs_adaptive.csv

# Count lines in CSV
wc -l results/exp03_parameter_sweep.csv

# Find optimal cutoff
grep -i "adaptive" results/exp02_fixed_vs_adaptive.csv

# List all figures generated
ls -lah results/figures/FigV1/

# Clean old results
rm -rf results/exp02_*_old.csv
```

---

## ✅ Checklist: Ready to Start

- [ ] Read PROJECT_STRUCTURE.md
- [ ] Run `python no_cutoff.py` (verify setup)
- [ ] Run `python Boxili_reproduction.py` (see example)
- [ ] Review QUICK_REFERENCE.md
- [ ] Explore MODULE_INDEX.md
- [ ] Choose your first experiment (02 or 03)
- [ ] Run and analyze results
- [ ] Modify config.py and run again
- [ ] Generate plots
- [ ] Export results to CSV

---

## 🚀 Next Steps After Reading

1. **Immediate** (5 min):
   - Run baseline: `python Boxili_reproduction.py`
   - Check results: `ls results/`

2. **Short term** (30 min):
   - Modify config.py
   - Run experiment 02
   - Visualize results

3. **Medium term** (2-4 hours):
   - Run parameter sweep (experiment 03)
   - Analyze heatmaps
   - Generate custom plots

4. **Long term** (1-2 weeks):
   - Implement new policies
   - Extend reward functions
   - Prepare for publication

---

**Happy simulating! 🎯**

*For questions about specific modules, refer to MODULE_INDEX.md*
*For conceptual questions, refer to QUICK_REFERENCE.md*
