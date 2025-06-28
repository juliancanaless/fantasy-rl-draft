# Fantasy Football Draft Assistant: A Deep Reinforcement Learning Experiment

**An exploration of whether RL agents can learn optimal fantasy football drafting strategies across multiple seasons**

## 🎯 Project Overview

This project investigates the application of deep reinforcement learning to fantasy football draft optimization. Using historical player data (2021-2024), I built custom gymnasium environments and trained PPO agents to draft superior teams compared to heuristic baselines. The journey revealed fascinating insights about the limits of RL in rapidly evolving strategic domains.

## 🧠 The Research Question

**Can a reinforcement learning agent learn fantasy football drafting strategies that generalize across seasons?**

Initial hypothesis: Yes, fundamental drafting principles (value, scarcity, positional need) should transfer across years.

**Spoiler alert:** The answer was more nuanced than expected.

## 🏗️ Technical Architecture

### Environment Design
- **Custom Gymnasium Environment**: 12-team snake draft simulation
- **Action Space**: Discrete choice from 300 available players  
- **Observation Space**: Multi-dict with roster state, positional needs, and top-5 available players per position
- **Reward Engineering**: Dense rewards for lineup improvement + final score vs baseline

### Key Features
- **Intelligent Opponents**: Heuristic bots that follow realistic drafting patterns (no K/DST before round 13, positional need awareness)
- **Position Flexibility**: Full FLEX position support with optimal lineup calculation
- **Draft Position Randomization**: Agent learns strategy-invariant principles across all draft slots
- **Multi-Year Training**: Weighted sampling across 2021-2023 data (50% 2023, 35% 2022, 15% 2021)

### Model Architecture
- **Algorithm**: Maskable PPO with action masking for invalid picks
- **Policy Network**: Multi-input CNN processing positional availability tables
- **Training Scale**: 1M+ timesteps across multiple experiments
- **Hardware**: NVIDIA T4 GPU on Google Colab

## 📊 Key Experiments & Results

### Experiment 1: Single-Year Overfitting (2021 Data)
**Hypothesis**: Agent should easily beat baseline on same-year data

```
Training: 2021 data only (500k timesteps)
Testing: 2021 data

Results:
Agent: 2,387 pts | Baseline: 1,680 pts | Improvement: +42%
Win Rate: 100% | Status: ✅ EXCELLENT
```

**Outcome**: Complete success. Agent learned to exploit 2021-specific patterns perfectly.

### Experiment 2: Multi-Year Generalization (2021-2023 → 2024)
**Hypothesis**: Agent should learn transferable drafting principles

```
Training: 2021-2023 multi-year (1M timesteps, weighted sampling)
Testing: Individual years + unseen 2024

Results by Year:
2021: +501 pts (+26.2%) ✅
2022: +437 pts (+22.6%) ✅  
2023: +170 pts (+8.7%)  🟡
2024: -331 pts (-16.8%) ❌

Generalization Gap: 700+ points
```

**Outcome**: Catastrophic failure. Agent memorized training patterns that became obsolete.

### Experiment 3: Anti-Overfitting Measures (2023 → 2024)
**Hypothesis**: Regularization might enable generalization

```
Training: 2023 only with ADP noise, higher entropy, smaller network
Testing: 2024

Results:
Agent: 1,521 pts | Baseline: 1,982 pts | Improvement: -23%
Overfitting Gap: Minimal (0.7%)
Generalization: Still failed
```

**Outcome**: Ruled out overfitting. The problem was fundamental domain shift.

## 🔍 Key Discoveries

### 1. **Fantasy Football Strategy Evolution**
The dramatic performance decline 2021→2023→2024 revealed that fantasy football strategy evolves rapidly:
- **2021**: Post-COVID patterns, more predictable
- **2022-2023**: Rule changes, new offensive schemes  
- **2024**: Evolved meta that invalidated historical patterns

### 2. **Limits of Multi-Year RL Training**
Traditional RL assumption: "More data = better generalization"
Reality: Historical data became adversarial when the underlying game changed.

### 3. **The Baseline Quality Problem**
The heuristic baseline was already quite strong (smart positional timing, value-based picks), leaving limited room for improvement through pattern memorization.

### 4. **Observation Space Insights**
Key features that drove performance:
- **Positional scarcity signals**: ADP gaps indicating tier breaks
- **Roster construction state**: K/DST timing, FLEX optimization
- **Draft context awareness**: Round-dependent position prioritization

## 🛠️ Technical Implementation Highlights

### Reward Engineering
```python
# Dense reward for immediate improvement
reward = (new_lineup_points - old_lineup_points) / 10.0

# Final reward comparing to baseline
if done:
    reward += (final_score - baseline_score) / lineup_scale
```

### Smart Baseline Calculation
```python
def _baseline_points(self):
    # Run complete heuristic draft simulation
    # Calculate average across all opponent teams
    # Ensures fair comparison excluding agent's slot
```

### Multi-Year Environment Optimization
- Pre-processed board templates for 10x faster resets
- Weighted year sampling with recency bias
- Draft position randomization for strategy-invariant learning

## 📈 Technical Metrics

- **Training Episodes**: 50,000+ complete drafts simulated
- **Environment Performance**: 2.5 drafts/second evaluation speed
- **Model Size**: 1.2M parameters (MultiInputPolicy CNN)
- **Data Scale**: 900+ players across 4 seasons
- **Evaluation Robustness**: 300 episodes per test with statistical significance

## 🎯 Strategic Insights for RL Practitioners

### When RL Works for Game Domains:
- **Stable rule sets** (chess, Go, poker)
- **Clear optimal strategies** that don't change over time
- **Sufficient signal-to-noise ratio** in the reward function

### When RL Struggles:
- **Rapidly evolving metas** (fantasy sports, real-time strategy games)
- **High variance outcomes** where luck dominates skill
- **Non-stationary environments** where historical data becomes misleading

### Alternative Approaches for Fantasy Sports:
- **Supervised learning** on season-specific patterns
- **Multi-task learning** with year-specific heads
- **Meta-learning** approaches for few-shot adaptation

## 💡 Project Lessons

### Technical Learnings
1. **Environment design matters more than model complexity**
2. **Reward shaping can mask fundamental domain problems**
3. **Evaluation methodology must match real-world deployment**
4. **Baseline quality determines the ceiling for RL improvement**

### Domain Insights
5. **Fantasy football strategy shifts faster than RL can adapt**
6. **Historical performance may be negatively correlated with future success**
7. **Human experts already encode most learnable patterns into ADP**

### Research Methodology
8. **Always test generalization early and often**
9. **Single-year overfitting can mask generalization failures**
10. **Cross-validation is critical for time-series domains**

## 🏆 Portfolio Significance

This project demonstrates:
- **Advanced RL Engineering**: Custom environments, reward design, multi-agent simulation
- **Rigorous Experimentation**: Proper baselines, statistical evaluation, controlled experiments  
- **Domain Expertise**: Deep understanding of fantasy football strategy and market dynamics
- **Critical Analysis**: Honest assessment of approach limitations and alternative directions
- **Production Considerations**: Performance optimization, evaluation frameworks, deployment readiness

The "failure" to achieve generalization is actually a valuable research contribution, highlighting the boundaries of current RL techniques in rapidly evolving strategic domains.

## 🔮 Future Directions

If continuing this research:

1. **Hybrid Approaches**: Combine RL for draft strategy with supervised learning for player valuation
2. **Meta-Learning**: Train agents to quickly adapt to new season patterns
3. **Multi-Objective Optimization**: Balance expected value against variance/risk
4. **Real-Time Adaptation**: Update models during the season as new information emerges
5. **Causal Inference**: Focus on stable causal relationships rather than correlational patterns

## 📊 Repository Structure

```
fantasy-rl-draft/
├── src/                    # Core RL environment and training
├── data/                   # Processed player data 2021-2024
├── models/                 # Trained PPO agents
├── tests/                  # Evaluation and debugging scripts
├── results/                # Experimental results and analysis
├── scripts/                # Data processing pipeline
└── docs/                   # Additional documentation
```

---

**The most important lesson: Sometimes the most valuable outcome of an ML project is learning when *not* to apply a particular technique to a given domain.**