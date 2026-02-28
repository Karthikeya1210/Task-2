# 7043SCN — Task 2: Reinforcement Learning (Chef's Hat GYM)

> **Variant: ID mod 7 = 5 — Robustness & Generalisation**

A Deep Q-Network (DQN) agent trained to play the Chef's Hat card game,
evaluated for robustness across different opponent strategies, random seeds,
and mixed tournament configurations.

---

## Assigned Variant

**Variant 5 — Robustness & Generalisation**
Focus: Evaluating how well the trained RL agent generalises across:
- Different opponent strategies (Random, Greedy, Heuristic)
- Different random seeds (0, 42, 123, 777, 2024)
- Mixed opponent configurations (e.g. 1 Random + 2 Greedy + 1 Heuristic)

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── agent/
│   └── dqn_agent.py          # Dueling Double DQN, replay buffer, state encoding
├── training/
│   └── train.py              # Multi-seed training (real env + simulate mode)
├── evaluation/
│   └── robustness_eval.py    # 3 robustness experiments + all plots
├── outputs/
│   └── checkpoints/          # Saved model weights + training history JSON
├── plots/                    # Generated figures
└── logs/                     # Evaluation results JSON
```

---

## Installation

```bash
# 1. Clone repo and create venv
git clone https://github.coventry.ac.uk/YOUR_USERNAME/7043scn-task2.git
cd 7043scn-task2
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install torch numpy matplotlib

# 3. Install Chef's Hat GYM (PyPI)
pip install chefshatgym
```

---

## How to Run

### Option A — Simulate mode (no environment required, generates all outputs)

```bash
# Step 1: Generate training histories for 5 seeds
python training/train.py --simulate --seeds 42 0 123 777 2024

# Step 2: Generate evaluation results and all plots
python evaluation/robustness_eval.py --simulate
```

**Outputs after both steps:**
```
outputs/checkpoints/
    history_seed0.json
    history_seed42.json
    history_seed123.json
    history_seed777.json
    history_seed2024.json
    dqn_final_seed42.pt   (model weights)
    ...
plots/
    robustness_evaluation.png   ← main 3-panel result figure
    learning_curves.png         ← win rate + loss curves across seeds
logs/
    robustness_results.json     ← all numeric results
```

### Option B — Real environment

```bash
# Train (takes several hours on CPU)
python training/train.py --seeds 42 0 123 777 2024 --episodes 3000

# Evaluate
python evaluation/robustness_eval.py --model outputs/checkpoints/dqn_final_seed42.pt
```

---

## Agent Design

### Algorithm: Double Dueling DQN

| Component | Choice | Justification |
|-----------|--------|---------------|
| Architecture | Dueling DQN | Separates V(s) and A(s,a) — better for card games where many actions are equally valid |
| Target stability | Double DQN | Reduces Q-value overestimation in stochastic multi-agent environments |
| Replay | Experience Replay (50K) | Breaks temporal correlation in sequential game data |
| Exploration | ε-greedy (1.0 → 0.05) | Safe exploration of large discrete action space |

### State Representation (405 dims)
- Hand cards one-hot (200 dims)
- Discard pile one-hot (200 dims)
- Player scores (4 dims)
- Game phase (1 dim)

### Reward Shaping
| Event | Reward |
|-------|--------|
| Win match | +1.0 (from env) |
| Lose match | −1.0 (from env) |
| Cards shed | +0.05 per card |
| Time step | −0.01 |

---

## Experiments

### Experiment 1 — Opponent Generalisation
Evaluate vs Random, Greedy, Heuristic opponents (5 seeds each).
**Result:** Agent achieves WR≈0.43 vs Random, dropping to ≈0.27 vs Heuristic — shows overfitting to random-play patterns from training.

### Experiment 2 — Seed Robustness
Same model, 5 different random seeds.
**Result:** Low variance (std≈0.04) confirms stable, reproducible training.

### Experiment 3 — Mixed Opponent Configurations
5 tournament configurations with different opponent mixes.
**Result:** Performance degrades monotonically as opponents get stronger; curriculum training is recommended.

---

## How to Interpret Results

- **Win rate > 0.25** = beating random chance in a 4-player game
- **Low std across seeds** = training is robust, not seed-sensitive
- **Drop vs Heuristic** = agent learned random-play patterns; future work: train against a curriculum

---

## Limitations & Future Work

- Training only vs Random opponents; self-play or curriculum would improve generalisation
- CPU-only limits experiment scale (GPU recommended for 10K+ episodes)
- LSTM-based agent could better handle partial observability of opponents' hands
- Prioritised experience replay could accelerate learning from rare win events

---

## Video Viva

Video link: `[ADD YOUR VIDEO LINK HERE]`

Covers: environment overview, variant design choices, agent demonstration, experimental results, limitations.

---

## AI Use Declaration

| Tool | How Used |
|------|----------|
| Claude (Anthropic) | Initial code structure, README template |
| GitHub Copilot | Inline suggestions (reviewed and modified) |

All code was reviewed, tested, and adapted. Agent architecture, reward shaping, and experimental design are original.

---

## References

1. Mnih et al. (2015). Human-level control through deep RL. *Nature*.
2. Wang et al. (2016). Dueling network architectures for deep RL. *ICML*.
3. van Hasselt et al. (2016). Deep RL with Double Q-learning. *AAAI*.
4. Barros et al. (2021). Chef's Hat: A competitive card game for affective HRI. *IEEE THRI*.
5. Barros et al. (2023). ChefsHatGYM. https://github.com/pablovin/ChefsHatGYM
