# Supply Chain Inventory Optimization with Reinforcement Learning

## Overview

A PPO agent trained to manage inventory replenishment across multiple retail items and store locations. The agent operates over 365-day episodes and learns to balance holding costs, stockout penalties, and order costs under stochastic demand and supply disruptions.

Demand is modeled by a LightGBM forecaster trained on the M5 Forecasting dataset. The policy is universal — trained simultaneously across all item-store combinations using a 27-dimensional observation that encodes inventory state, active disruptions, and counterfactual order outcomes.

---

## Live Demo
https://github.com/user-attachments/assets/0b53e6ab-1cd9-4592-b6d7-d9eb6e9ae22a
## Problem Statement

A retailer manages hundreds of item-store combinations. Each day, the manager must decide how many units to order from the supplier. The decision is difficult because demand is uncertain and variable, orders take several days to arrive (lead time), and the supply chain is occasionally disrupted by external events.

Ordering too little leads to stockouts — lost sales and backlogged demand. Ordering too much accumulates inventory that costs money to hold. The goal is to find a replenishment policy that maximizes service level while keeping total cost low, across all items and stores simultaneously.

This project formulates inventory management as a Markov Decision Process and trains a single universal PPO policy that generalizes across items with different demand scales, volatilities, and lead times.

---

## Key Variables

**State variables** — observed by the agent each day:

| Variable | Description |
|---|---|
| `inventory` | Units currently on hand |
| `backlog` | Accumulated unmet demand not yet fulfilled |
| `pipeline_qty` | Units ordered but not yet received (in transit) |
| `inventory_position` | `inventory + pipeline - backlog` — the primary replenishment signal |
| `demand` | Yesterday's realized demand |
| `demand_forecast` | LightGBM forecast for today |
| `lead_time` | Effective days from order to arrival (base + disruption delta) |

**Disruption variables** — describe active supply chain shocks:

| Variable | Description |
|---|---|
| `dis_type` | Type of active disruption: 0 = none, 1 = port closure, 2 = supplier failure, 3 = demand surge |
| `dis_days_remaining` | Days until the disruption resolves |
| `dis_lead_delta` | Extra lead time added by the disruption |
| `capacity_ratio` | Fraction of normal supplier capacity available |
| `demand_mult` | Multiplier on realized demand (>1 during a demand surge) |

**Decision variable:**

| Variable | Description |
|---|---|
| `order_quantity` | Units to order today. Discretized into 20 levels from 0 to `demand_mean * lead_time * 10` |

**Cost variables** — components of the daily cost signal:

| Variable | Description |
|---|---|
| `holding_cost` | `inventory * 0.05` per day |
| `stockout_cost` | `stockout_units * 2.5` per day |
| `order_cost` | `10 + 1.0 * order_quantity` when an order is placed |
| `service_level` | `units_sold / demand` for the day (1.0 = fully fulfilled) |

---

## Causal Reasoning Module

Standard RL agents react to what they observe. This project augments the observation with a causal lookahead: before deciding how much to order, the agent implicitly asks "what would happen over the next 7 days under each candidate order quantity?"

This is implemented as a Structural Causal Model (SCM) following the three-step do-calculus procedure:

**Step 1 — Abduction.** Given today's observation, infer the underlying noise terms that explain the current state:
- Lead time noise = observed lead time − base lead time − disruption delta
- Demand noise = realized demand − forecast

**Step 2 — Intervention.** Fix the noise terms and simulate forward for each of the 20 candidate order quantities, treating the order as an intervention `do(order = q)`.

**Step 3 — Prediction.** Run a 7-day rollout for each candidate and compute 8 aggregate statistics: minimum stockout rate, maximum service level, median-order stockout rate, minimum cost, best-order average inventory, cost sensitivity across candidates, fraction of risky orders, and the relative magnitude of the best order.

These 8 values (dims 16–23 of the observation) give the agent a model-based signal about the downstream consequences of its action, without requiring the agent to learn this reasoning from scratch through trial and error alone. The causal framing ensures the forward simulation conditions correctly on the current disruption state rather than on correlates of past behavior.

---

## Project Structure

```
.
├── configs/config.yaml             # All tunable parameters
├── data/raw/
│   ├── sales_train_evaluation.csv
│   └── calendar.csv
├── models/
│   ├── best_model.zip
│   ├── ppo_universal_final.zip
│   ├── vecnormalize.pkl
│   └── demand_lgbm.pkl
├── env/supply_chain_env.py         # Gymnasium environment
├── simulator/
│   ├── supply_chain_engine.py      # Core simulation logic
│   ├── disruption_engine.py        # Stochastic disruption model
│   └── shipment_pipeline.py        # In-transit order tracking
├── demand/
│   ├── demand_generator.py         # Online inference wrapper
│   ├── feature_engineering.py      # Lag and rolling features
│   └── lightgbm_trainer.py         # LightGBM training
├── rl/
│   ├── train.py                    # PPO training entry point
│   └── evaluate.py                 # Policy comparison
├── causal/
│   ├── dag.py                      # Causal graph definition
│   ├── scm.py                      # Structural causal model
│   └── counterfactual_engine.py    # Counterfactual rollout
├── viz/
│   └── run_viz.py                  # Pygame visualization
└── notebooks/
    ├── 01_eda_m5.ipynb
    ├── 02_simulator_test.ipynb
    ├── 03_disruption_analysis.ipynb
    ├── 04_observation_space.ipynb
    └── 05_policy_evaluation.ipynb
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate

pip install stable-baselines3[extra] lightgbm pandas numpy gymnasium pygame pyyaml matplotlib jupyter
```

Python 3.10 or 3.11 recommended.

---

## Data

Download the M5 Forecasting dataset from Kaggle and place the files as follows:

| File | Path |
|---|---|
| `sales_train_evaluation.csv` | `data/raw/sales_train_evaluation.csv` |
| `calendar.csv` | `data/raw/calendar.csv` |

To train the demand model from scratch:

```bash
python -c "from demand.lightgbm_trainer import train_lgbm; train_lgbm()"
```

---

## Quick Start

**Train:**
```bash
python rl/train.py
```

**Resume from checkpoint:**
```bash
python rl/train.py models/checkpoints/ppo_universal_250000_steps.zip
```

**Evaluate:**
```bash
python rl/evaluate.py
```

**Visualization:**
```bash
python -m viz.run_viz
```

---

## Configuration

All parameters are in `configs/config.yaml`. Key values:

| Section | Key | Default | Description |
|---|---|---|---|
| simulation | `base_lead_time` | 7 | Days from order to arrival (base) |
| simulation | `holding_cost` | 0.05 | Cost per unit held per day |
| simulation | `stockout_penalty` | 2.5 | Cost per unit of unmet demand |
| simulation | `episode_length` | 365 | Days per training episode |
| disruption | `mean_inter_arrival` | 60 | Mean days between disruptions |
| demand | `store_ids` | CA_1 … TX_2 | Stores included in training |
| demand | `n_items` | 50 | Items per store (by descending sales) |
| rl | `total_timesteps` | 3,000,000 | Training budget |
| rl | `n_order_levels` | 20 | Discrete action space size |
| rl | `cf_horizon` | 7 | Counterfactual lookahead (days) |

---

## Environment

### Action Space

`Discrete(20)`. At each episode reset, order levels are scaled per item:

```
order_levels = linspace(0, demand_mean * base_lead_time * 10, 20)
```

Action 0 = hold. Action 19 = item maximum.

### Observation Space (27 dimensions)

| Dims | Group | Description |
|---|---|---|
| 0–3 | Inventory state | On-hand inventory, backlog, pipeline quantity, inventory position (all log-normalized) |
| 4–5 | Demand | Yesterday's realized demand, LightGBM point forecast |
| 6–8 | Time | Effective lead time, episode progress |
| 9–10 | Seasonality | Sine/cosine encoding of day-of-week |
| 11–15 | Disruption | Type, days remaining, lead time extension, capacity ratio, demand surge |
| 16–23 | Counterfactual | 8 statistics from a 7-day forward simulation across candidate order levels |
| 24–26 | Product context | Demand CV, relative demand scale, base lead time fraction |

### Reward

```
reward = clip(2 * service_level - sqrt(cost / expected_cost) - overstock_penalty + disruption_bonus, -30, 3.5)
```

A disruption bonus of +0.5 is added when the agent maintains service level above 90% during an active disruption.

---

## Disruption Model

Three disruption types are sampled with mean inter-arrival of 60 days. An episode typically contains 5–7 disruptions.

| Type | Lead time delta | Supplier capacity | Demand multiplier | Duration |
|---|---|---|---|---|
| Port closure | +10 to +20 days | unchanged | 1.0 | 5–15 days |
| Supplier failure | +0 to +5 days | 20–50% of normal | 1.0 | 7–21 days |
| Demand surge | 0 | unchanged | 1.5 to 3.0x | 3–10 days |

---

## Counterfactual Module

At each step, the counterfactual engine runs a 7-day forward simulation under the current disruption state for 20 candidate order levels. It uses a structural causal model (abduction → intervention → prediction) to infer noise terms from the current observation and produce 8 aggregate statistics that are appended to the observation vector.

This gives the policy a model-based lookahead signal without requiring explicit planning. The tradeoff is computational: at `cf_horizon=7` with 20 candidates, this adds ~140 simulation steps per environment step and is the dominant cost in training throughput.

---

## Evaluation

Three policies are compared over 50 episodes:

| Policy | Description |
|---|---|
| Random | Uniform random action |
| Heuristic (s,S) | Order to level S=400 when inventory drops below s=150 |
| PPO Universal | Trained model, deterministic inference |

Note: the heuristic uses global constants not scaled per item, so performance degrades on high-demand items.

---

## Visualization Controls

| Key | Action |
|---|---|
| Space | Play / Pause |
| R | Reset episode |
| D | Trigger disruption manually |
| Up / Down | Increase / decrease speed (1x–5x) |
| Left / Right | Previous / next item |

---

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Key metrics logged every 1000 steps:

| Metric | Description |
|---|---|
| `supply_chain/service_level` | Mean fraction of demand fulfilled |
| `supply_chain/stockout_rate` | Fraction of steps with unmet demand |
| `supply_chain/avg_cost` | Mean total cost per step |
| `supply_chain/disruption_pct` | Fraction of steps under active disruption |
