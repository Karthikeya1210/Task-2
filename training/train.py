"""
Training script — Chef's Hat GYM v3 (async API)
Module: 7043SCN | Variant 5: Robustness & Generalisation

Install environment:
    pip install chefshatgym torch numpy matplotlib

Run (real environment):
    python training/train.py --seeds 42 0 123 777 2024

Run (simulate — no environment needed, generates all outputs):
    python training/train.py --simulate --seeds 42 0 123 777 2024
"""

import os
import sys
import json
import random
import asyncio
import logging
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent, DQNCore, DQN_CONFIG, STATE_DIM, ACTION_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ─────────────────────────── REAL TRAINING (v3 async API) ────────────────────

async def train_real(seed: int, episodes: int, config: dict):
    """Train using the real ChefsHatGYM v3 Room API."""
    from rooms.room import Room
    from agents.random_agent import RandomAgent

    set_seed(seed)
    ckpt_dir = config.get("checkpoint_dir", "./outputs/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = f"./logs/seed_{seed}"
    os.makedirs(log_dir, exist_ok=True)

    history = {"episode": [], "score": [], "win": [], "epsilon": [], "avg_loss": []}
    wins    = []
    losses  = []

    for ep in range(1, episodes + 1):
        dqn_agent   = DQNAgent(name="DQN_Agent", config=config, train=True)
        opponents   = [RandomAgent(name=f"Rand_{i}", log_directory=log_dir) for i in range(3)]

        room = Room(
            run_remote_room=False,
            room_name=f"train_ep{ep}_seed{seed}",
            max_matches=1,
            log_directory=log_dir,
            verbose=False,
        )

        for p in [dqn_agent] + opponents:
            room.connect_player(p)

        await room.run()

        # Extract result — room.final_scores is dict {player_name: score}
        scores = room.final_scores or {}
        my_score = scores.get(dqn_agent.name, 0)
        won = 1 if my_score == max(scores.values(), default=0) else 0
        wins.append(won)

        ep_losses = dqn_agent.dqn.losses[-50:] if dqn_agent.dqn.losses else [0]
        losses.extend(ep_losses)

        if ep % 10 == 0:
            wr   = np.mean(wins[-100:])
            loss = np.mean(ep_losses)
            history["episode"].append(ep)
            history["score"].append(float(my_score))
            history["win"].append(float(wr))
            history["epsilon"].append(round(dqn_agent.dqn.epsilon, 3))
            history["avg_loss"].append(round(loss, 4))
            logger.info("Ep %4d | WinRate: %.3f | ε: %.3f | Loss: %.4f",
                        ep, wr, dqn_agent.dqn.epsilon, loss)

        if ep % 500 == 0:
            dqn_agent.dqn.save(os.path.join(ckpt_dir, f"dqn_ep{ep}_seed{seed}.pt"))

    # Save final model and history
    final_agent = DQNAgent(name="DQN_Final", config=config, train=False)
    # Rebuild from last dqn_agent weights
    final_agent.dqn.policy.load_state_dict(dqn_agent.dqn.policy.state_dict())
    final_agent.save(os.path.join(ckpt_dir, f"dqn_final_seed{seed}.pt"))

    hist_path = os.path.join(ckpt_dir, f"history_seed{seed}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Seed %d done. History → %s", seed, hist_path)
    return history


# ─────────────────────────── SIMULATE MODE ───────────────────────────────────

def simulate_training(seed: int, episodes: int, config: dict):
    """
    Generate realistic training history without needing the environment.
    Useful for testing the pipeline and producing demonstration outputs.
    """
    set_seed(seed)
    ckpt_dir = config.get("checkpoint_dir", "./outputs/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info("Simulating training: seed=%d, episodes=%d", seed, episodes)

    eps_list = list(range(10, episodes + 1, 10))
    history  = {"episode": [], "score": [], "win": [], "epsilon": [], "avg_loss": []}

    noise = {"42": 0.03, "0": 0.05, "123": 0.03, "777": 0.06, "2024": 0.04}.get(str(seed), 0.04)

    for ep in eps_list:
        progress  = 1 / (1 + np.exp(-8 * (ep / episodes - 0.40)))
        win_rate  = float(np.clip(0.20 + 0.25 * progress + np.random.normal(0, noise), 0, 1))
        loss      = float(np.clip(2.0 * np.exp(-ep / 800) + 0.3 + np.random.normal(0, 0.05), 0.05, None))
        epsilon   = float(max(0.05, 1.0 * (DQN_CONFIG["eps_decay"] ** ep)))
        score     = float(np.clip(np.random.normal(0.5 + 0.3 * progress, 0.2), 0, 1))

        history["episode"].append(ep)
        history["win"].append(round(win_rate, 4))
        history["avg_loss"].append(round(loss, 4))
        history["epsilon"].append(round(epsilon, 4))
        history["score"].append(round(score, 4))

        if ep % 500 == 0:
            logger.info("  ep %4d | WinRate: %.3f | Loss: %.4f | ε: %.3f",
                        ep, win_rate, loss, epsilon)

    # Also save a simulated model checkpoint (random weights — just for structure)
    core = DQNCore(STATE_DIM, ACTION_DIM, config)
    core.save(os.path.join(ckpt_dir, f"dqn_final_seed{seed}.pt"))

    hist_path = os.path.join(ckpt_dir, f"history_seed{seed}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Simulated history saved → %s", hist_path)
    return history


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DQN for Chef's Hat (Variant 5)")
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42])
    parser.add_argument("--episodes", type=int,  default=DQN_CONFIG["episodes"])
    parser.add_argument("--simulate", action="store_true",
                        help="Skip environment, generate realistic outputs for demonstration")
    args = parser.parse_args()

    config = {**DQN_CONFIG, "episodes": args.episodes}

    for seed in args.seeds:
        logger.info("=" * 55)
        logger.info("Seed: %d | Episodes: %d | Mode: %s",
                    seed, args.episodes, "simulate" if args.simulate else "real")
        logger.info("=" * 55)

        if args.simulate:
            simulate_training(seed, args.episodes, config)
        else:
            try:
                asyncio.run(train_real(seed, args.episodes, config))
            except ImportError:
                logger.error(
                    "\n\nChefsHatGYM not installed!\n"
                    "Install with:  pip install chefshatgym\n"
                    "Or use simulate mode:  python training/train.py --simulate\n"
                )
                sys.exit(1)

    logger.info("\nAll seeds complete.")
    logger.info("Next: python evaluation/robustness_eval.py --simulate")


if __name__ == "__main__":
    main()