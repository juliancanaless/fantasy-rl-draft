import numpy as np, pandas as pd
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import tqdm

board = pd.read_csv("data/processed/training_data_2021.csv")

ROSTER = dict(QB=1, RB=2, WR=3, TE=1, K=1, DST=1, FLEX=1)

def make_env(slot):
    env = FantasyDraftEnv(
        board_df     = board,
        num_teams    = 12,
        my_slot      = slot,
        rounds       = 16,
        roster_slots = ROSTER,
        bench_spots  = 6,
    )
    return ActionMasker(env, lambda e: e.get_action_mask())

# ------------------------------------------------------------------
# Monte-Carlo baseline until mean stabilises (std error < 2 pts)
# ------------------------------------------------------------------
def baseline_mean(target_se=2.0, min_runs=100, max_runs=2000):
    scores = []
    for i in range(max_runs):
        env = make_env(np.random.randint(1, 13))
        obs, _ = env.reset()
        done = False
        while not done:
            # baseline uses internal heuristic for my slot
            env._opponent_pick(env.my_slot)
            env.pick_global += 1
            env._simulate_opponents()
            done = env.pick_global >= env.total_picks
        scores.append(env._lineup_points(env.board, env.my_picks))
        if len(scores) >= min_runs:
            se = np.std(scores) / np.sqrt(len(scores))
            if se < target_se:
                break
    return float(np.mean(scores))

baseline_pts = baseline_mean()
print(f"Baseline (heuristic) converged mean: {baseline_pts:.1f}")

# ---------------------- evaluate trained agent --------------------
model = MaskablePPO.load("ppo_12_2021_quick.zip")

def agent_episode():
    env = make_env(np.random.randint(1, 13))
    obs, info = env.reset()
    done = False
    while not done:
        mask = info["action_mask"]
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        obs, reward, done, _, info = env.step(action)
    return env._lineup_points(env.board, env.my_picks)

agent_scores = [agent_episode() for _ in tqdm.tqdm(range(300))]
print(f"Agent mean lineup pts:  {np.mean(agent_scores):.1f}")
print(f"Avg improvement:        {np.mean(agent_scores) - baseline_pts:+.1f}")
