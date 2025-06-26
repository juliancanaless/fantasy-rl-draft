# tests/eval_compare_quick.py
import numpy as np, pandas as pd, tqdm
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

board  = pd.read_csv("data/processed/training_data_2021.csv")
ROSTER = dict(QB=1, RB=2, WR=3, TE=1, K=1, DST=1, FLEX=1)

def make_env(slot: int):
    env = FantasyDraftEnv(
        board_df     = board,
        num_teams    = 12,
        my_slot      = slot,          # 1-based
        rounds       = 16,
        roster_slots = ROSTER,
        bench_spots  = 6,
    )
    return ActionMasker(env, lambda e: e.get_action_mask())

# ----------------------------------------------------------------------
# 1) Heuristic baseline
# ----------------------------------------------------------------------
def baseline_mean(runs: int = 500) -> float:
    pts = []
    for _ in range(runs):
        wenv = make_env(np.random.randint(1, 13))
        wenv.reset()
        # ask the underlying env for its deterministic baseline
        pts.append(wenv.unwrapped._baseline_points())
    return float(np.mean(pts))

baseline_pts = baseline_mean()
print(f"Heuristic baseline (mean of 500): {baseline_pts:.1f} pts")

# ----------------------------------------------------------------------
# 2) Load trained agent
# ----------------------------------------------------------------------
model = MaskablePPO.load("models/ppo_12_2021_quick")

def agent_episode() -> float:
    wenv = make_env(np.random.randint(1, 13))
    obs, info = wenv.reset()
    done = False
    while not done:
        action, _ = model.predict(
            obs,
            deterministic=False,
            action_masks=info["action_mask"]
        )
        obs, _, done, _, info = wenv.step(action)
    # lineup points must be computed on **the same env that ran the draft**
    return wenv.unwrapped._lineup_points(wenv.unwrapped.board,
                                         wenv.unwrapped.my_picks)

print("\nRunning 300 agent drafts …")
agent_scores = [agent_episode() for _ in tqdm.tqdm(range(300))]
mean_agent   = float(np.mean(agent_scores))

print(f"\nAgent mean lineup pts : {mean_agent:.1f}")
print(f"Average improvement   : {mean_agent - baseline_pts:+.1f}")
