import numpy as np, pandas as pd
from src.fantasyDraftEnv import FantasyDraftEnv

# 1. load a processed board (2021 data are fine for smoke)
board = pd.read_csv("data/processed/training_data_2021.csv")

# 2. env params (10-team league just to check flexibility)
env = FantasyDraftEnv(
    board_df     = board,
    num_teams    = 10,
    my_slot      = 3,
    rounds       = 16,
    roster_slots = dict(QB=1, RB=2, WR=2, TE=1, K=1, DST=1, FLEX=1),
    bench_spots  = 6,
)

obs, info = env.reset(seed=42)
done = False
tot_reward = 0.0

while not done:
    mask = info["action_mask"]
    action = int(np.random.choice(np.flatnonzero(mask)))  # random legal pick
    obs, r, done, _, info = env.step(action)
    tot_reward += r

print("Random draft finished.  Total reward vs baseline:", tot_reward)
