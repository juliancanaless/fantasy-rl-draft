import numpy as np, pandas as pd
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

board = pd.read_csv("data/processed/training_data_2021.csv")

def make_env(slot):
    base = FantasyDraftEnv(
        board_df     = board,
        num_teams    = 12,
        my_slot      = slot,
        rounds       = 16,
        roster_slots = dict(QB=1, RB=2, WR=3, TE=1, K=1, DST=1, FLEX=1),
        bench_spots  = 6,
    )
    masked = ActionMasker(base, lambda e: e.get_action_mask())
    return Monitor(masked)

# each reset picks a random slot so the net generalises a bit
class SlotRandomWrapper:
    def __init__(self):
        self.env = None
    def __call__(self):
        slot = np.random.randint(1, 13)
        self.env = make_env(slot)
        return self.env

from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([lambda: make_env(np.random.randint(1, 13))])

model = MaskablePPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    n_steps=2048,
    batch_size=512,
)

model.learn(total_timesteps=2_000_000)
model.save("tests/ppo_12_2021")
