import numpy as np, pandas as pd
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

board = pd.read_csv("data/processed/training_data_2021.csv")

def make_env(slot):
    env = FantasyDraftEnv(
        board_df     = board,
        num_teams    = 12,
        my_slot      = slot,
        rounds       = 16,
        roster_slots = dict(QB=1,RB=2,WR=3,TE=1,K=1,DST=1,FLEX=1),
        bench_spots  = 6,
    )
    env = ActionMasker(env, lambda e: e.get_action_mask())
    return Monitor(env)

# randomise slot each instance so policy learns invariance
vec_env = DummyVecEnv([
    lambda s=i: make_env(np.random.randint(1, 13)) for i in range(4)   # 4 parallel envs
])

model = MaskablePPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    n_steps=256,            # 4 envs × 256 = 1 024 steps per update
    batch_size=1_024,
)

model.learn(total_timesteps=100_000)   # ≈ 520 drafts, ~8 min on T4
model.save("ppo_12_2021_quick")
