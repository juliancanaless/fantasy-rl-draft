# tests/eval_compare_quick.py
import numpy as np, pandas as pd, tqdm
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

board   = pd.read_csv("data/processed/training_data_2021.csv")
ROSTER  = dict(QB=1, RB=2, WR=3, TE=1, K=1, DST=1, FLEX=1)

# -------------------------------------------------------------------
# helper to build a *wrapped* env; we’ll keep a shortcut to .unwrapped
# -------------------------------------------------------------------
def make_env(slot: int):
    env      = FantasyDraftEnv(
        board_df     = board,
        num_teams    = 12,
        my_slot      = slot,
        rounds       = 16,
        roster_slots = ROSTER,
        bench_spots  = 6,
    )
    wrapped  = ActionMasker(env, lambda e: e.get_action_mask())
    return wrapped

# -------------------------------------------------------------------
# Monte-Carlo heuristic baseline, stop when the SEM < 2 pts
# -------------------------------------------------------------------
def baseline_mean(target_se: float = 2.0,
                  min_runs: int   = 100,
                  max_runs: int   = 2000) -> float:
    scores = []
    for _ in range(max_runs):
        wenv               = make_env(np.random.randint(1, 13))
        env                = wenv.unwrapped            # <-- RAW env
        obs, _             = wenv.reset()
        done               = False

        # the heuristic always picks for *my* slot
        while not done:
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
print(f"Heuristic baseline (converged): {baseline_pts:6.1f} pts")

# -------------------------------------------------------------------
# load the trained agent
# -------------------------------------------------------------------
model = MaskablePPO.load("models/ppo_12_2021_quick.zip")

# run 300 evaluation drafts
def agent_episode() -> float:
    wenv           = make_env(np.random.randint(1, 13))
    env            = wenv.unwrapped                  # raw env
    obs, info      = wenv.reset()
    done           = False
    while not done:
        action, _  = model.predict(
            obs,
            deterministic=False,
            action_masks=info["action_mask"]
        )
        obs, _, done, _, info = wenv.step(action)
    return env._lineup_points(env.board, env.my_picks)


print("\nRunning agent episodes …")
agent_scores = [agent_episode() for _ in tqdm.tqdm(range(300))]
mean_agent   = float(np.mean(agent_scores))

print(f"\nAgent mean lineup pts : {mean_agent:6.1f}")
print(f"Average improvement    : {mean_agent - baseline_pts:+6.1f}")
