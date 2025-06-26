# eval_compare_quick.py
#
# Monte-Carlo evaluation of the heuristic baseline vs. a trained agent
# (uses the newest FantasyDraftEnv – observation = roster counts + ADPs only)

import numpy as np, pandas as pd, tqdm
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

board   = pd.read_csv("data/processed/training_data_2021.csv")
ROSTER  = dict(QB=1, RB=2, WR=3, TE=1, K=1, DST=1, FLEX=1)
N_TEAMS = 12
ROUNDS  = 16

# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #
def make_env(slot: int):
    """Return **wrapped** env (ActionMasker) for the given draft slot."""
    env = FantasyDraftEnv(
        board_df     = board,
        num_teams    = N_TEAMS,
        my_slot      = slot,
        rounds       = ROUNDS,
        roster_slots = ROSTER,
        bench_spots  = 6,
    )
    return ActionMasker(env, lambda e: e.get_action_mask())


# --------------------------------------------------------------------- #
# 1)  Monte-Carlo baseline                                              #
# --------------------------------------------------------------------- #
def baseline_mean(target_se: float = 2.0,
                  min_runs: int   = 100,
                  max_runs: int   = 2000) -> float:
    """
    Draw baseline drafts until the standard-error of the mean lineup
    points is below `target_se` or `max_runs` reached.
    """
    scores: list[float] = []
    rng = np.random.default_rng()

    for i in range(max_runs):
        wrapped = make_env(rng.integers(1, N_TEAMS + 1))
        wrapped.reset()
        inner = wrapped.unwrapped            # peel off ActionMasker

        scores.append(inner._baseline_points())

        if len(scores) >= min_runs:
            se = np.std(scores) / np.sqrt(len(scores))
            if se < target_se:
                break

    return float(np.mean(scores))


print("Computing heuristic baseline …")
baseline_pts = baseline_mean()
print(f"Baseline (heuristic) converged mean: {baseline_pts:.1f} pts\n")


# --------------------------------------------------------------------- #
# 2)  Evaluate the trained agent                                        #
# --------------------------------------------------------------------- #
model = MaskablePPO.load("models/ppo_12_2021_quick.zip")

def agent_episode() -> float:
    wrapped = make_env(np.random.randint(1, N_TEAMS + 1))
    obs, info = wrapped.reset()
    done = False

    while not done:
        mask   = info["action_mask"]
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        obs, _, done, _, info = wrapped.step(action)

    inner = wrapped.unwrapped
    return inner._roster_points()            # best-lineup season points


print("Running agent episodes …")
agent_scores = [agent_episode() for _ in tqdm.tqdm(range(300))]
agent_mean   = float(np.mean(agent_scores))

print(f"\nAgent mean lineup pts:  {agent_mean:.1f}")
print(f"Average improvement vs. baseline:  {agent_mean - baseline_pts:+.1f} pts")
