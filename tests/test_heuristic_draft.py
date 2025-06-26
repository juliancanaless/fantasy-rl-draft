"""
tests/test_heuristic_draft.py
-----------------------------
Drafts one full board with the opponent heuristic and checks:

  • proper snake order
  • each team meets ≥ required roster counts
  • kickers/defenses not drafted before the final four rounds
  • no duplicate players

Run:  python -m tests.test_heuristic_draft
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.fantasyDraftEnv import FantasyDraftEnv, BASE_POS

# --------------------------------------------------------------------------
NUM_TEAMS   = 12
ROUNDS      = 16
ROSTER_REQ  = dict(QB=1, RB=2, WR=3, TE=1, K=1, DST=1, FLEX=1)
BOARD_PATH  = Path("data/processed/training_data_2021.csv")
# --------------------------------------------------------------------------

board_df = pd.read_csv(BOARD_PATH)

def run_heuristic_draft():
    env = FantasyDraftEnv(
        board_df     = board_df,
        num_teams    = NUM_TEAMS,
        my_slot      = 1,          # slot unused in this test
        rounds       = ROUNDS,
        roster_slots = ROSTER_REQ,
        bench_spots  = 6,
    )

    board = env._board_template.copy()
    board["available"] = True

    team_counts = [
        {p: 0 for p in BASE_POS} | {"FLEX": 0}
        for _ in range(NUM_TEAMS)
    ]

    pick_log = []  # (round, team_idx, position)

    for pick_no in range(NUM_TEAMS * ROUNDS):
        round_idx = pick_no // NUM_TEAMS
        idx_in_round = pick_no % NUM_TEAMS

        # snake: even rounds left→right, odd rounds right→left
        if round_idx % 2 == 0:
            team_idx = idx_in_round
        else:
            team_idx = NUM_TEAMS - 1 - idx_in_round

        top_idx = env._heuristic_pick(board, team_counts[team_idx], pick_no)
        pos     = board.at[top_idx, "position"]
        name    = board.at[top_idx, "name"]
        pick_log.append((round_idx, team_idx, pos, name))

    return pick_log, board, team_counts

# --------------------------------------------------------------------------
# run draft and assertions
# --------------------------------------------------------------------------
picks, board_after, counts_list = run_heuristic_draft()

# 1. snake order
expected = []
for rnd in range(ROUNDS):
    order = list(range(NUM_TEAMS))
    if rnd % 2 == 1:
        order.reverse()
    expected.extend(order)
actual = [team for _, team, _, _ in picks]
assert actual == expected, "Snake order violated"

# 2. uniqueness
drafted = board_after[~board_after["available"]]
assert len(drafted) == NUM_TEAMS * ROUNDS, "Missing picks"
assert drafted["name"].is_unique, "Duplicate player drafted"

# 3. late K/DST
early = [
    (r + 1, pos) for r, _, pos, _ in picks
    if pos in ("K", "DST") and r < ROUNDS - 4
]
assert not early, "K/DST drafted early"

# 4. roster structure met
for tid, cnt in enumerate(counts_list):
    for pos, req in ROSTER_REQ.items():
        if pos == "FLEX":
            continue
        assert cnt[pos] >= req, f"Team {tid} lacks {pos}"
    assert cnt["FLEX"] >= ROSTER_REQ["FLEX"], f"Team {tid} lacks FLEX"

print("Heuristic draft passed all tests.")

def show_final_rosters(pick_log):
    rows = []
    for rnd, team, pos, name in pick_log:
        rows.append(dict(Round=rnd + 1, Team=team, Position=pos, Player=name))

    roster_df = pd.DataFrame(rows)
    for t in range(NUM_TEAMS):
        print(f"\nTeam {t} roster")
        print(
            roster_df[roster_df["Team"] == t]
            .sort_values("Round")
            .reset_index(drop=True)[["Round", "Position", "Player"]]
        )
    return roster_df


roster_df = show_final_rosters(picks)