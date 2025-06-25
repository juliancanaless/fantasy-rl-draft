# fantasy_draft_env.py
#
# Minimal fantasy-football snake-draft environment for gymnasium.
# Works with stable-baselines3 (or sb3-contrib MaskablePPO).
#
# Observation space:
#   └─ Dict {
#         "roster": int[6]      – counts of QB,RB,WR,TE,K,DST drafted so far
#         pos tables (6 keys)   – float32[5,2] containing [fant_pts, adp]
#                                for top 5 available at that position,
#                                padded with -1.0 when fewer than 5 remain
#       }
#
# Action space:
#   Discrete(N) where N = total rows in board_df
#   At each step only indices for still-available players are legal.
#
# Reward:
#   0 during draft, final reward =
#       my_roster_total_points  –  adp_auto_pick_baseline_points
#
# Author: 2025-06

from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List

# ---------------------------------------------------------------------------
# helper constants
# ---------------------------------------------------------------------------
BASE_POS = ["QB", "RB", "WR", "TE", "K", "DST"]
FLEX_POS = {"WR", "RB", "TE"}          # fixed per spec
FORESIGHT_K = 5                        # top-window size


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
class FantasyDraftEnv(gym.Env):
    metadata = {"render_modes": []}

    # -----------------------------------------------------------------------
    # constructor
    # -----------------------------------------------------------------------
    def __init__(
        self,
        board_df: pd.DataFrame,
        *,
        num_teams: int,
        my_slot: int,
        rounds: int = 16,
        roster_slots: Dict[str, int],
        bench_spots: int = 6,
    ):
        """
        board_df: DataFrame with at least columns
                  ['name','position','adp','fantasy_points']
        num_teams:  e.g. 8,10,12
        my_slot:    1-based draft position of the agent (1 … num_teams)
        rounds:     total draft rounds (starting + bench)
        roster_slots: starting lineup counts, e.g. {"QB":1,"RB":2,"WR":3,...}
        bench_spots: number of bench slots (not enforced in v1 logic)
        """
        super().__init__()

        if not 1 <= my_slot <= num_teams:
            raise ValueError("my_slot must be between 1 and num_teams inclusive")

        # immutable board template (sorted by ADP ascending)
        self._board_template = (
            board_df[["name", "position", "adp", "fantasy_points"]]
            .copy()
            .sort_values("adp")
            .reset_index(drop=True)
        )

        # config
        self.num_teams   = num_teams
        self.rounds      = rounds
        self.total_picks = num_teams * rounds
        self.my_slot     = my_slot - 1            # 0-based inside env
        self.roster_req  = roster_slots           # not strictly enforced yet
        self.bench_spots = bench_spots

        # action / observation spaces -------------------------------------------------
        self.action_space = spaces.Discrete(len(self._board_template))

        top_shape = (FORESIGHT_K, 2)              # [fant_pts, adp] rows
        self.observation_space = spaces.Dict(
            {
                "roster": spaces.Box(low=0, high=rounds, shape=(len(BASE_POS),), dtype=np.int32),
                **{
                    pos: spaces.Box(low=-1.0, high=1000.0, shape=top_shape, dtype=np.float32)
                    for pos in BASE_POS
                },
            }
        )

        # runtime state (populated in reset) ------------------------------------------
        self.board: pd.DataFrame | None = None
        self.roster_counts: Dict[str, int] = {}
        self.pick_global: int = 0

    # -----------------------------------------------------------------------
    # gymnasium methods
    # -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.board = self._board_template.copy()
        self.board["available"] = True

        # one roster-counter per team (agent handled separately)
        self.opp_counts = [
            {pos: 0 for pos in BASE_POS} | {"FLEX": 0}
            for _ in range(self.num_teams)
        ]

        self.roster_counts = {p: 0 for p in BASE_POS}
        self.pick_global = 0
        self._skip_to_my_turn()

        observation = self._build_obs()
        info = {"action_mask": self.get_action_mask()}
        return observation, info

    def step(self, action: int):
        assert self.board is not None

        avail_idx = np.flatnonzero(self.board["available"].values)
        if action >= len(avail_idx):
            # Illegal action – terminate with large negative reward
            terminated = True
            truncated = False
            reward = -10.0
            return self._build_obs(), reward, terminated, truncated, {}

        # draft the chosen player
        player_row = self.board.iloc[avail_idx[action]]
        self.board.at[player_row.name, "available"] = False
        self.roster_counts[player_row["position"]] += 1
        self.pick_global += 1

        # opponents pick until it's my next turn or draft ends
        self._simulate_opponents()

        terminated = self.pick_global >= self.total_picks
        truncated = False
        reward = self._final_reward() if terminated else 0.0

        observation = self._build_obs()
        info = {"action_mask": self.get_action_mask()}
        return observation, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # MaskablePPO support
    # -----------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        """Boolean mask the same length as action_space.n"""
        return self.board["available"].values.astype(bool)

    # -----------------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------------
    def _build_obs(self):
        # roster counts vector
        roster_vec = np.array(
            [self.roster_counts[p] for p in BASE_POS], dtype=np.int32
        )

        tables: Dict[str, np.ndarray] = {}
        for pos in BASE_POS:
            avail = (
                self.board[self.board["available"] & (self.board["position"] == pos)]
                .sort_values("adp")
                .head(FORESIGHT_K)
            )
            tbl = avail[["fantasy_points", "adp"]].to_numpy(dtype=np.float32)
            if len(tbl) < FORESIGHT_K:
                pad = np.full((FORESIGHT_K - len(tbl), 2), -1.0, dtype=np.float32)
                tbl = np.vstack([tbl, pad])
            tables[pos] = tbl

        return {"roster": roster_vec, **tables}

    # -----------------------------------------------------------------------
    # smarter opponents
    # -----------------------------------------------------------------------
    def _simulate_opponents(self):
        """
        Each opponent:
        1. Fills required starting slots   (roster_slots dict)
        2. Fills FLEX when possible        (FLEX = WR/RB/TE)
        3. Fills bench with best ADP
        """
        while self.pick_global < self.total_picks:
            team_idx = self.pick_global % self.num_teams
            if team_idx == self.my_slot:
                # my next turn
                break

            self._opponent_pick(team_idx)
            self.pick_global += 1


    def _opponent_pick(self, team_idx: int):
        """
        Human-like drafting heuristic:
        1.  Fill required starters (except K/DST) before bench.
        2.  Leave K and DST until the last 4 rounds.
        3.  Fill FLEX after all other starters are filled.
        4.  Draft bench depth with a bias toward RB and WR.
        """
        counts = self.opp_counts[team_idx]

        # ---------- context ----------
        round_idx  = self.pick_global // self.num_teams        # 0-based
        late4      = round_idx >= self.rounds - 4              # last 4 rounds?
        late2      = round_idx >= self.rounds - 2              # last 2 rounds?

        # ---------- 1) starters still needed (no K/DST yet) ----------
        needed_pos = [
            pos for pos, req in self.roster_req.items()
            if pos not in ("FLEX", "K", "DST") and counts[pos] < req
        ]

        # unlock K/DST starters only in last 4 rounds
        if late4:
            for pos in ("K", "DST"):
                if pos in self.roster_req and counts[pos] < self.roster_req[pos]:
                    needed_pos.append(pos)

        flex_needed = counts["FLEX"] < self.roster_req.get("FLEX", 0)

        # ---------- 2) build candidate position set ----------
        if needed_pos:
            cand = set(needed_pos)
        elif flex_needed:
            cand = {"WR", "RB", "TE"}
        else:
            cand = {"RB", "WR", "TE", "QB"}           # bench depth bias
            if late4:
                cand |= {"K", "DST"}

        # force missing K/DST in the final 2 rounds
        if late2:
            if "K" in self.roster_req and counts["K"] == 0:
                cand.add("K")
            if "DST" in self.roster_req and counts["DST"] == 0:
                cand.add("DST")

        # ---------- 3) pick best-ADP player among candidates ----------
        avail_mask = self.board["available"] & self.board["position"].isin(cand)
        if not avail_mask.any():
            avail_mask = self.board["available"]          # fall back, should rarely happen
            if not avail_mask.any():
                return                                    # board exhausted, safeguard

        top_idx = self.board[avail_mask].index[0]         # board is ADP-sorted
        self.board.at[top_idx, "available"] = False
        chosen_pos = self.board.at[top_idx, "position"]

        # ---------- 4) update that team’s counters ----------
        if chosen_pos in self.roster_req and counts[chosen_pos] < self.roster_req[chosen_pos]:
            counts[chosen_pos] += 1                       # filled a starter
        elif flex_needed and chosen_pos in {"WR", "RB", "TE"}:
            counts["FLEX"] += 1                           # filled FLEX
        else:
            counts[chosen_pos] += 1                       # bench depth




    def _skip_to_my_turn(self):
        """Skip initial opponent picks before my first turn."""
        while self.pick_global < self.total_picks:
            if (self.pick_global % self.num_teams) == self.my_slot:
                break
            top_idx = self.board[self.board["available"]].index[0]
            self.board.at[top_idx, "available"] = False
            self.pick_global += 1

    # -----------------------------------------------------------------------
    # reward helpers
    # -----------------------------------------------------------------------
    def _final_reward(self) -> float:
        """Season-points delta vs deterministic ADP baseline."""
        my_pts = self._roster_points()
        baseline_pts = self._baseline_points()
        return float(my_pts - baseline_pts)

    def _roster_points(self) -> float:
        drafted = self.board[~self.board["available"]]
        return drafted["fantasy_points"].sum()

    def _baseline_points(self) -> float:
        """Drafts a roster from my_slot using the same heuristic as opponents
        and returns its 2024 season-points total."""
        # fresh copy of board & counters
        board = self._board_template.copy()
        board["available"] = True

        opp_counts = [
            {pos: 0 for pos in BASE_POS} | {"FLEX": 0}
            for _ in range(self.num_teams)
        ]
        roster_idx = []                         # rows drafted by baseline team
        pick = 0
        while pick < self.total_picks:
            team_idx = pick % self.num_teams

            if team_idx == self.my_slot:
                # baseline team picks with heuristic
                top_idx = self._heuristic_pick(
                    board, opp_counts[self.my_slot], pick
                )
                roster_idx.append(top_idx)
            else:
                # opponent pick with same heuristic
                self._heuristic_pick(board, opp_counts[team_idx], pick)

            pick += 1

        return board.loc[roster_idx, "fantasy_points"].sum()


    # --- helper that mirrors _opponent_pick but stateless -------------------
    def _heuristic_pick(self, board: pd.DataFrame, counts: dict, pick_num: int):
        round_idx = pick_num // self.num_teams
        late4     = round_idx >= self.rounds - 4
        late2     = round_idx >= self.rounds - 2

        needed = [
            p for p, req in self.roster_req.items()
            if p not in ("FLEX", "K", "DST") and counts[p] < req
        ]
        if late4:
            for p in ("K", "DST"):
                if p in self.roster_req and counts[p] < self.roster_req[p]:
                    needed.append(p)

        flex_needed = counts["FLEX"] < self.roster_req.get("FLEX", 0)

        if needed:
            cand = set(needed)
        elif flex_needed:
            cand = {"WR", "RB", "TE"}
        else:
            cand = {"RB", "WR", "TE", "QB"}
            if late4:
                cand |= {"K", "DST"}
        if late2:
            if "K" in self.roster_req and counts["K"] == 0:
                cand.add("K")
            if "DST" in self.roster_req and counts["DST"] == 0:
                cand.add("DST")

        mask = board["available"] & board["position"].isin(cand)
        if not mask.any():                       # fall back
            mask = board["available"]
        top_idx = board[mask].index[0]
        board.at[top_idx, "available"] = False

        pos = board.at[top_idx, "position"]
        if pos in self.roster_req and counts[pos] < self.roster_req[pos]:
            counts[pos] += 1
        elif flex_needed and pos in {"WR", "RB", "TE"}:
            counts["FLEX"] += 1
        else:
            counts[pos] += 1

        return top_idx

