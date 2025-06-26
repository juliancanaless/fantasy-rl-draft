# fantasy_draft_env.py
#
# Observation space (no leakage of fantasy-points):
#   Dict{
#       "roster" : float32[6]             – counts / rounds (QB…DST)
#       pos keys : float32[5,1]           – ADP ÷ MAX_ADP of top-5 avail
#   }
#
# Reward = Δ(best-line-up points)/10 each pick  +  final_points / lineup_scale
# ---------------------------------------------------------------------------

from __future__ import annotations
import numpy as np, pandas as pd, gymnasium as gym
from gymnasium import spaces
from typing import Dict, List

# ---------------------------------------------------------------------------
BASE_POS    = ["QB", "RB", "WR", "TE", "K", "DST"]
FLEX_POS    = {"WR", "RB", "TE"}
FORESIGHT_K = 5
MAX_ADP     = 300
MAX_PLAYER_PTS = 450.0
DENSE_SCALE    = 10.0
# ---------------------------------------------------------------------------


class FantasyDraftEnv(gym.Env):
    metadata = {"render_modes": []}

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
        super().__init__()
        if not 1 <= my_slot <= num_teams:
            raise ValueError("my_slot must be between 1 and num_teams inclusive")

        self._board_template = (
            board_df[["name", "position", "adp", "fantasy_points"]]
            .copy()
            .sort_values("adp")
            .head(MAX_ADP)
            .reset_index(drop=True)
        )

        self.num_teams   = num_teams
        self.rounds      = rounds
        self.total_picks = num_teams * rounds
        self.my_slot     = my_slot - 1
        self.roster_req  = roster_slots
        self.bench_spots = bench_spots
        self.pos_max     = {"QB": 2, "TE": 3, "K": 1, "DST": 1}

        start_slots = sum(v for k, v in roster_slots.items() if k != "FLEX") \
                      + roster_slots.get("FLEX", 0)
        self.lineup_scale = MAX_PLAYER_PTS * start_slots

        # -------------------- spaces --------------------------------------
        self.action_space = spaces.Discrete(len(self._board_template))
        top_shape = (FORESIGHT_K, 1)                         # ADP only
        self.observation_space = spaces.Dict(
            {
                "roster": spaces.Box(0.0, 1.0, (len(BASE_POS),), np.float32),
                **{
                    p: spaces.Box(-1.0, 1.0, top_shape, np.float32)
                    for p in BASE_POS
                },
            }
        )

        # -------------------- runtime -------------------------------------
        self.board: pd.DataFrame | None = None
        self.roster_counts: Dict[str, int] = {}
        self.pick_global = 0
        self.my_picks: List[int] = []
        self._curr_lineup_pts = 0.0

    # ======================================================================
    # gymnasium API
    # ======================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._board_template.copy()
        self.board["available"] = True

        self.opp_counts = [
            {p: 0 for p in BASE_POS} | {"FLEX": 0} for _ in range(self.num_teams)
        ]
        self.roster_counts = {p: 0 for p in BASE_POS}
        self.my_picks.clear()
        self.pick_global = 0
        self._curr_lineup_pts = 0.0
        self._skip_to_my_turn()
        return self._build_obs(), {"action_mask": self.get_action_mask()}

    def step(self, action: int):
        if action >= len(self.board) or not self.board.at[action, "available"]:
            return self._build_obs(), -10.0, True, False, {}

        # my pick ----------------------------------------------------------
        self.board["available"].iat[action] = False
        pos = self.board["position"].iat[action]
        self.roster_counts[pos] += 1
        self.my_picks.append(action)
        self.pick_global += 1

        # dense reward -----------------------------------------------------
        new_pts = self._lineup_points(self.board, self.my_picks)
        reward  = (new_pts - self._curr_lineup_pts) / DENSE_SCALE
        self._curr_lineup_pts = new_pts

        # opponents --------------------------------------------------------
        self._simulate_opponents()

        done = self.pick_global >= self.total_picks
        if done:
            reward += new_pts / self.lineup_scale
        return self._build_obs(), reward, done, False, {"action_mask": self.get_action_mask()}

    # ======================================================================
    # observation builder
    # ======================================================================
    def _build_obs(self):
        roster_vec = np.array(
            [self.roster_counts[p] / self.rounds for p in BASE_POS], np.float32
        )

        tables = {}
        for pos in BASE_POS:
            avail = (
                self.board[self.board["available"] & (self.board["position"] == pos)]
                .sort_values("adp")
                .head(FORESIGHT_K)
            )
            col = avail["adp"].to_numpy(np.float32).reshape(-1, 1) / MAX_ADP
            if len(col) < FORESIGHT_K:
                pad = np.full((FORESIGHT_K - len(col), 1), -1.0, np.float32)
                col = np.vstack([col, pad])
            tables[pos] = col

        return {"roster": roster_vec, **tables}
    
    def get_action_mask(self) -> np.ndarray:
        """Boolean mask the same length as action_space.n"""
        return self.board["available"].values.astype(bool)

    # -----------------------------------------------------------------------
    # smarter opponents
    # -----------------------------------------------------------------------
    def _simulate_opponents(self):
        while self.pick_global < self.total_picks:
            tid = self.pick_global % self.num_teams
            if tid == self.my_slot:
                break
            self._opponent_pick(tid)
            self.pick_global += 1

    def _opponent_pick(self, team_idx: int):
        counts = self.opp_counts[team_idx]
        round_idx = self.pick_global // self.num_teams
        late4 = round_idx >= self.rounds - 4
        late2 = round_idx >= self.rounds - 2

        need = [p for p, req in self.roster_req.items()
                if p not in ("FLEX", "K", "DST") and counts[p] < req]
        if late4:
            for p in ("K", "DST"):
                if counts[p] < self.roster_req[p]:
                    need.append(p)

        flex_needed = counts["FLEX"] < self.roster_req.get("FLEX", 0)

        if need:
            cand = set(need)
        elif flex_needed:
            cand = {"WR", "RB", "TE"}
        else:
            cand = {"RB", "WR", "TE", "QB"}
            if late4:
                cand |= {"K", "DST"}

        if late2:
            if counts["K"] == 0:
                cand.add("K")
            if counts["DST"] == 0:
                cand.add("DST")

        cand = {p for p in cand if counts[p] < self.pos_max.get(p, 99)}
        if not cand:
            return

        mask = self.board["available"] & self.board["position"].isin(cand)
        if not mask.any():
            return
        idx = self.board[mask].index[0]
        self.board.at[idx, "available"] = False
        pos = self.board.at[idx, "position"]

        if pos in self.roster_req and counts[pos] < self.roster_req[pos]:
            counts[pos] += 1
        elif flex_needed and pos in FLEX_POS:
            counts["FLEX"] += 1
        else:
            counts[pos] += 1

    # -----------------------------------------------------------------------
    # reward helpers
    # -----------------------------------------------------------------------
    def _lineup_points(self, board: pd.DataFrame, idx_list: List[int]) -> float:
        roster = board.loc[idx_list]
        pos_gp = {p: roster[roster["position"] == p]
                          .sort_values("fantasy_points", ascending=False)
                  for p in BASE_POS}

        total = 0.0
        for pos, req in self.roster_req.items():
            if pos == "FLEX":
                continue
            total += pos_gp.get(pos, pd.DataFrame())["fantasy_points"].head(req).sum()

        flex_n = self.roster_req.get("FLEX", 0)
        if flex_n:
            flex_pool = pd.concat(
                [pos_gp[p].iloc[self.roster_req.get(p, 0):] for p in FLEX_POS],
                axis=0
            ).sort_values("fantasy_points", ascending=False)
            total += flex_pool["fantasy_points"].head(flex_n).sum()
        return float(total)

    def _final_reward(self) -> float:
        return self._roster_points() - self._baseline_points()

    def _roster_points(self) -> float:
        return self._lineup_points(self.board, self.my_picks)

    def _baseline_points(self) -> float:
        board = self._board_template.copy()
        board["available"] = True
        counts = [
            {p: 0 for p in BASE_POS} | {"FLEX": 0}
            for _ in range(self.num_teams)
        ]

        my_rows: List[int] = []
        for pick in range(self.total_picks):
            tid = pick % self.num_teams
            if tid == self.my_slot:
                idx = self._heuristic_pick(board, counts[tid], pick)
                my_rows.append(idx)
            else:
                self._heuristic_pick(board, counts[tid], pick)

        return self._lineup_points(board, my_rows)

    # -----------------------------------------------------------------------
    # _heuristic_pick 
    # -----------------------------------------------------------------------
    def _heuristic_pick(self, board: pd.DataFrame, counts: dict, pick_num: int):
        round_idx = pick_num // self.num_teams
        late4 = round_idx >= self.rounds - 4
        late2 = round_idx >= self.rounds - 2

        need = [p for p, r in self.roster_req.items()
                if p not in ("FLEX", "K", "DST") and counts[p] < r]
        if late4:
            for p in ("K", "DST"):
                if counts[p] < self.roster_req[p]:
                    need.append(p)

        flex_needed = counts["FLEX"] < self.roster_req.get("FLEX", 0)

        if need:
            cand = set(need)
        elif flex_needed:
            cand = {"WR", "RB", "TE"}
        else:
            cand = {"RB", "WR", "TE", "QB"}
            if late4:
                cand |= {"K", "DST"}

        if late2:
            if counts["K"] == 0:
                cand.add("K")
            if counts["DST"] == 0:
                cand.add("DST")

        cand = {p for p in cand if counts[p] < self.pos_max.get(p, 99)}
        if not cand:
            cand = {p for p in BASE_POS if counts[p] < self.pos_max.get(p, 99)}

        mask = board["available"] & board["position"].isin(cand)
        if not mask.any():
            mask = board["available"]
        idx = board[mask].index[0]
        board.at[idx, "available"] = False

        pos = board.at[idx, "position"]
        if pos in self.roster_req and counts[pos] < self.roster_req[pos]:
            counts[pos] += 1
        elif flex_needed and pos in FLEX_POS:
            counts["FLEX"] += 1
        else:
            counts[pos] += 1

        return idx

    def _skip_to_my_turn(self):
        while self.pick_global < self.total_picks:
            if (self.pick_global % self.num_teams) == self.my_slot:
                break
            top_idx = self.board[self.board["available"]].index[0]
            self.board.at[top_idx, "available"] = False
            self.pick_global += 1
