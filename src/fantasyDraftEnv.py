# fantasy_draft_env.py - PERFORMANCE OPTIMIZED VERSION
#
# Key optimizations:
# 1. Cache baseline points (calculate once)
# 2. Pre-filter position dataframes
# 3. Minimize DataFrame operations in hot paths
# 4. Use numpy operations where possible

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
TIER_GAP_THRESHOLD = 20.0  # ADP gap that indicates tier break
# ---------------------------------------------------------------------------


class FantasyDraftEnv(gym.Env):
    metadata = {"render_modes": []}

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

        # 🚀 MAJOR OPTIMIZATION: Pre-calculate baseline once!
        print("🔧 Pre-calculating baseline (one-time cost)...")
        self._cached_baseline = self._calculate_baseline_once()
        print(f"✅ Baseline cached: {self._cached_baseline:.1f} points")
        
        # 🚀 OPTIMIZATION: Pre-filter position dataframes
        self._position_masks = {}
        for pos in BASE_POS:
            self._position_masks[pos] = (self._board_template["position"] == pos).values

        # -------------------- spaces --------------------------------------
        self.action_space = spaces.Discrete(len(self._board_template))
        top_shape = (FORESIGHT_K, 2)  # ADP + tier_gap columns
        self.observation_space = spaces.Dict({
            "roster": spaces.Box(0.0, 1.0, (len(BASE_POS),), np.float32),
            "roster_needs": spaces.Box(0.0, 1.0, (len(BASE_POS),), np.float32),
            **{
                p: spaces.Box(-1.0, 1.0, top_shape, np.float32)
                for p in BASE_POS
            },
        })

        # -------------------- runtime -------------------------------------
        self.board: pd.DataFrame | None = None
        self.roster_counts: Dict[str, int] = {}
        self.pick_global = 0
        self.my_picks: List[int] = []
        self._curr_lineup_pts = 0.0

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
        idx = int(action)
        if idx >= len(self.board) or not self.board["available"].iat[idx]:
            return self._build_obs(), -10.0, True, False, {}

        # my pick ----------------------------------------------------------
        self.board["available"].iat[idx] = False
        pos = self.board["position"].iat[idx]
        self.roster_counts[pos] += 1
        self.my_picks.append(idx)
        self.pick_global += 1

        # 🚀 OPTIMIZED: Use fast lineup calculation
        new_pts = self._fast_lineup_points()
        reward  = (new_pts - self._curr_lineup_pts) / DENSE_SCALE
        self._curr_lineup_pts = new_pts

        # opponents --------------------------------------------------------
        self._simulate_opponents()

        done = self.pick_global >= self.total_picks
        if done:
            # 🚀 OPTIMIZATION: Use cached baseline instead of recalculating
            final_reward = new_pts - self._cached_baseline
            reward += final_reward / self.lineup_scale
            
        return self._build_obs(), reward, done, False, {"action_mask": self.get_action_mask()}

    def _build_obs(self):
        """🚀 OPTIMIZED observation builder using pre-computed masks."""
        roster_vec = np.array(
            [self.roster_counts[p] / self.rounds for p in BASE_POS], np.float32
        )

        needs_vec = self._get_roster_needs()

        # 🚀 OPTIMIZATION: Use numpy boolean indexing instead of DataFrame queries
        available_mask = self.board["available"].values
        
        tables = {}
        for i, pos in enumerate(BASE_POS):
            # Combine position mask with availability mask
            pos_available_mask = self._position_masks[pos] & available_mask
            pos_indices = np.where(pos_available_mask)[0]
            
            if len(pos_indices) == 0:
                tables[pos] = np.full((FORESIGHT_K, 2), -1.0, np.float32)
                continue
            
            # Get top K available players for this position (already sorted by ADP)
            top_indices = pos_indices[:FORESIGHT_K]
            
            # Extract ADP values directly from numpy array
            adp_values = self.board["adp"].iloc[top_indices].values.astype(np.float32)
            adp_col = adp_values / MAX_ADP
            
            # Calculate tier gaps
            tier_gaps = np.zeros_like(adp_col)
            for j in range(len(adp_col) - 1):
                raw_gap = adp_values[j + 1] - adp_values[j]
                tier_gaps[j] = min(raw_gap / TIER_GAP_THRESHOLD, 1.0)
            
            # Stack into 2-column array [ADP, tier_gap]
            features = np.column_stack([adp_col, tier_gaps])
            
            # Pad if necessary
            if len(features) < FORESIGHT_K:
                pad_rows = FORESIGHT_K - len(features)
                pad = np.full((pad_rows, 2), -1.0, np.float32)
                features = np.vstack([features, pad])
                
            tables[pos] = features

        return {
            "roster": roster_vec,
            "roster_needs": needs_vec,
            **tables
        }
    
    def get_action_mask(self) -> np.ndarray:
        """🚀 OPTIMIZED action mask using numpy operations."""
        mask = self.board["available"].values.copy()
        
        # Vectorized position limit checking
        for i, pos in enumerate(self.board["position"]):
            if not mask[i]:
                continue
                
            current_count = self.roster_counts[pos]
            max_allowed = self.pos_max.get(pos, 99)
            
            if current_count >= max_allowed:
                mask[i] = False
        
        return mask.astype(bool)

    def _fast_lineup_points(self) -> float:
        """🚀 OPTIMIZED lineup calculation - matches original logic exactly."""
        if not self.my_picks:
            return 0.0
            
        # Use the exact same logic as original _lineup_points
        roster = self.board.loc[self.my_picks]
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

    def _calculate_baseline_once(self) -> float:
        """🚀 OPTIMIZATION: Calculate baseline once and cache it.
        
        IMPORTANT: Excludes agent's slot to ensure fair comparison.
        Only averages the heuristic bot performance, not agent performance.
        """
        board = self._board_template.copy()
        board["available"] = True
        counts = [
            {p: 0 for p in BASE_POS} | {"FLEX": 0}
            for _ in range(self.num_teams)
        ]

        all_team_picks = [[] for _ in range(self.num_teams)]
        
        for pick in range(self.total_picks):
            round_idx = pick // self.num_teams
            pick_in_round = pick % self.num_teams
            
            # Proper snake draft order
            if round_idx % 2 == 0:
                tid = pick_in_round
            else:
                tid = self.num_teams - 1 - pick_in_round
                
            idx = self._heuristic_pick(board, counts[tid], pick)
            all_team_picks[tid].append(idx)

        # 🎯 FAIRNESS FIX: Only calculate baseline from opponent teams
        # Exclude agent's slot (self.my_slot) from baseline calculation
        opponent_scores = []
        for i, team_picks in enumerate(all_team_picks):
            if i != self.my_slot:  # Skip agent's slot
                score = self._lineup_points_static(board, team_picks)
                opponent_scores.append(score)
        
        baseline = float(np.mean(opponent_scores))
        print(f"🎯 Fair baseline calculated from {len(opponent_scores)} opponent teams: {baseline:.1f}")
        return baseline

    def _lineup_points_static(self, board: pd.DataFrame, idx_list: List[int]) -> float:
        """Static version of lineup calculation for baseline."""
        if not idx_list:
            return 0.0
            
        roster = board.iloc[idx_list]
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

    # Keep all the other methods the same (opponent simulation, etc.)
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

    def _get_roster_needs(self) -> np.ndarray:
        """Calculate roster needs - kept same as original."""
        needs = np.zeros(len(BASE_POS), dtype=np.float32)
        
        round_idx = self.pick_global // self.num_teams
        late_draft = round_idx >= self.rounds - 4
        very_late = round_idx >= self.rounds - 2
        
        flex_filled = 0
        for pos in FLEX_POS:
            required = self.roster_req.get(pos, 0)
            current = self.roster_counts[pos]
            if current > required:
                flex_filled += (current - required)
        
        flex_needed = max(0, self.roster_req.get("FLEX", 0) - flex_filled)
        
        for i, pos in enumerate(BASE_POS):
            required = self.roster_req.get(pos, 0)
            current = self.roster_counts[pos]
            
            if pos in ("K", "DST"):
                if current >= required:
                    needs[i] = 0.0
                elif very_late:
                    needs[i] = 1.0
                elif late_draft:
                    needs[i] = 0.7
                else:
                    needs[i] = 0.0
            elif current < required:
                needs[i] = 1.0
            elif pos in FLEX_POS and flex_needed > 0:
                needs[i] = 0.7
            elif pos in ("RB", "WR") and current < 4:
                needs[i] = 0.3
            elif pos == "QB" and current < 2:
                needs[i] = 0.2
            else:
                needs[i] = 0.0
                    
        return needs
