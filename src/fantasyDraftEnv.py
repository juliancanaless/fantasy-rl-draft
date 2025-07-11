
from __future__ import annotations
import numpy as np, pandas as pd, gymnasium as gym
from gymnasium import spaces
from typing import Dict, List

# ---------------------------------------------------------------------------
BASE_POS = ["QB", "RB", "WR", "TE", "K", "DST"]
FLEX_POS = {"WR", "RB", "TE"}
FORESIGHT_K = 5
MAX_ADP = 300
MAX_PLAYER_PTS = 450.0
DENSE_SCALE = 10.0
TIER_GAP_THRESHOLD = 20.0  # ADP gap that indicates tier break
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
        idx = int(action)
        if idx >= len(self.board) or not self.board["available"].iat[idx]:
            return self._build_obs(), -10.0, True, False, {}

        # my pick ----------------------------------------------------------
        self.board["available"].iat[idx] = False
        pos = self.board["position"].iat[idx]
        self.roster_counts[pos] += 1
        self.my_picks.append(idx)
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

        # roster needs
        needs_vec = self._get_roster_needs()

        tables = {}
        for pos in BASE_POS:
            avail = (
                self.board[self.board["available"] & (self.board["position"] == pos)]
                .sort_values("adp")
                .head(FORESIGHT_K)
            )
            
            if len(avail) == 0:
                # No players available at this position
                tables[pos] = np.full((FORESIGHT_K, 2), -1.0, np.float32)
                continue
                
            # ADP column (normalized)
            adp_col = avail["adp"].to_numpy(np.float32) / MAX_ADP
            
            # Tier gap column (normalized ADP difference to next player)
            tier_gaps = self._calculate_tier_gaps(avail)
            
            # Stack into 2-column array [ADP, tier_gap]
            features = np.column_stack([adp_col, tier_gaps])
            
            # Pad
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
        """Boolean mask that prevents invalid picks (unavailable + position limits)"""
        mask = self.board["available"].values.copy()
        
        # Enforce position limits
        for i, pos in enumerate(self.board["position"]):
            if not mask[i]:  # Already unavailable
                continue
                
            current_count = self.roster_counts[pos]
            max_allowed = self.pos_max.get(pos, 99)
            
            if current_count >= max_allowed:
                mask[i] = False  # Block this pick
        
        return mask.astype(bool)

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
        """Calculate baseline as AVERAGE of all teams in a heuristic draft."""
        board = self._board_template.copy()
        board["available"] = True
        counts = [
            {p: 0 for p in BASE_POS} | {"FLEX": 0}
            for _ in range(self.num_teams)
        ]

        # Track picks for ALL teams, not just mine
        all_team_picks = [[] for _ in range(self.num_teams)]
        
        for pick in range(self.total_picks):
            round_idx = pick // self.num_teams
            pick_in_round = pick % self.num_teams
            
            # FIXED: Proper snake draft order
            if round_idx % 2 == 0:
                tid = pick_in_round
            else:
                tid = self.num_teams - 1 - pick_in_round
                
            # Make pick for current team
            idx = self._heuristic_pick(board, counts[tid], pick)
            all_team_picks[tid].append(idx)

        # Calculate lineup points for ALL teams
        team_scores = []
        for team_picks in all_team_picks:
            score = self._lineup_points(board, team_picks)
            team_scores.append(score)
        
        # Return AVERAGE across all teams
        return float(np.mean(team_scores))


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

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------
    def _get_roster_needs(self) -> np.ndarray:
        """
        Calculate urgency flags for each position based on:
        1. Required starter slots still unfilled
        2. FLEX eligibility 
        3. Late-round K/DST timing (special case)
        4. Round context (early vs late draft)
        
        Returns float array [0.0, 1.0] where:
        - 1.0 = urgent need (missing required starter)
        - 0.7 = high need (FLEX eligible, late K/DST)
        - 0.3 = moderate need (depth/value play)
        - 0.0 = no immediate need
        """
        needs = np.zeros(len(BASE_POS), dtype=np.float32)
        
        round_idx = self.pick_global // self.num_teams
        late_draft = round_idx >= self.rounds - 4  # Last 4 rounds
        very_late = round_idx >= self.rounds - 2   # Last 2 rounds
        
        # Calculate current FLEX usage
        flex_filled = 0
        for pos in FLEX_POS:
            # Count excess players beyond required starters as potential FLEX
            required = self.roster_req.get(pos, 0)
            current = self.roster_counts[pos]
            if current > required:
                flex_filled += (current - required)
        
        flex_needed = max(0, self.roster_req.get("FLEX", 0) - flex_filled)
        
        for i, pos in enumerate(BASE_POS):
            required = self.roster_req.get(pos, 0)
            current = self.roster_counts[pos]
            
            # SPECIAL CASE: K/DST timing logic (overrides basic requirement logic)
            if pos in ("K", "DST"):
                if current >= required:
                    needs[i] = 0.0  # Already have enough
                elif very_late:
                    needs[i] = 1.0  # Urgent in final 2 rounds
                elif late_draft:
                    needs[i] = 0.7  # High need in final 4 rounds
                else:
                    needs[i] = 0.0  # Don't draft early, even if "required"
                    
            # REGULAR POSITIONS: Standard requirement logic
            elif current < required:
                needs[i] = 1.0  # Urgent: missing required starters
                
            # FLEX-eligible positions when FLEX needed
            elif pos in FLEX_POS and flex_needed > 0:
                needs[i] = 0.7  # High need for FLEX fill
                
            # DEPTH CONSIDERATIONS
            elif pos in ("RB", "WR") and current < 4:  # Want RB/WR depth
                needs[i] = 0.3  # Moderate need for depth
                
            elif pos == "QB" and current < 2:  # Limited QB depth
                needs[i] = 0.2  # Low need for QB2
                
            else:
                needs[i] = 0.0  # No immediate need
                    
        return needs
    
    def _calculate_tier_gaps(self, position_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate normalized tier gaps for position rankings.
        
        Large gaps indicate tier breaks where waiting might be costly.
        Small gaps suggest similar player quality.
        
        Returns array where:
        - 1.0 = large gap (major tier break)
        - 0.5 = moderate gap  
        - 0.0 = small gap (similar tier)
        """
        adps = position_df["adp"].to_numpy()
        gaps = np.zeros_like(adps, dtype=np.float32)
        
        for i in range(len(adps) - 1):
            raw_gap = adps[i + 1] - adps[i]
            
            # Normalize gap by threshold and cap at 1.0
            normalized_gap = min(raw_gap / TIER_GAP_THRESHOLD, 1.0)
            gaps[i] = normalized_gap
            
        # Last player has no "next" player, so gap = 0
        if len(gaps) > 0:
            gaps[-1] = 0.0
            
        return gaps