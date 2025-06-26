# Debug script to understand why agent is getting 2387 points

import numpy as np, pandas as pd
from src.fantasyDraftEnv import FantasyDraftEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def debug_agent_draft():
    """Debug a single agent draft to see what's happening."""
    
    board = pd.read_csv("data/processed/training_data_2021.csv")
    
    print("DEBUGGING AGENT'S HIGH SCORING")
    print("=" * 50)
    
    # Load model and create environment
    model = MaskablePPO.load("models/ppo_12_2021_quick")
    
    env = FantasyDraftEnv(
        board_df=board,
        num_teams=12,
        my_slot=6,  # Middle position
        rounds=16,
        roster_slots={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1, "FLEX": 1},
        bench_spots=6,
    )
    
    wrapped_env = ActionMasker(env, lambda e: e.get_action_mask())
    
    # Run one draft with detailed logging
    obs, info = wrapped_env.reset()
    done = False
    pick_count = 0
    
    print("AGENT'S DRAFT PICKS:")
    print("-" * 40)
    
    while not done:
        action, _ = model.predict(obs, deterministic=False, action_masks=info["action_mask"])
        
        # Get player info before taking action
        player_idx = int(action)
        player_info = env.board.iloc[player_idx]
        
        obs, reward, done, _, info = wrapped_env.step(action)
        pick_count += 1
        
        print(f"Pick {pick_count:2d}: {player_info['name']:20s} ({player_info['position']}) "
              f"- {player_info['fantasy_points']:6.1f} pts (ADP: {player_info['adp']:6.1f})")
    
    # Analyze the final roster
    print("\n" + "=" * 50)
    print("FINAL ROSTER ANALYSIS:")
    print("-" * 40)
    
    my_picks = env.my_picks
    final_roster = env.board.iloc[my_picks]
    
    print(f"Total players drafted: {len(my_picks)}")
    print(f"Total fantasy points (all players): {final_roster['fantasy_points'].sum():.1f}")
    
    # Break down by position
    print(f"\nROSTER BY POSITION:")
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        pos_players = final_roster[final_roster['position'] == pos]
        if len(pos_players) > 0:
            total_pos_points = pos_players['fantasy_points'].sum()
            print(f"{pos:3s}: {len(pos_players)} players, {total_pos_points:6.1f} total pts")
            for _, player in pos_players.iterrows():
                print(f"     {player['name']:20s} - {player['fantasy_points']:6.1f} pts")
    
    # Calculate optimal lineup using _lineup_points
    optimal_score = env._lineup_points(env.board, my_picks)
    print(f"\nOPTIMAL LINEUP SCORE: {optimal_score:.1f}")
    
    # Manually verify the optimal lineup calculation
    print(f"\nMANUAL LINEUP VERIFICATION:")
    roster = env.board.loc[my_picks]
    pos_groups = {p: roster[roster["position"] == p].sort_values("fantasy_points", ascending=False)
                  for p in ["QB", "RB", "WR", "TE", "K", "DST"]}
    
    # Required starters
    lineup_total = 0
    lineup_players = []
    
    # QB (1), TE (1), K (1), DST (1)
    for pos, req in [("QB", 1), ("TE", 1), ("K", 1), ("DST", 1)]:
        if len(pos_groups[pos]) >= req:
            starters = pos_groups[pos].head(req)
            lineup_total += starters["fantasy_points"].sum()
            for _, p in starters.iterrows():
                lineup_players.append(f"{pos}: {p['name']} - {p['fantasy_points']:.1f}")
    
    # RB (2), WR (3)
    for pos, req in [("RB", 2), ("WR", 3)]:
        if len(pos_groups[pos]) >= req:
            starters = pos_groups[pos].head(req)
            lineup_total += starters["fantasy_points"].sum()
            for _, p in starters.iterrows():
                lineup_players.append(f"{pos}: {p['name']} - {p['fantasy_points']:.1f}")
    
    # FLEX (1) - best remaining RB/WR/TE
    flex_pool = pd.concat([
        pos_groups["RB"].iloc[2:],  # RBs beyond RB1/RB2
        pos_groups["WR"].iloc[3:],  # WRs beyond WR1/WR2/WR3
        pos_groups["TE"].iloc[1:]   # TEs beyond TE1
    ]).sort_values("fantasy_points", ascending=False)
    
    if len(flex_pool) > 0:
        flex_player = flex_pool.iloc[0]
        lineup_total += flex_player["fantasy_points"]
        lineup_players.append(f"FLEX: {flex_player['name']} ({flex_player['position']}) - {flex_player['fantasy_points']:.1f}")
    
    print(f"Manual calculation total: {lineup_total:.1f}")
    print(f"_lineup_points() result:  {optimal_score:.1f}")
    print(f"Difference: {abs(optimal_score - lineup_total):.1f}")
    
    print(f"\nOPTIMAL STARTING LINEUP:")
    for player in lineup_players:
        print(f"  {player}")
    
    # Compare to realistic expectations
    print(f"\n" + "=" * 50)
    print("REALITY CHECK:")
    print(f"Agent's score: {optimal_score:.1f}")
    print(f"Baseline (avg): 1680.0")
    print(f"Elite 2021 lineup: ~3120")
    print(f"Very good lineup: ~2000-2200")
    
    if optimal_score > 2800:
        print("🚨 Score suspiciously high - possible bug")
    elif optimal_score > 2200:
        print("😮 Excellent score - agent found elite players")
    elif optimal_score > 1800:
        print("😊 Good score - agent performing well")
    else:
        print("😐 Average score - room for improvement")
    
    return optimal_score, final_roster

if __name__ == "__main__":
    score, roster = debug_agent_draft()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"If agent consistently scores {score:.0f}+, it's either:")
    print(f"1. Genuinely learned optimal fantasy strategy")
    print(f"2. Found a bug/exploit in the environment")
    print(f"3. Overfitting to 2021 data patterns")