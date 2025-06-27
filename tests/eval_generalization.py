# tests/eval_generalization.py - Test model on unseen 2024 data

import numpy as np
import pandas as pd
import tqdm
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.fantasyDraftEnv import FantasyDraftEnv

def evaluate_on_year(model, board_data, year_name, num_episodes=300):
    """Evaluate model performance on a specific year's data."""
    
    def run_episode():
        # Random draft position each episode
        slot = np.random.randint(1, 13)
        
        env = FantasyDraftEnv(
            board_df=board_data,
            num_teams=12,
            my_slot=slot,
            rounds=16,
            roster_slots={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1, "FLEX": 1},
            bench_spots=6,
        )
        wrapped_env = ActionMasker(env, lambda e: e.get_action_mask())
        
        # Run the draft
        obs, info = wrapped_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=False,
                action_masks=info["action_mask"]
            )
            obs, _, done, _, info = wrapped_env.step(action)
        
        # Get final scores
        agent_score = wrapped_env.unwrapped._lineup_points(
            wrapped_env.unwrapped.board,
            wrapped_env.unwrapped.my_picks
        )
        baseline_score = wrapped_env.unwrapped._baseline_points()
        
        return agent_score, baseline_score, slot
    
    print(f"Evaluating {year_name} ({num_episodes} episodes)...")
    
    agent_scores = []
    baseline_scores = []
    positions = []
    
    for _ in tqdm.tqdm(range(num_episodes)):
        agent_score, baseline_score, position = run_episode()
        agent_scores.append(agent_score)
        baseline_scores.append(baseline_score)
        positions.append(position)
    
    return {
        "year": year_name,
        "agent_scores": agent_scores,
        "baseline_scores": baseline_scores,
        "positions": positions,
        "agent_mean": np.mean(agent_scores),
        "baseline_mean": np.mean(baseline_scores),
        "improvement": np.mean(agent_scores) - np.mean(baseline_scores),
        "improvement_pct": (np.mean(agent_scores) / np.mean(baseline_scores) - 1) * 100,
        "win_rate": np.mean([a > b for a, b in zip(agent_scores, baseline_scores)])
    }

def full_generalization_test():
    """Run complete generalization evaluation."""
    
    # Load the multi-year trained model
    model_path = "models/ppo_multi_year_generalization"
    if not Path(model_path + ".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = MaskablePPO.load(model_path)
    
    # Load all available data
    data_dir = Path("data/processed")
    results = {}
    
    # Test on all years (including training years for comparison)
    test_years = [2021, 2022, 2023, 2024]
    
    for year in test_years:
        file_path = data_dir / f"training_data_{year}.csv"
        if file_path.exists():
            board_data = pd.read_csv(file_path)
            year_results = evaluate_on_year(model, board_data, year)
            results[year] = year_results
            
            print(f"\n{year} Results:")
            print(f"  Agent: {year_results['agent_mean']:.1f} pts")
            print(f"  Baseline: {year_results['baseline_mean']:.1f} pts") 
            print(f"  Improvement: {year_results['improvement']:+.1f} pts ({year_results['improvement_pct']:+.1f}%)")
            print(f"  Win Rate: {year_results['win_rate']:.1%}")
            
            # Mark if this was training or test data
            if year in [2021, 2022, 2023]:
                print(f"  (Training data)")
            else:
                print(f"  (Test data - GENERALIZATION)")
    
    # Summary analysis
    print(f"\nGENERALIZATION ANALYSIS")
    print("=" * 50)
    
    if 2024 in results:
        test_result = results[2024]
        train_results = [results[y] for y in [2021, 2022, 2023] if y in results]
        
        if train_results:
            avg_train_improvement = np.mean([r['improvement'] for r in train_results])
            test_improvement = test_result['improvement']
            
            print(f"Training years avg improvement: {avg_train_improvement:+.1f} pts")
            print(f"Test year (2024) improvement: {test_improvement:+.1f} pts")
            print(f"Generalization gap: {test_improvement - avg_train_improvement:+.1f} pts")
            
            # Assessment
            if test_improvement > 0:
                if test_improvement > avg_train_improvement * 0.8:
                    print(f"EXCELLENT: Strong generalization!")
                elif test_improvement > avg_train_improvement * 0.5:
                    print(f"GOOD: Decent generalization")
                else:
                    print(f"WEAK: Poor generalization")
            else:
                print(f"FAILED: No generalization (worse than baseline)")
    
    return results

if __name__ == "__main__":
    results = full_generalization_test()
    
    # Save detailed results
    if results:
        import json
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        json_results = {}
        for year, result in results.items():
            json_results[str(year)] = {
                "agent_mean": result["agent_mean"],
                "baseline_mean": result["baseline_mean"], 
                "improvement": result["improvement"],
                "improvement_pct": result["improvement_pct"],
                "win_rate": result["win_rate"]
            }
        
        with open(results_dir / "generalization_results.json", "w") as f:
            json.dump(json_results, f, indent=2)