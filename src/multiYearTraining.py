# src/multi_year_training.py - Complete pipeline for multi-year training

import pandas as pd
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.fantasyDraftEnv import FantasyDraftEnv

class MultiYearDraftEnv(FantasyDraftEnv):
    """
    Modified environment that samples from multiple years during training.
    Each episode randomly selects a year based on specified weights.
    """
    
    def __init__(self, year_data_dict, year_weights, **kwargs):
        """
        Args:
            year_data_dict: {year: dataframe} mapping
            year_weights: {year: weight} for sampling (e.g., {2023: 0.5, 2022: 0.35, 2021: 0.15})
        """
        self.year_data = year_data_dict
        self.years = list(year_data_dict.keys())
        
        # Convert weights to probabilities
        total_weight = sum(year_weights.values())
        self.year_probs = [year_weights[year] / total_weight for year in self.years]
        
        self.current_year = None
        
        # Initialize with first year (will change on reset)
        super().__init__(board_df=year_data_dict[self.years[0]], **kwargs)
        

    
    def reset(self, *, seed=None, options=None):
        """Reset with randomly sampled year based on weights."""
        super().reset(seed=seed, options=options)
        
        # Randomize draft position each episode
        self.my_slot = np.random.randint(0, self.num_teams)  # 0-based internal representation
        
        # Sample year for this episode
        self.current_year = np.random.choice(self.years, p=self.year_probs)
        
        # Update board template with selected year
        self._board_template = (
            self.year_data[self.current_year][["name", "position", "adp", "fantasy_points"]]
            .copy()
            .sort_values("adp")
            .head(300)  # MAX_ADP
            .reset_index(drop=True)
        )
        
        # Reinitialize environment state
        self.board = self._board_template.copy()
        self.board["available"] = True
        
        self.opp_counts = [
            {p: 0 for p in ["QB", "RB", "WR", "TE", "K", "DST"]} | {"FLEX": 0} 
            for _ in range(self.num_teams)
        ]
        self.roster_counts = {p: 0 for p in ["QB", "RB", "WR", "TE", "K", "DST"]}
        self.my_picks.clear()
        self.pick_global = 0
        self._curr_lineup_pts = 0.0
        
        self._skip_to_my_turn()
        
        return self._build_obs(), {"action_mask": self.get_action_mask(), "year": self.current_year}

def load_and_combine_data():
    """Load data for all years and return organized structure."""
    data_dir = Path("data/processed")
    
    year_data = {}
    for year in [2021, 2022, 2023]:
        file_path = data_dir / f"training_data_{year}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            year_data[year] = df
        else:
            raise FileNotFoundError(f"Missing data for {year}: {file_path}")
    
    # Load test data
    test_file = data_dir / "training_data_2024.csv"
    test_data = None
    if test_file.exists():
        test_data = pd.read_csv(test_file)
    else:
        raise FileNotFoundError(f"Missing 2024 test data: {test_file}")
    
    return year_data, test_data

def create_training_environment():
    """Create the multi-year training environment."""
    year_data, test_data = load_and_combine_data()
    
    if len(year_data) < 3:
        raise ValueError("Need data for 2021, 2022, and 2023")
    
    # Recency bias weights: 50% 2023, 35% 2022, 15% 2021
    year_weights = {2023: 0.50, 2022: 0.35, 2021: 0.15}
    
    env = MultiYearDraftEnv(
        year_data_dict=year_data,
        year_weights=year_weights,
        num_teams=12,
        my_slot=1,  # Will be randomized on each reset
        rounds=16,
        roster_slots={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1, "FLEX": 1},
        bench_spots=6,
    )
    
    wrapped_env = ActionMasker(env, lambda e: e.get_action_mask())
    return wrapped_env, test_data

def train_multi_year_model():
    """Train the multi-year generalization model."""
    
    # Create environment
    env, test_data = create_training_environment()
    
    # Create model
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device="cuda",  # Use GPU if available
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Training multi-year model (2M timesteps)...")
    
    # Train for 2M timesteps
    model.learn(
        total_timesteps=2_000_000,
        tb_log_name="multi_year_generalization",
        progress_bar=True
    )
    
    # Save model
    model_path = "models/ppo_multi_year_generalization"
    model.save(model_path)
    
    return model, test_data

if __name__ == "__main__":
    # Run the training
    model, test_data = train_multi_year_model()