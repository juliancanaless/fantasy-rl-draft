import pandas as pd
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.fantasyDraftEnv import FantasyDraftEnv
import warnings
import time

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Kernel._parent_header.*')

class OptimizedMultiYearDraftEnv(FantasyDraftEnv):
    """
    Optimized environment that pre-processes templates to reduce reset() overhead.
    """
    
    def __init__(self, year_data_dict, year_weights, **kwargs):
        """
        Args:
            year_data_dict: {year: dataframe} mapping
            year_weights: {year: weight} for sampling
        """
        print("Pre-processing board templates...")
        
        self.year_data = year_data_dict
        self.years = list(year_data_dict.keys())
        
        # Convert weights to probabilities
        total_weight = sum(year_weights.values())
        self.year_probs = [year_weights[year] / total_weight for year in self.years]
        
        # Pre-process board templates
        self.board_templates = {}
        for year, df in year_data_dict.items():
            template = (
                df[["name", "position", "adp", "fantasy_points"]]
                .copy()
                .sort_values("adp")
                .head(300)
                .reset_index(drop=True)
            )
            template["available"] = True
            self.board_templates[year] = template
            print(f"  {year}: {len(template)} players processed")
        
        self.current_year = None
        
        # Initialize with first year
        super().__init__(board_df=year_data_dict[self.years[0]], **kwargs)
        print("Environment optimization complete!")

    def reset(self, *, seed=None, options=None):
        """Reset with pre-processed templates for faster performance."""
        super().reset(seed=seed, options=options)
        
        # Randomize draft position each episode
        self.my_slot = np.random.randint(0, self.num_teams)
        
        # Sample year for this episode
        self.current_year = np.random.choice(self.years, p=self.year_probs)
        
        # Use pre-processed template
        self.board = self.board_templates[self.current_year].copy()
        
        # Reset counters
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
    """Load data for all years."""
    print("Loading training data...")
    data_dir = Path("data/processed")
    
    year_data = {}
    for year in [2021, 2022, 2023]:
        file_path = data_dir / f"training_data_{year}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            year_data[year] = df
            print(f"  {year}: {len(df)} players loaded")
        else:
            raise FileNotFoundError(f"Missing data for {year}: {file_path}")
    
    # Load test data
    test_file = data_dir / "training_data_2024.csv"
    test_data = None
    if test_file.exists():
        test_data = pd.read_csv(test_file)
        print(f"  2024 test: {len(test_data)} players loaded")
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
    
    print("Creating multi-year environment...")
    env = OptimizedMultiYearDraftEnv(
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

def benchmark_environment_speed():
    """Quick benchmark to test environment performance."""
    print("Benchmarking environment speed...")
    env, _ = create_training_environment()
    
    # Time 50 resets
    reset_times = []
    for i in range(50):
        start = time.time()
        env.reset()
        reset_times.append(time.time() - start)
    
    avg_reset_time = np.mean(reset_times)
    print(f"Average reset time: {avg_reset_time:.4f}s")
    
    if avg_reset_time > 0.01:
        print("Warning: Reset time is high - environment may be bottleneck")
    else:
        print("Reset time looks good!")
    
    # Estimate iterations per second
    estimated_its_per_sec = 1.0 / (avg_reset_time * 2048 / 1000)
    print(f"Estimated training speed: ~{estimated_its_per_sec:.0f} its/sec")
    
    return env

def train_multi_year_model():
    """Train the multi-year generalization model."""
    
    # Quick benchmark first
    env = benchmark_environment_speed()
    
    # Create model
    print("Creating PPO model...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        device="cuda",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Starting multi-year training...")
    print("Training 1M timesteps...")
    
    start_time = time.time()
    
    # Train for 1M timesteps
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="multi_year_optimized",
        progress_bar=True
    )
    
    total_time = time.time() - start_time
    its_per_sec = 1_000_000 / total_time
    
    print(f"Training complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average speed: {its_per_sec:.0f} its/sec")
    
    # Save model
    model_path = "models/ppo_multi_year_optimized"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return model, None

if __name__ == "__main__":
    model, test_data = train_multi_year_model()