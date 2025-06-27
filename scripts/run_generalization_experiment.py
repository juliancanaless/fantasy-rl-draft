# run_generalization_experiment.py - One-click generalization testing

import subprocess
import sys
from pathlib import Path

def check_data_availability():
    """Check if all required data files exist."""
    data_dir = Path("data/processed")
    required_files = [
        "training_data_2021.csv",
        "training_data_2022.csv", 
        "training_data_2023.csv",
        "training_data_2024.csv"
    ]
    
    missing = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required data files:")
        for file in missing:
            print(f"  - {file}")
        print("\nPlease ensure all data files are in data/processed/")
        return False
    
    print("✅ All required data files found!")
    return True

def run_training():
    """Run the multi-year training."""
    print("\n🚀 PHASE 1: Multi-Year Training")
    print("=" * 50)
    print("Training on 2021-2023 with recency bias (50%/35%/15%)")
    print("This will take several hours...")
    
    try:
        result = subprocess.run([
            sys.executable, "-c",
            "from src.multi_year_training import train_multi_year_model; train_multi_year_model()"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed:")
        print(f"Error: {e.stderr}")
        return False

def run_evaluation():
    """Run the generalization evaluation."""
    print("\n🧪 PHASE 2: Generalization Testing")
    print("=" * 50)
    print("Testing on 2024 data (unseen during training)...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/eval_generalization.py"
        ], check=True)
        
        print("✅ Evaluation completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def main():
    """Run the complete generalization experiment."""
    print("🎯 FANTASY DRAFT GENERALIZATION EXPERIMENT")
    print("=" * 60)
    print("This will:")
    print("1. Train on 2021-2023 data (recency weighted)")
    print("2. Test generalization on 2024 data")
    print("3. Compare performance vs baseline")
    
    # Check prerequisites
    if not check_data_availability():
        return
    
    print(f"\nReady to proceed? This will take several hours...")
    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Experiment cancelled.")
        return
    
    # Run training
    if not run_training():
        print("Stopping due to training failure.")
        return
    
    # Run evaluation  
    if not run_evaluation():
        print("Training succeeded but evaluation failed.")
        return
    
    print("\n🎉 EXPERIMENT COMPLETE!")
    print("=" * 50)
    print("Check the output above for generalization results.")
    print("Detailed results saved in: results/generalization_results.json")
    
    # Quick summary
    try:
        import json
        with open("results/generalization_results.json", "r") as f:
            results = json.load(f)
        
        if "2024" in results:
            test_result = results["2024"]
            print(f"\n📊 QUICK SUMMARY:")
            print(f"2024 Test Performance: {test_result['improvement']:+.1f} pts ({test_result['improvement_pct']:+.1f}%)")
            if test_result['improvement'] > 0:
                print("🎉 Agent generalized successfully!")
            else:
                print("😟 Agent failed to generalize - likely overfitting")
    except:
        pass

if __name__ == "__main__":
    main()