import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

from evaluator import StrategyEvaluator
from config_hyperparameters import STRATEGIES
import json
import os

def main():
    # Paths configuration
    print(f"Will save results to: {os.path.abspath('evaluation_results')}")
    if not os.path.exists('evaluation_results'):
        os.makedirs('evaluation_results')
        print("Created evaluation_results directory")
    else:
        print("Directory already exists")
    original_data_path = "./DATA_0/training_set/training_data.csv"
    data_dirs = {
        "DATA_1": "./DATA_1/synthetic_training_set/synthetic_data_0_1000_TVAE.csv",
        "DATA_2": "./DATA_2/synthetic_training_set/synthetic_data_0_1000_TVAE.csv",
        "DATA_3": "./DATA_3/synthetic_training_set/synthetic_data_0_2500_TVAE.csv",
        "DATA_4": "./DATA_4/synthetic_training_set/synthetic_data_0_1000_TVAE.csv",
        "DATA_5": "./DATA_5/synthetic_training_set/synthetic_data_0_1000_TVAE.csv",
        "DATA_6": "./DATA_6/synthetic_training_set/synthetic_data_0_2500_TVAE.csv",
        "DATA_7": "./DATA_7/synthetic_training_set/synthetic_data_0_2500_TVAE.csv",
        "DATA_8": "./DATA_8/synthetic_training_set/synthetic_data_0_1000_TVAE.csv"
    }
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(original_data_path)
    
    # Evaluate each strategy
    for strategy_name, data_path in data_dirs.items():
        print(f"\nEvaluating {strategy_name}...")
        if os.path.exists(data_path):
            evaluator.evaluate_strategy(strategy_name, data_path)
        else:
            print(f"Warning: Path not found - {data_path}")
    
    # Generate visualizations and save results
    evaluator.visualize_comparison()
    evaluator.save_results()
    
    print("\nEvaluation complete. Results saved to evaluation_results/")

if __name__ == "__main__":
    main()