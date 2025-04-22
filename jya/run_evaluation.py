import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

from evaluator import StrategyEvaluator
from config_hyperparameters import STRATEGIES
import json
import os
import pandas as pd

def main():
    # Paths configuration
    print(f"Will save results to: {os.path.abspath('evaluation_results')}")
    if not os.path.exists('evaluation_results'):
        os.makedirs('evaluation_results')
        print("Created evaluation_results directory")
    else:
        print("Directory already exists")
    # original_data_path = "./DATA_1/training_set/training_data.csv"

    # Initialize evaluator with primary ground truth
    evaluator = StrategyEvaluator("./GROUND_TRUTH_DATA0/training_set/training_data.csv")

    ground_truth_paths = {
        "GroundTruth_DATA_0": "./GROUND_TRUTH_DATA0/training_set/training_data.csv",
        "GroundTruth_DATA_1": "./GROUND_TRUTH_DATA1/training_set/training_data.csv"
    }

    for gt_name, gt_path in ground_truth_paths.items():
            print(f"\nEvaluating {gt_name}...")
            if os.path.exists(gt_path):
                # For ground truths, compare against themselves (no processing)
                evaluator.metrics[gt_name] = {
                    'strategy': gt_name,
                    'class_distribution': evaluator._calculate_class_distribution(pd.read_csv(gt_path)),
                    'feature_metrics': evaluator._calculate_feature_metrics(pd.read_csv(gt_path)),
                    'minority_quality': evaluator._evaluate_minority_quality(pd.read_csv(gt_path)),
                    'data_quality': evaluator._assess_data_quality(pd.read_csv(gt_path))
                }
            else:
                print(f"Warning: Ground truth not found - {gt_path}")

    data_dirs = {

        # All with smote applied
        "DATA_1": "./DATA_1_1/synthetic_training_set/synthetic_data_0_1000_TVAESynthesizer.csv",
        "DATA_2": "./DATA_1_2/synthetic_training_set/synthetic_data_0_1000_TVAESynthesizer.csv",
        "DATA_3": "./DATA_1_3/synthetic_training_set/synthetic_data_0_2500_TVAESynthesizer.csv",
        "DATA_4": "./DATA_1_4/synthetic_training_set/synthetic_data_0_1000_TVAESynthesizer.csv",
        "DATA_5": "./DATA_1_5/synthetic_training_set/synthetic_data_0_1000_CTGANSynthesizer.csv",
        "DATA_6": "./DATA_1_6/synthetic_training_set/synthetic_data_0_2500_CTGANSynthesizer.csv",
        "DATA_7": "./DATA_1_7/synthetic_training_set/synthetic_data_0_2500_CTGANSynthesizer.csv",
        "DATA_8": "./DATA_1_8/synthetic_training_set/synthetic_data_0_1000_CTGANSynthesizer.csv",
        "DATA_9": "./DATA_9/synthetic_training_set/synthetic_data_0_1000_TVAESynthesizer.csv",
        "DATA_10": "./DATA_10/synthetic_training_set/synthetic_data_0_1000_CTGANSynthesizer.csv"
    }
    
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