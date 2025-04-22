import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import os
from pathlib import Path

class StrategyEvaluator:
    def __init__(self, original_data_path):
        self.original_data = pd.read_csv(original_data_path)
        self.metrics = {}
        self.output_dir = Path('evaluation_results').absolute()
        self.output_dir.mkdir(exist_ok=True)
        
    def evaluate_strategy(self, strategy_name, processed_data_path):
        """Evaluate a single strategy"""
        processed_data = pd.read_csv(processed_data_path)
        
        # Ensure DR column exists
        if 'DR' not in processed_data.columns:
            raise ValueError("Processed data must contain 'DR' column")
            
        metrics = {
            'strategy': strategy_name,
            'class_distribution': self._calculate_class_distribution(processed_data),
            'feature_metrics': self._calculate_feature_metrics(processed_data),
            'minority_quality': self._evaluate_minority_quality(processed_data),
            'data_quality': self._assess_data_quality(processed_data)
        }
        
        self.metrics[strategy_name] = metrics
        return metrics
    
    def _calculate_class_distribution(self, processed_data):
        """Calculate class distribution metrics"""
        orig_counts = self.original_data['DR'].value_counts()
        proc_counts = processed_data['DR'].value_counts()
        
        return {
            'original': orig_counts.to_dict(),
            'processed': proc_counts.to_dict(),
            'balance_ratio': proc_counts[1]/proc_counts[0],
            'minority_increase': proc_counts[1]/orig_counts[1]
        }
    
    def _calculate_feature_metrics(self, processed_data):
        """Calculate feature distribution metrics"""
        shared_cols = [c for c in self.original_data.columns 
                      if c in processed_data.columns and c != 'DR']
        
        feature_metrics = {}
        for col in shared_cols:
            orig = self.original_data[col].dropna()
            proc = processed_data[col].dropna()
            
            feature_metrics[col] = {
                'ks_stat': ks_2samp(orig, proc).statistic,
                'wasserstein': wasserstein_distance(orig, proc),
                'mean_diff': abs(orig.mean() - proc.mean())/orig.mean(),
                'std_diff': abs(orig.std() - proc.std())/orig.std()
            }
        
        # Calculate aggregate metrics
        avg_ks = np.mean([m['ks_stat'] for m in feature_metrics.values()])
        avg_wasserstein = np.mean([m['wasserstein'] for m in feature_metrics.values()])
        
        return {
            'per_feature': feature_metrics,
            'avg_ks': avg_ks,
            'avg_wasserstein': avg_wasserstein
        }
    
    def _evaluate_minority_quality(self, processed_data):
        """Evaluate quality of minority class samples"""
        orig_minority = self.original_data[self.original_data['DR'] == 1].drop(columns=['DR'])
        proc_minority = processed_data[processed_data['DR'] == 1].drop(columns=['DR'])
        
        # Align columns and handle missing values
        shared_cols = list(set(orig_minority.columns) & set(proc_minority.columns))
        orig_minority = orig_minority[shared_cols]
        proc_minority = proc_minority[shared_cols]
        
        # Standardize data
        scaler = StandardScaler()
        orig_std = scaler.fit_transform(orig_minority)
        proc_std = scaler.transform(proc_minority)
        
        # Calculate nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=5).fit(orig_std)
        distances, _ = nbrs.kneighbors(proc_std)
        
        # PCA analysis
        pca = PCA(n_components=2)
        combined = np.vstack([orig_std, proc_std])
        pca.fit(combined)
        pca_results = pca.transform(combined)
        
        # Split back into original and processed
        orig_pca = pca_results[:len(orig_minority)]
        proc_pca = pca_results[len(orig_minority):]
        
        return {
            'avg_neighbor_distance': np.mean(distances),
            'pca_variance': pca.explained_variance_ratio_,
            'pca_results': {
                'original': orig_pca.tolist(),
                'processed': proc_pca.tolist()
            }
        }
    
    def _assess_data_quality(self, processed_data):
        """Check for data quality issues"""
        quality_issues = {
            'missing_values': processed_data.isna().sum().sum(),
            'duplicates': processed_data.duplicated().sum(),
            'outliers': self._detect_pca_outliers(processed_data)
        }
        return quality_issues
    
    def _detect_pca_outliers(self, processed_data):
        """Detect outliers in PCA space"""
        minority = processed_data[processed_data['DR'] == 1].drop(columns=['DR'])
        if len(minority) == 0:
            return 0
            
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(minority)
        
        # Calculate Mahalanobis distance
        cov = np.cov(pca_results.T)
        inv_cov = np.linalg.pinv(cov)
        mean = np.mean(pca_results, axis=0)
        
        distances = []
        for point in pca_results:
            distances.append(
                np.sqrt((point - mean).T @ inv_cov @ (point - mean))
            )

        # Consider points beyond 95th percentile as outliers
        threshold = np.percentile(distances, 95)
        return sum(d > threshold for d in distances)
    
    def visualize_comparison(self):
        """Generate comparison visualizations"""
        if not self.metrics:
            raise ValueError("No metrics to visualize - run evaluations first")
            
        ground_truths = [k for k in self.metrics.keys() if k.startswith("GroundTruth")]
        strategies = [k for k in self.metrics.keys() if k not in ground_truths]

        gt_colors = ['#666666', '#999999']  # Dark and light gray for ground truths
        strat_color = 'skyblue'           

        ground_truth_metrics = {
            'class_distribution': self._calculate_class_distribution(self.original_data),
            'feature_metrics': self._calculate_feature_metrics(self.original_data),
            'minority_quality': self._evaluate_minority_quality(self.original_data),
            'data_quality': self._assess_data_quality(self.original_data)
        }
        
        # 1. Class Balance Comparison - Fixed duplicate saving issue
        plt.figure(figsize=(16, 7))

        gt_ratios = [self.metrics[gt]['class_distribution']['balance_ratio'] 
                for gt in ground_truths]
        gt_bars = plt.bar(ground_truths, gt_ratios, color=gt_colors, alpha=0.7)

        strat_ratios = [self.metrics[s]['class_distribution']['balance_ratio'] 
                   for s in strategies]
        strat_bars = plt.bar(strategies, strat_ratios, color=strat_color)
        
        for bars in [gt_bars, strat_bars]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.title('Class Balance Ratio Comparison\n(1.0 = Perfect Balance)', fontsize=14, pad=20)
        plt.ylabel('Minority/Majority Ratio', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=gt_colors[0], label='Ground Truth (DATA_0)'),
            Patch(facecolor=gt_colors[1], label='Ground Truth (DATA_1)'),
            Patch(facecolor=strat_color, label='Synthetic Strategies'),
            Patch(facecolor='red', linestyle='--', label='Ideal Balance')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_balance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()  # Removed show() to prevent display when running script
        
        # 2. Feature Preservation (KS Statistics)
        plt.figure(figsize=(16, 6))

        # Ground truth would be 0 (perfect preservation)
        plt.bar(ground_truths, [0]*len(ground_truths), color=gt_colors, alpha=0.7)

        avg_ks = [m['feature_metrics']['avg_ks'] for m in self.metrics.values()]
        bars = plt.bar(strategies, avg_ks, color='lightgreen')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Feature Preservation (KS Statistic)\n(0 = identical distributions)', fontsize=14)
        plt.ylabel('Average KS Statistic', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_preservation_ks.png', dpi=300, bbox_inches='tight')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=gt_colors[0], label='Ground Truth (DATA_0)'),
            Patch(facecolor=gt_colors[1], label='Ground Truth (DATA_1)'),
            Patch(facecolor=strat_color, label='Synthetic Strategies')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.close()
        
        # 3. Minority Sample Quality
        plt.figure(figsize=(16, 6))

        # Ground truth would be 0 (perfect match)
        plt.bar(ground_truths, [0]*len(ground_truths), color=gt_colors, alpha=0.7)
        
        neighbor_dists = [m['minority_quality']['avg_neighbor_distance'] 
                         for m in self.metrics.values()]
        bars = plt.bar(strategies, neighbor_dists, color='salmon')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Minority Sample Quality\n(Avg. Distance to Original Samples)', fontsize=14)
        plt.ylabel('Distance (lower = better)', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'minority_sample_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. PCA Visualization (for best and worst by neighbor distance)
        best_strat = min(self.metrics.items(), 
                        key=lambda x: x[1]['minority_quality']['avg_neighbor_distance'])[0]
        worst_strat = max(self.metrics.items(), 
                         key=lambda x: x[1]['minority_quality']['avg_neighbor_distance'])[0]
        
        for strat in [best_strat, worst_strat]:
            pca_data = self.metrics[strat]['minority_quality']['pca_results']
            plt.figure(figsize=(8, 6))
            plt.scatter([x[0] for x in pca_data['original']], 
                       [x[1] for x in pca_data['original']], 
                       alpha=0.5, label='Original', color='blue')
            plt.scatter([x[0] for x in pca_data['processed']], 
                       [x[1] for x in pca_data['processed']], 
                       alpha=0.5, label='Processed', color='orange')
            plt.title(f'PCA - {strat}\n(Minority Class Samples)', fontsize=12)
            plt.xlabel('Principal Component 1', fontsize=10)
            plt.ylabel('Principal Component 2', fontsize=10)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f'pca_comparison_{strat}.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_comprehensive_report(self):
        """Generate a complete evaluation report with all metrics"""
        report = {
            "summary_stats": self._get_summary_stats(),
            "detailed_metrics": self.metrics,
            "visualizations": self._generate_all_visualizations()
        }
        
        # Save the full report
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/full_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

    def _get_summary_stats(self):
        """Calculate key summary statistics"""
        summary = []
        for strat, metrics in self.metrics.items():
            summary.append({
                "strategy": strat,
                "class_balance": metrics['class_distribution']['balance_ratio'],
                "feature_preservation": metrics['feature_metrics']['avg_ks'],
                "minority_quality": metrics['minority_quality']['avg_neighbor_distance'],
                "outlier_percentage": metrics['data_quality']['outliers'] / 
                                    metrics['class_distribution']['processed'][1] * 100
            })
        return summary

    def _generate_all_visualizations(self):
        """Generate all visualizations and return their paths"""
        vis_paths = {}
        
        # Class balance
        self.visualize_comparison()  # Using our enhanced version
        vis_paths['class_balance'] = 'evaluation_results/class_balance_enhanced.png'
        
        # Feature preservation radar chart
        self._plot_feature_preservation_radar()
        vis_paths['feature_radar'] = 'evaluation_results/feature_preservation_radar.png'
        
        return vis_paths

    def _plot_feature_preservation_radar(self):
        """Create radar chart of feature preservation"""
        features = list(next(iter(self.metrics.values()))['feature_metrics']['per_feature'].keys())
        stats = ['ks_stat', 'wasserstein', 'mean_diff']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for strat, metrics in self.metrics.items():
            values = []
            for feat in features[:5]:  # Show top 5 features for clarity
                values.append(np.mean([metrics['feature_metrics']['per_feature'][feat][s] for s in stats]))
            
            # Close the radar plot
            values = values + values[:1]
            
            angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, label=strat)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features[:5] + [features[0]])
        ax.set_title('Feature Preservation Radar Chart', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig('evaluation_results/feature_preservation_radar.png', dpi=300)
        plt.show()
    
    def save_results(self, output_dir='evaluation_results'):
        """Save all evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save summary table
        summary_data = []
        for strat, metrics in self.metrics.items():
            row = {
                'Strategy': strat,
                'Class Balance': metrics['class_distribution']['balance_ratio'],
                'Minority Increase': metrics['class_distribution']['minority_increase'],
                'Feature Preservation (KS)': metrics['feature_metrics']['avg_ks'],
                'Feature Preservation (Wasserstein)': metrics['feature_metrics']['avg_wasserstein'],
                'Minority Quality': metrics['minority_quality']['avg_neighbor_distance'],
                'Outliers (%)': metrics['data_quality']['outliers'] / 
                            metrics['class_distribution']['processed'][1] * 100,
                'Missing Values': metrics['data_quality']['missing_values'],
                'Duplicates': metrics['data_quality']['duplicates']
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/enhanced_summary.csv', index=False, float_format='%.3f')
        
        # 3. Generate markdown report
        self._generate_markdown_report(output_dir)