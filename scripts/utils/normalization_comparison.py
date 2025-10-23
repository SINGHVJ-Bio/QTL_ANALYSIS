#!/usr/bin/env python3
"""
Normalization Comparison Visualization
Creates side-by-side plots of raw vs normalized data for QTL analysis
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

class NormalizationComparison:
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        self.comparison_dir = os.path.join(results_dir, "normalization_comparison")
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C5C5C5']
        
    def generate_comprehensive_comparison(self, qtl_type, raw_data, normalized_data, normalization_method):
        """Generate comprehensive comparison plots for normalization"""
        logger.info(f"📊 Generating normalization comparison for {qtl_type}...")
        
        try:
            # Create subdirectory for this QTL type
            qtl_dir = os.path.join(self.comparison_dir, qtl_type)
            os.makedirs(qtl_dir, exist_ok=True)
            
            comparison_results = {
                'qtl_type': qtl_type,
                'normalization_method': normalization_method,
                'plots_generated': []
            }
            
            # 1. Distribution comparison
            dist_plot = self.create_distribution_comparison(raw_data, normalized_data, qtl_type, normalization_method, qtl_dir)
            if dist_plot:
                comparison_results['plots_generated'].append('distribution_comparison')
            
            # 2. Sample-wise comparison
            sample_plot = self.create_sample_comparison(raw_data, normalized_data, qtl_type, normalization_method, qtl_dir)
            if sample_plot:
                comparison_results['plots_generated'].append('sample_comparison')
            
            # 3. Feature-wise comparison
            feature_plot = self.create_feature_comparison(raw_data, normalized_data, qtl_type, normalization_method, qtl_dir)
            if feature_plot:
                comparison_results['plots_generated'].append('feature_comparison')
            
            # 4. Statistical summary comparison
            stats_plot = self.create_statistical_comparison(raw_data, normalized_data, qtl_type, normalization_method, qtl_dir)
            if stats_plot:
                comparison_results['plots_generated'].append('statistical_comparison')
            
            # 5. Interactive 3D PCA comparison
            pca_plot = self.create_pca_comparison(raw_data, normalized_data, qtl_type, normalization_method, qtl_dir)
            if pca_plot:
                comparison_results['plots_generated'].append('pca_comparison')
            
            # 6. Correlation structure comparison
            corr_plot = self.create_correlation_comparison(raw_data, normalized_data, qtl_type, normalization_method, qtl_dir)
            if corr_plot:
                comparison_results['plots_generated'].append('correlation_comparison')
            
            # 7. Generate comprehensive HTML report
            html_report = self.generate_comparison_html_report(comparison_results, qtl_type, qtl_dir)
            comparison_results['html_report'] = html_report
            
            logger.info(f"✅ Normalization comparison completed for {qtl_type}: {len(comparison_results['plots_generated'])} plots generated")
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Error generating normalization comparison for {qtl_type}: {e}")
            return {}
    
    def create_distribution_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create distribution comparison plots"""
        try:
            # Select random features for visualization
            n_features_to_plot = min(20, raw_data.shape[0])
            if n_features_to_plot == 0:
                return None
                
            feature_indices = np.random.choice(raw_data.index, n_features_to_plot, replace=False)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{qtl_type.upper()} Normalization: {method}\nDistribution Comparison', fontsize=16, fontweight='bold')
            
            # Plot 1: Overall distribution density
            self._plot_overall_density(raw_data, normalized_data, axes[0, 0], qtl_type)
            
            # Plot 2: Feature-wise distributions (sample a few features)
            self._plot_feature_distributions(raw_data, normalized_data, feature_indices, axes[0, 1], qtl_type)
            
            # Plot 3: Box plot comparison
            self._plot_box_comparison(raw_data, normalized_data, axes[1, 0], qtl_type)
            
            # Plot 4: QQ plot comparison
            self._plot_qq_comparison(raw_data, normalized_data, axes[1, 1], qtl_type)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_distribution_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create interactive version
            interactive_file = self._create_interactive_distribution(raw_data, normalized_data, qtl_type, method, output_dir)
            
            return {'static': plot_file, 'interactive': interactive_file}
            
        except Exception as e:
            logger.warning(f"Could not create distribution comparison: {e}")
            return None
    
    def _plot_overall_density(self, raw_data, normalized_data, ax, qtl_type):
        """Plot overall density comparison"""
        try:
            # Flatten data for density plot
            raw_flat = raw_data.values.flatten()
            norm_flat = normalized_data.values.flatten()
            
            # Remove outliers for better visualization
            raw_flat = raw_flat[~np.isnan(raw_flat)]
            norm_flat = norm_flat[~np.isnan(norm_flat)]
            
            # Limit to reasonable range
            if len(raw_flat) > 0:
                raw_mean, raw_std = np.mean(raw_flat), np.std(raw_flat)
                raw_flat = raw_flat[(raw_flat > raw_mean - 3*raw_std) & (raw_flat < raw_mean + 3*raw_std)]
            
            if len(norm_flat) > 0:
                norm_mean, norm_std = np.mean(norm_flat), np.std(norm_flat)
                norm_flat = norm_flat[(norm_flat > norm_mean - 3*norm_std) & (norm_flat < norm_mean + 3*norm_std)]
            
            ax.hist(raw_flat, bins=50, alpha=0.7, color=self.colors[0], density=True, label='Raw')
            ax.hist(norm_flat, bins=50, alpha=0.7, color=self.colors[1], density=True, label='Normalized')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title('Overall Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_feature_distributions(self, raw_data, normalized_data, feature_indices, ax, qtl_type):
        """Plot distributions for sample features"""
        try:
            for i, feature in enumerate(feature_indices[:5]):  # Plot first 5 features
                if feature in raw_data.index and feature in normalized_data.index:
                    raw_values = raw_data.loc[feature].dropna()
                    norm_values = normalized_data.loc[feature].dropna()
                    
                    if len(raw_values) > 0 and len(norm_values) > 0:
                        # Normalize for comparison
                        raw_norm = (raw_values - raw_values.mean()) / raw_values.std() if raw_values.std() > 0 else raw_values
                        norm_norm = (norm_values - norm_values.mean()) / norm_values.std() if norm_values.std() > 0 else norm_values
                        
                        ax.hist(raw_norm, bins=20, alpha=0.5, color=self.colors[0], 
                               density=True, label=f'Raw {feature}' if i == 0 else "")
                        ax.hist(norm_norm, bins=20, alpha=0.5, color=self.colors[1], 
                               density=True, label=f'Norm {feature}' if i == 0 else "")
            
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Density')
            ax.set_title('Feature Distributions (Sample)')
            if len(feature_indices) > 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_box_comparison(self, raw_data, normalized_data, ax, qtl_type):
        """Plot box plot comparison"""
        try:
            # Sample features for box plot
            n_sample = min(10, raw_data.shape[0])
            if n_sample > 0:
                sample_features = np.random.choice(raw_data.index, n_sample, replace=False)
                
                # Prepare data for box plot
                box_data_raw = []
                box_data_norm = []
                labels = []
                
                for feature in sample_features:
                    if feature in raw_data.index:
                        raw_values = raw_data.loc[feature].dropna()
                        if len(raw_values) > 0:
                            box_data_raw.append(raw_values.values)
                            labels.append(f'R_{feature[:8]}')
                    
                    if feature in normalized_data.index:
                        norm_values = normalized_data.loc[feature].dropna()
                        if len(norm_values) > 0:
                            box_data_norm.append(norm_values.values)
                            labels.append(f'N_{feature[:8]}')
                
                # Combine data
                all_data = box_data_raw + box_data_norm
                
                if len(all_data) > 0:
                    ax.boxplot(all_data, labels=labels)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel('Value')
                    ax.set_title('Box Plot Comparison\n(R=Raw, N=Normalized)')
                    ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_qq_comparison(self, raw_data, normalized_data, ax, qtl_type):
        """Plot QQ plot comparison"""
        try:
            # Flatten data
            raw_flat = raw_data.values.flatten()
            norm_flat = normalized_data.values.flatten()
            
            raw_flat = raw_flat[~np.isnan(raw_flat)]
            norm_flat = norm_flat[~np.isnan(norm_flat)]
            
            if len(raw_flat) > 1000 and len(norm_flat) > 1000:
                # Sample for performance
                raw_flat = np.random.choice(raw_flat, 1000, replace=False)
                norm_flat = np.random.choice(norm_flat, 1000, replace=False)
            
            if len(raw_flat) > 10 and len(norm_flat) > 10:
                # Raw data QQ plot
                raw_sorted = np.sort(raw_flat)
                raw_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(raw_sorted)))
                
                # Normalized data QQ plot
                norm_sorted = np.sort(norm_flat)
                norm_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(norm_sorted)))
                
                ax.scatter(raw_theoretical, raw_sorted, alpha=0.6, color=self.colors[0], 
                          s=20, label='Raw')
                ax.scatter(norm_theoretical, norm_sorted, alpha=0.6, color=self.colors[1], 
                          s=20, label='Normalized')
                
                # Add diagonal line
                min_val = min(raw_theoretical.min(), norm_theoretical.min(), 
                            raw_sorted.min(), norm_sorted.min())
                max_val = max(raw_theoretical.max(), norm_theoretical.max(), 
                            raw_sorted.max(), norm_sorted.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')
                ax.set_title('QQ Plot Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _create_interactive_distribution(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create interactive distribution plot"""
        try:
            # Sample data for performance
            n_samples = min(1000, raw_data.shape[0] * raw_data.shape[1])
            if n_samples == 0:
                return None
                
            # Flatten and sample
            raw_flat = raw_data.values.flatten()
            norm_flat = normalized_data.values.flatten()
            
            raw_flat = raw_flat[~np.isnan(raw_flat)]
            norm_flat = norm_flat[~np.isnan(norm_flat)]
            
            if len(raw_flat) > n_samples:
                raw_flat = np.random.choice(raw_flat, n_samples, replace=False)
            if len(norm_flat) > n_samples:
                norm_flat = np.random.choice(norm_flat, n_samples, replace=False)
            
            # Create interactive histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=raw_flat, 
                name='Raw Data',
                opacity=0.7,
                nbinsx=50,
                marker_color=self.colors[0]
            ))
            
            fig.add_trace(go.Histogram(
                x=norm_flat, 
                name='Normalized Data',
                opacity=0.7,
                nbinsx=50,
                marker_color=self.colors[1]
            ))
            
            fig.update_layout(
                title=f'{qtl_type.upper()} Distribution Comparison<br><sub>Normalization Method: {method}</sub>',
                xaxis_title='Value',
                yaxis_title='Count',
                barmode='overlay',
                height=600
            )
            
            plot_file = os.path.join(output_dir, f"{qtl_type}_interactive_distribution.html")
            fig.write_html(plot_file)
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create interactive distribution: {e}")
            return None
    
    def create_sample_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create sample-wise comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{qtl_type.upper()} Normalization: {method}\nSample-wise Comparison', fontsize=16, fontweight='bold')
            
            # Plot 1: Sample means
            self._plot_sample_means(raw_data, normalized_data, axes[0, 0])
            
            # Plot 2: Sample variances
            self._plot_sample_variances(raw_data, normalized_data, axes[0, 1])
            
            # Plot 3: Sample missingness
            self._plot_sample_missingness(raw_data, normalized_data, axes[1, 0])
            
            # Plot 4: Sample correlation
            self._plot_sample_correlation(raw_data, normalized_data, axes[1, 1])
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_sample_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create sample comparison: {e}")
            return None
    
    def _plot_sample_means(self, raw_data, normalized_data, ax):
        """Plot sample means comparison"""
        try:
            raw_means = raw_data.mean(axis=0)
            norm_means = normalized_data.mean(axis=0)
            
            ax.scatter(range(len(raw_means)), raw_means, alpha=0.7, color=self.colors[0], 
                      label='Raw', s=30)
            ax.scatter(range(len(norm_means)), norm_means, alpha=0.7, color=self.colors[1], 
                      label='Normalized', s=30)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Mean Value')
            ax.set_title('Sample Means')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_sample_variances(self, raw_data, normalized_data, ax):
        """Plot sample variances comparison"""
        try:
            raw_vars = raw_data.var(axis=0)
            norm_vars = normalized_data.var(axis=0)
            
            ax.scatter(range(len(raw_vars)), raw_vars, alpha=0.7, color=self.colors[0], 
                      label='Raw', s=30)
            ax.scatter(range(len(norm_vars)), norm_vars, alpha=0.7, color=self.colors[1], 
                      label='Normalized', s=30)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Variance')
            ax.set_title('Sample Variances')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_sample_missingness(self, raw_data, normalized_data, ax):
        """Plot sample missingness comparison"""
        try:
            raw_missing = raw_data.isna().sum(axis=0) / raw_data.shape[0] * 100
            norm_missing = normalized_data.isna().sum(axis=0) / normalized_data.shape[0] * 100
            
            ax.bar(np.arange(len(raw_missing)) - 0.2, raw_missing, width=0.4, 
                   color=self.colors[0], alpha=0.7, label='Raw')
            ax.bar(np.arange(len(norm_missing)) + 0.2, norm_missing, width=0.4, 
                   color=self.colors[1], alpha=0.7, label='Normalized')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Missingness (%)')
            ax.set_title('Sample Missingness')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_sample_correlation(self, raw_data, normalized_data, ax):
        """Plot sample correlation comparison"""
        try:
            # Calculate correlation matrices (sample a subset for performance)
            n_samples = min(20, raw_data.shape[1])
            if n_samples > 5:
                sample_cols = np.random.choice(raw_data.columns, n_samples, replace=False)
                
                raw_sample = raw_data[sample_cols].corr()
                norm_sample = normalized_data[sample_cols].corr()
                
                # Plot correlation differences
                corr_diff = norm_sample.values - raw_sample.values
                im = ax.imshow(corr_diff, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax.set_title('Correlation Difference (Norm - Raw)')
                ax.set_xticks(range(len(sample_cols)))
                ax.set_yticks(range(len(sample_cols)))
                ax.set_xticklabels([f'S{i+1}' for i in range(len(sample_cols))], rotation=45)
                ax.set_yticklabels([f'S{i+1}' for i in range(len(sample_cols))])
                plt.colorbar(im, ax=ax)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def create_feature_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create feature-wise comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{qtl_type.upper()} Normalization: {method}\nFeature-wise Comparison', fontsize=16, fontweight='bold')
            
            # Plot 1: Feature means
            self._plot_feature_means(raw_data, normalized_data, axes[0, 0])
            
            # Plot 2: Feature variances
            self._plot_feature_variances(raw_data, normalized_data, axes[0, 1])
            
            # Plot 3: Mean-Variance relationship
            self._plot_mean_variance(raw_data, normalized_data, axes[1, 0])
            
            # Plot 4: Feature detection rate
            self._plot_feature_detection(raw_data, normalized_data, axes[1, 1])
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_feature_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create feature comparison: {e}")
            return None
    
    def _plot_feature_means(self, raw_data, normalized_data, ax):
        """Plot feature means comparison"""
        try:
            raw_means = raw_data.mean(axis=1)
            norm_means = normalized_data.mean(axis=1)
            
            ax.scatter(raw_means, norm_means, alpha=0.6, color=self.colors[2], s=20)
            ax.plot([raw_means.min(), raw_means.max()], [raw_means.min(), raw_means.max()], 
                   'r--', alpha=0.8)
            
            ax.set_xlabel('Raw Mean')
            ax.set_ylabel('Normalized Mean')
            ax.set_title('Feature Means: Raw vs Normalized')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_feature_variances(self, raw_data, normalized_data, ax):
        """Plot feature variances comparison"""
        try:
            raw_vars = raw_data.var(axis=1)
            norm_vars = normalized_data.var(axis=1)
            
            ax.scatter(raw_vars, norm_vars, alpha=0.6, color=self.colors[2], s=20)
            ax.plot([raw_vars.min(), raw_vars.max()], [raw_vars.min(), raw_vars.max()], 
                   'r--', alpha=0.8)
            
            ax.set_xlabel('Raw Variance')
            ax.set_ylabel('Normalized Variance')
            ax.set_title('Feature Variances: Raw vs Normalized')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_mean_variance(self, raw_data, normalized_data, ax):
        """Plot mean-variance relationship"""
        try:
            raw_means = raw_data.mean(axis=1)
            raw_vars = raw_data.var(axis=1)
            norm_means = normalized_data.mean(axis=1)
            norm_vars = normalized_data.var(axis=1)
            
            ax.scatter(raw_means, raw_vars, alpha=0.6, color=self.colors[0], 
                      s=20, label='Raw')
            ax.scatter(norm_means, norm_vars, alpha=0.6, color=self.colors[1], 
                      s=20, label='Normalized')
            
            ax.set_xlabel('Mean')
            ax.set_ylabel('Variance')
            ax.set_title('Mean-Variance Relationship')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_feature_detection(self, raw_data, normalized_data, ax):
        """Plot feature detection rates"""
        try:
            # Calculate detection rate (non-zero/non-missing)
            raw_detection = (raw_data > 0).sum(axis=1) / raw_data.shape[1] * 100
            norm_detection = (normalized_data > normalized_data.quantile(0.01)).sum(axis=1) / normalized_data.shape[1] * 100
            
            ax.hist(raw_detection, bins=30, alpha=0.7, color=self.colors[0], 
                   density=True, label='Raw')
            ax.hist(norm_detection, bins=30, alpha=0.7, color=self.colors[1], 
                   density=True, label='Normalized')
            
            ax.set_xlabel('Detection Rate (%)')
            ax.set_ylabel('Density')
            ax.set_title('Feature Detection Rates')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def create_statistical_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create statistical summary comparison"""
        try:
            # Calculate statistical summaries
            raw_stats = self._calculate_statistical_summary(raw_data, 'Raw')
            norm_stats = self._calculate_statistical_summary(normalized_data, 'Normalized')
            
            # Create comparison table
            stats_comparison = pd.concat([raw_stats, norm_stats], axis=1)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{qtl_type.upper()} Normalization: {method}\nStatistical Summary', fontsize=16, fontweight='bold')
            
            # Plot 1: Basic statistics comparison
            self._plot_basic_stats(stats_comparison, axes[0, 0])
            
            # Plot 2: Skewness and kurtosis
            self._plot_skewness_kurtosis(raw_data, normalized_data, axes[0, 1])
            
            # Plot 3: Outlier comparison
            self._plot_outlier_comparison(raw_data, normalized_data, axes[1, 0])
            
            # Plot 4: Statistical test results
            self._plot_statistical_tests(raw_data, normalized_data, axes[1, 1])
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_statistical_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save statistical summary
            stats_file = os.path.join(output_dir, f"{qtl_type}_statistical_summary.txt")
            stats_comparison.to_csv(stats_file, sep='\t')
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create statistical comparison: {e}")
            return None
    
    def _calculate_statistical_summary(self, data, label):
        """Calculate comprehensive statistical summary"""
        try:
            flat_data = data.values.flatten()
            flat_data = flat_data[~np.isnan(flat_data)]
            
            if len(flat_data) == 0:
                return pd.Series([0] * 10, index=[
                    'Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 
                    'Kurtosis', 'Q1', 'Q3', 'IQR'
                ], name=label)
            
            return pd.Series({
                'Mean': np.mean(flat_data),
                'Median': np.median(flat_data),
                'Std': np.std(flat_data),
                'Min': np.min(flat_data),
                'Max': np.max(flat_data),
                'Skewness': stats.skew(flat_data),
                'Kurtosis': stats.kurtosis(flat_data),
                'Q1': np.percentile(flat_data, 25),
                'Q3': np.percentile(flat_data, 75),
                'IQR': np.percentile(flat_data, 75) - np.percentile(flat_data, 25)
            }, name=label)
            
        except Exception as e:
            logger.warning(f"Error calculating statistical summary: {e}")
            return pd.Series()
    
    def _plot_basic_stats(self, stats_comparison, ax):
        """Plot basic statistics comparison"""
        try:
            metrics = ['Mean', 'Median', 'Std', 'Min', 'Max']
            raw_values = stats_comparison.loc[metrics, 'Raw']
            norm_values = stats_comparison.loc[metrics, 'Normalized']
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, raw_values, width, label='Raw', color=self.colors[0], alpha=0.7)
            ax.bar(x + width/2, norm_values, width, label='Normalized', color=self.colors[1], alpha=0.7)
            
            ax.set_xlabel('Statistical Measures')
            ax.set_ylabel('Value')
            ax.set_title('Basic Statistics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_skewness_kurtosis(self, raw_data, normalized_data, ax):
        """Plot skewness and kurtosis comparison"""
        try:
            # Calculate for each feature
            raw_skew = raw_data.apply(lambda x: stats.skew(x.dropna()), axis=1)
            norm_skew = normalized_data.apply(lambda x: stats.skew(x.dropna()), axis=1)
            
            raw_kurt = raw_data.apply(lambda x: stats.kurtosis(x.dropna()), axis=1)
            norm_kurt = normalized_data.apply(lambda x: stats.kurtosis(x.dropna()), axis=1)
            
            ax.scatter(raw_skew, raw_kurt, alpha=0.6, color=self.colors[0], 
                      s=20, label='Raw')
            ax.scatter(norm_skew, norm_kurt, alpha=0.6, color=self.colors[1], 
                      s=20, label='Normalized')
            
            ax.set_xlabel('Skewness')
            ax.set_ylabel('Kurtosis')
            ax.set_title('Skewness vs Kurtosis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_outlier_comparison(self, raw_data, normalized_data, ax):
        """Plot outlier comparison"""
        try:
            # Calculate outliers using IQR method
            def count_outliers(data):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return ((data < lower_bound) | (data > upper_bound)).sum()
            
            raw_outliers = raw_data.apply(count_outliers, axis=0)
            norm_outliers = normalized_data.apply(count_outliers, axis=0)
            
            ax.hist(raw_outliers, bins=30, alpha=0.7, color=self.colors[0], 
                   density=True, label='Raw')
            ax.hist(norm_outliers, bins=30, alpha=0.7, color=self.colors[1], 
                   density=True, label='Normalized')
            
            ax.set_xlabel('Number of Outliers per Sample')
            ax.set_ylabel('Density')
            ax.set_title('Outlier Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_statistical_tests(self, raw_data, normalized_data, ax):
        """Plot statistical test results"""
        try:
            # Perform normality tests on sample of features
            n_tests = min(100, raw_data.shape[0])
            if n_tests > 10:
                test_features = np.random.choice(raw_data.index, n_tests, replace=False)
                
                raw_normality = []
                norm_normality = []
                
                for feature in test_features:
                    if feature in raw_data.index:
                        raw_values = raw_data.loc[feature].dropna()
                        if len(raw_values) > 3:
                            _, raw_p = stats.normaltest(raw_values)
                            raw_normality.append(-np.log10(raw_p) if raw_p > 0 else 10)
                    
                    if feature in normalized_data.index:
                        norm_values = normalized_data.loc[feature].dropna()
                        if len(norm_values) > 3:
                            _, norm_p = stats.normaltest(norm_values)
                            norm_normality.append(-np.log10(norm_p) if norm_p > 0 else 10)
                
                if len(raw_normality) > 0 and len(norm_normality) > 0:
                    ax.boxplot([raw_normality, norm_normality], 
                              labels=['Raw', 'Normalized'])
                    ax.set_ylabel('-log10(p-value)')
                    ax.set_title('Normality Test (Higher = Less Normal)')
                    ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def create_pca_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create PCA comparison plot"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for PCA
            raw_clean = raw_data.dropna(axis=1, how='all').T
            norm_clean = normalized_data.dropna(axis=1, how='all').T
            
            if raw_clean.shape[0] > 5 and norm_clean.shape[0] > 5:
                # Perform PCA
                pca_raw = PCA(n_components=3)
                pca_norm = PCA(n_components=3)
                
                # Standardize before PCA
                scaler_raw = StandardScaler()
                scaler_norm = StandardScaler()
                
                raw_scaled = scaler_raw.fit_transform(raw_clean.fillna(raw_clean.mean()))
                norm_scaled = scaler_norm.fit_transform(norm_clean.fillna(norm_clean.mean()))
                
                pca_raw_result = pca_raw.fit_transform(raw_scaled)
                pca_norm_result = pca_norm.fit_transform(norm_scaled)
                
                # Create 3D scatter plot
                fig = plt.figure(figsize=(15, 6))
                
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                
                # Plot raw PCA
                scatter1 = ax1.scatter(pca_raw_result[:, 0], pca_raw_result[:, 1], pca_raw_result[:, 2],
                                     c=range(len(pca_raw_result)), cmap='viridis', s=50)
                ax1.set_title(f'Raw Data PCA\nExplained Variance: {pca_raw.explained_variance_ratio_[:3].sum():.2f}')
                ax1.set_xlabel(f'PC1 ({pca_raw.explained_variance_ratio_[0]:.2f})')
                ax1.set_ylabel(f'PC2 ({pca_raw.explained_variance_ratio_[1]:.2f})')
                ax1.set_zlabel(f'PC3 ({pca_raw.explained_variance_ratio_[2]:.2f})')
                
                # Plot normalized PCA
                scatter2 = ax2.scatter(pca_norm_result[:, 0], pca_norm_result[:, 1], pca_norm_result[:, 2],
                                     c=range(len(pca_norm_result)), cmap='viridis', s=50)
                ax2.set_title(f'Normalized Data PCA\nExplained Variance: {pca_norm.explained_variance_ratio_[:3].sum():.2f}')
                ax2.set_xlabel(f'PC1 ({pca_norm.explained_variance_ratio_[0]:.2f})')
                ax2.set_ylabel(f'PC2 ({pca_norm.explained_variance_ratio_[1]:.2f})')
                ax2.set_zlabel(f'PC3 ({pca_norm.explained_variance_ratio_[2]:.2f})')
                
                plt.tight_layout()
                plot_file = os.path.join(output_dir, f"{qtl_type}_pca_comparison.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create interactive 3D plot
                interactive_file = self._create_interactive_pca(pca_raw_result, pca_norm_result, 
                                                              pca_raw, pca_norm, qtl_type, method, output_dir)
                
                return {'static': plot_file, 'interactive': interactive_file}
            
        except Exception as e:
            logger.warning(f"Could not create PCA comparison: {e}")
            return None
    
    def _create_interactive_pca(self, pca_raw, pca_norm, pca_raw_obj, pca_norm_obj, qtl_type, method, output_dir):
        """Create interactive 3D PCA plot"""
        try:
            fig = make_subplots(rows=1, cols=2, 
                              specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                              subplot_titles=(
                                  f'Raw Data PCA (Variance: {pca_raw_obj.explained_variance_ratio_[:3].sum():.3f})',
                                  f'Normalized Data PCA (Variance: {pca_norm_obj.explained_variance_ratio_[:3].sum():.3f})'
                              ))
            
            # Raw PCA
            fig.add_trace(
                go.Scatter3d(
                    x=pca_raw[:, 0], y=pca_raw[:, 1], z=pca_raw[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(pca_raw)),
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f'Sample {i}' for i in range(len(pca_raw))],
                    name='Raw'
                ),
                row=1, col=1
            )
            
            # Normalized PCA
            fig.add_trace(
                go.Scatter3d(
                    x=pca_norm[:, 0], y=pca_norm[:, 1], z=pca_norm[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(pca_norm)),
                        colorscale='Viridis', 
                        opacity=0.8
                    ),
                    text=[f'Sample {i}' for i in range(len(pca_norm))],
                    name='Normalized'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f'{qtl_type.upper()} PCA Comparison - Normalization: {method}',
                height=600,
                showlegend=False
            )
            
            plot_file = os.path.join(output_dir, f"{qtl_type}_interactive_pca.html")
            fig.write_html(plot_file)
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create interactive PCA: {e}")
            return None
    
    def create_correlation_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create correlation structure comparison"""
        try:
            # Sample features for correlation analysis
            n_features = min(50, raw_data.shape[0])
            if n_features < 5:
                return None
                
            feature_sample = np.random.choice(raw_data.index, n_features, replace=False)
            
            raw_sample = raw_data.loc[feature_sample].T.corr()
            norm_sample = normalized_data.loc[feature_sample].T.corr()
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{qtl_type.upper()} Normalization: {method}\nCorrelation Structure', fontsize=16, fontweight='bold')
            
            # Plot 1: Raw correlation heatmap
            im1 = axes[0, 0].imshow(raw_sample, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[0, 0].set_title('Raw Data Correlation')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot 2: Normalized correlation heatmap
            im2 = axes[0, 1].imshow(norm_sample, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[0, 1].set_title('Normalized Data Correlation')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Plot 3: Correlation difference
            corr_diff = norm_sample - raw_sample
            im3 = axes[1, 0].imshow(corr_diff, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[1, 0].set_title('Correlation Difference (Norm - Raw)')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Plot 4: Correlation distribution
            axes[1, 1].hist(raw_sample.values.flatten(), bins=50, alpha=0.7, 
                           color=self.colors[0], density=True, label='Raw')
            axes[1, 1].hist(norm_sample.values.flatten(), bins=50, alpha=0.7, 
                           color=self.colors[1], density=True, label='Normalized')
            axes[1, 1].set_xlabel('Correlation Coefficient')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Correlation Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_correlation_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create interactive correlation plot
            interactive_file = self._create_interactive_correlation(raw_sample, norm_sample, qtl_type, method, output_dir)
            
            return {'static': plot_file, 'interactive': interactive_file}
            
        except Exception as e:
            logger.warning(f"Could not create correlation comparison: {e}")
            return None
    
    def _create_interactive_correlation(self, raw_corr, norm_corr, qtl_type, method, output_dir):
        """Create interactive correlation heatmap"""
        try:
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Raw Data Correlation', 'Normalized Data Correlation'))
            
            fig.add_trace(
                go.Heatmap(
                    z=raw_corr.values,
                    x=raw_corr.columns,
                    y=raw_corr.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1,
                    colorbar=dict(x=0.45, title="Correlation")
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=norm_corr.values,
                    x=norm_corr.columns,
                    y=norm_corr.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1,
                    colorbar=dict(x=1.0, title="Correlation")
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f'{qtl_type.upper()} Correlation Comparison - Normalization: {method}',
                height=600
            )
            
            plot_file = os.path.join(output_dir, f"{qtl_type}_interactive_correlation.html")
            fig.write_html(plot_file)
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create interactive correlation: {e}")
            return None
    
    def generate_comparison_html_report(self, comparison_results, qtl_type, output_dir):
        """Generate comprehensive HTML report for normalization comparison"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Normalization Comparison Report - {qtl_type.upper()}</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                    .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #2E86AB, #A23B72); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; text-align: center; }}
                    .header h1 {{ margin: 0; font-size: 2.5em; }}
                    .section {{ margin: 25px 0; padding: 20px; border: 1px solid #e9ecef; border-radius: 8px; }}
                    .section h2 {{ color: #2E86AB; border-bottom: 2px solid #F18F01; padding-bottom: 10px; }}
                    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .plot-item {{ text-align: center; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    .plot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                    .interactive-plot {{ height: 500px; border: 1px solid #ddd; border-radius: 5px; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                    .stat-card {{ background: linear-gradient(135deg, #2E86AB, #A23B72); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                    .stat-number {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
                    .summary-box {{ background: #e7f3ff; border-left: 4px solid #2E86AB; padding: 15px; margin: 15px 0; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔬 Normalization Comparison Report</h1>
                        <p><strong>{qtl_type.upper()} Analysis</strong></p>
                        <p>Method: {comparison_results.get('normalization_method', 'Unknown')} | Plots Generated: {len(comparison_results.get('plots_generated', []))}</p>
                    </div>
                    
                    <div class="section">
                        <h2>📊 Summary</h2>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-label">QTL Type</div>
                                <div class="stat-number">{qtl_type.upper()}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Normalization</div>
                                <div class="stat-number">{comparison_results.get('normalization_method', 'Unknown')}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Plots Generated</div>
                                <div class="stat-number">{len(comparison_results.get('plots_generated', []))}</div>
                            </div>
                        </div>
                        
                        <div class="summary-box">
                            <strong>🎯 Analysis Overview:</strong>
                            <p>This report compares the raw input data with the normalized data used for QTL analysis. 
                            The side-by-side visualizations help understand the impact of normalization on data structure, 
                            distribution, and quality.</p>
                        </div>
                    </div>
            """
            
            # Add plot sections
            if 'distribution_comparison' in comparison_results.get('plots_generated', []):
                html_content += """
                    <div class="section">
                        <h2>📈 Distribution Comparison</h2>
                        <div class="plot-grid">
                            <div class="plot-item">
                                <h3>Static Plots</h3>
                                <img src="distribution_comparison.png" alt="Distribution Comparison">
                                <p>Overall distribution, feature distributions, box plots, and QQ plots comparing raw vs normalized data.</p>
                            </div>
                """
                
                # Check if interactive plot exists
                interactive_dist_file = f"{qtl_type}_interactive_distribution.html"
                if os.path.exists(os.path.join(output_dir, interactive_dist_file)):
                    html_content += f"""
                            <div class="plot-item">
                                <h3>Interactive Distribution</h3>
                                <iframe src="{interactive_dist_file}" class="interactive-plot"></iframe>
                                <p>Interactive histogram showing the distribution of raw vs normalized values.</p>
                            </div>
                    """
                
                html_content += "</div></div>"
            
            if 'sample_comparison' in comparison_results.get('plots_generated', []):
                html_content += f"""
                    <div class="section">
                        <h2>👥 Sample-wise Comparison</h2>
                        <div class="plot-item">
                            <img src="sample_comparison.png" alt="Sample Comparison">
                            <p>Comparison of sample means, variances, missingness, and correlation structure.</p>
                        </div>
                    </div>
                """
            
            if 'feature_comparison' in comparison_results.get('plots_generated', []):
                html_content += f"""
                    <div class="section">
                        <h2>🔍 Feature-wise Comparison</h2>
                        <div class="plot-item">
                            <img src="feature_comparison.png" alt="Feature Comparison">
                            <p>Comparison of feature means, variances, mean-variance relationship, and detection rates.</p>
                        </div>
                    </div>
                """
            
            if 'statistical_comparison' in comparison_results.get('plots_generated', []):
                html_content += f"""
                    <div class="section">
                        <h2>📋 Statistical Summary</h2>
                        <div class="plot-item">
                            <img src="statistical_comparison.png" alt="Statistical Comparison">
                            <p>Statistical measures, normality tests, outlier analysis, and comprehensive summary statistics.</p>
                        </div>
                    </div>
                """
            
            if 'pca_comparison' in comparison_results.get('plots_generated', []):
                html_content += """
                    <div class="section">
                        <h2>🎯 PCA Comparison</h2>
                        <div class="plot-grid">
                            <div class="plot-item">
                                <h3>Static 3D PCA</h3>
                                <img src="pca_comparison.png" alt="PCA Comparison">
                                <p>3D PCA plots showing sample clustering in raw vs normalized space.</p>
                            </div>
                """
                
                # Check if interactive PCA exists
                interactive_pca_file = f"{qtl_type}_interactive_pca.html"
                if os.path.exists(os.path.join(output_dir, interactive_pca_file)):
                    html_content += f"""
                            <div class="plot-item">
                                <h3>Interactive 3D PCA</h3>
                                <iframe src="{interactive_pca_file}" class="interactive-plot"></iframe>
                                <p>Interactive 3D PCA visualization for exploring sample relationships.</p>
                            </div>
                    """
                
                html_content += "</div></div>"
            
            if 'correlation_comparison' in comparison_results.get('plots_generated', []):
                html_content += """
                    <div class="section">
                        <h2>🔄 Correlation Structure</h2>
                        <div class="plot-grid">
                            <div class="plot-item">
                                <img src="correlation_comparison.png" alt="Correlation Comparison">
                                <p>Correlation heatmaps and distribution comparisons.</p>
                            </div>
                """
                
                # Check if interactive correlation exists
                interactive_corr_file = f"{qtl_type}_interactive_correlation.html"
                if os.path.exists(os.path.join(output_dir, interactive_corr_file)):
                    html_content += f"""
                            <div class="plot-item">
                                <h3>Interactive Correlation</h3>
                                <iframe src="{interactive_corr_file}" class="interactive-plot"></iframe>
                                <p>Interactive correlation heatmaps for detailed exploration.</p>
                            </div>
                    """
                
                html_content += "</div></div>"
            
            html_content += """
                    <div class="section">
                        <h2>💡 Interpretation Guide</h2>
                        <div class="summary-box">
                            <strong>Key Points to Observe:</strong>
                            <ul>
                                <li><strong>Distribution Changes:</strong> Look for normalization effects on data spread and shape</li>
                                <li><strong>Outlier Handling:</strong> Check if normalization reduces extreme values appropriately</li>
                                <li><strong>Mean-Variance Relationship:</strong> Ideal normalization should decouple mean and variance</li>
                                <li><strong>Correlation Structure:</strong> Observe if biological signals are preserved while technical noise is reduced</li>
                                <li><strong>PCA Patterns:</strong> Check if batch effects are reduced while biological structure is maintained</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📁 Generated Files</h2>
                        <div class="summary-box">
                            <strong>Available in: {}</strong>
                            <ul>
            """.format(output_dir)
            
            # List generated files
            for file in os.listdir(output_dir):
                if file.endswith(('.png', '.html', '.txt')):
                    html_content += f'<li>{file}</li>'
            
            html_content += """
                            </ul>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            report_file = os.path.join(output_dir, f"{qtl_type}_normalization_report.html")
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"💾 Normalization comparison HTML report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Error generating HTML report: {e}")
            return None