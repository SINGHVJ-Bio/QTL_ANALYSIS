#!/usr/bin/env python3
"""
Comprehensive plotting utilities for QTL and GWAS results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
from pathlib import Path
import logging

logger = logging.getLogger('QTLPipeline')

class QTLPlotter:
    def __init__(self, config, results, plots_dir):
        self.config = config
        self.results = results
        self.plots_dir = plots_dir
        self.plot_config = config.get('plotting', {})
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Setup matplotlib and seaborn style"""
        style = self.plot_config.get('style', 'seaborn')
        plt.style.use(style)
        
        # Set color palette
        colors = self.plot_config.get('colors', {})
        self.primary_color = colors.get('primary', '#2E86AB')
        self.secondary_color = colors.get('secondary', '#A23B72')
        self.significant_color = colors.get('significant', '#F18F01')
        self.nonsignificant_color = colors.get('nonsignificant', '#C5C5C5')
        
    def create_qtl_plots(self, qtl_type, result):
        """Create all plots for a QTL analysis"""
        if result['status'] != 'completed' or not os.path.exists(result['result_file']):
            return
            
        plot_types = self.plot_config.get('plot_types', [])
        
        for plot_type in plot_types:
            try:
                if plot_type == 'manhattan':
                    self.create_qtl_manhattan(qtl_type, result)
                elif plot_type == 'qq':
                    self.create_qtl_qq(qtl_type, result)
                elif plot_type == 'volcano':
                    self.create_qtl_volcano(qtl_type, result)
                elif plot_type == 'distribution':
                    self.create_qtl_distribution(qtl_type, result)
            except Exception as e:
                logger.error(f"Failed to create {plot_type} plot for {qtl_type}: {e}")
                
    def create_gwas_plots(self, gwas_result):
        """Create plots for GWAS results"""
        if gwas_result['status'] != 'completed' or not os.path.exists(gwas_result['result_file']):
            return
            
        try:
            self.create_gwas_manhattan(gwas_result)
            self.create_gwas_qq(gwas_result)
        except Exception as e:
            logger.error(f"Failed to create GWAS plots: {e}")
            
    def create_summary_plots(self):
        """Create summary comparison plots"""
        try:
            self.create_analysis_summary()
            self.create_significance_comparison()
        except Exception as e:
            logger.error(f"Failed to create summary plots: {e}")
    
    def create_qtl_manhattan(self, qtl_type, result):
        """Create Manhattan plot for QTL results"""
        df = pd.read_csv(result['result_file'], sep='\t')
        if len(df) == 0:
            return
            
        df = self.prepare_manhattan_data(df)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot points
        nonsig_df = df[df['-log10p'] < -np.log10(5e-8)]
        sig_df = df[df['-log10p'] >= -np.log10(5e-8)]
        
        if len(nonsig_df) > 0:
            ax.scatter(nonsig_df['pos'], nonsig_df['-log10p'], 
                      color=self.nonsignificant_color, alpha=0.6, s=10)
        
        if len(sig_df) > 0:
            ax.scatter(sig_df['pos'], sig_df['-log10p'], 
                      color=self.significant_color, alpha=0.8, s=20)
        
        # Add significance lines
        ax.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.8, label='Genome-wide significant')
        ax.axhline(y=-np.log10(1e-5), color='orange', linestyle='--', alpha=0.8, label='Suggestive')
        
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title(f'{qtl_type.upper()} Manhattan Plot')
        ax.legend()
        
        if 'chromosome' in df.columns:
            chrom_ticks = df.groupby('chromosome')['pos'].median()
            ax.set_xticks(chrom_ticks.values)
            ax.set_xticklabels(chrom_ticks.index)
        
        plt.tight_layout()
        self.save_plot(f"{qtl_type}_manhattan")
        plt.close()
    
    def create_qtl_qq(self, qtl_type, result):
        """Create QQ plot for QTL results"""
        df = pd.read_csv(result['result_file'], sep='\t')
        if len(df) == 0:
            return
            
        p_values = df['p_value'].dropna()
        p_values = p_values[p_values > 0]
        
        if len(p_values) == 0:
            return
        
        expected = -np.log10(np.linspace(1/len(p_values), 1, len(p_values)))
        observed = -np.log10(np.sort(p_values))
        lambda_gc = self.calculate_lambda_gc(p_values)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(expected, observed, alpha=0.6, color=self.primary_color)
        
        min_val = min(expected.min(), observed.min())
        max_val = max(expected.max(), observed.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel('Expected -log10(p)')
        ax.set_ylabel('Observed -log10(p)')
        ax.set_title(f'{qtl_type.upper()} QQ Plot (λ = {lambda_gc:.3f})')
        
        plt.tight_layout()
        self.save_plot(f"{qtl_type}_qq")
        plt.close()
    
    def create_qtl_volcano(self, qtl_type, result):
        """Create volcano plot for QTL results"""
        df = pd.read_csv(result['result_file'], sep='\t')
        if len(df) == 0 or 'beta' not in df.columns:
            return
            
        df = df.dropna(subset=['beta', 'p_value'])
        df['-log10p'] = -np.log10(df['p_value'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = []
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and abs(row['beta']) > 0.1:
                colors.append(self.significant_color)
            else:
                colors.append(self.nonsignificant_color)
        
        ax.scatter(df['beta'], df['-log10p'], c=colors, alpha=0.6, s=20)
        
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.8)
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.8)
        ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Effect Size (Beta)')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title(f'{qtl_type.upper()} Volcano Plot')
        
        plt.tight_layout()
        self.save_plot(f"{qtl_type}_volcano")
        plt.close()
    
    def create_qtl_distribution(self, qtl_type, result):
        """Create distribution plots for QTL results"""
        df = pd.read_csv(result['result_file'], sep='\t')
        if len(df) == 0:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        p_values = df['p_value'].dropna()
        axes[0].hist(p_values, bins=50, alpha=0.7, color=self.primary_color)
        axes[0].set_xlabel('P-value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('P-value Distribution')
        
        if 'beta' in df.columns:
            betas = df['beta'].dropna()
            axes[1].hist(betas, bins=50, alpha=0.7, color=self.secondary_color)
            axes[1].set_xlabel('Effect Size (Beta)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Effect Size Distribution')
        else:
            log_pvals = -np.log10(p_values)
            axes[1].hist(log_pvals, bins=50, alpha=0.7, color=self.secondary_color)
            axes[1].set_xlabel('-log10(P-value)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('-log10(P-value) Distribution')
        
        plt.suptitle(f'{qtl_type.upper()} Distribution Plots')
        plt.tight_layout()
        self.save_plot(f"{qtl_type}_distribution")
        plt.close()
    
    def create_gwas_manhattan(self, gwas_result):
        """Create Manhattan plot for GWAS results"""
        df = pd.read_csv(gwas_result['result_file'], sep='\t')
        if len(df) == 0:
            return
            
        # Prepare GWAS data for Manhattan plot
        if 'CHR' in df.columns and 'BP' in df.columns and 'P' in df.columns:
            df = df.rename(columns={'CHR': 'chromosome', 'BP': 'pos', 'P': 'p_value'})
            df['-log10p'] = -np.log10(df['p_value'])
        else:
            logger.warning("GWAS results don't have expected columns for Manhattan plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Color points by chromosome
        chromosomes = sorted(df['chromosome'].unique())
        colors = [self.primary_color, self.secondary_color] * (len(chromosomes) // 2 + 1)
        
        for i, chrom in enumerate(chromosomes):
            chrom_data = df[df['chromosome'] == chrom]
            color = colors[i % len(colors)]
            
            # Plot non-significant
            nonsig = chrom_data[chrom_data['-log10p'] < -np.log10(5e-8)]
            if len(nonsig) > 0:
                ax.scatter(nonsig['pos'], nonsig['-log10p'], color=color, alpha=0.6, s=10)
            
            # Plot significant
            sig = chrom_data[chrom_data['-log10p'] >= -np.log10(5e-8)]
            if len(sig) > 0:
                ax.scatter(sig['pos'], sig['-log10p'], color=self.significant_color, alpha=0.8, s=20)
        
        ax.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.8, label='Genome-wide significant')
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('GWAS Manhattan Plot')
        ax.legend()
        
        plt.tight_layout()
        self.save_plot("gwas_manhattan")
        plt.close()
    
    def create_gwas_qq(self, gwas_result):
        """Create QQ plot for GWAS results"""
        df = pd.read_csv(gwas_result['result_file'], sep='\t')
        if len(df) == 0:
            return
            
        if 'P' in df.columns:
            p_values = df['P'].dropna()
        elif 'p_value' in df.columns:
            p_values = df['p_value'].dropna()
        else:
            logger.warning("No p-value column found in GWAS results")
            return
            
        p_values = p_values[p_values > 0]
        
        if len(p_values) == 0:
            return
        
        expected = -np.log10(np.linspace(1/len(p_values), 1, len(p_values)))
        observed = -np.log10(np.sort(p_values))
        lambda_gc = self.calculate_lambda_gc(p_values)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(expected, observed, alpha=0.6, color=self.primary_color)
        
        min_val = min(expected.min(), observed.min())
        max_val = max(expected.max(), observed.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel('Expected -log10(p)')
        ax.set_ylabel('Observed -log10(p)')
        ax.set_title(f'GWAS QQ Plot (λ = {lambda_gc:.3f})')
        
        plt.tight_layout()
        self.save_plot("gwas_qq")
        plt.close()
    
    def create_analysis_summary(self):
        """Create summary bar plot of all analyses"""
        counts = {}
        
        # QTL counts
        if 'qtl' in self.results:
            for qtl_type, result in self.results['qtl'].items():
                if result['status'] == 'completed':
                    counts[qtl_type.upper()] = result.get('significant_count', 0)
        
        # GWAS count
        if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
            counts['GWAS'] = self.results['gwas'].get('significant_count', 0)
        
        if not counts:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        analysis_types = list(counts.keys())
        sig_counts = list(counts.values())
        
        bars = ax.bar(analysis_types, sig_counts, 
                     color=[self.primary_color, self.secondary_color, self.significant_color][:len(analysis_types)])
        
        for bar, count in zip(bars, sig_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        ax.set_ylabel('Number of Significant Associations')
        ax.set_title('Analysis Summary - Significant Associations')
        ax.set_ylim(0, max(sig_counts) * 1.2 if max(sig_counts) > 0 else 10)
        
        plt.tight_layout()
        self.save_plot("analysis_summary")
        plt.close()
    
    def create_significance_comparison(self):
        """Create comparison plot of significance levels"""
        # This would compare p-value distributions across analyses
        # Implementation depends on specific comparison needs
        pass
    
    def prepare_manhattan_data(self, df):
        """Prepare data for Manhattan plot"""
        if 'variant_id' in df.columns:
            df[['chromosome', 'pos']] = df['variant_id'].str.split('_', n=1, expand=True)
            df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
        df['-log10p'] = -np.log10(df['p_value'])
        return df.dropna(subset=['chromosome', 'pos', '-log10p'])
    
    def calculate_lambda_gc(self, p_values):
        """Calculate genomic control lambda"""
        try:
            chi_squared = stats.chi2.ppf(1 - p_values, 1)
            lambda_gc = np.median(chi_squared) / 0.4549364
            return lambda_gc
        except:
            return 1.0
    
    def save_plot(self, name):
        """Save plot with configured format and DPI"""
        format = self.plot_config.get('format', 'png')
        dpi = self.plot_config.get('dpi', 300)
        plt.savefig(os.path.join(self.plots_dir, f"{name}.{format}"), 
                   dpi=dpi, bbox_inches='tight')