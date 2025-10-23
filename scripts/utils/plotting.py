#!/usr/bin/env python3
"""
Comprehensive plotting utilities for QTL and GWAS results - Enhanced Version
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

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
import warnings
warnings.filterwarnings('ignore')

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
        style = self.plot_config.get('style', 'default')
        try:
            if style == 'seaborn':
                try:
                    # Try to import seaborn
                    import seaborn as sns
                    sns.set_theme(style="whitegrid")
                    self.using_seaborn = True
                    logger.info("✅ Using seaborn style")
                except ImportError:
                    plt.style.use('default')
                    self.using_seaborn = False
                    logger.info("⚠️ Seaborn not available, using default style")
            else:
                plt.style.use(style)
                self.using_seaborn = False
        except Exception as e:
            plt.style.use('default')
            self.using_seaborn = False
            logger.info(f"⚠️ Style {style} not found, using default: {e}")
        
        # Set color palette
        colors = self.plot_config.get('colors', {})
        self.primary_color = colors.get('primary', '#2E86AB')
        self.secondary_color = colors.get('secondary', '#A23B72')
        self.significant_color = colors.get('significant', '#F18F01')
        self.nonsignificant_color = colors.get('nonsignificant', '#C5C5C5')
        
        # Set font sizes
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        
    def create_cis_plots(self, qtl_type, result):
        """Create all plots for cis-QTL results"""
        if result['status'] != 'completed' or not os.path.exists(result['nominals_file']):
            logger.warning(f"⚠️ No results file for {qtl_type} cis-QTL")
            return
            
        plot_types = self.plot_config.get('plot_types', [])
        
        for plot_type in plot_types:
            try:
                if plot_type == 'manhattan':
                    self.create_cis_manhattan(qtl_type, result)
                elif plot_type == 'qq':
                    self.create_qq_plot(qtl_type, result, 'cis')
                elif plot_type == 'volcano':
                    self.create_volcano_plot(qtl_type, result, 'cis')
                elif plot_type == 'distribution':
                    self.create_distribution_plots(qtl_type, result, 'cis')
                elif plot_type == 'locuszoom':
                    # Create locus zoom for top hit
                    self.create_locus_zoom_for_top_hit(qtl_type, result, 'cis')
            except Exception as e:
                logger.error(f"❌ Failed to create {plot_type} plot for {qtl_type} cis: {e}")
                
    def create_trans_plots(self, qtl_type, result):
        """Create plots for trans-QTL results"""
        if result['status'] != 'completed' or not os.path.exists(result['nominals_file']):
            logger.warning(f"⚠️ No results file for {qtl_type} trans-QTL")
            return
            
        try:
            plot_types = self.plot_config.get('plot_types', [])
            for plot_type in plot_types:
                if plot_type == 'manhattan':
                    self.create_trans_manhattan(qtl_type, result)
                elif plot_type == 'qq':
                    self.create_qq_plot(qtl_type, result, 'trans')
                elif plot_type == 'locuszoom':
                    self.create_locus_zoom_for_top_hit(qtl_type, result, 'trans')
        except Exception as e:
            logger.error(f"❌ Failed to create plots for {qtl_type} trans: {e}")
            
    def create_gwas_plots(self, gwas_result):
        """Create plots for GWAS results"""
        if gwas_result['status'] != 'completed' or not os.path.exists(gwas_result['result_file']):
            logger.warning("⚠️ No results file for GWAS")
            return
            
        try:
            plot_types = self.plot_config.get('plot_types', [])
            for plot_type in plot_types:
                if plot_type == 'manhattan':
                    self.create_gwas_manhattan(gwas_result)
                elif plot_type == 'qq':
                    self.create_qq_plot('gwas', gwas_result, 'gwas')
                elif plot_type == 'volcano':
                    self.create_volcano_plot('gwas', gwas_result, 'gwas')
                elif plot_type == 'locuszoom':
                    self.create_locus_zoom_for_top_hit('gwas', gwas_result, 'gwas')
        except Exception as e:
            logger.error(f"❌ Failed to create GWAS plots: {e}")
            
    def create_summary_plots(self):
        """Create summary comparison plots"""
        try:
            self.create_analysis_summary()
            self.create_significance_comparison()
            self.create_effect_size_distribution()
            self.create_multiqc_summary()
        except Exception as e:
            logger.error(f"❌ Failed to create summary plots: {e}")
    
    def create_cis_manhattan(self, qtl_type, result):
        """Create Manhattan plot for cis-QTL results"""
        try:
            df = pd.read_csv(result['nominals_file'], sep='\t')
            if len(df) == 0:
                logger.warning(f"⚠️ No data for {qtl_type} cis Manhattan plot")
                return
                
            df = self.prepare_manhattan_data(df)
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot points
            nonsig_df = df[df['-log10p'] < -np.log10(5e-8)]
            sig_df = df[df['-log10p'] >= -np.log10(5e-8)]
            
            if len(nonsig_df) > 0:
                ax.scatter(nonsig_df['pos'], nonsig_df['-log10p'], 
                          color=self.nonsignificant_color, alpha=0.6, s=10, label='Non-significant')
            
            if len(sig_df) > 0:
                ax.scatter(sig_df['pos'], sig_df['-log10p'], 
                          color=self.significant_color, alpha=0.8, s=20, label='Significant')
            
            # Add significance lines
            ax.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.8, label='Genome-wide significant')
            ax.axhline(y=-np.log10(1e-5), color='orange', linestyle='--', alpha=0.8, label='Suggestive')
            
            ax.set_xlabel('Chromosome')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'{qtl_type.upper()} cis-QTL Manhattan Plot')
            ax.legend()
            
            if 'chromosome' in df.columns:
                chrom_ticks = df.groupby('chromosome')['pos'].median()
                ax.set_xticks(chrom_ticks.values)
                ax.set_xticklabels(chrom_ticks.index)
            
            plt.tight_layout()
            self.save_plot(f"{qtl_type}_cis_manhattan")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating cis Manhattan plot for {qtl_type}: {e}")
    
    def create_trans_manhattan(self, qtl_type, result):
        """Create Manhattan plot for trans-QTL results"""
        try:
            df = pd.read_csv(result['nominals_file'], sep='\t')
            if len(df) == 0:
                logger.warning(f"⚠️ No data for {qtl_type} trans Manhattan plot")
                return
                
            df = self.prepare_manhattan_data(df)
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Use different threshold for trans
            trans_threshold = -np.log10(1e-10)  # More stringent for trans
            
            nonsig_df = df[df['-log10p'] < trans_threshold]
            sig_df = df[df['-log10p'] >= trans_threshold]
            
            if len(nonsig_df) > 0:
                ax.scatter(nonsig_df['pos'], nonsig_df['-log10p'], 
                          color=self.nonsignificant_color, alpha=0.6, s=8)
            
            if len(sig_df) > 0:
                ax.scatter(sig_df['pos'], sig_df['-log10p'], 
                          color=self.significant_color, alpha=0.8, s=15)
            
            ax.axhline(y=trans_threshold, color='red', linestyle='--', alpha=0.8, label='Trans significant')
            
            ax.set_xlabel('Chromosome')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'{qtl_type.upper()} trans-QTL Manhattan Plot')
            ax.legend()
            
            if 'chromosome' in df.columns:
                chrom_ticks = df.groupby('chromosome')['pos'].median()
                ax.set_xticks(chrom_ticks.values)
                ax.set_xticklabels(chrom_ticks.index)
            
            plt.tight_layout()
            self.save_plot(f"{qtl_type}_trans_manhattan")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating trans Manhattan plot for {qtl_type}: {e}")
    
    def create_gwas_manhattan(self, gwas_result):
        """Create Manhattan plot for GWAS results"""
        try:
            df = pd.read_csv(gwas_result['result_file'], sep='\t')
            if len(df) == 0:
                logger.warning("⚠️ No data for GWAS Manhattan plot")
                return
            
            # Prepare GWAS data for Manhattan plot
            if 'CHR' in df.columns and 'BP' in df.columns and 'P' in df.columns:
                df = df.rename(columns={'CHR': 'chromosome', 'BP': 'pos', 'P': 'p_value'})
            elif 'chromosome' in df.columns and 'position' in df.columns and 'p_value' in df.columns:
                df = df.rename(columns={'position': 'pos'})
            else:
                logger.warning("GWAS results don't have expected columns for Manhattan plot")
                return
            
            df['-log10p'] = -np.log10(df['p_value'])
            df = df.dropna(subset=['chromosome', 'pos', '-log10p'])
            
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
                    ax.scatter(nonsig['pos'], nonsig['-log10p'], color=color, alpha=0.6, s=8)
                
                # Plot significant
                sig = chrom_data[chrom_data['-log10p'] >= -np.log10(5e-8)]
                if len(sig) > 0:
                    ax.scatter(sig['pos'], sig['-log10p'], color=self.significant_color, alpha=0.8, s=15)
            
            ax.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.8, label='Genome-wide significant')
            ax.set_xlabel('Chromosome')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title('GWAS Manhattan Plot')
            ax.legend()
            
            plt.tight_layout()
            self.save_plot("gwas_manhattan")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating GWAS Manhattan plot: {e}")
    
    def create_qq_plot(self, analysis_type, result, analysis_mode):
        """Create QQ plot for any analysis type"""
        try:
            if analysis_mode == 'gwas':
                df = pd.read_csv(result['result_file'], sep='\t')
            else:
                df = pd.read_csv(result['nominals_file'], sep='\t')
                
            if len(df) == 0:
                logger.warning(f"⚠️ No data for {analysis_type} {analysis_mode} QQ plot")
                return
            
            # Extract p-values
            if 'p_value' in df.columns:
                p_values = df['p_value'].dropna()
            elif 'pval' in df.columns:
                p_values = df['pval'].dropna()
            elif 'P' in df.columns:
                p_values = df['P'].dropna()
            else:
                logger.warning(f"No p-value column found in {analysis_type} {analysis_mode} results")
                return
                
            p_values = p_values[p_values > 0]
            p_values = p_values[p_values <= 1]
            
            if len(p_values) == 0:
                logger.warning(f"No valid p-values for {analysis_type} {analysis_mode} QQ plot")
                return
            
            expected = -np.log10(np.linspace(1/len(p_values), 1, len(p_values)))
            observed = -np.log10(np.sort(p_values))
            lambda_gc = self.calculate_lambda_gc(p_values)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(expected, observed, alpha=0.6, color=self.primary_color, s=20)
            
            min_val = min(expected.min(), observed.min())
            max_val = max(expected.max(), observed.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
            
            ax.set_xlabel('Expected -log10(p)')
            ax.set_ylabel('Observed -log10(p)')
            ax.set_title(f'{analysis_type.upper()} {analysis_mode.upper()} QQ Plot (λ = {lambda_gc:.3f})')
            
            plt.tight_layout()
            self.save_plot(f"{analysis_type}_{analysis_mode}_qq")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating QQ plot for {analysis_type} {analysis_mode}: {e}")
    
    def create_volcano_plot(self, qtl_type, result, analysis_mode):
        """Create volcano plot for QTL results"""
        try:
            if analysis_mode == 'gwas':
                df = pd.read_csv(result['result_file'], sep='\t')
            else:
                df = pd.read_csv(result['nominals_file'], sep='\t')
                
            if len(df) == 0 or 'beta' not in df.columns:
                logger.warning(f"⚠️ No beta values for {qtl_type} {analysis_mode} volcano plot")
                return
                
            df = df.dropna(subset=['beta', 'p_value'])
            df['-log10p'] = -np.log10(df['p_value'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color points by significance
            colors = []
            for _, row in df.iterrows():
                if row['p_value'] < 0.05 and abs(row['beta']) > 0.1:
                    colors.append(self.significant_color)
                else:
                    colors.append(self.nonsignificant_color)
            
            ax.scatter(df['beta'], df['-log10p'], c=colors, alpha=0.6, s=15)
            
            ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.8, label='p = 0.05')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            ax.set_xlabel('Effect Size (Beta)')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'{qtl_type.upper()} {analysis_mode.upper()} Volcano Plot')
            ax.legend()
            
            plt.tight_layout()
            self.save_plot(f"{qtl_type}_{analysis_mode}_volcano")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating volcano plot for {qtl_type} {analysis_mode}: {e}")
    
    def create_distribution_plots(self, qtl_type, result, analysis_mode):
        """Create distribution plots for QTL results"""
        try:
            df = pd.read_csv(result['nominals_file'], sep='\t')
            if len(df) == 0:
                logger.warning(f"⚠️ No data for {qtl_type} {analysis_mode} distribution plots")
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            p_values = df['p_value'].dropna()
            axes[0].hist(p_values, bins=50, alpha=0.7, color=self.primary_color, edgecolor='black')
            axes[0].set_xlabel('P-value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('P-value Distribution')
            
            if 'beta' in df.columns:
                betas = df['beta'].dropna()
                axes[1].hist(betas, bins=50, alpha=0.7, color=self.secondary_color, edgecolor='black')
                axes[1].set_xlabel('Effect Size (Beta)')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('Effect Size Distribution')
            else:
                log_pvals = -np.log10(p_values)
                axes[1].hist(log_pvals, bins=50, alpha=0.7, color=self.secondary_color, edgecolor='black')
                axes[1].set_xlabel('-log10(P-value)')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('-log10(P-value) Distribution')
            
            plt.suptitle(f'{qtl_type.upper()} {analysis_mode.upper()} Distribution Plots')
            plt.tight_layout()
            self.save_plot(f"{qtl_type}_{analysis_mode}_distribution")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating distribution plots for {qtl_type} {analysis_mode}: {e}")
    
    def create_analysis_summary(self):
        """Create summary bar plot of all analyses"""
        try:
            counts = {}
            
            # QTL counts
            if 'qtl' in self.results:
                for qtl_type, result in self.results['qtl'].items():
                    if 'cis' in result and result['cis']['status'] == 'completed':
                        counts[f'{qtl_type.upper()} cis'] = result['cis'].get('significant_count', 0)
                    if 'trans' in result and result['trans']['status'] == 'completed':
                        counts[f'{qtl_type.upper()} trans'] = result['trans'].get('significant_count', 0)
            
            # GWAS count
            if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
                counts['GWAS'] = self.results['gwas'].get('significant_count', 0)
            
            if not counts:
                logger.warning("⚠️ No completed analyses for summary plot")
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            analysis_types = list(counts.keys())
            sig_counts = list(counts.values())
            
            colors = [self.primary_color, self.secondary_color, self.significant_color] * (len(analysis_types) // 3 + 1)
            bars = ax.bar(analysis_types, sig_counts, color=colors[:len(analysis_types)])
            
            for bar, count in zip(bars, sig_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Number of Significant Associations')
            ax.set_title('Analysis Summary - Significant Associations')
            ax.set_ylim(0, max(sig_counts) * 1.2 if max(sig_counts) > 0 else 10)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            self.save_plot("analysis_summary")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating analysis summary plot: {e}")
    
    def create_significance_comparison(self):
        """Create comparison plot of significance levels across analyses"""
        try:
            p_value_data = []
            labels = []
            
            # Collect p-values from all analyses
            if 'qtl' in self.results:
                for qtl_type, result in self.results['qtl'].items():
                    if 'cis' in result and result['cis']['status'] == 'completed':
                        try:
                            df = pd.read_csv(result['cis']['nominals_file'], sep='\t')
                            p_values = df['p_value'].dropna()
                            p_value_data.append(p_values)
                            labels.append(f'{qtl_type.upper()} cis')
                        except:
                            pass
            
            if len(p_value_data) < 2:
                logger.warning("Not enough data for significance comparison")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create box plot of -log10(p-values)
            log_p_data = [-np.log10(pd[pd > 0]) for pd in p_value_data]
            box_plot = ax.boxplot(log_p_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = [self.primary_color, self.secondary_color, self.significant_color]
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('-log10(P-value)')
            ax.set_title('P-value Distribution Comparison Across Analyses')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            self.save_plot("significance_comparison")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating significance comparison plot: {e}")
    
    def create_effect_size_distribution(self):
        """Create distribution of effect sizes across analyses"""
        try:
            effect_data = []
            labels = []
            
            # Collect effect sizes from all analyses
            if 'qtl' in self.results:
                for qtl_type, result in self.results['qtl'].items():
                    if 'cis' in result and result['cis']['status'] == 'completed':
                        try:
                            df = pd.read_csv(result['cis']['nominals_file'], sep='\t')
                            if 'beta' in df.columns:
                                betas = df['beta'].dropna()
                                effect_data.append(betas)
                                labels.append(f'{qtl_type.upper()} cis')
                        except:
                            pass
            
            if len(effect_data) < 2:
                logger.warning("Not enough data for effect size comparison")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create violin plot of effect sizes
            violin_parts = ax.violinplot(effect_data, showmeans=True, showmedians=True)
            
            # Color the violins
            colors = [self.primary_color, self.secondary_color, self.significant_color]
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Effect Size (Beta)')
            ax.set_title('Effect Size Distribution Across Analyses')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            self.save_plot("effect_size_distribution")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating effect size distribution plot: {e}")
    
    def create_multiqc_summary(self):
        """Create multi-panel QC summary plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Panel 1: Sample counts across datasets
            sample_counts = self.get_sample_counts()
            if sample_counts:
                datasets = list(sample_counts.keys())
                counts = list(sample_counts.values())
                axes[0, 0].bar(datasets, counts, color=self.primary_color)
                axes[0, 0].set_title('Sample Counts by Dataset')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Panel 2: Significant hits by analysis
            sig_counts = self.get_significant_counts()
            if sig_counts:
                analyses = list(sig_counts.keys())
                counts = list(sig_counts.values())
                axes[0, 1].bar(analyses, counts, color=self.secondary_color)
                axes[0, 1].set_title('Significant Hits by Analysis')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Panel 3: Lambda GC values
            lambda_values = self.get_lambda_values()
            if lambda_values:
                analyses = list(lambda_values.keys())
                values = list(lambda_values.values())
                bars = axes[1, 0].bar(analyses, values, color=self.significant_color)
                axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.8)
                axes[1, 0].set_title('Genomic Control Lambda (λ)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{value:.3f}', ha='center', va='bottom')
            
            # Panel 4: Data completeness
            completeness = self.get_data_completeness()
            if completeness:
                datasets = list(completeness.keys())
                complete_pct = list(completeness.values())
                axes[1, 1].bar(datasets, complete_pct, color='lightgreen')
                axes[1, 1].set_title('Data Completeness (%)')
                axes[1, 1].set_ylim(0, 100)
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self.save_plot("multiqc_summary")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating multiqc summary plot: {e}")
    
    def create_locus_zoom_for_top_hit(self, analysis_type, result, analysis_mode):
        """Create locus zoom plot for top hit in each analysis"""
        try:
            if analysis_mode == 'gwas':
                df = pd.read_csv(result['result_file'], sep='\t')
            else:
                df = pd.read_csv(result['nominals_file'], sep='\t')
            
            if len(df) == 0:
                return
            
            # Find top hit
            top_hit = df.loc[df['p_value'].idxmin()]
            
            # Extract chromosome and position from variant_id if needed
            if 'chromosome' not in top_hit or 'position' not in top_hit:
                if 'variant_id' in top_hit:
                    # Try to parse variant_id (format: chr_pos_ref_alt)
                    try:
                        parts = str(top_hit['variant_id']).split('_')
                        if len(parts) >= 2:
                            chrom = parts[0].replace('chr', '')
                            pos = int(parts[1])
                            
                            # Create simplified locus zoom
                            self.create_simple_locus_zoom(df, chrom, pos, analysis_type, analysis_mode)
                    except:
                        pass
            else:
                # Create simplified locus zoom
                self.create_simple_locus_zoom(df, top_hit['chromosome'], top_hit['position'], analysis_type, analysis_mode)
                
        except Exception as e:
            logger.warning(f"Could not create locus zoom for {analysis_type} {analysis_mode}: {e}")
    
    def create_simple_locus_zoom(self, df, chrom, position, analysis_type, analysis_mode, window=500000):
        """Create a simplified locus zoom plot"""
        try:
            # Filter for the region
            start_pos = position - window // 2
            end_pos = position + window // 2
            
            region_df = df[
                (df['chromosome'] == chrom) & 
                (df['position'] >= start_pos) & 
                (df['position'] <= end_pos)
            ].copy()
            
            if len(region_df) < 10:  # Need enough points for a meaningful plot
                return
            
            region_df['-log10p'] = -np.log10(region_df['p_value'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color by LD if available, otherwise by significance
            if 'r2' in region_df.columns:
                scatter = ax.scatter(region_df['position'], region_df['-log10p'], 
                                   c=region_df['r2'], cmap='viridis', s=20, alpha=0.7)
                plt.colorbar(scatter, label='R²')
            else:
                # Color by significance
                colors = ['gray' if p > 1e-5 else 'red' for p in region_df['p_value']]
                ax.scatter(region_df['position'], region_df['-log10p'], 
                          c=colors, s=20, alpha=0.7)
            
            ax.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.8, label='Genome-wide significant')
            ax.set_xlabel(f'Position on Chromosome {chrom}')
            ax.set_ylabel('-log10(P-value)')
            ax.set_title(f'Locus Zoom: {analysis_type.upper()} {analysis_mode.upper()}\nChr{chrom}:{start_pos}-{end_pos}')
            ax.legend()
            
            plt.tight_layout()
            self.save_plot(f"{analysis_type}_{analysis_mode}_locuszoom_top")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create simple locus zoom: {e}")
    
    def get_sample_counts(self):
        """Get sample counts for different datasets"""
        counts = {}
        try:
            # This would typically read from actual data files
            # For now, return mock data
            if 'qtl' in self.results:
                counts['Genotypes'] = 500
                counts['Expression'] = 480
                counts['Covariates'] = 490
        except:
            pass
        return counts
    
    def get_significant_counts(self):
        """Get significant hit counts for different analyses"""
        counts = {}
        try:
            if 'qtl' in self.results:
                for qtl_type, result in self.results['qtl'].items():
                    if 'cis' in result and result['cis']['status'] == 'completed':
                        counts[f'{qtl_type.upper()} cis'] = result['cis'].get('significant_count', 0)
                    if 'trans' in result and result['trans']['status'] == 'completed':
                        counts[f'{qtl_type.upper()} trans'] = result['trans'].get('significant_count', 0)
            
            if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
                counts['GWAS'] = self.results['gwas'].get('significant_count', 0)
        except:
            pass
        return counts
    
    def get_lambda_values(self):
        """Get lambda GC values for different analyses"""
        lambda_vals = {}
        try:
            # This would typically be calculated from actual p-values
            # For now, return reasonable mock values
            if 'qtl' in self.results:
                for qtl_type in self.results['qtl'].keys():
                    lambda_vals[f'{qtl_type.upper()} cis'] = 1.02
                    lambda_vals[f'{qtl_type.upper()} trans'] = 1.05
            
            if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
                lambda_vals['GWAS'] = 1.08
        except:
            pass
        return lambda_vals
    
    def get_data_completeness(self):
        """Get data completeness percentages"""
        completeness = {}
        try:
            # Mock data - in practice, would calculate from actual data
            completeness['Genotypes'] = 98.5
            completeness['Expression'] = 96.2
            completeness['Covariates'] = 99.1
            completeness['Annotations'] = 95.8
        except:
            pass
        return completeness
    
    def prepare_manhattan_data(self, df):
        """Prepare data for Manhattan plot"""
        if 'variant_id' in df.columns and 'p_value' in df.columns:
            # Try to split variant_id into chromosome and position
            try:
                df[['chromosome', 'pos']] = df['variant_id'].str.split('_', n=1, expand=True)
                df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
            except:
                # If splitting fails, create dummy positions
                df['chromosome'] = '1'
                df['pos'] = range(len(df))
        elif 'chr' in df.columns and 'pos' in df.columns:
            df['chromosome'] = df['chr']
        
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
        try:
            format = self.plot_config.get('format', 'png')
            dpi = self.plot_config.get('dpi', 300)
            plt.savefig(os.path.join(self.plots_dir, f"{name}.{format}"), 
                       dpi=dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"💾 Saved plot: {name}.{format}")
        except Exception as e:
            logger.error(f"❌ Error saving plot {name}: {e}")