#!/usr/bin/env python3
"""
Advanced plotting utilities for QTL analysis with interactive plots
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

logger = logging.getLogger('QTLPipeline')

class AdvancedPlotter:
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        self.setup_plotly()
        
    def setup_plotly(self):
        """Setup Plotly configuration"""
        pio.templates.default = "plotly_white"
        
    def create_locus_zoom(self, result_file, gene_id, chrom, position, window_size=1000000, output_prefix=None):
        """Create locus zoom plot for specific genomic region"""
        logger.info(f"📊 Creating locus zoom plot for {gene_id} on {chrom}:{position}")
        
        try:
            # Read results file
            if result_file.endswith('.csv'):
                df = pd.read_csv(result_file)
            else:
                df = pd.read_csv(result_file, sep='\t')
            
            # Filter for the region of interest
            start_pos = position - window_size // 2
            end_pos = position + window_size // 2
            
            region_df = df[
                (df['chromosome'] == chrom) & 
                (df['position'] >= start_pos) & 
                (df['position'] <= end_pos)
            ].copy()
            
            if region_df.empty:
                logger.warning(f"No variants found in region {chrom}:{start_pos}-{end_pos}")
                return None
            
            # Create the plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add association signals
            region_df['-log10p'] = -np.log10(region_df['p_value'])
            
            # Color points by LD (if available) or by significance
            if 'r2' in region_df.columns:
                colors = region_df['r2']
                colorbar_title = "R²"
            else:
                colors = region_df['-log10p']
                colorbar_title = "-log10(p)"
            
            # Main association plot
            fig.add_trace(
                go.Scatter(
                    x=region_df['position'],
                    y=region_df['-log10p'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=colorbar_title)
                    ),
                    text=region_df.apply(lambda row: f"Variant: {row.get('variant_id', 'N/A')}<br>"
                                                   f"Position: {row['position']}<br>"
                                                   f"P-value: {row['p_value']:.2e}<br>"
                                                   f"-log10(p): {row['-log10p']:.2f}", axis=1),
                    hovertemplate="%{text}<extra></extra>",
                    name="Association"
                ),
                secondary_y=False
            )
            
            # Add gene annotations if available
            self.add_gene_annotations(fig, chrom, start_pos, end_pos)
            
            # Add significance line
            sig_threshold = -np.log10(5e-8)
            fig.add_hline(y=sig_threshold, line_dash="dash", line_color="red", 
                         annotation_text="Genome-wide significant", 
                         annotation_position="top left")
            
            # Update layout
            fig.update_layout(
                title=f"Locus Zoom Plot: {gene_id} (Chr{chrom}:{start_pos}-{end_pos})",
                xaxis_title=f"Position on Chromosome {chrom}",
                yaxis_title="-log10(p-value)",
                height=600,
                showlegend=False
            )
            
            # Save plot
            if output_prefix is None:
                output_prefix = f"locus_zoom_{gene_id}"
            
            output_file = os.path.join(self.plots_dir, f"{output_prefix}.html")
            fig.write_html(output_file)
            
            # Also save as static image
            static_file = os.path.join(self.plots_dir, f"{output_prefix}.png")
            fig.write_image(static_file, width=1200, height=600)
            
            logger.info(f"💾 Locus zoom plot saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating locus zoom plot: {e}")
            return None
    
    def add_gene_annotations(self, fig, chrom, start_pos, end_pos):
        """Add gene annotations to locus zoom plot"""
        try:
            # This would typically read from a gene annotation file
            # For now, we'll create some dummy annotations for demonstration
            genes = [
                {"name": "GeneA", "start": start_pos + 100000, "end": start_pos + 300000, "strand": "+"},
                {"name": "GeneB", "start": start_pos + 400000, "end": start_pos + 600000, "strand": "-"},
                {"name": "GeneC", "start": start_pos + 700000, "end": start_pos + 900000, "strand": "+"},
            ]
            
            for gene in genes:
                # Add gene as a rectangle
                fig.add_shape(
                    type="rect",
                    x0=gene["start"],
                    x1=gene["end"],
                    y0=0.95,
                    y1=1.0,
                    line=dict(color="blue"),
                    fillcolor="lightblue",
                    opacity=0.6
                )
                
                # Add gene name
                fig.add_annotation(
                    x=(gene["start"] + gene["end"]) / 2,
                    y=0.9,
                    text=gene["name"],
                    showarrow=False,
                    yshift=10
                )
                
        except Exception as e:
            logger.warning(f"Could not add gene annotations: {e}")
    
    def create_correlation_heatmap(self, data_matrix, output_prefix, title="Correlation Heatmap"):
        """Create correlation heatmap with clustering"""
        logger.info(f"📊 Creating correlation heatmap: {title}")
        
        try:
            # Calculate correlation matrix
            corr_matrix = data_matrix.corr()
            
            # Create interactive heatmap
            fig = px.imshow(
                corr_matrix,
                title=title,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                width=800,
                height=700,
                xaxis_title="Samples",
                yaxis_title="Samples"
            )
            
            # Save interactive version
            output_file = os.path.join(self.plots_dir, f"{output_prefix}_correlation_heatmap.html")
            fig.write_html(output_file)
            
            # Create static version with clustering
            plt.figure(figsize=(10, 8))
            sns.clustermap(corr_matrix, cmap="RdBu_r", center=0, annot=False)
            plt.title(title)
            static_file = os.path.join(self.plots_dir, f"{output_prefix}_correlation_heatmap.png")
            plt.savefig(static_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"💾 Correlation heatmap saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating correlation heatmap: {e}")
            return None
    
    def create_network_plot(self, significant_associations, output_prefix, max_nodes=50):
        """Create network plot of significant QTL associations"""
        logger.info("📊 Creating network plot...")
        
        try:
            if len(significant_associations) == 0:
                logger.warning("No significant associations for network plot")
                return None
            
            # Limit number of nodes for clarity
            if len(significant_associations) > max_nodes:
                significant_associations = significant_associations.head(max_nodes)
            
            # Create nodes and edges
            nodes = set()
            edges = []
            
            for _, row in significant_associations.iterrows():
                gene_node = row.get('gene_id', 'Unknown')
                variant_node = row.get('variant_id', 'Unknown')
                
                nodes.add(gene_node)
                nodes.add(variant_node)
                
                edges.append({
                    'source': variant_node,
                    'target': gene_node,
                    'value': -np.log10(row['p_value']),
                    'beta': row.get('beta', 0)
                })
            
            # Create network plot
            fig = go.Figure()
            
            # Add edges
            for edge in edges:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(width=2, color='gray'),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            # Add nodes (simplified implementation)
            node_x = []
            node_y = []
            node_text = []
            
            for i, node in enumerate(nodes):
                node_x.append(np.cos(2 * np.pi * i / len(nodes)))
                node_y.append(np.sin(2 * np.pi * i / len(nodes)))
                node_text.append(node)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=20, color='lightblue'),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                name="Nodes"
            ))
            
            fig.update_layout(
                title="QTL Network Plot",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Network plot of significant gene-variant associations",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            output_file = os.path.join(self.plots_dir, f"{output_prefix}_network.html")
            fig.write_html(output_file)
            
            logger.info(f"💾 Network plot saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating network plot: {e}")
            return None
    
    def create_interactive_manhattan(self, result_file, output_prefix, title="Manhattan Plot"):
        """Create interactive Manhattan plot with Plotly"""
        logger.info(f"📊 Creating interactive Manhattan plot: {title}")
        
        try:
            # Read results
            if result_file.endswith('.csv'):
                df = pd.read_csv(result_file)
            else:
                df = pd.read_csv(result_file, sep='\t')
            
            if df.empty:
                logger.warning("No data for Manhattan plot")
                return None
            
            # Prepare data
            df['-log10p'] = -np.log10(df['p_value'])
            
            # Create chromosome-wise positions
            df_sorted = df.sort_values(['chromosome', 'position'])
            df_sorted['pos_index'] = range(len(df_sorted))
            
            # Create the plot
            fig = px.scatter(
                df_sorted,
                x='pos_index',
                y='-log10p',
                color='chromosome',
                color_continuous_scale='Viridis',
                hover_data={
                    'chromosome': True,
                    'position': True,
                    'p_value': ':.2e',
                    'pos_index': False
                },
                title=title
            )
            
            # Add significance line
            sig_threshold = -np.log10(5e-8)
            fig.add_hline(y=sig_threshold, line_dash="dash", line_color="red", 
                         annotation_text="Genome-wide significant")
            
            # Update layout
            fig.update_layout(
                xaxis_title="Chromosome",
                yaxis_title="-log10(p-value)",
                height=600,
                showlegend=False
            )
            
            # Set x-axis ticks to chromosome midpoints
            chrom_ticks = {}
            for chrom in df_sorted['chromosome'].unique():
                chrom_data = df_sorted[df_sorted['chromosome'] == chrom]
                if len(chrom_data) > 0:
                    chrom_ticks[chrom] = chrom_data['pos_index'].median()
            
            fig.update_xaxes(
                tickvals=list(chrom_ticks.values()),
                ticktext=[f"Chr{int(ch)}" if isinstance(ch, (int, float)) else f"Chr{ch}" 
                         for ch in chrom_ticks.keys()]
            )
            
            # Save plot
            output_file = os.path.join(self.plots_dir, f"{output_prefix}_interactive_manhattan.html")
            fig.write_html(output_file)
            
            logger.info(f"💾 Interactive Manhattan plot saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating interactive Manhattan plot: {e}")
            return None
    
    def create_multi_panel_figure(self, qtl_results, output_prefix):
        """Create multi-panel figure with summary of all analyses"""
        logger.info("📊 Creating multi-panel summary figure...")
        
        try:
            # Create subplots: 2 rows, 3 columns
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    "QTL Summary", "GWAS Results", "Sample QC",
                    "MAF Distribution", "P-value Distribution", "Top Associations"
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
                ]
            )
            
            # Panel 1: QTL Summary (bar plot)
            if 'qtl' in qtl_results:
                analysis_types = []
                significant_counts = []
                
                for qtl_type, result in qtl_results['qtl'].items():
                    if 'cis' in result and result['cis'].get('significant_count', 0) > 0:
                        analysis_types.append(f"{qtl_type.upper()} cis")
                        significant_counts.append(result['cis']['significant_count'])
                    if 'trans' in result and result['trans'].get('significant_count', 0) > 0:
                        analysis_types.append(f"{qtl_type.upper()} trans")
                        significant_counts.append(result['trans']['significant_count'])
                
                if analysis_types:
                    fig.add_trace(
                        go.Bar(x=analysis_types, y=significant_counts, name="Significant QTLs"),
                        row=1, col=1
                    )
            
            # Panel 2: GWAS Results (if available)
            if 'gwas' in qtl_results and qtl_results['gwas'].get('significant_count', 0) > 0:
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=qtl_results['gwas']['significant_count'],
                        title={"text": "GWAS Hits"},
                        number={'suffix': " significant"},
                        domain={'row': 0, 'column': 1}
                    ),
                    row=1, col=2
                )
            
            # Panel 3: Sample QC (placeholder)
            fig.add_trace(
                go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='lines', name='QC Metric'),
                row=1, col=3
            )
            
            # Panel 4: MAF Distribution (placeholder)
            fig.add_trace(
                go.Histogram(x=np.random.normal(0.1, 0.05, 1000), nbinsx=30, name='MAF'),
                row=2, col=1
            )
            
            # Panel 5: P-value Distribution (placeholder)
            fig.add_trace(
                go.Histogram(x=np.random.uniform(0, 1, 1000), nbinsx=30, name='P-values'),
                row=2, col=2
            )
            
            # Panel 6: Top Associations (placeholder)
            fig.add_trace(
                go.Scatter(x=[1, 2, 3, 4, 5], y=[10, 11, 12, 13, 14], mode='markers', name='Top Hits'),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text="Comprehensive Analysis Summary",
                showlegend=False
            )
            
            # Save plot
            output_file = os.path.join(self.plots_dir, f"{output_prefix}_multi_panel_summary.html")
            fig.write_html(output_file)
            
            # Static version
            static_file = os.path.join(self.plots_dir, f"{output_prefix}_multi_panel_summary.png")
            fig.write_image(static_file, width=1200, height=800)
            
            logger.info(f"💾 Multi-panel summary saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating multi-panel figure: {e}")
            return None
    
    def create_expression_heatmap(self, expression_data, output_prefix, top_genes=50):
        """Create expression heatmap for top variable genes"""
        logger.info(f"📊 Creating expression heatmap for top {top_genes} genes...")
        
        try:
            # Calculate variance and select top genes
            variances = expression_data.var(axis=1)
            top_genes_idx = variances.nlargest(top_genes).index
            top_expression = expression_data.loc[top_genes_idx]
            
            # Z-score normalization
            z_scores = (top_expression - top_expression.mean(axis=1).values.reshape(-1, 1)) / top_expression.std(axis=1).values.reshape(-1, 1)
            
            # Create heatmap
            fig = px.imshow(
                z_scores,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"Expression Heatmap (Top {top_genes} Variable Genes)"
            )
            
            fig.update_layout(
                width=1000,
                height=800,
                xaxis_title="Samples",
                yaxis_title="Genes"
            )
            
            output_file = os.path.join(self.plots_dir, f"{output_prefix}_expression_heatmap.html")
            fig.write_html(output_file)
            
            logger.info(f"💾 Expression heatmap saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating expression heatmap: {e}")
            return None
    
    def create_qq_plot_interactive(self, p_values, output_prefix, title="QQ Plot"):
        """Create interactive QQ plot"""
        logger.info(f"📊 Creating interactive QQ plot: {title}")
        
        try:
            p_values = np.array(p_values)
            p_values = p_values[~np.isnan(p_values)]
            p_values = p_values[p_values > 0]
            p_values = p_values[p_values <= 1]
            
            if len(p_values) == 0:
                logger.warning("No valid p-values for QQ plot")
                return None
            
            # Calculate expected and observed
            n = len(p_values)
            expected = -np.log10(np.linspace(1/n, 1, n))
            observed = -np.log10(np.sort(p_values))
            
            # Calculate lambda GC
            chi_squared = stats.chi2.ppf(1 - p_values, 1)
            lambda_gc = np.median(chi_squared) / 0.4549364
            
            # Create plot
            fig = go.Figure()
            
            # Add points
            fig.add_trace(go.Scatter(
                x=expected,
                y=observed,
                mode='markers',
                marker=dict(size=6, color='#2E86AB', opacity=0.6),
                name='Variants'
            ))
            
            # Add diagonal line
            max_val = max(expected.max(), observed.max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Expected'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{title} (λ = {lambda_gc:.3f})",
                xaxis_title="Expected -log10(p)",
                yaxis_title="Observed -log10(p)",
                showlegend=False,
                height=600
            )
            
            output_file = os.path.join(self.plots_dir, f"{output_prefix}_interactive_qq.html")
            fig.write_html(output_file)
            
            logger.info(f"💾 Interactive QQ plot saved: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating interactive QQ plot: {e}")
            return None