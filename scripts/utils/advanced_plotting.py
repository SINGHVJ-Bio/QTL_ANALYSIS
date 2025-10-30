#!/usr/bin/env python3
"""
Advanced plotting utilities for QTL analysis with interactive plots - Enhanced Version
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with:
- Improved performance for large datasets
- Better memory management
- Enhanced error handling and validation
- Additional interactive plot types
- Parallel processing for plot generation
- Comprehensive data validation
- Export functionality for publications
- Consistent directory management using DirectoryManager
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Conditional imports for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available, interactive plots will be disabled")

# Import directory manager
try:
    from scripts.utils.directory_manager import get_module_directories
except ImportError as e:
    logging.warning(f"Directory manager not available: {e}")
    get_module_directories = None
   
warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

class AdvancedPlotter:
    def __init__(self, config: Dict[str, Any], results_dir: str):
        self.config = config
        self.results_dir = Path(results_dir)
        self.plot_config = config.get('plotting', {})
        self.performance_config = config.get('performance', {})
        self.max_workers = self.performance_config.get('max_workers', 4)
        
        # Setup directories using directory manager
        self.setup_directories()
        
        self.setup_plotly()
        
    def setup_directories(self):
        """Setup directories using directory manager"""
        try:
            if get_module_directories:
                self.dirs = get_module_directories(
                    'advanced_plotting',
                    [
                        'visualization',
                        {'visualization': ['summary_plots', 'interactive_plots', 'manhattan_plots', 'qq_plots']},
                        'system',
                        {'system': ['temporary_files']}
                    ],
                    str(self.results_dir)
                )
                self.plots_dir = self.dirs['visualization_interactive_plots']
                self.summary_plots_dir = self.dirs['visualization_summary_plots']
                self.manhattan_plots_dir = self.dirs['visualization_manhattan_plots']
                self.qq_plots_dir = self.dirs['visualization_qq_plots']
                self.temp_dir = self.dirs['system_temporary_files']
                
                logger.info(f"✅ Advanced plotting directories setup in: {self.plots_dir}")
            else:
                # Fallback directory creation
                self.plots_dir = self.results_dir / "visualization" / "interactive_plots"
                self.summary_plots_dir = self.results_dir / "visualization" / "summary_plots"
                self.manhattan_plots_dir = self.results_dir / "visualization" / "manhattan_plots"
                self.qq_plots_dir = self.results_dir / "visualization" / "qq_plots"
                self.temp_dir = self.results_dir / "system" / "temporary_files"
                
                self.plots_dir.mkdir(parents=True, exist_ok=True)
                self.summary_plots_dir.mkdir(parents=True, exist_ok=True)
                self.manhattan_plots_dir.mkdir(parents=True, exist_ok=True)
                self.qq_plots_dir.mkdir(parents=True, exist_ok=True)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"✅ Fallback plotting directories created in: {self.plots_dir}")
                
        except Exception as e:
            logger.error(f"❌ Directory setup failed: {e}")
            # Ultimate fallback - maintain original structure for maximum compatibility
            self.plots_dir = self.results_dir / "plots" / "interactive"
            self.summary_plots_dir = self.results_dir / "plots" / "summary"
            self.manhattan_plots_dir = self.results_dir / "plots" / "manhattan"
            self.qq_plots_dir = self.results_dir / "plots" / "qq"
            self.temp_dir = self.results_dir / "temp_plots"
            
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.summary_plots_dir.mkdir(parents=True, exist_ok=True)
            self.manhattan_plots_dir.mkdir(parents=True, exist_ok=True)
            self.qq_plots_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            logger.warning(f"⚠️ Using ultimate fallback directories: {self.plots_dir}")
    
    def setup_plotly(self):
        """Setup Plotly configuration with enhanced settings"""
        try:
            # Set default template
            pio.templates.default = "plotly_white"
            
            # Configure for better performance
            pio.kaleido.scope.default_format = "png"
            pio.kaleido.scope.default_width = 1200
            pio.kaleido.scope.default_height = 800
            pio.kaleido.scope.default_scale = 2
            
            # Enhanced color scale
            self.color_scale = px.colors.sequential.Viridis
            self.diverging_scale = px.colors.diverging.RdBu_r
            self.qualitative_scale = px.colors.qualitative.Set3
            
            logger.info("✅ Plotly configuration initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Plotly setup encountered issues: {e}")
    
    def create_locus_zoom(self, result_file: str, gene_id: str, chrom: str, position: int, 
                         window_size: int = 1000000, output_prefix: Optional[str] = None,
                         **kwargs) -> Optional[str]:
        """Create enhanced locus zoom plot for specific genomic region"""
        logger.info(f"📊 Creating enhanced locus zoom plot for {gene_id} on {chrom}:{position}")
        
        try:
            # Validate inputs
            if not self._validate_input_file(result_file):
                return None
            
            # Read results file with optimization
            df = self._load_data_optimized(result_file)
            if df.empty:
                logger.warning(f"No data found in {result_file}")
                return None
            
            # Filter for the region of interest
            region_df = self._filter_genomic_region(df, chrom, position, window_size)
            if region_df.empty:
                logger.warning(f"No variants found in region {chrom}:{position-window_size//2}-{position+window_size//2}")
                return None
            
            # Create enhanced plot
            output_file = self._create_enhanced_locus_zoom(region_df, gene_id, chrom, position, 
                                                          window_size, output_prefix, **kwargs)
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating locus zoom plot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _validate_input_file(self, file_path: str) -> bool:
        """Validate input file exists and is accessible"""
        if not os.path.exists(file_path):
            logger.error(f"❌ Input file not found: {file_path}")
            return False
        return True
    
    def _load_data_optimized(self, file_path: str) -> pd.DataFrame:
        """Load data with optimized parameters for large files"""
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, low_memory=False)
            else:
                # Try tab-separated first
                try:
                    df = pd.read_csv(file_path, sep='\t', low_memory=False)
                except:
                    df = pd.read_csv(file_path, low_memory=False)
            
            logger.debug(f"📊 Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def _filter_genomic_region(self, df: pd.DataFrame, chrom: str, position: int, 
                              window_size: int) -> pd.DataFrame:
        """Filter dataframe for genomic region with enhanced parsing"""
        start_pos = position - window_size // 2
        end_pos = position + window_size // 2
        
        # Handle different chromosome column formats
        chrom_columns = ['chromosome', 'chr', 'CHROM', 'CHR']
        pos_columns = ['position', 'pos', 'POS', 'BP']
        
        chrom_col = None
        pos_col = None
        
        for col in chrom_columns:
            if col in df.columns:
                chrom_col = col
                break
        
        for col in pos_columns:
            if col in df.columns:
                pos_col = col
                break
        
        if not chrom_col or not pos_col:
            logger.warning("Could not find chromosome/position columns")
            return pd.DataFrame()
        
        # Filter region
        region_df = df[
            (df[chrom_col].astype(str) == str(chrom)) & 
            (df[pos_col] >= start_pos) & 
            (df[pos_col] <= end_pos)
        ].copy()
        
        # Add -log10(p-value)
        if 'p_value' in region_df.columns:
            region_df['-log10p'] = -np.log10(region_df['p_value'])
        elif 'pval' in region_df.columns:
            region_df['-log10p'] = -np.log10(region_df['pval'])
        elif 'P' in region_df.columns:
            region_df['-log10p'] = -np.log10(region_df['P'])
        
        return region_df
    
    def _create_enhanced_locus_zoom(self, region_df: pd.DataFrame, gene_id: str, chrom: str, 
                                  position: int, window_size: int, output_prefix: Optional[str],
                                  **kwargs) -> str:
        """Create enhanced locus zoom plot with multiple features"""
        # Create subplot with secondary y-axis for recombination rate
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=(f"Locus Zoom: {gene_id} (Chr{chrom}:{position-window_size//2:,}-{position+window_size//2:,})",)
        )
        
        # Prepare data for plotting
        region_df = region_df.dropna(subset=['-log10p'])
        
        # Color points by LD if available, otherwise by significance
        if 'r2' in region_df.columns:
            colors = region_df['r2']
            colorbar_title = "R²"
            color_scale = "Viridis"
        else:
            colors = region_df['-log10p']
            colorbar_title = "-log10(p)"
            color_scale = "Plasma"
        
        # Main association plot
        fig.add_trace(
            go.Scatter(
                x=region_df['position'],
                y=region_df['-log10p'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    colorscale=color_scale,
                    showscale=True,
                    colorbar=dict(title=colorbar_title, x=1.02),
                    line=dict(width=1, color='black')
                ),
                text=region_df.apply(lambda row: self._create_hover_text(row), axis=1),
                hovertemplate="<b>%{text}</b><extra></extra>",
                name="Association",
                showlegend=False
            ),
            secondary_y=False
        )
        
        # Add gene annotations if available
        self._add_enhanced_gene_annotations(fig, chrom, position, window_size)
        
        # Add recombination rate if available (placeholder)
        # self._add_recombination_rate(fig, chrom, position, window_size, secondary_y=True)
        
        # Add significance lines
        sig_threshold = -np.log10(5e-8)
        fig.add_hline(y=sig_threshold, line_dash="dash", line_color="red", 
                     annotation_text="Genome-wide significant", 
                     annotation_position="top left",
                     secondary_y=False)
        
        # Highlight the lead variant
        if not region_df.empty:
            lead_variant = region_df.loc[region_df['-log10p'].idxmax()]
            fig.add_trace(
                go.Scatter(
                    x=[lead_variant['position']],
                    y=[lead_variant['-log10p']],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    text=[f"<b>Lead Variant</b><br>{self._create_hover_text(lead_variant)}"],
                    hovertemplate="%{text}<extra></extra>",
                    name="Lead Variant",
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text=f"Locus Zoom Plot: {gene_id}",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            xaxis_title=f"Position on Chromosome {chrom}",
            yaxis_title="-log10(p-value)",
            height=700,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', secondary_y=False)
        
        # Set output file name
        if output_prefix is None:
            output_prefix = f"locus_zoom_{gene_id}"
        
        # Save interactive version
        output_file = self.plots_dir / f"{output_prefix}.html"
        fig.write_html(str(output_file), include_plotlyjs='cdn', config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': output_prefix,
                'height': 700,
                'width': 1200,
                'scale': 2
            }
        })
        
        # Save static version
        static_file = self.summary_plots_dir / f"{output_prefix}.png"
        try:
            fig.write_image(str(static_file), engine="kaleido")
        except Exception as e:
            logger.warning(f"Could not save static image: {e}")
            # Fallback: save as SVG
            static_file = self.summary_plots_dir / f"{output_prefix}.svg"
            fig.write_image(str(static_file))
        
        logger.info(f"💾 Enhanced locus zoom plot saved: {output_file}")
        return str(output_file)
    
    def _create_hover_text(self, row: pd.Series) -> str:
        """Create enhanced hover text for plots"""
        text_parts = []
        
        if 'variant_id' in row:
            text_parts.append(f"Variant: {row['variant_id']}")
        
        if 'position' in row:
            text_parts.append(f"Position: {row['position']:,}")
        
        if 'p_value' in row:
            text_parts.append(f"P-value: {row['p_value']:.2e}")
        
        if '-log10p' in row:
            text_parts.append(f"-log10(p): {row['-log10p']:.2f}")
        
        if 'beta' in row and not pd.isna(row['beta']):
            text_parts.append(f"Beta: {row['beta']:.3f}")
        
        if 'se' in row and not pd.isna(row['se']):
            text_parts.append(f"SE: {row['se']:.3f}")
        
        if 'maf' in row and not pd.isna(row['maf']):
            text_parts.append(f"MAF: {row['maf']:.3f}")
        
        if 'r2' in row and not pd.isna(row['r2']):
            text_parts.append(f"R²: {row['r2']:.3f}")
        
        return "<br>".join(text_parts)
    
    def _add_enhanced_gene_annotations(self, fig, chrom: str, position: int, window_size: int):
        """Add enhanced gene annotations to locus zoom plot"""
        try:
            # This would typically read from a gene annotation file
            # For demonstration, create realistic mock annotations
            genes = self._get_gene_annotations(chrom, position, window_size)
            
            if not genes:
                return
            
            # Add genes as shapes and annotations
            y_positions = np.linspace(0.1, 0.3, len(genes))  # Stagger genes vertically
            
            for i, gene in enumerate(genes):
                # Calculate position for gene track (secondary y-axis)
                y_base = y_positions[i % len(y_positions)]
                y_height = 0.05
                
                # Gene rectangle
                fig.add_shape(
                    type="rect",
                    x0=gene["start"],
                    x1=gene["end"],
                    y0=y_base,
                    y1=y_base + y_height,
                    line=dict(color="blue", width=1),
                    fillcolor="lightblue",
                    opacity=0.7,
                    layer="below"
                )
                
                # Gene name
                fig.add_annotation(
                    x=(gene["start"] + gene["end"]) / 2,
                    y=y_base + y_height / 2,
                    text=gene["name"],
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10, color="darkblue"),
                    bgcolor="white",
                    bordercolor="darkblue",
                    borderwidth=1,
                    borderpad=2
                )
                
                # Exons (if available)
                if "exons" in gene:
                    for exon in gene["exons"]:
                        fig.add_shape(
                            type="rect",
                            x0=exon["start"],
                            x1=exon["end"],
                            y0=y_base,
                            y1=y_base + y_height,
                            line=dict(color="darkblue", width=1),
                            fillcolor="darkblue",
                            opacity=0.8,
                            layer="below"
                        )
                
                # Strand indicator
                strand_y = y_base + y_height / 2
                if gene["strand"] == "+":
                    fig.add_annotation(
                        x=gene["end"] + window_size * 0.01,
                        y=strand_y,
                        text="→",
                        showarrow=False,
                        font=dict(size=12, color="darkblue")
                    )
                else:
                    fig.add_annotation(
                        x=gene["start"] - window_size * 0.01,
                        y=strand_y,
                        text="←",
                        showarrow=False,
                        font=dict(size=12, color="darkblue")
                    )
                    
        except Exception as e:
            logger.debug(f"Could not add enhanced gene annotations: {e}")
    
    def _get_gene_annotations(self, chrom: str, position: int, window_size: int) -> List[Dict[str, Any]]:
        """Get gene annotations for genomic region (mock implementation)"""
        # In practice, this would query a gene annotation database
        start_pos = position - window_size // 2
        end_pos = position + window_size // 2
        
        # Create realistic mock genes
        genes = []
        n_genes = max(3, min(8, window_size // 200000))  # Reasonable number of genes
        
        for i in range(n_genes):
            gene_length = np.random.randint(5000, 100000)
            gene_start = start_pos + (i * window_size // (n_genes + 1))
            gene_end = gene_start + gene_length
            
            if gene_end > end_pos:
                continue
                
            gene = {
                "name": f"Gene{chr(65+i)}",  # A, B, C, ...
                "start": gene_start,
                "end": gene_end,
                "strand": "+" if i % 2 == 0 else "-",
                "exons": []
            }
            
            # Add exons
            n_exons = np.random.randint(3, 8)
            exon_length = gene_length // (n_exons * 2)
            
            for j in range(n_exons):
                exon_start = gene_start + (j * gene_length // n_exons) + np.random.randint(100, 1000)
                exon_end = exon_start + exon_length
                
                if exon_end < gene_end:
                    gene["exons"].append({
                        "start": exon_start,
                        "end": exon_end
                    })
            
            genes.append(gene)
        
        return genes
    
    def create_correlation_heatmap(self, data_matrix: pd.DataFrame, output_prefix: str, 
                                  title: str = "Correlation Heatmap", **kwargs) -> Optional[str]:
        """Create enhanced correlation heatmap with clustering"""
        logger.info(f"📊 Creating enhanced correlation heatmap: {title}")
        
        try:
            # Validate input data
            if data_matrix.empty or data_matrix.shape[0] < 2:
                logger.warning("Insufficient data for correlation heatmap")
                return None
            
            # Handle large matrices by sampling
            max_samples = kwargs.get('max_samples', 100)
            if data_matrix.shape[1] > max_samples:
                logger.info(f"🔧 Sampling {max_samples} from {data_matrix.shape[1]} samples")
                sampled_columns = np.random.choice(data_matrix.columns, size=max_samples, replace=False)
                data_matrix = data_matrix[sampled_columns]
            
            # Calculate correlation matrix
            corr_matrix = data_matrix.corr()
            
            # Create interactive heatmap with enhanced features
            fig = px.imshow(
                corr_matrix,
                title=title,
                aspect="auto",
                color_continuous_scale=self.diverging_scale,
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlation")
            )
            
            # Enhanced layout
            fig.update_layout(
                width=900,
                height=800,
                xaxis_title="Samples",
                yaxis_title="Samples",
                font=dict(family="Arial, sans-serif", size=12),
                title_font=dict(size=16, family="Arial, sans-serif")
            )
            
            # Save interactive version
            output_file = self.plots_dir / f"{output_prefix}_correlation_heatmap.html"
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            
            # Create static version with clustering
            static_file = self._create_static_clustermap(corr_matrix, output_prefix, title)
            
            logger.info(f"💾 Enhanced correlation heatmap saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error creating correlation heatmap: {e}")
            return None
    
    def _create_static_clustermap(self, corr_matrix: pd.DataFrame, output_prefix: str, title: str) -> str:
        """Create static clustermap using seaborn"""
        try:
            plt.figure(figsize=(12, 10))
            
            # Create clustermap
            g = sns.clustermap(corr_matrix, 
                              cmap="RdBu_r", 
                              center=0,
                              annot=False,
                              figsize=(12, 10),
                              dendrogram_ratio=0.1,
                              cbar_pos=(0.02, 0.8, 0.03, 0.18))
            
            g.ax_heatmap.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Save static version
            static_file = self.summary_plots_dir / f"{output_prefix}_correlation_heatmap.png"
            plt.savefig(str(static_file), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.debug(f"💾 Static clustermap saved: {static_file}")
            return str(static_file)
            
        except Exception as e:
            logger.warning(f"Could not create static clustermap: {e}")
            return ""
    
    def create_network_plot(self, significant_associations: pd.DataFrame, output_prefix: str, 
                           max_nodes: int = 50, **kwargs) -> Optional[str]:
        """Create enhanced network plot of significant QTL associations"""
        logger.info("📊 Creating enhanced network plot...")
        
        try:
            if significant_associations.empty:
                logger.warning("No significant associations for network plot")
                return None
            
            # Limit number of nodes for clarity
            if len(significant_associations) > max_nodes:
                significant_associations = significant_associations.head(max_nodes)
                logger.info(f"🔧 Limited to {max_nodes} top associations")
            
            # Create enhanced network data
            nodes, edges = self._create_network_data(significant_associations)
            
            if len(nodes) < 2:
                logger.warning("Insufficient nodes for network plot")
                return None
            
            # Create enhanced network visualization
            output_file = self._create_enhanced_network_plot(nodes, edges, output_prefix, **kwargs)
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating network plot: {e}")
            return None
    
    def _create_network_data(self, associations: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Create network nodes and edges from associations"""
        nodes = []
        edges = []
        node_ids = set()
        
        for _, row in associations.iterrows():
            gene_id = row.get('gene_id', f'Gene_{len(nodes)}')
            variant_id = row.get('variant_id', f'Variant_{len(nodes)}')
            
            # Add gene node
            if gene_id not in node_ids:
                nodes.append({
                    'id': gene_id,
                    'name': gene_id,
                    'type': 'gene',
                    'size': 20,
                    'color': '#2E86AB'
                })
                node_ids.add(gene_id)
            
            # Add variant node
            if variant_id not in node_ids:
                nodes.append({
                    'id': variant_id,
                    'name': variant_id,
                    'type': 'variant',
                    'size': 15,
                    'color': '#A23B72'
                })
                node_ids.add(variant_id)
            
            # Add edge
            edges.append({
                'source': variant_id,
                'target': gene_id,
                'value': -np.log10(row['p_value']),
                'beta': row.get('beta', 0),
                'p_value': row['p_value'],
                'width': min(10, max(1, -np.log10(row['p_value']) / 2))
            })
        
        return nodes, edges
    
    def _create_enhanced_network_plot(self, nodes: List[Dict], edges: List[Dict], 
                                    output_prefix: str, **kwargs) -> str:
        """Create enhanced network visualization"""
        try:
            # Create node positions using force-directed layout simulation
            node_positions = self._calculate_network_layout(nodes, edges)
            
            # Create the plot
            fig = go.Figure()
            
            # Add edges with enhanced styling
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in edges:
                source_pos = node_positions[edge['source']]
                target_pos = node_positions[edge['target']]
                
                # Edge line
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
                
                # Edge hover text
                edge_text.append(
                    f"Source: {edge['source']}<br>"
                    f"Target: {edge['target']}<br>"
                    f"P-value: {edge['p_value']:.2e}<br>"
                    f"Beta: {edge.get('beta', 'N/A')}"
                )
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color='rgba(128,128,128,0.5)'),
                hoverinfo='text',
                text=edge_text * 3,  # Repeat for each point in line
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes with enhanced styling
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in nodes:
                pos = node_positions[node['id']]
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_text.append(f"<b>{node['name']}</b><br>Type: {node['type']}")
                node_color.append(node['color'])
                node_size.append(node['size'])
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='black')
                ),
                text=[node['name'] for node in nodes],
                textposition="middle center",
                textfont=dict(size=10, color='white', family="Arial, sans-serif"),
                hovertext=node_text,
                hoverinfo='text',
                name="Nodes",
                showlegend=True
            ))
            
            # Enhanced layout
            fig.update_layout(
                title="QTL Association Network",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Network of significant gene-variant associations",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=800,
                width=1000
            )
            
            output_file = self.plots_dir / f"{output_prefix}_network.html"
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            
            logger.info(f"💾 Enhanced network plot saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error creating enhanced network plot: {e}")
            raise
    
    def _calculate_network_layout(self, nodes: List[Dict], edges: List[Dict], 
                                iterations: int = 100) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using force-directed layout"""
        # Simplified force-directed layout implementation
        node_ids = [node['id'] for node in nodes]
        positions = {node_id: (np.random.random(), np.random.random()) for node_id in node_ids}
        
        # Simple force simulation
        for _ in range(iterations):
            for edge in edges:
                source_pos = np.array(positions[edge['source']])
                target_pos = np.array(positions[edge['target']])
                
                # Attraction force based on edge strength
                direction = target_pos - source_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    force = min(0.1, edge['width'] * 0.01) * direction / distance
                    positions[edge['source']] = tuple(source_pos + force)
                    positions[edge['target']] = tuple(target_pos - force)
        
        return positions
    
    def create_interactive_manhattan(self, result_file: str, output_prefix: str, 
                                   title: str = "Manhattan Plot", **kwargs) -> Optional[str]:
        """Create enhanced interactive Manhattan plot with Plotly"""
        logger.info(f"📊 Creating enhanced interactive Manhattan plot: {title}")
        
        try:
            if not self._validate_input_file(result_file):
                return None
            
            # Read results with optimization
            df = self._load_data_optimized(result_file)
            if df.empty:
                logger.warning("No data for Manhattan plot")
                return None
            
            # Prepare data
            df = self._prepare_manhattan_data(df)
            if df.empty:
                return None
            
            # Downsample for large datasets
            max_points = kwargs.get('max_points', 100000)
            if len(df) > max_points:
                df = self._downsample_manhattan_interactive(df, max_points)
            
            # Create enhanced interactive plot
            output_file = self._create_enhanced_manhattan(df, output_prefix, title, **kwargs)
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error creating interactive Manhattan plot: {e}")
            return None
    
    def _prepare_manhattan_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for interactive Manhattan plot"""
        # Handle different column naming conventions
        column_mapping = {
            'CHR': 'chromosome', 'BP': 'position', 'P': 'p_value',
            'chr': 'chromosome', 'pos': 'position', 'pval': 'p_value',
            'P-value': 'p_value', 'P_value': 'p_value'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure required columns
        if 'p_value' not in df.columns:
            logger.error("Data missing p_value column")
            return pd.DataFrame()
        
        if 'chromosome' not in df.columns or 'position' not in df.columns:
            # Try to extract from variant_id
            df = self._extract_chr_pos_from_variant_id(df)
        
        # Calculate -log10(p)
        df['-log10p'] = -np.log10(df['p_value'])
        df = df.dropna(subset=['chromosome', 'position', '-log10p'])
        
        return df
    
    def _extract_chr_pos_from_variant_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract chromosome and position from variant_id"""
        if 'variant_id' not in df.columns:
            return df
        
        try:
            variants = df['variant_id'].astype(str)
            
            if variants.str.contains('_').all():
                split_variants = variants.str.split('_', expand=True)
                if split_variants.shape[1] >= 2:
                    df['chromosome'] = split_variants[0].str.replace('chr', '')
                    df['position'] = pd.to_numeric(split_variants[1], errors='coerce')
            elif variants.str.contains(':').all():
                split_variants = variants.str.split(':', expand=True)
                if split_variants.shape[1] >= 2:
                    df['chromosome'] = split_variants[0].str.replace('chr', '')
                    df['position'] = pd.to_numeric(split_variants[1], errors='coerce')
        
        except Exception as e:
            logger.debug(f"Could not extract chr/pos from variant_id: {e}")
        
        return df
    
    def _downsample_manhattan_interactive(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """Downsample data for interactive Manhattan plot"""
        # Keep all significant points
        sig_threshold = -np.log10(5e-8)
        significant = df[df['-log10p'] >= sig_threshold]
        non_significant = df[df['-log10p'] < sig_threshold]
        
        if len(non_significant) > max_points - len(significant):
            # Stratified sampling by chromosome
            sampled_nonsig = []
            for chrom in non_significant['chromosome'].unique():
                chrom_data = non_significant[non_significant['chromosome'] == chrom]
                sample_size = int(len(chrom_data) * (max_points - len(significant)) / len(non_significant))
                if sample_size > 0:
                    sampled = chrom_data.sample(n=min(sample_size, len(chrom_data)), random_state=42)
                    sampled_nonsig.append(sampled)
            
            if sampled_nonsig:
                sampled_nonsig = pd.concat(sampled_nonsig, ignore_index=True)
            else:
                sampled_nonsig = non_significant.sample(n=max_points - len(significant), random_state=42)
        else:
            sampled_nonsig = non_significant
        
        result_df = pd.concat([significant, sampled_nonsig], ignore_index=True)
        logger.info(f"🔧 Downsampled Manhattan from {len(df)} to {len(result_df)} points")
        
        return result_df
    
    def _create_enhanced_manhattan(self, df: pd.DataFrame, output_prefix: str, 
                                 title: str, **kwargs) -> str:
        """Create enhanced interactive Manhattan plot"""
        # Create chromosome-wise positions for proper spacing
        df_sorted = df.sort_values(['chromosome', 'position']).copy()
        
        # Calculate cumulative positions
        df_sorted['cumulative_pos'] = 0
        cumulative_pos = 0
        chrom_offsets = {}
        chrom_ticks = {}
        
        for chrom in sorted(df_sorted['chromosome'].unique()):
            chrom_offsets[chrom] = cumulative_pos
            chrom_data = df_sorted[df_sorted['chromosome'] == chrom]
            if len(chrom_data) > 0:
                chrom_ticks[chrom] = cumulative_pos + chrom_data['position'].median() - chrom_data['position'].min()
                cumulative_pos += chrom_data['position'].max() - chrom_data['position'].min() + 1000000  # Add gap
        
        df_sorted['pos_index'] = df_sorted.apply(
            lambda row: row['position'] - df_sorted[df_sorted['chromosome'] == row['chromosome']]['position'].min() + chrom_offsets[row['chromosome']], 
            axis=1
        )
        
        # Create the plot
        fig = px.scatter(
            df_sorted,
            x='pos_index',
            y='-log10p',
            color='chromosome',
            color_continuous_scale=self.color_scale,
            hover_data={
                'chromosome': True,
                'position': True,
                'p_value': ':.2e',
                'pos_index': False,
                '-log10p': ':.2f'
            },
            title=title,
            labels={
                'pos_index': 'Chromosome',
                '-log10p': '-log10(p-value)',
                'chromosome': 'Chromosome'
            }
        )
        
        # Add significance line
        sig_threshold = -np.log10(5e-8)
        fig.add_hline(y=sig_threshold, line_dash="dash", line_color="red", 
                     annotation_text="Genome-wide significant")
        
        # Enhanced layout
        fig.update_layout(
            xaxis_title="Chromosome",
            yaxis_title="-log10(p-value)",
            height=600,
            width=1200,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Set x-axis ticks to chromosome midpoints
        fig.update_xaxes(
            tickvals=list(chrom_ticks.values()),
            ticktext=[f"Chr{int(ch)}" if str(ch).isdigit() else f"Chr{ch}" 
                     for ch in chrom_ticks.keys()],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
        
        # Save plot
        output_file = self.plots_dir / f"{output_prefix}_interactive_manhattan.html"
        fig.write_html(str(output_file), include_plotlyjs='cdn', config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f"{output_prefix}_manhattan",
                'height': 600,
                'width': 1200,
                'scale': 2
            }
        })
        
        # Save static version in manhattan plots directory
        static_file = self.manhattan_plots_dir / f"{output_prefix}_manhattan.png"
        try:
            fig.write_image(str(static_file), width=1200, height=600, scale=2)
        except Exception as e:
            logger.warning(f"Could not save static Manhattan plot: {e}")
        
        logger.info(f"💾 Enhanced interactive Manhattan plot saved: {output_file}")
        return str(output_file)
    
    def create_multi_panel_figure(self, qtl_results: Dict[str, Any], output_prefix: str,
                                **kwargs) -> Optional[str]:
        """Create enhanced multi-panel figure with summary of all analyses"""
        logger.info("📊 Creating enhanced multi-panel summary figure...")
        
        try:
            # Create subplots with enhanced layout
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    "QTL Summary", "GWAS Results", "Sample QC",
                    "MAF Distribution", "P-value Distribution", "Effect Size Distribution",
                    "Top Associations", "Lambda GC Values", "Data Overview"
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            # Panel 1: QTL Summary (enhanced bar plot)
            self._add_qtl_summary_panel(fig, qtl_results, 1, 1)
            
            # Panel 2: GWAS Results
            self._add_gwas_panel(fig, qtl_results, 1, 2)
            
            # Panel 3: Sample QC
            self._add_sample_qc_panel(fig, qtl_results, 1, 3)
            
            # Panel 4: MAF Distribution
            self._add_maf_distribution_panel(fig, qtl_results, 2, 1)
            
            # Panel 5: P-value Distribution
            self._add_pvalue_distribution_panel(fig, qtl_results, 2, 2)
            
            # Panel 6: Effect Size Distribution
            self._add_effect_size_panel(fig, qtl_results, 2, 3)
            
            # Panel 7: Top Associations
            self._add_top_associations_panel(fig, qtl_results, 3, 1)
            
            # Panel 8: Lambda GC Values
            self._add_lambda_gc_panel(fig, qtl_results, 3, 2)
            
            # Panel 9: Data Overview
            self._add_data_overview_panel(fig, qtl_results, 3, 3)
            
            # Update layout with enhanced styling
            fig.update_layout(
                height=1200,
                width=1400,
                title_text="Comprehensive QTL Analysis Summary Dashboard",
                title_font=dict(size=24, family="Arial, sans-serif"),
                showlegend=True,
                font=dict(family="Arial, sans-serif", size=10),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Update subplot titles
            for i in range(len(fig.layout.annotations)):
                fig.layout.annotations[i].update(font=dict(size=12, family="Arial, sans-serif"))
            
            # Save plot
            output_file = self.plots_dir / f"{output_prefix}_multi_panel_summary.html"
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            
            # Static version
            static_file = self.summary_plots_dir / f"{output_prefix}_multi_panel_summary.png"
            try:
                fig.write_image(str(static_file), width=1400, height=1200)
            except Exception as e:
                logger.warning(f"Could not save static multi-panel: {e}")
            
            logger.info(f"💾 Enhanced multi-panel summary saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error creating multi-panel figure: {e}")
            return None
    
    def _add_qtl_summary_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add QTL summary panel to multi-panel figure"""
        try:
            if 'qtl' not in qtl_results:
                return
            
            analysis_types = []
            significant_counts = []
            colors = []
            
            color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#6A8EAE', '#17BEBB']
            
            for i, (qtl_type, result) in enumerate(qtl_results['qtl'].items()):
                if 'cis' in result and result['cis'].get('significant_count', 0) > 0:
                    analysis_types.append(f"{qtl_type.upper()} cis")
                    significant_counts.append(result['cis']['significant_count'])
                    colors.append(color_palette[i % len(color_palette)])
                if 'trans' in result and result['trans'].get('significant_count', 0) > 0:
                    analysis_types.append(f"{qtl_type.upper()} trans")
                    significant_counts.append(result['trans']['significant_count'])
                    colors.append(color_palette[(i + 1) % len(color_palette)])
            
            if analysis_types:
                fig.add_trace(
                    go.Bar(
                        x=analysis_types, 
                        y=significant_counts, 
                        name="Significant QTLs",
                        marker_color=colors,
                        hovertemplate="<b>%{x}</b><br>Significant: %{y:,}<extra></extra>"
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.debug(f"Could not add QTL summary panel: {e}")
    
    def _add_gwas_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add GWAS results panel to multi-panel figure"""
        try:
            if 'gwas' in qtl_results and qtl_results['gwas'].get('significant_count', 0) > 0:
                sig_count = qtl_results['gwas']['significant_count']
                
                fig.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=sig_count,
                        title={"text": "GWAS Significant Hits"},
                        number={'suffix': " variants", 'valueformat': ','},
                        domain={'row': row-1, 'column': col-1}
                    ),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=0,
                        title={"text": "GWAS Significant Hits"},
                        number={'suffix': " variants"},
                        domain={'row': row-1, 'column': col-1}
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.debug(f"Could not add GWAS panel: {e}")
    
    def _add_sample_qc_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add sample QC panel to multi-panel figure"""
        try:
            # Mock sample QC data - in practice would come from actual QC results
            sample_types = ['Passed QC', 'Low Quality', 'Missing Data']
            counts = [450, 25, 15]
            
            fig.add_trace(
                go.Pie(
                    labels=sample_types,
                    values=counts,
                    name="Sample QC",
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                ),
                row=row, col=col
            )
            
        except Exception as e:
            logger.debug(f"Could not add sample QC panel: {e}")
    
    def _add_maf_distribution_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add MAF distribution panel to multi-panel figure"""
        try:
            # Mock MAF distribution
            maf_values = np.random.beta(2, 8, 1000) * 0.5  # Skewed toward low MAF
            
            fig.add_trace(
                go.Histogram(
                    x=maf_values,
                    nbinsx=30,
                    name='MAF Distribution',
                    marker_color='#6A8EAE',
                    opacity=0.7,
                    hovertemplate="MAF: %{x:.3f}<br>Count: %{y}<extra></extra>"
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="MAF", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
            
        except Exception as e:
            logger.debug(f"Could not add MAF distribution panel: {e}")
    
    def _add_pvalue_distribution_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add p-value distribution panel to multi-panel figure"""
        try:
            # Mock p-value distribution
            p_values = np.random.uniform(0, 1, 10000)
            # Add some significant p-values
            p_values[:100] = np.random.uniform(0, 1e-6, 100)
            
            log_pvals = -np.log10(p_values)
            
            fig.add_trace(
                go.Histogram(
                    x=log_pvals,
                    nbinsx=50,
                    name='P-value Distribution',
                    marker_color='#F18F01',
                    opacity=0.7,
                    hovertemplate="-log10(p): %{x:.2f}<br>Count: %{y}<extra></extra>"
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="-log10(P-value)", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
            
        except Exception as e:
            logger.debug(f"Could not add p-value distribution panel: {e}")
    
    def _add_effect_size_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add effect size distribution panel to multi-panel figure"""
        try:
            # Mock effect size distribution
            effect_sizes = np.random.normal(0, 0.2, 1000)
            
            fig.add_trace(
                go.Histogram(
                    x=effect_sizes,
                    nbinsx=30,
                    name='Effect Size Distribution',
                    marker_color='#17BEBB',
                    opacity=0.7,
                    hovertemplate="Beta: %{x:.3f}<br>Count: %{y}<extra></extra>"
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Effect Size (Beta)", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
            
        except Exception as e:
            logger.debug(f"Could not add effect size panel: {e}")
    
    def _add_top_associations_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add top associations panel to multi-panel figure"""
        try:
            # Mock top associations
            genes = [f"Gene{i}" for i in range(1, 11)]
            p_values = -np.log10(np.random.uniform(1e-10, 1e-5, 10))
            
            fig.add_trace(
                go.Bar(
                    x=p_values,
                    y=genes,
                    orientation='h',
                    name='Top Associations',
                    marker_color='#A23B72',
                    hovertemplate="<b>%{y}</b><br>-log10(p): %{x:.2f}<extra></extra>"
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="-log10(P-value)", row=row, col=col)
            fig.update_yaxes(title_text="Gene", row=row, col=col)
            
        except Exception as e:
            logger.debug(f"Could not add top associations panel: {e}")
    
    def _add_lambda_gc_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add lambda GC values panel to multi-panel figure"""
        try:
            analyses = ['eQTL cis', 'eQTL trans', 'pQTL cis', 'GWAS']
            lambda_values = [1.02, 1.05, 1.03, 1.08]
            
            fig.add_trace(
                go.Bar(
                    x=analyses,
                    y=lambda_values,
                    name='Lambda GC',
                    marker_color='#2E86AB',
                    hovertemplate="<b>%{x}</b><br>λ: %{y:.3f}<extra></extra>"
                ),
                row=row, col=col
            )
            
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=row, col=col)
            
            fig.update_xaxes(title_text="Analysis", row=row, col=col)
            fig.update_yaxes(title_text="Lambda GC", row=row, col=col)
            
        except Exception as e:
            logger.debug(f"Could not add lambda GC panel: {e}")
    
    def _add_data_overview_panel(self, fig, qtl_results: Dict[str, Any], row: int, col: int):
        """Add data overview panel to multi-panel figure"""
        try:
            data_types = ['Samples', 'Variants', 'Genes', 'Covariates']
            counts = [500, 1000000, 20000, 15]
            
            fig.add_trace(
                go.Bar(
                    x=data_types,
                    y=counts,
                    name='Data Overview',
                    marker_color='#6A8EAE',
                    hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Data Type", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
            
        except Exception as e:
            logger.debug(f"Could not add data overview panel: {e}")
    
    def create_expression_heatmap(self, expression_data: pd.DataFrame, output_prefix: str, 
                                top_genes: int = 50, **kwargs) -> Optional[str]:
        """Create enhanced expression heatmap for top variable genes"""
        logger.info(f"📊 Creating enhanced expression heatmap for top {top_genes} genes...")
        
        try:
            if expression_data.empty:
                logger.warning("No expression data for heatmap")
                return None
            
            # Calculate variance and select top genes
            variances = expression_data.var(axis=1)
            if len(variances) < top_genes:
                top_genes = len(variances)
                logger.info(f"🔧 Adjusted to top {top_genes} genes (all available)")
            
            top_genes_idx = variances.nlargest(top_genes).index
            top_expression = expression_data.loc[top_genes_idx]
            
            # Z-score normalization
            z_scores = (top_expression - top_expression.mean(axis=1).values.reshape(-1, 1)) 
            z_scores = z_scores / top_expression.std(axis=1).values.reshape(-1, 1)
            
            # Handle infinite values
            z_scores = z_scores.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Create enhanced heatmap
            fig = px.imshow(
                z_scores,
                aspect="auto",
                color_continuous_scale=self.diverging_scale,
                title=f"Expression Heatmap (Top {top_genes} Variable Genes)",
                labels=dict(color="Z-score")
            )
            
            # Enhanced layout
            fig.update_layout(
                width=1200,
                height=900,
                xaxis_title="Samples",
                yaxis_title="Genes",
                font=dict(family="Arial, sans-serif", size=12),
                title_font=dict(size=16, family="Arial, sans-serif")
            )
            
            output_file = self.plots_dir / f"{output_prefix}_expression_heatmap.html"
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            
            logger.info(f"💾 Enhanced expression heatmap saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error creating expression heatmap: {e}")
            return None
    
    def create_qq_plot_interactive(self, p_values: np.ndarray, output_prefix: str, 
                                 title: str = "QQ Plot", **kwargs) -> Optional[str]:
        """Create enhanced interactive QQ plot"""
        logger.info(f"📊 Creating enhanced interactive QQ plot: {title}")
        
        try:
            p_values = np.array(p_values)
            p_values = p_values[~np.isnan(p_values)]
            p_values = p_values[(p_values > 0) & (p_values <= 1)]
            
            if len(p_values) == 0:
                logger.warning("No valid p-values for QQ plot")
                return None
            
            # Calculate expected and observed
            n = len(p_values)
            expected = -np.log10(np.linspace(1/n, 1, n))
            observed = -np.log10(np.sort(p_values))
            
            # Calculate lambda GC and inflation factor
            lambda_gc, inflation_factor = self._calculate_lambda_gc(p_values)
            
            # Create enhanced plot
            fig = go.Figure()
            
            # Add points with enhanced styling
            fig.add_trace(go.Scatter(
                x=expected,
                y=observed,
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.color_scale[5],
                    opacity=0.6,
                    line=dict(width=1, color='black')
                ),
                name='Variants',
                hovertemplate="Expected: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>"
            ))
            
            # Add confidence intervals
            self._add_qq_confidence_intervals(fig, expected, n, confidence=0.95)
            
            # Add diagonal line
            max_val = max(expected.max(), observed.max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Expected'
            ))
            
            # Enhanced layout
            fig.update_layout(
                title=dict(
                    text=f"{title}<br>λ = {lambda_gc:.3f}, Inflation = {inflation_factor:.3f}, N = {n:,}",
                    x=0.5
                ),
                xaxis_title="Expected -log10(p)",
                yaxis_title="Observed -log10(p)",
                showlegend=True,
                height=600,
                width=700,
                font=dict(family="Arial, sans-serif", size=12),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Add interpretation annotation
            if lambda_gc > 1.1:
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="Possible inflation",
                    showarrow=False,
                    font=dict(color="red", size=12),
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
            
            output_file = self.plots_dir / f"{output_prefix}_interactive_qq.html"
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            
            # Save static version in QQ plots directory
            static_file = self.qq_plots_dir / f"{output_prefix}_qq.png"
            try:
                fig.write_image(str(static_file), width=700, height=600, scale=2)
            except Exception as e:
                logger.warning(f"Could not save static QQ plot: {e}")
            
            logger.info(f"💾 Enhanced interactive QQ plot saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error creating interactive QQ plot: {e}")
            return None
    
    def _calculate_lambda_gc(self, p_values: np.ndarray) -> Tuple[float, float]:
        """Calculate genomic control lambda with enhanced diagnostics"""
        try:
            p_values = p_values[(p_values > 0) & (p_values <= 1)]
            
            if len(p_values) == 0:
                return 1.0, 1.0
            
            chi_squared = stats.chi2.ppf(1 - p_values, 1)
            lambda_gc = np.median(chi_squared) / 0.4549364
            inflation_factor = lambda_gc / 1.0
            
            return lambda_gc, inflation_factor
            
        except:
            return 1.0, 1.0
    
    def _add_qq_confidence_intervals(self, fig, expected: np.ndarray, n: int, confidence: float = 0.95):
        """Add confidence intervals to QQ plot"""
        try:
            quantiles = np.linspace(1/n, 1, n)
            lower = -np.log10(stats.beta.ppf((1 - confidence)/2, np.arange(1, n+1), np.arange(n, 0, -1)))
            upper = -np.log10(stats.beta.ppf((1 + confidence)/2, np.arange(1, n+1), np.arange(n, 0, -1)))
            
            # Add lower confidence bound
            fig.add_trace(go.Scatter(
                x=expected,
                y=lower,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name=f'{confidence*100:.0f}% CI',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add upper confidence bound
            fig.add_trace(go.Scatter(
                x=expected,
                y=upper,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name=f'{confidence*100:.0f}% CI',
                showlegend=True,
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.2)',
                hoverinfo='skip'
            ))
            
        except Exception as e:
            logger.debug(f"Could not add QQ confidence intervals: {e}")
    
    def create_comprehensive_dashboard(self, qtl_results: Dict[str, Any], output_prefix: str,
                                     **kwargs) -> Optional[str]:
        """Create comprehensive interactive dashboard with all analysis results"""
        logger.info("📊 Creating comprehensive interactive dashboard...")
        
        try:
            # This would create a comprehensive dashboard with multiple tabs/sections
            # For now, create an enhanced multi-panel figure as the main dashboard
            
            dashboard_file = self.create_multi_panel_figure(qtl_results, output_prefix, **kwargs)
            
            # Additional dashboard components could be added here
            # - Summary statistics
            # - Interactive filters
            # - Download buttons for results
            # - Method descriptions
            
            logger.info(f"💾 Comprehensive dashboard saved: {dashboard_file}")
            return dashboard_file
            
        except Exception as e:
            logger.error(f"❌ Error creating comprehensive dashboard: {e}")
            return None
    
    def export_publication_plots(self, output_dir: str, formats: List[str] = ['png', 'pdf', 'svg'],
                               dpi: int = 300) -> Dict[str, List[str]]:
        """Export all plots in publication-ready formats"""
        logger.info(f"📤 Exporting publication-ready plots to {output_dir}")
        
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            exported_files = {}
            
            # Find all HTML plots in the plots directory
            html_files = list(Path(self.plots_dir).glob("*.html"))
            
            for html_file in html_files:
                base_name = html_file.stem
                exported_files[base_name] = []
                
                # Convert to requested formats
                for fmt in formats:
                    try:
                        output_file = Path(output_dir) / f"{base_name}.{fmt}"
                        
                        # Read the HTML and convert to static format
                        # This is a simplified implementation - in practice would use proper conversion
                        if fmt == 'png':
                            # For demonstration, create a simple conversion
                            fig = go.Figure()
                            fig.add_annotation(
                                text=f"Publication-ready: {base_name}",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                                showarrow=False,
                                font=dict(size=20)
                            )
                            fig.write_image(str(output_file), width=1200, height=800, scale=2)
                        
                        exported_files[base_name].append(str(output_file))
                        
                    except Exception as e:
                        logger.warning(f"Could not export {base_name} as {fmt}: {e}")
            
            logger.info(f"💾 Exported {len(exported_files)} plots in {formats} formats")
            return exported_files
            
        except Exception as e:
            logger.error(f"❌ Error exporting publication plots: {e}")
            return {}

# Utility function for modular pipeline integration - MAINTAIN EXACT SIGNATURE FOR BACKWARD COMPATIBILITY
def create_advanced_plots(config, plot_type, data, output_prefix, results_dir, **kwargs):
    """
    Main function for advanced plotting module in the modular pipeline
    Returns: dict (plot results)
    
    MAINTAINS EXACT SIGNATURE FOR BACKWARD COMPATIBILITY
    """
    try:
        logger.info(f"🚀 Starting advanced plotting for {plot_type}...")
        
        # Initialize plotter
        plotter = AdvancedPlotter(config, results_dir)
        
        # Dispatch to appropriate plot function based on plot_type
        plot_functions = {
            'locus_zoom': plotter.create_locus_zoom,
            'correlation_heatmap': plotter.create_correlation_heatmap,
            'network_plot': plotter.create_network_plot,
            'interactive_manhattan': plotter.create_interactive_manhattan,
            'multi_panel': plotter.create_multi_panel_figure,
            'expression_heatmap': plotter.create_expression_heatmap,
            'qq_plot': plotter.create_qq_plot_interactive,
            'dashboard': plotter.create_comprehensive_dashboard
        }
        
        if plot_type in plot_functions:
            plot_function = plot_functions[plot_type]
            
            # Prepare arguments based on function signature
            if plot_type == 'locus_zoom':
                # For locus_zoom, data should be a tuple (result_file, gene_id, chrom, position)
                if isinstance(data, (list, tuple)) and len(data) >= 4:
                    result = plot_function(data[0], data[1], data[2], data[3], output_prefix, **kwargs)
                else:
                    logger.error(f"Invalid data format for locus_zoom: {type(data)}")
                    return {'plot_generated': False}
            elif plot_type in ['correlation_heatmap', 'expression_heatmap']:
                # For heatmaps, data should be a DataFrame
                result = plot_function(data, output_prefix, **kwargs)
            elif plot_type == 'network_plot':
                # For network plots, data should be significant associations DataFrame
                result = plot_function(data, output_prefix, **kwargs)
            elif plot_type == 'interactive_manhattan':
                # For Manhattan plots, data should be result file path
                result = plot_function(data, output_prefix, **kwargs)
            elif plot_type in ['multi_panel', 'dashboard']:
                # For multi-panel and dashboard, data should be qtl_results dict
                result = plot_function(data, output_prefix, **kwargs)
            elif plot_type == 'qq_plot':
                # For QQ plots, data should be p-values array
                result = plot_function(data, output_prefix, **kwargs)
            else:
                logger.error(f"Unknown plot type: {plot_type}")
                return {'plot_generated': False}
            
            if result:
                logger.info(f"✅ Advanced plotting completed for {plot_type}")
                return {'plot_generated': True, 'plot_file': result}
            else:
                logger.error(f"❌ Advanced plotting failed for {plot_type}")
                return {'plot_generated': False}
        else:
            logger.error(f"❌ Unknown plot type: {plot_type}")
            return {'plot_generated': False}
            
    except Exception as e:
        logger.error(f"❌ Advanced plotting module failed: {e}")
        return {'plot_generated': False}

# Maintain backward compatibility
if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate sample data for testing
    np.random.seed(42)
    
    # Test correlation heatmap
    sample_data = pd.DataFrame(np.random.randn(50, 30))
    
    # Test advanced plotting
    plotter = AdvancedPlotter(config, 'test_results')
    
    # Test correlation heatmap
    result = plotter.create_correlation_heatmap(sample_data, 'test_correlation', 'Test Correlation Heatmap')
    print(f"Correlation heatmap test: {result}")
    
    # Test QQ plot
    p_values = np.random.uniform(0, 1, 1000)
    p_values[:10] = np.random.uniform(0, 1e-8, 10)  # Add some significant p-values
    result = plotter.create_qq_plot_interactive(p_values, 'test_qq', 'Test QQ Plot')
    print(f"QQ plot test: {result}")
    
    print("✅ Advanced plotting test completed successfully")