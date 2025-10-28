"""
QTL Pipeline Utilities Package - Enhanced Version
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with modular pipeline support and backward compatibility.
"""

import logging

# Set up logger
logger = logging.getLogger('QTLPipeline')

from .validation import validate_inputs
from .genotype_processing import GenotypeProcessor, process_genotypes
from .qtl_analysis import prepare_genotypes, run_cis_analysis, run_trans_analysis, process_expression_data, run_qtl_mapping
from .gwas_analysis import run_gwas_analysis
from .plotting import QTLPlotter, generate_all_plots
from .report_generator import generate_html_report, generate_summary_report, generate_reports
from .enhanced_qc import EnhancedQC, run_data_preparation, run_quality_control
from .advanced_plotting import AdvancedPlotter

# Import analysis modules
try:
    from ..analysis.interaction_analysis import InteractionAnalysis, run_interaction_analysis
    from ..analysis.fine_mapping import FineMapping, run_fine_mapping
except ImportError as e:
    # These might not be available if the analysis directory doesn't exist yet
    logger.warning(f"Analysis modules not available: {e}")
    InteractionAnalysis = None
    FineMapping = None
    run_interaction_analysis = None
    run_fine_mapping = None

__all__ = [
    # Core functionality
    'validate_inputs',
    'GenotypeProcessor', 
    'prepare_genotypes',
    'run_cis_analysis',
    'run_trans_analysis',
    'run_gwas_analysis',
    'QTLPlotter',
    'generate_html_report',
    'generate_summary_report',
    'EnhancedQC',
    'AdvancedPlotter',
    
    # Modular pipeline functions
    'process_genotypes',
    'process_expression_data', 
    'run_qtl_mapping',
    'run_data_preparation',
    'run_quality_control',
    'generate_all_plots',
    'generate_reports',
    
    # Analysis modules
    'InteractionAnalysis',
    'FineMapping',
    'run_interaction_analysis',
    'run_fine_mapping'
]

# Version info
__version__ = "2.0.0"
__author__ = "Dr. Vijay Singh"
__email__ = "vijay.s.gautam@gmail.com"