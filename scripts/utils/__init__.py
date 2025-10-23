"""
QTL Pipeline Utilities Package - Enhanced Version
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

from .validation import validate_inputs
from .genotype_processing import GenotypeProcessor
from .qtl_analysis import prepare_genotypes, run_cis_analysis, run_trans_analysis
from .gwas_analysis import run_gwas_analysis
from .plotting import QTLPlotter
from .report_generator import generate_html_report, generate_summary_report
from .enhanced_qc import EnhancedQC
from .advanced_plotting import AdvancedPlotter

# Import analysis modules
try:
    from ..analysis.interaction_analysis import InteractionAnalysis
    from ..analysis.fine_mapping import FineMapping
except ImportError:
    # These might not be available if the analysis directory doesn't exist yet
    InteractionAnalysis = None
    FineMapping = None

__all__ = [
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
    'InteractionAnalysis',
    'FineMapping'
]