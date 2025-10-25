"""
Advanced analysis modules for QTL pipeline - Enhanced Version
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with modular pipeline support and function exports.
"""

import logging

# Set up logger
logger = logging.getLogger('QTLPipeline')

from .interaction_analysis import InteractionAnalysis, run_interaction_analysis
from .fine_mapping import FineMapping, run_fine_mapping

__all__ = [
    'InteractionAnalysis',
    'FineMapping',
    'run_interaction_analysis', 
    'run_fine_mapping'
]

# Version info for analysis modules
__version__ = "2.0.0"
__author__ = "Dr. Vijay Singh"
__email__ = "vijay.s.gautam@gmail.com"