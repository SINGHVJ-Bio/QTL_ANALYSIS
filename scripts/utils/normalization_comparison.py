#!/usr/bin/env python3
"""
Enhanced Normalization Comparison Visualization with Performance Optimizations
Creates comprehensive side-by-side plots of raw vs normalized data for QTL analysis
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with parallel processing, memory optimization, and comprehensive reporting.
Now includes batch correction pipeline integration and Python-based DESeq2 VST.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gc
import psutil
import yaml
import subprocess
import tempfile

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

try:
    from scripts.utils.batch_correction import run_batch_correction_pipeline
    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    logging.warning("Batch correction module not available, batch correction will be disabled")

# Import DESeq2 VST Python implementation
try:
    from scripts.utils.deseq2_vst_python import deseq2_vst_python, simple_vst_fallback
    DESEQ2_VST_AVAILABLE = True
    logging.info("✅ DESeq2 VST Python module successfully imported")
except ImportError as e:
    DESEQ2_VST_AVAILABLE = False
    logging.warning(f"⚠️ DESeq2 VST Python module not available: {e}")
   
warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

class NormalizationComparison:
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        self.comparison_dir = os.path.join(results_dir, "normalization_comparison")
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Performance optimization settings
        self.max_samples = config.get('performance', {}).get('max_comparison_samples', 1000)
        self.max_features = config.get('performance', {}).get('max_comparison_features', 100)
        self.parallel_processing = config.get('performance', {}).get('parallel_comparison', True)
        self.num_workers = min(4, config.get('performance', {}).get('num_threads', 4))
        
        # Enhanced pipeline settings
        self.enable_enhanced_pipeline = config.get('enhanced_pipeline', {}).get('enable', True)
        self.enable_batch_correction = config.get('batch_correction', {}).get('enabled', {}).get('eqtl', True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C5C5C5']
        
        # Cache for expensive computations
        self._cache = {}
        
    def generate_comprehensive_comparison(self, qtl_type, raw_data, normalized_data, normalization_method):
        """Generate comprehensive comparison plots for normalization with performance optimizations"""
        logger.info(f"📊 Generating enhanced normalization comparison for {qtl_type}...")
        
        try:
            # Create subdirectory for this QTL type
            qtl_dir = os.path.join(self.comparison_dir, qtl_type)
            os.makedirs(qtl_dir, exist_ok=True)
            
            # Sample data for performance if needed
            raw_data, normalized_data = self._sample_data_for_performance(raw_data, normalized_data)
            
            comparison_results = {
                'qtl_type': qtl_type,
                'normalization_method': normalization_method,
                'raw_data_shape': raw_data.shape,
                'normalized_data_shape': normalized_data.shape,
                'plots_generated': []
            }
            
            # Run comparisons in parallel where possible
            if self.parallel_processing:
                parallel_results = self._run_parallel_comparisons(
                    raw_data, normalized_data, qtl_type, normalization_method, qtl_dir
                )
                comparison_results.update(parallel_results)
                # Ensure plots_generated is maintained
                if 'plots_generated' in parallel_results:
                    comparison_results['plots_generated'].extend(parallel_results['plots_generated'])
            else:
                sequential_results = self._run_sequential_comparisons(
                    raw_data, normalized_data, qtl_type, normalization_method, qtl_dir
                )
                comparison_results.update(sequential_results)
                # Ensure plots_generated is maintained
                if 'plots_generated' in sequential_results:
                    comparison_results['plots_generated'].extend(sequential_results['plots_generated'])
            
            # Generate comprehensive HTML report
            html_report = self.generate_comparison_html_report(comparison_results, qtl_type, qtl_dir)
            comparison_results['html_report'] = html_report
            
            # Clear cache to free memory
            self._cache.clear()
            gc.collect()
            
            logger.info(f"✅ Enhanced normalization comparison completed for {qtl_type}: {len(comparison_results['plots_generated'])} plots generated")
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Error generating normalization comparison for {qtl_type}: {e}")
            return {'plots_generated': []}
    
    def generate_enhanced_pipeline_comparison(self, qtl_type, raw_data, normalization_method=None):
        """
        Enhanced pipeline: Filtering → Normalization → Batch Correction → Comparisons
        Maintains backward compatibility while adding new functionality
        """
        try:
            logger.info(f"🚀 Starting enhanced pipeline for {qtl_type}...")
            
            # Get normalization method from config if not provided
            if normalization_method is None:
                normalization_method = self._get_normalization_method(qtl_type)
            
            # Step 1: Run normalization pipeline
            normalized_file, normalized_data = self._run_normalization_pipeline(raw_data, qtl_type, normalization_method)
            
            # Step 2: Run batch correction if enabled
            batch_correction_results = self._run_batch_correction_pipeline(normalized_data, qtl_type, normalization_method)
            
            # Step 3: Run comparisons
            comparison_results = self._run_enhanced_comparisons(
                qtl_type, raw_data, normalized_data, batch_correction_results, normalization_method
            )
            
            # Save pipeline summary
            self._save_enhanced_pipeline_summary(qtl_type, normalization_method, batch_correction_results, comparison_results)
            
            logger.info(f"✅ Enhanced pipeline completed for {qtl_type}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced pipeline failed for {qtl_type}: {e}")
            # Fall back to basic comparison
            return self.generate_comprehensive_comparison(qtl_type, raw_data, raw_data, normalization_method)
    
    def _get_normalization_method(self, qtl_type):
        """Get normalization method from config for specific QTL type"""
        normalization_config = self.config.get('normalization', {})
        qtl_config = normalization_config.get(qtl_type, {})
        
        method = qtl_config.get('method', 'vst')
        logger.info(f"🔧 Using normalization method for {qtl_type}: {method}")
        return method
    
    def _run_normalization_pipeline(self, raw_data, qtl_type, method):
        """Run normalization pipeline using Python instead of R"""
        try:
            logger.info(f"🔧 Running {method} normalization using Python for {qtl_type}...")
            
            # Create temporary directory for processing
            temp_dir = os.path.join(self.results_dir, "temp_normalization")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Apply normalization using Python
            if method == 'vst':
                normalized_data = self._apply_vst_normalization_python(raw_data, qtl_type)
            elif method == 'log2':
                normalized_data = self._apply_log2_normalization_python(raw_data, qtl_type)
            elif method == 'quantile':
                normalized_data = self._apply_quantile_normalization_python(raw_data, qtl_type)
            elif method == 'zscore':
                normalized_data = self._apply_zscore_normalization_python(raw_data, qtl_type)
            elif method == 'arcsinh':
                normalized_data = self._apply_arcsinh_normalization_python(raw_data, qtl_type)
            elif method == 'tpm':
                normalized_data = self._apply_tpm_normalization_python(raw_data, qtl_type)
            else:
                logger.warning(f"⚠️ Unknown normalization method '{method}', using raw data")
                normalized_data = raw_data.copy()
            
            # Save normalized data
            normalized_file = os.path.join(temp_dir, f"{qtl_type}.normalised.{method}.tsv")
            normalized_data.to_csv(normalized_file, sep='\t')
            
            logger.info(f"✅ Python {method} normalization completed: {normalized_data.shape}")
            
            return normalized_file, normalized_data
            
        except Exception as e:
            logger.error(f"❌ Python normalization pipeline failed: {e}")
            raise
    
    def _apply_vst_normalization_python(self, raw_data, qtl_type):
        """Apply VST normalization using Python implementation"""
        if qtl_type != 'eqtl':
            logger.warning("VST normalization is typically for expression data")

        logger.info("🔢 Applying DESeq2 VST normalization (Python implementation)...")
        
        if not DESEQ2_VST_AVAILABLE:
            raise ImportError("DESeq2 VST Python implementation not available")
        
        # Get parameters from config
        normalization_config = self.config.get('normalization', {})
        qtl_config = normalization_config.get(qtl_type, {})
        blind = qtl_config.get('vst_blind', True)
        fit_type = qtl_config.get('fit_type', 'parametric')
        
        try:
            # Ensure data is appropriate for VST (non-negative)
            if (raw_data < 0).any().any():
                logger.warning("Negative values found in data. Taking absolute values for VST.")
                raw_data = raw_data.abs()
            
            # Ensure data is numeric and finite
            raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
            raw_data = raw_data.replace([np.inf, -np.inf], np.nan)
            raw_data = raw_data.fillna(0)
            
            # Apply VST normalization
            try:
                vst_df = deseq2_vst_python(raw_data, blind=blind, fit_type=fit_type)
                logger.info("✅ DESeq2 VST normalization completed (full implementation)")
            except Exception as e:
                logger.warning(f"⚠️ Full VST implementation failed: {e}, using simplified version")
                vst_df = simple_vst_fallback(raw_data)
                logger.info("✅ Simplified VST normalization completed")
            
            return vst_df
            
        except Exception as e:
            logger.error(f"❌ VST normalization failed: {e}")
            # Fallback to log2 if VST fails
            logger.info("🔄 Falling back to log2 normalization")
            return self._apply_log2_normalization_python(raw_data, qtl_type)
    
    def _apply_log2_normalization_python(self, raw_data, qtl_type):
        """Apply log2 transformation using Python"""
        normalization_config = self.config.get('normalization', {})
        qtl_config = normalization_config.get(qtl_type, {})
        pseudocount = qtl_config.get('log2_pseudocount', 1)
        remove_zeros = qtl_config.get('remove_zeros', True)
        
        if remove_zeros:
            # Replace zeros with NaN and remove all-zero features
            original_count = raw_data.shape[0]
            raw_data = raw_data.replace(0, np.nan)
            raw_data = raw_data.dropna(how='all')
            zeros_removed = original_count - raw_data.shape[0]
            if zeros_removed > 0:
                logger.info(f"🔧 Removed {zeros_removed} features with all zeros")
        
        normalized_df = np.log2(raw_data + pseudocount)
        logger.info(f"✅ Applied log2 transformation (pseudocount={pseudocount})")
        return normalized_df
    
    def _apply_quantile_normalization_python(self, raw_data, qtl_type):
        """Apply quantile normalization using Python"""
        try:
            from sklearn.preprocessing import quantile_transform
            
            # Handle missing values
            raw_data_filled = raw_data.fillna(raw_data.mean())
            
            normalized_array = quantile_transform(raw_data_filled.T, n_quantiles=min(1000, raw_data_filled.shape[0]))
            normalized_df = pd.DataFrame(normalized_array.T, index=raw_data.index, columns=raw_data.columns)
            
            logger.info("✅ Quantile normalization completed")
            return normalized_df
        except ImportError:
            logger.error("scikit-learn not available for quantile normalization")
            raise
        except Exception as e:
            logger.error(f"Quantile normalization failed: {e}")
            return raw_data
    
    def _apply_zscore_normalization_python(self, raw_data, qtl_type):
        """Apply z-score normalization per feature using Python"""
        try:
            # Handle missing values
            raw_data_filled = raw_data.fillna(raw_data.mean())
            
            normalized_df = (raw_data_filled - raw_data_filled.mean(axis=1).values.reshape(-1, 1)) 
            normalized_df = normalized_df / raw_data_filled.std(axis=1).values.reshape(-1, 1)
            
            # Handle constant features (std=0)
            constant_mask = raw_data_filled.std(axis=1) == 0
            if constant_mask.any():
                normalized_df.loc[constant_mask] = 0
                logger.warning(f"⚠️ Found {constant_mask.sum()} constant features, setting z-score to 0")
            
            logger.info("✅ Z-score normalization completed")
            return normalized_df
        except Exception as e:
            logger.error(f"Z-score normalization failed: {e}")
            return raw_data
    
    def _apply_arcsinh_normalization_python(self, raw_data, qtl_type):
        """Apply arcsinh transformation using Python"""
        normalization_config = self.config.get('normalization', {})
        qtl_config = normalization_config.get(qtl_type, {})
        cofactor = qtl_config.get('arcsinh_cofactor', 1)
        
        try:
            normalized_df = np.arcsinh(raw_data / cofactor)
            logger.info(f"✅ Arcsinh transformation completed (cofactor={cofactor})")
            return normalized_df
        except Exception as e:
            logger.error(f"Arcsinh normalization failed: {e}")
            return raw_data
    
    def _apply_tpm_normalization_python(self, raw_data, qtl_type):
        """Apply TPM-like normalization using Python"""
        try:
            # Simplified TPM calculation (without gene lengths)
            rpm_df = raw_data.div(raw_data.sum(axis=0)) * 1e6
            logger.info("✅ TPM-like normalization completed")
            return rpm_df
        except Exception as e:
            logger.error(f"TPM normalization failed: {e}")
            return raw_data
    
    def _run_batch_correction_pipeline(self, normalized_data, qtl_type, method):
        """Run batch correction pipeline - ALWAYS save to batch_corrected directory"""
        # Create batch_corrected directory
        corrected_dir = os.path.join(self.results_dir, "batch_corrected")
        os.makedirs(corrected_dir, exist_ok=True)
        
        # Always prepare the corrected file path
        corrected_file = os.path.join(
            corrected_dir, 
            f"{qtl_type}.normalised.{method}.corrected.tsv"
        )
        
        if not BATCH_CORRECTION_AVAILABLE:
            logger.warning("⏭️ Batch correction module not available, saving normalized data as corrected")
            # Save normalized data as corrected (for consistency)
            normalized_data.to_csv(corrected_file, sep='\t')
            return {
                'batch_correction_skipped': True, 
                'reason': 'Module not available',
                'corrected_file': corrected_file,
                'batch_correction_applied': False
            }
        
        try:
            # Check if batch correction is enabled for this QTL type
            batch_config = self.config.get('batch_correction', {})
            enabled = batch_config.get('enabled', {}).get(qtl_type, True)
            
            if not enabled:
                logger.info(f"⏭️ Batch correction disabled for {qtl_type}, saving normalized data as corrected")
                # Save normalized data as corrected (for consistency)
                normalized_data.to_csv(corrected_file, sep='\t')
                return {
                    'batch_correction_skipped': True, 
                    'reason': 'Disabled in config',
                    'corrected_file': corrected_file,
                    'batch_correction_applied': False
                }
            
            # Skip batch correction for splicing data but STILL save the file
            if qtl_type == 'sqtl' and not batch_config.get('enabled', {}).get('sqtl', False):
                logger.info("⏭️ Skipping batch correction for sQTL data but saving normalized file")
                normalized_data.to_csv(corrected_file, sep='\t')
                return {
                    'batch_correction_skipped': True, 
                    'reason': 'sQTL typically does not require batch correction',
                    'corrected_file': corrected_file,
                    'batch_correction_applied': False
                }
            
            logger.info(f"🔄 Running batch correction for {qtl_type}...")
            
            # Run batch correction with enhanced configuration
            corrected_data, correction_info = self._run_enhanced_batch_correction(
                normalized_data=normalized_data,
                qtl_type=qtl_type,
                config=self.config
            )
            
            # Save corrected data
            if correction_info.get('batch_correction_applied', False):
                corrected_data.to_csv(corrected_file, sep='\t')
                correction_info['corrected_file'] = corrected_file
                logger.info(f"💾 Batch-corrected data saved: {corrected_file}")
            else:
                # If batch correction was attempted but not applied, still save normalized data
                normalized_data.to_csv(corrected_file, sep='\t')
                correction_info['corrected_file'] = corrected_file
                correction_info['batch_correction_applied'] = False
                logger.info(f"💾 Normalized data saved as corrected (no batch effects): {corrected_file}")
            
            return correction_info
            
        except Exception as e:
            logger.error(f"❌ Batch correction failed, saving normalized data as corrected: {e}")
            # On error, still save normalized data to maintain file structure
            normalized_data.to_csv(corrected_file, sep='\t')
            return {
                'batch_correction_failed': True, 
                'error': str(e),
                'corrected_file': corrected_file,
                'batch_correction_applied': False
            }
    
    def _run_enhanced_batch_correction(self, normalized_data, qtl_type, config):
        """
        Enhanced batch correction with new configuration structure
        Uses categorical and linear covariates from config
        """
        try:
            # Get batch correction configuration
            batch_config = config.get('batch_correction', {})
            exp_covariates = batch_config.get('exp_covariates', {})
            covariate_design_file = batch_config.get('exp_covariate_design')
            
            if not covariate_design_file or not os.path.exists(covariate_design_file):
                logger.warning(f"⏭️ No covariate design file found: {covariate_design_file}")
                return normalized_data, {
                    'batch_correction_skipped': True,
                    'reason': 'No covariate design file found',
                    'batch_correction_applied': False
                }
            
            # Load covariate design matrix with format detection
            logger.info(f"📁 Loading covariate design from: {covariate_design_file}")
            
            # First, detect the file format by reading a few lines
            with open(covariate_design_file, 'r') as f:
                first_line = f.readline().strip()
            
            # Check if first column indicates samples × covariates format (CORRECT format)
            first_column = first_line.split('\t')[0].lower()
            sample_indicators = ['sample', 'id', 'iid', 'individual', 'individual_id', 'subject', 'subject_id']
            
            if first_column in sample_indicators:
                first_columnN = first_line.split('\t')[0]
                logger.info(f"✅ Detected CORRECT samples × covariates format (first column: '{first_columnN}') - keeping as is")
                # Read as samples × covariates - DO NOT TRANSPOSE, this is correct
                covariate_design = pd.read_csv(covariate_design_file, sep='\t', index_col=0)
                logger.info(f"✅ Loaded batch covariates: {covariate_design.shape[0]} samples, {covariate_design.shape[1]} covariates")
            else:
                # This would be covariates × samples format (less common)
                logger.info(f"Detected covariates × samples format (first column: '{first_line.split('\t')[0]}'), transposing...")
                covariate_design = pd.read_csv(covariate_design_file, sep='\t', index_col=0)
                covariate_design = covariate_design.T  # Transpose to samples × covariates
                logger.info(f"Loaded batch covariates: {covariate_design.shape[0]} samples, {covariate_design.shape[1]} covariates")
            
            # Get sample IDs from normalized data
            sample_ids = normalized_data.columns.tolist()
            
            # Subset covariate design to available samples - samples are in INDEX
            available_samples = [s for s in sample_ids if s in covariate_design.index]
            if len(available_samples) != len(sample_ids):
                logger.warning(f"⚠️ Only {len(available_samples)}/{len(sample_ids)} samples found in covariate design")
            
            if len(available_samples) == 0:
                logger.error("❌ No overlapping samples found for batch correction")
                logger.error(f"💡 Expression samples: {sample_ids[:5]}")
                logger.error(f"💡 Covariate samples: {covariate_design.index.tolist()[:5]}")
                return normalized_data, {
                    'batch_correction_skipped': True,
                    'reason': 'No overlapping samples found',
                    'batch_correction_applied': False
                }
            
            # CRITICAL: Ensure samples are in the SAME ORDER for both expression and covariates
            available_samples_sorted = sorted(available_samples)
            
            covariate_design = covariate_design.loc[available_samples_sorted]
            normalized_data_subset = normalized_data[available_samples_sorted]
            
            logger.info(f"✅ Using {len(available_samples_sorted)} samples for batch correction")
            
            # Verify sample alignment
            expr_samples = normalized_data_subset.columns.tolist()
            covar_samples = covariate_design.index.tolist()
            
            if expr_samples != covar_samples:
                logger.error("❌ CRITICAL: Sample order mismatch between expression data and covariates!")
                logger.error(f"💡 First 3 expression samples: {expr_samples[:3]}")
                logger.error(f"💡 First 3 covariate samples: {covar_samples[:3]}")
                raise ValueError("Sample order mismatch between expression data and covariates")
            else:
                logger.info("✅ Sample order verified: expression data and covariates are aligned")
            
            # Get categorical and linear covariates from config
            categorical_covariates = exp_covariates.get('categorical', [])
            linear_covariates = exp_covariates.get('linear', [])
            
            logger.info(f"🔧 Categorical covariates from config: {categorical_covariates}")
            logger.info(f"🔧 Linear covariates from config: {linear_covariates}")
            
            # Check if all specified covariates exist in the design matrix
            all_covariates = categorical_covariates + linear_covariates
            missing_covariates = [cov for cov in all_covariates if cov not in covariate_design.columns]
            
            if missing_covariates:
                logger.warning(f"⚠️ Missing covariates in design matrix: {missing_covariates}")
                logger.info(f"💡 Available covariates: {covariate_design.columns.tolist()}")
            
            # Remove missing covariates from lists
            categorical_covariates = [cov for cov in categorical_covariates if cov in covariate_design.columns]
            linear_covariates = [cov for cov in linear_covariates if cov in covariate_design.columns]
            
            logger.info(f"🔧 Final categorical covariates: {categorical_covariates}")
            logger.info(f"🔧 Final linear covariates: {linear_covariates}")
            
            # Prepare batches (categorical covariates)
            batches = []
            for covariate in categorical_covariates:
                if covariate in covariate_design.columns:
                    batch_data = covariate_design[covariate].values
                    batches.append(batch_data)
                    logger.info(f"  - Categorical: {covariate}: {len(np.unique(batch_data))} levels")
            
            # Prepare covariates (linear covariates)
            covariates = None
            if linear_covariates:
                covariates = covariate_design[linear_covariates].copy()
                # Convert to numeric, coerce errors to NaN
                for col in linear_covariates:
                    covariates[col] = pd.to_numeric(covariates[col], errors='coerce')
                logger.info(f"  - Linear covariates: {linear_covariates}")
            
            # Skip if no batches or covariates found
            if not batches and covariates is None:
                logger.info(f"⏭️ No batch/covariate effects found for {qtl_type}, skipping batch correction")
                return normalized_data, {
                    'batch_correction_skipped': True,
                    'reason': 'No batch/covariate effects found',
                    'batch_correction_applied': False
                }
            
            # Perform batch correction using custom function
            from scripts.utils.batch_correction import remove_batch_effect_custom
            
            corrected_data = remove_batch_effect_custom(
                x=normalized_data_subset,
                batches=batches,
                covariates=covariates,
                design=None  # Using default intercept design
            )
            
            # Prepare correction info
            correction_info = {
                'covariate_design_file': covariate_design_file,
                'samples_used': available_samples_sorted,
                'categorical_covariates': categorical_covariates,
                'linear_covariates': linear_covariates,
                'input_shape': normalized_data.shape,
                'output_shape': corrected_data.shape,
                'correction_method': 'linear_regression',
                'qtl_type': qtl_type,
                'batch_correction_applied': True
            }
            
            logger.info(f"✅ Enhanced batch correction completed for {qtl_type}")
            return corrected_data, correction_info
            
        except Exception as e:
            logger.error(f"❌ Enhanced batch correction failed for {qtl_type}: {e}")
            raise
    
    def _run_enhanced_comparisons(self, qtl_type, raw_data, normalized_data, batch_correction_info, method):
        """Run enhanced comparisons including batch correction"""
        comparison_results = {}
        
        # Comparison 1: Raw vs Normalized (original functionality)
        logger.info(f"📊 Comparison 1: Raw vs Normalized ({method})")
        comparison1 = self.generate_comprehensive_comparison(
            qtl_type, raw_data, normalized_data, method
        )
        comparison_results['raw_vs_normalized'] = comparison1
        
        # Comparison 2: Normalized vs "Corrected" (ALWAYS generated now)
        # Even if no batch correction was applied, we still have a file in batch_corrected
        corrected_file = batch_correction_info.get('corrected_file')
        if corrected_file and os.path.exists(corrected_file):
            logger.info(f"📊 Comparison 2: Normalized vs Batch-Corrected ({method})")
            
            corrected_data = pd.read_csv(corrected_file, sep='\t', index_col=0)
            
            # Create special directory for batch correction comparisons
            batch_comp_dir = os.path.join(self.comparison_dir, "batch_correction")
            os.makedirs(batch_comp_dir, exist_ok=True)
            
            # Determine comparison type based on whether batch correction was actually applied
            if batch_correction_info.get('batch_correction_applied', False):
                comparison_type = f"{method}_corrected"
                comparison_name = "Batch-Corrected"
            else:
                comparison_type = f"{method}_normalized_as_corrected"
                comparison_name = "Normalized (as Corrected)"
            
            comparison2 = self.generate_comprehensive_comparison(
                f"{qtl_type}_batch_corrected", 
                normalized_data, 
                corrected_data, 
                comparison_type
            )
            comparison_results['normalized_vs_corrected'] = comparison2
            comparison_results['correction_status'] = batch_correction_info.get('batch_correction_applied', False)
            comparison_results['correction_reason'] = batch_correction_info.get('reason', 'Unknown')
        else:
            logger.warning("⚠️ Corrected data file not found, skipping batch correction comparison")
            comparison_results['normalized_vs_corrected'] = {'skipped': True, 'reason': 'Corrected data file not found'}
        
        return comparison_results
    
    def _save_enhanced_pipeline_summary(self, qtl_type, method, batch_correction_info, comparison_results):
        """Save enhanced pipeline execution summary"""
        summary = {
            'qtl_type': qtl_type,
            'normalization_method': method,
            'batch_correction_info': batch_correction_info,
            'comparisons_performed': list(comparison_results.keys()),
            'timestamp': pd.Timestamp.now().isoformat(),
            'pipeline_version': 'enhanced_1.0'
        }
        
        summary_file = os.path.join(self.comparison_dir, f"{qtl_type}_enhanced_pipeline_summary.yaml")
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"💾 Enhanced pipeline summary saved: {summary_file}")
    
    def _sample_data_for_performance(self, raw_data, normalized_data):
        """Sample data for performance optimization"""
        if raw_data.shape[0] > self.max_features or raw_data.shape[1] > self.max_samples:
            logger.info(f"🔧 Sampling data for performance: {raw_data.shape} -> max {self.max_features} features, {self.max_samples} samples")
            
            # Sample features
            if raw_data.shape[0] > self.max_features:
                feature_indices = np.random.choice(
                    raw_data.index, 
                    min(self.max_features, raw_data.shape[0]), 
                    replace=False
                )
                raw_data = raw_data.loc[feature_indices]
                normalized_data = normalized_data.loc[feature_indices]
            
            # Sample samples
            if raw_data.shape[1] > self.max_samples:
                sample_indices = np.random.choice(
                    raw_data.columns,
                    min(self.max_samples, raw_data.shape[1]),
                    replace=False
                )
                raw_data = raw_data[sample_indices]
                normalized_data = normalized_data[sample_indices]
        
        return raw_data, normalized_data
    
    def _run_parallel_comparisons(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Run comparison tasks in parallel"""
        results = {'plots_generated': []}
        
        # Define tasks for parallel execution
        tasks = [
            ('distribution', self.create_distribution_comparison, (raw_data, normalized_data, qtl_type, method, output_dir)),
            ('statistical', self.create_statistical_comparison, (raw_data, normalized_data, qtl_type, method, output_dir)),
            ('correlation', self.create_correlation_comparison, (raw_data, normalized_data, qtl_type, method, output_dir)),
        ]
        
        # Sample-based tasks (less memory intensive)
        sample_tasks = [
            ('sample', self.create_sample_comparison, (raw_data, normalized_data, qtl_type, method, output_dir)),
            ('feature', self.create_feature_comparison, (raw_data, normalized_data, qtl_type, method, output_dir)),
        ]
        
        if raw_data.shape[0] * raw_data.shape[1] < 1000000:  # Only if data is not too large
            tasks.append(('pca', self.create_pca_comparison, (raw_data, normalized_data, qtl_type, method, output_dir)))
        
        # Execute tasks in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {
                executor.submit(task[1], *task[2]): task[0] for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        results[f'{task_name}_comparison'] = result
                        results['plots_generated'].append(task_name + '_comparison')
                except Exception as e:
                    logger.warning(f"Parallel task {task_name} failed: {e}")
        
        # Run sample tasks sequentially to avoid memory issues
        for task_name, task_func, task_args in sample_tasks:
            try:
                result = task_func(*task_args)
                if result:
                    results[f'{task_name}_comparison'] = result
                    results['plots_generated'].append(task_name + '_comparison')
            except Exception as e:
                logger.warning(f"Sample task {task_name} failed: {e}")
        
        return results
    
    def _run_sequential_comparisons(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Run comparison tasks sequentially"""
        results = {'plots_generated': []}
        
        comparison_tasks = [
            (self.create_distribution_comparison, 'distribution_comparison'),
            (self.create_sample_comparison, 'sample_comparison'),
            (self.create_feature_comparison, 'feature_comparison'),
            (self.create_statistical_comparison, 'statistical_comparison'),
            (self.create_correlation_comparison, 'correlation_comparison'),
        ]
        
        # Only run PCA for smaller datasets
        if raw_data.shape[0] * raw_data.shape[1] < 1000000:
            comparison_tasks.append((self.create_pca_comparison, 'pca_comparison'))
        
        for task_func, task_name in comparison_tasks:
            try:
                result = task_func(raw_data, normalized_data, qtl_type, method, output_dir)
                if result:
                    results[task_name] = result
                    results['plots_generated'].append(task_name)
            except Exception as e:
                logger.warning(f"Task {task_name} failed: {e}")
        
        return results
    
    def create_distribution_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create optimized distribution comparison plots"""
        try:
            # Use cached sampled data if available
            cache_key = f"dist_{qtl_type}"
            if cache_key not in self._cache:
                self._cache[cache_key] = self._prepare_distribution_data(raw_data, normalized_data)
            
            sampled_raw, sampled_norm = self._cache[cache_key]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{qtl_type.upper()} Normalization: {method}\nDistribution Comparison', fontsize=16, fontweight='bold')
            
            # Plot 1: Overall distribution density
            self._plot_overall_density(sampled_raw, sampled_norm, axes[0, 0], qtl_type)
            
            # Plot 2: Feature-wise distributions
            self._plot_feature_distributions(raw_data, normalized_data, axes[0, 1], qtl_type)
            
            # Plot 3: Box plot comparison
            self._plot_box_comparison(raw_data, normalized_data, axes[1, 0], qtl_type)
            
            # Plot 4: QQ plot comparison
            self._plot_qq_comparison(sampled_raw, sampled_norm, axes[1, 1], qtl_type)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_distribution_comparison.png")
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')  # Reduced DPI for performance
            plt.close()
            
            # Create interactive version if plotly available
            interactive_file = None
            if PLOTLY_AVAILABLE:
                interactive_file = self._create_interactive_distribution(sampled_raw, sampled_norm, qtl_type, method, output_dir)
            
            return {'static': plot_file, 'interactive': interactive_file}
            
        except Exception as e:
            logger.warning(f"Could not create distribution comparison: {e}")
            return None
    
    def _prepare_distribution_data(self, raw_data, normalized_data, max_points=10000):
        """Prepare sampled data for distribution plots"""
        # Flatten and sample for performance
        raw_flat = raw_data.values.flatten()
        norm_flat = normalized_data.values.flatten()
        
        raw_flat = raw_flat[~np.isnan(raw_flat)]
        norm_flat = norm_flat[~np.isnan(norm_flat)]
        
        # Sample for performance
        if len(raw_flat) > max_points:
            raw_flat = np.random.choice(raw_flat, max_points, replace=False)
        if len(norm_flat) > max_points:
            norm_flat = np.random.choice(norm_flat, max_points, replace=False)
        
        return raw_flat, norm_flat
    
    def _plot_overall_density(self, raw_flat, norm_flat, ax, qtl_type):
        """Plot optimized overall density comparison"""
        try:
            # Remove outliers for better visualization
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
    
    def _plot_feature_distributions(self, raw_data, normalized_data, ax, qtl_type):
        """Plot distributions for sample features with optimization"""
        try:
            # Sample a few features
            n_features = min(5, raw_data.shape[0])
            if n_features == 0:
                return
                
            feature_indices = np.random.choice(raw_data.index, n_features, replace=False)
            
            for i, feature in enumerate(feature_indices):
                if feature in raw_data.index and feature in normalized_data.index:
                    raw_values = raw_data.loc[feature].dropna()
                    norm_values = normalized_data.loc[feature].dropna()
                    
                    if len(raw_values) > 10 and len(norm_values) > 10:
                        # Sample for performance
                        if len(raw_values) > 1000:
                            raw_values = np.random.choice(raw_values, 1000, replace=False)
                        if len(norm_values) > 1000:
                            norm_values = np.random.choice(norm_values, 1000, replace=False)
                        
                        # Normalize for comparison
                        raw_norm = (raw_values - raw_values.mean()) / raw_values.std() if raw_values.std() > 0 else raw_values
                        norm_norm = (norm_values - norm_values.mean()) / norm_values.std() if norm_values.std() > 0 else norm_values
                        
                        ax.hist(raw_norm, bins=20, alpha=0.5, color=self.colors[0], 
                               density=True, label=f'Raw {feature}' if i == 0 else "", histtype='stepfilled')
                        ax.hist(norm_norm, bins=20, alpha=0.5, color=self.colors[1], 
                               density=True, label=f'Norm {feature}' if i == 0 else "", histtype='stepfilled')
            
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Density')
            ax.set_title('Feature Distributions (Sample)')
            if n_features > 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_box_comparison(self, raw_data, normalized_data, ax, qtl_type):
        """Optimized box plot comparison"""
        try:
            # Sample features for box plot
            n_sample = min(8, raw_data.shape[0])
            if n_sample > 0:
                sample_features = np.random.choice(raw_data.index, n_sample, replace=False)
                
                # Prepare data for box plot
                box_data = []
                labels = []
                
                for feature in sample_features:
                    if feature in raw_data.index:
                        raw_values = raw_data.loc[feature].dropna()
                        if len(raw_values) > 0:
                            # Sample for performance
                            if len(raw_values) > 500:
                                raw_values = np.random.choice(raw_values, 500, replace=False)
                            box_data.append(raw_values)
                            labels.append(f'R_{feature[:6]}')
                    
                    if feature in normalized_data.index:
                        norm_values = normalized_data.loc[feature].dropna()
                        if len(norm_values) > 0:
                            if len(norm_values) > 500:
                                norm_values = np.random.choice(norm_values, 500, replace=False)
                            box_data.append(norm_values)
                            labels.append(f'N_{feature[:6]}')
                
                if len(box_data) > 0:
                    ax.boxplot(box_data, labels=labels, showfliers=False)  # Hide outliers for clarity
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel('Value')
                    ax.set_title('Box Plot Comparison\n(R=Raw, N=Normalized)')
                    ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_qq_comparison(self, raw_flat, norm_flat, ax, qtl_type):
        """Optimized QQ plot comparison"""
        try:
            if len(raw_flat) > 100 and len(norm_flat) > 100:
                # Sample for performance
                if len(raw_flat) > 1000:
                    raw_flat = np.random.choice(raw_flat, 1000, replace=False)
                if len(norm_flat) > 1000:
                    norm_flat = np.random.choice(norm_flat, 1000, replace=False)
                
                # Raw data QQ plot
                raw_sorted = np.sort(raw_flat)
                raw_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(raw_sorted)))
                
                # Normalized data QQ plot
                norm_sorted = np.sort(norm_flat)
                norm_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(norm_sorted)))
                
                ax.scatter(raw_theoretical, raw_sorted, alpha=0.6, color=self.colors[0], 
                          s=10, label='Raw')  # Smaller points
                ax.scatter(norm_theoretical, norm_sorted, alpha=0.6, color=self.colors[1], 
                          s=10, label='Normalized')
                
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
    
    def _create_interactive_distribution(self, raw_flat, norm_flat, qtl_type, method, output_dir):
        """Create interactive distribution plot"""
        if not PLOTLY_AVAILABLE:
            return None
            
        try:
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
                height=500  # Smaller for performance
            )
            
            plot_file = os.path.join(output_dir, f"{qtl_type}_interactive_distribution.html")
            fig.write_html(plot_file, include_plotlyjs='cdn')  # Use CDN for smaller files
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create interactive distribution: {e}")
            return None

    def create_sample_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create optimized sample-wise comparison plots"""
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
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create sample comparison: {e}")
            return None
    
    def _plot_sample_means(self, raw_data, normalized_data, ax):
        """Plot sample means comparison with sampling"""
        try:
            raw_means = raw_data.mean(axis=0)
            norm_means = normalized_data.mean(axis=0)
            
            # Sample for performance
            if len(raw_means) > 100:
                indices = np.random.choice(len(raw_means), 100, replace=False)
                raw_means = raw_means.iloc[indices]
                norm_means = norm_means.iloc[indices]
            
            ax.scatter(range(len(raw_means)), raw_means, alpha=0.7, color=self.colors[0], 
                      label='Raw', s=20)
            ax.scatter(range(len(norm_means)), norm_means, alpha=0.7, color=self.colors[1], 
                      label='Normalized', s=20)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Mean Value')
            ax.set_title('Sample Means')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_sample_variances(self, raw_data, normalized_data, ax):
        """Plot sample variances comparison with sampling"""
        try:
            raw_vars = raw_data.var(axis=0)
            norm_vars = normalized_data.var(axis=0)
            
            # Sample for performance
            if len(raw_vars) > 100:
                indices = np.random.choice(len(raw_vars), 100, replace=False)
                raw_vars = raw_vars.iloc[indices]
                norm_vars = norm_vars.iloc[indices]
            
            ax.scatter(range(len(raw_vars)), raw_vars, alpha=0.7, color=self.colors[0], 
                      label='Raw', s=20)
            ax.scatter(range(len(norm_vars)), norm_vars, alpha=0.7, color=self.colors[1], 
                      label='Normalized', s=20)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Variance')
            ax.set_title('Sample Variances')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_sample_missingness(self, raw_data, normalized_data, ax):
        """Plot sample missingness comparison with sampling"""
        try:
            raw_missing = raw_data.isna().sum(axis=0) / raw_data.shape[0] * 100
            norm_missing = normalized_data.isna().sum(axis=0) / normalized_data.shape[0] * 100
            
            # Sample for performance
            if len(raw_missing) > 50:
                indices = np.random.choice(len(raw_missing), 50, replace=False)
                raw_missing = raw_missing.iloc[indices]
                norm_missing = norm_missing.iloc[indices]
            
            x_pos = np.arange(len(raw_missing))
            ax.bar(x_pos - 0.2, raw_missing, width=0.4, 
                   color=self.colors[0], alpha=0.7, label='Raw')
            ax.bar(x_pos + 0.2, norm_missing, width=0.4, 
                   color=self.colors[1], alpha=0.7, label='Normalized')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Missingness (%)')
            ax.set_title('Sample Missingness')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_sample_correlation(self, raw_data, normalized_data, ax):
        """Plot sample correlation comparison with optimization"""
        try:
            # Calculate correlation on a subset for performance
            n_samples = min(15, raw_data.shape[1])
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
        """Create optimized feature-wise comparison plots"""
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
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create feature comparison: {e}")
            return None
    
    def _plot_feature_means(self, raw_data, normalized_data, ax):
        """Plot feature means comparison with sampling"""
        try:
            raw_means = raw_data.mean(axis=1)
            norm_means = normalized_data.mean(axis=1)
            
            # Sample for performance
            if len(raw_means) > 1000:
                indices = np.random.choice(len(raw_means), 1000, replace=False)
                raw_means = raw_means.iloc[indices]
                norm_means = norm_means.iloc[indices]
            
            ax.scatter(raw_means, norm_means, alpha=0.6, color=self.colors[2], s=10)
            min_val = min(raw_means.min(), norm_means.min())
            max_val = max(raw_means.max(), norm_means.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Raw Mean')
            ax.set_ylabel('Normalized Mean')
            ax.set_title('Feature Means: Raw vs Normalized')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_feature_variances(self, raw_data, normalized_data, ax):
        """Plot feature variances comparison with sampling"""
        try:
            raw_vars = raw_data.var(axis=1)
            norm_vars = normalized_data.var(axis=1)
            
            # Sample for performance
            if len(raw_vars) > 1000:
                indices = np.random.choice(len(raw_vars), 1000, replace=False)
                raw_vars = raw_vars.iloc[indices]
                norm_vars = norm_vars.iloc[indices]
            
            ax.scatter(raw_vars, norm_vars, alpha=0.6, color=self.colors[2], s=10)
            min_val = min(raw_vars.min(), norm_vars.min())
            max_val = max(raw_vars.max(), norm_vars.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Raw Variance')
            ax.set_ylabel('Normalized Variance')
            ax.set_title('Feature Variances: Raw vs Normalized')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_mean_variance(self, raw_data, normalized_data, ax):
        """Plot mean-variance relationship with sampling"""
        try:
            raw_means = raw_data.mean(axis=1)
            raw_vars = raw_data.var(axis=1)
            norm_means = normalized_data.mean(axis=1)
            norm_vars = normalized_data.var(axis=1)
            
            # Sample for performance
            if len(raw_means) > 1000:
                indices = np.random.choice(len(raw_means), 1000, replace=False)
                raw_means = raw_means.iloc[indices]
                raw_vars = raw_vars.iloc[indices]
                norm_means = norm_means.iloc[indices]
                norm_vars = norm_vars.iloc[indices]
            
            ax.scatter(raw_means, raw_vars, alpha=0.6, color=self.colors[0], 
                      s=10, label='Raw')
            ax.scatter(norm_means, norm_vars, alpha=0.6, color=self.colors[1], 
                      s=10, label='Normalized')
            
            ax.set_xlabel('Mean')
            ax.set_ylabel('Variance')
            ax.set_title('Mean-Variance Relationship')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_feature_detection(self, raw_data, normalized_data, ax):
        """Plot feature detection rates with sampling"""
        try:
            # Calculate detection rate (non-zero/non-missing)
            raw_detection = (raw_data > 0).sum(axis=1) / raw_data.shape[1] * 100
            norm_detection = (normalized_data > normalized_data.quantile(0.01)).sum(axis=1) / normalized_data.shape[1] * 100
            
            # Sample for performance
            if len(raw_detection) > 1000:
                raw_detection = np.random.choice(raw_detection, 1000, replace=False)
                norm_detection = np.random.choice(norm_detection, 1000, replace=False)
            
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
        """Create optimized statistical summary comparison"""
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
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            # Save statistical summary
            stats_file = os.path.join(output_dir, f"{qtl_type}_statistical_summary.txt")
            stats_comparison.to_csv(stats_file, sep='\t')
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create statistical comparison: {e}")
            return None
    
    def _calculate_statistical_summary(self, data, label):
        """Calculate comprehensive statistical summary with optimization"""
        try:
            # Sample data for performance
            if data.size > 1000000:
                sampled_data = data.sample(n=1000000, axis=0).values.flatten()
            else:
                sampled_data = data.values.flatten()
            
            sampled_data = sampled_data[~np.isnan(sampled_data)]
            
            if len(sampled_data) == 0:
                return pd.Series([0] * 10, index=[
                    'Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 
                    'Kurtosis', 'Q1', 'Q3', 'IQR'
                ], name=label)
            
            return pd.Series({
                'Mean': np.mean(sampled_data),
                'Median': np.median(sampled_data),
                'Std': np.std(sampled_data),
                'Min': np.min(sampled_data),
                'Max': np.max(sampled_data),
                'Skewness': stats.skew(sampled_data),
                'Kurtosis': stats.kurtosis(sampled_data),
                'Q1': np.percentile(sampled_data, 25),
                'Q3': np.percentile(sampled_data, 75),
                'IQR': np.percentile(sampled_data, 75) - np.percentile(sampled_data, 25)
            }, name=label)
            
        except Exception as e:
            logger.warning(f"Error calculating statistical summary: {e}")
            return pd.Series()
    
    def _plot_basic_stats(self, stats_comparison, ax):
        """Plot basic statistics comparison"""
        try:
            metrics = ['Mean', 'Median', 'Std', 'Min', 'Max']
            if all(metric in stats_comparison.index for metric in metrics):
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
        """Plot skewness and kurtosis comparison with sampling"""
        try:
            # Sample features for performance
            n_features = min(100, raw_data.shape[0])
            if n_features > 10:
                feature_indices = np.random.choice(raw_data.index, n_features, replace=False)
                
                raw_skew = []
                raw_kurt = []
                norm_skew = []
                norm_kurt = []
                
                for feature in feature_indices:
                    if feature in raw_data.index:
                        raw_values = raw_data.loc[feature].dropna()
                        if len(raw_values) > 10:
                            raw_skew.append(stats.skew(raw_values))
                            raw_kurt.append(stats.kurtosis(raw_values))
                    
                    if feature in normalized_data.index:
                        norm_values = normalized_data.loc[feature].dropna()
                        if len(norm_values) > 10:
                            norm_skew.append(stats.skew(norm_values))
                            norm_kurt.append(stats.kurtosis(norm_values))
                
                if raw_skew and raw_kurt and norm_skew and norm_kurt:
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
        """Plot outlier comparison with sampling"""
        try:
            # Calculate outliers using IQR method on sampled data
            def count_outliers(data):
                if len(data) < 10:
                    return 0
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                if IQR == 0:
                    return 0
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Sample columns for performance
            n_samples = min(50, raw_data.shape[1])
            if n_samples > 5:
                sample_cols = np.random.choice(raw_data.columns, n_samples, replace=False)
                
                raw_outliers = [count_outliers(raw_data[col].dropna()) for col in sample_cols]
                norm_outliers = [count_outliers(normalized_data[col].dropna()) for col in sample_cols]
                
                ax.hist(raw_outliers, bins=20, alpha=0.7, color=self.colors[0], 
                       density=True, label='Raw')
                ax.hist(norm_outliers, bins=20, alpha=0.7, color=self.colors[1], 
                       density=True, label='Normalized')
                
                ax.set_xlabel('Number of Outliers per Sample')
                ax.set_ylabel('Density')
                ax.set_title('Outlier Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha='center')
    
    def _plot_statistical_tests(self, raw_data, normalized_data, ax):
        """Plot statistical test results with sampling"""
        try:
            # Perform normality tests on sample of features
            n_tests = min(50, raw_data.shape[0])
            if n_tests > 10:
                test_features = np.random.choice(raw_data.index, n_tests, replace=False)
                
                raw_normality = []
                norm_normality = []
                
                for feature in test_features:
                    if feature in raw_data.index:
                        raw_values = raw_data.loc[feature].dropna()
                        if len(raw_values) > 10:
                            _, raw_p = stats.normaltest(raw_values)
                            raw_normality.append(-np.log10(raw_p) if raw_p > 0 else 10)
                    
                    if feature in normalized_data.index:
                        norm_values = normalized_data.loc[feature].dropna()
                        if len(norm_values) > 10:
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
        """Create optimized PCA comparison plot"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for PCA - use sampling for large datasets
            if raw_data.shape[1] > 100:
                sample_cols = np.random.choice(raw_data.columns, 100, replace=False)
                raw_clean = raw_data[sample_cols].dropna(axis=1, how='all').T
                norm_clean = normalized_data[sample_cols].dropna(axis=1, how='all').T
            else:
                raw_clean = raw_data.dropna(axis=1, how='all').T
                norm_clean = normalized_data.dropna(axis=1, how='all').T
            
            if raw_clean.shape[0] > 5 and norm_clean.shape[0] > 5:
                # Sample features for PCA if too many
                if raw_clean.shape[1] > 1000:
                    feature_sample = np.random.choice(raw_clean.columns, 1000, replace=False)
                    raw_clean = raw_clean[feature_sample]
                    norm_clean = norm_clean[feature_sample]
                
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
                                     c=range(len(pca_raw_result)), cmap='viridis', s=30)
                ax1.set_title(f'Raw Data PCA\nExplained Variance: {pca_raw.explained_variance_ratio_[:3].sum():.2f}')
                ax1.set_xlabel(f'PC1 ({pca_raw.explained_variance_ratio_[0]:.2f})')
                ax1.set_ylabel(f'PC2 ({pca_raw.explained_variance_ratio_[1]:.2f})')
                ax1.set_zlabel(f'PC3 ({pca_raw.explained_variance_ratio_[2]:.2f})')
                
                # Plot normalized PCA
                scatter2 = ax2.scatter(pca_norm_result[:, 0], pca_norm_result[:, 1], pca_norm_result[:, 2],
                                     c=range(len(pca_norm_result)), cmap='viridis', s=30)
                ax2.set_title(f'Normalized Data PCA\nExplained Variance: {pca_norm.explained_variance_ratio_[:3].sum():.2f}')
                ax2.set_xlabel(f'PC1 ({pca_norm.explained_variance_ratio_[0]:.2f})')
                ax2.set_ylabel(f'PC2 ({pca_norm.explained_variance_ratio_[1]:.2f})')
                ax2.set_zlabel(f'PC3 ({pca_norm.explained_variance_ratio_[2]:.2f})')
                
                plt.tight_layout()
                plot_file = os.path.join(output_dir, f"{qtl_type}_pca_comparison.png")
                plt.savefig(plot_file, dpi=200, bbox_inches='tight')
                plt.close()
                
                # Create interactive 3D plot if plotly available
                interactive_file = None
                if PLOTLY_AVAILABLE:
                    interactive_file = self._create_interactive_pca(pca_raw_result, pca_norm_result, 
                                                                  pca_raw, pca_norm, qtl_type, method, output_dir)
                
                return {'static': plot_file, 'interactive': interactive_file}
            
        except Exception as e:
            logger.warning(f"Could not create PCA comparison: {e}")
            return None
    
    def _create_interactive_pca(self, pca_raw, pca_norm, pca_raw_obj, pca_norm_obj, qtl_type, method, output_dir):
        """Create interactive 3D PCA plot"""
        if not PLOTLY_AVAILABLE:
            return None
            
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
                        size=6,
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
                        size=6,
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
                height=500,
                showlegend=False
            )
            
            plot_file = os.path.join(output_dir, f"{qtl_type}_interactive_pca.html")
            fig.write_html(plot_file, include_plotlyjs='cdn')
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create interactive PCA: {e}")
            return None
    
    def create_correlation_comparison(self, raw_data, normalized_data, qtl_type, method, output_dir):
        """Create optimized correlation structure comparison"""
        try:
            # Sample features for correlation analysis
            n_features = min(30, raw_data.shape[0])
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
            axes[1, 1].hist(raw_sample.values.flatten(), bins=30, alpha=0.7, 
                           color=self.colors[0], density=True, label='Raw')
            axes[1, 1].hist(norm_sample.values.flatten(), bins=30, alpha=0.7, 
                           color=self.colors[1], density=True, label='Normalized')
            axes[1, 1].set_xlabel('Correlation Coefficient')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Correlation Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{qtl_type}_correlation_comparison.png")
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            # Create interactive correlation plot if plotly available
            interactive_file = None
            if PLOTLY_AVAILABLE:
                interactive_file = self._create_interactive_correlation(raw_sample, norm_sample, qtl_type, method, output_dir)
            
            return {'static': plot_file, 'interactive': interactive_file}
            
        except Exception as e:
            logger.warning(f"Could not create correlation comparison: {e}")
            return None
    
    def _create_interactive_correlation(self, raw_corr, norm_corr, qtl_type, method, output_dir):
        """Create interactive correlation heatmap"""
        if not PLOTLY_AVAILABLE:
            return None
            
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
                height=500
            )
            
            plot_file = os.path.join(output_dir, f"{qtl_type}_interactive_correlation.html")
            fig.write_html(plot_file, include_plotlyjs='cdn')
            
            return plot_file
            
        except Exception as e:
            logger.warning(f"Could not create interactive correlation: {e}")
            return None
    
    def generate_comparison_html_report(self, comparison_results, qtl_type, output_dir):
        """Generate comprehensive HTML report for normalization comparison"""
        try:
            html_content = self._generate_html_report_content(comparison_results, qtl_type, output_dir)
            report_file = os.path.join(output_dir, f"{qtl_type}_normalization_report.html")
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"💾 Normalization comparison HTML report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Error generating HTML report: {e}")
            return None
    
    def _generate_html_report_content(self, comparison_results, qtl_type, output_dir):
        """Generate HTML content for normalization comparison report"""
        plots_generated = comparison_results.get('plots_generated', [])
        
        # List available files
        available_files = []
        if os.path.exists(output_dir):
            available_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.html', '.txt'))]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Normalization Comparison Report - {qtl_type.upper()}</title>
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
                .performance-info {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔬 Enhanced Normalization Comparison Report</h1>
                    <p><strong>{qtl_type.upper()} Analysis</strong></p>
                    <p>Method: {comparison_results.get('normalization_method', 'Unknown')} | Plots Generated: {len(plots_generated)}</p>
                    <p>Data: {comparison_results.get('raw_data_shape', 'Unknown')} (raw) → {comparison_results.get('normalized_data_shape', 'Unknown')} (normalized)</p>
                </div>
                
                <div class="section">
                    <h2>📊 Enhanced Summary</h2>
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
                            <div class="stat-number">{len(plots_generated)}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Data Points</div>
                            <div class="stat-number" style="font-size: 1.2em;">{comparison_results.get('raw_data_shape', 'N/A')[0]} × {comparison_results.get('raw_data_shape', 'N/A')[1]}</div>
                        </div>
                    </div>
                    
                    <div class="performance-info">
                        <strong>⚡ Performance Optimizations Applied:</strong>
                        <ul>
                            <li>Parallel processing: {self.parallel_processing}</li>
                            <li>Max samples: {self.max_samples}</li>
                            <li>Max features: {self.max_features}</li>
                            <li>Data sampling for large datasets</li>
                            <li>Optimized plotting parameters</li>
                        </ul>
                    </div>
                    
                    <div class="summary-box">
                        <strong>🎯 Analysis Overview:</strong>
                        <p>This enhanced report compares the raw input data with the normalized data used for QTL analysis. 
                        The comprehensive visualizations help understand the impact of normalization on data structure, 
                        distribution, and quality with performance optimizations for large datasets.</p>
                    </div>
                </div>
        """
        
        # Add plot sections dynamically
        plot_sections = {
            'distribution_comparison': ('📈 Distribution Comparison', 'Overall distribution, feature distributions, box plots, and QQ plots'),
            'sample_comparison': ('👥 Sample-wise Comparison', 'Sample means, variances, missingness, and correlation structure'),
            'feature_comparison': ('🔍 Feature-wise Comparison', 'Feature means, variances, mean-variance relationship, and detection rates'),
            'statistical_comparison': ('📋 Statistical Summary', 'Statistical measures, normality tests, outlier analysis, and comprehensive statistics'),
            'pca_comparison': ('🎯 PCA Comparison', '3D PCA plots showing sample clustering in raw vs normalized space'),
            'correlation_comparison': ('🔄 Correlation Structure', 'Correlation heatmaps and distribution comparisons')
        }
        
        for plot_type in plots_generated:
            if plot_type in plot_sections:
                title, description = plot_sections[plot_type]
                html_content += f"""
                <div class="section">
                    <h2>{title}</h2>
                    <div class="plot-grid">
                """
                
                # Add static plot
                static_file = f"{qtl_type}_{plot_type}.png"
                if static_file in available_files:
                    html_content += f"""
                        <div class="plot-item">
                            <h3>Static Plot</h3>
                            <img src="{static_file}" alt="{title}">
                            <p>{description}</p>
                        </div>
                    """
                
                # Add interactive plot if available
                interactive_file = f"{qtl_type}_interactive_{plot_type.split('_')[0]}.html"
                if interactive_file in available_files:
                    html_content += f"""
                        <div class="plot-item">
                            <h3>Interactive Plot</h3>
                            <iframe src="{interactive_file}" class="interactive-plot"></iframe>
                            <p>Interactive visualization for detailed exploration</p>
                        </div>
                    """
                
                html_content += "</div></div>"
        
        html_content += f"""
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
                            <li><strong>Statistical Properties:</strong> Assess changes in skewness, kurtosis, and normality</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📁 Generated Files</h2>
                    <div class="summary-box">
                        <strong>Available in: {output_dir}</strong>
                        <ul>
        """
        
        for file in available_files:
            html_content += f'<li>{file}</li>'
        
        html_content += """
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

# Utility function for modular pipeline integration - MAINTAIN EXACT SIGNATURE FOR BACKWARD COMPATIBILITY
def run_normalization_comparison(config, qtl_type, raw_data, normalized_data, normalization_method, results_dir):
    """
    Main function for normalization comparison module in the modular pipeline
    Returns: dict (comparison results)
    
    MAINTAINS EXACT SIGNATURE FOR BACKWARD COMPATIBILITY
    """
    try:
        logger.info(f"🚀 Starting normalization comparison for {qtl_type}...")
        
        # Initialize comparator
        comparator = NormalizationComparison(config, results_dir)
        
        # Check if enhanced pipeline should be used
        enable_enhanced = config.get('enhanced_pipeline', {}).get('enable', False)
        
        if enable_enhanced and normalized_data is None:
            # Use enhanced pipeline if no normalized data provided
            logger.info("🔧 Using enhanced pipeline with batch correction")
            comparison_results = comparator.generate_enhanced_pipeline_comparison(
                qtl_type, raw_data, normalization_method
            )
        else:
            # Use original comparison functionality
            logger.info("🔧 Using standard normalization comparison")
            comparison_results = comparator.generate_comprehensive_comparison(
                qtl_type, raw_data, normalized_data, normalization_method
            )
        
        if comparison_results and comparison_results.get('plots_generated'):
            logger.info(f"✅ Normalization comparison completed for {qtl_type}")
            return comparison_results
        else:
            logger.error(f"❌ Normalization comparison failed for {qtl_type}")
            return {'plots_generated': []}
            
    except Exception as e:
        logger.error(f"❌ Normalization comparison module failed: {e}")
        return {'plots_generated': []}

# NEW: Enhanced pipeline function for explicit enhanced pipeline usage
def run_enhanced_normalization_pipeline(config, qtl_type, raw_data, normalization_method=None, results_dir='results'):
    """
    Enhanced pipeline function with batch correction support
    Use this for explicit enhanced pipeline execution
    """
    try:
        logger.info(f"🚀 Starting enhanced normalization pipeline for {qtl_type}...")
        
        # Initialize comparator
        comparator = NormalizationComparison(config, results_dir)
        
        # Run enhanced pipeline
        comparison_results = comparator.generate_enhanced_pipeline_comparison(
            qtl_type, raw_data, normalization_method
            )
        
        if comparison_results:
            logger.info(f"✅ Enhanced normalization pipeline completed for {qtl_type}")
            return comparison_results
        else:
            logger.error(f"❌ Enhanced normalization pipeline failed for {qtl_type}")
            return {'plots_generated': []}
            
    except Exception as e:
        logger.error(f"❌ Enhanced normalization pipeline failed: {e}")
        return {'plots_generated': []}

# Maintain backward compatibility
if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate sample data for testing
    np.random.seed(42)
    raw_data = pd.DataFrame(np.random.randn(100, 50))
    normalized_data = pd.DataFrame(np.random.randn(100, 50) * 0.5 + 1)
    
    # Test standard comparison (backward compatibility)
    results = run_normalization_comparison(config, 'eqtl', raw_data, normalized_data, 'log2', 'test_results')
    print(f"Standard comparison completed: {len(results.get('plots_generated', []))} plots generated")
    
    # Test enhanced pipeline
    enhanced_results = run_enhanced_normalization_pipeline(config, 'eqtl', raw_data, 'vst', 'test_enhanced_results')
    print(f"Enhanced pipeline completed: {len(enhanced_results.get('plots_generated', []))} plots generated")
