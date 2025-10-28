#!/usr/bin/env python3
"""
Optimized QTL analysis utilities with tensorQTL v1.0.10 compatibility
Complete pipeline for cis/trans QTL analysis with robust error handling
and optimized CPU/GPU utilization

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Optimized for tensorQTL v1.0.10 with enhanced performance and error handling.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess
import warnings
import psutil
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import tempfile
import gc
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import sys

# Set up basic logger first for import errors
if 'logger' not in locals():
    logger = logging.getLogger('QTLPipeline')
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import tensorQTL with v1.0.10 specific imports
TENSORQTL_AVAILABLE = False
TENSORQTL_IMPORT_ERROR = None

try:
    # First try to import torch (required by tensorqtl)
    try:
        import torch
        TORCH_AVAILABLE = True
        logger.info(f"✅ PyTorch successfully imported (version: {torch.__version__})")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA is available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                logger.info(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("🔢 CUDA not available, using CPU")
            
    except ImportError as e:
        TORCH_AVAILABLE = False
        logger.error(f"❌ PyTorch import failed: {e}")
        logger.error("Please install PyTorch: pip install torch")
        raise

    # Now try to import tensorqtl v1.0.10
    try:
        import tensorqtl
        from tensorqtl import genotypeio, cis, trans, calculate_qvalues
        logger.info(f"✅ tensorQTL v1.0.10 successfully imported")
        TENSORQTL_AVAILABLE = True
        
    except ImportError as e:
        TENSORQTL_AVAILABLE = False
        TENSORQTL_IMPORT_ERROR = str(e)
        logger.error(f"❌ tensorQTL v1.0.10 import failed: {e}")
        logger.error("Please install tensorqtl v1.0.10: pip install tensorqtl==1.0.10")

except Exception as e:
    logger.error(f"❌ Unexpected error during tensorQTL import: {e}")
    TENSORQTL_AVAILABLE = False

# Import other dependencies with fallbacks
try:
    from .normalization_comparison import NormalizationComparison
    from .genotype_processing import GenotypeProcessor
except ImportError:
    try:
        from scripts.utils.normalization_comparison import NormalizationComparison
        from scripts.utils.genotype_processing import GenotypeProcessor
    except ImportError as e:
        if 'logger' in locals():
            logger.warning(f"Some optional dependencies not available: {e}")
        NormalizationComparison = None
        GenotypeProcessor = None

warnings.filterwarnings('ignore')

class DynamicDataHandler:
    """Optimized handler for dynamic QTL data alignment and processing"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data_handling', {})
        
    def align_qtl_data(self, genotype_samples, phenotype_df, covariate_df=None):
        """Align genotype, phenotype, and covariate data with comprehensive validation"""
        logger.info("🔍 Aligning QTL data across all datasets...")
        
        # Convert all sample identifiers to strings to ensure hashable types
        genotype_samples = [str(sample) for sample in genotype_samples]
        phenotype_samples = [str(sample) for sample in phenotype_df.columns]
        
        if covariate_df is not None and not covariate_df.empty:
            covariate_samples = [str(sample) for sample in covariate_df.columns]
        else:
            covariate_samples = []
        
        # Convert to sets for fast operations
        genotype_set = set(genotype_samples)
        phenotype_set = set(phenotype_samples)
        
        if covariate_samples:
            covariate_set = set(covariate_samples)
        else:
            covariate_set = set()
        
        # Find common samples
        if covariate_set:
            common_samples = genotype_set & phenotype_set & covariate_set
        else:
            common_samples = genotype_set & phenotype_set
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found across genotype, phenotype, and covariate data")
        
        common_samples = sorted(common_samples)
        
        # Subset data to common samples
        aligned_phenotype = phenotype_df[common_samples]
        
        if covariate_set:
            aligned_covariates = covariate_df[common_samples]
        else:
            aligned_covariates = pd.DataFrame()
        
        # Log alignment results
        logger.info(f"✅ Data alignment completed: {len(common_samples)} common samples")
        
        return {
            'phenotype': aligned_phenotype,
            'covariates': aligned_covariates,
            'common_samples': common_samples
        }
    
    def validate_phenotype_data(self, phenotype_df, qtl_type):
        """Validate phenotype data with QTL-type specific checks"""
        logger.info(f"🔍 Validating {qtl_type} phenotype data...")
        
        if phenotype_df.empty:
            raise ValueError(f"{qtl_type} phenotype data is empty")
        
        # Check for numeric data
        non_numeric = phenotype_df.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            logger.warning(f"⚠️ Found {non_numeric.shape[1]} non-numeric columns, attempting conversion")
            try:
                phenotype_df = phenotype_df.apply(pd.to_numeric, errors='coerce')
                phenotype_df = phenotype_df.dropna(axis=1, how='all')
            except Exception as e:
                logger.error(f"❌ Could not convert non-numeric columns: {e}")
                raise
        
        # Check for constant features
        constant_features = phenotype_df.std(axis=1) == 0
        if constant_features.any():
            logger.warning(f"⚠️ Removing {constant_features.sum()} constant features")
            phenotype_df = phenotype_df[~constant_features]
        
        # Check for excessive missingness
        missing_threshold = self.data_config.get('missing_value_threshold', 0.2)
        missing_rates = phenotype_df.isna().sum(axis=1) / phenotype_df.shape[1]
        high_missing = missing_rates > missing_threshold
        
        if high_missing.any():
            logger.warning(f"⚠️ Removing {high_missing.sum()} features with >{missing_threshold*100}% missing values")
            phenotype_df = phenotype_df[~high_missing]
        
        logger.info(f"✅ {qtl_type} validation: {phenotype_df.shape[0]} features, {phenotype_df.shape[1]} samples")
        return phenotype_df

    def generate_enhanced_covariates(self, phenotype_df, existing_covariates=None):
        """Generate enhanced covariates including PCA components"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            logger.info("🔧 Generating enhanced covariates...")
            
            # Prepare data for PCA
            pheno_for_pca = phenotype_df.T.fillna(phenotype_df.T.mean())
            
            # Remove constant features
            constant_mask = pheno_for_pca.std() == 0
            if constant_mask.any():
                pheno_for_pca = pheno_for_pca.loc[:, ~constant_mask]
            
            if pheno_for_pca.shape[1] < 2:
                logger.warning("⚠️ Insufficient features for PCA")
                return existing_covariates
            
            # Standardize and perform PCA
            scaler = StandardScaler()
            pheno_scaled = scaler.fit_transform(pheno_for_pca)
            
            n_components = min(10, pheno_scaled.shape[1], pheno_scaled.shape[0] - 1)
            if n_components < 1:
                return existing_covariates
            
            pca = PCA(n_components=n_components)
            pc_components = pca.fit_transform(pheno_scaled)
            
            # Create PCA covariates
            pc_columns = [f'PC{i+1}' for i in range(n_components)]
            pc_df = pd.DataFrame(pc_components, index=pheno_for_pca.index, columns=pc_columns)
            pc_df = pc_df.T  # Transpose to covariates x samples
            
            # Combine with existing covariates
            if existing_covariates is not None and not existing_covariates.empty:
                enhanced_covariates = pd.concat([existing_covariates, pc_df])
            else:
                enhanced_covariates = pc_df
            
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"✅ Enhanced covariates: {n_components} PC components "
                       f"(explained variance: {explained_variance:.3f})")
            
            return enhanced_covariates
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced covariate generation failed: {e}")
            return existing_covariates

class HardwareOptimizer:
    """Optimize hardware utilization for tensorQTL analysis"""
    
    def __init__(self, config):
        self.config = config
        self.performance_config = config.get('performance', {})
        self.tensorqtl_config = config.get('tensorqtl', {})
        
    def setup_hardware(self):
        """Setup optimal hardware configuration for tensorQTL"""
        device_info = self.detect_available_devices()
        
        use_gpu = self.tensorqtl_config.get('use_gpu', False) and device_info['gpu_available']
        
        if use_gpu:
            device = self.setup_gpu()
        else:
            device = self.setup_cpu_optimized()
        
        return device, device_info
    
    def detect_available_devices(self):
        """Detect available hardware devices"""
        device_info = {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'cpu_cores': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # Check GPU availability
        if TENSORQTL_AVAILABLE and 'torch' in sys.modules:
            if torch.cuda.is_available():
                device_info['gpu_available'] = True
                device_info['gpu_count'] = torch.cuda.device_count()
                device_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(device_info['gpu_count'])]
        
        logger.info(f"🖥️  Hardware: {device_info['cpu_cores']} CPU cores, "
                   f"{device_info['memory_gb']:.1f} GB RAM, "
                   f"{device_info['gpu_count']} GPUs")
        
        return device_info
    
    def setup_gpu(self):
        """Setup GPU configuration for optimal performance"""
        if not TENSORQTL_AVAILABLE or 'torch' not in sys.modules:
            logger.warning("⚠️ GPU setup skipped: tensorQTL or PyTorch not available")
            return self.setup_cpu_optimized()
            
        try:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.backends.cudnn.benchmark = True
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🎯 GPU acceleration enabled")
            return torch.device('cuda')
            
        except Exception as e:
            logger.warning(f"⚠️ GPU setup failed: {e}, falling back to CPU")
            return self.setup_cpu_optimized()
    
    def setup_cpu_optimized(self):
        """Setup CPU configuration for optimal multi-core performance"""
        try:
            num_threads = self.performance_config.get('num_threads', min(16, os.cpu_count()))
            
            if 'torch' in sys.modules:
                try:
                    torch.set_num_threads(num_threads)
                    torch.set_num_interop_threads(num_threads)
                except RuntimeError:
                    logger.warning("⚠️ Torch threads already initialized")
            
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            
            logger.info(f"🔢 Using {num_threads} CPU threads")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ CPU optimization failed: {e}")
            return None

class QTLConfig:
    """Optimized QTL configuration management for tensorQTL v1.0.10"""
    def __init__(self, config):
        self.config = config
        self.tensorqtl_config = config.get('tensorqtl', {})
        self.performance_config = config.get('performance', {})
        
        # Initialize hardware optimizer
        self.hardware_optimizer = HardwareOptimizer(config)
        
    def get_analysis_params(self, analysis_type):
        """Get analysis parameters optimized for tensorQTL v1.0.10"""
        # Setup hardware first
        device, device_info = self.hardware_optimizer.setup_hardware()
        
        base_params = {
            'cis_window': self.tensorqtl_config.get('cis_window', 1000000),
            'maf_threshold': self.tensorqtl_config.get('maf_threshold', 0.05),
            'min_maf': self.tensorqtl_config.get('min_maf', 0.01),
            'fdr_threshold': self.tensorqtl_config.get('fdr_threshold', 0.05),
            'num_permutations': self.tensorqtl_config.get('num_permutations', 1000),
            'batch_size': self.tensorqtl_config.get('batch_size', 20000),
            'seed': self.tensorqtl_config.get('seed', 12345),
            'run_eigenmt': self.tensorqtl_config.get('run_eigenmt', False),
            'device': device,
            'device_info': device_info,
            'use_gpu': device_info['gpu_available'] and self.tensorqtl_config.get('use_gpu', False)
        }
        
        # Analysis-specific adjustments
        if analysis_type == 'trans':
            base_params.update({
                'batch_size': self.tensorqtl_config.get('trans_batch_size', base_params['batch_size']),
                'pval_threshold': self.tensorqtl_config.get('trans_pval_threshold', 1e-5),
                'return_sparse': self.tensorqtl_config.get('return_sparse', True)
            })
        
        # Performance tuning based on hardware
        if base_params['use_gpu']:
            logger.info("🚀 Using GPU-optimized parameters")
            base_params['batch_size'] = base_params['batch_size'] * 2
        else:
            logger.info("🔢 Using CPU-optimized parameters")
            base_params['num_threads'] = self.performance_config.get('num_threads', min(16, os.cpu_count()))
        
        return base_params

class PhenotypeProcessor:
    """Optimized phenotype data processing for tensorQTL v1.0.10"""
    
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        self.qc_config = config.get('qc', {})
        self.normalization_config = config.get('normalization', {})
        self.data_handler = DynamicDataHandler(config)
        
    def prepare_phenotype_data(self, qtl_type, genotype_samples=None):
        """Prepare phenotype data optimized for tensorQTL v1.0.10"""
        logger.info(f"🔧 Preparing {qtl_type} phenotype data...")
        
        try:
            # Get phenotype file path
            config_key = self._map_qtl_type_to_config_key(qtl_type)
            pheno_file = self.config['input_files'].get(config_key)
            
            if not pheno_file or not os.path.exists(pheno_file):
                raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")
            
            # Load phenotype data
            pheno_df = self._load_phenotype_data(pheno_file, qtl_type)
            logger.info(f"📊 Loaded {qtl_type} data: {pheno_df.shape[0]} features, {pheno_df.shape[1]} samples")
            
            # Load covariates if available
            covariates_file = self.config['input_files'].get('covariates')
            if covariates_file and os.path.exists(covariates_file):
                cov_df = self._load_covariate_data(covariates_file)
                logger.info(f"📊 Loaded covariates: {cov_df.shape[0]} covariates")
            else:
                cov_df = pd.DataFrame()
                logger.info("ℹ️ No covariate file found")
            
            # Align data if genotype samples are provided
            if genotype_samples is not None:
                aligned_data = self.data_handler.align_qtl_data(genotype_samples, pheno_df, cov_df)
                pheno_df = aligned_data['phenotype']
                cov_df = aligned_data['covariates']
                common_samples = aligned_data['common_samples']
            else:
                common_samples = pheno_df.columns.tolist()
            
            # Validate phenotype data
            pheno_df = self.data_handler.validate_phenotype_data(pheno_df, qtl_type)
            
            # Apply QC filters
            if self.qc_config.get('filter_low_expressed', True):
                pheno_df = self._apply_qc_filters(pheno_df, qtl_type)
            
            # NEW: Apply expression-specific filtering for eQTL data
            if qtl_type == 'eqtl' and self.qc_config.get('filter_lowly_expressed_genes', True):
                pheno_df = self._filter_lowly_expressed_genes(pheno_df, qtl_type)
            
            # Apply normalization
            if self.qc_config.get('normalize', True):
                normalized_df = self._apply_normalization(pheno_df, qtl_type)
                normalization_method = self.normalization_config.get(qtl_type, {}).get('method', 'unknown')
                logger.info(f"🔄 Applied {normalization_method} normalization")
            else:
                normalized_df = pheno_df
                logger.info("📊 Using raw data without normalization")
            
            # Generate enhanced covariates if enabled
            if self.config.get('enhanced_qc', {}).get('generate_enhanced_covariates', True):
                enhanced_cov_df = self.data_handler.generate_enhanced_covariates(normalized_df, cov_df)
                if enhanced_cov_df is not None:
                    cov_df = enhanced_cov_df
            
            # Generate normalization comparison if enabled
            if self.config.get('enhanced_qc', {}).get('generate_normalization_plots', True) and NormalizationComparison:
                self._generate_normalization_comparison(pheno_df, normalized_df, qtl_type)
            
            # Prepare for tensorQTL (samples x features)
            normalized_df = normalized_df.T
            
            # Save processed data
            output_files = self._save_processed_data(normalized_df, qtl_type, cov_df)
            
            final_feature_count = normalized_df.shape[1]
            logger.info(f"✅ Prepared {qtl_type} data: {final_feature_count} features retained")
            
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Phenotype preparation failed for {qtl_type}: {e}")
            raise

    def _filter_lowly_expressed_genes(self, pheno_df, qtl_type):
        """
        NEW: Filter out genes with low expression in more than X% of samples
        Based on QTL analysis best practices - removes genes that are lowly expressed in too many samples
        """
        if qtl_type != 'eqtl':
            return pheno_df  # Only apply to expression data
        
        logger.info("🔍 Filtering lowly expressed genes based on QTL best practices...")
        
        # Get configuration parameters with defaults
        low_expression_threshold = self.qc_config.get('low_expression_threshold', 0.1)
        max_low_expression_samples_percentage = self.qc_config.get('max_low_expression_samples_percentage', 10)
        
        original_count = pheno_df.shape[0]
        
        # Calculate percentage of samples with low expression for each gene
        low_expression_mask = pheno_df < low_expression_threshold
        low_expression_percentage = (low_expression_mask.sum(axis=1) / pheno_df.shape[1]) * 100
        
        # Filter genes where low expression percentage is below threshold
        keep_genes = low_expression_percentage <= max_low_expression_samples_percentage
        filtered_df = pheno_df[keep_genes]
        
        filtered_count = filtered_df.shape[0]
        removed_count = original_count - filtered_count
        
        logger.info(f"🔧 Low expression filtering: {filtered_count}/{original_count} genes retained "
                   f"(threshold: {low_expression_threshold}, max low samples: {max_low_expression_samples_percentage}%, "
                   f"removed: {removed_count})")
        
        return filtered_df
    
    def _map_qtl_type_to_config_key(self, qtl_type):
        """Map QTL type to config file key"""
        mapping = {
            'eqtl': 'expression',
            'pqtl': 'protein', 
            'sqtl': 'splicing'
        }
        return mapping.get(qtl_type, qtl_type)
    
    def _load_phenotype_data(self, file_path, qtl_type):
        """Load phenotype data with robust error handling"""
        try:
            logger.info(f"📁 Loading {qtl_type} phenotype file: {file_path}")
            
            # Try different separators
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(file_path, sep=sep, index_col=0)
                    if not df.empty:
                        logger.info(f"✅ Successfully loaded with separator '{sep}'")
                        df = df.apply(pd.to_numeric, errors='coerce')
                        return df
                except:
                    continue
            
            # Fallback
            df = pd.read_csv(file_path, index_col=0)
            df = df.apply(pd.to_numeric, errors='coerce')
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading phenotype data: {e}")
            raise
    
    def _load_covariate_data(self, covariate_file):
        """Load covariate data with dynamic format handling"""
        try:
            logger.info(f"📁 Loading covariate file: {covariate_file}")
            
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(covariate_file, sep=sep, index_col=0)
                    if not df.empty:
                        logger.info(f"✅ Successfully loaded covariates with separator '{sep}'")
                        df = df.apply(pd.to_numeric, errors='coerce')
                        return df
                except:
                    continue
            
            df = pd.read_csv(covariate_file, index_col=0)
            df = df.apply(pd.to_numeric, errors='coerce')
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading covariate data: {e}")
            return pd.DataFrame()
    
    def _log_file_preview(self, df, file_type, max_cols=5, max_rows=3):
        """Log file preview with dimensions and sample data"""
        try:
            rows, cols = df.shape
            logger.info(f"📊 {file_type} file dimensions: {rows} rows × {cols} columns")
            
            # Show limited preview
            preview_cols = min(cols, max_cols)
            preview_rows = min(rows, max_rows)
            
            if preview_cols > 0 and preview_rows > 0:
                logger.info(f"🔍 {file_type} file preview (first {preview_cols} columns, {preview_rows} rows):")
                
                # Get preview data
                preview_df = df.iloc[:preview_rows, :preview_cols]
                
                # Create formatted preview
                preview_str = str(preview_df)
                
                # Log in chunks if too long
                lines = preview_str.split('\n')
                for line in lines[:preview_rows + 2]:
                    logger.info(f"   {line}")
                
                if len(lines) > preview_rows + 2:
                    logger.info(f"   ... (showing first {preview_rows} rows and {preview_cols} columns)")
            
            # Log data types summary
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).shape[1]
            logger.info(f"📋 {file_type} data types: {numeric_cols} numeric, {non_numeric_cols} non-numeric columns")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not generate file preview for {file_type}: {e}")
    
    def _apply_qc_filters(self, pheno_df, qtl_type):
        """Apply quality control filters"""
        original_count = pheno_df.shape[0]
        filtered_df = pheno_df.copy()
        
        # Remove constant features
        constant_threshold = self.qc_config.get('constant_threshold', 0.95)
        non_constant_mask = (filtered_df.nunique(axis=1) / filtered_df.shape[1]) > (1 - constant_threshold)
        filtered_df = filtered_df[non_constant_mask]
        constant_removed = original_count - filtered_df.shape[0]
        
        # Remove features with too many missing values
        missing_threshold = self.qc_config.get('missing_value_threshold', 0.2)
        low_missing_mask = (filtered_df.isna().sum(axis=1) / filtered_df.shape[1]) < missing_threshold
        filtered_df = filtered_df[low_missing_mask]
        missing_removed = original_count - constant_removed - filtered_df.shape[0]
        
        # QTL-type specific filtering
        if qtl_type == 'eqtl':
            threshold = self.qc_config.get('expression_threshold', 0.1)
            mean_expression = filtered_df.mean(axis=1)
            expressed_mask = mean_expression > threshold
            filtered_df = filtered_df[expressed_mask]
            low_expression_removed = original_count - constant_removed - missing_removed - filtered_df.shape[0]
        elif qtl_type in ['pqtl', 'sqtl']:
            # Filter based on variance
            variance_threshold = filtered_df.var(axis=1).quantile(0.1)
            high_variance_mask = filtered_df.var(axis=1) > variance_threshold
            filtered_df = filtered_df[high_variance_mask]
            low_variance_removed = original_count - constant_removed - missing_removed - filtered_df.shape[0]
        else:
            low_expression_removed = 0
            low_variance_removed = 0
        
        filtered_count = filtered_df.shape[0]
        logger.info(f"🔧 QC filtering: {filtered_count}/{original_count} features retained")
        
        return filtered_df
    
    def _apply_normalization(self, pheno_df, qtl_type):
        """Apply normalization based on user-defined method"""
        norm_config = self.normalization_config.get(qtl_type, {})
        method = norm_config.get('method', 'log2')
        
        logger.info(f"🔄 Applying {method} normalization for {qtl_type}")
        
        if method == 'log2':
            return self._apply_log2_normalization(pheno_df, qtl_type)
        elif method == 'vst':
            return self._apply_vst_normalization(pheno_df, qtl_type)
        elif method == 'quantile':
            return self._apply_quantile_normalization(pheno_df, qtl_type)
        elif method == 'zscore':
            return self._apply_zscore_normalization(pheno_df, qtl_type)
        elif method == 'arcsinh':
            return self._apply_arcsinh_normalization(pheno_df, qtl_type)
        elif method == 'tpm':
            return self._apply_tpm_normalization(pheno_df, qtl_type)
        elif method == 'raw':
            return pheno_df
        else:
            logger.warning(f"⚠️ Unknown normalization method '{method}', using raw data")
            return pheno_df
    
    def _apply_log2_normalization(self, pheno_df, qtl_type):
        """Apply log2 transformation"""
        norm_config = self.normalization_config.get(qtl_type, {})
        pseudocount = norm_config.get('log2_pseudocount', 1)
        remove_zeros = norm_config.get('remove_zeros', True)
        
        if remove_zeros:
            # Replace zeros with NaN and remove all-zero features
            original_count = pheno_df.shape[0]
            pheno_df = pheno_df.replace(0, np.nan)
            pheno_df = pheno_df.dropna(how='all')
            zeros_removed = original_count - pheno_df.shape[0]
            if zeros_removed > 0:
                logger.info(f"🔧 Removed {zeros_removed} features with all zeros")
        
        normalized_df = np.log2(pheno_df + pseudocount)
        logger.info(f"✅ Applied log2 transformation (pseudocount={pseudocount})")
        return normalized_df
    
    def _apply_vst_normalization(self, pheno_df, qtl_type):
        """Apply VST normalization using DESeq2"""
        if qtl_type != 'eqtl':
            logger.warning("VST normalization is typically for expression data")
        
        logger.info("🔢 Adding +1 and rounding counts for VST normalization...")
        pheno_df = pheno_df + 1
        pheno_df = np.round(pheno_df).astype(int)
        
        temp_input_path = None
        temp_output_path = None
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='_input.txt', delete=False) as temp_input, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='_vst.txt', delete=False) as temp_output:
                
                temp_input_path = temp_input.name
                temp_output_path = temp_output.name
            
            # Save data for R processing
            pheno_df.reset_index().to_csv(temp_input_path, sep='\t', index=False)
            
            # Get R script path
            r_script_path = self.config['paths'].get('r_script_deseq2', 'scripts/utils/deseq2_vst.R')
            if not os.path.exists(r_script_path):
                raise FileNotFoundError(f"DESeq2 R script not found: {r_script_path}")
            
            # Build R command
            norm_config = self.normalization_config.get(qtl_type, {})
            blind = norm_config.get('vst_blind', True)
            fit_type = norm_config.get('fit_type', 'parametric')
            
            cmd = f"Rscript {r_script_path} {temp_input_path} {temp_output_path} {blind} {fit_type}"
            
            # Execute R script
            result = run_command(cmd, "DESeq2 VST normalization", self.config, check=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"DESeq2 VST normalization failed")
            
            # Load normalized data
            if os.path.exists(temp_output_path):
                vst_df = pd.read_csv(temp_output_path, sep='\t', index_col=0)
                vst_df = vst_df.apply(pd.to_numeric, errors='coerce')
                logger.info("✅ VST normalization completed")
                return vst_df
            else:
                raise FileNotFoundError("VST output file not generated")
                
        except Exception as e:
            logger.error(f"❌ VST normalization failed: {e}")
            raise RuntimeError(f"VST normalization failed: {e}")
        finally:
            # Clean up temporary files
            for temp_file in [temp_input_path, temp_output_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def _apply_quantile_normalization(self, pheno_df, qtl_type):
        """Apply quantile normalization"""
        try:
            from sklearn.preprocessing import quantile_transform
            
            normalized_array = quantile_transform(pheno_df.T, n_quantiles=min(1000, pheno_df.shape[0]))
            normalized_df = pd.DataFrame(normalized_array.T, index=pheno_df.index, columns=pheno_df.columns)
            
            logger.info("✅ Quantile normalization completed")
            return normalized_df
        except ImportError:
            raise ImportError("scikit-learn not available for quantile normalization")
    
    def _apply_zscore_normalization(self, pheno_df, qtl_type):
        """Apply z-score normalization per feature"""
        normalized_df = (pheno_df - pheno_df.mean(axis=1).values.reshape(-1, 1)) 
        normalized_df = normalized_df / pheno_df.std(axis=1).values.reshape(-1, 1)
        
        # Handle constant features (std=0)
        constant_mask = pheno_df.std(axis=1) == 0
        if constant_mask.any():
            normalized_df.loc[constant_mask] = 0
            logger.warning(f"⚠️ Found {constant_mask.sum()} constant features, setting z-score to 0")
        
        logger.info("✅ Z-score normalization completed")
        return normalized_df

    def _apply_arcsinh_normalization(self, pheno_df, qtl_type):
        """Apply arcsinh transformation"""
        norm_config = self.normalization_config.get(qtl_type, {})
        cofactor = norm_config.get('arcsinh_cofactor', 1)
        
        normalized_df = np.arcsinh(pheno_df / cofactor)
        logger.info(f"✅ Arcsinh transformation completed (cofactor={cofactor})")
        return normalized_df

    def _apply_tpm_normalization(self, pheno_df, qtl_type):
        """Apply TPM-like normalization"""
        # Simplified TPM calculation (without gene lengths)
        rpm_df = pheno_df.div(pheno_df.sum(axis=0)) * 1e6
        logger.info("✅ TPM-like normalization completed")
        return rpm_df

    def _generate_normalization_comparison(self, raw_df, normalized_df, qtl_type):
        """Generate normalization comparison plots"""
        try:
            if NormalizationComparison:
                comparison = NormalizationComparison(self.config, self.results_dir)
                comparison.generate_comprehensive_comparison(
                    qtl_type, raw_df.copy(), normalized_df, 
                    self.normalization_config[qtl_type].get('method', 'unknown')
                )
                logger.info(f"📊 Normalization comparison completed for {qtl_type}")
            else:
                logger.warning("NormalizationComparison module not available")
        except Exception as e:
            logger.warning(f"⚠️ Normalization comparison failed: {e}")
    
    def _save_processed_data(self, normalized_df, qtl_type, covariate_df):
        """Save processed phenotype data"""
        # Ensure all data is numeric
        normalized_df = normalized_df.apply(pd.to_numeric, errors='coerce')
        
        if not covariate_df.empty:
            covariate_df = covariate_df.apply(pd.to_numeric, errors='coerce')
        
        # Save phenotype matrix
        output_format = self.config.get('tensorqtl', {}).get('output_format', 'parquet')
        if output_format == 'parquet':
            pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.parquet")
            try:
                normalized_df.to_parquet(pheno_file)
            except Exception as e:
                logger.warning(f"⚠️ Parquet save failed, using CSV: {e}")
                pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.txt.gz")
                normalized_df.to_csv(pheno_file, sep='\t', compression='gzip')
        else:
            pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.txt.gz")
            normalized_df.to_csv(pheno_file, sep='\t', compression='gzip')
        
        # Save covariates if available
        if not covariate_df.empty:
            cov_file = os.path.join(self.results_dir, f"{qtl_type}_covariates.parquet")
            try:
                covariate_df.to_parquet(cov_file)
            except Exception as e:
                logger.warning(f"⚠️ Covariate parquet save failed, using CSV: {e}")
                cov_file = os.path.join(self.results_dir, f"{qtl_type}_covariates.txt.gz")
                covariate_df.to_csv(cov_file, sep='\t', compression='gzip')
        else:
            cov_file = None
        
        # Create and save phenotype positions
        pheno_pos_file = os.path.join(self.results_dir, f"{qtl_type}_phenotype_positions.parquet")
        pheno_pos_df = self._create_phenotype_positions(normalized_df.columns, qtl_type)
        try:
            pheno_pos_df.to_parquet(pheno_pos_file)
        except Exception as e:
            logger.warning(f"⚠️ Phenotype positions parquet save failed, using CSV: {e}")
            pheno_pos_file = os.path.join(self.results_dir, f"{qtl_type}_phenotype_positions.txt.gz")
            pheno_pos_df.to_csv(pheno_pos_file, sep='\t', compression='gzip')
        
        logger.info(f"💾 Saved processed data: {pheno_file}")
        
        return {
            'phenotype_file': pheno_file,
            'phenotype_pos_file': pheno_pos_file,
            'covariate_file': cov_file,
            'phenotype_df': normalized_df,
            'phenotype_pos_df': pheno_pos_df,
            'covariate_df': covariate_df
        }
    
    def _create_phenotype_positions(self, feature_ids, qtl_type):
        """Create phenotype positions DataFrame"""
        annotation_file = self.config['input_files']['annotations']
        
        try:
            annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
        except:
            try:
                annot_df = pd.read_csv(annotation_file, sep='\t')
            except Exception as e:
                logger.warning(f"⚠️ Could not read annotation file: {e}, creating default positions")
                annot_df = pd.DataFrame()
        
        positions_data = []
        
        for feature_id in feature_ids:
            if not annot_df.empty and 'gene_id' in annot_df.columns:
                feature_annot = annot_df[annot_df['gene_id'] == feature_id]
                if len(feature_annot) > 0:
                    feature_annot = feature_annot.iloc[0]
                    positions_data.append({
                        'phenotype_id': feature_id,
                        'chr': str(feature_annot['chr']),
                        'start': int(feature_annot['start']),
                        'end': int(feature_annot['end']),
                        'strand': feature_annot.get('strand', '+')
                    })
                else:
                    # Create default annotation if not found
                    positions_data.append({
                        'phenotype_id': feature_id,
                        'chr': '1',
                        'start': 1,
                        'end': 1000,
                        'strand': '+'
                    })
            else:
                # Create default annotations
                positions_data.append({
                    'phenotype_id': feature_id,
                    'chr': '1',
                    'start': 1,
                    'end': 1000,
                    'strand': '+'
                })
        
        positions_df = pd.DataFrame(positions_data)
        positions_df = positions_df.set_index('phenotype_id')
        return positions_df

class GenotypeLoader:
    """Optimized genotype data loading for tensorQTL v1.0.10"""
    
    def __init__(self, config):
        self.config = config
        self.genotype_processing_config = config.get('genotype_processing', {})
    
    def load_genotypes(self, genotype_file):
        """Load genotype data optimized for tensorQTL v1.0.10"""
        logger.info("🔧 Loading genotype data for tensorQTL...")
        
        if not TENSORQTL_AVAILABLE:
            error_msg = "tensorQTL is not available. "
            if TENSORQTL_IMPORT_ERROR:
                error_msg += f"Import error: {TENSORQTL_IMPORT_ERROR}. "
            error_msg += "Please install: pip install tensorqtl==1.0.10"
            raise ImportError(error_msg)
        
        try:
            if genotype_file.endswith('.bed'):
                plink_prefix = genotype_file.replace('.bed', '')
                
                # Log PLINK file components
                for ext in ['.bed', '.bim', '.fam']:
                    plink_file = plink_prefix + ext
                    if os.path.exists(plink_file):
                        file_size = os.path.getsize(plink_file) / (1024**2)
                        logger.info(f"📄 PLINK component: {plink_file} ({file_size:.2f} MB)")
                
                # Load PLINK data
                pr = genotypeio.read_plink(plink_prefix)
                
                # Handle different return formats
                if isinstance(pr, tuple):
                    # Older tensorQTL versions return tuple
                    logger.info("📦 Detected tuple return format")
                    genotypes, variants, samples = pr
                    
                    logger.info(f"📊 Genotype data: {genotypes.shape[0]} variants × {genotypes.shape[1]} samples")
                    
                    # Create container for compatibility
                    class GenotypeContainer:
                        def __init__(self, genotypes, variants, samples):
                            self.genotypes = genotypes
                            self.variants = variants
                            self.samples = samples
                    
                    pr_container = GenotypeContainer(genotypes, variants, samples)
                    return pr_container
                else:
                    # Newer tensorQTL versions return object
                    logger.info(f"📊 Genotype data: {pr.genotypes.shape[0]} variants × {pr.genotypes.shape[1]} samples")
                    return pr
            else:
                raise ValueError(f"Unsupported genotype format: {genotype_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading genotype data: {e}")
            raise
    
    def optimize_genotype_data(self, genotype_reader):
        """Optimize genotype data for analysis"""
        if hasattr(genotype_reader, 'genotypes'):
            original_count = genotype_reader.genotypes.shape[0]
            genotypes_obj = genotype_reader.genotypes
            
            # Apply MAF filtering
            maf_threshold = self.genotype_processing_config.get('min_maf', 0.01)
            if maf_threshold > 0:
                if hasattr(genotypes_obj, 'maf'):
                    maf = genotypes_obj.maf()
                    keep_variants = maf >= maf_threshold
                    genotypes_obj = genotypes_obj[keep_variants]
                    maf_filtered_count = genotypes_obj.shape[0]
                    logger.info(f"🔧 MAF filtering: {maf_filtered_count}/{original_count} variants retained")
                else:
                    logger.warning("⚠️ MAF method not available, using manual MAF calculation")
                    maf = self._calculate_maf_manual(genotypes_obj)
                    keep_variants = maf >= maf_threshold
                    genotypes_obj = genotypes_obj[keep_variants]
                    maf_filtered_count = genotypes_obj.shape[0]
                    logger.info(f"🔧 MAF filtering (manual): {maf_filtered_count}/{original_count} variants retained")
            
            # Apply call rate filtering if needed
            call_rate_threshold = self.genotype_processing_config.get('min_call_rate', 0.95)
            if call_rate_threshold < 1.0:
                if hasattr(genotypes_obj, 'isnan'):
                    call_rate = 1 - genotypes_obj.isnan().mean(axis=1)
                else:
                    # Fallback for regular DataFrames
                    call_rate = 1 - genotypes_obj.isna().mean(axis=1)
                keep_variants = call_rate >= call_rate_threshold
                genotypes_obj = genotypes_obj[keep_variants]
                call_rate_filtered_count = genotypes_obj.shape[0]
                logger.info(f"🔧 Call rate filtering: {call_rate_filtered_count} variants retained (call rate >= {call_rate_threshold})")
            
            # Update the genotype reader
            genotype_reader.genotypes = genotypes_obj
            
            final_count = genotype_reader.genotypes.shape[0]
            logger.info(f"🔧 Genotype optimization: {final_count}/{original_count} variants retained")
                
        return genotype_reader
    
    def _calculate_maf_manual(self, genotypes_df):
        """Calculate Minor Allele Frequency manually - FIXED version"""
        try:
            if hasattr(genotypes_df, 'values'):
                geno_array = genotypes_df.values
            else:
                geno_array = genotypes_df
                
            # Ensure numeric data type
            if geno_array.dtype == np.object_ or geno_array.dtype.kind in 'OUS':
                logger.warning("⚠️ Genotype data contains string values, converting to numeric")
                try:
                    geno_array = pd.DataFrame(geno_array).apply(pd.to_numeric, errors='coerce').values
                except Exception as e:
                    logger.error(f"❌ Cannot convert string genotypes: {e}")
                    return np.ones(genotypes_df.shape[0])
            
            # Calculate allele frequencies
            allele_counts = np.nansum(geno_array, axis=1)
            valid_counts = np.sum(~np.isnan(geno_array), axis=1) * 2
            
            # Avoid division by zero
            valid_counts[valid_counts == 0] = 1
            
            freq = allele_counts / valid_counts
            maf = np.minimum(freq, 1 - freq)
            
            return maf
            
        except Exception as e:
            logger.error(f"❌ Manual MAF calculation failed: {e}")
            return np.ones(genotypes_df.shape[0])

def prepare_genotypes(config, results_dir):
    """Prepare genotype data optimized for tensorQTL"""
    logger.info("🔧 Preparing genotype data for tensorQTL...")
    
    try:
        if GenotypeProcessor:
            processor = GenotypeProcessor(config)
            input_file = config['input_files']['genotypes']
            genotype_file = processor.process_genotypes(input_file, results_dir)
        else:
            genotype_file = config['input_files']['genotypes']
            logger.warning("Using direct genotype processing")
        
        # Convert VCF to PLINK if needed
        if genotype_file.endswith('.vcf.gz') or genotype_file.endswith('.vcf'):
            plink_base = os.path.join(results_dir, "genotypes_plink")
            logger.info("🔄 Converting VCF to PLINK format...")
            
            plink_path = config['paths'].get('plink', 'plink')
            plink_threads = config.get('genotype_processing', {}).get('plink_threads', 1)
            
            cmd = f"{plink_path} --vcf {genotype_file} --make-bed --out {plink_base} --threads {plink_threads}"
            run_command(cmd, "Converting VCF to PLINK", config)
            
            genotype_file = plink_base + ".bed"
        
        logger.info(f"✅ Genotype preparation completed: {genotype_file}")
        return genotype_file
        
    except Exception as e:
        logger.error(f"❌ Genotype preparation failed: {e}")
        raise

def load_covariates(config, results_dir, qtl_type='eqtl'):
    """Load and prepare covariates for tensorQTL"""
    logger.info(f"🔧 Loading covariates for {qtl_type}...")
    
    try:
        # Try to load pre-processed covariates first
        cov_file = os.path.join(results_dir, f"{qtl_type}_covariates.parquet")
        if os.path.exists(cov_file):
            try:
                cov_df = pd.read_parquet(cov_file)
                logger.info(f"✅ Loaded pre-processed covariates: {cov_df.shape[1]} samples")
                return cov_df.T
            except Exception:
                cov_file = os.path.join(results_dir, f"{qtl_type}_covariates.txt.gz")
                if os.path.exists(cov_file):
                    cov_df = pd.read_csv(cov_file, sep='\t', index_col=0)
                    return cov_df.T
        
        # Fallback to original covariate file
        covariates_file = config['input_files'].get('covariates')
        if not covariates_file or not os.path.exists(covariates_file):
            logger.warning("⚠️ No covariate file found")
            return None
        
        # Load with dynamic format handling
        data_handler = DynamicDataHandler(config)
        cov_df = data_handler._load_covariate_data(covariates_file)
        
        if cov_df.empty:
            logger.warning("⚠️ Covariate data is empty")
            return None
        
        # Convert all sample names to strings
        cov_df.columns = [str(col) for col in cov_df.columns]
        cov_df.index = [str(idx) for idx in cov_df.index]
        
        # Transpose for tensorQTL
        cov_df = cov_df.T
        
        # Remove constant covariates
        constant_covariates = cov_df.columns[cov_df.nunique() <= 1]
        if len(constant_covariates) > 0:
            cov_df = cov_df.drop(columns=constant_covariates)
            logger.warning(f"⚠️ Removed {len(constant_covariates)} constant covariates")
        
        # Check for missing values
        missing_count = cov_df.isna().sum().sum()
        if missing_count > 0:
            logger.warning(f"⚠️ Covariates contain {missing_count} missing values, imputing with mean")
            cov_df = cov_df.fillna(cov_df.mean())
        
        logger.info(f"✅ Loaded covariates: {cov_df.shape[1]} covariates")
        return cov_df
        
    except Exception as e:
        logger.error(f"❌ Error loading covariates: {e}")
        return None

def run_cis_analysis(config, genotype_file, qtl_type, results_dir):
    """Run cis-QTL analysis using tensorQTL v1.0.10"""
    if not TENSORQTL_AVAILABLE:
        error_msg = "tensorQTL is not available. "
        if TENSORQTL_IMPORT_ERROR:
            error_msg += f"Import error: {TENSORQTL_IMPORT_ERROR}. "
        error_msg += "Please install: pip install tensorqtl==1.0.10"
        raise ImportError(error_msg)
    
    logger.info(f"🔍 Running {qtl_type} cis-QTL analysis...")
    
    try:
        # Initialize configuration
        qtl_config = QTLConfig(config)
        params = qtl_config.get_analysis_params('cis')
        
        # Log hardware configuration
        if params['use_gpu']:
            logger.info(f"🎮 Using GPU for {qtl_type} cis-QTL analysis")
        else:
            logger.info(f"🔢 Using CPU for {qtl_type} cis-QTL analysis")
        
        # Load and optimize genotype data
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        pr = genotype_loader.optimize_genotype_data(pr)
        
        # Extract samples and ensure string format
        if hasattr(pr, 'samples'):
            genotype_samples = [str(sample) for sample in pr.samples]
        elif hasattr(pr, 'genotypes') and hasattr(pr.genotypes, 'columns'):
            genotype_samples = [str(sample) for sample in pr.genotypes.columns.tolist()]
        else:
            try:
                genotype_samples = [str(sample) for sample in list(pr.genotypes.columns)]
            except:
                raise AttributeError("Cannot extract samples from genotype reader")
        
        # Prepare phenotype data
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type, genotype_samples)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir, qtl_type)
        
        # Prepare data for tensorQTL
        phenotype_df_t = pheno_data['phenotype_df'].T
        phenotype_pos_df = pheno_data['phenotype_pos_df']
        
        # Ensure all sample names are strings
        phenotype_df_t.index = [str(idx) for idx in phenotype_df_t.index]
        phenotype_df_t.columns = [str(col) for col in phenotype_df_t.columns]
        
        # Run cis-QTL analysis
        logger.info("🔬 Running tensorQTL cis mapping...")
        
        cis_df = cis.map_cis(
            pr,
            phenotype_df_t, 
            phenotype_pos_df,
            covariates_df=covariates_df,
            window=params['cis_window'],
            seed=params['seed'],
            run_eigenmt=params['run_eigenmt']
        )
        
        # Save results
        result_file = os.path.join(results_dir, f"{qtl_type}_cis.cis_qtl.txt.gz")
        cis_df.to_csv(result_file, sep='\t', compression='gzip')
        
        # Count significant associations
        significant_count = count_significant_associations(results_dir, f"{qtl_type}_cis", params['fdr_threshold'])
        
        logger.info(f"✅ {qtl_type} cis: Found {significant_count} significant associations")
        
        # Clean up GPU memory if used
        if params['use_gpu'] and TENSORQTL_AVAILABLE and 'torch' in sys.modules:
            torch.cuda.empty_cache()
        
        return {
            'result_file': result_file,
            'significant_count': significant_count,
            'status': 'completed',
            'hardware_used': 'GPU' if params['use_gpu'] else 'CPU'
        }
        
    except Exception as e:
        logger.error(f"❌ cis-QTL analysis failed for {qtl_type}: {e}")
        return {
            'result_file': "",
            'significant_count': 0,
            'status': 'failed',
            'error': str(e)
        }

def run_trans_analysis(config, genotype_file, qtl_type, results_dir):
    """Run trans-QTL analysis using tensorQTL v1.0.10"""
    if not TENSORQTL_AVAILABLE:
        error_msg = "tensorQTL is not available. "
        if TENSORQTL_IMPORT_ERROR:
            error_msg += f"Import error: {TENSORQTL_IMPORT_ERROR}. "
        error_msg += "Please install: pip install tensorqtl==1.0.10"
        raise ImportError(error_msg)
    
    logger.info(f"🔍 Running {qtl_type} trans-QTL analysis...")
    
    try:
        # Initialize configuration
        qtl_config = QTLConfig(config)
        params = qtl_config.get_analysis_params('trans')
        
        if params['use_gpu']:
            logger.info(f"🎮 Using GPU for {qtl_type} trans-QTL analysis")
        else:
            logger.info(f"🔢 Using CPU for {qtl_type} trans-QTL analysis")
        
        # Load and optimize genotype data
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        pr = genotype_loader.optimize_genotype_data(pr)
        
        # Extract samples and ensure string format
        if hasattr(pr, 'samples'):
            genotype_samples = [str(sample) for sample in pr.samples]
        elif hasattr(pr, 'genotypes') and hasattr(pr.genotypes, 'columns'):
            genotype_samples = [str(sample) for sample in pr.genotypes.columns.tolist()]
        else:
            try:
                genotype_samples = [str(sample) for sample in list(pr.genotypes.columns)]
            except:
                raise AttributeError("Cannot extract samples from genotype reader")
        
        # Prepare phenotype data
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type, genotype_samples)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir, qtl_type)
        
        # Prepare data for tensorQTL
        phenotype_df_t = pheno_data['phenotype_df'].T
        
        # Ensure all sample names are strings
        phenotype_df_t.index = [str(idx) for idx in phenotype_df_t.index]
        phenotype_df_t.columns = [str(col) for col in phenotype_df_t.columns]
        
        # Run trans-QTL analysis
        logger.info("🔬 Running tensorQTL trans mapping...")
        
        trans_df = trans.map_trans(
            pr,
            phenotype_df_t,
            covariates_df=covariates_df,
            batch_size=params['batch_size'],
            return_sparse=params.get('return_sparse', True),
            pval_threshold=params.get('pval_threshold', 1e-5)
        )
        
        # Save results
        trans_file = os.path.join(results_dir, f"{qtl_type}_trans.trans_qtl.txt.gz")
        if trans_df is not None and len(trans_df) > 0:
            # Apply FDR correction using tensorQTL's calculate_qvalues
            if 'pval' in trans_df.columns:
                fdr = calculate_qvalues(trans_df['pval'])
                trans_df['fdr'] = fdr
                significant_count = (fdr < params['fdr_threshold']).sum()
            else:
                significant_count = len(trans_df)
            
            trans_df.to_csv(trans_file, sep='\t', compression='gzip')
            logger.info(f"✅ Saved {len(trans_df)} trans associations")
        else:
            significant_count = 0
            empty_df = pd.DataFrame(columns=['phenotype_id', 'variant_id', 'pval', 'beta', 'se'])
            empty_df.to_csv(trans_file, sep='\t', compression='gzip', index=False)
            logger.warning(f"⚠️ No significant trans associations found")
        
        logger.info(f"✅ {qtl_type} trans: Found {significant_count} significant associations")
        
        # Clean up GPU memory if used
        if params['use_gpu'] and TENSORQTL_AVAILABLE and 'torch' in sys.modules:
            torch.cuda.empty_cache()
        
        return {
            'result_file': trans_file,
            'significant_count': significant_count,
            'status': 'completed',
            'hardware_used': 'GPU' if params['use_gpu'] else 'CPU'
        }
        
    except Exception as e:
        logger.error(f"❌ trans-QTL analysis failed for {qtl_type}: {e}")
        return {
            'result_file': "",
            'significant_count': 0,
            'status': 'failed',
            'error': str(e)
        }

def count_significant_associations(results_dir, prefix, fdr_threshold=0.05):
    """Count significant associations from tensorQTL output"""
    result_file = os.path.join(results_dir, f"{prefix}.cis_qtl.txt.gz")
    
    if not os.path.exists(result_file):
        return 0
    
    try:
        df = pd.read_csv(result_file, sep='\t')
        
        if df.empty:
            return 0
        
        if 'qval' in df.columns:
            significant_count = len(df[df['qval'] < fdr_threshold])
        elif 'pval_perm' in df.columns:
            significant_count = len(df[df['pval_perm'] < fdr_threshold])
        elif 'pval_nominal' in df.columns:
            bonferroni_threshold = fdr_threshold / len(df)
            significant_count = len(df[df['pval_nominal'] < bonferroni_threshold])
        else:
            significant_count = len(df)
            logger.warning("No FDR column found, counting all results")
        
        return significant_count
        
    except Exception as e:
        logger.warning(f"⚠️ Could not count significant associations: {e}")
        return 0

def run_command(cmd, description, config, check=True):
    """Run shell command with error handling"""
    logger.info(f"Executing: {description}")
    
    timeout = config.get('large_data', {}).get('command_timeout', 7200)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True,
            executable='/bin/bash',
            timeout=timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
        else:
            logger.warning(f"⚠️ {description} completed with exit code {result.returncode}")
            
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        if check:
            raise RuntimeError(f"Command failed: {description}") from e
        return e
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after {timeout} seconds")
        if check:
            raise RuntimeError(f"Command timed out: {description}")
        return None
    except Exception as e:
        logger.error(f"❌ {description} failed with unexpected error: {e}")
        if check:
            raise
        return None

# Backward compatibility functions - ALL MISSING FUNCTIONS RESTORED
def apply_normalization(pheno_df, config, qtl_type, results_dir):
    """Apply normalization - Compatibility wrapper"""
    processor = PhenotypeProcessor(config, results_dir)
    return processor._apply_normalization(pheno_df, qtl_type)

def filter_low_expressed_features(pheno_df, config, qtl_type):
    """Filter lowly expressed features - Compatibility wrapper"""
    processor = PhenotypeProcessor(config, ".")
    return processor._apply_qc_filters(pheno_df, qtl_type)

def prepare_phenotype_data(config, qtl_type, results_dir):
    """Prepare phenotype data for tensorQTL - Compatibility wrapper"""
    processor = PhenotypeProcessor(config, results_dir)
    return processor.prepare_phenotype_data(qtl_type)

# Keep all your original normalization function definitions for direct calls
def apply_vst_normalization(pheno_df, config, results_dir):
    """Apply VST normalization using DESeq2"""
    processor = PhenotypeProcessor(config, results_dir)
    return processor._apply_vst_normalization(pheno_df, 'expression')

def apply_log2_normalization(pheno_df, config, qtl_type):
    """Apply log2 transformation with pseudocount"""
    processor = PhenotypeProcessor(config, ".")
    return processor._apply_log2_normalization(pheno_df, qtl_type)

def apply_quantile_normalization(pheno_df):
    """Apply quantile normalization"""
    processor = PhenotypeProcessor({}, ".")
    return processor._apply_quantile_normalization(pheno_df, 'generic')

def apply_zscore_normalization(pheno_df):
    """Apply z-score normalization per feature"""
    processor = PhenotypeProcessor({}, ".")
    return processor._apply_zscore_normalization(pheno_df, 'generic')

def apply_arcsinh_normalization(pheno_df, config, qtl_type):
    """Apply arcsinh transformation"""
    processor = PhenotypeProcessor(config, ".")
    return processor._apply_arcsinh_normalization(pheno_df, qtl_type)

def apply_tpm_normalization(pheno_df, config):
    """Apply TPM normalization"""
    processor = PhenotypeProcessor(config, ".")
    return processor._apply_tpm_normalization(pheno_df, 'expression')

def create_phenotype_positions(feature_ids, annot_df, qtl_type):
    """Legacy function for backward compatibility"""
    processor = PhenotypeProcessor({}, ".")
    return processor._create_phenotype_positions(feature_ids, qtl_type)

def map_qtl_type_to_config_key(qtl_type):
    """Map QTL analysis types to config file keys - Enhanced version"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing',
        'expression': 'expression',
        'protein': 'protein',
        'splicing': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

# Additional utility functions for modular pipeline - MISSING FUNCTION RESTORED
def process_expression_data(config, results_dir=None):
    """Process expression data for modular pipeline - FIXED: Added results_dir parameter"""
    if results_dir is None:
        # Fallback to config if results_dir not provided
        results_dir = config.get('results_dir', 'results')
        logger.warning(f"results_dir not provided, using config value: {results_dir}")
    
    return prepare_phenotype_data(config, 'eqtl', results_dir)

def run_qtl_mapping(config, genotype_file, qtl_type, results_dir, analysis_mode='cis'):
    """Unified QTL mapping function for modular pipeline"""
    if analysis_mode == 'cis':
        return run_cis_analysis(config, genotype_file, qtl_type, results_dir)
    elif analysis_mode == 'trans':
        return run_trans_analysis(config, genotype_file, qtl_type, results_dir)
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")

# Performance monitoring utilities
def monitor_performance():
    """Monitor current performance metrics"""
    gpu_memory = None
    if TENSORQTL_AVAILABLE and 'torch' in sys.modules:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    logger.info(f"📊 Performance: CPU {cpu_percent}%, RAM {memory.percent}%")
    if gpu_memory:
        logger.info(f"📊 GPU memory: {gpu_memory:.2f} GB")

if __name__ == "__main__":
    """Standalone QTL analysis script"""
    import yaml
    import sys
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/config.yaml"
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        genotype_file = prepare_genotypes(config, config['results_dir'])
        qtl_types = config['analysis']['qtl_types']
        
        if isinstance(qtl_types, str) and qtl_types != 'all':
            qtl_types = [qtl_types]
        elif qtl_types == 'all':
            qtl_types = ['eqtl']
        
        for qtl_type in qtl_types:
            if config['analysis']['qtl_mode'] in ['cis', 'both']:
                run_cis_analysis(config, genotype_file, qtl_type, config['results_dir'])
            if config['analysis']['qtl_mode'] in ['trans', 'both']:
                run_trans_analysis(config, genotype_file, qtl_type, config['results_dir'])
        
        logger.info("✅ QTL analysis completed successfully")
        
    except Exception as e:
        logger.error(f"❌ QTL analysis failed: {e}")
        sys.exit(1)