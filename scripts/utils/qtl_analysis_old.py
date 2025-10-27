#!/usr/bin/env python3
"""
Enhanced QTL analysis utilities with tensorQTL-specific capabilities - Production Version
Complete pipeline for cis/trans QTL analysis using tensorQTL with robust error handling
and optimized CPU/GPU utilization

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with comprehensive CPU/GPU optimization, memory management, performance tuning, and dynamic data handling.
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

# Import tensorQTL with comprehensive error handling and better diagnostics
TENSORQTL_AVAILABLE = False
CALCULATE_QVALUES_AVAILABLE = False
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

    # Now try to import tensorqtl
    try:
        import tensorqtl
        from tensorqtl import genotypeio, cis, trans
        logger.info(f"✅ tensorQTL successfully imported (version: {tensorqtl.__version__})")
        TENSORQTL_AVAILABLE = True
        
        # Enhanced calculate_qvalues import with multiple fallbacks
        CALCULATE_QVALUES_AVAILABLE = False
        calculate_qvalues = None
        
        # Try multiple import strategies for calculate_qvalues
        import_strategies = [
            # Strategy 1: Direct import from tensorqtl
            lambda: __import__('tensorqtl').calculate_qvalues,
            # Strategy 2: From tensorqtl.utils
            lambda: __import__('tensorqtl.utils', fromlist=['calculate_qvalues']).calculate_qvalues,
            # Strategy 3: Try to find it in the module
            lambda: getattr(tensorqtl, 'calculate_qvalues', None),
        ]
        
        for i, strategy in enumerate(import_strategies):
            try:
                calculate_qvalues = strategy()
                if calculate_qvalues is not None:
                    CALCULATE_QVALUES_AVAILABLE = True
                    logger.info(f"✅ calculate_qvalues found using strategy {i+1}")
                    break
            except (ImportError, AttributeError) as e:
                logger.debug(f"❌ Import strategy {i+1} failed: {e}")
                continue
        
        if not CALCULATE_QVALUES_AVAILABLE:
            # Try statsmodels as final fallback
            try:
                from statsmodels.stats.multitest import multipletests
                logger.info("✅ Using statsmodels for FDR calculation")
                # We'll create a wrapper function later
            except ImportError:
                logger.warning("❌ Neither tensorqtl.calculate_qvalues nor statsmodels available")
        
    except ImportError as e:
        TENSORQTL_AVAILABLE = False
        TENSORQTL_IMPORT_ERROR = str(e)
        logger.error(f"❌ tensorQTL import failed: {e}")
        logger.error("Please install tensorqtl: pip install tensorqtl")
        logger.error("If using GPU, try: pip install tensorqtl[gpu]")

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
    """Enhanced handler for dynamic QTL data alignment and processing"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data_handling', {})
        
    def align_qtl_data(self, genotype_samples, phenotype_df, covariate_df=None):
        """Align genotype, phenotype, and covariate data with comprehensive validation"""
        logger.info("🔍 Aligning QTL data across all datasets...")
        
        # Convert to sets for fast operations
        genotype_set = set(genotype_samples)
        phenotype_set = set(phenotype_df.columns)
        
        if covariate_df is not None and not covariate_df.empty:
            covariate_set = set(covariate_df.columns)
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
        logger.info(f"✅ Data alignment completed: {len(common_samples)} common samples "
                   f"(genotype: {len(genotype_set)}, phenotype: {len(phenotype_set)}, "
                   f"covariates: {len(covariate_set) if covariate_set else 0})")
        
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
            logger.warning(f"⚠️ Found {non_numeric.shape[1]} non-numeric columns in {qtl_type} data, attempting conversion")
            try:
                # Convert to numeric, coercing errors to NaN
                phenotype_df = phenotype_df.apply(pd.to_numeric, errors='coerce')
                # Drop columns that couldn't be converted
                phenotype_df = phenotype_df.dropna(axis=1, how='all')
            except Exception as e:
                logger.error(f"❌ Could not convert non-numeric columns to numeric: {e}")
                raise
        
        # Check for constant features
        constant_features = phenotype_df.std(axis=1) == 0
        if constant_features.any():
            logger.warning(f"⚠️ Removing {constant_features.sum()} constant features from {qtl_type} data")
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
    """Optimize hardware utilization for tensorQTL analysis with your existing config"""
    
    def __init__(self, config):
        self.config = config
        self.performance_config = config.get('performance', {})
        self.tensorqtl_config = config.get('tensorqtl', {})
        
    def setup_hardware(self):
        """Setup optimal hardware configuration for tensorQTL based on your config"""
        device_info = self.detect_available_devices()
        
        # Use GPU if available and enabled in config, otherwise use optimized CPU
        use_gpu = self.tensorqtl_config.get('use_gpu', False) and device_info['gpu_available']
        
        if use_gpu:
            device = self.setup_gpu()
        else:
            device = self.setup_cpu_optimized()
        
        # Set memory optimization based on available resources
        self.setup_memory_optimization(device_info)
        
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
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_info['gpu_available'] = True
                device_info['gpu_names'] = ['Apple MPS']
        
        logger.info(f"🖥️  Hardware detected: {device_info['cpu_cores']} CPU cores, "
                   f"{device_info['memory_gb']:.1f} GB RAM, "
                   f"{device_info['gpu_count']} GPUs available")
        
        if device_info['gpu_available']:
            logger.info(f"🎮 GPUs: {', '.join(device_info['gpu_names'])}")
        
        return device_info
    
    def setup_gpu(self):
        """Setup GPU configuration for optimal performance"""
        if not TENSORQTL_AVAILABLE or 'torch' not in sys.modules:
            logger.warning("⚠️ GPU setup skipped: tensorQTL or PyTorch not available")
            return self.setup_cpu_optimized()
            
        try:
            # Set default tensor type to CUDA
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            
            # Configure CUDA optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🎯 GPU acceleration enabled for tensorQTL")
            return torch.device('cuda')
            
        except Exception as e:
            logger.warning(f"⚠️ GPU setup failed: {e}, falling back to CPU")
            return self.setup_cpu_optimized()
    
    def setup_cpu_optimized(self):
        """Setup CPU configuration for optimal multi-core performance"""
        try:
            # Set CPU threads for optimal performance - use your config or auto-detect
            num_threads = self.performance_config.get('num_threads', min(16, os.cpu_count()))
            
            # Only set torch threads if torch is available AND hasn't been initialized yet
            if 'torch' in sys.modules and not self._torch_already_initialized():
                try:
                    torch.set_num_threads(num_threads)
                    torch.set_num_interop_threads(num_threads)
                    logger.info(f"🔢 Successfully set torch threads: {num_threads}")
                except RuntimeError as e:
                    if "after parallel work has started" in str(e):
                        logger.warning("⚠️ Torch threads already initialized, using current configuration")
                    else:
                        raise e
            
            # Set environment variables for OpenMP (safe to do anytime)
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            
            logger.info(f"🔢 Using {num_threads} CPU threads for tensor operations")
            
            return None  # Return None for CPU device
            
        except Exception as e:
            logger.warning(f"⚠️ CPU optimization failed: {e}")
            return None
    
    def _torch_already_initialized(self):
        """Check if torch has already been initialized (to avoid runtime errors)"""
        if 'torch' not in sys.modules:
            return False
        try:
            # Try to get current thread settings - if this fails, torch isn't fully initialized
            torch.get_num_threads()
            return True
        except:
            return False
    
    def setup_memory_optimization(self, device_info):
        """Setup memory optimization parameters based on available resources"""
        memory_gb = device_info['memory_gb']
        
        # Adaptive batch sizing based on available memory and your config
        config_batch_size = self.tensorqtl_config.get('batch_size', 20000)
        config_chunk_size = self.tensorqtl_config.get('chunk_size', 200)
        
        # Adjust based on available memory
        if memory_gb > 64:
            # High memory system - can use larger batches
            optimized_batch_size = min(config_batch_size * 2, 50000)
            optimized_chunk_size = min(config_chunk_size * 2, 500)
        elif memory_gb > 32:
            # Medium memory system - use config values
            optimized_batch_size = config_batch_size
            optimized_chunk_size = config_chunk_size
        else:
            # Low memory system - reduce batch size
            optimized_batch_size = max(config_batch_size // 2, 5000)
            optimized_chunk_size = max(config_chunk_size // 2, 100)
        
        logger.info(f"💾 Memory optimization: {memory_gb:.1f} GB available, "
                   f"batch_size={optimized_batch_size}, chunk_size={optimized_chunk_size}")

class QTLConfig:
    """Enhanced QTL configuration management with robust error handling and hardware optimization"""
    def __init__(self, config):
        self.config = config
        self.qtl_config = config.get('qtl', {})
        self.tensorqtl_config = config.get('tensorqtl', {})
        self.normalization_config = config.get('normalization', {})
        self.performance_config = config.get('performance', {})
        self.large_data_config = config.get('large_data', {})
        
        # Initialize hardware optimizer
        self.hardware_optimizer = HardwareOptimizer(config)
        
    def get_analysis_params(self, analysis_type):
        """Get analysis parameters with comprehensive fallbacks and hardware optimization"""
        # Setup hardware first
        device, device_info = self.hardware_optimizer.setup_hardware()
        
        # Use your existing config values with hardware optimization
        base_params = {
            'cis_window': self.tensorqtl_config.get('cis_window', 1000000),
            'maf_threshold': self.tensorqtl_config.get('maf_threshold', 0.05),
            'min_maf': self.tensorqtl_config.get('min_maf', 0.01),
            'fdr_threshold': self.tensorqtl_config.get('fdr_threshold', 0.05),
            'num_permutations': self.tensorqtl_config.get('num_permutations', 1000),
            'batch_size': self.tensorqtl_config.get('batch_size', 20000),
            'chunk_size': self.tensorqtl_config.get('chunk_size', 200),
            'seed': self.tensorqtl_config.get('seed', 12345),
            'run_permutations': self.tensorqtl_config.get('run_permutations', True),
            'write_stats': self.tensorqtl_config.get('write_stats', True),
            'write_top_results': self.tensorqtl_config.get('write_top_results', True),
            'run_eigenmt': self.tensorqtl_config.get('run_eigenmt', False),
            'output_format': self.tensorqtl_config.get('output_format', 'parquet'),
            'write_sparse': self.tensorqtl_config.get('write_sparse', True),
            'impute_missing': self.tensorqtl_config.get('impute_missing', True),
            'center_features': self.tensorqtl_config.get('center_features', True),
            'standardize_features': self.tensorqtl_config.get('standardize_features', True),
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
            # GPU-optimized parameters
            logger.info("🚀 Using GPU-optimized parameters")
            base_params.update({
                'batch_size': base_params['batch_size'] * 2,  # Larger batches for GPU
                'chunk_size': min(base_params['chunk_size'] * 2, 500)
            })
        else:
            # CPU-optimized parameters
            logger.info("🔢 Using CPU-optimized parameters")
            base_params.update({
                'num_threads': self.performance_config.get('num_threads', min(16, os.cpu_count()))
            })
        
        # Large data handling
        if self.large_data_config.get('process_by_chromosome', False):
            base_params['chromosome_batch_size'] = self.large_data_config.get('max_concurrent_chromosomes', 2)
        
        return base_params
    
    def validate_parameters(self, analysis_type):
        """Validate analysis parameters"""
        params = self.get_analysis_params(analysis_type)
        errors = []
        
        if params['cis_window'] <= 0:
            errors.append("cis_window must be positive")
        if not (0 < params['maf_threshold'] <= 0.5):
            errors.append("maf_threshold must be between 0 and 0.5")
        if not (0 < params['fdr_threshold'] <= 1):
            errors.append("fdr_threshold must be between 0 and 1")
        if params['num_permutations'] < 10:
            errors.append("num_permutations should be at least 10 for meaningful results")
        if params['batch_size'] <= 0:
            errors.append("batch_size must be positive")
        if params['chunk_size'] <= 0:
            errors.append("chunk_size must be positive")
        
        if errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
        
        return True

class PhenotypeProcessor:
    """Enhanced phenotype data processing with robust error handling and comprehensive normalization"""
    
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        self.qc_config = config.get('qc', {})
        self.normalization_config = config.get('normalization', {})
        self.performance_config = config.get('performance', {})
        self.data_handler = DynamicDataHandler(config)
        
    def prepare_phenotype_data(self, qtl_type, genotype_samples=None):
        """Prepare phenotype data with comprehensive processing and error handling"""
        logger.info(f"🔧 Preparing {qtl_type} phenotype data with dynamic handling...")
        
        try:
            # Get phenotype file path using proper mapping
            config_key = self._map_qtl_type_to_config_key(qtl_type)
            pheno_file = self.config['input_files'].get(config_key)
            
            if not pheno_file:
                raise FileNotFoundError(f"Phenotype file not configured for {qtl_type} (key: {config_key})")
            if not os.path.exists(pheno_file):
                raise FileNotFoundError(f"Phenotype file not found for {qtl_type}: {pheno_file}")
            
            # Load phenotype data
            pheno_df = self._load_phenotype_data(pheno_file, qtl_type)
            original_feature_count = pheno_df.shape[0]
            original_sample_count = pheno_df.shape[1]
            logger.info(f"📊 Loaded {qtl_type} data: {original_feature_count} features, {original_sample_count} samples")
            
            # Load covariates if available
            covariates_file = self.config['input_files'].get('covariates')
            if covariates_file and os.path.exists(covariates_file):
                cov_df = self._load_covariate_data(covariates_file)
                logger.info(f"📊 Loaded covariates: {cov_df.shape[0]} covariates, {cov_df.shape[1]} samples")
            else:
                cov_df = pd.DataFrame()
                logger.info("ℹ️ No covariate file found or specified")
            
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
            
            # Apply normalization - STRICT USER-DEFINED METHOD (NO FALLBACK)
            if self.qc_config.get('normalize', True):
                normalized_df = self._apply_normalization_strict(pheno_df, qtl_type)
                normalization_method = self.normalization_config.get(qtl_type, {}).get('method', 'unknown')
                logger.info(f"🔄 Applied {normalization_method} normalization for {qtl_type}")
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
            logger.info(f"✅ Prepared {qtl_type} data: {final_feature_count}/{original_feature_count} features retained, "
                       f"{len(common_samples)} samples, {cov_df.shape[0] if not cov_df.empty else 0} covariates")
            
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
        
        # Log detailed statistics about the filtering
        if removed_count > 0:
            low_expr_stats = low_expression_percentage.describe()
            logger.info(f"📊 Low expression statistics - "
                       f"Mean: {low_expr_stats['mean']:.1f}%, "
                       f"Median: {low_expr_stats['50%']:.1f}%, "
                       f"Max: {low_expr_stats['max']:.1f}%")
        
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
        """Load phenotype data with robust error handling and format detection"""
        try:
            # Try different encodings and separators
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(file_path, sep=sep, index_col=0)
                    if not df.empty:
                        logger.info(f"✅ Successfully loaded {qtl_type} data with separator '{sep}'")
                        # Ensure all data is numeric
                        df = df.apply(pd.to_numeric, errors='coerce')
                        return df
                except:
                    continue
            
            # If standard separators fail, try with header detection
            df = pd.read_csv(file_path, index_col=0)
            if df.empty:
                raise ValueError(f"Could not read {qtl_type} file with any standard separator")
            
            # Ensure all data is numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading phenotype data from {file_path}: {e}")
            raise
    
    def _load_covariate_data(self, covariate_file):
        """Load covariate data with dynamic format handling"""
        try:
            # Try different separators
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(covariate_file, sep=sep, index_col=0)
                    if not df.empty:
                        logger.info(f"✅ Successfully loaded covariate data with separator '{sep}'")
                        # Ensure all data is numeric
                        df = df.apply(pd.to_numeric, errors='coerce')
                        return df
                except:
                    continue
            
            # Fallback
            df = pd.read_csv(covariate_file, index_col=0)
            # Ensure all data is numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading covariate data: {e}")
            return pd.DataFrame()
    
    def _apply_qc_filters(self, pheno_df, qtl_type):
        """Apply comprehensive quality control filters with FIXED variable initialization"""
        original_count = pheno_df.shape[0]
        filtered_df = pheno_df.copy()
        
        # Initialize all removal counters to avoid uninitialized variable errors
        constant_removed = 0
        missing_removed = 0
        low_expression_removed = 0
        low_variance_removed = 0
        
        # Remove constant features
        constant_threshold = self.qc_config.get('constant_threshold', 0.95)
        non_constant_mask = (filtered_df.nunique(axis=1) / filtered_df.shape[1]) > (1 - constant_threshold)
        filtered_df_after_constant = filtered_df[non_constant_mask]
        constant_removed = original_count - filtered_df_after_constant.shape[0]
        filtered_df = filtered_df_after_constant
        
        # Remove features with too many missing values
        missing_threshold = self.qc_config.get('missing_value_threshold', 0.2)
        low_missing_mask = (filtered_df.isna().sum(axis=1) / filtered_df.shape[1]) < missing_threshold
        filtered_df_after_missing = filtered_df[low_missing_mask]
        missing_removed = filtered_df.shape[0] - filtered_df_after_missing.shape[0]
        filtered_df = filtered_df_after_missing
        
        # QTL-type specific filtering
        if qtl_type == 'eqtl':
            threshold = self.qc_config.get('expression_threshold', 0.1)
            mean_expression = filtered_df.mean(axis=1)
            expressed_mask = mean_expression > threshold
            filtered_df_after_expr = filtered_df[expressed_mask]
            low_expression_removed = filtered_df.shape[0] - filtered_df_after_expr.shape[0]
            filtered_df = filtered_df_after_expr
        elif qtl_type in ['pqtl', 'sqtl']:
            # Filter based on variance
            variance_threshold = filtered_df.var(axis=1).quantile(0.1)
            high_variance_mask = filtered_df.var(axis=1) > variance_threshold
            filtered_df_after_var = filtered_df[high_variance_mask]
            low_variance_removed = filtered_df.shape[0] - filtered_df_after_var.shape[0]
            filtered_df = filtered_df_after_var
        
        filtered_count = filtered_df.shape[0]
        logger.info(f"🔧 QC filtering: {filtered_count}/{original_count} features retained "
                   f"(constant: {constant_removed}, missing: {missing_removed}, "
                   f"low_expr: {low_expression_removed}, low_var: {low_variance_removed})")
        
        return filtered_df
    
    def _apply_normalization_strict(self, pheno_df, qtl_type):
        """
        STRICT normalization based on user-defined method only - NO FALLBACK
        """
        norm_config = self.normalization_config.get(qtl_type, {})
        method = norm_config.get('method', 'log2')
        
        logger.info(f"🔄 Applying STRICT {method} normalization for {qtl_type} (no fallback)...")
        
        normalization_methods = {
            'vst': self._apply_vst_normalization_strict,
            'log2': self._apply_log2_normalization_strict,
            'quantile': self._apply_quantile_normalization_strict,
            'zscore': self._apply_zscore_normalization_strict,
            'arcsinh': self._apply_arcsinh_normalization_strict,
            'tpm': self._apply_tpm_normalization_strict,
            'raw': lambda x, y: x  # No normalization
        }
        
        if method in normalization_methods:
            return normalization_methods[method](pheno_df, qtl_type)
        else:
            raise ValueError(f"Unknown normalization method '{method}'. Available methods: {list(normalization_methods.keys())}")
    
    def _apply_vst_normalization_strict(self, pheno_df, qtl_type):
        """
        STRICT VST normalization using DESeq2 - NO FALLBACK
        Adds +1 and rounds counts before normalization as required
        """
        if qtl_type != 'eqtl':
            logger.warning("VST normalization is typically for expression data")
        
        # ADD +1 AND ROUND COUNTS BEFORE VST NORMALIZATION
        logger.info("🔢 Adding +1 and rounding counts for VST normalization...")
        pheno_df = pheno_df + 1
        pheno_df = np.round(pheno_df).astype(int)
        logger.info(f"✅ Added +1 and rounded counts for VST normalization. Data shape: {pheno_df.shape}")
        
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
                raise RuntimeError(f"DESeq2 VST normalization failed with exit code {result.returncode}")
            
            # Load normalized data
            if os.path.exists(temp_output_path):
                vst_df = pd.read_csv(temp_output_path, sep='\t', index_col=0)
                # Ensure all data is numeric
                vst_df = vst_df.apply(pd.to_numeric, errors='coerce')
                logger.info("✅ VST normalization completed successfully")
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
    
    def _apply_log2_normalization_strict(self, pheno_df, qtl_type):
        """Apply log2 transformation with comprehensive options - NO FALLBACK"""
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
        
        # Apply log2 transformation
        normalized_df = np.log2(pheno_df + pseudocount)
        logger.info(f"✅ Applied log2 transformation (pseudocount={pseudocount})")
        
        return normalized_df
    
    def _apply_quantile_normalization_strict(self, pheno_df, qtl_type):
        """Apply quantile normalization - NO FALLBACK"""
        try:
            from sklearn.preprocessing import quantile_transform
            
            # Transpose for sample-wise normalization
            normalized_array = quantile_transform(pheno_df.T, n_quantiles=min(1000, pheno_df.shape[0]))
            normalized_df = pd.DataFrame(normalized_array.T, index=pheno_df.index, columns=pheno_df.columns)
            
            logger.info("✅ Quantile normalization completed")
            return normalized_df
        except ImportError:
            raise ImportError("scikit-learn not available for quantile normalization")
    
    def _apply_zscore_normalization_strict(self, pheno_df, qtl_type):
        """Apply z-score normalization per feature with constant feature handling - NO FALLBACK"""
        normalized_df = (pheno_df - pheno_df.mean(axis=1).values.reshape(-1, 1)) 
        normalized_df = normalized_df / pheno_df.std(axis=1).values.reshape(-1, 1)
        
        # Handle constant features (std=0)
        constant_mask = pheno_df.std(axis=1) == 0
        if constant_mask.any():
            normalized_df.loc[constant_mask] = 0
            logger.warning(f"⚠️ Found {constant_mask.sum()} constant features, setting z-score to 0")
        
        logger.info("✅ Z-score normalization completed")
        return normalized_df
    
    def _apply_arcsinh_normalization_strict(self, pheno_df, qtl_type):
        """Apply arcsinh transformation - NO FALLBACK"""
        norm_config = self.normalization_config.get(qtl_type, {})
        cofactor = norm_config.get('arcsinh_cofactor', 1)
        
        normalized_df = np.arcsinh(pheno_df / cofactor)
        logger.info(f"✅ Arcsinh transformation completed (cofactor={cofactor})")
        return normalized_df
    
    def _apply_tpm_normalization_strict(self, pheno_df, qtl_type):
        """Apply TPM-like normalization - NO FALLBACK"""
        # Simplified TPM calculation (without gene lengths)
        rpm_df = pheno_df.div(pheno_df.sum(axis=0)) * 1e6
        logger.info("✅ TPM-like normalization completed")
        return rpm_df
    
    # Keep original normalization methods for backward compatibility
    def _apply_normalization(self, pheno_df, qtl_type):
        """Original normalization method with fallbacks - kept for backward compatibility"""
        logger.warning("Using legacy normalization method with fallbacks. Consider using _apply_normalization_strict instead.")
        return self._apply_normalization_strict(pheno_df, qtl_type)
    
    def _apply_vst_normalization(self, pheno_df, qtl_type):
        """Original VST method - kept for backward compatibility"""
        logger.warning("Using legacy VST normalization method. Consider using _apply_vst_normalization_strict instead.")
        return self._apply_vst_normalization_strict(pheno_df, qtl_type)
    
    def _apply_log2_normalization(self, pheno_df, qtl_type):
        """Original log2 method - kept for backward compatibility"""
        logger.warning("Using legacy log2 normalization method. Consider using _apply_log2_normalization_strict instead.")
        return self._apply_log2_normalization_strict(pheno_df, qtl_type)
    
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
        """Save processed phenotype data with comprehensive output options"""
        # Ensure all data is numeric before saving
        logger.info("🔧 Ensuring all data is numeric before saving...")
        
        # Clean and convert normalized_df
        normalized_df = normalized_df.apply(pd.to_numeric, errors='coerce')
        
        # Clean and convert covariate_df if it exists
        if not covariate_df.empty:
            covariate_df = covariate_df.apply(pd.to_numeric, errors='coerce')
        
        # Save phenotype matrix based on config format
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
                logger.info(f"💾 Saved covariates: {cov_file}")
            except Exception as e:
                logger.warning(f"⚠️ Covariate parquet save failed, using CSV: {e}")
                cov_file = os.path.join(self.results_dir, f"{qtl_type}_covariates.txt.gz")
                covariate_df.to_csv(cov_file, sep='\t', compression='gzip')
                logger.info(f"💾 Saved covariates: {cov_file}")
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
        """Create phenotype positions DataFrame with robust annotation handling"""
        annotation_file = self.config['input_files']['annotations']
        
        try:
            # Try different comment characters for annotation file
            try:
                annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
            except:
                annot_df = pd.read_csv(annotation_file, sep='\t')
        except Exception as e:
            logger.warning(f"⚠️ Could not read annotation file: {e}, creating default positions")
            annot_df = pd.DataFrame()
        
        positions_data = []
        missing_annotations = 0
        
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
                    missing_annotations += 1
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
        
        if missing_annotations > 0:
            logger.warning(f"⚠️ Could not find annotations for {missing_annotations} features, using defaults")
        
        positions_df = pd.DataFrame(positions_data)
        positions_df = positions_df.set_index('phenotype_id')
        return positions_df

class GenotypeLoader:
    """Enhanced genotype data loading with memory optimization and comprehensive error handling"""
    
    def __init__(self, config):
        self.config = config
        self.genotype_processing_config = config.get('genotype_processing', {})
        self.performance_config = config.get('performance', {})
    
    def load_genotypes(self, genotype_file):
        """Load genotype data with comprehensive error handling and version compatibility"""
        logger.info("🔧 Loading genotype data for tensorQTL...")
        
        if not TENSORQTL_AVAILABLE:
            error_msg = "tensorQTL is not available. "
            if TENSORQTL_IMPORT_ERROR:
                error_msg += f"Import error: {TENSORQTL_IMPORT_ERROR}. "
            error_msg += "Please install: pip install tensorqtl"
            if TENSORQTL_IMPORT_ERROR and "torch" in TENSORQTL_IMPORT_ERROR.lower():
                error_msg += " and make sure PyTorch is installed: pip install torch"
            raise ImportError(error_msg)
        
        try:
            if genotype_file.endswith('.bed'):
                # Load PLINK data
                plink_prefix = genotype_file.replace('.bed', '')
                
                # Hardware optimization for tensorQTL
                hardware_optimizer = HardwareOptimizer(self.config)
                device, device_info = hardware_optimizer.setup_hardware()
                
                # FIXED: Handle different tensorQTL versions
                pr = genotypeio.read_plink(plink_prefix)
                
                # Check if it's a tuple (older tensorQTL versions) or object (newer versions)
                if isinstance(pr, tuple):
                    # Older tensorQTL versions return tuple: (genotypes, variants, samples)
                    logger.info("📦 Detected older tensorQTL version (tuple return format)")
                    genotypes, variants, samples = pr
                    # Create a mock object with the required attributes for compatibility
                    class GenotypeContainer:
                        def __init__(self, genotypes, variants, samples):
                            self.genotypes = genotypes
                            self.variants = variants
                            self.samples = samples
                    
                    pr_container = GenotypeContainer(genotypes, variants, samples)
                    logger.info(f"✅ Loaded PLINK data (old format): {genotypes.shape[0]} variants, {genotypes.shape[1]} samples")
                    return pr_container
                else:
                    # Newer tensorQTL versions return an object with attributes
                    logger.info(f"✅ Loaded PLINK data (new format): {pr.genotypes.shape[0]} variants, {pr.genotypes.shape[1]} samples")
                    return pr
            else:
                raise ValueError(f"Unsupported genotype format: {genotype_file}. Use PLINK format for best performance.")
                
        except Exception as e:
            logger.error(f"❌ Error loading genotype data: {e}")
            raise
    
    def optimize_genotype_data(self, genotype_reader):
        """Optimize genotype data for analysis with comprehensive filtering - FIXED for version compatibility"""
        # FIXED: Handle both tuple format and object format
        if hasattr(genotype_reader, 'genotypes'):
            # Object format (newer tensorQTL)
            original_count = genotype_reader.genotypes.shape[0]
            genotypes_obj = genotype_reader.genotypes
        else:
            # This shouldn't happen with our fix above, but keep for safety
            raise AttributeError("Genotype reader doesn't have 'genotypes' attribute")
        
        # Apply MAF filtering
        maf_threshold = self.genotype_processing_config.get('min_maf', 0.01)
        if maf_threshold > 0:
            maf = genotypes_obj.maf()
            keep_variants = maf >= maf_threshold
            genotypes_obj = genotypes_obj[keep_variants]
            maf_filtered_count = genotypes_obj.shape[0]
            logger.info(f"🔧 MAF filtering: {maf_filtered_count}/{original_count} variants retained (MAF >= {maf_threshold})")
        
        # Apply call rate filtering if needed
        call_rate_threshold = self.genotype_processing_config.get('min_call_rate', 0.95)
        if call_rate_threshold < 1.0:
            call_rate = 1 - genotypes_obj.isnan().mean(axis=1)
            keep_variants = call_rate >= call_rate_threshold
            genotypes_obj = genotypes_obj[keep_variants]
            call_rate_filtered_count = genotypes_obj.shape[0]
            logger.info(f"🔧 Call rate filtering: {call_rate_filtered_count} variants retained (call rate >= {call_rate_threshold})")
        
        # Update the genotype reader with filtered data
        genotype_reader.genotypes = genotypes_obj
        
        final_count = genotype_reader.genotypes.shape[0]
        logger.info(f"🔧 Genotype optimization: {final_count}/{original_count} variants retained after filtering")
        
        return genotype_reader

def prepare_genotypes(config, results_dir):
    """Prepare genotype data optimized for tensorQTL with comprehensive error handling"""
    logger.info("🔧 Preparing genotype data for tensorQTL...")
    
    try:
        # Initialize genotype processor
        if GenotypeProcessor:
            processor = GenotypeProcessor(config)
            
            # Get input file path
            input_file = config['input_files']['genotypes']
            
            # Process genotypes - tensorQTL prefers PLINK format
            genotype_file = processor.process_genotypes(input_file, results_dir)
        else:
            # Fallback to direct processing
            genotype_file = config['input_files']['genotypes']
            logger.warning("Using direct genotype processing - GenotypeProcessor not available")
        
        # Ensure PLINK format for tensorQTL
        if genotype_file.endswith('.vcf.gz') or genotype_file.endswith('.vcf'):
            # Convert VCF to PLINK for tensorQTL
            plink_base = os.path.join(results_dir, "genotypes_plink")
            logger.info("🔄 Converting VCF to PLINK format for tensorQTL...")
            
            plink_path = config['paths'].get('plink', 'plink')
            plink_threads = config.get('genotype_processing', {}).get('plink_threads', 1)
            
            cmd = f"{plink_path} --vcf {genotype_file} --make-bed --out {plink_base} --threads {plink_threads}"
            run_command(cmd, "Converting VCF to PLINK", config)
            
            genotype_file = plink_base + ".bed"
            logger.info(f"✅ Converted VCF to PLINK format: {genotype_file}")
        
        logger.info(f"✅ Genotype preparation completed: {genotype_file}")
        return genotype_file
        
    except Exception as e:
        logger.error(f"❌ Genotype preparation failed: {e}")
        raise

def load_covariates(config, results_dir, qtl_type='eqtl'):
    """Load and prepare covariates for tensorQTL with enhanced processing and dynamic handling"""
    logger.info(f"🔧 Loading covariates for {qtl_type} with dynamic handling...")
    
    try:
        # Try to load pre-processed covariates first
        cov_file = os.path.join(results_dir, f"{qtl_type}_covariates.parquet")
        if os.path.exists(cov_file):
            try:
                cov_df = pd.read_parquet(cov_file)
                logger.info(f"✅ Loaded pre-processed covariates: {cov_df.shape[1]} samples, {cov_df.shape[0]} covariates")
                return cov_df.T  # Return samples x covariates for tensorQTL
            except Exception as e:
                logger.warning(f"⚠️ Could not read parquet covariates, trying CSV: {e}")
                cov_file = os.path.join(results_dir, f"{qtl_type}_covariates.txt.gz")
                if os.path.exists(cov_file):
                    cov_df = pd.read_csv(cov_file, sep='\t', index_col=0)
                    return cov_df.T
        
        # Fallback to original covariate file
        covariates_file = config['input_files'].get('covariates')
        if not covariates_file or not os.path.exists(covariates_file):
            logger.warning("⚠️ No covariate file found, proceeding without covariates")
            return None
        
        # Load with dynamic format handling
        data_handler = DynamicDataHandler(config)
        cov_df = data_handler._load_covariate_data(covariates_file)
        
        if cov_df.empty:
            logger.warning("⚠️ Covariate data is empty, proceeding without covariates")
            return None
        
        # Transpose for tensorQTL (samples x covariates)
        cov_df = cov_df.T
        
        # Remove constant covariates
        constant_covariates = cov_df.columns[cov_df.nunique() <= 1]
        if len(constant_covariates) > 0:
            cov_df = cov_df.drop(columns=constant_covariates)
            logger.warning(f"⚠️ Removed {len(constant_covariates)} constant covariates")
        
        # Check for missing values
        missing_count = cov_df.isna().sum().sum()
        if missing_count > 0:
            logger.warning(f"⚠️ Covariates contain {missing_count} missing values, they will be imputed")
            # Simple imputation with mean
            cov_df = cov_df.fillna(cov_df.mean())
        
        logger.info(f"✅ Loaded covariates: {cov_df.shape[1]} covariates, {cov_df.shape[0]} samples")
        return cov_df
        
    except Exception as e:
        logger.error(f"❌ Error loading covariates: {e}")
        return None

def calculate_fdr(pvalues, method='bh'):
    """Calculate FDR using available methods with enhanced fallbacks"""
    # Try tensorQTL calculate_qvalues first if available
    if CALCULATE_QVALUES_AVAILABLE and TENSORQTL_AVAILABLE:
        try:
            # Dynamically import calculate_qvalues if not already available
            if 'calculate_qvalues' not in globals():
                # Try multiple import strategies
                try:
                    from tensorqtl import calculate_qvalues
                except ImportError:
                    try:
                        from tensorqtl.utils import calculate_qvalues
                    except ImportError:
                        # Try to get it from the main tensorqtl module
                        import tensorqtl
                        calculate_qvalues = getattr(tensorqtl, 'calculate_qvalues', None)
            
            if calculate_qvalues is not None:
                return calculate_qvalues(pvalues)
            else:
                raise ImportError("calculate_qvalues not found in tensorqtl")
                
        except Exception as e:
            logger.warning(f"⚠️ tensorQTL calculate_qvalues failed, using statsmodels: {e}")
    
    # Use statsmodels as fallback
    try:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(pvalues, method=method)
        return fdr
    except ImportError:
        logger.error("❌ Neither tensorqtl.calculate_qvalues nor statsmodels available for FDR calculation")
        # Return raw pvalues if no FDR method is available
        return pvalues

def run_cis_analysis(config, genotype_file, qtl_type, results_dir):
    """Run cis-QTL analysis using tensorQTL with enhanced error handling and performance"""
    if not TENSORQTL_AVAILABLE:
        error_msg = "tensorQTL is not available. "
        if TENSORQTL_IMPORT_ERROR:
            error_msg += f"Import error: {TENSORQTL_IMPORT_ERROR}. "
        error_msg += "Please install it: pip install tensorqtl"
        if TENSORQTL_IMPORT_ERROR and "torch" in TENSORQTL_IMPORT_ERROR.lower():
            error_msg += " and make sure PyTorch is installed: pip install torch"
        raise ImportError(error_msg)
    
    logger.info(f"🔍 Running {qtl_type} cis-QTL analysis with tensorQTL...")
    
    try:
        # Initialize configuration with hardware optimization
        qtl_config = QTLConfig(config)
        qtl_config.validate_parameters('cis')
        params = qtl_config.get_analysis_params('cis')
        
        # Log hardware configuration
        device_info = params['device_info']
        if params['use_gpu']:
            logger.info(f"🎮 Using GPU for {qtl_type} cis-QTL analysis")
        else:
            logger.info(f"🔢 Using CPU for {qtl_type} cis-QTL analysis (threads: {params.get('num_threads', 'auto')})")
        
        # Load and optimize genotype data first to get samples
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        pr = genotype_loader.optimize_genotype_data(pr)
        
        # FIXED: Handle different genotype reader formats safely
        if hasattr(pr, 'samples'):
            genotype_samples = pr.samples
        elif hasattr(pr, 'genotypes') and hasattr(pr.genotypes, 'columns'):
            genotype_samples = pr.genotypes.columns.tolist()
        else:
            # Fallback: try to extract samples from the genotype object
            try:
                genotype_samples = list(pr.genotypes.columns)
            except:
                raise AttributeError("Cannot extract samples from genotype reader")
        
        # Prepare phenotype data with genotype samples for alignment
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type, genotype_samples)
        
        # Load covariates (will use pre-processed ones if available)
        covariates_df = load_covariates(config, results_dir, qtl_type)
        
        # Set output prefix
        output_prefix = os.path.join(results_dir, f"{qtl_type}_cis")
        
        # Run cis-QTL analysis with hardware optimization
        logger.info("🔬 Running tensorQTL cis mapping...")
        
        # Convert to tensorQTL-compatible format if needed
        phenotype_df_t = pheno_data['phenotype_df'].T  # tensorQTL expects samples x features
        phenotype_pos_df = pheno_data['phenotype_pos_df']
        
        # Map cis-QTLs with hardware optimization - using proper tensorQTL API
        cis_df = cis.map_cis(
            pr,  # Pass the genotype reader object directly
            phenotype_df_t, 
            phenotype_pos_df,
            covariates_df=covariates_df,
            window=params['cis_window'],
            seed=params['seed'],
            run_eigenmt=params['run_eigenmt']
        )
        
        # Run permutations for FDR estimation if requested
        if params['run_permutations']:
            logger.info("🔬 Running tensorQTL cis permutations...")
            
            cis_df = cis.map_cis(
                pr,
                phenotype_df_t,
                phenotype_pos_df,
                covariates_df=covariates_df,
                window=params['cis_window'],
                seed=params['seed'],
                run_eigenmt=params['run_eigenmt'],
                nperm=params['num_permutations']
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
            'nominals_file': result_file,
            'significant_count': significant_count,
            'status': 'completed',
            'params': params,
            'hardware_used': 'GPU' if params['use_gpu'] else 'CPU'
        }
        
    except Exception as e:
        logger.error(f"❌ cis-QTL analysis failed for {qtl_type}: {e}")
        return {
            'result_file': "",
            'nominals_file': "",
            'significant_count': 0,
            'status': 'failed',
            'error': str(e)
        }

def run_trans_analysis(config, genotype_file, qtl_type, results_dir):
    """Run trans-QTL analysis using tensorQTL with enhanced performance and memory optimization"""
    if not TENSORQTL_AVAILABLE:
        error_msg = "tensorQTL is not available. "
        if TENSORQTL_IMPORT_ERROR:
            error_msg += f"Import error: {TENSORQTL_IMPORT_ERROR}. "
        error_msg += "Please install it: pip install tensorqtl"
        if TENSORQTL_IMPORT_ERROR and "torch" in TENSORQTL_IMPORT_ERROR.lower():
            error_msg += " and make sure PyTorch is installed: pip install torch"
        raise ImportError(error_msg)
    
    logger.info(f"🔍 Running {qtl_type} trans-QTL analysis with tensorQTL...")
    
    try:
        # Initialize configuration with hardware optimization
        qtl_config = QTLConfig(config)
        params = qtl_config.get_analysis_params('trans')
        
        # Log hardware configuration
        device_info = params['device_info']
        if params['use_gpu']:
            logger.info(f"🎮 Using GPU for {qtl_type} trans-QTL analysis")
        else:
            logger.info(f"🔢 Using CPU for {qtl_type} trans-QTL analysis")
        
        # Load and optimize genotype data first to get samples
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        pr = genotype_loader.optimize_genotype_data(pr)
        
        # FIXED: Handle different genotype reader formats safely
        if hasattr(pr, 'samples'):
            genotype_samples = pr.samples
        elif hasattr(pr, 'genotypes') and hasattr(pr.genotypes, 'columns'):
            genotype_samples = pr.genotypes.columns.tolist()
        else:
            # Fallback: try to extract samples from the genotype object
            try:
                genotype_samples = list(pr.genotypes.columns)
            except:
                raise AttributeError("Cannot extract samples from genotype reader")
        
        # Prepare phenotype data with genotype samples for alignment
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type, genotype_samples)
        
        # Load covariates (will use pre-processed ones if available)
        covariates_df = load_covariates(config, results_dir, qtl_type)
        
        # Set output prefix
        output_prefix = os.path.join(results_dir, f"{qtl_type}_trans")
        
        # Run trans-QTL analysis with memory optimization
        logger.info("🔬 Running tensorQTL trans mapping...")
        
        # Convert to tensorQTL-compatible format
        phenotype_df_t = pheno_data['phenotype_df'].T  # tensorQTL expects samples x features
        
        # Use chunked processing for large datasets with hardware optimization
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
            # Apply FDR correction
            if 'pval' in trans_df.columns:
                fdr = calculate_fdr(trans_df['pval'])
                trans_df['fdr'] = fdr
                significant_count = (fdr < params['fdr_threshold']).sum()
            else:
                significant_count = len(trans_df)
            
            trans_df.to_csv(trans_file, sep='\t', compression='gzip')
            logger.info(f"✅ Saved {len(trans_df)} trans associations to {trans_file}")
        else:
            significant_count = 0
            # Create empty result file with proper columns
            empty_df = pd.DataFrame(columns=['phenotype_id', 'variant_id', 'pval', 'beta', 'se'])
            empty_df.to_csv(trans_file, sep='\t', compression='gzip', index=False)
            logger.warning(f"⚠️ No significant trans associations found for {qtl_type}")
        
        logger.info(f"✅ {qtl_type} trans: Found {significant_count} significant associations")
        
        # Clean up GPU memory if used
        if params['use_gpu'] and TENSORQTL_AVAILABLE and 'torch' in sys.modules:
            torch.cuda.empty_cache()
        
        return {
            'result_file': trans_file,
            'nominals_file': trans_file,
            'significant_count': significant_count,
            'status': 'completed',
            'params': params,
            'hardware_used': 'GPU' if params['use_gpu'] else 'CPU'
        }
        
    except Exception as e:
        logger.error(f"❌ trans-QTL analysis failed for {qtl_type}: {e}")
        return {
            'result_file': "",
            'nominals_file': "",
            'significant_count': 0,
            'status': 'failed',
            'error': str(e)
        }

def count_significant_associations(results_dir, prefix, fdr_threshold=0.05):
    """Count significant associations from tensorQTL output with enhanced parsing"""
    result_file = os.path.join(results_dir, f"{prefix}.cis_qtl.txt.gz")
    
    if not os.path.exists(result_file):
        return 0
    
    try:
        df = pd.read_csv(result_file, sep='\t')
        
        if df.empty:
            return 0
        
        # Check for different FDR/p-value columns
        if 'qval' in df.columns:
            significant_count = len(df[df['qval'] < fdr_threshold])
        elif 'pval_perm' in df.columns:
            significant_count = len(df[df['pval_perm'] < fdr_threshold])
        elif 'pval_nominal' in df.columns:
            # Use Bonferroni correction for nominal p-values
            bonferroni_threshold = fdr_threshold / len(df)
            significant_count = len(df[df['pval_nominal'] < bonferroni_threshold])
        else:
            # Count all results if no FDR column
            significant_count = len(df)
            logger.warning("No FDR column found in tensorQTL output, counting all results")
        
        return significant_count
        
    except Exception as e:
        logger.warning(f"⚠️ Could not count significant associations: {e}")
        return 0

def run_command(cmd, description, config, check=True):
    """Run shell command with comprehensive error handling and timeout"""
    logger.info(f"Executing: {description}")
    logger.debug(f"Command: {cmd}")
    
    # Set timeout from config
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
            if result.stderr:
                logger.warning(f"Stderr: {result.stderr[:500]}...")
            
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        logger.error(f"Command: {e.cmd}")
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

# Backward compatibility functions - maintain all your original function signatures
def apply_normalization(pheno_df, config, qtl_type, results_dir):
    """Apply proper normalization based on QTL type - Compatibility wrapper"""
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

# Additional utility functions for modular pipeline
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
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Run analysis based on config
        genotype_file = prepare_genotypes(config, config['results_dir'])
        qtl_types = config['analysis']['qtl_types']
        
        if isinstance(qtl_types, str) and qtl_types != 'all':
            qtl_types = [qtl_types]
        elif qtl_types == 'all':
            qtl_types = ['eqtl']  # Default to eQTL
        
        for qtl_type in qtl_types:
            if config['analysis']['qtl_mode'] in ['cis', 'both']:
                run_cis_analysis(config, genotype_file, qtl_type, config['results_dir'])
            if config['analysis']['qtl_mode'] in ['trans', 'both']:
                run_trans_analysis(config, genotype_file, qtl_type, config['results_dir'])
        
        logger.info("✅ QTL analysis completed successfully")
        
    except Exception as e:
        logger.error(f"❌ QTL analysis failed: {e}")
        sys.exit(1)