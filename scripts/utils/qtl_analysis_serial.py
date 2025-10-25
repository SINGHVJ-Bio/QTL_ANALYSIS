#!/usr/bin/env python3
"""
Enhanced QTL analysis utilities with tensorQTL-specific capabilities - Production Version
Complete pipeline for cis/trans QTL analysis using tensorQTL with robust error handling
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
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

# Import tensorQTL with comprehensive error handling
try:
    import tensorqtl
    from tensorqtl import genotypeio, cis, trans
    import torch
    
    # Try different imports for calculate_qvalues based on tensorQTL version
    try:
        # Alternative import for newer versions
        from tensorqtl import calculate_qvalues
        CALCULATE_QVALUES_AVAILABLE = True
    except ImportError:
        # Fallback to statsmodels for FDR calculation
        from statsmodels.stats.multitest import multipletests
        CALCULATE_QVALUES_AVAILABLE = False
        logger = logging.getLogger('QTLPipeline')
        logger.warning("tensorqtl.utils.calculate_qvalues not available, using statsmodels fallback")
    
    TENSORQTL_AVAILABLE = True
    # Initialize logger after import
    if 'logger' not in locals():
        logger = logging.getLogger('QTLPipeline')
    logger.info("✅ tensorQTL successfully imported")
    
except ImportError as e:
    # Initialize logger before using it
    if 'logger' not in locals():
        logger = logging.getLogger('QTLPipeline')
    logger.error(f"❌ tensorQTL import failed: {e}")
    TENSORQTL_AVAILABLE = False
    CALCULATE_QVALUES_AVAILABLE = False

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

class QTLConfig:
    """Enhanced QTL configuration management with robust error handling"""
    def __init__(self, config):
        self.config = config
        self.qtl_config = config.get('qtl', {})
        self.tensorqtl_config = config.get('tensorqtl', {})
        self.normalization_config = config.get('normalization', {})
        self.performance_config = config.get('performance', {})
        self.large_data_config = config.get('large_data', {})
        
    def get_analysis_params(self, analysis_type):
        """Get analysis parameters with comprehensive fallbacks"""
        base_params = {
            'cis_window': self.tensorqtl_config.get('cis_window', 1000000),
            'maf_threshold': self.tensorqtl_config.get('maf_threshold', 0.05),
            'min_maf': self.tensorqtl_config.get('min_maf', 0.01),
            'fdr_threshold': self.tensorqtl_config.get('fdr_threshold', 0.05),
            'num_permutations': self.tensorqtl_config.get('num_permutations', 1000),
            'batch_size': self.tensorqtl_config.get('batch_size', 10000),
            'seed': self.tensorqtl_config.get('seed', 42),
            'run_permutations': self.tensorqtl_config.get('run_permutations', True),
            'write_stats': self.tensorqtl_config.get('write_stats', True),
            'write_top_results': self.tensorqtl_config.get('write_top_results', True)
        }
        
        # Analysis-specific adjustments
        if analysis_type == 'trans':
            base_params.update({
                'batch_size': self.tensorqtl_config.get('trans_batch_size', 5000),
                'pval_threshold': self.tensorqtl_config.get('trans_pval_threshold', 1e-5),
                'return_sparse': self.tensorqtl_config.get('return_sparse', True)
            })
        
        # Performance tuning
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
        
    def prepare_phenotype_data(self, qtl_type):
        """Prepare phenotype data with comprehensive processing and error handling"""
        logger.info(f"🔧 Preparing {qtl_type} phenotype data...")
        
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
            
            # Apply QC filters
            if self.qc_config.get('filter_low_expressed', True):
                pheno_df = self._apply_qc_filters(pheno_df, qtl_type)
            
            # Apply normalization
            if self.qc_config.get('normalize', True):
                normalized_df = self._apply_normalization(pheno_df, qtl_type)
                normalization_method = self.normalization_config.get(qtl_type, {}).get('method', 'unknown')
                logger.info(f"🔄 Applied {normalization_method} normalization for {qtl_type}")
            else:
                normalized_df = pheno_df
                logger.info("📊 Using raw data without normalization")
            
            # Generate normalization comparison if enabled
            if self.config.get('enhanced_qc', {}).get('generate_normalization_plots', True) and NormalizationComparison:
                self._generate_normalization_comparison(pheno_df, normalized_df, qtl_type)
            
            # Prepare for tensorQTL (samples x features)
            normalized_df = normalized_df.T
            
            # Save processed data
            output_files = self._save_processed_data(normalized_df, qtl_type)
            
            final_feature_count = normalized_df.shape[1]
            logger.info(f"✅ Prepared {qtl_type} data: {final_feature_count}/{original_feature_count} features retained")
            
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Phenotype preparation failed for {qtl_type}: {e}")
            raise
    
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
                        break
                except:
                    continue
            else:
                raise ValueError(f"Could not read {qtl_type} file with any standard separator")
            
            if df.empty:
                raise ValueError(f"Phenotype file is empty: {file_path}")
            
            # Check for numeric data
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                logger.warning(f"⚠️ Found {len(non_numeric_cols)} non-numeric columns in {qtl_type} data")
                # Try to convert to numeric
                for col in non_numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check if conversion was successful
                remaining_non_numeric = df.select_dtypes(exclude=[np.number]).columns
                if len(remaining_non_numeric) > 0:
                    logger.warning(f"⚠️ Could not convert {len(remaining_non_numeric)} columns to numeric, they will be dropped")
                    df = df.select_dtypes(include=[np.number])
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading phenotype data from {file_path}: {e}")
            raise
    
    def _apply_qc_filters(self, pheno_df, qtl_type):
        """Apply comprehensive quality control filters"""
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
        missing_removed = (original_count - constant_removed) - filtered_df.shape[0]
        
        # QTL-type specific filtering
        if qtl_type == 'eqtl':
            threshold = self.qc_config.get('expression_threshold', 0.1)
            mean_expression = filtered_df.mean(axis=1)
            expressed_mask = mean_expression > threshold
            filtered_df = filtered_df[expressed_mask]
            low_expression_removed = filtered_df.shape[0] - expressed_mask.sum()
        elif qtl_type in ['pqtl', 'sqtl']:
            # Filter based on variance
            variance_threshold = filtered_df.var(axis=1).quantile(0.1)
            high_variance_mask = filtered_df.var(axis=1) > variance_threshold
            filtered_df = filtered_df[high_variance_mask]
            low_variance_removed = filtered_df.shape[0] - high_variance_mask.sum()
        else:
            low_expression_removed = 0
            low_variance_removed = 0
        
        filtered_count = filtered_df.shape[0]
        logger.info(f"🔧 QC filtering: {filtered_count}/{original_count} features retained "
                   f"(constant: {constant_removed}, missing: {missing_removed}, "
                   f"low_expr: {low_expression_removed}, low_var: {low_variance_removed})")
        
        return filtered_df
    
    def _apply_normalization(self, pheno_df, qtl_type):
        """Apply appropriate normalization based on QTL type with comprehensive error handling"""
        norm_config = self.normalization_config.get(qtl_type, {})
        method = norm_config.get('method', 'log2')
        
        logger.info(f"🔄 Applying {method} normalization for {qtl_type}...")
        
        normalization_methods = {
            'vst': self._apply_vst_normalization,
            'log2': self._apply_log2_normalization,
            'quantile': self._apply_quantile_normalization,
            'zscore': self._apply_zscore_normalization,
            'arcsinh': self._apply_arcsinh_normalization,
            'tpm': self._apply_tpm_normalization,
            'raw': lambda x, y: x  # No normalization
        }
        
        if method in normalization_methods:
            try:
                return normalization_methods[method](pheno_df, qtl_type)
            except Exception as e:
                logger.error(f"❌ {method} normalization failed: {e}, falling back to log2")
                return self._apply_log2_normalization(pheno_df, qtl_type)
        else:
            logger.warning(f"⚠️ Unknown normalization method '{method}', using log2")
            return self._apply_log2_normalization(pheno_df, qtl_type)
    
    def _apply_vst_normalization(self, pheno_df, qtl_type):
        """Apply VST normalization using DESeq2 with robust error handling"""
        if qtl_type != 'eqtl':
            logger.warning("VST normalization is typically for expression data, using log2 instead")
            return self._apply_log2_normalization(pheno_df, qtl_type)
        
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
            result = run_command(cmd, "DESeq2 VST normalization", self.config, check=False)
            
            if result.returncode != 0:
                logger.warning("❌ DESeq2 VST normalization failed, falling back to log2")
                return self._apply_log2_normalization(pheno_df, qtl_type)
            
            # Load normalized data
            if os.path.exists(temp_output_path):
                vst_df = pd.read_csv(temp_output_path, sep='\t', index_col=0)
                logger.info("✅ VST normalization completed successfully")
                return vst_df
            else:
                raise FileNotFoundError("VST output file not generated")
                
        except Exception as e:
            logger.error(f"❌ VST normalization failed: {e}, falling back to log2")
            return self._apply_log2_normalization(pheno_df, qtl_type)
        finally:
            # Clean up temporary files
            for temp_file in [temp_input_path, temp_output_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def _apply_log2_normalization(self, pheno_df, qtl_type):
        """Apply log2 transformation with comprehensive options"""
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
    
    def _apply_quantile_normalization(self, pheno_df, qtl_type):
        """Apply quantile normalization"""
        try:
            from sklearn.preprocessing import quantile_transform
            
            # Transpose for sample-wise normalization
            normalized_array = quantile_transform(pheno_df.T, n_quantiles=min(1000, pheno_df.shape[0]))
            normalized_df = pd.DataFrame(normalized_array.T, index=pheno_df.index, columns=pheno_df.columns)
            
            logger.info("✅ Quantile normalization completed")
            return normalized_df
        except ImportError:
            logger.warning("⚠️ scikit-learn not available for quantile normalization, using log2")
            return self._apply_log2_normalization(pheno_df, qtl_type)
    
    def _apply_zscore_normalization(self, pheno_df, qtl_type):
        """Apply z-score normalization per feature with constant feature handling"""
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
    
    def _save_processed_data(self, normalized_df, qtl_type):
        """Save processed phenotype data with comprehensive output options"""
        # Save phenotype matrix
        output_format = self.config.get('output', {}).get('phenotype_format', 'parquet')
        if output_format == 'parquet':
            pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.parquet")
            normalized_df.to_parquet(pheno_file)
        else:
            pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.txt.gz")
            normalized_df.to_csv(pheno_file, sep='\t', compression='gzip')
        
        # Create and save phenotype positions
        pheno_pos_file = os.path.join(self.results_dir, f"{qtl_type}_phenotype_positions.parquet")
        pheno_pos_df = self._create_phenotype_positions(normalized_df.columns, qtl_type)
        pheno_pos_df.to_parquet(pheno_pos_file)
        
        logger.info(f"💾 Saved processed data: {pheno_file}")
        
        return {
            'phenotype_file': pheno_file,
            'phenotype_pos_file': pheno_pos_file,
            'phenotype_df': normalized_df,
            'phenotype_pos_df': pheno_pos_df
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
        """Load genotype data with comprehensive error handling and performance optimization"""
        logger.info("🔧 Loading genotype data for tensorQTL...")
        
        if not TENSORQTL_AVAILABLE:
            raise ImportError("tensorQTL is not available. Please install: pip install tensorqtl")
        
        try:
            if genotype_file.endswith('.bed'):
                # Load PLINK data
                plink_prefix = genotype_file.replace('.bed', '')
                
                # Set tensorQTL parameters for performance
                if self.performance_config.get('use_gpu', False) and torch.cuda.is_available():
                    logger.info("🚀 Using GPU for tensorQTL analysis")
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
                
                pr = genotypeio.read_plink(plink_prefix)
                logger.info(f"✅ Loaded PLINK data: {pr.genotypes.shape[0]} variants, {pr.genotypes.shape[1]} samples")
                return pr
            else:
                raise ValueError(f"Unsupported genotype format: {genotype_file}. Use PLINK format for best performance.")
                
        except Exception as e:
            logger.error(f"❌ Error loading genotype data: {e}")
            raise
    
    def optimize_genotype_data(self, genotype_reader):
        """Optimize genotype data for analysis with comprehensive filtering"""
        original_count = genotype_reader.genotypes.shape[0]
        
        # Apply MAF filtering
        maf_threshold = self.genotype_processing_config.get('min_maf', 0.01)
        if maf_threshold > 0:
            maf = genotype_reader.genotypes.maf()
            keep_variants = maf >= maf_threshold
            genotype_reader.genotypes = genotype_reader.genotypes[keep_variants]
            maf_filtered_count = genotype_reader.genotypes.shape[0]
            logger.info(f"🔧 MAF filtering: {maf_filtered_count}/{original_count} variants retained (MAF >= {maf_threshold})")
        
        # Apply call rate filtering if needed
        call_rate_threshold = self.genotype_processing_config.get('min_call_rate', 0.95)
        if call_rate_threshold < 1.0:
            call_rate = 1 - genotype_reader.genotypes.isnan().mean(axis=1)
            keep_variants = call_rate >= call_rate_threshold
            genotype_reader.genotypes = genotype_reader.genotypes[keep_variants]
            call_rate_filtered_count = genotype_reader.genotypes.shape[0]
            logger.info(f"🔧 Call rate filtering: {call_rate_filtered_count} variants retained (call rate >= {call_rate_threshold})")
        
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

def load_covariates(config, results_dir):
    """Load and prepare covariates for tensorQTL with enhanced processing"""
    logger.info("🔧 Loading covariates for tensorQTL...")
    
    try:
        covariates_file = config['input_files']['covariates']
        cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
        
        # Transpose for tensorQTL (samples x covariates)
        cov_df = cov_df.T
        
        # Check for enhanced covariates
        enhanced_cov_file = os.path.join(results_dir, "enhanced_covariates.txt")
        if os.path.exists(enhanced_cov_file):
            enhanced_cov_df = pd.read_csv(enhanced_cov_file, sep='\t', index_col=0)
            enhanced_cov_df = enhanced_cov_df.T
            # Merge with original covariates
            cov_df = pd.concat([cov_df, enhanced_cov_df], axis=1)
            logger.info("✅ Using enhanced covariates with PCA components")
        
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
    """Calculate FDR using available methods"""
    if CALCULATE_QVALUES_AVAILABLE:
        try:
            return calculate_qvalues(pvalues)
        except:
            # Fallback to statsmodels if tensorQTL method fails
            pass
    
    # Use statsmodels as fallback
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(pvalues, method=method)
    return fdr

def run_cis_analysis(config, genotype_file, qtl_type, results_dir):
    """Run cis-QTL analysis using tensorQTL with enhanced error handling and performance"""
    if not TENSORQTL_AVAILABLE:
        raise ImportError("tensorQTL is not available. Please install it: pip install tensorqtl")
    
    logger.info(f"🔍 Running {qtl_type} cis-QTL analysis with tensorQTL...")
    
    try:
        # Initialize configuration
        qtl_config = QTLConfig(config)
        qtl_config.validate_parameters('cis')
        params = qtl_config.get_analysis_params('cis')
        
        # Prepare phenotype data
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type)
        
        # Load and optimize genotype data
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        pr = genotype_loader.optimize_genotype_data(pr)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir)
        
        # Set output prefix
        output_prefix = os.path.join(results_dir, f"{qtl_type}_cis")
        
        # Run cis-QTL analysis
        logger.info("🔬 Running tensorQTL cis mapping...")
        
        # Map cis-QTLs
        cis_df = cis.map_cis(
            genotype_df=pr.genotypes,
            phenotype_df=pheno_data['phenotype_df'],
            phenotype_pos_df=pheno_data['phenotype_pos_df'],
            covariates_df=covariates_df,
            window=params['cis_window'],
            maf_threshold=params['maf_threshold'],
            seed=params['seed'],
            output_dir=results_dir,
            prefix=f"{qtl_type}_cis",
            write_stats=params['write_stats'],
            write_top=params['write_top_results'],
            run_eigenmt=False
        )
        
        # Run permutations for FDR estimation if requested
        if params['run_permutations']:
            logger.info("🔬 Running tensorQTL cis permutations...")
            
            cis_df = cis.map_cis(
                genotype_df=pr.genotypes,
                phenotype_df=pheno_data['phenotype_df'],
                phenotype_pos_df=pheno_data['phenotype_pos_df'],
                covariates_df=covariates_df,
                window=params['cis_window'],
                maf_threshold=params['maf_threshold'],
                seed=params['seed'],
                output_dir=results_dir,
                prefix=f"{qtl_type}_cis",
                write_stats=params['write_stats'],
                write_top=params['write_top_results'],
                run_eigenmt=False,
                nperm=params['num_permutations']
            )
        
        # Count significant associations
        significant_count = count_significant_associations(results_dir, f"{qtl_type}_cis", params['fdr_threshold'])
        
        logger.info(f"✅ {qtl_type} cis: Found {significant_count} significant associations")
        
        return {
            'result_file': os.path.join(results_dir, f"{qtl_type}_cis.cis_qtl.txt.gz"),
            'nominals_file': os.path.join(results_dir, f"{qtl_type}_cis.cis_qtl.txt.gz"),
            'significant_count': significant_count,
            'status': 'completed',
            'params': params
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
        raise ImportError("tensorQTL is not available. Please install it: pip install tensorqtl")
    
    logger.info(f"🔍 Running {qtl_type} trans-QTL analysis with tensorQTL...")
    
    try:
        # Initialize configuration
        qtl_config = QTLConfig(config)
        params = qtl_config.get_analysis_params('trans')
        
        # Prepare phenotype data
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type)
        
        # Load and optimize genotype data
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        pr = genotype_loader.optimize_genotype_data(pr)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir)
        
        # Set output prefix
        output_prefix = os.path.join(results_dir, f"{qtl_type}_trans")
        
        # Run trans-QTL analysis with memory optimization
        logger.info("🔬 Running tensorQTL trans mapping...")
        
        # Use chunked processing for large datasets
        trans_df = trans.map_trans(
            genotype_df=pr.genotypes,
            phenotype_df=pheno_data['phenotype_df'],
            covariates_df=covariates_df,
            batch_size=params['batch_size'],
            maf_threshold=params['maf_threshold'],
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
        
        return {
            'result_file': trans_file,
            'nominals_file': trans_file,
            'significant_count': significant_count,
            'status': 'completed',
            'params': params
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
def process_expression_data(config, results_dir):
    """Process expression data for modular pipeline"""
    return prepare_phenotype_data(config, 'eqtl', results_dir)

def run_qtl_mapping(config, genotype_file, qtl_type, results_dir, analysis_mode='cis'):
    """Unified QTL mapping function for modular pipeline"""
    if analysis_mode == 'cis':
        return run_cis_analysis(config, genotype_file, qtl_type, results_dir)
    elif analysis_mode == 'trans':
        return run_trans_analysis(config, genotype_file, qtl_type, results_dir)
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")

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