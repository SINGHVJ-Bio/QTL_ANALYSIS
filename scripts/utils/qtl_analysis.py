#!/usr/bin/env python3
"""
Enhanced QTL analysis utilities with tensorQTL-specific capabilities
Complete pipeline for cis/trans QTL analysis using tensorQTL
With proper normalization: VST for eQTL, log2 for pQTL/sQTL
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
import warnings
import tempfile
from .normalization_comparison import NormalizationComparison

# Import tensorQTL
try:
    import tensorqtl
    from tensorqtl import genotypeio, cis, trans
    import torch
    TENSORQTL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"tensorQTL not available: {e}")
    TENSORQTL_AVAILABLE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

def prepare_genotypes(config, results_dir):
    """Prepare genotype data optimized for tensorQTL"""
    logger.info("🔧 Preparing genotype data for tensorQTL...")
    
    # Initialize genotype processor
    from .genotype_processing import GenotypeProcessor
    processor = GenotypeProcessor(config)
    
    # Get input file path
    input_file = config['input_files']['genotypes']
    
    # Process genotypes - tensorQTL prefers PLINK format
    genotype_file = processor.process_genotypes(input_file, results_dir)
    
    # Ensure we have PLINK format for tensorQTL
    if genotype_file.endswith('.vcf.gz'):
        # Convert VCF to PLINK for tensorQTL
        plink_base = os.path.join(results_dir, "genotypes_plink")
        logger.info("🔄 Converting VCF to PLINK format for tensorQTL...")
        run_command(
            f"{config['paths']['plink']} --vcf {genotype_file} --make-bed --out {plink_base}",
            "Converting VCF to PLINK for tensorQTL", config
        )
        genotype_file = plink_base + ".bed"
    
    logger.info(f"✅ Genotype preparation completed: {genotype_file}")
    return genotype_file

def prepare_phenotype_data(config, qtl_type, results_dir):
    """Prepare phenotype data for tensorQTL with proper normalization"""
    logger.info(f"🔧 Preparing {qtl_type} phenotype data with {config['normalization'][qtl_type]['method']} normalization...")
    
    try:
        # Read phenotype data
        pheno_file = config['input_files'][qtl_type]
        if not os.path.exists(pheno_file):
            raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")
            
        pheno_df = pd.read_csv(pheno_file, sep='\t', index_col=0)
        logger.info(f"📊 Loaded {qtl_type} data: {pheno_df.shape[0]} features, {pheno_df.shape[1]} samples")
        
        # Apply QC filters if enabled
        if config['qc'].get('filter_low_expressed', True):
            pheno_df = filter_low_expressed_features(pheno_df, config, qtl_type)
        
        # Apply proper normalization based on QTL type
        if config['qc'].get('normalize', True):
            normalized_df = apply_normalization(pheno_df, config, qtl_type, results_dir)
        else:
            normalized_df = pheno_df
        
        # Generate normalization comparison plots if enabled
        if config.get('enhanced_qc', {}).get('generate_normalization_plots', True):
            try:
                comparison = NormalizationComparison(config, results_dir)
                comparison_results = comparison.generate_comprehensive_comparison(
                    qtl_type, pheno_df.copy(), normalized_df, 
                    config['normalization'][qtl_type]['method']
                )
                logger.info(f"📊 Normalization comparison completed for {qtl_type}")
            except Exception as e:
                logger.warning(f"⚠️ Normalization comparison failed: {e}")
        
        # Transpose for tensorQTL (samples x features)
        normalized_df = normalized_df.T
        
        # Save phenotype data for tensorQTL
        pheno_output_file = os.path.join(results_dir, f"{qtl_type}_phenotypes.parquet")
        normalized_df.to_parquet(pheno_output_file)
        
        # Prepare phenotype positions
        annotation_file = config['input_files']['annotations']
        annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
        
        # Create phenotype positions DataFrame
        pheno_pos_df = create_phenotype_positions(normalized_df.columns, annot_df, qtl_type)
        
        # Save phenotype positions
        pheno_pos_file = os.path.join(results_dir, f"{qtl_type}_phenotype_positions.parquet")
        pheno_pos_df.to_parquet(pheno_pos_file)
        
        logger.info(f"✅ Prepared {qtl_type} data for tensorQTL: {normalized_df.shape[1]} features")
        
        return {
            'phenotype_file': pheno_output_file,
            'phenotype_pos_file': pheno_pos_file,
            'phenotype_df': normalized_df,
            'phenotype_pos_df': pheno_pos_df
        }
        
    except Exception as e:
        logger.error(f"❌ Error preparing {qtl_type} data: {e}")
        raise

def create_phenotype_positions(feature_ids, annot_df, qtl_type):
    """Create phenotype positions DataFrame for tensorQTL"""
    positions_data = []
    
    for feature_id in feature_ids:
        feature_annot = annot_df[annot_df['gene_id'] == feature_id]
        
        if len(feature_annot) == 0:
            # If no annotation found, use default values
            positions_data.append({
                'phenotype_id': feature_id,
                'chr': '1',
                'start': 1,
                'end': 2,
                'strand': '+'
            })
        else:
            feature_annot = feature_annot.iloc[0]
            positions_data.append({
                'phenotype_id': feature_id,
                'chr': feature_annot['chr'],
                'start': feature_annot['start'],
                'end': feature_annot['end'],
                'strand': feature_annot.get('strand', '+')
            })
    
    positions_df = pd.DataFrame(positions_data)
    positions_df = positions_df.set_index('phenotype_id')
    return positions_df

def load_genotype_data(genotype_file, config):
    """Load genotype data for tensorQTL"""
    logger.info("🔧 Loading genotype data for tensorQTL...")
    
    try:
        if genotype_file.endswith('.bed'):
            # Load PLINK data
            plink_prefix = genotype_file.replace('.bed', '')
            genotype_df = genotypeio.load_genotypes(plink_prefix)
        else:
            raise ValueError(f"Unsupported genotype format: {genotype_file}")
        
        logger.info(f"✅ Loaded genotype data: {genotype_df.shape[0]} variants, {genotype_df.shape[1]} samples")
        return genotype_df
        
    except Exception as e:
        logger.error(f"❌ Error loading genotype data: {e}")
        raise

def load_covariates(config, results_dir):
    """Load and prepare covariates for tensorQTL"""
    logger.info("🔧 Loading covariates for tensorQTL...")
    
    try:
        covariates_file = config['input_files']['covariates']
        cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
        
        # Transpose for tensorQTL (samples x covariates)
        cov_df = cov_df.T
        
        # Use enhanced covariates if available
        enhanced_cov_file = os.path.join(results_dir, "enhanced_covariates.txt")
        if os.path.exists(enhanced_cov_file):
            enhanced_cov_df = pd.read_csv(enhanced_cov_file, sep='\t', index_col=0)
            enhanced_cov_df = enhanced_cov_df.T
            # Merge with original covariates
            cov_df = pd.concat([cov_df, enhanced_cov_df], axis=1)
            logger.info("✅ Using enhanced covariates with PCA components")
        
        logger.info(f"✅ Loaded covariates: {cov_df.shape[1]} covariates, {cov_df.shape[0]} samples")
        return cov_df
        
    except Exception as e:
        logger.error(f"❌ Error loading covariates: {e}")
        return None

def run_cis_analysis(config, genotype_file, qtl_type, results_dir):
    """Run cis-QTL analysis using tensorQTL"""
    if not TENSORQTL_AVAILABLE:
        raise ImportError("tensorQTL is not available. Please install it: pip install tensorqtl")
    
    logger.info(f"🔍 Running {qtl_type} cis-QTL analysis with tensorQTL...")
    
    try:
        # Prepare phenotype data
        pheno_data = prepare_phenotype_data(config, qtl_type, results_dir)
        
        # Load genotype data
        genotype_df = load_genotype_data(genotype_file, config)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir)
        
        # Set output paths
        output_prefix = os.path.join(results_dir, f"{qtl_type}_cis")
        
        # Get tensorQTL parameters
        tqtl_config = config.get('tensorqtl', {})
        
        # Run cis-QTL analysis
        logger.info("🔬 Running tensorQTL cis mapping...")
        
        cis_df = cis.map_cis(
            genotype_df, 
            pheno_data['phenotype_df'], 
            pheno_data['phenotype_pos_df'],
            covariates_df=covariates_df,
            window=tqtl_config.get('cis_window', 1000000),
            maf_threshold=tqtl_config.get('maf_threshold', 0.05),
            min_maf=tqtl_config.get('min_maf', 0.01),
            seed=tqtl_config.get('seed', 12345),
            output_dir=results_dir,
            prefix=f"{qtl_type}_cis",
            write_stats=tqtl_config.get('write_stats', True),
            write_top=tqtl_config.get('write_top_results', True),
            run_eigenmt=False  # Can be enabled for multiple testing correction
        )
        
        # Run permutations if requested
        if tqtl_config.get('run_permutations', True):
            logger.info("🔬 Running tensorQTL cis permutations...")
            
            cis_df = cis.map_cis(
                genotype_df, 
                pheno_data['phenotype_df'], 
                pheno_data['phenotype_pos_df'],
                covariates_df=covariates_df,
                window=tqtl_config.get('cis_window', 1000000),
                maf_threshold=tqtl_config.get('maf_threshold', 0.05),
                min_maf=tqtl_config.get('min_maf', 0.01),
                seed=tqtl_config.get('seed', 12345),
                output_dir=results_dir,
                prefix=f"{qtl_type}_cis",
                write_stats=tqtl_config.get('write_stats', True),
                write_top=tqtl_config.get('write_top_results', True),
                run_eigenmt=False,
                nperm=tqtl_config.get('num_permutations', 1000)
            )
        
        # Count significant associations
        significant_count = count_tensorqtl_significant(results_dir, f"{qtl_type}_cis", tqtl_config.get('fdr_threshold', 0.05))
        
        logger.info(f"✅ {qtl_type} cis: Found {significant_count} significant associations")
        
        return {
            'result_file': os.path.join(results_dir, f"{qtl_type}_cis.cis_qtl.txt.gz"),
            'nominals_file': os.path.join(results_dir, f"{qtl_type}_cis.cis_qtl.txt.gz"),
            'significant_count': significant_count,
            'status': 'completed'
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
    """Run trans-QTL analysis using tensorQTL"""
    if not TENSORQTL_AVAILABLE:
        raise ImportError("tensorQTL is not available. Please install it: pip install tensorqtl")
    
    logger.info(f"🔍 Running {qtl_type} trans-QTL analysis with tensorQTL...")
    
    try:
        # Prepare phenotype data
        pheno_data = prepare_phenotype_data(config, qtl_type, results_dir)
        
        # Load genotype data
        genotype_df = load_genotype_data(genotype_file, config)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir)
        
        # Set output paths
        output_prefix = os.path.join(results_dir, f"{qtl_type}_trans")
        
        # Get tensorQTL parameters
        tqtl_config = config.get('tensorqtl', {})
        
        # Run trans-QTL analysis
        logger.info("🔬 Running tensorQTL trans mapping...")
        
        # For large datasets, use chunked processing
        trans_df = trans.map_trans(
            genotype_df, 
            pheno_data['phenotype_df'],
            covariates_df=covariates_df,
            batch_size=tqtl_config.get('batch_size', 10000),
            maf_threshold=tqtl_config.get('maf_threshold', 0.05),
            min_maf=tqtl_config.get('min_maf', 0.01),
            return_sparse=True,
            pval_threshold=0.05  # Initial p-value threshold
        )
        
        # Save results
        trans_file = os.path.join(results_dir, f"{qtl_type}_trans.trans_qtl.txt.gz")
        trans_df.to_csv(trans_file, sep='\t', compression='gzip')
        
        # Count significant associations
        significant_count = len(trans_df) if trans_df is not None else 0
        
        logger.info(f"✅ {qtl_type} trans: Found {significant_count} significant associations")
        
        return {
            'result_file': trans_file,
            'nominals_file': trans_file,
            'significant_count': significant_count,
            'status': 'completed'
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

def count_tensorqtl_significant(results_dir, prefix, fdr_threshold=0.05):
    """Count significant associations from tensorQTL output"""
    result_file = os.path.join(results_dir, f"{prefix}.cis_qtl.txt.gz")
    
    if not os.path.exists(result_file):
        return 0
    
    try:
        df = pd.read_csv(result_file, sep='\t')
        
        # tensorQTL outputs qval column for FDR
        if 'qval' in df.columns:
            significant_count = len(df[df['qval'] < fdr_threshold])
        elif 'pval_perm' in df.columns:
            # Use permutation p-values
            significant_count = len(df[df['pval_perm'] < fdr_threshold])
        elif 'pval_nominal' in df.columns:
            # Use nominal p-values with Bonferroni correction
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

# Keep the normalization functions from your original code
def apply_normalization(pheno_df, config, qtl_type, results_dir):
    """Apply proper normalization based on QTL type"""
    normalization_config = config['normalization'][qtl_type]
    method = normalization_config.get('method', 'log2')
    
    logger.info(f"🔄 Applying {method} normalization for {qtl_type}...")
    
    if method == 'vst' and qtl_type == 'eqtl':
        return apply_vst_normalization(pheno_df, config, results_dir)
    elif method == 'log2':
        return apply_log2_normalization(pheno_df, config, qtl_type)
    elif method == 'quantile':
        return apply_quantile_normalization(pheno_df)
    elif method == 'zscore':
        return apply_zscore_normalization(pheno_df)
    elif method == 'arcsinh':
        return apply_arcsinh_normalization(pheno_df, config, qtl_type)
    elif method == 'tpm':
        return apply_tpm_normalization(pheno_df, config)
    elif method == 'raw':
        logger.info("📊 Using raw data without normalization")
        return pheno_df
    else:
        logger.warning(f"⚠️ Unknown normalization method '{method}' for {qtl_type}, using log2")
        return apply_log2_normalization(pheno_df, config, qtl_type)

def apply_vst_normalization(pheno_df, config, results_dir):
    """Apply VST normalization using DESeq2 for eQTL data"""
    logger.info("🔬 Applying VST normalization using DESeq2...")
    
    try:
        # Save temporary input file
        temp_input = os.path.join(results_dir, "temp_expression_counts.txt")
        temp_output = os.path.join(results_dir, "temp_expression_vst.txt")
        
        # Save data with gene IDs
        pheno_df.reset_index().to_csv(temp_input, sep='\t', index=False)
        
        # Get DESeq2 parameters
        vst_config = config['normalization']['eqtl']
        blind = vst_config.get('vst_blind', True)
        fit_type = vst_config.get('fit_type', 'parametric')
        
        # Build R command
        r_script_path = config['paths'].get('r_script_deseq2', 'scripts/utils/deseq2_vst.R')
        cmd = f"Rscript {r_script_path} {temp_input} {temp_output} {blind} {fit_type}"
        
        # Run DESeq2 VST normalization
        result = run_command(cmd, "DESeq2 VST normalization", config)
        
        if result.returncode != 0:
            logger.warning("❌ DESeq2 VST normalization failed, falling back to log2")
            # Clean up and fall back to log2
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return apply_log2_normalization(pheno_df, config, 'eqtl')
        
        # Read VST normalized data
        vst_df = pd.read_csv(temp_output, sep='\t', index_col=0)
        
        # Clean up temporary files
        os.remove(temp_input)
        os.remove(temp_output)
        
        logger.info(f"✅ VST normalization completed: {vst_df.shape[0]} features")
        return vst_df
        
    except Exception as e:
        logger.error(f"❌ VST normalization failed: {e}, falling back to log2")
        return apply_log2_normalization(pheno_df, config, 'eqtl')

def apply_log2_normalization(pheno_df, config, qtl_type):
    """Apply log2 transformation with pseudocount"""
    norm_config = config['normalization'][qtl_type]
    pseudocount = norm_config.get('log2_pseudocount', 1)
    remove_zeros = norm_config.get('remove_zeros', True)
    
    # Remove zeros if configured
    if remove_zeros:
        # Replace zeros with NaN
        pheno_df = pheno_df.replace(0, np.nan)
        # Remove features with all zeros
        pheno_df = pheno_df.dropna(how='all')
    
    # Apply log2 transformation
    pheno_df = np.log2(pheno_df + pseudocount)
    
    logger.info(f"✅ Applied log2 transformation (pseudocount={pseudocount})")
    return pheno_df

def apply_quantile_normalization(pheno_df):
    """Apply quantile normalization"""
    logger.info("🔧 Applying quantile normalization...")
    
    # Rank-based quantile normalization
    ranked_df = pheno_df.rank(axis=1, method='average')
    quantiles = ranked_df.mean(axis=0)
    
    # Map ranks to quantiles
    normalized_df = ranked_df.apply(lambda x: np.interp(x, np.arange(1, len(x)+1), quantiles), axis=1)
    
    logger.info("✅ Quantile normalization completed")
    return normalized_df

def apply_zscore_normalization(pheno_df):
    """Apply z-score normalization per feature"""
    logger.info("🔧 Applying z-score normalization per feature...")
    
    # Standardize each feature (row)
    normalized_df = (pheno_df - pheno_df.mean(axis=1).values.reshape(-1, 1)) / pheno_df.std(axis=1).values.reshape(-1, 1)
    
    logger.info("✅ Z-score normalization completed")
    return normalized_df

def apply_arcsinh_normalization(pheno_df, config, qtl_type):
    """Apply arcsinh transformation (useful for PSI values in sQTL)"""
    logger.info("🔧 Applying arcsinh transformation...")
    
    norm_config = config['normalization'][qtl_type]
    cofactor = norm_config.get('arcsinh_cofactor', 1)
    
    # Apply arcsinh transformation
    normalized_df = np.arcsinh(pheno_df / cofactor)
    
    logger.info(f"✅ Arcsinh transformation completed (cofactor={cofactor})")
    return normalized_df

def apply_tpm_normalization(pheno_df, config):
    """Apply TPM normalization (requires gene lengths)"""
    logger.info("🔧 Applying TPM normalization...")
    
    # This is a simplified TPM calculation
    # In practice, you would need gene lengths
    logger.warning("⚠️ TPM normalization requires gene length information")
    logger.warning("   Using simplified TPM-like normalization")
    
    # Read counts per million (RPM)
    rpm_df = pheno_df.div(pheno_df.sum(axis=0)) * 1e6
    
    # For true TPM, you would divide by gene length in kb
    # tpm_df = rpm_df.div(gene_lengths_kb, axis=0)
    
    logger.info("✅ Simplified TPM normalization completed")
    return rpm_df

def filter_low_expressed_features(pheno_df, config, qtl_type):
    """Filter lowly expressed features based on QTL type"""
    global_config = config['normalization']['global']
    
    # Remove constant features
    if global_config.get('remove_constant_features', True):
        constant_threshold = global_config.get('constant_threshold', 0.95)
        non_constant_mask = (pheno_df.nunique(axis=1) / pheno_df.shape[1]) > (1 - constant_threshold)
        pheno_df = pheno_df[non_constant_mask]
        logger.info(f"🔧 Removed constant features: {non_constant_mask.sum()} features remaining")
    
    # Remove features with too many missing values
    if global_config.get('missing_value_threshold', 0.2) > 0:
        missing_threshold = global_config.get('missing_value_threshold', 0.2)
        low_missing_mask = (pheno_df.isna().sum(axis=1) / pheno_df.shape[1]) < missing_threshold
        pheno_df = pheno_df[low_missing_mask]
        logger.info(f"🔧 Removed high-missing features: {low_missing_mask.sum()} features remaining")
    
    # QTL-type specific filtering
    if qtl_type == 'eqtl':
        threshold = config['qc'].get('expression_threshold', 0.1)
        mean_expression = pheno_df.mean(axis=1)
        expressed_mask = mean_expression > threshold
        pheno_df = pheno_df[expressed_mask]
        logger.info(f"🔧 Filtered low expressed genes: {expressed_mask.sum()} features remaining")
    
    elif qtl_type in ['pqtl', 'sqtl']:
        # For protein and splicing, filter based on variance
        variance_threshold = pheno_df.var(axis=1).quantile(0.1)  # Keep top 90% by variance
        high_variance_mask = pheno_df.var(axis=1) > variance_threshold
        pheno_df = pheno_df[high_variance_mask]
        logger.info(f"🔧 Filtered low variance features: {high_variance_mask.sum()} features remaining")
    
    return pheno_df

def run_command(cmd, description, config, check=True):
    """Run shell command with comprehensive error handling"""
    logger.info(f"Executing: {description}")
    logger.debug(f"Command: {cmd}")
    
    # Set timeout for large dataset operations
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
        if check and result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
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