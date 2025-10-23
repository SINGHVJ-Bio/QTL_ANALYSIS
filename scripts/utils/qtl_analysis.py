#!/usr/bin/env python3
"""
Enhanced QTL analysis utilities with QTLtools-specific capabilities
Complete pipeline for cis/trans QTL analysis using QTLtools
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import warnings
from .normalization_comparison import NormalizationComparison

warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

def prepare_genotypes(config, results_dir):
    """Prepare genotype data optimized for QTLtools"""
    logger.info("🔧 Preparing genotype data for QTLtools...")
    
    # Initialize genotype processor
    from .genotype_processing import GenotypeProcessor
    processor = GenotypeProcessor(config)
    
    # Get input file path
    input_file = config['input_files']['genotypes']
    
    # Process genotypes
    genotype_file = processor.process_genotypes(input_file, results_dir)
    
    logger.info(f"✅ Genotype preparation completed: {genotype_file}")
    return genotype_file

def prepare_phenotype_data(config, qtl_type, results_dir):
    """Prepare phenotype data in BED format with proper normalization"""
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
            pheno_df = apply_normalization(pheno_df, config, qtl_type, results_dir)
        
        # Apply batch effect correction if enabled (using QTLtools covariates)
        if config['enhanced_qc'].get('batch_effect_correction', False):
            pheno_df = prepare_batch_covariates(pheno_df, config, results_dir)
        
        # Read annotation data
        annotation_file = config['input_files']['annotations']
        annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
        logger.info(f"📊 Loaded annotation data: {annot_df.shape[0]} features")
        
        # Create BED format for QTLtools
        bed_data = create_qtltools_bed_format(pheno_df, annot_df, qtl_type)
        
        if len(bed_data) == 0:
            raise ValueError(f"❌ No valid features found for {qtl_type} after annotation matching")
        
        # Save BED file
        bed_file = os.path.join(results_dir, f"{qtl_type}_phenotypes.bed")
        bed_data.to_csv(bed_file, sep='\t', index=False)
        
        # Compress and index for QTLtools
        bed_gz = bed_file + '.gz'
        run_command(
            f"{config['paths']['bgzip']} -c {bed_file} > {bed_gz}",
            f"Compressing {qtl_type} BED file", config
        )
        
        run_command(
            f"{config['paths']['tabix']} -p bed {bed_gz}",
            f"Indexing {qtl_type} BED file", config
        )
        
        logger.info(f"✅ Prepared {qtl_type} data for QTLtools: {len(bed_data)} features")
        return bed_gz
        
    except Exception as e:
        logger.error(f"❌ Error preparing {qtl_type} data: {e}")
        raise

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

def create_qtltools_bed_format(pheno_df, annot_df, qtl_type):
    """Create QTLtools-compatible BED format"""
    bed_data = []
    missing_annotations = 0
    
    for feature_id in pheno_df.index:
        feature_annot = annot_df[annot_df['gene_id'] == feature_id]
        
        if len(feature_annot) == 0:
            missing_annotations += 1
            continue
            
        feature_annot = feature_annot.iloc[0]
        
        # Create BED entry with QTLtools format
        bed_entry = {
            'chr': feature_annot['chr'],
            'start': feature_annot['start'],
            'end': feature_annot['end'],
            'pid': feature_id,  # QTLtools uses 'pid' for phenotype ID
            'gid': feature_id,  # QTLtools uses 'gid' for gene ID
            'strand': feature_annot.get('strand', '+')
        }
        
        # Add phenotype values for all samples
        for sample in pheno_df.columns:
            bed_entry[sample] = pheno_df.loc[feature_id, sample]
        
        bed_data.append(bed_entry)
    
    if missing_annotations > 0:
        logger.warning(f"⚠️ {missing_annotations} features missing annotations")
    
    # Create BED dataframe with proper column order
    if bed_data:
        bed_df = pd.DataFrame(bed_data)
        # Ensure proper column order: chr, start, end, pid, gid, strand, then samples
        sample_cols = [col for col in bed_df.columns if col not in ['chr', 'start', 'end', 'pid', 'gid', 'strand']]
        bed_df = bed_df[['chr', 'start', 'end', 'pid', 'gid', 'strand'] + sample_cols]
        return bed_df
    else:
        return pd.DataFrame()

def prepare_batch_covariates(pheno_df, config, results_dir):
    """Prepare batch effect covariates for QTLtools"""
    logger.info("🔧 Preparing batch effect covariates...")
    
    try:
        # Read original covariates
        covariates_file = config['input_files']['covariates']
        cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
        
        # Perform PCA to capture batch effects
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Transpose for sample-wise PCA
        pheno_samples = pheno_df.T.fillna(pheno_df.T.mean())
        
        # Standardize
        scaler = StandardScaler()
        pheno_scaled = scaler.fit_transform(pheno_samples)
        
        # Perform PCA
        n_pcs = config['enhanced_qc'].get('num_pc_covariates', 5)
        pca = PCA(n_components=n_pcs)
        pcs = pca.fit_transform(pheno_scaled)
        
        # Create PCA covariates
        pc_cov_df = pd.DataFrame(
            pcs, 
            columns=[f'PC{i+1}' for i in range(n_pcs)],
            index=pheno_samples.index
        )
        
        # Combine with original covariates
        combined_cov_df = pd.concat([cov_df.T, pc_cov_df], axis=1)
        
        # Save enhanced covariates
        enhanced_cov_file = os.path.join(results_dir, "enhanced_covariates.txt")
        combined_cov_df.T.to_csv(enhanced_cov_file, sep='\t')
        
        logger.info(f"✅ Created enhanced covariates with {n_pcs} PCs: {enhanced_cov_file}")
        
        return pheno_df  # Return unchanged phenotype data
        
    except Exception as e:
        logger.warning(f"⚠️ Batch effect correction failed: {e}")
        return pheno_df

def prepare_phenotype_data_with_comparison(config, qtl_type, results_dir):
    """Prepare phenotype data with comprehensive normalization comparison"""
    logger.info(f"🔧 Preparing {qtl_type} phenotype data with {config['normalization'][qtl_type]['method']} normalization...")
    
    try:
        # Read phenotype data
        pheno_file = config['input_files'][qtl_type]
        if not os.path.exists(pheno_file):
            raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")
            
        raw_df = pd.read_csv(pheno_file, sep='\t', index_col=0)
        logger.info(f"📊 Loaded {qtl_type} data: {raw_df.shape[0]} features, {raw_df.shape[1]} samples")
        
        # Store raw data for comparison
        raw_data_for_comparison = raw_df.copy()
        
        # Apply QC filters if enabled
        if config['qc'].get('filter_low_expressed', True):
            raw_df = filter_low_expressed_features(raw_df, config, qtl_type)
        
        # Apply proper normalization based on QTL type
        if config['qc'].get('normalize', True):
            normalized_df = apply_normalization(raw_df, config, qtl_type, results_dir)
        else:
            normalized_df = raw_df
        
        # Generate normalization comparison plots if enabled
        if config.get('enhanced_qc', {}).get('generate_normalization_plots', True):
            try:
                comparison = NormalizationComparison(config, results_dir)
                comparison_results = comparison.generate_comprehensive_comparison(
                    qtl_type, raw_data_for_comparison, normalized_df, 
                    config['normalization'][qtl_type]['method']
                )
                logger.info(f"📊 Normalization comparison completed for {qtl_type}")
            except Exception as e:
                logger.warning(f"⚠️ Normalization comparison failed: {e}")
        
        # Continue with original processing...
        # Read annotation data
        annotation_file = config['input_files']['annotations']
        annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
        logger.info(f"📊 Loaded annotation data: {annot_df.shape[0]} features")
        
        # Create BED format for QTLtools
        bed_data = create_qtltools_bed_format(normalized_df, annot_df, qtl_type)
        
        if len(bed_data) == 0:
            raise ValueError(f"❌ No valid features found for {qtl_type} after annotation matching")
        
        # Save BED file
        bed_file = os.path.join(results_dir, f"{qtl_type}_phenotypes.bed")
        bed_data.to_csv(bed_file, sep='\t', index=False)
        
        # Compress and index for QTLtools
        bed_gz = bed_file + '.gz'
        run_command(
            f"{config['paths']['bgzip']} -c {bed_file} > {bed_gz}",
            f"Compressing {qtl_type} BED file", config
        )
        
        run_command(
            f"{config['paths']['tabix']} -p bed {bed_gz}",
            f"Indexing {qtl_type} BED file", config
        )
        
        logger.info(f"✅ Prepared {qtl_type} data for QTLtools: {len(bed_data)} features")
        return bed_gz
        
    except Exception as e:
        logger.error(f"❌ Error preparing {qtl_type} data: {e}")
        raise

def run_cis_analysis(config, genotype_file, qtl_type, results_dir):
    """Run cis-QTL analysis using QTLtools with comprehensive options"""
    logger.info(f"🔍 Running {qtl_type} cis-QTL analysis with QTLtools...")
    
    try:
        # Prepare phenotype data with proper normalization
        bed_gz = prepare_phenotype_data_with_comparison(config, qtl_type, results_dir)
        if not bed_gz or not os.path.exists(bed_gz):
            raise FileNotFoundError(f"Could not prepare {qtl_type} phenotype data")
            
        # Use enhanced covariates if available
        enhanced_cov_file = os.path.join(results_dir, "enhanced_covariates.txt")
        if os.path.exists(enhanced_cov_file):
            covariates_file = enhanced_cov_file
            logger.info("✅ Using enhanced covariates with PCA components")
        else:
            covariates_file = config['input_files']['covariates']
            
        if not os.path.exists(covariates_file):
            raise FileNotFoundError(f"Covariates file not found: {covariates_file}")
        
        # Set output paths
        output_prefix = os.path.join(results_dir, f"{qtl_type}_cis")
        
        # Get QTL parameters
        qtl_config = config.get('qtl', {})
        
        # Build QTLtools cis command with comprehensive options
        if genotype_file.endswith('.bed'):  # PLINK format
            base_name = genotype_file.replace('.bed', '')
            cmd = (
                f"{config['paths']['qtltools']} cis "
                f"--plink {base_name} "
                f"--bed {bed_gz} "
                f"--cov {covariates_file} "
            )
        else:  # VCF format
            cmd = (
                f"{config['paths']['qtltools']} cis "
                f"--vcf {genotype_file} "
                f"--bed {bed_gz} "
                f"--cov {covariates_file} "
            )
        
        # Add QTLtools-specific parameters
        cmd += (
            f"--window {qtl_config.get('cis_window', 1000000)} "
            f"--permute {qtl_config.get('permutations', 1000)} "
            f"--maf-threshold {qtl_config.get('maf_threshold', 0.05)} "
            f"--ma-min {qtl_config.get('min_maf', 0.01)} "
            f"--seed {qtl_config.get('seed', 12345)} "
        )
        
        # Add conditional analysis if enabled
        if qtl_config.get('run_conditional', False):
            cmd += f" --conditional {qtl_config.get('conditional_quantiles', 10)}"
        
        # Add normal approximation if enabled
        if qtl_config.get('normal', True):
            cmd += " --normal"
        
        # Add output options
        cmd += (
            f"--out {output_prefix}_nominals.txt "
            f"--log {output_prefix}_qtl.log"
        )
        
        # Add performance options
        if config['performance'].get('num_threads', 1) > 1:
            cmd += f" --chunk {qtl_config.get('chunk_size', 100)}"
        
        logger.info(f"Running QTLtools cis command")
        run_command(cmd, f"{qtl_type} cis QTL analysis", config)
        
        # Check if results were generated
        nominals_file = f"{output_prefix}_nominals.txt"
        if not os.path.exists(nominals_file) or os.path.getsize(nominals_file) == 0:
            logger.warning(f"⚠️ No results generated for {qtl_type} cis analysis")
            return {
                'result_file': "",
                'nominals_file': nominals_file,
                'significant_count': 0,
                'status': 'completed'
            }
        
        # Run QTLtools correct for FDR
        fdr_cmd = (
            f"{config['paths']['qtltools']} correct "
            f"--qtl {nominals_file} "
            f"--out {output_prefix}_significant.txt "
            f"--threshold {qtl_config.get('fdr_threshold', 0.05)} "
            f"--log {output_prefix}_fdr.log"
        )
        
        if qtl_config.get('group', 'genes'):
            fdr_cmd += f" --grp {qtl_config['group']}"
        
        run_command(fdr_cmd, f"{qtl_type} cis FDR correction", config)
        
        # Count significant associations
        sig_file = f"{output_prefix}_significant.txt"
        significant_count = count_qtltools_significant(sig_file, qtl_config.get('fdr_threshold', 0.05))
        
        logger.info(f"✅ {qtl_type} cis: Found {significant_count} significant associations")
        
        return {
            'result_file': sig_file,
            'nominals_file': nominals_file,
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
    """Run trans-QTL analysis using QTLtools"""
    logger.info(f"🔍 Running {qtl_type} trans-QTL analysis with QTLtools...")
    
    try:
        # Prepare phenotype data with proper normalization
        bed_gz = prepare_phenotype_data_with_comparison(config, qtl_type, results_dir)
        if not bed_gz or not os.path.exists(bed_gz):
            raise FileNotFoundError(f"Could not prepare {qtl_type} phenotype data")
            
        # Use enhanced covariates if available
        enhanced_cov_file = os.path.join(results_dir, "enhanced_covariates.txt")
        if os.path.exists(enhanced_cov_file):
            covariates_file = enhanced_cov_file
            logger.info("✅ Using enhanced covariates with PCA components")
        else:
            covariates_file = config['input_files']['covariates']
            
        if not os.path.exists(covariates_file):
            raise FileNotFoundError(f"Covariates file not found: {covariates_file}")
        
        # Set output paths
        output_prefix = os.path.join(results_dir, f"{qtl_type}_trans")
        
        # Get QTL parameters
        qtl_config = config.get('qtl', {})
        
        # Build QTLtools trans command
        if genotype_file.endswith('.bed'):  # PLINK format
            base_name = genotype_file.replace('.bed', '')
            cmd = (
                f"{config['paths']['qtltools']} trans "
                f"--plink {base_name} "
                f"--bed {bed_gz} "
                f"--cov {covariates_file} "
            )
        else:  # VCF format
            cmd = (
                f"{config['paths']['qtltools']} trans "
                f"--vcf {genotype_file} "
                f"--bed {bed_gz} "
                f"--cov {covariates_file} "
            )
        
        # Add trans-specific parameters
        cmd += (
            f"--window {qtl_config.get('trans_window', 5000000)} "
            f"--maf-threshold {qtl_config.get('maf_threshold', 0.05)} "
            f"--ma-min {qtl_config.get('min_maf', 0.01)} "
            f"--seed {qtl_config.get('seed', 12345)} "
            f"--out {output_prefix}_nominals.txt "
            f"--log {output_prefix}_trans.log"
        )
        
        # Add performance options for trans analysis
        if config['performance'].get('num_threads', 1) > 1:
            cmd += f" --chunk {qtl_config.get('chunk_size', 100)}"
            
        run_command(cmd, f"{qtl_type} trans associations", config)
        
        # Check if results were generated
        nominals_file = f"{output_prefix}_nominals.txt"
        if not os.path.exists(nominals_file) or os.path.getsize(nominals_file) == 0:
            logger.warning(f"⚠️ No results generated for {qtl_type} trans analysis")
            return {
                'result_file': "",
                'nominals_file': nominals_file,
                'significant_count': 0,
                'status': 'completed'
            }
        
        # Apply FDR correction
        fdr_cmd = (
            f"{config['paths']['qtltools']} correct "
            f"--qtl {nominals_file} "
            f"--out {output_prefix}_significant.txt "
            f"--threshold {qtl_config.get('fdr_threshold', 0.05)} "
            f"--log {output_prefix}_trans_fdr.log"
        )
        
        run_command(fdr_cmd, f"{qtl_type} trans FDR correction", config)
        
        # Count significant associations
        sig_file = f"{output_prefix}_significant.txt"
        significant_count = count_qtltools_significant(sig_file, qtl_config.get('fdr_threshold', 0.05))
        
        logger.info(f"✅ {qtl_type} trans: Found {significant_count} significant associations")
            
        return {
            'result_file': sig_file,
            'nominals_file': nominals_file,
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

def count_qtltools_significant(result_file, fdr_threshold=0.05):
    """Count significant associations from QTLtools output"""
    if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
        return 0
    
    try:
        # QTLtools output format may vary, try different column names
        df = pd.read_csv(result_file, sep='\t')
        
        # Try different FDR column names used by QTLtools
        fdr_columns = ['FDR', 'fdr', 'qval', 'q-value', 'bh_fdr']
        fdr_column = None
        
        for col in fdr_columns:
            if col in df.columns:
                fdr_column = col
                break
        
        if fdr_column:
            significant_count = len(df[df[fdr_column] < fdr_threshold])
        else:
            # If no FDR column, count all results
            significant_count = len(df)
            logger.warning("No FDR column found in QTLtools output, counting all results")
        
        return significant_count
        
    except Exception as e:
        logger.warning(f"⚠️ Could not count significant associations: {e}")
        return 0

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