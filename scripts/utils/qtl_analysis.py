#!/usr/bin/env python3
"""
Enhanced QTL analysis utilities with cis/trans capabilities
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess

# Import the genotype processor
from .genotype_processing import GenotypeProcessor

logger = logging.getLogger('QTLPipeline')

def prepare_genotypes(config, results_dir):
    """Prepare genotype data with comprehensive pre-processing"""
    logger.info("🔧 Preparing genotype data with enhanced processing...")
    
    # Initialize genotype processor
    processor = GenotypeProcessor(config)
    
    # Get input file path
    input_file = config['input_files']['genotypes']
    
    # Process genotypes
    vcf_gz = processor.process_genotypes(input_file, results_dir)
    
    logger.info(f"✅ Genotype preparation completed: {vcf_gz}")
    return vcf_gz

def prepare_phenotype_data(config, qtl_type, results_dir):
    """Prepare phenotype data in BED format with comprehensive QC"""
    logger.info(f"🔧 Preparing {qtl_type} phenotype data...")
    
    try:
        # Read phenotype data
        pheno_file = config['input_files'][qtl_type]
        if not os.path.exists(pheno_file):
            raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")
            
        pheno_df = pd.read_csv(pheno_file, sep='\t', index_col=0)
        logger.info(f"📊 Loaded {qtl_type} data: {pheno_df.shape[0]} features, {pheno_df.shape[1]} samples")
        
        # Apply QC filters if enabled
        if config['qc'].get('filter_low_expressed', True) and qtl_type == 'expression':
            pheno_df = filter_low_expressed(pheno_df, config)
        
        if config['qc'].get('normalize', True):
            pheno_df = normalize_phenotypes(pheno_df)
        
        # Read annotation data
        annotation_file = config['input_files']['annotations']
        annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
        logger.info(f"📊 Loaded annotation data: {annot_df.shape[0]} features")
        
        # Create BED format
        bed_data = []
        missing_annotations = 0
        
        for feature_id in pheno_df.index:
            feature_annot = annot_df[annot_df['gene_id'] == feature_id]
            
            if len(feature_annot) == 0:
                missing_annotations += 1
                continue
                
            feature_annot = feature_annot.iloc[0]
            
            bed_entry = {
                'chr': feature_annot['chr'],
                'start': feature_annot['start'],
                'end': feature_annot['end'],
                'feature_id': feature_id,
                'score': 0,
                'strand': feature_annot.get('strand', '+')
            }
            
            # Add phenotype values for all samples
            for sample in pheno_df.columns:
                bed_entry[sample] = pheno_df.loc[feature_id, sample]
            
            bed_data.append(bed_entry)
        
        if missing_annotations > 0:
            logger.warning(f"⚠️ {missing_annotations} features missing annotations")
        
        # Create BED dataframe
        bed_df = pd.DataFrame(bed_data)
        
        if len(bed_df) == 0:
            raise ValueError(f"❌ No valid features found for {qtl_type} after annotation matching")
        
        # Save BED file
        bed_file = os.path.join(results_dir, f"{qtl_type}_phenotypes.bed")
        bed_df.to_csv(bed_file, sep='\t', index=False)
        
        # Compress and index
        bed_gz = bed_file + '.gz'
        run_command(
            f"{config['paths']['bgzip']} -c {bed_file} > {bed_gz}",
            f"Compressing {qtl_type} BED file", config
        )
        
        run_command(
            f"{config['paths']['tabix']} -p bed {bed_gz}",
            f"Indexing {qtl_type} BED file", config
        )
        
        logger.info(f"✅ Prepared {qtl_type} data: {len(bed_df)} features")
        return bed_gz
        
    except Exception as e:
        logger.error(f"❌ Error preparing {qtl_type} data: {e}")
        raise

def run_cis_analysis(config, vcf_gz, qtl_type, results_dir):
    """Run cis-QTL analysis with comprehensive error handling"""
    logger.info(f"🔍 Running {qtl_type} cis-QTL analysis...")
    
    try:
        # Prepare phenotype data
        bed_gz = prepare_phenotype_data(config, qtl_type, results_dir)
        covariates_file = config['input_files']['covariates']
        
        # Set output paths
        output_prefix = os.path.join(results_dir, f"{qtl_type}_cis")
        
        # Get QTL parameters
        qtl_config = config.get('qtl', {})
        
        # Base QTLTools cis command
        cmd = (
            f"{config['paths']['qtltools']} cis "
            f"--vcf {vcf_gz} "
            f"--bed {bed_gz} "
            f"--cov {covariates_file} "
            f"--window {qtl_config.get('cis_window', 1000000)} "
            f"--permute {qtl_config.get('permutations', 1000)} "
            f"--maf-threshold {qtl_config.get('maf_threshold', 0.05)} "
            f"--out {output_prefix}_nominals.txt "
            f"--log {output_prefix}_qtl.log"
        )
        
        # Add optional parameters
        if 'min_maf' in qtl_config:
            cmd += f" --ma-min {qtl_config['min_maf']}"
            
        run_command(cmd, f"{qtl_type} cis nominal associations", config)
        
        # Run FDR correction
        fdr_cmd = (
            f"{config['paths']['qtltools']} correct "
            f"--qtl {output_prefix}_nominals.txt "
            f"--out {output_prefix}_significant.txt "
            f"--threshold {qtl_config.get('fdr_threshold', 0.05)}"
        )
        
        run_command(fdr_cmd, f"{qtl_type} cis FDR correction", config)
        
        # Count significant associations
        sig_file = f"{output_prefix}_significant.txt"
        significant_count = 0
        if os.path.exists(sig_file):
            try:
                sig_df = pd.read_csv(sig_file, sep='\t')
                significant_count = len(sig_df)
                logger.info(f"✅ {qtl_type} cis: Found {significant_count} significant associations")
            except Exception as e:
                logger.warning(f"⚠️ Could not read significant associations file: {e}")
        else:
            logger.warning(f"⚠️ No significant associations file created for {qtl_type} cis")
            
        return {
            'result_file': sig_file,
            'nominals_file': f"{output_prefix}_nominals.txt",
            'significant_count': significant_count
        }
        
    except Exception as e:
        logger.error(f"❌ cis-QTL analysis failed for {qtl_type}: {e}")
        raise

def run_trans_analysis(config, vcf_gz, qtl_type, results_dir):
    """Run trans-QTL analysis"""
    logger.info(f"🔍 Running {qtl_type} trans-QTL analysis...")
    
    try:
        # Prepare phenotype data
        bed_gz = prepare_phenotype_data(config, qtl_type, results_dir)
        covariates_file = config['input_files']['covariates']
        
        # Set output paths
        output_prefix = os.path.join(results_dir, f"{qtl_type}_trans")
        
        # Get QTL parameters
        qtl_config = config.get('qtl', {})
        
        # QTLTools trans command (using matrix eQTL method)
        cmd = (
            f"{config['paths']['qtltools']} trans "
            f"--vcf {vcf_gz} "
            f"--bed {bed_gz} "
            f"--cov {covariates_file} "
            f"--window {qtl_config.get('trans_window', 5000000)} "
            f"--maf-threshold {qtl_config.get('maf_threshold', 0.05)} "
            f"--out {output_prefix}_nominals.txt "
            f"--log {output_prefix}_trans.log"
        )
        
        run_command(cmd, f"{qtl_type} trans associations", config)
        
        # Apply FDR correction
        fdr_cmd = (
            f"{config['paths']['qtltools']} correct "
            f"--qtl {output_prefix}_nominals.txt "
            f"--out {output_prefix}_significant.txt "
            f"--threshold {qtl_config.get('fdr_threshold', 0.05)}"
        )
        
        run_command(fdr_cmd, f"{qtl_type} trans FDR correction", config)
        
        # Count significant associations
        sig_file = f"{output_prefix}_significant.txt"
        significant_count = 0
        if os.path.exists(sig_file):
            try:
                sig_df = pd.read_csv(sig_file, sep='\t')
                significant_count = len(sig_df)
                logger.info(f"✅ {qtl_type} trans: Found {significant_count} significant associations")
            except Exception as e:
                logger.warning(f"⚠️ Could not read trans significant associations file: {e}")
        else:
            logger.warning(f"⚠️ No significant associations file created for {qtl_type} trans")
            
        return {
            'result_file': sig_file,
            'nominals_file': f"{output_prefix}_nominals.txt", 
            'significant_count': significant_count
        }
        
    except Exception as e:
        logger.error(f"❌ trans-QTL analysis failed for {qtl_type}: {e}")
        raise

def filter_low_expressed(pheno_df, config):
    """Filter lowly expressed genes"""
    threshold = config['qc'].get('expression_threshold', 0.1)
    
    # Calculate mean expression per gene
    mean_expression = pheno_df.mean(axis=1)
    
    # Filter genes with mean expression above threshold
    filtered_df = pheno_df[mean_expression > threshold]
    
    filtered_count = len(pheno_df) - len(filtered_df)
    logger.info(f"🔧 Filtered {filtered_count} low expressed genes: {len(pheno_df)} -> {len(filtered_df)}")
    
    return filtered_df

def normalize_phenotypes(pheno_df):
    """Normalize phenotype data"""
    # Log transform if needed (avoid log(0))
    if pheno_df.min().min() >= 0:
        # Add small constant to avoid log(0)
        pheno_df = np.log2(pheno_df + 1)
        logger.info("🔧 Applied log2 transformation to phenotype data")
    
    # Standardize (z-score normalization)
    pheno_df = (pheno_df - pheno_df.mean()) / pheno_df.std()
    logger.info("🔧 Applied z-score normalization to phenotype data")
    
    return pheno_df

def run_command(cmd, description, config, check=True):
    """Run shell command with comprehensive error handling"""
    logger.info(f"Executing: {description}")
    logger.debug(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True,
            executable='/bin/bash'
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