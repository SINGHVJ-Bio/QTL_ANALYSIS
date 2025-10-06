#!/usr/bin/env python3
"""
GWAS analysis utilities
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess

logger = logging.getLogger('QTLPipeline')

def run_gwas_analysis(config, vcf_gz, results_dir):
    """Run GWAS analysis using PLINK"""
    logger.info("Running GWAS analysis...")
    
    try:
        # Prepare GWAS data
        gwas_data = prepare_gwas_data(config, results_dir)
        
        # Run GWAS using PLINK
        gwas_results = run_plink_gwas(config, vcf_gz, gwas_data, results_dir)
        
        # Count significant associations
        significant_count = count_significant_gwas(gwas_results['result_file'])
        logger.info(f"Found {significant_count} significant GWAS associations")
        
        return {
            'result_file': gwas_results['result_file'],
            'significant_count': significant_count,
            'method': config['gwas'].get('method', 'linear')
        }
        
    except Exception as e:
        logger.error(f"GWAS analysis failed: {e}")
        raise

def prepare_gwas_data(config, results_dir):
    """Prepare GWAS phenotype and covariate data"""
    logger.info("Preparing GWAS data...")
    
    # Read GWAS phenotype file
    gwas_file = config['input_files'].get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
    gwas_df = pd.read_csv(gwas_file, sep='\t')
    
    # Ensure sample_id column exists
    if 'sample_id' not in gwas_df.columns:
        raise ValueError("GWAS phenotype file must contain 'sample_id' column")
    
    # Read covariates
    covariates_file = config['input_files']['covariates']
    cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
    
    # Merge phenotype and covariates
    phenotype_cols = [col for col in gwas_df.columns if col != 'sample_id']
    
    # Create PLINK compatible files
    plink_pheno_file = os.path.join(results_dir, "gwas_phenotype.txt")
    plink_cov_file = os.path.join(results_dir, "gwas_covariates.txt")
    
    # Prepare phenotype file for PLINK
    pheno_output = gwas_df[['sample_id'] + phenotype_cols]
    pheno_output.to_csv(plink_pheno_file, sep='\t', index=False)
    
    # Prepare covariate file for PLINK
    cov_output = cov_df.T.reset_index()
    cov_output.columns = ['sample_id'] + list(cov_df.index)
    cov_output.to_csv(plink_cov_file, sep='\t', index=False)
    
    return {
        'phenotype_file': plink_pheno_file,
        'covariate_file': plink_cov_file,
        'phenotype_cols': phenotype_cols
    }

def run_plink_gwas(config, vcf_gz, gwas_data, results_dir):
    """Run GWAS using PLINK"""
    logger.info("Running PLINK GWAS...")
    
    gwas_config = config.get('gwas', {})
    method = gwas_config.get('method', 'linear')
    
    # Convert VCF to PLINK format if needed
    plink_base = os.path.join(results_dir, "genotypes")
    
    if not os.path.exists(plink_base + ".bed"):
        run_command(
            f"{config['paths']['plink']} --vcf {vcf_gz} --make-bed --out {plink_base}",
            "Converting VCF to PLINK format", config
        )
    
    # Run GWAS for each phenotype
    all_results = []
    
    for phenotype in gwas_data['phenotype_cols']:
        logger.info(f"Running GWAS for phenotype: {phenotype}")
        
        output_prefix = os.path.join(results_dir, f"gwas_{phenotype}")
        
        # Build PLINK command
        cmd = f"{config['paths']['plink']} --bfile {plink_base} --pheno {gwas_data['phenotype_file']} --mpheno {phenotype}"
        
        if gwas_config.get('covariates', True):
            cmd += f" --covar {gwas_data['covariate_file']}"
        
        if method == 'linear':
            cmd += " --linear --ci 0.95"
        elif method == 'logistic':
            cmd += " --logistic --ci 0.95"
        else:
            raise ValueError(f"Unsupported GWAS method: {method}")
        
        # Add filters
        cmd += f" --maf {gwas_config.get('maf_threshold', 0.01)}"
        cmd += f" --out {output_prefix}"
        
        run_command(cmd, f"GWAS for {phenotype}", config)
        
        # Process results
        result_file = f"{output_prefix}.assoc.{method}"
        if os.path.exists(result_file):
            results_df = pd.read_csv(result_file, delim_whitespace=True)
            results_df['PHENOTYPE'] = phenotype
            all_results.append(results_df)
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(results_dir, "gwas_combined_results.txt")
        combined_results.to_csv(combined_file, sep='\t', index=False)
        
        return {
            'result_file': combined_file,
            'individual_files': [f"{os.path.join(results_dir, f'gwas_{phenotype}.assoc.{method}')}" 
                               for phenotype in gwas_data['phenotype_cols']]
        }
    else:
        raise ValueError("No GWAS results were generated")

def count_significant_gwas(result_file, pval_threshold=5e-8):
    """Count significant GWAS hits"""
    if not os.path.exists(result_file):
        return 0
    
    try:
        results_df = pd.read_csv(result_file, sep='\t')
        if 'P' in results_df.columns:
            return len(results_df[results_df['P'] < pval_threshold])
        else:
            return 0
    except:
        return 0

def run_command(cmd, description, config):
    """Run shell command with error handling"""
    logger.info(f"Executing: {description}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            executable='/bin/bash'
        )
        logger.info(f"✓ {description} completed successfully")
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        raise