#!/usr/bin/env python3
"""
Enhanced input validation utilities with genotype format detection
"""

import os
import pandas as pd
import yaml
from pathlib import Path
import subprocess
import logging

logger = logging.getLogger('QTLPipeline')

def validate_inputs(config):
    """Validate all input files and data consistency with genotype format detection"""
    input_files = config['input_files']
    
    errors = []
    warnings = []
    
    print("🔍 Validating input files...")
    
    # Check mandatory files
    mandatory_files = ['genotypes', 'covariates', 'annotations']
    for file_type in mandatory_files:
        file_path = input_files.get(file_type)
        if not file_path:
            errors.append(f"Missing mandatory input file: {file_type}")
            continue
            
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path} (for {file_type})")
        else:
            print(f"✓ Found {file_type} file: {file_path}")
            
            # Special validation for genotype files
            if file_type == 'genotypes':
                genotype_errors, genotype_warnings = validate_genotype_file(file_path, config)
                errors.extend(genotype_errors)
                warnings.extend(genotype_warnings)
    
    # Check phenotype files based on analysis types
    analysis_types = get_qtl_types_from_config(config)
    for analysis_type in analysis_types:
        file_path = input_files.get(analysis_type)
        if not file_path:
            errors.append(f"Missing phenotype file for {analysis_type}")
            continue
            
        if not os.path.exists(file_path):
            errors.append(f"Phenotype file not found: {file_path} (for {analysis_type})")
        else:
            validate_phenotype_file(file_path, analysis_type, warnings)
    
    # Check GWAS phenotype file if GWAS is enabled
    if config['analysis'].get('run_gwas', False):
        gwas_file = input_files.get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
        if not gwas_file:
            errors.append("GWAS enabled but no gwas_phenotype specified")
        elif not os.path.exists(gwas_file):
            errors.append(f"GWAS phenotype file not found: {gwas_file}")
        else:
            validate_gwas_phenotype_file(gwas_file, warnings)
    
    # Check sample concordance
    if config['qc'].get('check_sample_concordance', True):
        sample_warnings = check_sample_concordance(config, input_files)
        warnings.extend(sample_warnings)
    
    # Report results
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if errors:
        print("\n❌ VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Input validation failed")
    else:
        print("\n✅ All inputs validated successfully!")

def validate_genotype_file(file_path, config):
    """Validate genotype file format and content"""
    errors = []
    warnings = []
    
    # Detect format
    format_info = detect_genotype_format(file_path)
    print(f"✓ Genotype format: {format_info['format']}")
    
    # Check for required tools based on format
    if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
        if not check_tool_exists(config['paths']['bcftools']):
            errors.append("bcftools not found but required for VCF/BCF processing")
    
    # Check file accessibility
    if not os.access(file_path, os.R_OK):
        errors.append(f"Cannot read genotype file: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
    if file_size > 10:
        warnings.append(f"Genotype file is large ({file_size:.2f} GB), analysis may be slow")
    
    # Validate specific format requirements
    if format_info['format'] in ['vcf', 'vcf.gz']:
        vcf_errors, vcf_warnings = validate_vcf_file(file_path, config)
        errors.extend(vcf_errors)
        warnings.extend(vcf_warnings)
    
    return errors, warnings

def detect_genotype_format(file_path):
    """Detect genotype file format"""
    if file_path.endswith('.vcf.gz'):
        return {'format': 'vcf.gz', 'compressed': True}
    elif file_path.endswith('.vcf'):
        return {'format': 'vcf', 'compressed': False}
    elif file_path.endswith('.bcf'):
        return {'format': 'bcf', 'compressed': True}
    elif file_path.endswith('.bed'):
        return {'format': 'plink_bed', 'compressed': False}
    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        return {'format': 'hdf5', 'compressed': True}
    else:
        # Try to detect by content
        try:
            if file_path.endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline()
            else:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
            
            if first_line.startswith('##fileformat=VCF'):
                return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
            elif first_line.startswith('#CHROM'):
                return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
            else:
                return {'format': 'unknown', 'compressed': file_path.endswith('.gz')}
        except:
            return {'format': 'unknown', 'compressed': file_path.endswith('.gz')}

def validate_vcf_file(file_path, config):
    """Validate VCF file structure"""
    errors = []
    warnings = []
    
    try:
        # Check if bcftools can read the file
        result = subprocess.run(
            f"{config['paths']['bcftools']} view -h {file_path} | head -5",
            shell=True, capture_output=True, text=True, check=False
        )
        
        if result.returncode != 0:
            errors.append(f"VCF file cannot be read by bcftools: {file_path}")
            return errors, warnings
        
        # Check for GT field in a sample of variants
        result = subprocess.run(
            f"{config['paths']['bcftools']} view {file_path} | grep -v '^#' | head -10",
            shell=True, capture_output=True, text=True, check=False
        )
        
        if "GT" not in result.stdout:
            warnings.append("GT field not found in VCF - ensure genotypes are properly encoded")
        
        # Check sample names
        result = subprocess.run(
            f"{config['paths']['bcftools']} query -l {file_path}",
            shell=True, capture_output=True, text=True, check=False
        )
        
        samples = [s.strip() for s in result.stdout.split('\n') if s.strip()]
        if not samples:
            errors.append("No samples found in VCF file")
        else:
            print(f"✓ Found {len(samples)} samples in genotype file")
            
    except Exception as e:
        errors.append(f"VCF validation error: {e}")
    
    return errors, warnings

def check_tool_exists(tool_name):
    """Check if a required tool exists in PATH"""
    try:
        subprocess.run(f"which {tool_name}", shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_qtl_types_from_config(config):
    """Extract QTL types from config"""
    qtl_types = config['analysis']['qtl_types']
    if qtl_types == 'all':
        return ['eqtl', 'pqtl', 'sqtl']
    elif isinstance(qtl_types, str):
        return [t.strip() for t in qtl_types.split(',')]
    else:
        return qtl_types

def validate_phenotype_file(file_path, qtl_type, warnings):
    """Validate phenotype file structure"""
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        print(f"✓ {qtl_type}: {df.shape[0]} features, {df.shape[1]} samples")
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            warnings.append(f"{qtl_type} has {missing_count} missing values")
            
        # Check for constant features
        constant_features = (df.nunique() == 1).sum()
        if constant_features > 0:
            warnings.append(f"{qtl_type} has {constant_features} constant features")
            
    except Exception as e:
        raise ValueError(f"Error reading {qtl_type} file {file_path}: {e}")

def validate_gwas_phenotype_file(file_path, warnings):
    """Validate GWAS phenotype file"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"✓ GWAS phenotype: {df.shape[0]} samples, {df.shape[1]-1} phenotypes")
        
        # Check required columns
        if 'sample_id' not in df.columns:
            raise ValueError("GWAS phenotype file must contain 'sample_id' column")
            
    except Exception as e:
        raise ValueError(f"Error reading GWAS phenotype file {file_path}: {e}")

def check_sample_concordance(config, input_files):
    """Check sample concordance across all input files"""
    warnings = []
    
    try:
        # Get samples from genotype file
        geno_file = input_files['genotypes']
        result = subprocess.run(
            f"{config['paths']['bcftools']} query -l {geno_file}",
            shell=True, capture_output=True, text=True, check=False
        )
        geno_samples = set([s.strip() for s in result.stdout.split('\n') if s.strip()])
        
        # Get samples from covariate file
        cov_df = pd.read_csv(input_files['covariates'], sep='\t', index_col=0)
        cov_samples = set(cov_df.columns)
        
        # Check genotype-covariate overlap
        overlap = geno_samples.intersection(cov_samples)
        if len(overlap) == 0:
            warnings.append("No sample overlap between genotypes and covariates")
        elif len(overlap) < min(len(geno_samples), len(cov_samples)):
            warnings.append(f"Partial sample overlap: {len(overlap)} common samples")
        
        # Check phenotype files
        analysis_types = get_qtl_types_from_config(config)
        for analysis_type in analysis_types:
            if analysis_type in input_files:
                pheno_df = pd.read_csv(input_files[analysis_type], sep='\t', index_col=0)
                pheno_samples = set(pheno_df.columns)
                overlap = geno_samples.intersection(pheno_samples)
                if len(overlap) == 0:
                    warnings.append(f"No sample overlap between genotypes and {analysis_type}")
                elif len(overlap) < min(len(geno_samples), len(pheno_samples)):
                    warnings.append(f"Partial sample overlap for {analysis_type}: {len(overlap)} common samples")
                    
    except Exception as e:
        warnings.append(f"Sample concordance check failed: {e}")
    
    return warnings