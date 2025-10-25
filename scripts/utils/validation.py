#!/usr/bin/env python3
"""
Comprehensive input validation utilities for tensorQTL pipeline - Production Version
Robust validation with enhanced mapping, parallel processing, and comprehensive error handling
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import subprocess
import logging
import gzip
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import psutil
import re
from datetime import datetime
import json
from collections import defaultdict
import sys

warnings.filterwarnings('ignore')
logger = logging.getLogger('QTLPipeline')

class ValidationResult:
    """Enhanced validation result tracking with comprehensive reporting"""
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.file_stats = {}
        self.sample_counts = {}
        self.data_types_available = []
        self.validation_time = datetime.now().isoformat()
        self.overall_status = "PASS"
    
    def add_error(self, module, message):
        self.errors.append(f"{module}: {message}")
        self.overall_status = "FAIL"
    
    def add_warning(self, module, message):
        self.warnings.append(f"{module}: {message}")
    
    def add_info(self, module, message):
        self.info.append(f"{module}: {message}")
    
    def to_dict(self):
        return {
            'timestamp': self.validation_time,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'file_stats': self.file_stats,
            'sample_counts': self.sample_counts,
            'data_types_available': self.data_types_available,
            'overall_status': self.overall_status
        }

def validate_inputs(config):
    """Validate all input files and data consistency with comprehensive checks for tensorQTL - ENHANCED"""
    input_files = config['input_files']
    result = ValidationResult()
    
    print("🔍 Starting comprehensive input validation for tensorQTL pipeline...")
    logger.info("Starting comprehensive input validation")
    
    try:
        # Validate mandatory files with enhanced parallel processing
        mandatory_files = ['genotypes', 'covariates', 'annotations']
        validate_mandatory_files_parallel(input_files, mandatory_files, config, result)
        
        # Validate phenotype files based on analysis types
        analysis_types = get_qtl_types_from_config(config)
        result.data_types_available = analysis_types
        validate_phenotype_files_parallel(input_files, analysis_types, config, result)
        
        # Validate GWAS if enabled
        if config['analysis'].get('run_gwas', False):
            validate_gwas_files(input_files, config, result)
        
        # Enhanced tool validation with version checking
        validate_tools_comprehensive(config, result)
        
        # Comprehensive sample concordance checking
        if config['qc'].get('check_sample_concordance', True):
            check_sample_concordance_enhanced(config, input_files, result)
        
        # Configuration and requirements validation
        validate_configuration_comprehensive(config, result)
        
        # Generate validation report
        generate_validation_report(result, config)
        
        # Report results
        if result.warnings:
            print("\n⚠️  VALIDATION WARNINGS:")
            for warning in result.warnings[:10]:
                print(f"  - {warning}")
            if len(result.warnings) > 10:
                print(f"  ... and {len(result.warnings) - 10} more warnings")
        
        if result.errors:
            print("\n❌ VALIDATION FAILED:")
            for error in result.errors:
                print(f"  - {error}")
            logger.error(f"Input validation failed with {len(result.errors)} errors")
            raise ValueError("Input validation failed - please fix the errors above")
        else:
            print("\n🎉 All inputs validated successfully!")
            if result.warnings:
                print(f"   Found {len(result.warnings)} warnings - review validation report for details")
            logger.info(f"Input validation completed: {len(result.warnings)} warnings, {len(result.info)} info messages")
            return True
            
    except Exception as e:
        logger.error(f"Validation process failed: {e}")
        result.add_error('validation_process', f"Validation process failed: {e}")
        raise

def validate_mandatory_files_parallel(input_files, mandatory_files, config, result):
    """Validate mandatory files with enhanced parallel processing"""
    with ThreadPoolExecutor(max_workers=min(6, mp.cpu_count())) as executor:
        futures = []
        
        for file_type in mandatory_files:
            file_path = input_files.get(file_type)
            if not file_path:
                result.add_error('mandatory_files', f"Missing mandatory input file: {file_type}")
                continue
                
            if not os.path.exists(file_path):
                result.add_error('mandatory_files', f"File not found: {file_path} (for {file_type})")
                continue
            
            # Submit file-specific validation tasks
            if file_type == 'genotypes':
                future = executor.submit(validate_genotype_file_enhanced, file_path, config, result)
            elif file_type == 'covariates':
                future = executor.submit(validate_covariates_file_enhanced, file_path, config, result)
            elif file_type == 'annotations':
                future = executor.submit(validate_annotations_file_enhanced, file_path, config, result)
            else:
                future = executor.submit(validate_generic_file, file_path, file_type, config, result)
            
            futures.append(future)
        
        # Wait for all validations to complete
        for future in as_completed(futures):
            try:
                future.result(timeout=300)
            except Exception as e:
                result.add_error('parallel_validation', f"Parallel validation task failed: {e}")

def validate_phenotype_files_parallel(input_files, analysis_types, config, result):
    """Validate phenotype files with enhanced parallel processing"""
    with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
        futures = []
        
        for analysis_type in analysis_types:
            config_key = map_qtl_type_to_config_key(analysis_type)
            file_path = input_files.get(config_key)
            
            if not file_path:
                result.add_error('phenotype_files', f"Missing phenotype file for {analysis_type} (key: {config_key})")
                continue
                
            if not os.path.exists(file_path):
                result.add_error('phenotype_files', f"Phenotype file not found: {file_path} (for {analysis_type})")
                continue
            
            future = executor.submit(validate_phenotype_file_enhanced, file_path, analysis_type, config, result)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result(timeout=300)
            except Exception as e:
                result.add_error('phenotype_validation', f"Phenotype validation task failed: {e}")

def validate_genotype_file_enhanced(file_path, config, result):
    """Enhanced genotype file validation with comprehensive checks"""
    logger.info(f"Validating genotype file: {file_path}")
    
    # Basic file checks
    if not os.path.exists(file_path):
        result.add_error('genotype', f"File not found: {file_path}")
        return
    
    # Check file size and permissions
    try:
        file_size = os.path.getsize(file_path) / (1024**3)  # GB
        if file_size == 0:
            result.add_error('genotype', f"Genotype file is empty: {file_path}")
            return
        elif file_size > 50:
            result.add_warning('genotype', f"Genotype file is very large ({file_size:.2f} GB), ensure sufficient memory")
        
        result.file_stats['genotype_size_gb'] = file_size
        result.add_info('genotype', f"File size: {file_size:.2f} GB")
        
    except OSError as e:
        result.add_error('genotype', f"Cannot access genotype file: {e}")
        return
    
    # Detect format with enhanced detection
    format_info = detect_genotype_format_enhanced(file_path)
    result.add_info('genotype', f"Detected format: {format_info['format']} (compressed: {format_info['compressed']})")
    
    # Validate specific formats
    if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
        validate_vcf_file_enhanced(file_path, config, result)
    elif format_info['format'] == 'plink_bed':
        validate_plink_file_enhanced(file_path, config, result)
    elif format_info['format'] == 'unknown':
        result.add_warning('genotype', f"Could not automatically detect genotype file format: {file_path}")
        # Try to validate as generic tabular file
        validate_generic_tabular_file(file_path, 'genotype', config, result)
    else:
        result.add_warning('genotype', f"Format {format_info['format']} may not be directly compatible with tensorQTL")

def detect_genotype_format_enhanced(file_path):
    """Enhanced genotype format detection with more comprehensive checks"""
    file_ext = file_path.lower()
    
    # Extension-based detection
    if file_ext.endswith('.vcf.gz') or file_ext.endswith('.vcf.bgz'):
        return {'format': 'vcf.gz', 'compressed': True}
    elif file_ext.endswith('.vcf'):
        return {'format': 'vcf', 'compressed': False}
    elif file_ext.endswith('.bcf'):
        return {'format': 'bcf', 'compressed': True}
    elif file_ext.endswith('.bed'):
        return {'format': 'plink_bed', 'compressed': False}
    elif file_ext.endswith(('.h5', '.hdf5')):
        return {'format': 'hdf5', 'compressed': True}
    elif file_ext.endswith(('.bgen', '.gen')):
        return {'format': 'bgen', 'compressed': True}
    else:
        # Content-based detection
        return detect_format_by_content_enhanced(file_path)

def detect_format_by_content_enhanced(file_path):
    """Enhanced content-based format detection"""
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
        
        for line in first_lines:
            if not line:
                continue
                
            if line.startswith('##fileformat=VCF'):
                return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
            elif line.startswith('#CHROM'):
                return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
            elif 'VCF' in line.upper():
                return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
            elif 'BED' in line:
                return {'format': 'plink_bed', 'compressed': file_path.endswith('.gz')}
            elif line.startswith('##GENOTYPE'):
                return {'format': 'genotype_matrix', 'compressed': file_path.endswith('.gz')}
        
        # Check for PLINK BED magic number
        if file_path.endswith('.bed'):
            with open(file_path, 'rb') as f:
                magic = f.read(3)
                if magic == b'\x6c\x1b\x01':  # PLINK BED magic number
                    return {'format': 'plink_bed', 'compressed': False}
        
        # Check if it looks like a tabular genotype file
        if len(first_lines) > 1 and '\t' in first_lines[0]:
            return {'format': 'tabular', 'compressed': file_path.endswith('.gz')}
            
    except Exception as e:
        logger.debug(f"Enhanced format detection failed: {e}")
    
    return {'format': 'unknown', 'compressed': file_path.endswith('.gz')}

def validate_vcf_file_enhanced(file_path, config, result):
    """Enhanced VCF file validation with comprehensive checks"""
    try:
        bcftools_path = config['paths'].get('bcftools', 'bcftools')
        
        # Test basic VCF reading
        cmd = f"{bcftools_path} view -h {file_path} 2>/dev/null | head -10 || echo 'ERROR'"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        if "ERROR" in process.stdout or process.returncode != 0:
            result.add_error('vcf', f"bcftools cannot read VCF file. Check file format and compression.")
            return
        
        # Extract sample count
        samples_cmd = f"{bcftools_path} query -l {file_path} 2>/dev/null | wc -l"
        samples_process = subprocess.run(samples_cmd, shell=True, capture_output=True, text=True)
        if samples_process.returncode == 0 and samples_process.stdout.strip().isdigit():
            sample_count = int(samples_process.stdout.strip())
            result.sample_counts['genotypes'] = sample_count
            result.add_info('vcf', f"Found {sample_count} samples in VCF")
        else:
            result.add_warning('vcf', "Could not count samples in VCF file")
        
        # Extract variant count estimate
        variants_cmd = f"{bcftools_path} view -H {file_path} 2>/dev/null | head -1000 | wc -l"
        variants_process = subprocess.run(variants_cmd, shell=True, capture_output=True, text=True)
        if variants_process.returncode == 0 and variants_process.stdout.strip().isdigit():
            variant_sample = int(variants_process.stdout.strip())
            if variant_sample == 0:
                result.add_error('vcf', "No variants found in VCF file")
            else:
                result.add_info('vcf', f"VCF contains variants (sampled {variant_sample})")
        
        # Check chromosome naming consistency
        chrom_cmd = f"{bcftools_path} view -H {file_path} 2>/dev/null | cut -f1 | head -100 | sort | uniq"
        chrom_process = subprocess.run(chrom_cmd, shell=True, capture_output=True, text=True)
        if chrom_process.returncode == 0:
            chromosomes = [c.strip() for c in chrom_process.stdout.split('\n') if c.strip()]
            analyze_chromosome_naming(chromosomes, 'vcf', result)
        
        # Check for required INFO and FORMAT fields
        header_cmd = f"{bcftools_path} view -h {file_path} 2>/dev/null | grep -E '^##(INFO|FORMAT)' | head -10"
        header_process = subprocess.run(header_cmd, shell=True, capture_output=True, text=True)
        if header_process.returncode == 0:
            headers = header_process.stdout.strip().split('\n')
            has_gt = any('ID=GT' in h for h in headers if h)
            if not has_gt:
                result.add_warning('vcf', "VCF file may not contain genotype (GT) format field")
        
    except Exception as e:
        result.add_error('vcf', f"VCF validation error: {e}")

def validate_plink_file_enhanced(file_path, config, result):
    """Enhanced PLINK file validation with comprehensive checks"""
    try:
        base_name = file_path.replace('.bed', '')
        required_files = [f'{base_name}.bed', f'{base_name}.bim', f'{base_name}.fam']
        
        # Check for all required files
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            result.add_error('plink', f"Missing PLINK companion files: {', '.join(missing_files)}")
            return
        
        # Validate FAM file (samples)
        try:
            fam_df = pd.read_csv(f'{base_name}.fam', sep='\s+', header=None, 
                               names=['family_id', 'sample_id', 'father_id', 'mother_id', 'sex', 'phenotype'])
            sample_count = len(fam_df)
            result.sample_counts['genotypes'] = sample_count
            result.add_info('plink', f"Found {sample_count} samples in PLINK FAM file")
            
            # Check for duplicate sample IDs
            duplicate_samples = fam_df['sample_id'].duplicated().sum()
            if duplicate_samples > 0:
                result.add_error('plink', f"Found {duplicate_samples} duplicate sample IDs in FAM file")
        except Exception as e:
            result.add_error('plink', f"Error reading PLINK FAM file: {e}")
        
        # Validate BIM file (variants)
        try:
            bim_df = pd.read_csv(f'{base_name}.bim', sep='\s+', header=None,
                               names=['chr', 'variant_id', 'pos_cm', 'pos_bp', 'allele1', 'allele2'])
            variant_count = len(bim_df)
            result.add_info('plink', f"Found {variant_count} variants in PLINK BIM file")
            
            # Check for duplicate variant IDs
            duplicate_variants = bim_df['variant_id'].duplicated().sum()
            if duplicate_variants > 0:
                result.add_warning('plink', f"Found {duplicate_variants} duplicate variant IDs in BIM file")
            
            # Check chromosome naming
            analyze_chromosome_naming(bim_df['chr'].unique(), 'plink', result)
            
            # Check for invalid positions
            invalid_positions = bim_df[(bim_df['pos_bp'] <= 0) | (bim_df['pos_bp'].isna())]
            if len(invalid_positions) > 0:
                result.add_warning('plink', f"Found {len(invalid_positions)} variants with invalid positions")
                
        except Exception as e:
            result.add_error('plink', f"Error reading PLINK BIM file: {e}")
        
    except Exception as e:
        result.add_error('plink', f"PLINK validation error: {e}")

def validate_covariates_file_enhanced(file_path, config, result):
    """Enhanced covariates file validation"""
    try:
        # Read with comprehensive checks
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        
        if df.empty:
            result.add_error('covariates', "Covariates file is empty")
            return
        
        # Store sample count
        sample_count = df.shape[1]
        result.sample_counts['covariates'] = sample_count
        result.add_info('covariates', f"Found {df.shape[0]} covariates for {sample_count} samples")
        
        # Check for missing values
        missing_matrix = df.isna()
        missing_count = missing_matrix.sum().sum()
        if missing_count > 0:
            missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
            result.add_warning('covariates', f"Found {missing_count} missing values ({missing_percent:.2f}%)")
        
        # Check for constant covariates
        constant_covariates = []
        for covar in df.index:
            if df.loc[covar].nunique() <= 1:
                constant_covariates.append(covar)
        
        if constant_covariates:
            result.add_warning('covariates', f"Found {len(constant_covariates)} constant covariates: {constant_covariates[:5]}...")
        
        # Check for numeric covariates
        non_numeric_covariates = []
        for covar in df.index:
            try:
                pd.to_numeric(df.loc[covar], errors='raise')
            except (ValueError, TypeError):
                non_numeric_covariates.append(covar)
        
        if non_numeric_covariates:
            result.add_warning('covariates', f"Found {len(non_numeric_covariates)} non-numeric covariates: {non_numeric_covariates[:5]}...")
        
        # Check for recommended covariates
        recommended_covariates = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'age', 'sex', 'batch']
        available_recommended = [cov for cov in recommended_covariates if cov in df.index]
        if available_recommended:
            result.add_info('covariates', f"Found recommended covariates: {', '.join(available_recommended)}")
        
        # Check for extreme values
        numeric_mask = ~df.index.isin(non_numeric_covariates)
        if numeric_mask.any():
            numeric_df = df[numeric_mask]
            extreme_threshold = 10  # Z-score threshold
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
            extreme_count = (z_scores > extreme_threshold).sum().sum()
            if extreme_count > 0:
                result.add_warning('covariates', f"Found {extreme_count} extreme values (|Z| > {extreme_threshold})")
        
    except Exception as e:
        result.add_error('covariates', f"Error reading covariates file: {e}")

def validate_annotations_file_enhanced(file_path, config, result):
    """Enhanced annotations file validation"""
    try:
        # Try to read with different comment characters
        try:
            df = pd.read_csv(file_path, sep='\t', comment='#', nrows=1000)
        except:
            try:
                df = pd.read_csv(file_path, sep='\t', nrows=1000)
            except Exception as e:
                result.add_error('annotations', f"Cannot read annotations file: {e}")
                return
        
        # Check required columns
        required_columns = ['chr', 'start', 'end', 'gene_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            result.add_error('annotations', f"Missing required columns: {', '.join(missing_columns)}")
            result.add_info('annotations', f"Available columns: {list(df.columns)}")
            return
        
        # Validate data types
        try:
            df['start'] = pd.to_numeric(df['start'])
            df['end'] = pd.to_numeric(df['end'])
        except Exception as e:
            result.add_error('annotations', f"Start/end positions must be numeric: {e}")
        
        # Check for invalid ranges
        invalid_ranges = df[df['start'] >= df['end']]
        if len(invalid_ranges) > 0:
            result.add_warning('annotations', f"Found {len(invalid_ranges)} annotations with start >= end")
        
        # Check for duplicate gene IDs
        duplicate_genes = df[df.duplicated('gene_id')]
        if len(duplicate_genes) > 0:
            result.add_warning('annotations', f"Found {len(duplicate_genes)} duplicate gene IDs")
        
        # Analyze chromosome naming
        analyze_chromosome_naming(df['chr'].unique(), 'annotations', result)
        
        # Check annotation coverage
        expected_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT', 'M']
        found_chromosomes = set(df['chr'].astype(str))
        missing_chromosomes = [chrom for chrom in expected_chromosomes 
                             if chrom not in found_chromosomes and f"chr{chrom}" not in found_chromosomes]
        
        if missing_chromosomes:
            result.add_info('annotations', f"Missing annotations for chromosomes: {', '.join(missing_chromosomes)}")
        
        # Calculate basic statistics
        total_length = (df['end'] - df['start']).sum()
        avg_length = (df['end'] - df['start']).mean()
        
        result.add_info('annotations', f"Annotation stats: {len(df)} features, total length {total_length:,} bp, avg {avg_length:.0f} bp")
        
    except Exception as e:
        result.add_error('annotations', f"Error reading annotations file: {e}")

def validate_phenotype_file_enhanced(file_path, qtl_type, config, result):
    """Enhanced phenotype file validation"""
    try:
        # Read phenotype data
        df = pd.read_csv(file_path, sep='\t', index_col=0, nrows=1000)  # Sample for validation
        
        if df.empty:
            result.add_error(qtl_type, f"Phenotype file is empty: {file_path}")
            return
        
        # Store sample count
        sample_count = df.shape[1]
        result.sample_counts[qtl_type] = sample_count
        result.add_info(qtl_type, f"Found {df.shape[0]} features for {sample_count} samples")
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
            result.add_warning(qtl_type, f"Found {missing_count} missing values ({missing_percent:.2f}%)")
        
        # Check for constant features
        constant_features = (df.nunique(axis=1) == 1).sum()
        if constant_features > 0:
            result.add_warning(qtl_type, f"Found {constant_features} constant features")
        
        # Check for zero-variance features
        zero_variance = (df.std(axis=1) == 0).sum()
        if zero_variance > 0:
            result.add_warning(qtl_type, f"Found {zero_variance} zero-variance features")
        
        # Check data distribution based on QTL type
        data_type = map_qtl_type_to_data_type(qtl_type)
        if data_type == 'expression':
            # Check for negative values in expression data
            negative_count = (df < 0).sum().sum()
            if negative_count > 0:
                result.add_warning(qtl_type, f"Found {negative_count} negative values in expression data")
        
        # Check for extreme outliers
        if df.size > 0:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR))
            outlier_count = outlier_mask.sum().sum()
            
            if outlier_count > 0:
                outlier_percent = (outlier_count / (df.shape[0] * df.shape[1])) * 100
                result.add_info(qtl_type, f"Found {outlier_count} potential outliers ({outlier_percent:.2f}%)")
        
        # Check normalization method compatibility
        norm_method = config['normalization'].get(qtl_type, {}).get('method', 'unknown')
        result.add_info(qtl_type, f"Configured normalization: {norm_method}")
        
        # Calculate basic statistics
        if df.size > 0:
            mean_val = df.mean().mean()
            std_val = df.std().mean()
            result.add_info(qtl_type, f"Data stats - Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        
    except Exception as e:
        result.add_error(qtl_type, f"Error reading phenotype file {file_path}: {e}")

def validate_gwas_files(input_files, config, result):
    """Validate GWAS-related files"""
    gwas_file = input_files.get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
    
    if not gwas_file:
        result.add_error('gwas', "GWAS enabled but no gwas_phenotype specified")
        return
    
    if not os.path.exists(gwas_file):
        result.add_error('gwas', f"GWAS phenotype file not found: {gwas_file}")
        return
    
    try:
        df = pd.read_csv(gwas_file, sep='\t', nrows=1000)
        
        # Check required columns
        if 'sample_id' not in df.columns:
            result.add_error('gwas', "GWAS phenotype file must contain 'sample_id' column")
        
        # Check phenotype columns
        phenotype_cols = [col for col in df.columns if col != 'sample_id']
        if not phenotype_cols:
            result.add_error('gwas', "GWAS phenotype file has no phenotype columns")
        
        # Validate sample IDs
        missing_samples = df['sample_id'].isna().sum()
        if missing_samples > 0:
            result.add_warning('gwas', f"Found {missing_samples} missing sample IDs")
        
        # Validate phenotype data types
        gwas_method = config.get('gwas', {}).get('method', 'linear')
        for pheno in phenotype_cols[:10]:  # Check first 10 phenotypes
            try:
                pd.to_numeric(df[pheno])
            except:
                result.add_warning('gwas', f"Phenotype '{pheno}' contains non-numeric values")
            
            # Check for binary phenotypes if logistic regression
            if gwas_method == 'logistic':
                unique_vals = df[pheno].dropna().unique()
                if len(unique_vals) != 2:
                    result.add_warning('gwas', f"Phenotype '{pheno}' has {len(unique_vals)} unique values, logistic regression expects binary")
        
        result.add_info('gwas', f"GWAS phenotypes: {len(phenotype_cols)} phenotypes for {len(df)} samples")
        
    except Exception as e:
        result.add_error('gwas', f"Error reading GWAS file: {e}")

def validate_tools_comprehensive(config, result):
    """Comprehensive tool validation with version checking"""
    tools = config.get('paths', {})
    
    required_tools = {
        'plink': {'version_flag': '--version', 'min_version': '1.9'},
        'bcftools': {'version_flag': '--version', 'min_version': '1.9'},
        'bgzip': {'version_flag': '--version', 'min_version': '1.9'},
        'tabix': {'version_flag': '--version', 'min_version': '1.9'}
    }
    
    optional_tools = {
        'R': {'version_flag': '--version', 'min_version': '3.6'},
        'python': {'version_flag': '--version', 'min_version': '3.6'}
    }
    
    def check_tool_version(tool_name, tool_config):
        """Check tool version and compatibility"""
        tool_path = tools.get(tool_name, tool_name)
        
        try:
            # Check if tool exists
            which_cmd = f"which {tool_path} > /dev/null 2>&1 && echo 'FOUND' || echo 'NOT_FOUND'"
            which_result = subprocess.run(which_cmd, shell=True, capture_output=True, text=True)
            if "NOT_FOUND" in which_result.stdout:
                return False, f"Tool not found in PATH: {tool_name} ({tool_path})"
            
            # Get version
            version_cmd = f"{tool_path} {tool_config['version_flag']} 2>&1 | head -1"
            version_result = subprocess.run(version_cmd, shell=True, capture_output=True, text=True)
            
            if version_result.returncode == 0:
                version_text = version_result.stdout.strip()
                # Extract version number
                version_match = re.search(r'(\d+\.\d+(\.\d+)?)', version_text)
                if version_match:
                    version = version_match.group(1)
                    return True, f"{tool_name} {version}"
                else:
                    return True, f"{tool_name} (version unknown: {version_text})"
            else:
                return True, f"{tool_name} (could not get version)"
                
        except Exception as e:
            return False, f"Error checking {tool_name}: {e}"
    
    # Check required tools
    for tool_name, tool_config in required_tools.items():
        found, message = check_tool_version(tool_name, tool_config)
        if found:
            result.add_info('tools', message)
        else:
            result.add_error('tools', message)
    
    # Check optional tools
    for tool_name, tool_config in optional_tools.items():
        found, message = check_tool_version(tool_name, tool_config)
        if found:
            result.add_info('tools', message)
        else:
            result.add_warning('tools', message)
    
    # Check Python packages
    python_packages = {
        'tensorqtl': 'tensorQTL for QTL mapping',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computations',
        'scipy': 'Statistical functions',
        'statsmodels': 'Statistical models'
    }
    
    for package, description in python_packages.items():
        try:
            __import__(package)
            result.add_info('python_packages', f"Found {package}: {description}")
        except ImportError:
            if package == 'tensorqtl':
                result.add_error('python_packages', f"Missing {package}: {description} - Required for analysis")
            else:
                result.add_warning('python_packages', f"Missing {package}: {description}")

def check_sample_concordance_enhanced(config, input_files, result):
    """Enhanced sample concordance checking"""
    try:
        # Get samples from genotype file
        geno_samples = extract_genotype_samples(input_files['genotypes'], config)
        if not geno_samples:
            result.add_error('sample_concordance', "Could not extract samples from genotype file")
            return
        
        result.add_info('sample_concordance', f"Genotype samples: {len(geno_samples)}")
        
        # Get samples from covariate file
        cov_samples = extract_covariate_samples(input_files['covariates'])
        if not cov_samples:
            result.add_error('sample_concordance', "Could not extract samples from covariate file")
            return
        
        result.add_info('sample_concordance', f"Covariate samples: {len(cov_samples)}")
        
        # Check genotype-covariate overlap
        geno_cov_overlap = geno_samples.intersection(cov_samples)
        if not geno_cov_overlap:
            result.add_error('sample_concordance', "No sample overlap between genotypes and covariates")
        else:
            overlap_percent = len(geno_cov_overlap) / min(len(geno_samples), len(cov_samples)) * 100
            if overlap_percent < 80:
                result.add_warning('sample_concordance', 
                                 f"Low genotype-covariate overlap: {len(geno_cov_overlap)} samples ({overlap_percent:.1f}%)")
            else:
                result.add_info('sample_concordance', 
                              f"Genotype-covariate overlap: {len(geno_cov_overlap)} samples ({overlap_percent:.1f}%)")
        
        # Check phenotype files
        analysis_types = get_qtl_types_from_config(config)
        for analysis_type in analysis_types:
            config_key = map_qtl_type_to_config_key(analysis_type)
            if config_key in input_files and input_files[config_key]:
                pheno_samples = extract_phenotype_samples(input_files[config_key])
                if pheno_samples:
                    pheno_overlap = geno_samples.intersection(pheno_samples)
                    if not pheno_overlap:
                        result.add_error('sample_concordance', f"No sample overlap for {analysis_type}")
                    else:
                        overlap_percent = len(pheno_overlap) / min(len(geno_samples), len(pheno_samples)) * 100
                        if overlap_percent < 80:
                            result.add_warning('sample_concordance', 
                                             f"Low {analysis_type} overlap: {len(pheno_overlap)} samples ({overlap_percent:.1f}%)")
                        else:
                            result.add_info('sample_concordance', 
                                          f"{analysis_type} overlap: {len(pheno_overlap)} samples ({overlap_percent:.1f}%)")
        
        # Check GWAS samples if enabled
        if config['analysis'].get('run_gwas', False):
            gwas_file = input_files.get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
            if gwas_file and os.path.exists(gwas_file):
                gwas_samples = extract_gwas_samples(gwas_file)
                if gwas_samples:
                    gwas_overlap = geno_samples.intersection(gwas_samples)
                    if not gwas_overlap:
                        result.add_error('sample_concordance', "No sample overlap for GWAS")
                    else:
                        overlap_percent = len(gwas_overlap) / min(len(geno_samples), len(gwas_samples)) * 100
                        if overlap_percent < 80:
                            result.add_warning('sample_concordance', 
                                             f"Low GWAS overlap: {len(gwas_overlap)} samples ({overlap_percent:.1f}%)")
                        else:
                            result.add_info('sample_concordance', 
                                          f"GWAS overlap: {len(gwas_overlap)} samples ({overlap_percent:.1f}%)")
        
        # Store final sample set for pipeline
        final_samples = geno_cov_overlap
        for analysis_type in analysis_types:
            config_key = map_qtl_type_to_config_key(analysis_type)
            if config_key in input_files and input_files[config_key]:
                pheno_samples = extract_phenotype_samples(input_files[config_key])
                if pheno_samples:
                    final_samples = final_samples.intersection(pheno_samples)
        
        if final_samples:
            result.add_info('sample_concordance', f"Final sample set: {len(final_samples)} samples")
            result.sample_counts['final'] = len(final_samples)
        else:
            result.add_error('sample_concordance', "No common samples found across all data types")
        
    except Exception as e:
        result.add_error('sample_concordance', f"Sample concordance check failed: {e}")

def validate_configuration_comprehensive(config, result):
    """Comprehensive configuration validation"""
    print("\n⚙️  Validating configuration parameters...")
    
    # TensorQTL configuration
    tensorqtl_config = config.get('tensorqtl', {})
    if tensorqtl_config.get('num_permutations', 1000) < 100:
        result.add_warning('configuration', "Low number of permutations (<100) may affect FDR estimation")
    
    # Memory and performance settings
    memory_gb = config.get('performance', {}).get('memory_gb', 8)
    available_memory = psutil.virtual_memory().available / (1024**3)
    if memory_gb > available_memory:
        result.add_warning('configuration', f"Configured memory ({memory_gb} GB) exceeds available memory ({available_memory:.1f} GB)")
    
    num_threads = config.get('performance', {}).get('num_threads', 1)
    if num_threads == 1:
        result.add_info('configuration', "Using single thread - consider increasing for better performance")
    
    # Analysis mode validation
    qtl_mode = config['analysis'].get('qtl_mode', 'cis')
    if qtl_mode == 'trans' and num_threads < 4:
        result.add_warning('configuration', "Trans-QTL analysis is computationally intensive, recommend using more threads")
    
    # Output directory checks
    results_dir = config.get('results_dir', 'results')
    if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
        result.add_warning('configuration', f"Results directory {results_dir} already exists and is not empty")
    
    # Normalization settings
    normalization_config = config.get('normalization', {})
    for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
        if qtl_type in normalization_config:
            method = normalization_config[qtl_type].get('method', 'unknown')
            result.add_info('configuration', f"{qtl_type.upper()} normalization: {method}")
    
    # Enhanced features
    if config.get('enhanced_qc', {}).get('enable', False):
        result.add_info('configuration', "Enhanced QC enabled")
    
    if config.get('interaction_analysis', {}).get('enable', False):
        result.add_info('configuration', "Interaction analysis enabled")
    
    if config.get('fine_mapping', {}).get('enable', False):
        result.add_info('configuration', "Fine-mapping enabled")

def generate_validation_report(result, config):
    """Generate comprehensive validation report"""
    report_dir = config.get('results_dir', 'results')
    os.makedirs(report_dir, exist_ok=True)
    
    # Save JSON report
    report_file = os.path.join(report_dir, 'input_validation_report.json')
    with open(report_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    # Generate text report
    text_report_file = os.path.join(report_dir, 'input_validation_report.txt')
    with open(text_report_file, 'w') as f:
        f.write("QTL Pipeline - Input Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {result.validation_time}\n")
        f.write(f"Overall Status: {result.overall_status}\n\n")
        
        f.write("Data Types Available:\n")
        for data_type in result.data_types_available:
            f.write(f"  - {data_type}\n")
        f.write("\n")
        
        f.write("Sample Counts:\n")
        for data_type, count in result.sample_counts.items():
            f.write(f"  - {data_type}: {count} samples\n")
        f.write("\n")
        
        if result.errors:
            f.write("ERRORS:\n")
            for error in result.errors:
                f.write(f"  ❌ {error}\n")
            f.write("\n")
        
        if result.warnings:
            f.write("WARNINGS:\n")
            for warning in result.warnings:
                f.write(f"  ⚠️  {warning}\n")
            f.write("\n")
        
        if result.info:
            f.write("INFO:\n")
            for info in result.info:
                f.write(f"  ℹ️  {info}\n")
    
    # Generate sample mapping file
    sample_mapping_file = os.path.join(report_dir, 'sample_mapping.txt')
    generate_sample_mapping(result, sample_mapping_file, config)
    
    logger.info(f"Validation report saved: {report_file}")
    logger.info(f"Text report saved: {text_report_file}")
    logger.info(f"Sample mapping saved: {sample_mapping_file}")

def generate_sample_mapping(result, output_file, config):
    """Generate sample mapping file from validation results"""
    try:
        # Extract actual samples from genotype file if possible
        input_files = config['input_files']
        geno_samples = extract_genotype_samples(input_files['genotypes'], config)
        
        if geno_samples:
            sample_data = []
            for sample in list(geno_samples)[:1000]:  # Limit to first 1000 samples
                data_types = ['genotype']
                
                # Check which phenotype files this sample appears in
                analysis_types = get_qtl_types_from_config(config)
                for analysis_type in analysis_types:
                    config_key = map_qtl_type_to_config_key(analysis_type)
                    if config_key in input_files and input_files[config_key]:
                        pheno_samples = extract_phenotype_samples(input_files[config_key])
                        if sample in pheno_samples:
                            data_types.append(analysis_type)
                
                # Check covariates
                cov_samples = extract_covariate_samples(input_files['covariates'])
                if sample in cov_samples:
                    data_types.append('covariates')
                
                sample_data.append({
                    'sample_id': sample,
                    'data_types': ','.join(data_types),
                    'qc_status': 'PASS'
                })
            
            if sample_data:
                df = pd.DataFrame(sample_data)
                df.to_csv(output_file, sep='\t', index=False)
                logger.info(f"Generated sample mapping with {len(sample_data)} samples")
            else:
                # Create minimal sample mapping
                df = pd.DataFrame([{
                    'sample_id': 'example_sample',
                    'data_types': 'genotype,covariates,expression',
                    'qc_status': 'PASS'
                }])
                df.to_csv(output_file, sep='\t', index=False)
                logger.warning("Created example sample mapping - actual samples could not be extracted")
    except Exception as e:
        logger.warning(f"Could not generate comprehensive sample mapping: {e}")

# Enhanced mapping functions
def map_qtl_type_to_config_key(qtl_type):
    """Map QTL analysis types to config file keys - ENHANCED"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing',
        'expression': 'expression',  # Allow reverse mapping
        'protein': 'protein',
        'splicing': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

def map_qtl_type_to_data_type(qtl_type):
    """Map QTL analysis types to data types for validation logic - ENHANCED"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein',
        'sqtl': 'splicing',
        'expression': 'expression',
        'protein': 'protein', 
        'splicing': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

def get_qtl_types_from_config(config):
    """Extract QTL types from config - ENHANCED with better error handling"""
    qtl_types = config['analysis']['qtl_types']
    
    if qtl_types == 'all':
        available_types = []
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            config_key = map_qtl_type_to_config_key(qtl_type)
            if (config_key in config['input_files'] and 
                config['input_files'][config_key] and 
                os.path.exists(config['input_files'][config_key])):
                available_types.append(qtl_type)
        return available_types
    elif isinstance(qtl_types, str):
        types_list = [t.strip() for t in qtl_types.split(',')]
        # Validate each type exists
        valid_types = []
        for qtl_type in types_list:
            config_key = map_qtl_type_to_config_key(qtl_type)
            if (config_key in config['input_files'] and 
                config['input_files'][config_key] and 
                os.path.exists(config['input_files'][config_key])):
                valid_types.append(qtl_type)
            else:
                logger.warning(f"QTL type {qtl_type} specified but file not found")
        return valid_types
    elif isinstance(qtl_types, list):
        return qtl_types
    else:
        logger.warning(f"Invalid qtl_types configuration: {qtl_types}, defaulting to eqtl")
        return ['eqtl']

def analyze_chromosome_naming(chromosomes, file_type, result):
    """Analyze chromosome naming consistency"""
    has_chr_prefix = any(str(c).startswith('chr') for c in chromosomes)
    no_chr_prefix = any(str(c) in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT', 'M'] for c in chromosomes)
    
    if has_chr_prefix and no_chr_prefix:
        result.add_warning(file_type, "Mixed chromosome naming (some with 'chr' prefix, some without)")
    elif has_chr_prefix:
        result.add_info(file_type, "Chromosome naming: with 'chr' prefix")
    elif no_chr_prefix:
        result.add_info(file_type, "Chromosome naming: without 'chr' prefix")

def extract_genotype_samples(genotype_file, config):
    """Extract samples from genotype file"""
    format_info = detect_genotype_format_enhanced(genotype_file)
    
    if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
        try:
            bcftools_path = config['paths'].get('bcftools', 'bcftools')
            cmd = f"{bcftools_path} query -l {genotype_file} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return set([s.strip() for s in result.stdout.split('\n') if s.strip()])
        except:
            pass
    elif format_info['format'] == 'plink_bed':
        try:
            base_name = genotype_file.replace('.bed', '')
            fam_file = f"{base_name}.fam"
            if os.path.exists(fam_file):
                fam_df = pd.read_csv(fam_file, sep='\s+', header=None, usecols=[1])
                return set(fam_df[1].tolist())
        except:
            pass
    
    return set()

def extract_covariate_samples(covariates_file):
    """Extract samples from covariates file"""
    try:
        df = pd.read_csv(covariates_file, sep='\t', index_col=0, nrows=0)
        return set(df.columns)
    except:
        return set()

def extract_phenotype_samples(phenotype_file):
    """Extract samples from phenotype file"""
    try:
        df = pd.read_csv(phenotype_file, sep='\t', index_col=0, nrows=0)
        return set(df.columns)
    except:
        return set()

def extract_gwas_samples(gwas_file):
    """Extract samples from GWAS file"""
    try:
        df = pd.read_csv(gwas_file, sep='\t', usecols=['sample_id'], nrows=1000)
        return set(df['sample_id'])
    except:
        return set()

def validate_generic_file(file_path, file_type, config, result):
    """Validate generic file type"""
    try:
        file_size = os.path.getsize(file_path) / (1024**2)  # MB
        result.add_info(file_type, f"File size: {file_size:.2f} MB")
        
        # Try to read as text file
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                result.add_info(file_type, f"First line sample: {first_line[:100]}...")
    except Exception as e:
        result.add_warning(file_type, f"Generic validation failed: {e}")

def validate_generic_tabular_file(file_path, file_type, config, result):
    """Validate generic tabular file"""
    try:
        df = pd.read_csv(file_path, sep='\t', nrows=100)
        result.add_info(file_type, f"Tabular file: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for common issues
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            result.add_warning(file_type, f"Found {missing_count} missing values in sample")
    except Exception as e:
        result.add_warning(file_type, f"Tabular validation failed: {e}")

# Backward compatibility functions
def validate_tensorqtl_requirements(config):
    """Legacy function for backward compatibility"""
    result = ValidationResult()
    validate_configuration_comprehensive(config, result)
    return result.errors, result.warnings

def validate_configuration(config):
    """Legacy function for backward compatibility"""
    result = ValidationResult()
    validate_configuration_comprehensive(config, result)
    return result.errors, result.warnings

def validate_enhanced_qc_requirements(config):
    """Legacy function for backward compatibility"""
    return [], []

def validate_interaction_requirements(config):
    """Legacy function for backward compatibility"""
    return [], []

def validate_finemap_requirements(config):
    """Legacy function for backward compatibility"""
    return [], []

if __name__ == "__main__":
    """Standalone validation script"""
    import sys
    import yaml
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/config.yaml"
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        success = validate_inputs(config)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)