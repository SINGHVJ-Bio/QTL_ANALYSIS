#!/usr/bin/env python3
"""
Comprehensive input validation utilities for tensorQTL pipeline - Optimized Version
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

warnings.filterwarnings('ignore')
logger = logging.getLogger('QTLPipeline')

def validate_inputs(config):
    """Validate all input files and data consistency with comprehensive checks for tensorQTL - OPTIMIZED"""
    input_files = config['input_files']
    
    errors = []
    warnings = []
    
    print("🔍 Starting comprehensive input validation for tensorQTL pipeline...")
    
    # Check mandatory files - parallel file existence checks
    mandatory_files = ['genotypes', 'covariates', 'annotations']
    file_validation_futures = []
    
    with ThreadPoolExecutor(max_workers=min(6, mp.cpu_count())) as executor:
        for file_type in mandatory_files:
            file_path = input_files.get(file_type)
            if not file_path:
                errors.append(f"Missing mandatory input file: {file_type}")
                continue
                
            if not os.path.exists(file_path):
                errors.append(f"File not found: {file_path} (for {file_type})")
            else:
                print(f"✅ Found {file_type} file: {file_path}")
                
                # Submit file-specific validation tasks in parallel
                if file_type == 'genotypes':
                    future = executor.submit(validate_genotype_file, file_path, config)
                    file_validation_futures.append((future, 'genotypes'))
                elif file_type == 'covariates':
                    future = executor.submit(validate_covariates_file, file_path, config)
                    file_validation_futures.append((future, 'covariates'))
                elif file_type == 'annotations':
                    future = executor.submit(validate_annotations_file, file_path, config)
                    file_validation_futures.append((future, 'annotations'))
        
        # Process completed file validation tasks
        for future, file_type in file_validation_futures:
            try:
                task_errors, task_warnings = future.result(timeout=300)  # 5 minute timeout
                errors.extend(task_errors)
                warnings.extend(task_warnings)
            except Exception as e:
                errors.append(f"Validation failed for {file_type}: {e}")
    
    # Check phenotype files based on analysis types - parallel processing
    analysis_types = get_qtl_types_from_config(config)
    pheno_futures = []
    
    with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
        for analysis_type in analysis_types:
            # Map analysis type to config key
            config_key = map_qtl_type_to_config_key(analysis_type)
            file_path = input_files.get(config_key)
            
            if not file_path:
                errors.append(f"Missing phenotype file for {analysis_type} (looking for key: {config_key})")
                continue
                
            if not os.path.exists(file_path):
                errors.append(f"Phenotype file not found: {file_path} (for {analysis_type})")
            else:
                future = executor.submit(validate_phenotype_file, file_path, analysis_type, config)
                pheno_futures.append(future)
        
        for future in as_completed(pheno_futures):
            try:
                pheno_errors, pheno_warnings = future.result(timeout=300)
                errors.extend(pheno_errors)
                warnings.extend(pheno_warnings)
            except Exception as e:
                errors.append(f"Phenotype validation failed: {e}")
    
    # Check GWAS phenotype file if GWAS is enabled
    if config['analysis'].get('run_gwas', False):
        gwas_file = input_files.get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
        if not gwas_file:
            errors.append("GWAS enabled but no gwas_phenotype specified")
        elif not os.path.exists(gwas_file):
            errors.append(f"GWAS phenotype file not found: {gwas_file}")
        else:
            gwas_errors, gwas_warnings = validate_gwas_phenotype_file(gwas_file, config)
            errors.extend(gwas_errors)
            warnings.extend(gwas_warnings)
    
    # Check required tools for tensorQTL pipeline - parallel tool checking
    tool_errors, tool_warnings = check_required_tools_parallel(config)
    errors.extend(tool_errors)
    warnings.extend(tool_warnings)
    
    # Check sample concordance
    if config['qc'].get('check_sample_concordance', True):
        sample_errors, sample_warnings = check_sample_concordance_optimized(config, input_files)
        errors.extend(sample_errors)
        warnings.extend(sample_warnings)
    
    # Parallel configuration validations
    config_validation_tasks = [
        (validate_configuration, [config]),
        (validate_tensorqtl_requirements, [config])
    ]
    
    # Add optional validations if enabled
    if config.get('enhanced_qc', {}).get('enable', False):
        config_validation_tasks.append((validate_enhanced_qc_requirements, [config]))
    
    if config.get('interaction_analysis', {}).get('enable', False):
        config_validation_tasks.append((validate_interaction_requirements, [config]))
    
    if config.get('fine_mapping', {}).get('enable', False):
        config_validation_tasks.append((validate_finemap_requirements, [config]))
    
    # Execute configuration validations in parallel
    with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
        future_to_task = {}
        for func, args in config_validation_tasks:
            future = executor.submit(func, *args)
            future_to_task[future] = func.__name__
        
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                task_errors, task_warnings = future.result(timeout=120)
                errors.extend(task_errors)
                warnings.extend(task_warnings)
            except Exception as e:
                errors.append(f"Configuration validation {task_name} failed: {e}")
    
    # Report results
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if errors:
        print("\n❌ VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Input validation failed - please fix the errors above")
    else:
        print("\n🎉 All inputs validated successfully!")
        if warnings:
            print("   Some warnings were found but analysis can proceed")
        return True

def map_qtl_type_to_config_key(qtl_type):
    """Map QTL analysis types to config file keys"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

def map_qtl_type_to_data_type(qtl_type):
    """Map QTL analysis types to data types for validation logic"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein',
        'sqtl': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

def validate_tensorqtl_requirements(config):
    """Validate tensorQTL specific requirements - PRESERVED"""
    errors = []
    warnings = []
    
    print("\n🧬 Validating tensorQTL requirements...")
    
    try:
        import tensorqtl
        import torch
        
        # Check tensorQTL version
        tqtl_version = getattr(tensorqtl, '__version__', 'unknown')
        print(f"✅ tensorQTL version: {tqtl_version}")
        
        # Check PyTorch configuration
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("💡 Using CPU for tensorQTL analysis")
            # Check if we have enough memory for CPU analysis
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 16:
                warnings.append(f"Low system memory ({memory_gb:.1f}GB) for CPU-based tensorQTL analysis")
        
        # Test basic tensorQTL functionality
        try:
            from tensorqtl import cis, trans, genotypeio
            print("✅ tensorQTL modules imported successfully")
        except ImportError as e:
            errors.append(f"tensorQTL module import error: {e}")
            
    except ImportError as e:
        errors.append(f"tensorQTL import failed: {e}")
        errors.append("Install tensorqtl: pip install tensorqtl")
    
    # Check tensorQTL configuration parameters
    tensorqtl_config = config.get('tensorqtl', {})
    if tensorqtl_config.get('batch_size', 10000) < 1000:
        warnings.append("Very small batch_size may slow down tensorQTL analysis")
    
    if tensorqtl_config.get('num_permutations', 1000) < 100:
        warnings.append("Low number of permutations may affect FDR estimation accuracy")
    
    return errors, warnings

@lru_cache(maxsize=32)
def detect_genotype_format_cached(file_path):
    """Cached version of format detection"""
    return detect_genotype_format(file_path)

def validate_genotype_file(file_path, config):
    """Validate genotype file format and content for tensorQTL - OPTIMIZED"""
    errors = []
    warnings = []
    
    # Basic file checks
    if not os.path.exists(file_path):
        errors.append(f"Genotype file not found: {file_path}")
        return errors, warnings
    
    # Check file size
    file_size = os.path.getsize(file_path) / (1024**3)  # GB
    if file_size == 0:
        errors.append(f"Genotype file is empty: {file_path}")
    elif file_size > 50:
        warnings.append(f"Genotype file is very large ({file_size:.2f} GB), consider using PLINK format for better performance")
    
    # Detect format with caching
    format_info = detect_genotype_format_cached(file_path)
    print(f"📁 Detected genotype format: {format_info['format']}")
    
    # Validate specific formats
    if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
        vcf_errors, vcf_warnings = validate_vcf_file_optimized(file_path, config)
        errors.extend(vcf_errors)
        warnings.extend(vcf_warnings)
    elif format_info['format'] == 'plink_bed':
        plink_errors, plink_warnings = validate_plink_file_optimized(file_path, config)
        errors.extend(plink_errors)
        warnings.extend(plink_warnings)
    elif format_info['format'] == 'unknown':
        warnings.append(f"Could not automatically detect genotype file format: {file_path}")
    
    # Check if format is compatible with tensorQTL
    if format_info['format'] not in ['vcf', 'vcf.gz', 'bcf', 'plink_bed']:
        warnings.append(f"Format {format_info['format']} may not be directly compatible with tensorQTL")
    
    return errors, warnings

def detect_genotype_format(file_path):
    """Detect genotype file format - PRESERVED"""
    file_ext = file_path.lower()
    
    if file_ext.endswith('.vcf.gz') or file_ext.endswith('.vcf.bgz'):
        return {'format': 'vcf.gz', 'compressed': True}
    elif file_ext.endswith('.vcf'):
        return {'format': 'vcf', 'compressed': False}
    elif file_ext.endswith('.bcf'):
        return {'format': 'bcf', 'compressed': True}
    elif file_ext.endswith('.bed'):
        return {'format': 'plink_bed', 'compressed': False}
    elif file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        return {'format': 'hdf5', 'compressed': True}
    else:
        # Try to detect by content
        return detect_format_by_content(file_path)

def detect_format_by_content(file_path):
    """Detect file format by examining content - PRESERVED"""
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                first_line = f.readline()
        else:
            with open(file_path, 'r') as f:
                first_line = f.readline()
        
        if first_line.startswith('##fileformat=VCF'):
            return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
        elif first_line.startswith('#CHROM'):
            return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
        elif first_line.startswith('#fileformat=VCF'):
            return {'format': 'vcf', 'compressed': file_path.endswith('.gz')}
        elif 'BED' in first_line:
            return {'format': 'plink_bed', 'compressed': file_path.endswith('.gz')}
        else:
            # Check for PLINK BED magic number
            if file_path.endswith('.bed'):
                with open(file_path, 'rb') as f:
                    magic = f.read(2)
                    if magic == b'\x6c\x1b':  # PLINK BED magic number
                        return {'format': 'plink_bed', 'compressed': False}
            
            return {'format': 'unknown', 'compressed': file_path.endswith('.gz')}
    except Exception as e:
        logger.warning(f"Format detection failed: {e}")
        return {'format': 'unknown', 'compressed': file_path.endswith('.gz')}

def validate_vcf_file_optimized(file_path, config):
    """Validate VCF file structure for tensorQTL - OPTIMIZED"""
    errors = []
    warnings = []
    
    try:
        # Check if bcftools can read the file - single command for multiple checks
        cmd = f"""
        {config['paths']['bcftools']} view -h {file_path} | head -5 && \
        {config['paths']['bcftools']} query -l {file_path} | head -5 && \
        {config['paths']['bcftools']} view -H {file_path} | head -10 | grep -o 'GT' | head -1
        """
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        
        if "command not found" in result.stderr:
            errors.append(f"bcftools not available: {result.stderr}")
            return errors, warnings
        
        # Get sample count quickly
        result_samples = subprocess.run(
            f"{config['paths']['bcftools']} query -l {file_path} | wc -l",
            shell=True, capture_output=True, text=True, check=False
        )
        
        samples_count = int(result_samples.stdout.strip()) if result_samples.stdout.strip().isdigit() else 0
        if samples_count == 0:
            errors.append("No samples found in VCF file")
        else:
            print(f"✅ Found {samples_count} samples in genotype file")
            
        # Get variant count estimate quickly
        result_variants = subprocess.run(
            f"{config['paths']['bcftools']} view -H {file_path} | head -1000 | wc -l",
            shell=True, capture_output=True, text=True, check=False
        )
        
        variant_sample = int(result_variants.stdout.strip()) if result_variants.stdout.strip().isdigit() else 0
        if variant_sample == 0:
            errors.append("No variants found in VCF file")
        else:
            print(f"✅ VCF contains variants (sampled {variant_sample})")
            
        # Check chromosome naming consistency quickly
        result_chrom = subprocess.run(
            f"{config['paths']['bcftools']} view -H {file_path} | cut -f1 | head -20 | sort | uniq",
            shell=True, capture_output=True, text=True, check=False
        )
        
        chromosomes = [c.strip() for c in result_chrom.stdout.split('\n') if c.strip()]
        has_chr_prefix = any(str(c).startswith('chr') for c in chromosomes)
        no_chr_prefix = any(str(c) in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT'] for c in chromosomes)
        
        if has_chr_prefix and no_chr_prefix:
            warnings.append("Mixed chromosome naming (some with 'chr' prefix, some without)")
        elif has_chr_prefix:
            print("✅ Chromosome naming: with 'chr' prefix")
        elif no_chr_prefix:
            print("✅ Chromosome naming: without 'chr' prefix")
            
    except Exception as e:
        errors.append(f"VCF validation error: {e}")
    
    return errors, warnings

def validate_plink_file_optimized(file_path, config):
    """Validate PLINK BED file for tensorQTL - OPTIMIZED"""
    errors = []
    warnings = []
    
    try:
        base_name = file_path.replace('.bed', '')
        
        # Check for required companion files
        required_files = [f'{base_name}.bed', f'{base_name}.bim', f'{base_name}.fam']
        for req_file in required_files:
            if not os.path.exists(req_file):
                errors.append(f"PLINK companion file not found: {req_file}")
        
        if errors:
            return errors, warnings
            
        # Fast file reading - just get counts and basic info
        try:
            # Get sample count quickly
            if os.path.exists(f'{base_name}.fam'):
                with open(f'{base_name}.fam', 'r') as f:
                    sample_count = sum(1 for _ in f)
                print(f"✅ PLINK samples: {sample_count}")
            
            # Get variant count quickly  
            if os.path.exists(f'{base_name}.bim'):
                with open(f'{base_name}.bim', 'r') as f:
                    variant_count = sum(1 for _ in f)
                print(f"✅ PLINK variants: {variant_count}")
            
            # Quick duplicate check on a sample of variants
            try:
                bim_sample = pd.read_csv(f'{base_name}.bim', sep='\t', header=None, 
                                       names=['chr', 'variant_id', 'pos_cm', 'pos_bp', 'allele1', 'allele2'],
                                       nrows=1000)
                duplicate_variants = bim_sample.duplicated('variant_id').sum()
                if duplicate_variants > 0:
                    warnings.append(f"Found duplicate variant IDs in BIM file sample")
            except:
                pass  # Skip duplicate check if it fails
                
        except Exception as e:
            errors.append(f"Error reading PLINK files: {e}")
            
    except Exception as e:
        errors.append(f"PLINK validation error: {e}")
    
    return errors, warnings

def validate_covariates_file(file_path, config):
    """Validate covariates file for tensorQTL - OPTIMIZED"""
    errors = []
    warnings = []
    
    try:
        # Read only first 1000 rows for validation
        df = pd.read_csv(file_path, sep='\t', index_col=0, nrows=1000)
        
        # Check dimensions
        if df.shape[0] == 0:
            errors.append("Covariates file has no rows")
        if df.shape[1] == 0:
            errors.append("Covariates file has no samples")
        
        # Check for missing values on sample
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            warnings.append(f"Covariates file has missing values (sampled {missing_count})")
        
        # Check for constant covariates on sample
        constant_covariates = []
        for covariate in df.index:
            if df.loc[covariate].nunique() == 1:
                constant_covariates.append(covariate)
                if len(constant_covariates) >= 10:  # Limit check
                    break
        
        if constant_covariates:
            warnings.append(f"Constant covariates found (sample): {', '.join(constant_covariates[:5])}")
        
        # Check for numeric covariates (sampled)
        non_numeric_covariates = []
        for covariate in df.index[:50]:  # Check first 50
            try:
                pd.to_numeric(df.loc[covariate])
            except:
                non_numeric_covariates.append(covariate)
        
        if non_numeric_covariates:
            warnings.append(f"Non-numeric covariates found (sample): {', '.join(non_numeric_covariates[:5])}")
        
        # Check for recommended covariates
        recommended_covariates = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        missing_recommended = [cov for cov in recommended_covariates if cov not in df.index]
        if missing_recommended:
            warnings.append(f"Recommended covariates missing: {', '.join(missing_recommended)}")
        
        # Get full dimensions without loading entire file
        with open(file_path, 'r') as f:
            total_covariates = sum(1 for _ in f) - 1  # Subtract header
        
        print(f"✅ Covariates: {total_covariates} covariates, {df.shape[1]} samples")
        print(f"✅ Numeric covariates: {len(df.index) - len(non_numeric_covariates)} checked")
        if non_numeric_covariates:
            print(f"⚠️  Non-numeric covariates: {len(non_numeric_covariates)} found in sample")
        
    except Exception as e:
        errors.append(f"Error reading covariates file {file_path}: {e}")
    
    return errors, warnings

def validate_annotations_file(file_path, config):
    """Validate annotations file (BED format) for tensorQTL - OPTIMIZED"""
    errors = []
    warnings = []
    
    try:
        # Read only first 1000 rows for validation
        df = pd.read_csv(file_path, sep='\t', comment='#', nrows=1000)
        
        # Check required columns
        required_columns = ['chr', 'start', 'end', 'gene_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Annotations file missing required columns: {', '.join(missing_columns)}")
            return errors, warnings
        
        # Check data types on sample
        try:
            df['start'] = pd.to_numeric(df['start'])
            df['end'] = pd.to_numeric(df['end'])
        except:
            errors.append("Annotation start/end positions must be numeric")
        
        # Check for invalid ranges on sample
        invalid_ranges = df[df['start'] >= df['end']]
        if len(invalid_ranges) > 0:
            warnings.append(f"Found {len(invalid_ranges)} annotations with start >= end (in sample)")
        
        # Check for duplicate gene IDs on sample
        duplicate_genes = df[df.duplicated('gene_id')]
        if len(duplicate_genes) > 0:
            warnings.append(f"Found {len(duplicate_genes)} duplicate gene IDs (in sample)")
        
        # Check chromosome naming consistency
        chromosomes = df['chr'].unique()
        has_chr_prefix = any(str(c).startswith('chr') for c in chromosomes)
        no_chr_prefix = any(str(c) in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT'] for c in chromosomes)
        
        if has_chr_prefix and no_chr_prefix:
            warnings.append("Mixed chromosome naming in annotations file")
        elif has_chr_prefix:
            print("✅ Annotation chromosomes: with 'chr' prefix")
        elif no_chr_prefix:
            print("✅ Annotation chromosomes: without 'chr' prefix")
        
        # Get total annotation count without loading entire file
        with open(file_path, 'r') as f:
            total_annotations = sum(1 for _ in f)
            if file_path.endswith('.gz'):
                # For gzipped files, we can't easily count without reading
                total_annotations = "many" if total_annotations > 1000 else total_annotations
        
        print(f"✅ Annotations: {total_annotations} features (sampled {len(df)})")
        
        # Check if annotations cover expected chromosomes
        expected_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        missing_chromosomes = [chrom for chrom in expected_chromosomes if chrom not in chromosomes and f"chr{chrom}" not in chromosomes]
        if missing_chromosomes:
            warnings.append(f"Annotations missing for chromosomes (in sample): {', '.join(missing_chromosomes[:5])}")
        
    except Exception as e:
        errors.append(f"Error reading annotations file {file_path}: {e}")
    
    return errors, warnings

def validate_phenotype_file(file_path, qtl_type, config):
    """Validate phenotype file structure for tensorQTL - OPTIMIZED with naming mapping"""
    errors = []
    warnings = []
    
    try:
        # Read only first 1000 rows for validation
        df = pd.read_csv(file_path, sep='\t', index_col=0, nrows=1000)
        
        # Check dimensions
        if df.shape[0] == 0:
            errors.append(f"{qtl_type} file has no features")
        if df.shape[1] == 0:
            errors.append(f"{qtl_type} file has no samples")
        
        # Check for missing values on sample
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            warnings.append(f"{qtl_type} has missing values (sampled {missing_count})")
        
        # Check for constant features on sample
        constant_features = (df.nunique() == 1).sum()
        if constant_features > 0:
            warnings.append(f"{qtl_type} has {constant_features} constant features (in sample)")
        
        # Check for extreme values based on phenotype type using mapping
        if df.size > 0:
            # Map qtl_type to data type for extreme value checks
            data_type = map_qtl_type_to_data_type(qtl_type)
            
            if data_type == 'expression':
                extreme_low = (df < -10).sum().sum()
                extreme_high = (df > 10).sum().sum()
            elif data_type == 'protein':
                extreme_low = (df < -20).sum().sum()
                extreme_high = (df > 20).sum().sum()
            elif data_type == 'splicing':
                extreme_low = (df < -10).sum().sum()
                extreme_high = (df > 10).sum().sum()
            else:
                extreme_low = (df < -10).sum().sum()
                extreme_high = (df > 10).sum().sum()
                
            if extreme_low > 0 or extreme_high > 0:
                warnings.append(f"{qtl_type} has extreme values (consider checking normalization)")
        
        # Get full feature count without loading entire file
        with open(file_path, 'r') as f:
            total_features = sum(1 for _ in f) - 1  # Subtract header
        
        # Calculate basic statistics on sample
        try:
            mean_val = df.mean().mean()
            std_val = df.std().mean()
            print(f"✅ {qtl_type}: {total_features} features, {df.shape[1]} samples")
            print(f"   Mean: {mean_val:.3f}, Std: {std_val:.3f} (sampled)")
            
            # Check normalization method from config - handles new naming
            norm_method = config['normalization'].get(qtl_type, {}).get('method', 'unknown')
            print(f"   Normalization method: {norm_method}")
            
        except:
            print(f"✅ {qtl_type}: {total_features} features, {df.shape[1]} samples")
        
    except Exception as e:
        errors.append(f"Error reading {qtl_type} file {file_path}: {e}")
    
    return errors, warnings

def validate_gwas_phenotype_file(file_path, config):
    """Validate GWAS phenotype file - OPTIMIZED"""
    errors = []
    warnings = []
    
    try:
        # Read only first 1000 rows for validation
        df = pd.read_csv(file_path, sep='\t', nrows=1000)
        
        # Check required columns
        if 'sample_id' not in df.columns:
            errors.append("GWAS phenotype file must contain 'sample_id' column")
        
        # Check phenotype columns
        phenotype_cols = [col for col in df.columns if col != 'sample_id']
        if len(phenotype_cols) == 0:
            errors.append("GWAS phenotype file has no phenotype columns")
        
        # Check for missing values in sample IDs
        missing_samples = df['sample_id'].isna().sum()
        if missing_samples > 0:
            errors.append(f"GWAS phenotype file has {missing_samples} missing sample IDs (in sample)")
        
        # Check phenotype data types on sample
        for pheno in phenotype_cols[:10]:  # Check first 10 phenotypes
            try:
                pd.to_numeric(df[pheno])
            except:
                warnings.append(f"GWAS phenotype '{pheno}' contains non-numeric values")
        
        # Check for binary phenotypes if logistic regression is specified
        gwas_method = config.get('gwas', {}).get('method', 'linear')
        if gwas_method == 'logistic':
            for pheno in phenotype_cols[:5]:  # Check first 5 phenotypes
                unique_vals = df[pheno].dropna().unique()
                if len(unique_vals) != 2:
                    warnings.append(f"GWAS phenotype '{pheno}' has {len(unique_vals)} unique values, but logistic regression expects binary outcomes")
        
        # Get total sample count without loading entire file
        with open(file_path, 'r') as f:
            total_samples = sum(1 for _ in f) - 1  # Subtract header
        
        print(f"✅ GWAS phenotype: {total_samples} samples, {len(phenotype_cols)} phenotypes")
        
    except Exception as e:
        errors.append(f"Error reading GWAS phenotype file {file_path}: {e}")
    
    return errors, warnings

def check_required_tools_parallel(config):
    """Check if all required tools are available - OPTIMIZED with parallel execution"""
    errors = []
    warnings = []
    
    tools = config.get('paths', {})
    
    required_tools = [
        'plink',    # For genotype processing
        'bcftools', # For VCF/BCF manipulation
        'bgzip',    # For file compression
        'tabix'     # For file indexing
    ]
    
    optional_tools = ['R']  # For DESeq2 normalization
    
    def check_single_tool(tool_name, tool_path):
        """Check a single tool and return result"""
        try:
            result = subprocess.run(
                f"which {tool_path}", 
                shell=True, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                return tool_name, False, f"Required tool not found: {tool_name} ({tool_path})"
            
            # Get version information
            version_result = subprocess.run(
                f"{tool_path} --version 2>&1 | head -1", 
                shell=True, capture_output=True, text=True, check=False
            )
            version_info = version_result.stdout.strip() if version_result.returncode == 0 else "version unknown"
            return tool_name, True, f"Found {tool_name}: {version_info}"
            
        except Exception as e:
            return tool_name, False, f"Error checking tool {tool_name}: {e}"
    
    # Check required tools in parallel
    with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
        # Submit required tools
        future_to_tool = {}
        for tool_name in required_tools:
            tool_path = tools.get(tool_name, tool_name)
            future = executor.submit(check_single_tool, tool_name, tool_path)
            future_to_tool[future] = ('required', tool_name)
        
        # Submit optional tools
        for tool_name in optional_tools:
            tool_path = tools.get(tool_name, tool_name)
            future = executor.submit(check_single_tool, tool_name, tool_path)
            future_to_tool[future] = ('optional', tool_name)
        
        # Process results
        for future in as_completed(future_to_tool):
            tool_type, tool_name = future_to_tool[future]
            try:
                name, found, message = future.result(timeout=30)
                if found:
                    print(f"✅ {message}")
                else:
                    if tool_type == 'required':
                        errors.append(message)
                    else:
                        warnings.append(message)
            except Exception as e:
                error_msg = f"Tool check failed for {tool_name}: {e}"
                if tool_type == 'required':
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
    
    # Check Python packages for enhanced features
    if config.get('enhanced_qc', {}).get('enable', False):
        try:
            import sklearn
            print("✅ Found sklearn: for PCA analysis")
        except ImportError:
            warnings.append("sklearn not found - enhanced QC PCA analysis will be limited")
    
    if config.get('plotting', {}).get('plot_types', []):
        if 'interactive' in config['plotting']['plot_types']:
            try:
                import plotly
                print("✅ Found plotly: for interactive plots")
            except ImportError:
                warnings.append("plotly not found - interactive plots will be disabled")
    
    # Check for R packages if R is available and DESeq2 normalization is used
    if any(config['normalization'].get(qtl_type, {}).get('use_deseq2', False) 
           for qtl_type in ['eqtl', 'pqtl', 'sqtl'] if qtl_type in config['normalization']):
        try:
            result = subprocess.run(
                "Rscript -e 'library(DESeq2)'", 
                shell=True, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print("✅ Found DESeq2 R package")
            else:
                warnings.append("DESeq2 R package not found - VST normalization will not be available")
        except:
            warnings.append("Could not check for DESeq2 R package")
    
    return errors, warnings

def check_sample_concordance_optimized(config, input_files):
    """Check sample concordance across all input files - OPTIMIZED with naming mapping"""
    errors = []
    warnings = []
    
    try:
        # Get samples from genotype file
        geno_file = input_files['genotypes']
        format_info = detect_genotype_format_cached(geno_file)
        
        geno_samples = set()
        if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
            result = subprocess.run(
                f"{config['paths']['bcftools']} query -l {geno_file}",
                shell=True, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                geno_samples = set([s.strip() for s in result.stdout.split('\n') if s.strip()])
        elif format_info['format'] == 'plink_bed':
            base_name = geno_file.replace('.bed', '')
            fam_file = f"{base_name}.fam"
            if os.path.exists(fam_file):
                # Read only sample IDs from FAM
                fam_samples = pd.read_csv(fam_file, sep='\t', header=None, usecols=[1])
                geno_samples = set(fam_samples[1].tolist())
        
        if not geno_samples:
            warnings.append("Could not extract samples from genotype file")
            return errors, warnings
        
        print(f"✅ Genotype samples: {len(geno_samples)}")
        
        # Get samples from covariate file quickly
        cov_samples = set()
        try:
            cov_df = pd.read_csv(input_files['covariates'], sep='\t', index_col=0, nrows=0)
            cov_samples = set(cov_df.columns)
        except Exception as e:
            warnings.append(f"Could not read covariates samples: {e}")
            return errors, warnings
        
        print(f"✅ Covariate samples: {len(cov_samples)}")
        
        # Check genotype-covariate overlap
        overlap = geno_samples.intersection(cov_samples)
        if len(overlap) == 0:
            errors.append("No sample overlap between genotypes and covariates")
        else:
            overlap_percent = len(overlap) / min(len(geno_samples), len(cov_samples)) * 100
            if overlap_percent < 80:
                warnings.append(f"Low sample overlap between genotypes and covariates: {len(overlap)} samples ({overlap_percent:.1f}%)")
            else:
                print(f"✅ Genotype-covariate overlap: {len(overlap)} samples ({overlap_percent:.1f}%)")
        
        # Check phenotype files using naming mapping
        analysis_types = get_qtl_types_from_config(config)
        for analysis_type in analysis_types:
            # Map analysis type to config key
            config_key = map_qtl_type_to_config_key(analysis_type)
            if config_key in input_files and input_files[config_key]:
                try:
                    pheno_df = pd.read_csv(input_files[config_key], sep='\t', index_col=0, nrows=0)
                    pheno_samples = set(pheno_df.columns)
                    overlap = geno_samples.intersection(pheno_samples)
                    
                    if len(overlap) == 0:
                        errors.append(f"No sample overlap between genotypes and {analysis_type}")
                    else:
                        overlap_percent = len(overlap) / min(len(geno_samples), len(pheno_samples)) * 100
                        if overlap_percent < 80:
                            warnings.append(f"Low sample overlap for {analysis_type}: {len(overlap)} samples ({overlap_percent:.1f}%)")
                        else:
                            print(f"✅ {analysis_type} overlap: {len(overlap)} samples ({overlap_percent:.1f}%)")
                except Exception as e:
                    warnings.append(f"Could not check samples for {analysis_type}: {e}")
        
        # Check GWAS samples if enabled
        if config['analysis'].get('run_gwas', False):
            gwas_file = input_files.get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
            if gwas_file and os.path.exists(gwas_file):
                try:
                    gwas_df = pd.read_csv(gwas_file, sep='\t', usecols=['sample_id'], nrows=1000)
                    gwas_samples = set(gwas_df['sample_id'])
                    overlap = geno_samples.intersection(gwas_samples)
                    
                    if len(overlap) == 0:
                        errors.append("No sample overlap between genotypes and GWAS phenotypes")
                    else:
                        overlap_percent = len(overlap) / min(len(geno_samples), len(gwas_samples)) * 100
                        if overlap_percent < 80:
                            warnings.append(f"Low sample overlap for GWAS: {len(overlap)} samples ({overlap_percent:.1f}%)")
                        else:
                            print(f"✅ GWAS overlap: {len(overlap)} samples ({overlap_percent:.1f}%)")
                except Exception as e:
                    warnings.append(f"Could not check GWAS samples: {e}")
                    
    except Exception as e:
        warnings.append(f"Sample concordance check failed: {e}")
    
    return errors, warnings

def validate_configuration(config):
    """Validate configuration parameters for tensorQTL - PRESERVED"""
    errors = []
    warnings = []
    
    print("\n⚙️  Validating configuration parameters...")
    
    # Check tensorQTL parameters
    tensorqtl_config = config.get('tensorqtl', {})
    if tensorqtl_config.get('num_permutations', 1000) < 100:
        warnings.append("Low number of permutations (<100) may affect FDR estimation accuracy")
    
    if tensorqtl_config.get('fdr_threshold', 0.05) > 0.1:
        warnings.append("High FDR threshold (>0.1) may result in many false positives")
    
    # Check cis window size
    cis_window = tensorqtl_config.get('cis_window', 1000000)
    if cis_window < 100000:
        warnings.append("Very small cis window (<100kb) may miss true associations")
    elif cis_window > 2000000:
        warnings.append("Very large cis window (>2Mb) may increase multiple testing burden")
    
    # Check analysis mode
    qtl_mode = config['analysis'].get('qtl_mode', 'cis')
    if qtl_mode == 'trans' and config.get('performance', {}).get('num_threads', 1) < 4:
        warnings.append("Trans-QTL analysis is computationally intensive, consider using more threads")
    
    # Check genotype processing parameters
    processing_config = config.get('genotype_processing', {})
    if processing_config.get('min_maf', 0) > 0.1:
        warnings.append("High MAF threshold (>0.1) may remove too many variants")
    
    if processing_config.get('min_call_rate', 1) > 0.99:
        warnings.append("Very high call rate threshold (>0.99) may be too stringent")
    
    # Check memory requirements
    memory_gb = config.get('performance', {}).get('memory_gb', 8)
    if memory_gb < 8:
        warnings.append("Low memory allocation (<8 GB) may cause performance issues with tensorQTL")
    elif memory_gb >= 32:
        print("✅ Sufficient memory allocated for large datasets")
    
    # Check CPU thread configuration
    num_threads = config.get('performance', {}).get('num_threads', 1)
    if num_threads == 1:
        warnings.append("Using only 1 thread - consider increasing for better performance")
    else:
        print(f"✅ Using {num_threads} threads for parallel processing")
    
    # Check output directory
    results_dir = config.get('results_dir', 'results')
    if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
        warnings.append(f"Results directory {results_dir} already exists and is not empty")
    
    # Check normalization settings
    normalization_config = config.get('normalization', {})
    for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
        if qtl_type in normalization_config:
            method = normalization_config[qtl_type].get('method', 'unknown')
            print(f"✅ {qtl_type.upper()} normalization: {method}")
    
    return errors, warnings

def validate_enhanced_qc_requirements(config):
    """Validate requirements for enhanced QC - PRESERVED"""
    errors = []
    warnings = []
    
    qc_config = config.get('enhanced_qc', {})
    
    if qc_config.get('run_pca', False):
        # Check if we have enough samples for meaningful PCA
        try:
            geno_file = config['input_files']['genotypes']
            format_info = detect_genotype_format_cached(geno_file)
            
            sample_count = 0
            if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
                result = subprocess.run(
                    f"{config['paths']['bcftools']} query -l {geno_file} | wc -l",
                    shell=True, capture_output=True, text=True, check=False
                )
                sample_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            elif format_info['format'] == 'plink_bed':
                base_name = geno_file.replace('.bed', '')
                fam_file = f"{base_name}.fam"
                if os.path.exists(fam_file):
                    with open(fam_file, 'r') as f:
                        sample_count = sum(1 for _ in f)
            
            if sample_count < 50:
                warnings.append("PCA analysis may not be meaningful with fewer than 50 samples")
            else:
                print(f"✅ Sufficient samples ({sample_count}) for PCA analysis")
        except:
            pass
    
    return errors, warnings

def validate_interaction_requirements(config):
    """Validate requirements for interaction analysis - PRESERVED"""
    errors = []
    warnings = []
    
    interaction_config = config.get('interaction_analysis', {})
    interaction_covariates = interaction_config.get('interaction_covariates', [])
    
    if not interaction_covariates:
        errors.append("No interaction covariates specified for interaction analysis")
        return errors, warnings
    
    # Check if interaction covariates exist in covariates file
    try:
        covariates_file = config['input_files']['covariates']
        # Read only the index (covariate names) to check existence
        cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0, nrows=0)
        
        missing_covariates = [cov for cov in interaction_covariates if cov not in cov_df.index]
        if missing_covariates:
            errors.append(f"Interaction covariates not found in covariates file: {', '.join(missing_covariates)}")
        else:
            print(f"✅ Found all interaction covariates: {', '.join(interaction_covariates)}")
            
    except Exception as e:
        errors.append(f"Error validating interaction covariates: {e}")
    
    return errors, warnings

def validate_finemap_requirements(config):
    """Validate requirements for fine-mapping - PRESERVED"""
    errors = []
    warnings = []
    
    finemap_config = config.get('fine_mapping', {})
    method = finemap_config.get('method', 'susie')
    
    if method not in ['susie', 'finemap']:
        errors.append(f"Unknown fine-mapping method: {method}")
    
    credible_threshold = finemap_config.get('credible_set_threshold', 0.95)
    if credible_threshold <= 0 or credible_threshold > 1:
        errors.append(f"Invalid credible set threshold: {credible_threshold} (must be between 0 and 1)")
    
    max_causal = finemap_config.get('max_causal_variants', 5)
    if max_causal < 1:
        errors.append(f"Invalid max causal variants: {max_causal} (must be at least 1)")
    
    return errors, warnings

def get_qtl_types_from_config(config):
    """Extract QTL types from config - UPDATED for naming mapping"""
    qtl_types = config['analysis']['qtl_types']
    
    if qtl_types == 'all':
        available_types = []
        # Check for each QTL type using the config key mapping
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            config_key = map_qtl_type_to_config_key(qtl_type)
            if (config_key in config['input_files'] and 
                config['input_files'][config_key] and 
                os.path.exists(config['input_files'][config_key])):
                available_types.append(qtl_type)
        return available_types
    elif isinstance(qtl_types, str):
        return [t.strip() for t in qtl_types.split(',')]
    elif isinstance(qtl_types, list):
        return qtl_types
    else:
        raise ValueError(f"Invalid qtl_types configuration: {qtl_types}")