#!/usr/bin/env python3
"""
Comprehensive input validation utilities for tensorQTL pipeline - Enhanced Version
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
warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

def validate_inputs(config):
    """Validate all input files and data consistency with comprehensive checks for tensorQTL"""
    input_files = config['input_files']
    
    errors = []
    warnings = []
    
    print("🔍 Starting comprehensive input validation for tensorQTL pipeline...")
    
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
            print(f"✅ Found {file_type} file: {file_path}")
            
            # Special validation for each file type
            if file_type == 'genotypes':
                genotype_errors, genotype_warnings = validate_genotype_file(file_path, config)
                errors.extend(genotype_errors)
                warnings.extend(genotype_warnings)
            elif file_type == 'covariates':
                cov_errors, cov_warnings = validate_covariates_file(file_path, config)
                errors.extend(cov_errors)
                warnings.extend(cov_warnings)
            elif file_type == 'annotations':
                annot_errors, annot_warnings = validate_annotations_file(file_path, config)
                errors.extend(annot_errors)
                warnings.extend(annot_warnings)
    
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
            pheno_errors, pheno_warnings = validate_phenotype_file(file_path, analysis_type, config)
            errors.extend(pheno_errors)
            warnings.extend(pheno_warnings)
    
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
    
    # Check required tools for tensorQTL pipeline
    tool_errors, tool_warnings = check_required_tools(config)
    errors.extend(tool_errors)
    warnings.extend(tool_warnings)
    
    # Check sample concordance
    if config['qc'].get('check_sample_concordance', True):
        sample_errors, sample_warnings = check_sample_concordance(config, input_files)
        errors.extend(sample_errors)
        warnings.extend(sample_warnings)
    
    # Check configuration validity for tensorQTL
    config_errors, config_warnings = validate_configuration(config)
    errors.extend(config_errors)
    warnings.extend(config_warnings)
    
    # Check tensorQTL specific requirements
    tensorqtl_errors, tensorqtl_warnings = validate_tensorqtl_requirements(config)
    errors.extend(tensorqtl_errors)
    warnings.extend(tensorqtl_warnings)
    
    # Check enhanced QC requirements
    if config.get('enhanced_qc', {}).get('enable', False):
        qc_errors, qc_warnings = validate_enhanced_qc_requirements(config)
        errors.extend(qc_errors)
        warnings.extend(qc_warnings)
    
    # Check interaction analysis requirements
    if config.get('interaction_analysis', {}).get('enable', False):
        interaction_errors, interaction_warnings = validate_interaction_requirements(config)
        errors.extend(interaction_errors)
        warnings.extend(interaction_warnings)
    
    # Check fine-mapping requirements
    if config.get('fine_mapping', {}).get('enable', False):
        finemap_errors, finemap_warnings = validate_finemap_requirements(config)
        errors.extend(finemap_errors)
        warnings.extend(finemap_warnings)
    
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

def validate_tensorqtl_requirements(config):
    """Validate tensorQTL specific requirements"""
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
            import psutil
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

def validate_genotype_file(file_path, config):
    """Validate genotype file format and content for tensorQTL"""
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
    
    # Detect format
    format_info = detect_genotype_format(file_path)
    print(f"📁 Detected genotype format: {format_info['format']}")
    
    # Validate specific formats
    if format_info['format'] in ['vcf', 'vcf.gz', 'bcf']:
        vcf_errors, vcf_warnings = validate_vcf_file(file_path, config)
        errors.extend(vcf_errors)
        warnings.extend(vcf_warnings)
    elif format_info['format'] == 'plink_bed':
        plink_errors, plink_warnings = validate_plink_file(file_path, config)
        errors.extend(plink_errors)
        warnings.extend(plink_warnings)
    elif format_info['format'] == 'unknown':
        warnings.append(f"Could not automatically detect genotype file format: {file_path}")
    
    # Check if format is compatible with tensorQTL
    if format_info['format'] not in ['vcf', 'vcf.gz', 'bcf', 'plink_bed']:
        warnings.append(f"Format {format_info['format']} may not be directly compatible with tensorQTL")
    
    return errors, warnings

def detect_genotype_format(file_path):
    """Detect genotype file format"""
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
    """Detect file format by examining content"""
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

def validate_vcf_file(file_path, config):
    """Validate VCF file structure for tensorQTL"""
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
            print(f"✅ Found {len(samples)} samples in genotype file")
            
        # Check variant count
        result = subprocess.run(
            f"{config['paths']['bcftools']} view -H {file_path} | wc -l",
            shell=True, capture_output=True, text=True, check=False
        )
        
        variant_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        if variant_count == 0:
            errors.append("No variants found in VCF file")
        else:
            print(f"✅ Found {variant_count} variants in genotype file")
            
        # Check chromosome naming consistency
        result = subprocess.run(
            f"{config['paths']['bcftools']} view -H {file_path} | cut -f1 | sort | uniq | head -10",
            shell=True, capture_output=True, text=True, check=False
        )
        
        chromosomes = [c.strip() for c in result.stdout.split('\n') if c.strip()]
        has_chr_prefix = any(c.startswith('chr') for c in chromosomes)
        no_chr_prefix = any(c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT'] for c in chromosomes)
        
        if has_chr_prefix and no_chr_prefix:
            warnings.append("Mixed chromosome naming (some with 'chr' prefix, some without)")
        elif has_chr_prefix:
            print("✅ Chromosome naming: with 'chr' prefix")
        elif no_chr_prefix:
            print("✅ Chromosome naming: without 'chr' prefix")
            
    except Exception as e:
        errors.append(f"VCF validation error: {e}")
    
    return errors, warnings

def validate_plink_file(file_path, config):
    """Validate PLINK BED file for tensorQTL"""
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
            
        # Check if we can read the files
        try:
            # Try to read BIM file
            bim_df = pd.read_csv(f'{base_name}.bim', sep='\t', header=None, 
                               names=['chr', 'variant_id', 'pos_cm', 'pos_bp', 'allele1', 'allele2'])
            print(f"✅ PLINK BIM: {len(bim_df)} variants")
            
            # Try to read FAM file
            fam_df = pd.read_csv(f'{base_name}.fam', sep='\t', header=None,
                               names=['fam_id', 'sample_id', 'father', 'mother', 'sex', 'phenotype'])
            print(f"✅ PLINK FAM: {len(fam_df)} samples")
            
            # Check for duplicate variant IDs
            duplicate_variants = bim_df.duplicated('variant_id').sum()
            if duplicate_variants > 0:
                warnings.append(f"Found {duplicate_variants} duplicate variant IDs in BIM file")
            
        except Exception as e:
            errors.append(f"Error reading PLINK files: {e}")
            
    except Exception as e:
        errors.append(f"PLINK validation error: {e}")
    
    return errors, warnings

def validate_covariates_file(file_path, config):
    """Validate covariates file for tensorQTL"""
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        
        # Check dimensions
        if df.shape[0] == 0:
            errors.append("Covariates file has no rows")
        if df.shape[1] == 0:
            errors.append("Covariates file has no samples")
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            warnings.append(f"Covariates file has {missing_count} missing values")
        
        # Check for constant covariates
        constant_covariates = []
        for covariate in df.index:
            if df.loc[covariate].nunique() == 1:
                constant_covariates.append(covariate)
        
        if constant_covariates:
            warnings.append(f"Constant covariates found: {', '.join(constant_covariates)}")
        
        # Check for numeric covariates (except for categorical ones that should be encoded)
        numeric_covariates = []
        non_numeric_covariates = []
        
        for covariate in df.index:
            try:
                pd.to_numeric(df.loc[covariate])
                numeric_covariates.append(covariate)
            except:
                non_numeric_covariates.append(covariate)
        
        if non_numeric_covariates:
            warnings.append(f"Non-numeric covariates found: {', '.join(non_numeric_covariates)}")
        
        # Check for recommended covariates
        recommended_covariates = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        missing_recommended = [cov for cov in recommended_covariates if cov not in df.index]
        if missing_recommended:
            warnings.append(f"Recommended covariates missing: {', '.join(missing_recommended)}")
        
        print(f"✅ Covariates: {df.shape[0]} covariates, {df.shape[1]} samples")
        print(f"✅ Numeric covariates: {len(numeric_covariates)}")
        if non_numeric_covariates:
            print(f"⚠️  Non-numeric covariates: {len(non_numeric_covariates)}")
        
    except Exception as e:
        errors.append(f"Error reading covariates file {file_path}: {e}")
    
    return errors, warnings

def validate_annotations_file(file_path, config):
    """Validate annotations file (BED format) for tensorQTL"""
    errors = []
    warnings = []
    
    try:
        # Try to read as BED file
        df = pd.read_csv(file_path, sep='\t', comment='#')
        
        # Check required columns
        required_columns = ['chr', 'start', 'end', 'gene_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Annotations file missing required columns: {', '.join(missing_columns)}")
            return errors, warnings
        
        # Check data types
        try:
            df['start'] = pd.to_numeric(df['start'])
            df['end'] = pd.to_numeric(df['end'])
        except:
            errors.append("Annotation start/end positions must be numeric")
        
        # Check for invalid ranges
        invalid_ranges = df[df['start'] >= df['end']]
        if len(invalid_ranges) > 0:
            errors.append(f"Found {len(invalid_ranges)} annotations with start >= end")
        
        # Check for duplicate gene IDs
        duplicate_genes = df[df.duplicated('gene_id')]
        if len(duplicate_genes) > 0:
            warnings.append(f"Found {len(duplicate_genes)} duplicate gene IDs")
        
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
        
        # Check annotation count
        print(f"✅ Annotations: {len(df)} features")
        
        # Check if annotations cover expected chromosomes
        expected_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        missing_chromosomes = [chrom for chrom in expected_chromosomes if chrom not in chromosomes and f"chr{chrom}" not in chromosomes]
        if missing_chromosomes:
            warnings.append(f"Annotations missing for chromosomes: {', '.join(missing_chromosomes)}")
        
    except Exception as e:
        errors.append(f"Error reading annotations file {file_path}: {e}")
    
    return errors, warnings

def validate_phenotype_file(file_path, qtl_type, config):
    """Validate phenotype file structure for tensorQTL"""
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        
        # Check dimensions
        if df.shape[0] == 0:
            errors.append(f"{qtl_type} file has no features")
        if df.shape[1] == 0:
            errors.append(f"{qtl_type} file has no samples")
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            warnings.append(f"{qtl_type} has {missing_count} missing values")
        
        # Check for constant features
        constant_features = (df.nunique() == 1).sum()
        if constant_features > 0:
            warnings.append(f"{qtl_type} has {constant_features} constant features")
        
        # Check for extreme values based on phenotype type
        if df.size > 0:
            if qtl_type == 'expression':
                # Expression data typically log-transformed
                extreme_low = (df < -10).sum().sum()
                extreme_high = (df > 10).sum().sum()
            elif qtl_type == 'protein':
                # Protein data often has wider range
                extreme_low = (df < -20).sum().sum()
                extreme_high = (df > 20).sum().sum()
            elif qtl_type == 'splicing':
                # Splicing data (PSI) typically between 0-1 or logit transformed
                extreme_low = (df < -10).sum().sum()
                extreme_high = (df > 10).sum().sum()
            else:
                extreme_low = (df < -10).sum().sum()
                extreme_high = (df > 10).sum().sum()
                
            if extreme_low > 0 or extreme_high > 0:
                warnings.append(f"{qtl_type} has extreme values (consider checking normalization)")
        
        # Check data distribution
        try:
            # Calculate basic statistics
            mean_val = df.mean().mean()
            std_val = df.std().mean()
            print(f"✅ {qtl_type}: {df.shape[0]} features, {df.shape[1]} samples")
            print(f"   Mean: {mean_val:.3f}, Std: {std_val:.3f}")
            
            # Check normalization method from config
            norm_method = config['normalization'].get(qtl_type, {}).get('method', 'unknown')
            print(f"   Normalization method: {norm_method}")
            
        except:
            print(f"✅ {qtl_type}: {df.shape[0]} features, {df.shape[1]} samples")
        
    except Exception as e:
        errors.append(f"Error reading {qtl_type} file {file_path}: {e}")
    
    return errors, warnings

def validate_gwas_phenotype_file(file_path, config):
    """Validate GWAS phenotype file"""
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        
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
            errors.append(f"GWAS phenotype file has {missing_samples} missing sample IDs")
        
        # Check phenotype data types
        for pheno in phenotype_cols:
            try:
                pd.to_numeric(df[pheno])
            except:
                warnings.append(f"GWAS phenotype '{pheno}' contains non-numeric values")
        
        # Check for binary phenotypes if logistic regression is specified
        gwas_method = config.get('gwas', {}).get('method', 'linear')
        if gwas_method == 'logistic':
            for pheno in phenotype_cols:
                unique_vals = df[pheno].dropna().unique()
                if len(unique_vals) != 2:
                    warnings.append(f"GWAS phenotype '{pheno}' has {len(unique_vals)} unique values, but logistic regression expects binary outcomes")
        
        print(f"✅ GWAS phenotype: {df.shape[0]} samples, {len(phenotype_cols)} phenotypes")
        
    except Exception as e:
        errors.append(f"Error reading GWAS phenotype file {file_path}: {e}")
    
    return errors, warnings

def check_required_tools(config):
    """Check if all required tools are available for tensorQTL pipeline"""
    errors = []
    warnings = []
    
    tools = config.get('paths', {})
    
    # Updated required tools for tensorQTL pipeline
    required_tools = [
        'plink',    # For genotype processing
        'bcftools', # For VCF/BCF manipulation
        'bgzip',    # For file compression
        'tabix'     # For file indexing
    ]
    
    for tool_name in required_tools:
        tool_path = tools.get(tool_name, tool_name)
        
        try:
            result = subprocess.run(
                f"which {tool_path}", 
                shell=True, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                errors.append(f"Required tool not found: {tool_name} ({tool_path})")
            else:
                # Get version information
                version_result = subprocess.run(
                    f"{tool_path} --version", 
                    shell=True, capture_output=True, text=True, check=False
                )
                if version_result.returncode == 0:
                    version_line = version_result.stdout.split('\n')[0]
                    print(f"✅ Found {tool_name}: {version_line}")
                else:
                    print(f"✅ Found {tool_name}: {tool_path}")
        except Exception as e:
            errors.append(f"Error checking tool {tool_name}: {e}")
    
    # Check optional tools
    optional_tools = ['R']  # For DESeq2 normalization
    for tool_name in optional_tools:
        tool_path = tools.get(tool_name, tool_name)
        result = subprocess.run(
            f"which {tool_path}", 
            shell=True, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            warnings.append(f"Optional tool not found: {tool_name} ({tool_path})")
        else:
            print(f"✅ Found {tool_name}: {tool_path}")
    
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

def check_sample_concordance(config, input_files):
    """Check sample concordance across all input files"""
    errors = []
    warnings = []
    
    try:
        # Get samples from genotype file
        geno_file = input_files['genotypes']
        format_info = detect_genotype_format(geno_file)
        
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
                fam_df = pd.read_csv(fam_file, sep='\t', header=None)
                geno_samples = set(fam_df[1].tolist())  # Sample IDs are in second column
        
        if not geno_samples:
            warnings.append("Could not extract samples from genotype file")
            return errors, warnings
        
        print(f"✅ Genotype samples: {len(geno_samples)}")
        
        # Get samples from covariate file
        cov_df = pd.read_csv(input_files['covariates'], sep='\t', index_col=0)
        cov_samples = set(cov_df.columns)
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
        
        # Check phenotype files
        analysis_types = get_qtl_types_from_config(config)
        for analysis_type in analysis_types:
            if analysis_type in input_files and input_files[analysis_type]:
                pheno_df = pd.read_csv(input_files[analysis_type], sep='\t', index_col=0)
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
        
        # Check GWAS samples if enabled
        if config['analysis'].get('run_gwas', False):
            gwas_file = input_files.get('gwas_phenotype') or config['analysis'].get('gwas_phenotype')
            if gwas_file and os.path.exists(gwas_file):
                gwas_df = pd.read_csv(gwas_file, sep='\t')
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
        warnings.append(f"Sample concordance check failed: {e}")
    
    return errors, warnings

def validate_configuration(config):
    """Validate configuration parameters for tensorQTL"""
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
    """Validate requirements for enhanced QC"""
    errors = []
    warnings = []
    
    qc_config = config.get('enhanced_qc', {})
    
    if qc_config.get('run_pca', False):
        # Check if we have enough samples for meaningful PCA
        try:
            geno_file = config['input_files']['genotypes']
            format_info = detect_genotype_format(geno_file)
            
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
                    fam_df = pd.read_csv(fam_file, sep='\t', header=None)
                    sample_count = len(fam_df)
            
            if sample_count < 50:
                warnings.append("PCA analysis may not be meaningful with fewer than 50 samples")
            else:
                print(f"✅ Sufficient samples ({sample_count}) for PCA analysis")
        except:
            pass
    
    return errors, warnings

def validate_interaction_requirements(config):
    """Validate requirements for interaction analysis"""
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
        cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
        
        missing_covariates = [cov for cov in interaction_covariates if cov not in cov_df.index]
        if missing_covariates:
            errors.append(f"Interaction covariates not found in covariates file: {', '.join(missing_covariates)}")
        else:
            print(f"✅ Found all interaction covariates: {', '.join(interaction_covariates)}")
            
    except Exception as e:
        errors.append(f"Error validating interaction covariates: {e}")
    
    return errors, warnings

def validate_finemap_requirements(config):
    """Validate requirements for fine-mapping"""
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
    """Extract QTL types from config"""
    qtl_types = config['analysis']['qtl_types']
    
    if qtl_types == 'all':
        available_types = []
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            if (qtl_type in config['input_files'] and 
                config['input_files'][qtl_type] and 
                os.path.exists(config['input_files'][qtl_type])):
                available_types.append(qtl_type)
        return available_types
    elif isinstance(qtl_types, str):
        return [t.strip() for t in qtl_types.split(',')]
    elif isinstance(qtl_types, list):
        return qtl_types
    else:
        raise ValueError(f"Invalid qtl_types configuration: {qtl_types}")