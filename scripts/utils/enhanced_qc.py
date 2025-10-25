#!/usr/bin/env python3
"""
Enhanced Quality Control with comprehensive sample and variant QC
Enhanced for modular pipeline integration with improved error handling and reporting

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Features:
- Comprehensive genotype and phenotype QC
- Sample concordance checking
- Population stratification (PCA)
- Advanced outlier detection
- Modular pipeline integration
- Enhanced reporting and visualization
- Full error handling and logging
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import subprocess
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set up logger
logger = logging.getLogger('QTLPipeline')

def map_qtl_type_to_config_key(qtl_type):
    """Map QTL analysis types to config file keys"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing',
        'gwas': 'gwas_phenotype'
    }
    return mapping.get(qtl_type, qtl_type)

class EnhancedQC:
    def __init__(self, config):
        self.config = config
        self.qc_config = config.get('enhanced_qc', {})
        self.results_dir = config.get('results_dir', 'results')
        self.setup_qc_directories()
        
    def setup_qc_directories(self):
        """Create comprehensive QC directory structure"""
        try:
            qc_dirs = [
                'QC_reports',
                'QC_plots',
                'QC_data',
                'sample_concordance',
                'pca_results',
                'outlier_analysis',
                'sample_lists',
                'variant_lists'
            ]
            
            for qc_dir in qc_dirs:
                full_path = os.path.join(self.results_dir, qc_dir)
                Path(full_path).mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ QC directory structure created")
        except Exception as e:
            logger.error(f"❌ Failed to create QC directories: {e}")
            raise

    def run_data_preparation(self):
        """
        Run data preparation module for modular pipeline
        Returns: bool (success)
        """
        logger.info("🚀 Starting data preparation module...")
        
        try:
            # Step 1: Validate input files
            logger.info("📋 Validating input files...")
            validation_results = self.validate_input_files()
            
            # Step 2: Basic data quality checks
            logger.info("🔍 Performing basic data quality checks...")
            quality_results = self.basic_data_quality_checks()
            
            # Step 3: Generate data preparation report
            logger.info("📊 Generating data preparation report...")
            report_success = self.generate_data_preparation_report(validation_results, quality_results)
            
            # Step 4: Create sample mapping files
            logger.info("👥 Creating sample mapping files...")
            mapping_success = self.create_sample_mapping_files()
            
            success = report_success and mapping_success
            
            if success:
                logger.info("✅ Data preparation module completed successfully")
            else:
                logger.error("❌ Data preparation module had issues")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Data preparation module failed: {e}")
            return False

    def run_quality_control(self):
        """
        Run quality control module for modular pipeline
        Returns: bool (success)
        """
        logger.info("🚀 Starting quality control module...")
        
        try:
            # Get input files from config
            if 'input_files' not in self.config:
                logger.error("❌ No input_files section in config")
                return False
                
            if 'genotypes' not in self.config['input_files']:
                logger.error("❌ No genotypes file specified in config")
                return False
                
            vcf_file = self.config['input_files']['genotypes']
            
            if not os.path.exists(vcf_file):
                logger.error(f"❌ Genotype file not found: {vcf_file}")
                return False
            
            qtl_types = self.get_qtl_types_from_config()
            
            # Run comprehensive QC
            qc_results = self.run_comprehensive_qc(vcf_file, qtl_types, self.results_dir)
            
            if qc_results:
                logger.info("✅ Quality control module completed successfully")
                return True
            else:
                logger.error("❌ Quality control module failed to produce results")
                return False
            
        except Exception as e:
            logger.error(f"❌ Quality control module failed: {e}")
            return False

    def validate_input_files(self):
        """Validate all input files exist and are accessible"""
        logger.info("🔍 Validating input files...")
        
        validation_results = {}
        input_files = self.config.get('input_files', {})
        
        required_files = ['genotypes', 'covariates', 'annotations']
        optional_files = ['expression', 'protein', 'splicing', 'gwas_phenotype']
        
        for file_type in required_files + optional_files:
            file_path = input_files.get(file_type)
            
            if file_type in required_files and not file_path:
                validation_results[file_type] = {
                    'status': 'REQUIRED_MISSING',
                    'path': None,
                    'size_gb': 0,
                    'accessible': False
                }
                logger.error(f"  ❌ {file_type}: REQUIRED BUT NOT SPECIFIED")
                continue
                
            if not file_path:
                validation_results[file_type] = {
                    'status': 'OPTIONAL_NOT_CONFIGURED',
                    'path': None,
                    'size_gb': 0,
                    'accessible': False
                }
                logger.info(f"  ⚠️  {file_type}: Optional (not configured)")
                continue
            
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path) / (1024**3)  # GB
                    validation_results[file_type] = {
                        'status': 'OK',
                        'path': file_path,
                        'size_gb': round(file_size, 2),
                        'accessible': True
                    }
                    logger.info(f"  ✅ {file_type}: {file_path} ({file_size:.2f} GB)")
                except Exception as e:
                    validation_results[file_type] = {
                        'status': 'ACCESS_ERROR',
                        'path': file_path,
                        'size_gb': 0,
                        'accessible': False,
                        'error': str(e)
                    }
                    logger.error(f"  ❌ {file_type}: {file_path} - ACCESS ERROR: {e}")
            else:
                validation_results[file_type] = {
                    'status': 'FILE_NOT_FOUND',
                    'path': file_path,
                    'size_gb': 0,
                    'accessible': False
                }
                if file_type in required_files:
                    logger.error(f"  ❌ {file_type}: {file_path} - REQUIRED FILE NOT FOUND")
                else:
                    logger.warning(f"  ⚠️  {file_type}: {file_path} - OPTIONAL FILE NOT FOUND")
        
        return validation_results

    def basic_data_quality_checks(self):
        """Perform basic data quality checks"""
        logger.info("🔍 Performing basic data quality checks...")
        
        quality_results = {}
        input_files = self.config.get('input_files', {})
        
        # Check genotype file
        if 'genotypes' in input_files and input_files['genotypes']:
            geno_quality = self.check_genotype_file_quality(input_files['genotypes'])
            quality_results['genotypes'] = geno_quality
        
        # Check phenotype files
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            config_key = map_qtl_type_to_config_key(qtl_type)
            if config_key in input_files and input_files[config_key]:
                pheno_quality = self.check_phenotype_file_quality(input_files[config_key], qtl_type)
                quality_results[qtl_type] = pheno_quality
        
        # Check covariates file
        if 'covariates' in input_files and input_files['covariates']:
            covar_quality = self.check_covariates_file_quality(input_files['covariates'])
            quality_results['covariates'] = covar_quality
        
        # Check annotations file
        if 'annotations' in input_files and input_files['annotations']:
            annot_quality = self.check_annotations_file_quality(input_files['annotations'])
            quality_results['annotations'] = annot_quality
        
        return quality_results

    def check_genotype_file_quality(self, vcf_file):
        """Check genotype file quality using bcftools"""
        logger.info(f"  🧬 Checking genotype file: {vcf_file}")
        
        try:
            quality_info = {
                'file_type': 'genotype',
                'checks_passed': 0,
                'checks_total': 4,
                'details': {}
            }
            
            # Check 1: File exists and is accessible
            if os.path.exists(vcf_file):
                quality_info['checks_passed'] += 1
                quality_info['details']['file_exists'] = True
                
                # Get file size
                file_size = os.path.getsize(vcf_file) / (1024**3)  # GB
                quality_info['details']['file_size_gb'] = round(file_size, 2)
            else:
                quality_info['details']['file_exists'] = False
                quality_info['details']['error'] = "File not found"
                return quality_info
            
            # Check 2: Basic stats using bcftools
            stats_cmd = f"{self.config['paths']['bcftools']} stats {vcf_file} 2>/dev/null | head -20 || true"
            result = subprocess.run(stats_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0 and result.stdout.strip():
                quality_info['checks_passed'] += 1
                quality_info['details']['bcftools_accessible'] = True
                
                # Parse basic info from stats
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'number of samples:' in line.lower():
                        quality_info['details']['n_samples'] = line.split(':')[-1].strip()
                    elif 'number of records:' in line.lower():
                        quality_info['details']['n_variants'] = line.split(':')[-1].strip()
            else:
                quality_info['details']['bcftools_accessible'] = False
                quality_info['details']['bcftools_error'] = result.stderr
            
            # Check 3: File is indexed
            if os.path.exists(f"{vcf_file}.tbi") or os.path.exists(f"{vcf_file}.csi"):
                quality_info['checks_passed'] += 1
                quality_info['details']['indexed'] = True
            else:
                quality_info['details']['indexed'] = False
                quality_info['details']['index_warning'] = "File not indexed - some operations may be slow"
            
            # Check 4: File format
            if vcf_file.endswith(('.vcf', '.vcf.gz', '.bcf')):
                quality_info['checks_passed'] += 1
                quality_info['details']['format_ok'] = True
                quality_info['details']['format'] = os.path.splitext(vcf_file)[1]
            else:
                quality_info['details']['format_ok'] = False
                quality_info['details']['format'] = 'unknown'
            
            logger.info(f"    ✅ Genotype file checks: {quality_info['checks_passed']}/{quality_info['checks_total']} passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ Genotype quality check failed: {e}")
            return {
                'file_type': 'genotype',
                'checks_passed': 0,
                'checks_total': 4,
                'details': {'error': str(e)}
            }

    def check_phenotype_file_quality(self, pheno_file, pheno_type):
        """Check phenotype file quality"""
        logger.info(f"  📊 Checking {pheno_type} file: {pheno_file}")
        
        try:
            # Read the file
            df = pd.read_csv(pheno_file, sep='\t', index_col=0)
            
            quality_info = {
                'file_type': pheno_type,
                'checks_passed': 0,
                'checks_total': 6,
                'details': {
                    'n_features': df.shape[0],
                    'n_samples': df.shape[1],
                    'total_measurements': df.size,
                    'missing_percentage': (df.isna().sum().sum() / df.size) * 100,
                    'data_type': 'counts' if df.values.sum() > df.shape[0] * df.shape[1] else 'normalized'
                }
            }
            
            # Check 1: Non-empty file
            if df.shape[0] > 0 and df.shape[1] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['non_empty'] = True
            else:
                quality_info['details']['non_empty'] = False
            
            # Check 2: Reasonable missing rate (< 50%)
            if quality_info['details']['missing_percentage'] < 50:
                quality_info['checks_passed'] += 1
                quality_info['details']['missing_rate_acceptable'] = True
            else:
                quality_info['details']['missing_rate_acceptable'] = False
            
            # Check 3: No completely missing features
            if (df.isna().sum(axis=1) < df.shape[1]).all():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_completely_missing_features'] = True
            else:
                quality_info['details']['no_completely_missing_features'] = False
                quality_info['details']['completely_missing_features'] = (df.isna().sum(axis=1) == df.shape[1]).sum()
            
            # Check 4: No completely missing samples
            if (df.isna().sum(axis=0) < df.shape[0]).all():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_completely_missing_samples'] = True
            else:
                quality_info['details']['no_completely_missing_samples'] = False
                quality_info['details']['completely_missing_samples'] = (df.isna().sum(axis=0) == df.shape[0]).sum()
            
            # Check 5: No duplicate feature names
            if not df.index.duplicated().any():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_duplicate_features'] = True
            else:
                quality_info['details']['no_duplicate_features'] = False
                quality_info['details']['duplicate_features'] = df.index.duplicated().sum()
            
            # Check 6: Numeric data
            try:
                df.astype(float)
                quality_info['checks_passed'] += 1
                quality_info['details']['all_numeric'] = True
            except:
                quality_info['details']['all_numeric'] = False
                quality_info['details']['non_numeric_warning'] = "Contains non-numeric data"
            
            logger.info(f"    ✅ {pheno_type} file: {df.shape[0]} features, {df.shape[1]} samples - {quality_info['checks_passed']}/6 checks passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ {pheno_type} quality check failed: {e}")
            return {
                'file_type': pheno_type,
                'checks_passed': 0,
                'checks_total': 6,
                'details': {'error': str(e)}
            }

    def check_covariates_file_quality(self, covariates_file):
        """Check covariates file quality"""
        logger.info(f"  📈 Checking covariates file: {covariates_file}")
        
        try:
            df = pd.read_csv(covariates_file, sep='\t', index_col=0)
            
            quality_info = {
                'file_type': 'covariates',
                'checks_passed': 0,
                'checks_total': 5,
                'details': {
                    'n_covariates': df.shape[0],
                    'n_samples': df.shape[1],
                    'missing_percentage': (df.isna().sum().sum() / df.size) * 100
                }
            }
            
            # Check 1: Non-empty file
            if df.shape[0] > 0 and df.shape[1] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['non_empty'] = True
            else:
                quality_info['details']['non_empty'] = False
            
            # Check 2: Low missing rate (< 5%)
            if quality_info['details']['missing_percentage'] < 5:
                quality_info['checks_passed'] += 1
                quality_info['details']['missing_rate_acceptable'] = True
            else:
                quality_info['details']['missing_rate_acceptable'] = False
            
            # Check 3: No constant covariates (all same value)
            if not (df.std(axis=1) == 0).any():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_constant_covariates'] = True
            else:
                quality_info['details']['no_constant_covariates'] = False
                quality_info['details']['constant_covariates'] = (df.std(axis=1) == 0).sum()
            
            # Check 4: No duplicate covariate names
            if not df.index.duplicated().any():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_duplicate_covariates'] = True
            else:
                quality_info['details']['no_duplicate_covariates'] = False
                quality_info['details']['duplicate_covariates'] = df.index.duplicated().sum()
            
            # Check 5: Numeric data
            try:
                df.astype(float)
                quality_info['checks_passed'] += 1
                quality_info['details']['all_numeric'] = True
            except:
                quality_info['details']['all_numeric'] = False
                quality_info['details']['non_numeric_warning'] = "Contains non-numeric data"
            
            logger.info(f"    ✅ Covariates file: {df.shape[0]} covariates, {df.shape[1]} samples - {quality_info['checks_passed']}/5 checks passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ Covariates quality check failed: {e}")
            return {
                'file_type': 'covariates',
                'checks_passed': 0,
                'checks_total': 5,
                'details': {'error': str(e)}
            }

    def check_annotations_file_quality(self, annotations_file):
        """Check annotations file quality"""
        logger.info(f"  📖 Checking annotations file: {annotations_file}")
        
        try:
            # Try reading as BED format
            df = pd.read_csv(annotations_file, sep='\t', comment='#', header=None)
            
            quality_info = {
                'file_type': 'annotations',
                'checks_passed': 0,
                'checks_total': 3,
                'details': {
                    'n_annotations': df.shape[0],
                    'n_columns': df.shape[1],
                    'format': 'BED' if df.shape[1] >= 3 else 'unknown'
                }
            }
            
            # Check 1: Non-empty file
            if df.shape[0] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['non_empty'] = True
            else:
                quality_info['details']['non_empty'] = False
            
            # Check 2: Has minimum BED columns (chr, start, end)
            if df.shape[1] >= 3:
                quality_info['checks_passed'] += 1
                quality_info['details']['min_columns'] = True
            else:
                quality_info['details']['min_columns'] = False
            
            # Check 3: Chromosome column looks reasonable
            if df.shape[1] >= 1:
                first_col = df.iloc[:, 0].astype(str)
                chrom_like = first_col.str.startswith('chr').any() or first_col.isin([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']).any()
                if chrom_like:
                    quality_info['checks_passed'] += 1
                    quality_info['details']['chromosome_format_ok'] = True
                else:
                    quality_info['details']['chromosome_format_ok'] = False
                    quality_info['details']['chromosome_warning'] = "First column doesn't look like chromosome names"
            
            logger.info(f"    ✅ Annotations file: {df.shape[0]} annotations - {quality_info['checks_passed']}/3 checks passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ Annotations quality check failed: {e}")
            return {
                'file_type': 'annotations',
                'checks_passed': 0,
                'checks_total': 3,
                'details': {'error': str(e)}
            }

    def create_sample_mapping_files(self):
        """Create sample mapping files for downstream analysis"""
        logger.info("👥 Creating sample mapping files...")
        
        try:
            samples_dir = os.path.join(self.results_dir, "sample_lists")
            os.makedirs(samples_dir, exist_ok=True)
            
            input_files = self.config.get('input_files', {})
            all_samples = {}
            
            # Extract samples from genotype file
            if 'genotypes' in input_files and input_files['genotypes']:
                vcf_file = input_files['genotypes']
                geno_samples = self.extract_samples_from_vcf(vcf_file)
                if geno_samples:
                    all_samples['genotypes'] = geno_samples
                    with open(os.path.join(samples_dir, "genotype_samples.txt"), 'w') as f:
                        for sample in geno_samples:
                            f.write(f"{sample}\n")
            
            # Extract samples from phenotype files
            for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
                config_key = map_qtl_type_to_config_key(qtl_type)
                if config_key in input_files and input_files[config_key]:
                    pheno_file = input_files[config_key]
                    try:
                        df = pd.read_csv(pheno_file, sep='\t', index_col=0)
                        pheno_samples = list(df.columns)
                        all_samples[qtl_type] = pheno_samples
                        with open(os.path.join(samples_dir, f"{qtl_type}_samples.txt"), 'w') as f:
                            for sample in pheno_samples:
                                f.write(f"{sample}\n")
                    except Exception as e:
                        logger.warning(f"Could not extract samples from {qtl_type} file: {e}")
            
            # Extract samples from covariates file
            if 'covariates' in input_files and input_files['covariates']:
                covar_file = input_files['covariates']
                try:
                    df = pd.read_csv(covar_file, sep='\t', index_col=0)
                    covar_samples = list(df.columns)
                    all_samples['covariates'] = covar_samples
                    with open(os.path.join(samples_dir, "covariate_samples.txt"), 'w') as f:
                        for sample in covar_samples:
                            f.write(f"{sample}\n")
                except Exception as e:
                    logger.warning(f"Could not extract samples from covariates file: {e}")
            
            # Create sample intersection file
            if all_samples:
                # Find common samples across all datasets
                common_samples = set.intersection(*[set(samples) for samples in all_samples.values()])
                
                with open(os.path.join(samples_dir, "common_samples.txt"), 'w') as f:
                    for sample in sorted(common_samples):
                        f.write(f"{sample}\n")
                
                logger.info(f"✅ Sample mapping created: {len(common_samples)} common samples across {len(all_samples)} datasets")
                return True
            else:
                logger.warning("⚠️ No samples could be extracted from input files")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create sample mapping files: {e}")
            return False

    def extract_samples_from_vcf(self, vcf_file):
        """Extract sample names from VCF file"""
        try:
            # Try using bcftools first
            cmd = f"{self.config['paths']['bcftools']} query -l {vcf_file} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0 and result.stdout.strip():
                samples = [s.strip() for s in result.stdout.split('\n') if s.strip()]
                return samples
            
            # Fallback: try reading first few lines
            if vcf_file.endswith('.gz'):
                import gzip
                with gzip.open(vcf_file, 'rt') as f:
                    for line in f:
                        if line.startswith('#CHROM'):
                            samples = line.strip().split('\t')[9:]
                            return samples
            else:
                with open(vcf_file, 'r') as f:
                    for line in f:
                        if line.startswith('#CHROM'):
                            samples = line.strip().split('\t')[9:]
                            return samples
            
            logger.warning(f"Could not extract samples from VCF: {vcf_file}")
            return []
            
        except Exception as e:
            logger.warning(f"Error extracting samples from VCF: {e}")
            return []

    def generate_data_preparation_report(self, validation_results, quality_results):
        """Generate data preparation report"""
        logger.info("📝 Generating data preparation report...")
        
        try:
            report_dir = os.path.join(self.results_dir, "QC_reports")
            report_file = os.path.join(report_dir, "data_preparation_report.html")
            
            # Calculate overall status
            total_checks = sum(result.get('checks_total', 0) for result in quality_results.values() if isinstance(result, dict))
            passed_checks = sum(result.get('checks_passed', 0) for result in quality_results.values() if isinstance(result, dict))
            
            if total_checks > 0:
                pass_rate = passed_checks / total_checks
                if pass_rate > 0.8:
                    overall_status = "PASS"
                    status_class = "pass"
                elif pass_rate > 0.5:
                    overall_status = "WARNING"
                    status_class = "warning"
                else:
                    overall_status = "FAIL"
                    status_class = "fail"
            else:
                overall_status = "UNKNOWN"
                status_class = "warning"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Preparation Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .status-pass {{ color: #28a745; font-weight: bold; }}
                    .status-warning {{ color: #ffc107; font-weight: bold; }}
                    .status-fail {{ color: #dc3545; font-weight: bold; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
                    th {{ background-color: #f8f9fa; }}
                    .metric-bad {{ background-color: #f8d7da; }}
                    .metric-good {{ background-color: #d1ecf1; }}
                    .code {{ font-family: monospace; background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Data Preparation Report</h1>
                    <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Overall Status: <span class="status-{status_class}">{overall_status}</span></p>
                    <p>Checks Passed: {passed_checks} / {total_checks}</p>
                </div>
                
                <div class="section">
                    <h2>File Validation</h2>
                    <table>
                        <tr><th>File Type</th><th>Status</th><th>Path</th><th>Size (GB)</th></tr>
            """
            
            for file_type, result in validation_results.items():
                status_map = {
                    'OK': ('status-pass', 'OK'),
                    'REQUIRED_MISSING': ('status-fail', 'REQUIRED MISSING'),
                    'FILE_NOT_FOUND': ('status-fail', 'NOT FOUND'),
                    'ACCESS_ERROR': ('status-fail', 'ACCESS ERROR'),
                    'OPTIONAL_NOT_CONFIGURED': ('status-warning', 'OPTIONAL (NOT CONFIGURED)'),
                    'NOT_CONFIGURED': ('status-warning', 'NOT CONFIGURED')
                }
                
                status_class, status_text = status_map.get(result['status'], ('status-warning', result['status']))
                
                html_content += f"""
                        <tr>
                            <td>{file_type}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{result.get('path', 'N/A')}</td>
                            <td>{result.get('size_gb', 'N/A')}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Data Quality Checks</h2>
                    <table>
                        <tr><th>Data Type</th><th>Checks Passed</th><th>Total Checks</th><th>Details</th></tr>
            """
            
            for data_type, result in quality_results.items():
                if isinstance(result, dict):
                    checks_passed = result.get('checks_passed', 0)
                    checks_total = result.get('checks_total', 0)
                    
                    if checks_total > 0:
                        pass_rate = checks_passed / checks_total
                        if pass_rate >= 0.8:
                            status_class = "status-pass"
                        elif pass_rate >= 0.5:
                            status_class = "status-warning"
                        else:
                            status_class = "status-fail"
                    else:
                        status_class = "status-warning"
                    
                    details_html = "<ul>"
                    for key, value in result.get('details', {}).items():
                        details_html += f"<li><strong>{key}:</strong> {value}</li>"
                    details_html += "</ul>"
                    
                    html_content += f"""
                        <tr>
                            <td>{data_type}</td>
                            <td class="{status_class}">{checks_passed}/{checks_total}</td>
                            <td>{checks_total}</td>
                            <td>{details_html}</td>
                        </tr>
                    """
            
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Next Steps</h2>
                    <ul>
                        <li><strong>If status is PASS:</strong> Proceed with genotype and expression processing</li>
                        <li><strong>If status is WARNING:</strong> Review warnings and consider addressing issues</li>
                        <li><strong>If status is FAIL:</strong> Fix the reported issues before proceeding</li>
                    </ul>
                    <p>Run the next module: <span class="code">python run_QTLPipeline.py --modules genotype_processing expression_processing --auto-deps</span></p>
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"✅ Data preparation report generated: {report_file}")
            
            # Also save JSON version for programmatic access
            json_report = {
                'validation_results': validation_results,
                'quality_results': quality_results,
                'overall_status': overall_status,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            json_file = os.path.join(report_dir, "data_preparation_report.json")
            with open(json_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate data preparation report: {e}")
            return False

    def get_qtl_types_from_config(self):
        """Get QTL types from configuration"""
        try:
            qtl_types_config = self.config['analysis'].get('qtl_types', 'all')
            
            if qtl_types_config == 'all':
                available_types = []
                for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
                    config_key = map_qtl_type_to_config_key(qtl_type)
                    if (config_key in self.config['input_files'] and 
                        self.config['input_files'][config_key] and 
                        os.path.exists(self.config['input_files'][config_key])):
                        available_types.append(qtl_type)
                return available_types
            elif isinstance(qtl_types_config, str):
                return [t.strip() for t in qtl_types_config.split(',')]
            elif isinstance(qtl_types_config, list):
                return qtl_types_config
            else:
                logger.warning(f"Invalid qtl_types configuration: {qtl_types_config}")
                return ['eqtl']  # Default fallback
        except Exception as e:
            logger.error(f"Error getting QTL types from config: {e}")
            return ['eqtl']

    def run_comprehensive_qc(self, vcf_file, qtl_types, output_dir):
        """Run comprehensive QC on all data types"""
        logger.info("🔍 Running comprehensive quality control...")
        
        qc_results = {}
        
        try:
            # Create QC directory
            qc_dir = os.path.join(output_dir, "QC_reports")
            os.makedirs(qc_dir, exist_ok=True)
            
            # Get phenotype files using proper mapping
            phenotype_files = {}
            for qtl_type in qtl_types:
                config_key = map_qtl_type_to_config_key(qtl_type)
                phenotype_files[qtl_type] = self.config['input_files'].get(config_key)
            
            # Genotype QC
            logger.info("🧬 Running genotype QC...")
            qc_results['genotype'] = self.genotype_qc(vcf_file, qc_dir)
            
            # Phenotype QC
            for qtl_type, pheno_file in phenotype_files.items():
                if pheno_file and os.path.exists(pheno_file):
                    logger.info(f"📊 Running {qtl_type} phenotype QC...")
                    qc_results[qtl_type] = self.phenotype_qc(pheno_file, qtl_type, qc_dir)
            
            # Sample concordance
            logger.info("🔗 Checking sample concordance...")
            qc_results['concordance'] = self.sample_concordance_qc(vcf_file, phenotype_files, qc_dir)
            
            # Population stratification (PCA)
            if self.qc_config.get('run_pca', True):
                logger.info("📈 Running PCA analysis...")
                qc_results['pca'] = self.run_pca_analysis(vcf_file, qc_dir)
            
            # Advanced outlier detection
            if self.qc_config.get('advanced_outlier_detection', True):
                logger.info("🎯 Running advanced outlier detection...")
                qc_results['outliers'] = self.advanced_outlier_detection(vcf_file, phenotype_files, qc_dir)
            
            # Generate QC report
            logger.info("📋 Generating comprehensive QC report...")
            self.generate_qc_report(qc_results, qc_dir)
            
            # Save QC results for downstream modules
            self.save_qc_results(qc_results, qc_dir)
            
            logger.info("✅ Comprehensive QC completed")
            return qc_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive QC failed: {e}")
            return {}

    def genotype_qc(self, vcf_file, output_dir):
        """Comprehensive genotype QC using PLINK"""
        logger.info("🔬 Running genotype QC...")
        
        qc_metrics = {}
        
        try:
            # Convert VCF to PLINK format first
            plink_base = os.path.join(output_dir, "plink_qc")
            
            # Convert VCF to PLINK binary format
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed --out {plink_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode != 0:
                logger.warning("VCF to PLINK conversion failed, trying with different parameters")
                # Try without --make-bed
                cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --out {plink_base} 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
                if result.returncode != 0:
                    logger.error("VCF to PLINK conversion completely failed")
                    return {}
            
            # Sample missingness using PLINK
            sample_missingness = self.calculate_sample_missingness_plink(plink_base)
            qc_metrics['sample_missingness'] = sample_missingness
            
            # Variant missingness using PLINK
            variant_missingness = self.calculate_variant_missingness_plink(plink_base)
            qc_metrics['variant_missingness'] = variant_missingness
            
            # MAF distribution using PLINK
            maf_distribution = self.calculate_maf_distribution_plink(plink_base)
            qc_metrics['maf_distribution'] = maf_distribution
            
            # HWE testing using PLINK
            hwe_results = self.calculate_hwe_plink(plink_base, output_dir)
            qc_metrics['hwe'] = hwe_results
            
            # Sample heterozygosity using PLINK
            heterozygosity = self.calculate_heterozygosity_plink(plink_base)
            qc_metrics['heterozygosity'] = heterozygosity
            
            # Generate QC plots
            self.plot_genotype_qc(qc_metrics, output_dir)
            
            # Apply QC filters using PLINK
            filtered_file = self.apply_genotype_filters_plink(plink_base, output_dir, qc_metrics)
            qc_metrics['filtered_file'] = filtered_file
            
            # Calculate QC summary statistics
            qc_metrics['summary'] = self.calculate_genotype_qc_summary(qc_metrics)
            
            logger.info("✅ Genotype QC completed")
            return qc_metrics
            
        except Exception as e:
            logger.error(f"❌ Genotype QC failed: {e}")
            return {}

    def calculate_genotype_qc_summary(self, qc_metrics):
        """Calculate genotype QC summary statistics"""
        summary = {}
        
        try:
            # Sample missingness summary
            if 'sample_missingness' in qc_metrics and qc_metrics['sample_missingness']:
                missing_rates = list(qc_metrics['sample_missingness'].values())
                summary['sample_missingness'] = {
                    'mean': np.mean(missing_rates),
                    'median': np.median(missing_rates),
                    'max': np.max(missing_rates),
                    'samples_above_threshold': len([x for x in missing_rates if x > 0.1])
                }
            
            # Variant missingness summary
            if 'variant_missingness' in qc_metrics and qc_metrics['variant_missingness']:
                missing_rates = list(qc_metrics['variant_missingness'].values())
                summary['variant_missingness'] = {
                    'mean': np.mean(missing_rates),
                    'median': np.median(missing_rates),
                    'max': np.max(missing_rates),
                    'variants_above_threshold': len([x for x in missing_rates if x > 0.1])
                }
            
            # MAF summary
            if 'maf_distribution' in qc_metrics:
                maf_data = qc_metrics['maf_distribution']
                maf_values = maf_data.get('maf_values', [])
                summary['maf'] = {
                    'mean': maf_data.get('mean_maf', 0),
                    'median': maf_data.get('median_maf', 0),
                    'variants_maf_lt_01': len([x for x in maf_values if x < 0.01]),
                    'variants_maf_lt_05': len([x for x in maf_values if x < 0.05])
                }
            
            # HWE summary
            if 'hwe' in qc_metrics:
                summary['hwe'] = qc_metrics['hwe']
            
            return summary
            
        except Exception as e:
            logger.warning(f"Could not calculate genotype QC summary: {e}")
            return {}

    def calculate_sample_missingness_plink(self, plink_base):
        """Calculate sample-level missingness using PLINK"""
        logger.info("📊 Calculating sample missingness using PLINK...")
        
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --missing --out {plink_base}_missingness 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            sample_missingness = {}
            if result.returncode == 0:
                imiss_file = f"{plink_base}_missingness.imiss"
                if os.path.exists(imiss_file):
                    df = pd.read_csv(imiss_file, sep='\s+')
                    for _, row in df.iterrows():
                        sample_id = row['IID']
                        missing_rate = row['F_MISS']
                        sample_missingness[sample_id] = missing_rate
            
            return sample_missingness
            
        except Exception as e:
            logger.warning(f"Could not calculate sample missingness with PLINK: {e}")
            return {}

    def calculate_variant_missingness_plink(self, plink_base):
        """Calculate variant-level missingness using PLINK"""
        logger.info("📊 Calculating variant missingness using PLINK...")
        
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --missing --out {plink_base}_missingness 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            variant_missingness = {}
            if result.returncode == 0:
                lmiss_file = f"{plink_base}_missingness.lmiss"
                if os.path.exists(lmiss_file):
                    df = pd.read_csv(lmiss_file, sep='\s+')
                    for _, row in df.iterrows():
                        variant_id = f"{row['CHR']}:{row['SNP']}"
                        missing_rate = row['F_MISS']
                        variant_missingness[variant_id] = missing_rate
            
            return variant_missingness
            
        except Exception as e:
            logger.warning(f"Could not calculate variant missingness with PLINK: {e}")
            return {}

    def calculate_maf_distribution_plink(self, plink_base):
        """Calculate MAF distribution using PLINK"""
        logger.info("📊 Calculating MAF distribution using PLINK...")
        
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --freq --out {plink_base}_maf 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            maf_values = []
            if result.returncode == 0:
                frq_file = f"{plink_base}_maf.frq"
                if os.path.exists(frq_file):
                    df = pd.read_csv(frq_file, sep='\s+')
                    for _, row in df.iterrows():
                        maf = row['MAF']
                        maf_values.append(maf)
            
            return {
                'maf_values': maf_values,
                'mean_maf': np.mean(maf_values) if maf_values else 0,
                'median_maf': np.median(maf_values) if maf_values else 0,
                'maf_bins': np.histogram(maf_values, bins=20, range=(0, 0.5)) if maf_values else ([], [])
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate MAF distribution with PLINK: {e}")
            return {'maf_values': [], 'mean_maf': 0, 'median_maf': 0, 'maf_bins': ([], [])}

    def calculate_hwe_plink(self, plink_base, output_dir):
        """Calculate Hardy-Weinberg Equilibrium using PLINK"""
        logger.info("📊 Calculating HWE using PLINK...")
        
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --hardy --out {plink_base}_hwe 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                hwe_file = f"{plink_base}_hwe.hwe"
                if os.path.exists(hwe_file):
                    df = pd.read_csv(hwe_file, sep='\s+')
                    
                    hwe_threshold = self.qc_config.get('hwe_threshold', 1e-6)
                    violations = len(df[df['P'] < hwe_threshold])
                    total_variants = len(df)
                    
                    return {
                        'violations': violations,
                        'total_variants': total_variants,
                        'violation_rate': violations / total_variants if total_variants > 0 else 0
                    }
            
            return {'violations': 0, 'total_variants': 0, 'violation_rate': 0}
            
        except Exception as e:
            logger.warning(f"Could not calculate HWE with PLINK: {e}")
            return {'violations': 0, 'total_variants': 0, 'violation_rate': 0}

    def calculate_heterozygosity_plink(self, plink_base):
        """Calculate sample heterozygosity using PLINK"""
        logger.info("📊 Calculating heterozygosity using PLINK...")
        
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --het --out {plink_base}_het 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            heterozygosity = {}
            if result.returncode == 0:
                het_file = f"{plink_base}_het.het"
                if os.path.exists(het_file):
                    df = pd.read_csv(het_file, sep='\s+')
                    for _, row in df.iterrows():
                        sample_id = row['IID']
                        hom_count = row['O(HOM)']
                        nm_count = row['N(NM)']
                        het_rate = (nm_count - hom_count) / nm_count if nm_count > 0 else 0
                        heterozygosity[sample_id] = het_rate
            
            return heterozygosity
            
        except Exception as e:
            logger.warning(f"Could not calculate heterozygosity with PLINK: {e}")
            return {}

    def apply_genotype_filters_plink(self, plink_base, output_dir, qc_metrics):
        """Apply genotype filters using PLINK based on QC results"""
        logger.info("🔧 Applying genotype filters using PLINK...")
        
        try:
            filtered_base = os.path.join(output_dir, "filtered_genotypes")
            
            filter_args = []
            
            # MAF filter
            maf_threshold = self.qc_config.get('maf_threshold', 0.01)
            filter_args.append(f"--maf {maf_threshold}")
            
            # Variant missingness filter
            missing_threshold = self.qc_config.get('variant_missingness_threshold', 0.1)
            filter_args.append(f"--geno {missing_threshold}")
            
            # HWE filter
            hwe_threshold = self.qc_config.get('hwe_threshold', 1e-6)
            filter_args.append(f"--hwe {hwe_threshold}")
            
            # Sample missingness filter
            sample_missing_threshold = self.qc_config.get('sample_missingness_threshold', 0.1)
            filter_args.append(f"--mind {sample_missing_threshold}")
            
            filter_string = " ".join(filter_args)
            
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} {filter_string} --recode vcf --out {filtered_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                filtered_vcf = f"{filtered_base}.vcf"
                if os.path.exists(filtered_vcf):
                    # Compress and index the filtered VCF
                    compressed_vcf = f"{filtered_vcf}.gz"
                    subprocess.run(f"{self.config['paths']['bgzip']} -c {filtered_vcf} > {compressed_vcf} 2>/dev/null", shell=True, executable='/bin/bash')
                    subprocess.run(f"{self.config['paths']['tabix']} -p vcf {compressed_vcf} 2>/dev/null", shell=True, executable='/bin/bash')
                    return compressed_vcf
            else:
                logger.warning("PLINK filtering failed, using original file")
                return None
                
        except Exception as e:
            logger.warning(f"Genotype filtering with PLINK failed: {e}")
            return None

    def plot_genotype_qc(self, qc_metrics, output_dir):
        """Generate genotype QC plots"""
        try:
            plot_dir = os.path.join(output_dir, "QC_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # MAF distribution plot
            if 'maf_distribution' in qc_metrics and qc_metrics['maf_distribution']['maf_values']:
                plt.figure(figsize=(10, 6))
                plt.hist(qc_metrics['maf_distribution']['maf_values'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(self.qc_config.get('maf_threshold', 0.01), color='red', linestyle='--', label=f'MAF threshold ({self.qc_config.get("maf_threshold", 0.01)})')
                plt.xlabel('Minor Allele Frequency (MAF)')
                plt.ylabel('Number of Variants')
                plt.title('MAF Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'maf_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Sample missingness plot
            if 'sample_missingness' in qc_metrics and qc_metrics['sample_missingness']:
                plt.figure(figsize=(12, 6))
                missing_rates = list(qc_metrics['sample_missingness'].values())
                plt.hist(missing_rates, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
                plt.axvline(self.qc_config.get('sample_missingness_threshold', 0.1), color='red', linestyle='--', 
                           label=f'Missingness threshold ({self.qc_config.get("sample_missingness_threshold", 0.1)})')
                plt.xlabel('Sample Missing Rate')
                plt.ylabel('Number of Samples')
                plt.title('Sample Missingness Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'sample_missingness.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
            # Heterozygosity plot
            if 'heterozygosity' in qc_metrics and qc_metrics['heterozygosity']:
                plt.figure(figsize=(10, 6))
                het_rates = list(qc_metrics['heterozygosity'].values())
                plt.hist(het_rates, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.xlabel('Heterozygosity Rate')
                plt.ylabel('Number of Samples')
                plt.title('Sample Heterozygosity Distribution')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'heterozygosity.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate genotype QC plots: {e}")

    def phenotype_qc(self, pheno_file, pheno_type, output_dir):
        """Comprehensive phenotype QC with enhanced metrics"""
        logger.info(f"🔬 Running {pheno_type} QC...")
        
        try:
            df = pd.read_csv(pheno_file, sep='\t', index_col=0)
            qc_metrics = {}
            
            # Basic statistics
            qc_metrics['basic_stats'] = {
                'n_features': df.shape[0],
                'n_samples': df.shape[1],
                'total_measurements': df.size,
                'data_type': 'counts' if df.values.sum() > df.shape[0] * df.shape[1] else 'normalized'
            }
            
            # Missing values analysis
            missing_by_feature = df.isna().sum(axis=1)
            missing_by_sample = df.isna().sum(axis=0)
            
            qc_metrics['missingness'] = {
                'feature_missingness': missing_by_feature.describe().to_dict(),
                'sample_missingness': missing_by_sample.describe().to_dict(),
                'total_missing': df.isna().sum().sum(),
                'missing_percentage': (df.isna().sum().sum() / df.size) * 100,
                'features_completely_missing': (missing_by_feature == df.shape[1]).sum(),
                'samples_completely_missing': (missing_by_sample == df.shape[0]).sum()
            }
            
            # Distribution metrics
            qc_metrics['distribution'] = {
                'mean': df.mean(axis=1).describe().to_dict(),
                'std': df.std(axis=1).describe().to_dict(),
                'skewness': df.apply(lambda x: stats.skew(x.dropna()), axis=1).describe().to_dict(),
                'kurtosis': df.apply(lambda x: stats.kurtosis(x.dropna()), axis=1).describe().to_dict()
            }
            
            # Data quality flags
            qc_metrics['quality_flags'] = {
                'has_negative_values': (df < 0).any().any(),
                'has_zero_values': (df == 0).any().any(),
                'has_constant_features': (df.std(axis=1) == 0).any(),
                'has_duplicate_features': df.index.duplicated().any()
            }
            
            # Outlier detection
            qc_metrics['outliers'] = self.detect_phenotype_outliers(df)
            
            # Correlation structure
            qc_metrics['correlation'] = self.analyze_phenotype_correlation(df)
            
            # Generate phenotype QC plots
            self.plot_phenotype_qc(df, pheno_type, output_dir, qc_metrics)
            
            logger.info(f"✅ {pheno_type} QC completed")
            return qc_metrics
            
        except Exception as e:
            logger.error(f"❌ {pheno_type} QC failed: {e}")
            return {}

    def analyze_phenotype_correlation(self, df):
        """Analyze correlation structure in phenotype data"""
        try:
            # Calculate feature-feature correlation on a sample of data to save time
            sample_size = min(1000, df.shape[0])
            if sample_size < df.shape[0]:
                df_sample = df.sample(n=sample_size, random_state=42)
            else:
                df_sample = df
            
            corr_matrix = df_sample.T.corr()
            
            # Remove diagonal and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask).stack()
            
            return {
                'mean_correlation': corr_values.mean(),
                'median_correlation': corr_values.median(),
                'max_correlation': corr_values.max(),
                'min_correlation': corr_values.min(),
                'high_correlations': (np.abs(corr_values) > 0.8).sum()
            }
        except Exception as e:
            logger.warning(f"Could not analyze phenotype correlation: {e}")
            return {}

    def detect_phenotype_outliers(self, df):
        """Detect outliers in phenotype data"""
        outliers = {}
        
        try:
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(df, nan_policy='omit'))
            outlier_mask = z_scores > 3
            
            outliers['z_score'] = {
                'n_outliers': outlier_mask.sum().sum(),
                'outlier_percentage': (outlier_mask.sum().sum() / df.size) * 100
            }
            
            # IQR based outlier detection
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
            
            outliers['iqr'] = {
                'n_outliers': iqr_outlier_mask.sum().sum(),
                'outlier_percentage': (iqr_outlier_mask.sum().sum() / df.size) * 100
            }
            
            return outliers
            
        except Exception as e:
            logger.warning(f"Outlier detection failed: {e}")
            return {}

    def advanced_outlier_detection(self, vcf_file, phenotype_files, output_dir):
        """Advanced outlier detection using multiple methods"""
        logger.info("🎯 Running advanced outlier detection...")
        
        outlier_results = {}
        
        try:
            # Use PLINK for genotype-based outlier detection
            plink_base = os.path.join(output_dir, "outlier_detection")
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed --out {plink_base} 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            # Calculate heterozygosity and identify outliers
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --het --out {plink_base}_het 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            # Read heterozygosity results
            het_file = f"{plink_base}_het.het"
            if os.path.exists(het_file):
                het_df = pd.read_csv(het_file, sep='\s+')
                het_rates = (het_df['N(NM)'] - het_df['O(HOM)']) / het_df['N(NM)']
                
                # Identify heterozygosity outliers (mean ± 3 SD)
                mean_het = het_rates.mean()
                std_het = het_rates.std()
                outlier_mask = (het_rates < mean_het - 3 * std_het) | (het_rates > mean_het + 3 * std_het)
                
                outlier_results['heterozygosity_outliers'] = {
                    'n_outliers': outlier_mask.sum(),
                    'outlier_samples': het_df[outlier_mask]['IID'].tolist(),
                    'threshold_low': mean_het - 3 * std_het,
                    'threshold_high': mean_het + 3 * std_het
                }
            
            # Phenotype-based outlier detection
            for pheno_type, pheno_file in phenotype_files.items():
                if pheno_file and os.path.exists(pheno_file):
                    try:
                        df = pd.read_csv(pheno_file, sep='\t', index_col=0)
                        
                        # Use Isolation Forest for multivariate outlier detection on a sample of features
                        sample_features = min(1000, df.shape[0])
                        if sample_features < df.shape[0]:
                            df_sample = df.sample(n=sample_features, random_state=42)
                        else:
                            df_sample = df
                        
                        X = df_sample.T.fillna(df_sample.T.mean())
                        iso_forest = IsolationForest(contamination=0.05, random_state=42)
                        outlier_labels = iso_forest.fit_predict(X)
                        
                        outlier_results[f'{pheno_type}_outliers'] = {
                            'n_outliers': (outlier_labels == -1).sum(),
                            'outlier_samples': X.index[outlier_labels == -1].tolist(),
                            'method': 'Isolation Forest'
                        }
                    except Exception as e:
                        logger.warning(f"Could not run outlier detection for {pheno_type}: {e}")
                        continue
            
            # Generate outlier report
            self.generate_outlier_report(outlier_results, output_dir)
            
            return outlier_results
            
        except Exception as e:
            logger.warning(f"Advanced outlier detection failed: {e}")
            return {}

    def generate_outlier_report(self, outlier_results, output_dir):
        """Generate outlier detection report"""
        try:
            report_file = os.path.join(output_dir, "outlier_analysis_report.html")
            
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Outlier Analysis Report</title>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <h1>Outlier Analysis Report</h1>
            """
            
            for method, results in outlier_results.items():
                html_content += f"""
                <div class="section">
                    <h2>{method.replace('_', ' ').title()}</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                """
                
                for key, value in results.items():
                    if isinstance(value, list):
                        value_str = f"{len(value)} samples" if len(value) > 5 else ", ".join(map(str, value))
                    else:
                        value_str = str(value)
                    
                    html_content += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
                
                html_content += "</table></div>"
            
            html_content += "</body></html>"
            
            with open(report_file, 'w') as f:
                f.write(html_content)
                
            logger.info(f"✅ Outlier report generated: {report_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate outlier report: {e}")

    def save_qc_results(self, qc_results, output_dir):
        """Save QC results for downstream modules"""
        try:
            # Save comprehensive QC results
            results_file = os.path.join(output_dir, "comprehensive_qc_results.json")
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (pd.DataFrame, pd.Series)):
                    return obj.to_dict()
                return obj
            
            serializable_results = {}
            for key, value in qc_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: convert_for_json(v) for k, v in value.items()}
                else:
                    serializable_results[key] = convert_for_json(value)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"💾 QC results saved: {results_file}")
            
            # Save sample and variant lists for filtering
            self.save_filtering_lists(qc_results, output_dir)
            
        except Exception as e:
            logger.warning(f"Could not save QC results: {e}")

    def save_filtering_lists(self, qc_results, output_dir):
        """Save sample and variant lists for filtering in downstream modules"""
        try:
            # Save sample lists
            samples_dir = os.path.join(output_dir, "sample_lists")
            os.makedirs(samples_dir, exist_ok=True)
            
            # Get samples that pass QC from genotype data
            if 'genotype' in qc_results and 'sample_missingness' in qc_results['genotype']:
                sample_missingness = qc_results['genotype']['sample_missingness']
                missing_threshold = self.qc_config.get('sample_missingness_threshold', 0.1)
                
                passing_samples = [sample for sample, missing_rate in sample_missingness.items() 
                                 if missing_rate <= missing_threshold]
                
                with open(os.path.join(samples_dir, "samples_passing_qc.txt"), 'w') as f:
                    for sample in passing_samples:
                        f.write(f"{sample}\n")
            
            # Save variant lists
            variants_dir = os.path.join(output_dir, "variant_lists")
            os.makedirs(variants_dir, exist_ok=True)
            
            if 'genotype' in qc_results and 'variant_missingness' in qc_results['genotype']:
                variant_missingness = qc_results['genotype']['variant_missingness']
                missing_threshold = self.qc_config.get('variant_missingness_threshold', 0.1)
                
                passing_variants = [variant for variant, missing_rate in variant_missingness.items() 
                                  if missing_rate <= missing_threshold]
                
                with open(os.path.join(variants_dir, "variants_passing_qc.txt"), 'w') as f:
                    for variant in passing_variants:
                        f.write(f"{variant}\n")
            
            logger.info("💾 Sample and variant lists saved for downstream filtering")
            
        except Exception as e:
            logger.warning(f"Could not save filtering lists: {e}")

    def sample_concordance_qc(self, vcf_file, phenotype_files, output_dir):
        """Check sample concordance across all datasets"""
        logger.info("🔍 Checking sample concordance...")
        
        concordance_results = {}
        
        try:
            # Get samples from genotype file using PLINK
            plink_base = os.path.join(output_dir, "concordance_temp")
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed --out {plink_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                fam_file = f"{plink_base}.fam"
                if os.path.exists(fam_file):
                    fam_df = pd.read_csv(fam_file, sep='\s+', header=None)
                    geno_samples = set(fam_df[1].tolist())
                    
                    concordance_results['genotype_samples'] = list(geno_samples)
                    concordance_results['genotype_sample_count'] = len(geno_samples)
            
            # If PLINK failed, fall back to bcftools
            if not concordance_results.get('genotype_samples'):
                samples_cmd = f"{self.config['paths']['bcftools']} query -l {vcf_file} 2>/dev/null"
                samples_result = subprocess.run(samples_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
                if samples_result.returncode == 0 and samples_result.stdout.strip():
                    geno_samples = set([s.strip() for s in samples_result.stdout.split('\n') if s.strip()])
                    
                    concordance_results['genotype_samples'] = list(geno_samples)
                    concordance_results['genotype_sample_count'] = len(geno_samples)
                else:
                    logger.warning("Could not extract samples from genotype file")
                    return {}
            
            # Get samples from each phenotype file
            sample_overlap = {}
            for pheno_type, pheno_file in phenotype_files.items():
                if pheno_file and os.path.exists(pheno_file):
                    try:
                        df = pd.read_csv(pheno_file, sep='\t', index_col=0)
                        pheno_samples = set(df.columns)
                        overlap = geno_samples.intersection(pheno_samples)
                        
                        sample_overlap[pheno_type] = {
                            'pheno_samples': list(pheno_samples),
                            'pheno_sample_count': len(pheno_samples),
                            'overlap_samples': list(overlap),
                            'overlap_count': len(overlap),
                            'overlap_percentage': (len(overlap) / len(geno_samples)) * 100 if geno_samples else 0
                        }
                    except Exception as e:
                        logger.warning(f"Could not read phenotype file {pheno_file}: {e}")
                        continue
            
            concordance_results['sample_overlap'] = sample_overlap
            
            # Generate concordance plot
            self.plot_sample_concordance(concordance_results, output_dir)
            
            return concordance_results
            
        except Exception as e:
            logger.error(f"Sample concordance check failed: {e}")
            return {}

    def run_pca_analysis(self, vcf_file, output_dir):
        """Run PCA for population stratification using PLINK"""
        logger.info("📊 Running PCA analysis using PLINK...")
        
        try:
            plink_base = os.path.join(output_dir, "pca_input")
            
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --pca 10 --out {plink_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                pca_eigenvec = f"{plink_base}.eigenvec"
                if os.path.exists(pca_eigenvec):
                    pca_df = pd.read_csv(pca_eigenvec, sep='\s+', header=None)
                    pca_df.columns = ['FID', 'IID'] + [f'PC{i+1}' for i in range(10)]
                    
                    self.plot_pca_results(pca_df, output_dir)
                    
                    return {
                        'pca_file': pca_eigenvec,
                        'explained_variance': self.calculate_pca_variance(f"{plink_base}.eigenval"),
                        'pca_data': pca_df.iloc[:, 2:12].to_dict('list')
                    }
            
            return {}
            
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            return {}

    def calculate_pca_variance(self, eigenval_file):
        """Calculate explained variance from eigenvalues"""
        try:
            if os.path.exists(eigenval_file):
                eigenvalues = pd.read_csv(eigenval_file, header=None)[0].values
                total_variance = np.sum(eigenvalues)
                explained_variance = (eigenvalues / total_variance) * 100
                return explained_variance.tolist()
        except Exception as e:
            logger.warning(f"Could not calculate PCA variance: {e}")
        return []

    def plot_phenotype_qc(self, df, pheno_type, output_dir, qc_metrics):
        """Generate phenotype QC plots"""
        try:
            plot_dir = os.path.join(output_dir, "QC_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            n_features_to_plot = min(50, df.shape[0])
            features_to_plot = np.random.choice(df.index, n_features_to_plot, replace=False)
            
            for feature in features_to_plot:
                plt.hist(df.loc[feature].dropna(), bins=30, alpha=0.3, density=True)
            
            plt.xlabel('Expression Value')
            plt.ylabel('Density')
            plt.title(f'{pheno_type.upper()} Distribution\n({n_features_to_plot} random features)')
            
            plt.subplot(1, 2, 2)
            missing_data = df.isna().astype(int)
            plt.imshow(missing_data.values, aspect='auto', cmap='Reds', interpolation='nearest')
            plt.xlabel('Samples')
            plt.ylabel('Features')
            plt.title(f'{pheno_type.upper()} Missingness Pattern')
            plt.colorbar(label='Missing (1) / Present (0)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{pheno_type}_qc_plots.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate phenotype QC plots: {e}")

    def plot_sample_concordance(self, concordance_results, output_dir):
        """Plot sample concordance across datasets"""
        try:
            plot_dir = os.path.join(output_dir, "QC_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            if 'sample_overlap' in concordance_results and concordance_results['sample_overlap']:
                datasets = list(concordance_results['sample_overlap'].keys())
                overlap_percentages = [concordance_results['sample_overlap'][d]['overlap_percentage'] for d in datasets]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(datasets, overlap_percentages, color=['#2E86AB', '#A23B72', '#F18F01'][:len(datasets)])
                
                for bar, percentage in zip(bars, overlap_percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
                plt.ylabel('Sample Overlap Percentage')
                plt.title('Sample Concordance Across Datasets')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'sample_concordance.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate sample concordance plot: {e}")

    def plot_pca_results(self, pca_df, output_dir):
        """Plot PCA results"""
        try:
            plot_dir = os.path.join(output_dir, "QC_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, color='#2E86AB')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA: PC1 vs PC2')
            
            plt.subplot(1, 2, 2)
            pcs = range(1, 11)
            plt.plot(pcs, [100/i for i in pcs], 'o-', color='#A23B72')  # Placeholder values
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance (%)')
            plt.title('Scree Plot')
            plt.xticks(pcs)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate PCA plots: {e}")

    def generate_qc_report(self, qc_results, output_dir):
        """Generate comprehensive QC report"""
        logger.info("📊 Generating QC report...")
        
        try:
            report_file = os.path.join(output_dir, "comprehensive_qc_report.html")
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Comprehensive QC Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; }}
                    .good {{ color: green; font-weight: bold; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .bad {{ color: red; font-weight: bold; }}
                    .plot {{ text-align: center; margin: 20px 0; }}
                    .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Comprehensive Quality Control Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Genotype QC Summary</h2>
                    {self.generate_genotype_qc_summary(qc_results.get('genotype', {}))}
                </div>
                
                <div class="section">
                    <h2>Phenotype QC Summary</h2>
                    {self.generate_phenotype_qc_summary(qc_results)}
                </div>
                
                <div class="section">
                    <h2>Sample Concordance</h2>
                    {self.generate_concordance_summary(qc_results.get('concordance', {}))}
                </div>
                
                <div class="section">
                    <h2>QC Plots</h2>
                    {self.generate_plot_section(output_dir)}
                </div>
                
                <div class="section">
                    <h2>Next Steps</h2>
                    <p>Based on the QC results, you can proceed with the next modules:</p>
                    <ul>
                        <li><strong>If QC passes:</strong> Run QTL mapping: <code>python run_QTLPipeline.py --modules qtl_mapping --auto-deps</code></li>
                        <li><strong>If issues found:</strong> Review the warnings and consider re-running data preparation</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w') as f:
                f.write(html_content)
                
            logger.info(f"✅ QC report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Could not generate QC report: {e}")

    def generate_genotype_qc_summary(self, genotype_qc):
        """Generate genotype QC summary HTML"""
        if not genotype_qc:
            return "<p>No genotype QC data available.</p>"
        
        summary = "<table>"
        summary += "<tr><th>Metric</th><th>Value</th><th>Status</th></tr>"
        
        # Sample missingness
        if 'sample_missingness' in genotype_qc and genotype_qc['sample_missingness']:
            max_missing = max(genotype_qc['sample_missingness'].values()) if genotype_qc['sample_missingness'] else 0
            status = "good" if max_missing < 0.1 else "warning" if max_missing < 0.2 else "bad"
            summary += f"<tr><td>Max Sample Missingness</td><td>{max_missing:.3f}</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        # MAF summary
        if 'maf_distribution' in genotype_qc:
            mean_maf = genotype_qc['maf_distribution'].get('mean_maf', 0)
            summary += f"<tr><td>Mean MAF</td><td>{mean_maf:.4f}</td><td>-</td></tr>"
        
        # HWE violations
        if 'hwe' in genotype_qc:
            violation_rate = genotype_qc['hwe'].get('violation_rate', 0)
            status = "good" if violation_rate < 0.01 else "warning" if violation_rate < 0.05 else "bad"
            summary += f"<tr><td>HWE Violation Rate</td><td>{violation_rate:.4f}</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        # Heterozygosity
        if 'heterozygosity' in genotype_qc and genotype_qc['heterozygosity']:
            mean_het = np.mean(list(genotype_qc['heterozygosity'].values())) if genotype_qc['heterozygosity'] else 0
            summary += f"<tr><td>Mean Heterozygosity</td><td>{mean_het:.4f}</td><td>-</td></tr>"
        
        summary += "</table>"
        return summary

    def generate_phenotype_qc_summary(self, qc_results):
        """Generate phenotype QC summary HTML"""
        phenotype_types = [k for k in qc_results.keys() if k not in ['genotype', 'concordance', 'pca', 'outliers']]
        
        if not phenotype_types:
            return "<p>No phenotype QC data available.</p>"
        
        summary = "<table>"
        summary += "<tr><th>Phenotype Type</th><th>Features</th><th>Samples</th><th>Missing %</th><th>Status</th></tr>"
        
        for pheno_type in phenotype_types:
            pheno_qc = qc_results[pheno_type]
            if 'basic_stats' in pheno_qc:
                stats = pheno_qc['basic_stats']
                missing_pct = pheno_qc.get('missingness', {}).get('missing_percentage', 0)
                status = "good" if missing_pct < 5 else "warning" if missing_pct < 20 else "bad"
                
                summary += f"<tr><td>{pheno_type.upper()}</td><td>{stats.get('n_features', 0)}</td><td>{stats.get('n_samples', 0)}</td><td>{missing_pct:.2f}%</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        summary += "</table>"
        return summary

    def generate_concordance_summary(self, concordance_results):
        """Generate sample concordance summary HTML"""
        if not concordance_results or 'sample_overlap' not in concordance_results:
            return "<p>No concordance data available.</p>"
        
        summary = "<table>"
        summary += "<tr><th>Dataset</th><th>Samples</th><th>Overlap with Genotypes</th><th>Overlap %</th><th>Status</th></tr>"
        
        for dataset, data in concordance_results['sample_overlap'].items():
            overlap_pct = data.get('overlap_percentage', 0)
            status = "good" if overlap_pct >= 80 else "warning" if overlap_pct >= 50 else "bad"
            
            summary += f"<tr><td>{dataset.upper()}</td><td>{data.get('pheno_sample_count', 0)}</td><td>{data.get('overlap_count', 0)}</td><td>{overlap_pct:.1f}%</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        summary += f"<tr><td><strong>GENOTYPES</strong></td><td><strong>{concordance_results.get('genotype_sample_count', 0)}</strong></td><td>-</td><td>-</td><td>-</td></tr>"
        summary += "</table>"
        return summary

    def generate_plot_section(self, output_dir):
        """Generate HTML for QC plots"""
        plots_html = "<div class='plot-grid'>"
        
        plot_files = [
            ('maf_distribution.png', 'MAF Distribution'),
            ('sample_missingness.png', 'Sample Missingness'),
            ('heterozygosity.png', 'Heterozygosity'),
            ('sample_concordance.png', 'Sample Concordance'),
            ('pca_analysis.png', 'PCA Analysis')
        ]
        
        plot_dir = os.path.join(output_dir, "QC_plots")
        
        for plot_file, title in plot_files:
            plot_path = os.path.join(plot_dir, plot_file)
            if os.path.exists(plot_path):
                plots_html += f"""
                <div class="plot">
                    <h3>{title}</h3>
                    <img src="QC_plots/{plot_file}" alt="{title}">
                </div>
                """
        
        # Add phenotype plots
        for pheno_type in ['expression', 'protein', 'splicing']:
            pheno_plot = f"{pheno_type}_qc_plots.png"
            plot_path = os.path.join(plot_dir, pheno_plot)
            if os.path.exists(plot_path):
                plots_html += f"""
                <div class="plot">
                    <h3>{pheno_type.upper()} QC</h3>
                    <img src="QC_plots/{pheno_plot}" alt="{pheno_type.upper()} QC">
                </div>
                """
        
        plots_html += "</div>"
        return plots_html


# Modular pipeline functions
def run_data_preparation(config):
    """
    Main function for data preparation module in the modular pipeline
    Returns: bool (success)
    """
    try:
        logger.info("🚀 Starting data preparation module...")
        qc = EnhancedQC(config)
        success = qc.run_data_preparation()
        
        if success:
            logger.info("✅ Data preparation module completed successfully")
        else:
            logger.error("❌ Data preparation module failed")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Data preparation module failed: {e}")
        return False

def run_quality_control(config):
    """
    Main function for quality control module in the modular pipeline
    Returns: bool (success)
    """
    try:
        logger.info("🚀 Starting quality control module...")
        qc = EnhancedQC(config)
        success = qc.run_quality_control()
        
        if success:
            logger.info("✅ Quality control module completed successfully")
        else:
            logger.error("❌ Quality control module failed")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Quality control module failed: {e}")
        return False


# Maintain backward compatibility
if __name__ == "__main__":
    # Load config and run as standalone script
    import sys
    import yaml
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/config.yaml"
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Run comprehensive QC
        qc = EnhancedQC(config)
        vcf_file = config['input_files']['genotypes']
        qtl_types = qc.get_qtl_types_from_config()
        
        results = qc.run_comprehensive_qc(vcf_file, qtl_types, config.get('results_dir', 'results'))
        
        if results:
            print("✅ QC completed successfully!")
            sys.exit(0)
        else:
            print("❌ QC failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)