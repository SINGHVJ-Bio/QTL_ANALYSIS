#!/usr/bin/env python3
"""
Comprehensive input validation utilities for tensorQTL pipeline - Production Version
Robust validation with enhanced mapping, parallel processing, and comprehensive error handling
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

ENHANCED: Dynamic covariate and phenotype handling with flexible validation
FIXED: Covariate file parsing for your specific format
UPDATED: Consistent directory management using directory_manager
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

# Import directory manager for consistent directory structure
try:
    from scripts.utils.directory_manager import get_module_directories
except ImportError as e:
    logger.warning(f"Directory manager not available: {e}")
    get_module_directories = None

class ValidationResult:
    """Enhanced validation result tracking with comprehensive reporting"""
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.file_stats = {}
        self.sample_counts = {}
        self.covariate_info = {}  # NEW: Store covariate metadata
        self.phenotype_info = {}  # NEW: Store phenotype metadata
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
            'covariate_info': self.covariate_info,  # NEW
            'phenotype_info': self.phenotype_info,  # NEW
            'data_types_available': self.data_types_available,
            'overall_status': self.overall_status
        }

class DynamicCovariateAnalyzer:
    """NEW: Analyze covariates dynamically and provide recommendations"""
    
    def __init__(self, config):
        self.config = config
        self.enhanced_qc_config = config.get('enhanced_qc', {})
        
    def analyze_covariates(self, covariate_df, result):
        """Analyze covariates and provide dynamic recommendations"""
        if covariate_df.empty:
            result.add_warning('covariate_analysis', "No covariates provided for analysis")
            return
            
        logger.info("🔍 Analyzing covariate structure and composition...")
        
        # Store basic covariate information
        result.covariate_info['total_covariates'] = covariate_df.shape[0]
        result.covariate_info['sample_count'] = covariate_df.shape[1]
        result.covariate_info['covariate_names'] = covariate_df.index.tolist()
        
        # Analyze covariate types
        numeric_covariates = []
        categorical_covariates = []
        binary_covariates = []
        
        for covar in covariate_df.index:
            values = covariate_df.loc[covar]
            
            # Check if numeric
            try:
                numeric_vals = pd.to_numeric(values, errors='coerce')
                non_na_count = numeric_vals.notna().sum()
                total_count = len(values)
                
                if non_na_count == total_count:  # All values are numeric
                    unique_vals = numeric_vals.nunique()
                    
                    if unique_vals == 2:
                        binary_covariates.append(covar)
                        result.covariate_info[f'covariate_{covar}_type'] = 'binary'
                        result.covariate_info[f'covariate_{covar}_values'] = sorted(numeric_vals.unique().tolist())
                    else:
                        numeric_covariates.append(covar)
                        result.covariate_info[f'covariate_{covar}_type'] = 'numeric'
                        result.covariate_info[f'covariate_{covar}_stats'] = {
                            'mean': float(numeric_vals.mean()),
                            'std': float(numeric_vals.std()),
                            'min': float(numeric_vals.min()),
                            'max': float(numeric_vals.max())
                        }
                else:
                    # Mixed or categorical data
                    categorical_covariates.append(covar)
                    result.covariate_info[f'covariate_{covar}_type'] = 'categorical'
                    unique_values = values.unique()
                    # Limit stored values to avoid huge JSON
                    if len(unique_values) <= 20:
                        result.covariate_info[f'covariate_{covar}_values'] = unique_values.tolist()
                    else:
                        result.covariate_info[f'covariate_{covar}_values'] = f"{len(unique_values)} unique values"
                    
            except Exception as e:
                categorical_covariates.append(covar)
                result.covariate_info[f'covariate_{covar}_type'] = 'categorical'
                unique_values = values.unique()
                if len(unique_values) <= 20:
                    result.covariate_info[f'covariate_{covar}_values'] = unique_values.tolist()
                else:
                    result.covariate_info[f'covariate_{covar}_values'] = f"{len(unique_values)} unique values"
        
        # Store type counts
        result.covariate_info['numeric_covariates'] = numeric_covariates
        result.covariate_info['categorical_covariates'] = categorical_covariates
        result.covariate_info['binary_covariates'] = binary_covariates
        result.covariate_info['numeric_count'] = len(numeric_covariates)
        result.covariate_info['categorical_count'] = len(categorical_covariates)
        result.covariate_info['binary_count'] = len(binary_covariates)
        
        # Detect common covariate patterns
        self._detect_covariate_patterns(covariate_df, result)
        
        # Provide recommendations
        self._provide_covariate_recommendations(covariate_df, result)
        
        logger.info(f"✅ Covariate analysis: {len(numeric_covariates)} numeric, "
                   f"{len(categorical_covariates)} categorical, {len(binary_covariates)} binary covariates")
    
    def _detect_covariate_patterns(self, covariate_df, result):
        """Detect common covariate patterns and technical artifacts"""
        detected_patterns = []
        
        # Check for PCA components
        pca_pattern = re.compile(r'^PC\d+$', re.IGNORECASE)
        pca_covariates = [covar for covar in covariate_df.index if pca_pattern.match(str(covar))]
        if pca_covariates:
            detected_patterns.append(f"PCA components: {len(pca_covariates)} found")
            result.covariate_info['pca_components'] = pca_covariates
        
        # Check for batch effects
        batch_pattern = re.compile(r'.*batch.*', re.IGNORECASE)
        batch_covariates = [covar for covar in covariate_df.index if batch_pattern.match(str(covar))]
        if batch_covariates:
            detected_patterns.append(f"Batch covariates: {batch_covariates}")
            result.covariate_info['batch_covariates'] = batch_covariates
        
        # Check for demographic covariates
        demo_patterns = ['age', 'sex', 'gender', 'bmi', 'height', 'weight']
        demo_covariates = [covar for covar in covariate_df.index 
                          if any(pattern in str(covar).lower() for pattern in demo_patterns)]
        if demo_covariates:
            detected_patterns.append(f"Demographic covariates: {demo_covariates}")
            result.covariate_info['demographic_covariates'] = demo_covariates
        
        # Check for study-specific covariates
        study_patterns = ['study', 'cohort', 'center', 'site']
        study_covariates = [covar for covar in covariate_df.index 
                           if any(pattern in str(covar).lower() for pattern in study_patterns)]
        if study_covariates:
            detected_patterns.append(f"Study-specific covariates: {study_covariates}")
            result.covariate_info['study_covariates'] = study_covariates
        
        if detected_patterns:
            result.add_info('covariate_patterns', f"Detected patterns: {'; '.join(detected_patterns)}")
    
    def _provide_covariate_recommendations(self, covariate_df, result):
        """Provide dynamic recommendations based on covariate analysis"""
        recommendations = []
        
        # Check if PCA components are present
        pca_count = len(result.covariate_info.get('pca_components', []))
        if pca_count < 5 and self.enhanced_qc_config.get('run_pca', True):
            recommendations.append("Consider adding more PCA components (5-10 recommended)")
        
        # Check for missing common covariates
        expected_covariates = ['age', 'sex', 'batch']
        missing_expected = [covar for covar in expected_covariates 
                          if covar not in [c.lower() for c in covariate_df.index]]
        if missing_expected:
            recommendations.append(f"Common covariates not found: {missing_expected}")
        
        # Check for categorical covariates that might need encoding
        categorical_count = result.covariate_info.get('categorical_count', 0)
        if categorical_count > 0:
            categorical_covars = result.covariate_info.get('categorical_covariates', [])
            recommendations.append(f"Categorical covariates detected: {categorical_covars}. Ensure proper encoding.")
        
        # Check sample size vs covariate count
        sample_count = covariate_df.shape[1]
        covariate_count = covariate_df.shape[0]
        if covariate_count > sample_count / 10:
            recommendations.append(f"High covariate count ({covariate_count}) relative to sample size ({sample_count})")
        
        if recommendations:
            result.add_info('covariate_recommendations', f"Recommendations: {'; '.join(recommendations)}")

class DynamicPhenotypeAnalyzer:
    """NEW: Analyze phenotype data dynamically"""
    
    def __init__(self, config):
        self.config = config
        self.normalization_config = config.get('normalization', {})
        
    def analyze_phenotype_data(self, phenotype_df, qtl_type, result):
        """Analyze phenotype data and provide dynamic recommendations"""
        logger.info(f"🔍 Analyzing {qtl_type} phenotype data structure...")
        
        # Store basic phenotype information
        result.phenotype_info[qtl_type] = {
            'feature_count': phenotype_df.shape[0],
            'sample_count': phenotype_df.shape[1],
            'total_measurements': phenotype_df.size,
            'missing_count': int(phenotype_df.isna().sum().sum()),
            'missing_percentage': float((phenotype_df.isna().sum().sum() / phenotype_df.size) * 100)
        }
        
        # Analyze data distribution
        self._analyze_phenotype_distribution(phenotype_df, qtl_type, result)
        
        # Check normalization compatibility
        self._check_normalization_compatibility(phenotype_df, qtl_type, result)
        
        # Detect data quality issues
        self._detect_data_quality_issues(phenotype_df, qtl_type, result)
        
        logger.info(f"✅ {qtl_type} analysis: {phenotype_df.shape[0]} features, "
                   f"{phenotype_df.shape[1]} samples, "
                   f"{result.phenotype_info[qtl_type]['missing_percentage']:.2f}% missing")
    
    def _analyze_phenotype_distribution(self, phenotype_df, qtl_type, result):
        """Analyze phenotype data distribution"""
        try:
            # Sample a subset for efficiency
            if phenotype_df.shape[0] > 1000:
                sample_df = phenotype_df.sample(n=1000, random_state=42)
            else:
                sample_df = phenotype_df
            
            # Calculate basic statistics
            stats_data = {}
            
            # Mean and variance
            means = sample_df.mean(axis=1)
            variances = sample_df.var(axis=1)
            
            stats_data['mean_mean'] = float(means.mean())
            stats_data['mean_std'] = float(means.std())
            stats_data['variance_mean'] = float(variances.mean())
            stats_data['variance_std'] = float(variances.std())
            
            # Detect distribution type based on QTL type
            if qtl_type == 'eqtl':
                # Expression data - typically count-like or continuous
                if (sample_df >= 0).all().all():
                    if (sample_df == 0).any().any():
                        stats_data['distribution_type'] = 'count_like'
                    else:
                        stats_data['distribution_type'] = 'continuous_positive'
                else:
                    stats_data['distribution_type'] = 'continuous'
                    
            elif qtl_type == 'pqtl':
                # Protein data - often continuous
                stats_data['distribution_type'] = 'continuous'
                
            elif qtl_type == 'sqtl':
                # Splicing data - often proportions (0-1)
                if ((sample_df >= 0) & (sample_df <= 1)).all().all():
                    stats_data['distribution_type'] = 'proportion'
                else:
                    stats_data['distribution_type'] = 'continuous'
            
            result.phenotype_info[qtl_type]['distribution'] = stats_data
            
        except Exception as e:
            logger.warning(f"Phenotype distribution analysis failed: {e}")
    
    def _check_normalization_compatibility(self, phenotype_df, qtl_type, result):
        """Check if configured normalization is compatible with data"""
        norm_config = self.normalization_config.get(qtl_type, {})
        norm_method = norm_config.get('method', 'log2')
        
        compatibility_notes = []
        
        if qtl_type == 'eqtl':
            if norm_method == 'vst' and (phenotype_df < 0).any().any():
                compatibility_notes.append("VST normalization expects non-negative data")
            elif norm_method == 'log2' and (phenotype_df <= 0).any().any():
                compatibility_notes.append("Log2 normalization requires positive values")
                
        elif qtl_type == 'sqtl':
            if norm_method == 'log2' and ((phenotype_df < 0) | (phenotype_df > 1)).any().any():
                compatibility_notes.append("Log2 normalization may not be ideal for PSI values outside [0,1]")
        
        if compatibility_notes:
            result.add_warning(f'normalization_{qtl_type}', 
                             f"Normalization compatibility: {'; '.join(compatibility_notes)}")
        else:
            result.add_info(f'normalization_{qtl_type}', 
                          f"Configured normalization '{norm_method}' appears compatible")
    
    def _detect_data_quality_issues(self, phenotype_df, qtl_type, result):
        """Detect potential data quality issues"""
        issues = []
        
        # Check for constant features
        constant_features = (phenotype_df.nunique(axis=1) == 1).sum()
        if constant_features > 0:
            issues.append(f"{constant_features} constant features")
        
        # Check for zero-variance features
        zero_variance = (phenotype_df.std(axis=1) == 0).sum()
        if zero_variance > 0:
            issues.append(f"{zero_variance} zero-variance features")
        
        # Check for excessive missingness
        missing_threshold = 0.2  # 20% missing threshold
        high_missing_features = (phenotype_df.isna().sum(axis=1) / phenotype_df.shape[1] > missing_threshold).sum()
        if high_missing_features > 0:
            issues.append(f"{high_missing_features} features with >{missing_threshold*100}% missing values")
        
        # Check for outliers
        if phenotype_df.size > 0:
            Q1 = phenotype_df.quantile(0.25)
            Q3 = phenotype_df.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (phenotype_df < (Q1 - 3 * IQR)) | (phenotype_df > (Q3 + 3 * IQR))
            outlier_count = outlier_mask.sum().sum()
            if outlier_count > 0:
                outlier_percent = (outlier_count / phenotype_df.size) * 100
                if outlier_percent > 5:  # More than 5% outliers
                    issues.append(f"{outlier_percent:.1f}% potential outliers")
        
        if issues:
            result.add_warning(f'data_quality_{qtl_type}', 
                             f"Data quality issues: {'; '.join(issues)}")

def validate_inputs(config):
    """Validate all input files and data consistency with comprehensive checks for tensorQTL - ENHANCED"""
    input_files = config['input_files']
    result = ValidationResult()
    
    print("🔍 Starting comprehensive input validation for tensorQTL pipeline...")
    logger.info("Starting comprehensive input validation")
    
    try:
        # Setup directories using directory manager
        results_dir = config.get('results_dir', 'results')
        validation_dirs = setup_validation_directories(results_dir)
        
        # Initialize dynamic analyzers
        covariate_analyzer = DynamicCovariateAnalyzer(config)
        phenotype_analyzer = DynamicPhenotypeAnalyzer(config)
        
        # Validate mandatory files with enhanced parallel processing
        mandatory_files = ['genotypes', 'covariates', 'annotations']
        validate_mandatory_files_parallel(input_files, mandatory_files, config, result)
        
        # Validate phenotype files based on analysis types
        analysis_types = get_qtl_types_from_config(config)
        result.data_types_available = analysis_types
        validate_phenotype_files_parallel(input_files, analysis_types, config, result)
        
        # NEW: Perform dynamic analysis of covariates and phenotypes
        if 'covariates' in input_files and input_files['covariates'] and os.path.exists(input_files['covariates']):
            try:
                # FIXED: Use the corrected covariate reading function
                cov_df = read_covariates_file_robust(input_files['covariates'])
                if not cov_df.empty:
                    covariate_analyzer.analyze_covariates(cov_df, result)
            except Exception as e:
                logger.warning(f"Dynamic covariate analysis failed: {e}")
        
        for analysis_type in analysis_types:
            config_key = map_qtl_type_to_config_key(analysis_type)
            if config_key in input_files and input_files[config_key] and os.path.exists(input_files[config_key]):
                try:
                    pheno_df = pd.read_csv(input_files[config_key], sep='\t', index_col=0, nrows=1000)
                    if not pheno_df.empty:
                        phenotype_analyzer.analyze_phenotype_data(pheno_df, analysis_type, result)
                except Exception as e:
                    logger.warning(f"Dynamic phenotype analysis for {analysis_type} failed: {e}")
        
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
        
        # Generate validation report using consistent directories
        generate_validation_report(result, config, validation_dirs)
        
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
            
            # NEW: Print dynamic analysis summary
            print_dynamic_analysis_summary(result)
            
            return True
            
    except Exception as e:
        logger.error(f"Validation process failed: {e}")
        result.add_error('validation_process', f"Validation process failed: {e}")
        raise

def setup_validation_directories(results_dir):
    """Setup consistent directories for validation using directory manager"""
    try:
        if get_module_directories:
            validation_dirs = get_module_directories(
                'validation',
                [
                    'reports',
                    {'reports': ['pipeline_reports', 'qc_reports']},
                    'processed_data',
                    {'processed_data': ['quality_control']},
                    'system',
                    {'system': ['logs']}
                ],
                results_dir
            )
            logger.info(f"✅ Validation directories setup using directory manager")
            return validation_dirs
        else:
            # Fallback directory creation
            validation_dirs = {
                'reports_pipeline_reports': Path(results_dir) / "reports" / "pipeline_reports",
                'reports_qc_reports': Path(results_dir) / "reports" / "qc_reports",
                'processed_data_quality_control': Path(results_dir) / "processed_data" / "quality_control",
                'system_logs': Path(results_dir) / "system" / "logs"
            }
            
            for dir_path in validation_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Fallback validation directories created")
            return validation_dirs
            
    except Exception as e:
        logger.error(f"❌ Directory setup failed: {e}")
        # Ultimate fallback
        validation_dirs = {
            'reports_pipeline_reports': Path(results_dir) / "reports",
            'reports_qc_reports': Path(results_dir) / "reports",
            'processed_data_quality_control': Path(results_dir) / "quality_control",
            'system_logs': Path(results_dir) / "logs"
        }
        
        for dir_path in validation_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.warning(f"⚠️ Using ultimate fallback directories")
        return validation_dirs

def read_covariates_file_robust(file_path):
    """FIXED: Robust covariate file reading that handles your specific format"""
    try:
        # First attempt: standard tab-separated with header
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        logger.info(f"Successfully read covariates file with standard tab separation")
        return df
    except Exception as e:
        logger.warning(f"Standard reading failed: {e}, trying alternative methods")
        
        try:
            # Second attempt: try different separators
            for sep in ['\t', '  ', ' ', ',']:
                try:
                    df = pd.read_csv(file_path, sep=sep, index_col=0, engine='python')
                    if df.shape[1] > 1:  # Should have multiple samples
                        logger.info(f"Successfully read covariates file with separator: {repr(sep)}")
                        return df
                except:
                    continue
        except Exception as e2:
            logger.warning(f"Alternative separator reading failed: {e2}")
        
        # Final attempt: manual parsing for complex formats
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find header line
            header_line = None
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    # Check if this looks like a header with sample IDs
                    parts = line.strip().split()
                    if len(parts) > 3 and all(len(part) > 3 for part in parts[1:4]):  # Sample IDs usually have reasonable length
                        header_line = i
                        break
            
            if header_line is not None:
                # Parse header and data
                header = lines[header_line].strip().split()
                data_lines = []
                
                for i in range(header_line + 1, len(lines)):
                    if lines[i].strip() and not lines[i].startswith('#'):
                        data_lines.append(lines[i].strip().split())
                
                if data_lines and len(data_lines[0]) == len(header):
                    # Create DataFrame
                    data_dict = {}
                    for row in data_lines:
                        if len(row) == len(header):
                            covar_name = row[0]
                            values = row[1:]
                            data_dict[covar_name] = values
                    
                    df = pd.DataFrame(data_dict, index=header[1:]).T
                    logger.info(f"Successfully read covariates file with manual parsing")
                    return df
                    
        except Exception as e3:
            logger.error(f"Manual parsing also failed: {e3}")
            raise ValueError(f"Could not parse covariates file with any method: {e3}")
    
    raise ValueError("All covariate file reading methods failed")

def print_dynamic_analysis_summary(result):
    """NEW: Print summary of dynamic covariate and phenotype analysis"""
    print("\n📊 DYNAMIC ANALYSIS SUMMARY:")
    
    # Covariate summary
    if result.covariate_info:
        print("  📈 Covariates:")
        print(f"    - Total: {result.covariate_info.get('total_covariates', 'N/A')}")
        print(f"    - Numeric: {result.covariate_info.get('numeric_count', 'N/A')}")
        print(f"    - Categorical: {result.covariate_info.get('categorical_count', 'N/A')}")
        print(f"    - Binary: {result.covariate_info.get('binary_count', 'N/A')}")
        
        # Print detected patterns
        if 'pca_components' in result.covariate_info:
            pca_count = len(result.covariate_info['pca_components'])
            print(f"    - PCA components: {pca_count}")
        
        if 'batch_covariates' in result.covariate_info:
            print(f"    - Batch covariates: {result.covariate_info['batch_covariates']}")
    
    # Phenotype summary
    if result.phenotype_info:
        print("  🧬 Phenotypes:")
        for qtl_type, info in result.phenotype_info.items():
            print(f"    - {qtl_type.upper()}: {info.get('feature_count', 'N/A')} features, "
                  f"{info.get('sample_count', 'N/A')} samples, "
                  f"{info.get('missing_percentage', 0):.1f}% missing")
            
            dist_info = info.get('distribution', {})
            if 'distribution_type' in dist_info:
                print(f"      Distribution: {dist_info['distribution_type']}")

def validate_covariates_file_enhanced(file_path, config, result):
    """Enhanced covariates file validation with dynamic handling - FIXED for your format"""
    try:
        # FIXED: Use robust reading function
        df = read_covariates_file_robust(file_path)
        
        if df.empty:
            result.add_error('covariates', "Covariates file is empty")
            return
        
        # Store sample count
        sample_count = df.shape[1]
        result.sample_counts['covariates'] = sample_count
        result.add_info('covariates', f"Found {df.shape[0]} covariates for {sample_count} samples")
        
        # NEW: Flexible covariate validation - don't enforce specific covariates
        self_reported_covariates = df.index.tolist()
        result.add_info('covariates', f"Covariates provided: {', '.join(self_reported_covariates[:10])}"
                       f"{'...' if len(self_reported_covariates) > 10 else ''}")
        
        # Check for missing values with dynamic threshold
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
        
        # Check for numeric covariates with flexible handling
        non_numeric_covariates = []
        numeric_covariates = []
        
        for covar in df.index:
            try:
                numeric_series = pd.to_numeric(df.loc[covar], errors='coerce')
                non_na_count = numeric_series.notna().sum()
                total_count = len(df.loc[covar])
                
                if non_na_count == total_count:
                    numeric_covariates.append(covar)
                else:
                    non_numeric_covariates.append(covar)
            except (ValueError, TypeError):
                non_numeric_covariates.append(covar)
        
        if non_numeric_covariates:
            result.add_info('covariates', f"Found {len(non_numeric_covariates)} non-numeric covariates: {non_numeric_covariates[:5]}...")
        
        # NEW: Dynamic check for extreme values only on numeric covariates
        if numeric_covariates:
            numeric_df = df.loc[numeric_covariates].apply(pd.to_numeric, errors='coerce')
            extreme_threshold = 10  # Z-score threshold
            # Avoid division by zero
            numeric_std = numeric_df.std()
            numeric_std = numeric_std.replace(0, 1)  # Replace zero std with 1 to avoid division by zero
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_std)
            extreme_count = (z_scores > extreme_threshold).sum().sum()
            if extreme_count > 0:
                result.add_warning('covariates', f"Found {extreme_count} extreme values (|Z| > {extreme_threshold}) in numeric covariates")
        
        # NEW: Check covariate count relative to sample size
        if df.shape[0] > df.shape[1] / 2:
            result.add_warning('covariates', 
                             f"High covariate count ({df.shape[0]}) relative to sample size ({df.shape[1]})")
        
    except Exception as e:
        result.add_error('covariates', f"Error reading covariates file: {e}")

def validate_phenotype_file_enhanced(file_path, qtl_type, config, result):
    """Enhanced phenotype file validation with dynamic handling"""
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
        
        # NEW: Dynamic phenotype validation based on QTL type
        self_reported_features = df.index.tolist()[:5]  # First 5 features
        result.add_info(qtl_type, f"Sample features: {', '.join(self_reported_features)}...")
        
        # Check for missing values with dynamic thresholds
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
        
        # NEW: Dynamic data distribution checks based on QTL type
        data_type = map_qtl_type_to_data_type(qtl_type)
        if data_type == 'expression':
            # Check for negative values in expression data
            negative_count = (df < 0).sum().sum()
            if negative_count > 0:
                result.add_info(qtl_type, f"Found {negative_count} negative values - ensure proper normalization")
        
        elif data_type == 'splicing':
            # Check if data appears to be proportions (PSI values)
            if ((df >= 0) & (df <= 1)).all().all():
                result.add_info(qtl_type, "Data appears to be proportion-based (PSI values 0-1)")
            else:
                result.add_info(qtl_type, "Data range suggests non-standard PSI values")
        
        # NEW: Check for extreme outliers with dynamic thresholds
        if df.size > 0:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            # Avoid division by zero in IQR
            iqr_nonzero = IQR.replace(0, 1)
            outlier_mask = (df < (Q1 - 3 * iqr_nonzero)) | (df > (Q3 + 3 * iqr_nonzero))
            outlier_count = outlier_mask.sum().sum()
            
            if outlier_count > 0:
                outlier_percent = (outlier_count / (df.shape[0] * df.shape[1])) * 100
                result.add_info(qtl_type, f"Found {outlier_count} potential outliers ({outlier_percent:.2f}%)")
        
        # NEW: Check normalization method compatibility
        norm_method = config['normalization'].get(qtl_type, {}).get('method', 'unknown')
        result.add_info(qtl_type, f"Configured normalization: {norm_method}")
        
        # Calculate basic statistics
        if df.size > 0:
            mean_val = df.mean().mean()
            std_val = df.std().mean()
            result.add_info(qtl_type, f"Data stats - Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        
    except Exception as e:
        result.add_error(qtl_type, f"Error reading phenotype file {file_path}: {e}")

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
        bcftools_threads = config.get('genotype_processing', {}).get('bcftools_threads', 1)
        
        # Test basic VCF reading
        cmd = f"{bcftools_path} view -h {file_path} --threads {bcftools_threads} 2>/dev/null | head -10 || echo 'ERROR'"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        if "ERROR" in process.stdout or process.returncode != 0:
            result.add_error('vcf', f"bcftools cannot read VCF file. Check file format and compression.")
            return
        
        # Extract sample count
        samples_cmd = f"{bcftools_path} query -l {file_path} --threads {bcftools_threads} 2>/dev/null | wc -l"
        samples_process = subprocess.run(samples_cmd, shell=True, capture_output=True, text=True)
        if samples_process.returncode == 0 and samples_process.stdout.strip().isdigit():
            sample_count = int(samples_process.stdout.strip())
            result.sample_counts['genotypes'] = sample_count
            result.add_info('vcf', f"Found {sample_count} samples in VCF")
        else:
            result.add_warning('vcf', "Could not count samples in VCF file")
        
        # Extract variant count estimate
        variants_cmd = f"{bcftools_path} view -H {file_path} --threads {bcftools_threads} 2>/dev/null | head -1000 | wc -l"
        variants_process = subprocess.run(variants_cmd, shell=True, capture_output=True, text=True)
        if variants_process.returncode == 0 and variants_process.stdout.strip().isdigit():
            variant_sample = int(variants_process.stdout.strip())
            if variant_sample == 0:
                result.add_error('vcf', "No variants found in VCF file")
            else:
                result.add_info('vcf', f"VCF contains variants (sampled {variant_sample})")
        
        # Check chromosome naming consistency
        chrom_cmd = f"{bcftools_path} view -H {file_path} --threads {bcftools_threads} 2>/dev/null | cut -f1 | head -100 | sort | uniq"
        chrom_process = subprocess.run(chrom_cmd, shell=True, capture_output=True, text=True)
        if chrom_process.returncode == 0:
            chromosomes = [c.strip() for c in chrom_process.stdout.split('\n') if c.strip()]
            analyze_chromosome_naming(chromosomes, 'vcf', result)
        
        # Check for required INFO and FORMAT fields
        header_cmd = f"{bcftools_path} view -h {file_path} --threads {bcftools_threads} 2>/dev/null | grep -E '^##(INFO|FORMAT)' | head -10"
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
    """Comprehensive configuration validation with dynamic checks"""
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
    
    # NEW: Dynamic normalization configuration validation
    normalization_config = config.get('normalization', {})
    for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
        if qtl_type in normalization_config:
            method = normalization_config[qtl_type].get('method', 'unknown')
            pseudocount = normalization_config[qtl_type].get('log2_pseudocount', 1)
            
            result.add_info('configuration', f"{qtl_type.upper()} normalization: {method} (pseudocount: {pseudocount})")
            
            # Provide method-specific recommendations
            if method == 'vst' and qtl_type != 'eqtl':
                result.add_warning('configuration', f"VST normalization is typically for expression data, not {qtl_type}")
    
    # Enhanced features
    if config.get('enhanced_qc', {}).get('enable', False):
        result.add_info('configuration', "Enhanced QC enabled")
    
    if config.get('interaction_analysis', {}).get('enable', False):
        result.add_info('configuration', "Interaction analysis enabled")
    
    if config.get('fine_mapping', {}).get('enable', False):
        result.add_info('configuration', "Fine-mapping enabled")
    
    # NEW: Validate dynamic covariate handling
    covariates_file = config['input_files'].get('covariates')
    if covariates_file and os.path.exists(covariates_file):
        try:
            cov_df = read_covariates_file_robust(covariates_file)
            covariate_count = cov_df.shape[0]
            sample_count = cov_df.shape[1]
            result.add_info('configuration', f"Covariate structure: {covariate_count} covariates for {sample_count} samples")
        except:
            pass

def generate_validation_report(result, config, validation_dirs=None):
    """Generate comprehensive validation report using consistent directories"""
    if validation_dirs is None:
        results_dir = config.get('results_dir', 'results')
        validation_dirs = setup_validation_directories(results_dir)
    
    # Use consistent directories from directory manager
    report_dir = validation_dirs.get('reports_qc_reports', Path(config.get('results_dir', 'results')) / 'reports' / 'qc_reports')
    qc_dir = validation_dirs.get('processed_data_quality_control', Path(config.get('results_dir', 'results')) / 'processed_data' / 'quality_control')
    
    # Ensure directories exist
    report_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    report_file = report_dir / 'input_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    # Generate text report
    text_report_file = report_dir / 'input_validation_report.txt'
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
        
        # NEW: Add covariate information
        if result.covariate_info:
            f.write("Covariate Information:\n")
            f.write(f"  - Total covariates: {result.covariate_info.get('total_covariates', 'N/A')}\n")
            f.write(f"  - Numeric covariates: {result.covariate_info.get('numeric_count', 'N/A')}\n")
            f.write(f"  - Categorical covariates: {result.covariate_info.get('categorical_count', 'N/A')}\n")
            f.write(f"  - Binary covariates: {result.covariate_info.get('binary_count', 'N/A')}\n")
            if 'pca_components' in result.covariate_info:
                f.write(f"  - PCA components: {len(result.covariate_info['pca_components'])}\n")
            f.write("\n")
        
        # NEW: Add phenotype information
        if result.phenotype_info:
            f.write("Phenotype Information:\n")
            for qtl_type, info in result.phenotype_info.items():
                f.write(f"  - {qtl_type.upper()}: {info.get('feature_count', 'N/A')} features, "
                       f"{info.get('sample_count', 'N/A')} samples, "
                       f"{info.get('missing_percentage', 0):.1f}% missing\n")
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
    
    # Generate sample mapping file in quality control directory
    sample_mapping_file = qc_dir / 'sample_mapping.txt'
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
            bcftools_threads = config.get('genotype_processing', {}).get('bcftools_threads', 1)
            cmd = f"{bcftools_path} query -l {genotype_file} --threads {bcftools_threads} 2>/dev/null"
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
        # Use robust reading for covariate file
        df = read_covariates_file_robust(covariates_file)
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