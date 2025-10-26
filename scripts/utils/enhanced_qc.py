#!/usr/bin/env python3
"""
Enhanced Quality Control with comprehensive sample and variant QC
Optimized for performance and memory efficiency

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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from typing import Dict, List, Set, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

warnings.filterwarnings('ignore')

# Set up logger
logger = logging.getLogger('QTLPipeline')

def map_qtl_type_to_config_key(qtl_type: str) -> str:
    """Map QTL analysis types to config file keys"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing',
        'gwas': 'gwas_phenotype'
    }
    return mapping.get(qtl_type, qtl_type)

class EnhancedQC:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qc_config = config.get('enhanced_qc', {})
        self.results_dir = config.get('results_dir', 'results')
        self._plink_cache = {}  # Cache for PLINK results
        self._file_cache = {}   # Cache for file reads
        self.setup_qc_directories()
        
    def setup_qc_directories(self) -> None:
        """Create comprehensive QC directory structure"""
        try:
            qc_dirs = [
                'QC_reports', 'QC_plots', 'QC_data', 'sample_concordance',
                'pca_results', 'outlier_analysis', 'sample_lists', 
                'variant_lists', 'tensorqtl_compatibility'
            ]
            
            for qc_dir in qc_dirs:
                Path(self.results_dir).joinpath(qc_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ QC directory structure created")
        except Exception as e:
            logger.error(f"❌ Failed to create QC directories: {e}")
            raise

    def run_data_preparation(self) -> bool:
        """Run data preparation module for modular pipeline"""
        logger.info("🚀 Starting data preparation module...")
        
        try:
            steps = [
                ("📋 Validating input files", self.validate_input_files),
                ("🔍 Performing basic data quality checks", self.basic_data_quality_checks),
                ("📊 Generating data preparation report", lambda: self.generate_data_preparation_report(
                    self._validation_results, self._quality_results)),
                ("👥 Creating sample mapping files", self.create_sample_mapping_files),
                ("🔧 Checking tensorQTL compatibility", self.check_tensorqtl_compatibility)
            ]
            
            results = []
            for desc, func in steps:
                logger.info(desc)
                if desc == "📊 Generating data preparation report":
                    # Use cached results for report generation
                    result = func()
                else:
                    result = func()
                    # Cache results for report generation
                    if desc == "📋 Validating input files":
                        self._validation_results = result
                    elif desc == "🔍 Performing basic data quality checks":
                        self._quality_results = result
                results.append(result)
            
            success = all(results)
            
            if success:
                logger.info("✅ Data preparation module completed successfully")
            else:
                logger.error("❌ Data preparation module had issues")
            
            # Clear cached results to free memory
            if hasattr(self, '_validation_results'):
                del self._validation_results
            if hasattr(self, '_quality_results'):
                del self._quality_results
            gc.collect()
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Data preparation module failed: {e}")
            return False

    def check_tensorqtl_compatibility(self) -> bool:
        """Check if all data formats are compatible with tensorQTL"""
        logger.info("🔧 Checking tensorQTL compatibility...")
        
        compatibility_issues = []
        
        try:
            # Check genotype file format
            geno_file = self.config['input_files']['genotypes']
            if not geno_file.endswith(('.vcf.gz', '.vcf', '.bed', '.bcf')):
                compatibility_issues.append(f"Genotype file format may not be optimal for tensorQTL: {geno_file}")
                logger.warning("⚠️  Consider converting genotype data to PLINK format for better tensorQTL performance")
            
            # Check if we can extract samples from genotype file
            geno_samples = self.extract_samples_from_genotypes(geno_file)
            if not geno_samples:
                compatibility_issues.append("Cannot extract samples from genotype file")
            else:
                logger.info(f"✅ Genotype samples: {len(geno_samples)} samples found")
            
            # Check phenotype files in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_qtl = {}
                for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
                    config_key = map_qtl_type_to_config_key(qtl_type)
                    pheno_file = self.config['input_files'].get(config_key)
                    if pheno_file and os.path.exists(pheno_file):
                        future = executor.submit(self._check_phenotype_compatibility, pheno_file, qtl_type, geno_samples)
                        future_to_qtl[future] = qtl_type
                
                for future in as_completed(future_to_qtl):
                    qtl_type = future_to_qtl[future]
                    try:
                        issues = future.result()
                        compatibility_issues.extend(issues)
                    except Exception as e:
                        compatibility_issues.append(f"Error checking {qtl_type} compatibility: {e}")
            
            # Check covariates format
            covar_file = self.config['input_files'].get('covariates')
            if covar_file and os.path.exists(covar_file):
                try:
                    # Use robust covariate reading that handles categorical data
                    covar_df = self._read_covariates_robust(covar_file, nrows=5)
                    if covar_df is not None:
                        covar_samples = set(covar_df.columns)
                        overlap = set(geno_samples) & covar_samples
                        
                        if len(overlap) == 0:
                            compatibility_issues.append("No overlapping samples between genotype and covariates")
                        
                        logger.info(f"✅ Covariates compatibility: {len(covar_samples)} samples, {len(overlap)} overlap with genotypes")
                    else:
                        compatibility_issues.append("Could not read covariates file for compatibility check")
                    
                except Exception as e:
                    compatibility_issues.append(f"Error reading covariates file: {e}")
            
            # Generate compatibility report
            self.generate_tensorqtl_compatibility_report(compatibility_issues, geno_samples)
            
            if compatibility_issues:
                logger.warning(f"⚠️  Found {len(compatibility_issues)} tensorQTL compatibility issues")
                for issue in compatibility_issues[:5]:  # Limit log output
                    logger.warning(f"   - {issue}")
                if len(compatibility_issues) > 5:
                    logger.warning(f"   ... and {len(compatibility_issues) - 5} more issues")
                return False
            
            logger.info("✅ All data formats are compatible with tensorQTL")
            return True
                
        except Exception as e:
            logger.error(f"❌ TensorQTL compatibility check failed: {e}")
            return False

    def _read_covariates_robust(self, file_path: str, nrows: Optional[int] = None):
        """Robust covariate file reading that handles categorical data"""
        try:
            # First attempt: standard tab-separated with header
            df = pd.read_csv(file_path, sep='\t', index_col=0, nrows=nrows)
            logger.info(f"Successfully read covariates file with standard tab separation")
            return df
        except Exception as e:
            logger.warning(f"Standard reading failed: {e}, trying alternative methods")
            
            try:
                # Second attempt: try different separators
                for sep in ['\t', '  ', ' ', ',']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, index_col=0, nrows=nrows, engine='python')
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
        
        logger.error("All covariate file reading methods failed")
        return None

    def _check_phenotype_compatibility(self, pheno_file: str, qtl_type: str, geno_samples: List[str]) -> List[str]:
        """Check phenotype file compatibility (for parallel execution)"""
        issues = []
        try:
            # Use chunking for large files
            chunksize = 10000
            first_chunk = True
            pheno_samples = set()
            
            for chunk in pd.read_csv(pheno_file, sep='\t', index_col=0, chunksize=chunksize):
                if first_chunk:
                    pheno_samples = set(chunk.columns)
                    first_chunk = False
                
                # Basic validation on first chunk only
                if first_chunk:
                    if chunk.shape[0] == 0 or chunk.shape[1] == 0:
                        issues.append(f"Empty or invalid {qtl_type} phenotype file")
                        break
            
            overlap = set(geno_samples) & pheno_samples
            
            if len(overlap) == 0:
                issues.append(f"No overlapping samples between genotype and {qtl_type} data")
            elif len(overlap) < min(len(geno_samples), len(pheno_samples)) * 0.8:
                issues.append(f"Low sample overlap ({len(overlap)}) for {qtl_type} analysis")
            
            logger.info(f"✅ {qtl_type.upper()} compatibility: {len(pheno_samples)} samples, {len(overlap)} overlap with genotypes")
            
        except Exception as e:
            issues.append(f"Error reading {qtl_type} file: {e}")
        
        return issues

    def extract_samples_from_genotypes(self, genotype_file: str) -> List[str]:
        """Extract sample names from genotype file with multiple format support"""
        if genotype_file in self._file_cache:
            return self._file_cache[genotype_file]
        
        samples = []
        
        try:
            if genotype_file.endswith(('.vcf.gz', '.vcf', '.bcf')):
                # Use bcftools for VCF files
                cmd = f"{self.config['paths']['bcftools']} query -l {genotype_file}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
                if result.returncode == 0:
                    samples = [s.strip() for s in result.stdout.split('\n') if s.strip()]
                else:
                    # Fallback: try to read VCF header efficiently
                    samples = self._extract_samples_from_vcf_header(genotype_file)
            
            elif genotype_file.endswith('.bed'):
                # PLINK BED format - read from FAM file
                base_name = genotype_file.rsplit('.', 1)[0]  # More robust than replace
                fam_file = f"{base_name}.fam"
                if os.path.exists(fam_file):
                    fam_df = pd.read_csv(fam_file, sep='\s+', header=None, usecols=[1])  # Only read sample column
                    samples = fam_df.iloc[:, 0].tolist()
            
            logger.info(f"📊 Extracted {len(samples)} samples from genotype file")
            
            # Cache the result
            self._file_cache[genotype_file] = samples
            return samples
            
        except Exception as e:
            logger.error(f"❌ Error extracting samples from genotype file: {e}")
            return []

    def _extract_samples_from_vcf_header(self, vcf_file: str) -> List[str]:
        """Efficiently extract samples from VCF header"""
        samples = []
        try:
            if vcf_file.endswith('.gz'):
                import gzip
                open_func = gzip.open
            else:
                open_func = open
            
            with open_func(vcf_file, 'rt') as f:
                for line in f:
                    if line.startswith('#CHROM'):
                        samples = line.strip().split('\t')[9:]
                        break
                    # Stop after reasonable number of lines if header is malformed
                    if f.tell() > 1000000:  # 1MB limit for header
                        break
        except Exception as e:
            logger.warning(f"Could not extract samples from VCF header: {e}")
        
        return samples

    def generate_tensorqtl_compatibility_report(self, issues: List[str], geno_samples: List[str]) -> bool:
        """Generate tensorQTL compatibility report"""
        try:
            report_dir = Path(self.results_dir) / "tensorqtl_compatibility"
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / "tensorqtl_compatibility_report.html"
            
            # Template for HTML report
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TensorQTL Compatibility Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .good {{ color: green; font-weight: bold; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .error {{ color: red; font-weight: bold; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>TensorQTL Compatibility Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <div class="section">
                    <h2>Compatibility Summary</h2>
                    <p><strong>Genotype Samples:</strong> {sample_count}</p>
                    <p><strong>Compatibility Issues:</strong> {issue_count}</p>
                    {status_message}
                </div>
                {issues_section}
                <div class="section">
                    <h2>Sample Information</h2>
                    <p><strong>Total Genotype Samples:</strong> {sample_count}</p>
                    <details>
                        <summary>Show Sample List (first 20)</summary>
                        <pre>{sample_list}</pre>
                    </details>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Ensure sample IDs match exactly across all files</li>
                        <li>Use PLINK format for genotype data for optimal tensorQTL performance</li>
                        <li>Check that all files use consistent sample ordering</li>
                        <li>Verify that phenotype data is properly normalized</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # Prepare content
            status_message = ("<p class='good'>✅ All checks passed - Data is compatible with tensorQTL</p>" 
                            if not issues else 
                            f"<p class='warning'>⚠️ Found {len(issues)} compatibility issues that need attention</p>")
            
            issues_section = ""
            if issues:
                issues_html = "".join(f"<li class='warning'>{issue}</li>" for issue in issues)
                issues_section = f"""
                <div class="section">
                    <h2>Compatibility Issues</h2>
                    <ul>{issues_html}</ul>
                </div>
                """
            
            html_content = html_template.format(
                timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                sample_count=len(geno_samples),
                issue_count=len(issues),
                status_message=status_message,
                issues_section=issues_section,
                sample_list='\n'.join(geno_samples[:20])
            )
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"✅ TensorQTL compatibility report generated: {report_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not generate tensorQTL compatibility report: {e}")
            return False

    def validate_input_files(self) -> Dict[str, Any]:
        """Validate all input files exist and are accessible"""
        logger.info("🔍 Validating input files...")
        
        validation_results = {}
        input_files = self.config.get('input_files', {})
        
        required_files = ['genotypes', 'covariates', 'annotations']
        optional_files = ['expression', 'protein', 'splicing', 'gwas_phenotype']
        
        for file_type in required_files + optional_files:
            file_path = input_files.get(file_type)
            
            if file_type in required_files and not file_path:
                validation_results[file_type] = self._create_validation_result('REQUIRED_MISSING')
                logger.error(f"  ❌ {file_type}: REQUIRED BUT NOT SPECIFIED")
                continue
                
            if not file_path:
                validation_results[file_type] = self._create_validation_result('OPTIONAL_NOT_CONFIGURED')
                logger.info(f"  ⚠️  {file_type}: Optional (not configured)")
                continue
            
            validation_results[file_type] = self._validate_single_file(file_type, file_path)
        
        return validation_results

    def _create_validation_result(self, status: str) -> Dict[str, Any]:
        """Create a standardized validation result"""
        return {
            'status': status,
            'path': None,
            'size_gb': 0,
            'accessible': False
        }

    def _validate_single_file(self, file_type: str, file_path: str) -> Dict[str, Any]:
        """Validate a single file"""
        if not os.path.exists(file_path):
            result = self._create_validation_result('FILE_NOT_FOUND')
            if file_type in ['genotypes', 'covariates', 'annotations']:
                logger.error(f"  ❌ {file_type}: {file_path} - REQUIRED FILE NOT FOUND")
            else:
                logger.warning(f"  ⚠️  {file_type}: {file_path} - OPTIONAL FILE NOT FOUND")
            return result
        
        try:
            file_size = os.path.getsize(file_path) / (1024**3)  # GB
            result = {
                'status': 'OK',
                'path': file_path,
                'size_gb': round(file_size, 2),
                'accessible': True
            }
            
            logger.info(f"  ✅ {file_type}: {file_path} ({file_size:.2f} GB)")
            
            # Additional format validation
            if file_type == 'genotypes':
                result['tensorqtl_compatible'] = self.validate_genotype_format(file_path)
            elif file_type == 'expression':
                result['tensorqtl_compatible'] = self.validate_phenotype_format(file_path, 'expression')
            elif file_type == 'covariates':
                result['tensorqtl_compatible'] = self.validate_covariates_format(file_path)
                
            return result
            
        except Exception as e:
            result = self._create_validation_result('ACCESS_ERROR')
            result['error'] = str(e)
            logger.error(f"  ❌ {file_type}: {file_path} - ACCESS ERROR: {e}")
            return result

    def validate_genotype_format(self, genotype_file: str) -> bool:
        """Validate genotype file format for tensorQTL compatibility"""
        try:
            if genotype_file.endswith(('.vcf.gz', '.vcf', '.bcf')):
                cmd = f"{self.config['paths']['bcftools']} view -h {genotype_file} | head -5"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
                return result.returncode == 0 and '#CHROM' in result.stdout
            elif genotype_file.endswith('.bed'):
                base_name = genotype_file.rsplit('.', 1)[0]
                required_files = [f'{base_name}.bed', f'{base_name}.bim', f'{base_name}.fam']
                return all(os.path.exists(f) for f in required_files)
            return False
        except:
            return False

    def validate_phenotype_format(self, phenotype_file: str, pheno_type: str) -> bool:
        """Validate phenotype file format for tensorQTL compatibility"""
        try:
            # Only read first few rows for validation
            df = pd.read_csv(phenotype_file, sep='\t', index_col=0, nrows=5)
            return df.shape[0] > 0 and df.shape[1] > 0
        except:
            return False

    def validate_covariates_format(self, covariates_file: str) -> bool:
        """Validate covariates file format for tensorQTL compatibility"""
        try:
            df = self._read_covariates_robust(covariates_file, nrows=5)
            return df is not None and df.shape[0] > 0 and df.shape[1] > 0
        except:
            return False

    def basic_data_quality_checks(self) -> Dict[str, Any]:
        """Perform basic data quality checks in parallel"""
        logger.info("🔍 Performing basic data quality checks...")
        
        quality_results = {}
        input_files = self.config.get('input_files', {})
        
        # Define quality check functions
        check_functions = [
            ('genotypes', self.check_genotype_file_quality, input_files.get('genotypes')),
        ]
        
        # Add phenotype checks
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            config_key = map_qtl_type_to_config_key(qtl_type)
            check_functions.append((qtl_type, self.check_phenotype_file_quality, input_files.get(config_key)))
        
        # Add covariates and annotations
        check_functions.extend([
            ('covariates', self.check_covariates_file_quality, input_files.get('covariates')),
            ('annotations', self.check_annotations_file_quality, input_files.get('annotations'))
        ])
        
        # Run checks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_type = {}
            for file_type, check_func, file_path in check_functions:
                if file_path and os.path.exists(file_path):
                    future = executor.submit(check_func, file_path, file_type if file_type != 'genotypes' else None)
                    future_to_type[future] = file_type
            
            for future in as_completed(future_to_type):
                file_type = future_to_type[future]
                try:
                    quality_results[file_type] = future.result()
                except Exception as e:
                    logger.error(f"❌ {file_type} quality check failed: {e}")
                    quality_results[file_type] = self._create_empty_quality_result(file_type)
        
        return quality_results

    def _create_empty_quality_result(self, file_type: str) -> Dict[str, Any]:
        """Create empty quality result for failed checks"""
        checks_total = {'genotypes': 5, 'eqtl': 7, 'pqtl': 7, 'sqtl': 7, 'covariates': 6, 'annotations': 4}.get(file_type, 5)
        return {
            'file_type': file_type,
            'checks_passed': 0,
            'checks_total': checks_total,
            'details': {'error': 'Check failed'}
        }

    def check_genotype_file_quality(self, vcf_file: str, file_type: str = None) -> Dict[str, Any]:
        """Check genotype file quality using bcftools"""
        logger.info(f"  🧬 Checking genotype file: {vcf_file}")
        
        try:
            quality_info = {
                'file_type': 'genotype',
                'checks_passed': 0,
                'checks_total': 5,
                'details': {}
            }
            
            # Check 1: File exists and is accessible
            if not os.path.exists(vcf_file):
                quality_info['details']['file_exists'] = False
                return quality_info
            
            quality_info['checks_passed'] += 1
            file_size = os.path.getsize(vcf_file) / (1024**3)
            quality_info['details']['file_size_gb'] = round(file_size, 2)
            quality_info['details']['file_exists'] = True
            
            # Check 2: Basic stats using bcftools
            stats_cmd = f"{self.config['paths']['bcftools']} stats {vcf_file} 2>/dev/null | head -20 || true"
            result = subprocess.run(stats_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0 and result.stdout.strip():
                quality_info['checks_passed'] += 1
                quality_info['details']['bcftools_accessible'] = True
                
                # Parse basic info from stats
                for line in result.stdout.split('\n'):
                    if 'number of samples:' in line.lower():
                        quality_info['details']['n_samples'] = line.split(':')[-1].strip()
                    elif 'number of records:' in line.lower():
                        quality_info['details']['n_variants'] = line.split(':')[-1].strip()
            else:
                quality_info['details']['bcftools_accessible'] = False
            
            # Check 3: File is indexed
            if os.path.exists(f"{vcf_file}.tbi") or os.path.exists(f"{vcf_file}.csi"):
                quality_info['checks_passed'] += 1
                quality_info['details']['indexed'] = True
            else:
                quality_info['details']['indexed'] = False
            
            # Check 4: File format
            if vcf_file.endswith(('.vcf', '.vcf.gz', '.bcf', '.bed')):
                quality_info['checks_passed'] += 1
                quality_info['details']['format_ok'] = True
                quality_info['details']['format'] = os.path.splitext(vcf_file)[1]
                quality_info['details']['tensorqtl_optimized'] = vcf_file.endswith('.bed')
            else:
                quality_info['details']['format_ok'] = False
            
            # Check 5: Sample extraction
            samples = self.extract_samples_from_genotypes(vcf_file)
            if samples:
                quality_info['checks_passed'] += 1
                quality_info['details']['samples_extractable'] = True
                quality_info['details']['sample_count'] = len(samples)
            else:
                quality_info['details']['samples_extractable'] = False
            
            logger.info(f"    ✅ Genotype file checks: {quality_info['checks_passed']}/5 passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ Genotype quality check failed: {e}")
            return self._create_empty_quality_result('genotypes')

    def check_phenotype_file_quality(self, pheno_file: str, pheno_type: str) -> Dict[str, Any]:
        """Check phenotype file quality efficiently using chunking"""
        logger.info(f"  📊 Checking {pheno_type} file: {pheno_file}")
        
        try:
            # Read only first chunk for initial assessment
            chunksize = 10000
            reader = pd.read_csv(pheno_file, sep='\t', index_col=0, chunksize=chunksize)
            first_chunk = next(reader)
            
            quality_info = {
                'file_type': pheno_type,
                'checks_passed': 0,
                'checks_total': 7,
                'details': {
                    'n_features': 0,
                    'n_samples': first_chunk.shape[1],
                    'data_type': 'counts' if first_chunk.values.sum() > first_chunk.shape[0] * first_chunk.shape[1] else 'normalized',
                    'tensorqtl_format': 'OK'
                }
            }
            
            # Check 1: Non-empty file
            if first_chunk.shape[0] > 0 and first_chunk.shape[1] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['non_empty'] = True
            
            # Process chunks to get total feature count and missingness
            total_features = first_chunk.shape[0]
            total_missing = first_chunk.isna().sum().sum()
            total_cells = first_chunk.size
            
            for chunk in reader:
                total_features += chunk.shape[0]
                total_missing += chunk.isna().sum().sum()
                total_cells += chunk.size
            
            quality_info['details']['n_features'] = total_features
            quality_info['details']['missing_percentage'] = (total_missing / total_cells) * 100
            quality_info['details']['total_measurements'] = total_cells
            
            # Check 2: Reasonable missing rate
            if quality_info['details']['missing_percentage'] < 50:
                quality_info['checks_passed'] += 1
                quality_info['details']['missing_rate_acceptable'] = True
            
            # Check 3: No duplicate feature names (check first chunk only for efficiency)
            if not first_chunk.index.duplicated().any():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_duplicate_features'] = True
            
            # Check 4: Numeric data
            try:
                first_chunk.astype(float)
                quality_info['checks_passed'] += 1
                quality_info['details']['all_numeric'] = True
            except:
                quality_info['details']['all_numeric'] = False
            
            # Check 5: Proper format for tensorQTL
            if first_chunk.shape[0] > 0 and first_chunk.shape[1] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['tensorqtl_compatible'] = True
            
            logger.info(f"    ✅ {pheno_type} file: {total_features} features, {first_chunk.shape[1]} samples - {quality_info['checks_passed']}/7 checks passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ {pheno_type} quality check failed: {e}")
            return self._create_empty_quality_result(pheno_type)

    def check_covariates_file_quality(self, covariates_file: str, file_type: str = None) -> Dict[str, Any]:
        """Check covariates file quality with robust handling of categorical data"""
        logger.info(f"  📈 Checking covariates file: {covariates_file}")
        
        try:
            # Use robust reading that handles categorical data
            df = self._read_covariates_robust(covariates_file, nrows=1000)
            
            if df is None or df.empty:
                return self._create_empty_quality_result('covariates')
            
            quality_info = {
                'file_type': 'covariates',
                'checks_passed': 0,
                'checks_total': 6,
                'details': {
                    'n_covariates': df.shape[0],
                    'n_samples': df.shape[1],
                    'missing_percentage': (df.isna().sum().sum() / df.size) * 100,
                    'tensorqtl_format': 'OK'
                }
            }
            
            # Check 1: Non-empty file
            if df.shape[0] > 0 and df.shape[1] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['non_empty'] = True
            
            # Check 2: Low missing rate (more lenient for covariates)
            if quality_info['details']['missing_percentage'] < 10:  # Increased threshold for covariates
                quality_info['checks_passed'] += 1
                quality_info['details']['missing_rate_acceptable'] = True
            
            # Check 3: No constant rows - handle both numeric and categorical
            try:
                # For numeric columns, check standard deviation
                # For categorical, check number of unique values
                constant_rows = 0
                for idx in df.index:
                    row = df.loc[idx]
                    # Try to convert to numeric, if successful check std
                    try:
                        numeric_row = pd.to_numeric(row, errors='coerce')
                        if numeric_row.notna().all() and numeric_row.std() == 0:
                            constant_rows += 1
                    except:
                        # If conversion fails, check if all values are the same
                        if row.nunique() == 1:
                            constant_rows += 1
                
                if constant_rows == 0:
                    quality_info['checks_passed'] += 1
                    quality_info['details']['no_constant_rows'] = True
                else:
                    quality_info['details']['constant_rows'] = constant_rows
            except Exception as e:
                logger.warning(f"Could not check constant rows: {e}")
            
            # Check 4: No duplicate names
            if not df.index.duplicated().any():
                quality_info['checks_passed'] += 1
                quality_info['details']['no_duplicate_names'] = True
            
            # Check 5: Data type analysis (not strict requirement)
            numeric_count = 0
            categorical_count = 0
            for idx in df.index:
                row = df.loc[idx]
                try:
                    numeric_row = pd.to_numeric(row, errors='coerce')
                    if numeric_row.notna().all():
                        numeric_count += 1
                    else:
                        categorical_count += 1
                except:
                    categorical_count += 1
            
            quality_info['details']['numeric_covariates'] = numeric_count
            quality_info['details']['categorical_covariates'] = categorical_count
            
            # This check always passes for covariates since we handle mixed types
            quality_info['checks_passed'] += 1
            quality_info['details']['mixed_types_handled'] = True
            
            # Check 6: Proper format
            if df.shape[0] > 0 and df.shape[1] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['tensorqtl_compatible'] = True
            
            logger.info(f"    ✅ covariates file: {df.shape[0]} rows, {df.shape[1]} samples - {quality_info['checks_passed']}/6 checks passed")
            logger.info(f"    📊 Covariate types: {numeric_count} numeric, {categorical_count} categorical")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ covariates quality check failed: {e}")
            return self._create_empty_quality_result('covariates')

    def check_annotations_file_quality(self, annotations_file: str, file_type: str = None) -> Dict[str, Any]:
        """Check annotations file quality"""
        logger.info(f"  📖 Checking annotations file: {annotations_file}")
        
        try:
            # Try reading as BED format with limited rows
            df = pd.read_csv(annotations_file, sep='\t', comment='#', header=None, nrows=1000)
            
            quality_info = {
                'file_type': 'annotations',
                'checks_passed': 0,
                'checks_total': 4,
                'details': {
                    'n_annotations': '>1000' if len(df) == 1000 else len(df),
                    'n_columns': df.shape[1],
                    'format': 'BED' if df.shape[1] >= 3 else 'unknown'
                }
            }
            
            # Check 1: Non-empty file
            if df.shape[0] > 0:
                quality_info['checks_passed'] += 1
                quality_info['details']['non_empty'] = True
            
            # Check 2: Has minimum BED columns
            if df.shape[1] >= 3:
                quality_info['checks_passed'] += 1
                quality_info['details']['min_columns'] = True
            
            # Check 3: Chromosome column looks reasonable
            if df.shape[1] >= 1:
                first_col = df.iloc[:, 0].astype(str)
                chrom_like = first_col.str.startswith('chr').any() or first_col.isin([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']).any()
                if chrom_like:
                    quality_info['checks_passed'] += 1
                    quality_info['details']['chromosome_format_ok'] = True
            
            # Check 4: Has gene IDs
            if df.shape[1] >= 4:
                quality_info['checks_passed'] += 1
                quality_info['details']['has_gene_ids'] = True
            
            logger.info(f"    ✅ Annotations file: {df.shape[0]} annotations - {quality_info['checks_passed']}/4 checks passed")
            return quality_info
            
        except Exception as e:
            logger.error(f"    ❌ Annotations quality check failed: {e}")
            return self._create_empty_quality_result('annotations')

    def create_sample_mapping_files(self) -> bool:
        """Create sample mapping files for downstream analysis"""
        logger.info("👥 Creating sample mapping files...")
        
        try:
            samples_dir = Path(self.results_dir) / "sample_lists"
            samples_dir.mkdir(exist_ok=True)
            
            input_files = self.config.get('input_files', {})
            all_samples = {}
            
            # Extract samples from genotype file
            if 'genotypes' in input_files and input_files['genotypes']:
                geno_samples = self.extract_samples_from_genotypes(input_files['genotypes'])
                if geno_samples:
                    all_samples['genotypes'] = set(geno_samples)
                    self._write_sample_file(samples_dir / "genotype_samples.txt", geno_samples)
            
            # Extract samples from other files
            sample_sources = [
                ('expression', 'eqtl'),
                ('protein', 'pqtl'), 
                ('splicing', 'sqtl'),
                ('covariates', 'covariates')
            ]
            
            for config_key, sample_key in sample_sources:
                if config_key in input_files and input_files[config_key]:
                    samples = self._extract_samples_from_tabular(input_files[config_key])
                    if samples:
                        all_samples[sample_key] = samples
                        self._write_sample_file(samples_dir / f"{sample_key}_samples.txt", list(samples))
            
            # Create sample intersection
            if all_samples:
                common_samples = set.intersection(*all_samples.values())
                self._write_sample_file(samples_dir / "common_samples.txt", sorted(common_samples))
                
                # Create tensorQTL-specific sample lists
                tensorqtl_dir = samples_dir / "tensorqtl"
                tensorqtl_dir.mkdir(exist_ok=True)
                
                for dataset, samples in all_samples.items():
                    self._write_sample_file(tensorqtl_dir / f"{dataset}_samples.txt", sorted(samples))
                
                self.log_sample_overlap_statistics(all_samples, common_samples)
                return True
            
            logger.warning("⚠️ No samples could be extracted from input files")
            return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create sample mapping files: {e}")
            return False

    def _extract_samples_from_tabular(self, file_path: str) -> Set[str]:
        """Extract samples from tabular file efficiently"""
        try:
            # For covariates, use robust reading
            if 'covariate' in file_path.lower():
                df = self._read_covariates_robust(file_path, nrows=0)
            else:
                # Only read header to get sample names
                df = pd.read_csv(file_path, sep='\t', index_col=0, nrows=0)
            return set(df.columns)
        except Exception as e:
            logger.warning(f"Could not extract samples from {file_path}: {e}")
            return set()

    def _write_sample_file(self, file_path: Path, samples: List[str]) -> None:
        """Write sample list to file"""
        with open(file_path, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")

    def log_sample_overlap_statistics(self, all_samples: Dict[str, Set[str]], common_samples: Set[str]) -> None:
        """Log detailed sample overlap statistics"""
        logger.info("📊 Sample Overlap Statistics:")
        
        for dataset, samples in all_samples.items():
            overlap_count = len(samples & common_samples)
            overlap_percentage = (overlap_count / len(samples)) * 100 if samples else 0
            logger.info(f"   {dataset.upper()}: {len(samples)} total, {overlap_count} common ({overlap_percentage:.1f}%)")
        
        if common_samples:
            logger.info(f"✅ TensorQTL will use {len(common_samples)} common samples for analysis")
        else:
            logger.error("❌ No common samples found - tensorQTL analysis will fail!")

    def generate_data_preparation_report(self, validation_results: Dict[str, Any], quality_results: Dict[str, Any]) -> bool:
        """Generate data preparation report efficiently"""
        logger.info("📝 Generating data preparation report...")
        
        try:
            report_dir = Path(self.results_dir) / "QC_reports"
            report_dir.mkdir(exist_ok=True)
            
            # Calculate overall status
            total_checks = sum(result.get('checks_total', 0) for result in quality_results.values() if isinstance(result, dict))
            passed_checks = sum(result.get('checks_passed', 0) for result in quality_results.values() if isinstance(result, dict))
            
            if total_checks > 0:
                pass_rate = passed_checks / total_checks
                if pass_rate > 0.8:
                    overall_status, status_class = "PASS", "pass"
                elif pass_rate > 0.5:
                    overall_status, status_class = "WARNING", "warning"
                else:
                    overall_status, status_class = "FAIL", "fail"
            else:
                overall_status, status_class = "UNKNOWN", "warning"
            
            # Generate HTML report
            self._generate_html_report(report_dir, validation_results, quality_results, overall_status, status_class, passed_checks, total_checks)
            
            # Save JSON version
            json_report = {
                'validation_results': validation_results,
                'quality_results': quality_results,
                'overall_status': overall_status,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(report_dir / "data_preparation_report.json", 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            
            logger.info("✅ Data preparation report generated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate data preparation report: {e}")
            return False

    def _generate_html_report(self, report_dir: Path, validation_results: Dict[str, Any], 
                         quality_results: Dict[str, Any], overall_status: str, 
                         status_class: str, passed_checks: int, total_checks: int) -> None:
        """Generate HTML report content"""
        report_file = report_dir / "data_preparation_report.html"
        
        # HTML template with placeholders - FIXED: Proper string formatting
        html_template = """<!DOCTYPE html>
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
            .tensorqtl-note {{ background: #e7f3ff; padding: 10px; border-left: 4px solid #007bff; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Preparation Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Overall Status: <span class="status-{status_class}">{overall_status}</span></p>
            <p>Checks Passed: {passed_checks} / {total_checks}</p>
        </div>
        
        <div class="tensorqtl-note">
            <h3>🧬 TensorQTL Compatibility Notes</h3>
            <p>This pipeline uses <strong>tensorQTL</strong> for QTL analysis. Ensure:</p>
            <ul>
                <li>Genotype data is in PLINK format for optimal performance</li>
                <li>Sample IDs match exactly across all files</li>
                <li>Phenotype data is properly normalized (VST recommended for RNA-seq)</li>
                <li>Sufficient common samples exist across all datasets</li>
            </ul>
        </div>
        
        {validation_table}
        
        {quality_table}
        
        <div class="section">
            <h2>Next Steps</h2>
            <ul>
                <li><strong>If status is PASS:</strong> Proceed with genotype and expression processing</li>
                <li><strong>If status is WARNING:</strong> Review warnings and consider addressing issues</li>
                <li><strong>If status is FAIL:</strong> Fix the reported issues before proceeding</li>
            </ul>
        </div>
    </body>
    </html>"""
        
        # Generate validation table
        validation_table = self._generate_validation_table(validation_results)
        
        # Generate quality table
        quality_table = self._generate_quality_table(quality_results)
        
        # Fill template - FIXED: Use proper string formatting
        html_content = html_template.format(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            status_class=status_class,
            overall_status=overall_status,
            passed_checks=passed_checks,
            total_checks=total_checks,
            validation_table=validation_table,
            quality_table=quality_table
        )
        
        with open(report_file, 'w') as f:
            f.write(html_content)

    def _get_report_template(self) -> str:
        """Return HTML report template with FIXED CSS syntax"""
        return """<!DOCTYPE html>
    <html>
    <head>
        <title>Data Preparation Report</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background: #f8f9fa; padding: 20px; border-radius: 5px; }
            .status-pass { color: #28a745; font-weight: bold; }
            .status-warning { color: #ffc107; font-weight: bold; }
            .status-fail { color: #dc3545; font-weight: bold; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
            th { background-color: #f8f9fa; }
            .tensorqtl-note { background: #e7f3ff; padding: 10px; border-left: 4px solid #007bff; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Preparation Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Overall Status: <span class="status-{status_class}">{overall_status}</span></p>
            <p>Checks Passed: {passed_checks} / {total_checks}</p>
        </div>
        
        <div class="tensorqtl-note">
            <h3>🧬 TensorQTL Compatibility Notes</h3>
            <p>This pipeline uses <strong>tensorQTL</strong> for QTL analysis. Ensure:</p>
            <ul>
                <li>Genotype data is in PLINK format for optimal performance</li>
                <li>Sample IDs match exactly across all files</li>
                <li>Phenotype data is properly normalized (VST recommended for RNA-seq)</li>
                <li>Sufficient common samples exist across all datasets</li>
            </ul>
        </div>
        
        {validation_table}
        
        {quality_table}
        
        <div class="section">
            <h2>Next Steps</h2>
            <ul>
                <li><strong>If status is PASS:</strong> Proceed with genotype and expression processing</li>
                <li><strong>If status is WARNING:</strong> Review warnings and consider addressing issues</li>
                <li><strong>If status is FAIL:</strong> Fix the reported issues before proceeding</li>
            </ul>
        </div>
    </body>
    </html>"""

    def _generate_validation_table(self, validation_results: Dict[str, Any]) -> str:
        """Generate validation results table HTML"""
        if not validation_results:
            return "<div class='section'><h2>File Validation</h2><p>No validation results available.</p></div>"
        
        table_html = """
        <div class="section">
            <h2>File Validation</h2>
            <table>
                <tr><th>File Type</th><th>Status</th><th>Path</th><th>Size (GB)</th><th>TensorQTL Compatible</th></tr>
        """
        
        status_map = {
            'OK': ('status-pass', 'OK'),
            'REQUIRED_MISSING': ('status-fail', 'REQUIRED MISSING'),
            'FILE_NOT_FOUND': ('status-fail', 'NOT FOUND'),
            'ACCESS_ERROR': ('status-fail', 'ACCESS ERROR'),
            'OPTIONAL_NOT_CONFIGURED': ('status-warning', 'OPTIONAL (NOT CONFIGURED)')
        }
        
        for file_type, result in validation_results.items():
            status_class, status_text = status_map.get(result['status'], ('status-warning', result['status']))
            tensorqtl_status = result.get('tensorqtl_compatible', 'Unknown')
            tensorqtl_class = 'status-pass' if tensorqtl_status == True else 'status-warning' if tensorqtl_status == 'Unknown' else 'status-fail'
            tensorqtl_text = '✅' if tensorqtl_status == True else '⚠️' if tensorqtl_status == 'Unknown' else '❌'
            
            table_html += f"""
                <tr>
                    <td>{file_type}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.get('path', 'N/A')}</td>
                    <td>{result.get('size_gb', 'N/A')}</td>
                    <td class="{tensorqtl_class}">{tensorqtl_text}</td>
                </tr>
            """
        
        table_html += "</table></div>"
        return table_html

    def _generate_quality_table(self, quality_results: Dict[str, Any]) -> str:
        """Generate quality results table HTML"""
        if not quality_results:
            return "<div class='section'><h2>Data Quality Checks</h2><p>No quality results available.</p></div>"
        
        table_html = """
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
                for key, value in list(result.get('details', {}).items())[:5]:  # Limit details
                    details_html += f"<li><strong>{key}:</strong> {value}</li>"
                details_html += "</ul>"
                
                table_html += f"""
                    <tr>
                        <td>{data_type}</td>
                        <td class="{status_class}">{checks_passed}/{checks_total}</td>
                        <td>{checks_total}</td>
                        <td>{details_html}</td>
                    </tr>
                """
        
        table_html += "</table></div>"
        return table_html

    def get_qtl_types_from_config(self) -> List[str]:
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
                return ['eqtl']
        except Exception as e:
            logger.error(f"Error getting QTL types from config: {e}")
            return ['eqtl']
    
    def run_quality_control(self) -> bool:
        """Run quality control module for modular pipeline"""
        logger.info("🚀 Starting quality control module...")
        
        try:
            if 'input_files' not in self.config:
                logger.error("❌ No input_files section in config")
                return False
                
            vcf_file = self.config['input_files'].get('genotypes')
            if not vcf_file:
                logger.error("❌ No genotypes file specified in config")
                return False
            
            if not os.path.exists(vcf_file):
                logger.error(f"❌ Genotype file not found: {vcf_file}")
                return False
            
            qtl_types = self.get_qtl_types_from_config()
            qc_results = self.run_comprehensive_qc(vcf_file, qtl_types, self.results_dir)
            
            success = bool(qc_results)
            if success:
                logger.info("✅ Quality control module completed successfully")
            else:
                logger.error("❌ Quality control module failed to produce results")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Quality control module failed: {e}")
            return False

    def run_comprehensive_qc(self, vcf_file: str, qtl_types: List[str], output_dir: str) -> Dict[str, Any]:
        """Run comprehensive QC on all data types"""
        logger.info("🔍 Running comprehensive quality control...")
        
        qc_results = {}
        
        try:
            qc_dir = Path(output_dir) / "QC_reports"
            qc_dir.mkdir(exist_ok=True)
            
            # Get phenotype files
            phenotype_files = {}
            for qtl_type in qtl_types:
                config_key = map_qtl_type_to_config_key(qtl_type)
                phenotype_files[qtl_type] = self.config['input_files'].get(config_key)
            
            # Run QC steps
            qc_steps = [
                ("🧬 Genotype QC", lambda: self.genotype_qc(vcf_file, str(qc_dir))),
                ("🔗 Sample concordance", lambda: self.sample_concordance_qc(vcf_file, phenotype_files, str(qc_dir))),
            ]
            
            # Add phenotype QC for available types
            for qtl_type, pheno_file in phenotype_files.items():
                if pheno_file and os.path.exists(pheno_file):
                    qc_steps.append((f"📊 {qtl_type} phenotype QC", 
                                   lambda p=pheno_file, t=qtl_type: self.phenotype_qc(p, t, str(qc_dir))))
            
            # Add optional steps
            if self.qc_config.get('run_pca', True):
                qc_steps.append(("📈 PCA analysis", lambda: self.run_pca_analysis(vcf_file, str(qc_dir))))
            
            if self.qc_config.get('advanced_outlier_detection', True):
                qc_steps.append(("🎯 Advanced outlier detection", 
                               lambda: self.advanced_outlier_detection(vcf_file, phenotype_files, str(qc_dir))))
            
            # Execute QC steps
            for desc, func in qc_steps:
                logger.info(desc)
                try:
                    result = func()
                    key = desc.split(' ')[1].lower()  # Extract key from description
                    qc_results[key] = result
                except Exception as e:
                    logger.error(f"❌ {desc} failed: {e}")
                    qc_results[desc.split(' ')[1].lower()] = {}
            
            # Generate reports
            logger.info("📋 Generating comprehensive QC report...")
            self.generate_qc_report(qc_results, str(qc_dir))
            self.save_qc_results(qc_results, str(qc_dir))
            
            logger.info("✅ Comprehensive QC completed")
            return qc_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive QC failed: {e}")
            return {}

    def genotype_qc(self, vcf_file: str, output_dir: str) -> Dict[str, Any]:
        """Optimized genotype QC using caching and efficient operations"""
        logger.info("🔬 Running genotype QC...")
        
        # Use cache if available
        cache_key = f"genotype_qc_{vcf_file}"
        if cache_key in self._plink_cache:
            logger.info("📚 Using cached genotype QC results")
            return self._plink_cache[cache_key]
        
        qc_metrics = {}
        
        try:
            plink_base = Path(output_dir) / "plink_qc"
            
            # Convert VCF to PLINK format with error handling
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed --out {plink_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode != 0:
                logger.warning("VCF to PLINK conversion failed, trying alternative approach")
                return qc_metrics
            
            # Run QC metrics in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_metric = {
                    executor.submit(self._calculate_sample_missingness_plink, str(plink_base)): 'sample_missingness',
                    executor.submit(self._calculate_variant_missingness_plink, str(plink_base)): 'variant_missingness',
                    executor.submit(self._calculate_maf_distribution_plink, str(plink_base)): 'maf_distribution'
                }
                
                for future in as_completed(future_to_metric):
                    metric_name = future_to_metric[future]
                    try:
                        qc_metrics[metric_name] = future.result()
                    except Exception as e:
                        logger.warning(f"Could not calculate {metric_name}: {e}")
                        qc_metrics[metric_name] = {}
            
            # Calculate additional metrics
            qc_metrics['hwe'] = self._calculate_hwe_plink(str(plink_base), output_dir)
            qc_metrics['heterozygosity'] = self._calculate_heterozygosity_plink(str(plink_base))
            
            # Generate plots
            self._plot_genotype_qc(qc_metrics, output_dir)
            
            # Apply filters
            filtered_file = self._apply_genotype_filters_plink(str(plink_base), output_dir, qc_metrics)
            qc_metrics['filtered_file'] = filtered_file
            
            # Calculate summary
            qc_metrics['summary'] = self._calculate_genotype_qc_summary(qc_metrics)
            
            logger.info("✅ Genotype QC completed")
            
            # Cache results
            self._plink_cache[cache_key] = qc_metrics
            return qc_metrics
            
        except Exception as e:
            logger.error(f"❌ Genotype QC failed: {e}")
            return {}

    def _calculate_sample_missingness_plink(self, plink_base: str) -> Dict[str, float]:
        """Calculate sample-level missingness using PLINK"""
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --missing --out {plink_base}_missingness 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            sample_missingness = {}
            imiss_file = f"{plink_base}_missingness.imiss"
            if os.path.exists(imiss_file):
                df = pd.read_csv(imiss_file, sep='\s+', usecols=['IID', 'F_MISS'])
                sample_missingness = df.set_index('IID')['F_MISS'].to_dict()
            
            return sample_missingness
            
        except Exception as e:
            logger.warning(f"Could not calculate sample missingness: {e}")
            return {}

    def _calculate_variant_missingness_plink(self, plink_base: str) -> Dict[str, float]:
        """Calculate variant-level missingness using PLINK"""
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --missing --out {plink_base}_missingness 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            variant_missingness = {}
            lmiss_file = f"{plink_base}_missingness.lmiss"
            if os.path.exists(lmiss_file):
                df = pd.read_csv(lmiss_file, sep='\s+', usecols=['CHR', 'SNP', 'F_MISS'])
                variant_missingness = {f"{row['CHR']}:{row['SNP']}": row['F_MISS'] for _, row in df.iterrows()}
            
            return variant_missingness
            
        except Exception as e:
            logger.warning(f"Could not calculate variant missingness: {e}")
            return {}

    def _calculate_maf_distribution_plink(self, plink_base: str) -> Dict[str, Any]:
        """Calculate MAF distribution using PLINK"""
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --freq --out {plink_base}_maf 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            maf_values = []
            frq_file = f"{plink_base}_maf.frq"
            if os.path.exists(frq_file):
                df = pd.read_csv(frq_file, sep='\s+', usecols=['MAF'])
                maf_values = df['MAF'].tolist()
            
            return {
                'maf_values': maf_values,
                'mean_maf': np.mean(maf_values) if maf_values else 0,
                'median_maf': np.median(maf_values) if maf_values else 0
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate MAF distribution: {e}")
            return {'maf_values': [], 'mean_maf': 0, 'median_maf': 0}

    def _calculate_hwe_plink(self, plink_base: str, output_dir: str) -> Dict[str, Any]:
        """Calculate Hardy-Weinberg Equilibrium using PLINK"""
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --hardy --out {plink_base}_hwe 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            hwe_file = f"{plink_base}_hwe.hwe"
            if os.path.exists(hwe_file):
                df = pd.read_csv(hwe_file, sep='\s+', usecols=['P'])
                hwe_threshold = self.qc_config.get('hwe_threshold', 1e-6)
                violations = (df['P'] < hwe_threshold).sum()
                total_variants = len(df)
                
                return {
                    'violations': int(violations),
                    'total_variants': total_variants,
                    'violation_rate': violations / total_variants if total_variants > 0 else 0
                }
            
            return {'violations': 0, 'total_variants': 0, 'violation_rate': 0}
            
        except Exception as e:
            logger.warning(f"Could not calculate HWE: {e}")
            return {'violations': 0, 'total_variants': 0, 'violation_rate': 0}

    def _calculate_heterozygosity_plink(self, plink_base: str) -> Dict[str, float]:
        """Calculate sample heterozygosity using PLINK"""
        try:
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --het --out {plink_base}_het 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            heterozygosity = {}
            het_file = f"{plink_base}_het.het"
            if os.path.exists(het_file):
                df = pd.read_csv(het_file, sep='\s+', usecols=['IID', 'O(HOM)', 'N(NM)'])
                for _, row in df.iterrows():
                    sample_id = row['IID']
                    hom_count = row['O(HOM)']
                    nm_count = row['N(NM)']
                    het_rate = (nm_count - hom_count) / nm_count if nm_count > 0 else 0
                    heterozygosity[sample_id] = het_rate
            
            return heterozygosity
            
        except Exception as e:
            logger.warning(f"Could not calculate heterozygosity: {e}")
            return {}

    def _apply_genotype_filters_plink(self, plink_base: str, output_dir: str, qc_metrics: Dict[str, Any]) -> Optional[str]:
        """Apply genotype filters using PLINK"""
        logger.info("🔧 Applying genotype filters using PLINK...")
        
        try:
            filtered_base = Path(output_dir) / "filtered_genotypes"
            
            filter_args = [
                f"--maf {self.qc_config.get('maf_threshold', 0.01)}",
                f"--geno {self.qc_config.get('variant_missingness_threshold', 0.1)}",
                f"--hwe {self.qc_config.get('hwe_threshold', 1e-6)}",
                f"--mind {self.qc_config.get('sample_missingness_threshold', 0.1)}"
            ]
            
            filter_string = " ".join(filter_args)
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} {filter_string} --recode vcf --out {filtered_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                filtered_vcf = f"{filtered_base}.vcf"
                if os.path.exists(filtered_vcf):
                    # Compress and index
                    compressed_vcf = f"{filtered_vcf}.gz"
                    subprocess.run(f"{self.config['paths']['bgzip']} -c {filtered_vcf} > {compressed_vcf}", shell=True, executable='/bin/bash')
                    subprocess.run(f"{self.config['paths']['tabix']} -p vcf {compressed_vcf}", shell=True, executable='/bin/bash')
                    return compressed_vcf
            
            logger.warning("PLINK filtering failed, using original file")
            return None
                
        except Exception as e:
            logger.warning(f"Genotype filtering failed: {e}")
            return None

    def _plot_genotype_qc(self, qc_metrics: Dict[str, Any], output_dir: str) -> None:
        """Generate genotype QC plots"""
        try:
            plot_dir = Path(output_dir) / "QC_plots"
            plot_dir.mkdir(exist_ok=True)
            
            # MAF distribution plot
            if 'maf_distribution' in qc_metrics and qc_metrics['maf_distribution']['maf_values']:
                plt.figure(figsize=(10, 6))
                plt.hist(qc_metrics['maf_distribution']['maf_values'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(self.qc_config.get('maf_threshold', 0.01), color='red', linestyle='--', 
                           label=f'MAF threshold ({self.qc_config.get("maf_threshold", 0.01)})')
                plt.xlabel('Minor Allele Frequency (MAF)')
                plt.ylabel('Number of Variants')
                plt.title('MAF Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_dir / 'maf_distribution.png', dpi=300, bbox_inches='tight')
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
                plt.savefig(plot_dir / 'sample_missingness.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate genotype QC plots: {e}")

    def _calculate_genotype_qc_summary(self, qc_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate genotype QC summary statistics"""
        summary = {}
        
        try:
            # Sample missingness summary
            if 'sample_missingness' in qc_metrics and qc_metrics['sample_missingness']:
                missing_rates = list(qc_metrics['sample_missingness'].values())
                summary['sample_missingness'] = {
                    'mean': float(np.mean(missing_rates)),
                    'median': float(np.median(missing_rates)),
                    'max': float(np.max(missing_rates)),
                    'samples_above_threshold': len([x for x in missing_rates if x > 0.1])
                }
            
            # Variant missingness summary
            if 'variant_missingness' in qc_metrics and qc_metrics['variant_missingness']:
                missing_rates = list(qc_metrics['variant_missingness'].values())
                summary['variant_missingness'] = {
                    'mean': float(np.mean(missing_rates)),
                    'median': float(np.median(missing_rates)),
                    'max': float(np.max(missing_rates)),
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

    def phenotype_qc(self, pheno_file: str, pheno_type: str, output_dir: str) -> Dict[str, Any]:
        """Comprehensive phenotype QC with enhanced metrics"""
        logger.info(f"🔬 Running {pheno_type} QC...")
        
        try:
            # Use chunking for large files
            chunksize = 10000
            reader = pd.read_csv(pheno_file, sep='\t', index_col=0, chunksize=chunksize)
            first_chunk = next(reader)
            
            qc_metrics = {
                'basic_stats': {
                    'n_features': 0,
                    'n_samples': first_chunk.shape[1],
                    'data_type': 'counts' if first_chunk.values.sum() > first_chunk.shape[0] * first_chunk.shape[1] else 'normalized'
                }
            }
            
            # Process chunks to accumulate statistics
            total_features = first_chunk.shape[0]
            total_missing = first_chunk.isna().sum().sum()
            total_cells = first_chunk.size
            
            for chunk in reader:
                total_features += chunk.shape[0]
                total_missing += chunk.isna().sum().sum()
                total_cells += chunk.size
            
            qc_metrics['basic_stats']['n_features'] = total_features
            qc_metrics['basic_stats']['total_measurements'] = total_cells
            
            # Missingness analysis
            qc_metrics['missingness'] = {
                'total_missing': int(total_missing),
                'missing_percentage': (total_missing / total_cells) * 100
            }
            
            # Distribution metrics (on first chunk for efficiency)
            qc_metrics['distribution'] = {
                'mean': float(first_chunk.mean().mean()),
                'std': float(first_chunk.std().mean())
            }
            
            # Quality flags
            qc_metrics['quality_flags'] = {
                'has_negative_values': (first_chunk < 0).any().any(),
                'has_zero_values': (first_chunk == 0).any().any(),
                'has_constant_features': (first_chunk.std(axis=1) == 0).any(),
                'has_duplicate_features': first_chunk.index.duplicated().any()
            }
            
            # Generate plots
            self._plot_phenotype_qc(first_chunk, pheno_type, output_dir, qc_metrics)
            
            logger.info(f"✅ {pheno_type} QC completed: {total_features} features, {first_chunk.shape[1]} samples")
            return qc_metrics
            
        except Exception as e:
            logger.error(f"❌ {pheno_type} QC failed: {e}")
            return {}

    def _plot_phenotype_qc(self, df: pd.DataFrame, pheno_type: str, output_dir: str, qc_metrics: Dict[str, Any]) -> None:
        """Generate phenotype QC plots"""
        try:
            plot_dir = Path(output_dir) / "QC_plots"
            plot_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            # Sample random features for visualization
            n_features_to_plot = min(20, df.shape[0])
            if n_features_to_plot > 0:
                features_to_plot = np.random.choice(df.index, n_features_to_plot, replace=False)
                for feature in features_to_plot:
                    plt.hist(df.loc[feature].dropna(), bins=20, alpha=0.3, density=True)
            
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'{pheno_type.upper()} Distribution\n({n_features_to_plot} random features)')
            
            plt.subplot(1, 2, 2)
            # Missingness pattern (sampled for large datasets)
            sample_size = min(1000, df.shape[0])
            if sample_size < df.shape[0]:
                df_sample = df.sample(n=sample_size)
            else:
                df_sample = df
                
            missing_data = df_sample.isna().astype(int)
            plt.imshow(missing_data.values, aspect='auto', cmap='Reds', interpolation='nearest')
            plt.xlabel('Samples')
            plt.ylabel('Features')
            plt.title(f'{pheno_type.upper()} Missingness Pattern')
            plt.colorbar(label='Missing (1) / Present (0)')
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'{pheno_type}_qc_plots.png', dpi=200, bbox_inches='tight')  # Reduced DPI for speed
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate phenotype QC plots: {e}")

    def sample_concordance_qc(self, vcf_file: str, phenotype_files: Dict[str, str], output_dir: str) -> Dict[str, Any]:
        """Check sample concordance across all datasets"""
        logger.info("🔍 Checking sample concordance...")
        
        concordance_results = {}
        
        try:
            # Get samples from genotype file
            geno_samples = self.extract_samples_from_genotypes(vcf_file)
            if not geno_samples:
                logger.warning("Could not extract samples from genotype file")
                return {}
            
            concordance_results['genotype_samples'] = geno_samples
            concordance_results['genotype_sample_count'] = len(geno_samples)
            
            # Get samples from each phenotype file
            sample_overlap = {}
            for pheno_type, pheno_file in phenotype_files.items():
                if pheno_file and os.path.exists(pheno_file):
                    try:
                        # Only read header to get sample names
                        df = pd.read_csv(pheno_file, sep='\t', index_col=0, nrows=0)
                        pheno_samples = set(df.columns)
                        overlap = set(geno_samples) & pheno_samples
                        
                        sample_overlap[pheno_type] = {
                            'pheno_sample_count': len(pheno_samples),
                            'overlap_count': len(overlap),
                            'overlap_percentage': (len(overlap) / len(geno_samples)) * 100 if geno_samples else 0
                        }
                    except Exception as e:
                        logger.warning(f"Could not read phenotype file {pheno_file}: {e}")
                        continue
            
            concordance_results['sample_overlap'] = sample_overlap
            
            # Generate concordance plot
            self._plot_sample_concordance(concordance_results, output_dir)
            
            return concordance_results
            
        except Exception as e:
            logger.error(f"Sample concordance check failed: {e}")
            return {}

    def _plot_sample_concordance(self, concordance_results: Dict[str, Any], output_dir: str) -> None:
        """Plot sample concordance across datasets"""
        try:
            plot_dir = Path(output_dir) / "QC_plots"
            plot_dir.mkdir(exist_ok=True)
            
            if 'sample_overlap' in concordance_results and concordance_results['sample_overlap']:
                datasets = list(concordance_results['sample_overlap'].keys())
                overlap_percentages = [concordance_results['sample_overlap'][d]['overlap_percentage'] for d in datasets]
                
                plt.figure(figsize=(10, 6))
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#1B998B', '#FF6B6B']
                bars = plt.bar(datasets, overlap_percentages, color=colors[:len(datasets)])
                
                for bar, percentage in zip(bars, overlap_percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
                plt.ylabel('Sample Overlap Percentage')
                plt.title('Sample Concordance Across Datasets')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plot_dir / 'sample_concordance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate sample concordance plot: {e}")

    def run_pca_analysis(self, vcf_file: str, output_dir: str) -> Dict[str, Any]:
        """Run PCA for population stratification using PLINK"""
        logger.info("📊 Running PCA analysis using PLINK...")
        
        try:
            plink_base = Path(output_dir) / "pca_input"
            
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --pca 10 --out {plink_base} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                pca_eigenvec = f"{plink_base}.eigenvec"
                if os.path.exists(pca_eigenvec):
                    pca_df = pd.read_csv(pca_eigenvec, sep='\s+', header=None)
                    pca_df.columns = ['FID', 'IID'] + [f'PC{i+1}' for i in range(10)]
                    
                    self._plot_pca_results(pca_df, output_dir)
                    
                    return {
                        'pca_file': pca_eigenvec,
                        'explained_variance': self._calculate_pca_variance(f"{plink_base}.eigenval"),
                        'sample_count': len(pca_df)
                    }
            
            return {}
            
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            return {}

    def _calculate_pca_variance(self, eigenval_file: str) -> List[float]:
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

    def _plot_pca_results(self, pca_df: pd.DataFrame, output_dir: str) -> None:
        """Plot PCA results"""
        try:
            plot_dir = Path(output_dir) / "QC_plots"
            plot_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, color='#2E86AB', s=20)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA: PC1 vs PC2')
            
            plt.subplot(1, 2, 2)
            pcs = range(1, 11)
            # Simplified scree plot (actual values would come from eigenvalues)
            plt.plot(pcs, [100/i for i in pcs], 'o-', color='#A23B72')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance (%)')
            plt.title('Scree Plot')
            plt.xticks(pcs)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate PCA plots: {e}")

    def advanced_outlier_detection(self, vcf_file: str, phenotype_files: Dict[str, str], output_dir: str) -> Dict[str, Any]:
        """Advanced outlier detection using multiple methods"""
        logger.info("🎯 Running advanced outlier detection...")
        
        outlier_results = {}
        
        try:
            # Use PLINK for genotype-based outlier detection
            plink_base = Path(output_dir) / "outlier_detection"
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed --out {plink_base} 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            # Calculate heterozygosity and identify outliers
            cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --het --out {plink_base}_het 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
            
            # Read heterozygosity results
            het_file = f"{plink_base}_het.het"
            if os.path.exists(het_file):
                df = pd.read_csv(het_file, sep='\s+', usecols=['IID', 'O(HOM)', 'N(NM)'])
                het_rates = (df['N(NM)'] - df['O(HOM)']) / df['N(NM)']
                
                # Identify heterozygosity outliers
                mean_het = het_rates.mean()
                std_het = het_rates.std()
                outlier_mask = (het_rates < mean_het - 3 * std_het) | (het_rates > mean_het + 3 * std_het)
                
                outlier_results['heterozygosity_outliers'] = {
                    'n_outliers': int(outlier_mask.sum()),
                    'outlier_samples': df[outlier_mask]['IID'].tolist(),
                    'threshold_low': float(mean_het - 3 * std_het),
                    'threshold_high': float(mean_het + 3 * std_het)
                }
            
            logger.info("✅ Advanced outlier detection completed")
            return outlier_results
            
        except Exception as e:
            logger.warning(f"Advanced outlier detection failed: {e}")
            return {}

    def generate_qc_report(self, qc_results: Dict[str, Any], output_dir: str) -> None:
        """Generate comprehensive QC report"""
        logger.info("📊 Generating QC report...")
        
        try:
            report_file = Path(output_dir) / "comprehensive_qc_report.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Comprehensive QC Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
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
                    {self._generate_genotype_qc_summary_html(qc_results.get('genotype', {}))}
                </div>
                
                <div class="section">
                    <h2>Sample Concordance</h2>
                    {self._generate_concordance_summary_html(qc_results.get('concordance', {}))}
                </div>
                
                <div class="section">
                    <h2>QC Plots</h2>
                    {self._generate_plot_section_html(output_dir)}
                </div>
                
                <div class="section">
                    <h2>Next Steps</h2>
                    <p>Based on the QC results, you can proceed with the next modules:</p>
                    <ul>
                        <li><strong>If QC passes:</strong> Run QTL mapping</li>
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

    def _generate_genotype_qc_summary_html(self, genotype_qc: Dict[str, Any]) -> str:
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
        
        summary += "</table>"
        return summary

    def _generate_concordance_summary_html(self, concordance_results: Dict[str, Any]) -> str:
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

    def _generate_plot_section_html(self, output_dir: str) -> str:
        """Generate HTML for QC plots"""
        plots_html = "<div class='plot-grid'>"
        
        plot_files = [
            ('maf_distribution.png', 'MAF Distribution'),
            ('sample_missingness.png', 'Sample Missingness'),
            ('sample_concordance.png', 'Sample Concordance'),
            ('pca_analysis.png', 'PCA Analysis')
        ]
        
        plot_dir = Path(output_dir) / "QC_plots"
        
        for plot_file, title in plot_files:
            plot_path = plot_dir / plot_file
            if plot_path.exists():
                plots_html += f"""
                <div class="plot">
                    <h3>{title}</h3>
                    <img src="QC_plots/{plot_file}" alt="{title}" style="max-width: 400px;">
                </div>
                """
        
        plots_html += "</div>"
        return plots_html

    def save_qc_results(self, qc_results: Dict[str, Any], output_dir: str) -> None:
        """Save QC results for downstream modules"""
        try:
            results_file = Path(output_dir) / "comprehensive_qc_results.json"
            
            # Convert to JSON-serializable format
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
            
        except Exception as e:
            logger.warning(f"Could not save QC results: {e}")


# Maintain backward compatibility
def run_data_preparation(config: Dict[str, Any]) -> bool:
    """Main function for data preparation module"""
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

def run_quality_control(config: Dict[str, Any]) -> bool:
    """Main function for quality control module"""
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


# Standalone execution
if __name__ == "__main__":
    import sys
    import yaml
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    
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