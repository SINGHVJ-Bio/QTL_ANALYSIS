#!/usr/bin/env python3
"""
Optimized QTL analysis pipeline with tensorQTL v1.0.10 compatibility
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import os
import pandas as pd
import numpy as np
import logging
import subprocess
import warnings
import psutil
import sys
from pathlib import Path

# Configure logging
logger = logging.getLogger('QTLPipeline')
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from scripts.utils.batch_correction import run_batch_correction_pipeline
    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    logger.warning("Batch correction module not available")

try:
    from scripts.utils.deseq2_vst_python import deseq2_vst_python, simple_vst_fallback
    DESEQ2_VST_AVAILABLE = True
except ImportError:
    DESEQ2_VST_AVAILABLE = False
    logger.warning("DESeq2 VST module not available")

try:
    from normalization_comparison import NormalizationComparison
except ImportError:
    try:
        from scripts.utils.normalization_comparison import NormalizationComparison
    except ImportError:
        NormalizationComparison = None
        logger.warning("NormalizationComparison module not available")

# TensorQTL imports
TENSORQTL_AVAILABLE = False
TENSORQTL_IMPORT_ERROR = None
try:
    import torch
    import tensorqtl
    from tensorqtl import genotypeio, cis, trans, calculate_qvalues
    TENSORQTL_AVAILABLE = True
    logger.info(f"TensorQTL v1.0.10 successfully imported")
except ImportError as e:
    TENSORQTL_AVAILABLE = False
    TENSORQTL_IMPORT_ERROR = str(e)
    logger.error(f"TensorQTL import failed: {e}")

class PLINKVersionManager:
    """Manage PLINK version detection and command generation"""
    
    def __init__(self, config):
        self.config = config
        self.plink_paths = self._detect_plink_versions()
        self.preferred_version = self._select_preferred_version()
        
    def _detect_plink_versions(self):
        """Detect available PLINK versions"""
        plink_versions = {}
        versions_to_check = [
            ('plink2', self.config['paths'].get('plink2', 'plink2')),
            ('plink', self.config['paths'].get('plink', 'plink')),
            ('plink1.9', self.config['paths'].get('plink1.9', 'plink1.9'))
        ]
        
        for name, path in versions_to_check:
            version = self._get_plink_version(path, name)
            if version:
                plink_type = self._determine_plink_type(version) if name == 'plink' else name
                plink_versions[name] = {'path': path, 'version': version, 'type': plink_type}
        
        logger.info(f"Detected PLINK versions: {list(plink_versions.keys())}")
        return plink_versions
    
    def _get_plink_version(self, plink_path, plink_name):
        """Get PLINK version by running version command"""
        try:
            for cmd in [f"{plink_path} --version", f"{plink_path} -version"]:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    version_output = result.stdout.strip()
                    logger.info(f"{plink_name} version: {version_output.splitlines()[0] if version_output else 'Unknown'}")
                    return version_output
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    def _determine_plink_type(self, version_output):
        """Determine if plink is version 1.9 or 2.0"""
        version_str = version_output.lower()
        if 'v2.' in version_str or 'plink 2' in version_str:
            return 'plink2'
        return 'plink1.9'
    
    def _select_preferred_version(self):
        """Select preferred PLINK version"""
        preferred_order = ['plink2', 'plink1.9', 'plink']
        for version_name in preferred_order:
            if version_name in self.plink_paths:
                logger.info(f"Using {version_name} as preferred version")
                return self.plink_paths[version_name]
        logger.error("No suitable PLINK version found")
        return None
    
    def get_vcf_filtering_command(self, vcf_file, output_prefix, plink_threads=1):
        """
        Generate VCF filtering command based on PLINK version
        
        PLINK2: Uses --export vcf to output filtered VCF
        PLINK1.9: Uses --recode vcf to output filtered VCF
        """
        if not self.preferred_version:
            raise RuntimeError("No PLINK version available")
        
        plink_type = self.preferred_version['type']
        plink_path = self.preferred_version['path']
        
        # Get filtering parameters
        genotype_config = self.config.get('genotype_processing', {})
        maf_threshold = genotype_config.get('maf_threshold', 0.01)
        call_rate_threshold = genotype_config.get('call_rate_threshold', 0.95)
        hwe_threshold = genotype_config.get('hwe_threshold', 1e-6)
        
        common_args = f"--vcf {vcf_file} --out {output_prefix} --threads {plink_threads}"
        filtering_args = f"--maf {maf_threshold} --geno {1-call_rate_threshold} --hwe {hwe_threshold} midp"
        
        if plink_type == 'plink2':
            # PLINK2: --export vcf creates filtered VCF file
            cmd = f"{plink_path} --output-chr chrM {common_args} {filtering_args} --export vcf"
            output_file = f"{output_prefix}.vcf"
            logger.info(f"Using PLINK2 for VCF filtering (--export vcf)")
        else:  # plink1.9
            # PLINK1.9: --recode vcf creates filtered VCF file
            cmd = f"{plink_path} {common_args} {filtering_args} --recode vcf"
            output_file = f"{output_prefix}.vcf"
            logger.info(f"Using PLINK1.9 for VCF filtering (--recode vcf)")
        
        return cmd, output_file, plink_type
    
    def get_vcf_conversion_command(self, vcf_file, output_prefix, plink_threads=1):
        """
        Generate VCF to PLINK conversion command based on version
        
        PLINK2: Creates .pgen, .pvar, .psam files (binary format)
        PLINK1.9: Creates .bed, .bim, .fam files (binary format)
        """
        if not self.preferred_version:
            raise RuntimeError("No PLINK version available")
        
        plink_type = self.preferred_version['type']
        plink_path = self.preferred_version['path']
        common_args = f"--vcf {vcf_file} --out {output_prefix} --threads {plink_threads}"
        
        if plink_type == 'plink2':
            # PLINK2: Creates pgen/pvar/psam format
            cmd = f"{plink_path} --output-chr chrM {common_args}"
            output_files = {
                'genotype': f"{output_prefix}.pgen",
                'variant': f"{output_prefix}.pvar", 
                'sample': f"{output_prefix}.psam"
            }
            logger.info("Using PLINK2 for VCF conversion (pgen/pvar/psam format)")
        else:  # plink1.9
            # PLINK1.9: Creates bed/bim/fam format with --keep-allele-order
            cmd = f"{plink_path} --make-bed --keep-allele-order {common_args}"
            output_files = {
                'genotype': f"{output_prefix}.bed",
                'variant': f"{output_prefix}.bim",
                'sample': f"{output_prefix}.fam"
            }
            logger.info("Using PLINK1.9 for VCF conversion (bed/bim/fam format)")
        
        return cmd, output_files, plink_type

class FileFormatValidator:
    """Comprehensive file format validation for tensorQTL compatibility"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = config.get('results_dir', 'results')
        self.plink_manager = PLINKVersionManager(config)
        
    def validate_all_input_files(self):
        """Validate all input files for tensorQTL compatibility"""
        logger.info("Validating all input files...")
        
        validation_results = {}
        file_types = ['genotypes', 'expression', 'protein', 'splicing', 'covariates', 'annotations']
        
        for file_type in file_types:
            file_path = self.config['input_files'].get(file_type)
            if file_path and os.path.exists(file_path):
                if file_type == 'genotypes':
                    validation_results[file_type] = self.validate_genotype_file(file_path)
                elif file_type == 'annotations':
                    validation_results[file_type] = self.validate_annotation_file(file_path)
                else:
                    validation_results[file_type] = self.validate_data_file(file_path, file_type)
        
        all_valid = all(result.get('valid', False) for result in validation_results.values())
        status_msg = "successfully" if all_valid else "with warnings"
        logger.info(f"File validation completed {status_msg}")
        
        return {'all_valid': all_valid, 'details': validation_results}
    
    def validate_genotype_file(self, genotype_file):
        """Validate genotype file format"""
        logger.info(f"Validating genotype file: {genotype_file}")
        
        result = {'file': genotype_file, 'valid': False, 'format': 'unknown'}
        
        try:
            if genotype_file.endswith(('.vcf.gz', '.vcf')):
                result['format'] = 'VCF'
                plink_file, plink_version = self._process_vcf_file(genotype_file)
                if plink_file:
                    result.update({'valid': True, 'converted': True, 'converted_file': plink_file, 'plink_version': plink_version})
                else:
                    result['error'] = "VCF processing failed"
            
            elif genotype_file.endswith('.bed'):
                result['format'] = 'PLINK1.9'
                if self._check_plink_files(genotype_file, ['.bim', '.fam']):
                    result.update({'valid': True, 'plink_version': 'PLINK1.9'})
                else:
                    result['error'] = "Missing PLINK1.9 companion files"
            
            elif genotype_file.endswith('.pgen'):
                result['format'] = 'PLINK2'
                if self._check_plink_files(genotype_file, ['.pvar', '.psam']):
                    result.update({'valid': True, 'plink_version': 'PLINK2'})
                else:
                    result['error'] = "Missing PLINK2 companion files"
            
            else:
                result['error'] = f"Unsupported genotype format: {genotype_file}"
                
        except Exception as e:
            result['error'] = f"Genotype validation error: {str(e)}"
        
        return result
    
    def validate_data_file(self, file_path, file_type):
        """Validate data file format"""
        logger.info(f"Validating {file_type} file: {file_path}")
        
        result = {'file': file_path, 'valid': False}
        
        try:
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(file_path, sep=sep, index_col=0, nrows=5)
                    if not df.empty:
                        result.update({
                            'valid': True, 
                            'format': f'CSV (sep: {repr(sep)})',
                            'rows': len(df),
                            'cols': len(df.columns)
                        })
                        logger.info(f"{file_type} file validated: {len(df)} features")
                        break
                except:
                    continue
            
            if not result['valid']:
                result['error'] = "Could not read file with any separator"
                
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
        
        return result

    def validate_annotation_file(self, annotation_file):
        """Validate annotation file format (BED format)"""
        logger.info(f"Validating annotation file: {annotation_file}")
        
        result = {
            'file': annotation_file,
            'valid': False,
            'format': 'unknown',
            'rows': 0,
            'required_columns': ['chr', 'start', 'end', 'gene_id']
        }
        
        try:
            # Try to read as BED format
            df = pd.read_csv(annotation_file, sep='\t', comment='#', nrows=5)
            result['rows'] = self._count_file_lines(annotation_file) - 1  # Subtract header
            
            # Check for required columns
            required_cols = result['required_columns']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if not missing_cols:
                result['valid'] = True
                result['format'] = 'BED'
                logger.info(f"Annotation file validated: {result['rows']} annotations")
            else:
                result['error'] = f"Missing required columns: {missing_cols}"
                result['available_columns'] = df.columns.tolist()
                
        except Exception as e:
            result['error'] = f"Annotation validation error: {str(e)}"
        
        return result
    
    def _process_vcf_file(self, vcf_file):
        """Process VCF file: filter with PLINK and convert to PLINK format"""
        try:
            # Step 1: Filter VCF
            filtered_vcf_base = os.path.join(self.results_dir, "genotypes_filtered")
            filtered_vcf_file = filtered_vcf_base + ".vcf.gz"
            
            if not os.path.exists(filtered_vcf_file):
                logger.info("Filtering VCF file using PLINK...")
                plink_threads = self.config.get('genotype_processing', {}).get('plink_threads', 1)
                cmd, output_file, plink_type = self.plink_manager.get_vcf_filtering_command(
                    vcf_file, filtered_vcf_base, plink_threads
                )
                
                if run_command(cmd, "VCF filtering", self.config, check=False):
                    if not output_file.endswith('.gz'):
                        run_command(f"bgzip -f {output_file}", "Compressing filtered VCF", self.config)
                    logger.info("VCF filtering completed")
                else:
                    return None, None
            
            # Step 2: Convert to PLINK format
            plink_base = os.path.join(self.results_dir, "genotypes_plink")
            if self._check_existing_plink_files(plink_base):
                logger.info("PLINK files already exist")
                return self._get_plink_genotype_file(plink_base), 'existing'
            
            logger.info("Converting filtered VCF to PLINK format...")
            plink_threads = self.config.get('genotype_processing', {}).get('plink_threads', 1)
            cmd, output_files, plink_type = self.plink_manager.get_vcf_conversion_command(
                filtered_vcf_file, plink_base, plink_threads
            )
            
            if run_command(cmd, f"VCF to {plink_type} conversion", self.config, check=False):
                logger.info(f"VCF to {plink_type} conversion successful")
                return output_files['genotype'], plink_type
            
            return None, None
                
        except Exception as e:
            logger.error(f"VCF processing error: {e}")
            return None, None
    
    def _check_plink_files(self, main_file, extensions):
        """Check if PLINK companion files exist"""
        base = main_file.rsplit('.', 1)[0]
        return all(os.path.exists(base + ext) for ext in extensions)
    
    def _check_existing_plink_files(self, base_prefix):
        """Check if PLINK files already exist"""
        for plink_type in ['plink2', 'plink1.9']:
            extensions = ['.pgen', '.pvar', '.psam'] if plink_type == 'plink2' else ['.bed', '.bim', '.fam']
            if all(os.path.exists(base_prefix + ext) for ext in extensions):
                return True
        return False
    
    def _get_plink_genotype_file(self, base_prefix):
        """Get the main genotype file path"""
        for plink_type in ['plink2', 'plink1.9']:
            extensions = ['.pgen', '.pvar', '.psam'] if plink_type == 'plink2' else ['.bed', '.bim', '.fam']
            if all(os.path.exists(base_prefix + ext) for ext in extensions):
                return base_prefix + ('.pgen' if plink_type == 'plink2' else '.bed')
        return None

    def _count_file_lines(self, file_path):
        """Count total lines in file"""
        try:
            with open(file_path, 'r') as f:
                return sum(1 for line in f)
        except:
            return 0

class DynamicDataHandler:
    """Optimized handler for dynamic QTL data alignment and processing"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data_handling', {})
        
    def align_qtl_data(self, genotype_samples, phenotype_df, covariate_df=None):
        """Align genotype, phenotype, and covariate data"""
        logger.info("Aligning QTL data across all datasets...")
        
        # Convert all sample identifiers to strings
        genotype_samples = [str(sample) for sample in genotype_samples]
        phenotype_samples = [str(sample) for sample in phenotype_df.columns]
        covariate_samples = [str(sample) for sample in covariate_df.columns] if covariate_df is not None else []
        
        # Find common samples
        sample_sets = [set(genotype_samples), set(phenotype_samples)]
        if covariate_samples:
            sample_sets.append(set(covariate_samples))
        
        common_samples = set.intersection(*sample_sets)
        
        if not common_samples:
            raise ValueError("No common samples found across datasets")
        
        common_samples = sorted(common_samples)
        
        # Subset data to common samples
        aligned_phenotype = phenotype_df[common_samples]
        aligned_covariates = covariate_df[common_samples] if covariate_df is not None else pd.DataFrame()
        
        logger.info(f"Data alignment completed: {len(common_samples)} common samples")
        
        return {
            'phenotype': aligned_phenotype,
            'covariates': aligned_covariates,
            'common_samples': common_samples
        }
    
    def validate_phenotype_data(self, phenotype_df, qtl_type):
        """Validate phenotype data with QTL-type specific checks"""
        logger.info(f"Validating {qtl_type} phenotype data...")
        
        if phenotype_df.empty:
            raise ValueError(f"{qtl_type} phenotype data is empty")
        
        # Ensure numeric data
        non_numeric = phenotype_df.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            logger.warning(f"Found {non_numeric.shape[1]} non-numeric columns, converting")
            phenotype_df = phenotype_df.apply(pd.to_numeric, errors='coerce')
            phenotype_df = phenotype_df.dropna(axis=1, how='all')
        
        # Remove constant features
        constant_features = phenotype_df.std(axis=1) == 0
        if constant_features.any():
            logger.warning(f"Removing {constant_features.sum()} constant features")
            phenotype_df = phenotype_df[~constant_features]
        
        # Remove features with excessive missingness
        missing_threshold = self.data_config.get('missing_value_threshold', 0.2)
        missing_rates = phenotype_df.isna().sum(axis=1) / phenotype_df.shape[1]
        high_missing = missing_rates > missing_threshold
        
        if high_missing.any():
            logger.warning(f"Removing {high_missing.sum()} features with >{missing_threshold*100}% missing values")
            phenotype_df = phenotype_df[~high_missing]
        
        logger.info(f"{qtl_type} validation: {phenotype_df.shape[0]} features, {phenotype_df.shape[1]} samples")
        return phenotype_df

    def generate_enhanced_covariates(self, phenotype_df, existing_covariates=None):
        """Generate enhanced covariates including PCA components"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            logger.info("Generating enhanced covariates...")
            
            # Prepare data for PCA
            pheno_for_pca = phenotype_df.T.fillna(phenotype_df.T.mean())
            
            # Remove constant features
            constant_mask = pheno_for_pca.std() == 0
            if constant_mask.any():
                pheno_for_pca = pheno_for_pca.loc[:, ~constant_mask]
            
            if pheno_for_pca.shape[1] < 2:
                logger.warning("Insufficient features for PCA")
                return existing_covariates
            
            # Standardize and perform PCA
            scaler = StandardScaler()
            pheno_scaled = scaler.fit_transform(pheno_for_pca)
            
            n_components = min(10, pheno_scaled.shape[1], pheno_scaled.shape[0] - 1)
            if n_components < 1:
                return existing_covariates
            
            pca = PCA(n_components=n_components)
            pc_components = pca.fit_transform(pheno_scaled)
            
            # Create PCA covariates
            pc_columns = [f'PC{i+1}' for i in range(n_components)]
            pc_df = pd.DataFrame(pc_components, index=pheno_for_pca.index, columns=pc_columns)
            pc_df = pc_df.T  # Transpose to covariates x samples
            
            # Combine with existing covariates
            if existing_covariates is not None and not existing_covariates.empty:
                enhanced_covariates = pd.concat([existing_covariates, pc_df])
            else:
                enhanced_covariates = pc_df
            
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"Enhanced covariates: {n_components} PC components (explained variance: {explained_variance:.3f})")
            
            return enhanced_covariates
            
        except Exception as e:
            logger.warning(f"Enhanced covariate generation failed: {e}")
            return existing_covariates

class HardwareOptimizer:
    """Optimize hardware utilization for tensorQTL analysis"""
    
    def __init__(self, config):
        self.config = config
        
    def setup_hardware(self):
        """Setup optimal hardware configuration"""
        device_info = self.detect_available_devices()
        
        use_gpu = self.config.get('tensorqtl', {}).get('use_gpu', False) and device_info['gpu_available']
        
        if use_gpu:
            device = self.setup_gpu()
        else:
            device = self.setup_cpu_optimized()
        
        return device, device_info
    
    def detect_available_devices(self):
        """Detect available hardware devices"""
        device_info = {
            'gpu_available': False,
            'gpu_count': 0,
            'cpu_cores': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if TENSORQTL_AVAILABLE and torch.cuda.is_available():
            device_info.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            })
        
        logger.info(f"Hardware: {device_info['cpu_cores']} CPU cores, {device_info['memory_gb']:.1f} GB RAM, {device_info['gpu_count']} GPUs")
        return device_info
    
    def setup_gpu(self):
        """Setup GPU configuration"""
        if not TENSORQTL_AVAILABLE:
            return self.setup_cpu_optimized()
            
        try:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            logger.info("GPU acceleration enabled")
            return torch.device('cuda')
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}, falling back to CPU")
            return self.setup_cpu_optimized()
    
    def setup_cpu_optimized(self):
        """Setup CPU configuration for optimal performance"""
        num_threads = self.config.get('performance', {}).get('num_threads', min(16, os.cpu_count()))
        
        if TENSORQTL_AVAILABLE:
            try:
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(num_threads)
            except RuntimeError:
                pass
        
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        
        logger.info(f"Using {num_threads} CPU threads")
        return None

class PhenotypeProcessor:
    """Optimized phenotype data processing for tensorQTL"""
    
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        self.data_handler = DynamicDataHandler(config)
        
    def prepare_phenotype_data(self, qtl_type, genotype_samples=None):
        """Prepare phenotype data optimized for tensorQTL"""
        logger.info(f"Preparing {qtl_type} phenotype data...")
        
        try:
            # Load data
            config_key = self._map_qtl_type_to_config_key(qtl_type)
            pheno_file = self.config['input_files'].get(config_key)
            
            if not pheno_file or not os.path.exists(pheno_file):
                raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")
            
            raw_pheno_df = self._load_data_file(pheno_file, f"{qtl_type} phenotype")
            cov_df = self._load_covariate_data()
            
            # Align data if genotype samples provided
            if genotype_samples is not None:
                aligned_data = self.data_handler.align_qtl_data(genotype_samples, raw_pheno_df, cov_df)
                raw_pheno_df = aligned_data['phenotype']
                cov_df = aligned_data['covariates']
            
            # Validate and process data
            raw_pheno_df = self.data_handler.validate_phenotype_data(raw_pheno_df, qtl_type)
            raw_pheno_df = self._apply_qc_filters(raw_pheno_df, qtl_type)
            
            # Apply expression-specific filtering for eQTL data
            if qtl_type == 'eqtl' and self.config.get('qc', {}).get('filter_lowly_expressed_genes', True):
                raw_pheno_df = self._filter_lowly_expressed_genes(raw_pheno_df, qtl_type)
            
            # Enhanced pipeline with batch correction
            if self.config.get('enhanced_pipeline', {}).get('enable', True):
                logger.info(f"Using enhanced pipeline for {qtl_type}")
                final_phenotype_df, pipeline_info = self._run_enhanced_pipeline(raw_pheno_df, qtl_type)
            else:
                # Apply normalization using traditional method
                if self.config.get('qc', {}).get('normalize', True):
                    final_phenotype_df = self._apply_normalization(raw_pheno_df, qtl_type)
                else:
                    final_phenotype_df = raw_pheno_df
                    logger.info("Using raw data without normalization")
            
            # Generate enhanced covariates if enabled
            if self.config.get('enhanced_qc', {}).get('generate_enhanced_covariates', True):
                enhanced_cov_df = self.data_handler.generate_enhanced_covariates(final_phenotype_df, cov_df)
                if enhanced_cov_df is not None:
                    cov_df = enhanced_cov_df
            
            # Generate normalization comparison if enabled
            if self.config.get('enhanced_qc', {}).get('generate_normalization_plots', True) and NormalizationComparison:
                self._generate_normalization_comparison(raw_pheno_df, final_phenotype_df, qtl_type)
            
            # Prepare for tensorQTL (samples x features)
            final_phenotype_df = final_phenotype_df.T
            
            # Save processed data
            output_files = self._save_processed_data(final_phenotype_df, qtl_type, cov_df)
            
            logger.info(f"Prepared {qtl_type} data: {final_phenotype_df.shape[1]} features retained")
            return output_files
            
        except Exception as e:
            logger.error(f"Phenotype preparation failed for {qtl_type}: {e}")
            raise

    def _run_enhanced_pipeline(self, raw_pheno_df, qtl_type):
        """Run enhanced normalization and batch correction pipeline"""
        try:
            logger.info("Running enhanced pipeline...")
            
            # Get normalization method
            normalization_method = self._get_normalization_method(qtl_type)
            logger.info(f"Normalization method: {normalization_method}")
            
            # Apply normalization
            normalized_df = self._apply_normalization(raw_pheno_df, qtl_type)
            
            # Apply batch correction if enabled and available
            if self.config.get('batch_correction', {}).get('enabled', {}).get('eqtl', True) and BATCH_CORRECTION_AVAILABLE:
                logger.info("Applying batch correction...")
                
                try:
                    corrected_df, correction_info = run_batch_correction_pipeline(
                        normalized_data=normalized_df,
                        qtl_type=qtl_type, 
                        config=self.config
                    )
                    
                    if corrected_df is not None and not corrected_df.empty:
                        pipeline_info = {
                            'normalization_method': normalization_method,
                            'batch_correction_applied': True,
                            'correction_method': 'custom_linear_regression'
                        }
                        logger.info("Batch correction completed successfully")
                        return corrected_df, pipeline_info
                    else:
                        logger.warning("Batch correction returned empty data, using normalized data")
                        
                except Exception as e:
                    logger.error(f"Custom batch correction failed: {e}")
            
            # If batch correction not applied or failed, use normalized data
            pipeline_info = {
                'normalization_method': normalization_method,
                'batch_correction_applied': False,
                'reason': 'Not enabled or failed'
            }
            
            logger.info(f"Using {normalization_method} normalized data without batch correction")
            return normalized_df, pipeline_info
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}, falling back to standard normalization")
            normalized_df = self._apply_normalization(raw_pheno_df, qtl_type)
            pipeline_info = {
                'normalization_method': self._get_normalization_method(qtl_type),
                'batch_correction_applied': False,
                'reason': f'Pipeline error: {str(e)}'
            }
            
            return normalized_df, pipeline_info

    def _get_normalization_method(self, qtl_type):
        """Get normalization method from config for specific QTL type"""
        normalization_config = self.config.get('normalization', {})
        qtl_config = normalization_config.get(qtl_type, {})
        return qtl_config.get('method', 'vst')

    def _filter_lowly_expressed_genes(self, pheno_df, qtl_type):
        """Filter out genes with low expression in more than X% of samples"""
        logger.info("Filtering lowly expressed genes...")
        
        qc_config = self.config.get('qc', {})
        low_expression_threshold = qc_config.get('low_expression_threshold', 0.1)
        max_low_expression_samples_percentage = qc_config.get('max_low_expression_samples_percentage', 10)
        
        original_count = pheno_df.shape[0]
        
        # Calculate percentage of samples with low expression for each gene
        low_expression_mask = pheno_df < low_expression_threshold
        low_expression_percentage = (low_expression_mask.sum(axis=1) / pheno_df.shape[1]) * 100
        
        # Filter genes where low expression percentage is below threshold
        keep_genes = low_expression_percentage <= max_low_expression_samples_percentage
        filtered_df = pheno_df[keep_genes]
        
        filtered_count = filtered_df.shape[0]
        logger.info(f"Low expression filtering: {filtered_count}/{original_count} genes retained")
        
        return filtered_df

    def _load_data_file(self, file_path, description):
        """Load data file with robust error handling"""
        logger.info(f"Loading {description}: {file_path}")
        
        for sep in ['\t', ',', ' ']:
            try:
                df = pd.read_csv(file_path, sep=sep, index_col=0)
                if not df.empty:
                    logger.info(f"Successfully loaded with separator '{sep}': {df.shape[0]} features")
                    return df.apply(pd.to_numeric, errors='coerce')
            except:
                continue
        
        # Fallback
        df = pd.read_csv(file_path, index_col=0)
        return df.apply(pd.to_numeric, errors='coerce')
    
    def _load_covariate_data(self):
        """Load covariate data from either exp_covariate_design or original covariates file"""
        # Check for new covariate design structure first
        batch_config = self.config.get('batch_correction', {})
        exp_covariate_design_file = batch_config.get('exp_covariate_design')
        
        if exp_covariate_design_file and os.path.exists(exp_covariate_design_file):
            logger.info(f"Loading covariate design from: {exp_covariate_design_file}")
            try:
                cov_df = self._load_data_file(exp_covariate_design_file, "covariate design")
                logger.info(f"Loaded covariate design: {cov_df.shape[0]} covariates, {cov_df.shape[1]} samples")
                return cov_df
            except Exception as e:
                logger.warning(f"Failed to load exp_covariate_design file: {e}, falling back to original covariates")
        
        # Fallback to original covariates file
        covariates_file = self.config['input_files'].get('covariates')
        if not covariates_file or not os.path.exists(covariates_file):
            return pd.DataFrame()
        
        cov_df = self._load_data_file(covariates_file, "covariates")
        logger.info(f"Loaded covariates: {cov_df.shape[0]} covariates")
        return cov_df
    
    def _apply_qc_filters(self, pheno_df, qtl_type):
        """Apply quality control filters"""
        original_count = pheno_df.shape[0]
        qc_config = self.config.get('qc', {})
        
        # Remove constant features
        constant_threshold = qc_config.get('constant_threshold', 0.95)
        non_constant_mask = (pheno_df.nunique(axis=1) / pheno_df.shape[1]) > (1 - constant_threshold)
        pheno_df = pheno_df[non_constant_mask]
        
        # Remove features with too many missing values
        missing_threshold = qc_config.get('missing_value_threshold', 0.2)
        low_missing_mask = (pheno_df.isna().sum(axis=1) / pheno_df.shape[1]) < missing_threshold
        pheno_df = pheno_df[low_missing_mask]
        
        filtered_count = pheno_df.shape[0]
        logger.info(f"QC filtering: {filtered_count}/{original_count} features retained")
        return pheno_df
    
    def _apply_normalization(self, pheno_df, qtl_type):
        """Apply normalization based on user-defined method"""
        norm_config = self.config.get('normalization', {}).get(qtl_type, {})
        method = norm_config.get('method', 'log2')
        
        logger.info(f"Applying {method} normalization for {qtl_type}")
        
        normalization_methods = {
            'log2': self._apply_log2_normalization,
            'vst': self._apply_vst_normalization,
            'quantile': self._apply_quantile_normalization,
            'zscore': self._apply_zscore_normalization,
            'arcsinh': self._apply_arcsinh_normalization,
            'tpm': self._apply_tpm_normalization
        }
        
        if method in normalization_methods:
            return normalization_methods[method](pheno_df, qtl_type)
        else:
            logger.warning(f"Unknown normalization method '{method}', using raw data")
            return pheno_df
    
    def _apply_log2_normalization(self, pheno_df, qtl_type):
        """Apply log2 transformation"""
        pseudocount = self.config.get('normalization', {}).get(qtl_type, {}).get('log2_pseudocount', 1)
        normalized_df = np.log2(pheno_df + pseudocount)
        logger.info(f"Applied log2 transformation (pseudocount={pseudocount})")
        return normalized_df
    
    def _apply_vst_normalization(self, pheno_df, qtl_type):
        """Apply VST normalization"""
        if not DESEQ2_VST_AVAILABLE:
            raise ImportError("DESeq2 VST Python implementation not available")
        
        try:
            # Ensure data is appropriate for VST
            if (pheno_df < 0).any().any():
                logger.warning("Negative values found, taking absolute values for VST")
                pheno_df = pheno_df.abs()
            
            pheno_df = pheno_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            try:
                vst_df = deseq2_vst_python(pheno_df)
                logger.info("DESeq2 VST normalization completed")
            except Exception:
                vst_df = simple_vst_fallback(pheno_df)
                logger.info("Simplified VST normalization completed")
            
            return vst_df
            
        except Exception as e:
            logger.error(f"VST normalization failed: {e}, falling back to log2")
            return self._apply_log2_normalization(pheno_df, qtl_type)
    
    def _apply_quantile_normalization(self, pheno_df, qtl_type):
        """Apply quantile normalization"""
        try:
            from sklearn.preprocessing import quantile_transform
            pheno_filled = pheno_df.fillna(pheno_df.mean())
            normalized_array = quantile_transform(pheno_filled.T, n_quantiles=min(1000, pheno_filled.shape[0]))
            normalized_df = pd.DataFrame(normalized_array.T, index=pheno_df.index, columns=pheno_df.columns)
            logger.info("Quantile normalization completed")
            return normalized_df
        except Exception as e:
            logger.error(f"Quantile normalization failed: {e}")
            return pheno_df
    
    def _apply_zscore_normalization(self, pheno_df, qtl_type):
        """Apply z-score normalization"""
        pheno_filled = pheno_df.fillna(pheno_df.mean())
        normalized_df = (pheno_filled - pheno_filled.mean(axis=1).values.reshape(-1, 1)) 
        normalized_df = normalized_df / pheno_filled.std(axis=1).values.reshape(-1, 1)
        
        # Handle constant features
        constant_mask = pheno_filled.std(axis=1) == 0
        if constant_mask.any():
            normalized_df.loc[constant_mask] = 0
            logger.warning(f"Found {constant_mask.sum()} constant features, setting z-score to 0")
        
        logger.info("Z-score normalization completed")
        return normalized_df
    
    def _apply_arcsinh_normalization(self, pheno_df, qtl_type):
        """Apply arcsinh transformation"""
        cofactor = self.config.get('normalization', {}).get(qtl_type, {}).get('arcsinh_cofactor', 1)
        normalized_df = np.arcsinh(pheno_df / cofactor)
        logger.info(f"Arcsinh transformation completed (cofactor={cofactor})")
        return normalized_df
    
    def _apply_tpm_normalization(self, pheno_df, qtl_type):
        """Apply TPM-like normalization"""
        rpm_df = pheno_df.div(pheno_df.sum(axis=0)) * 1e6
        logger.info("TPM-like normalization completed")
        return rpm_df

    def _generate_normalization_comparison(self, raw_df, normalized_df, qtl_type):
        """Generate normalization comparison plots"""
        try:
            if NormalizationComparison:
                comparison = NormalizationComparison(self.config, self.results_dir)
                comparison.generate_comprehensive_comparison(
                    qtl_type, raw_df.copy(), normalized_df, 
                    self.config.get('normalization', {}).get(qtl_type, {}).get('method', 'unknown')
                )
                logger.info(f"Normalization comparison completed for {qtl_type}")
        except Exception as e:
            logger.warning(f"Normalization comparison failed: {e}")
    
    def _save_processed_data(self, normalized_df, qtl_type, covariate_df):
        """Save processed phenotype data"""
        # Ensure numeric data
        normalized_df = normalized_df.apply(pd.to_numeric, errors='coerce')
        if not covariate_df.empty:
            covariate_df = covariate_df.apply(pd.to_numeric, errors='coerce')
        
        # Save phenotype data
        output_format = self.config.get('tensorqtl', {}).get('output_format', 'parquet')
        if output_format == 'parquet':
            pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.parquet")
            try:
                normalized_df.to_parquet(pheno_file)
            except Exception:
                pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.txt.gz")
                normalized_df.to_csv(pheno_file, sep='\t', compression='gzip')
        else:
            pheno_file = os.path.join(self.results_dir, f"{qtl_type}_phenotypes.txt.gz")
            normalized_df.to_csv(pheno_file, sep='\t', compression='gzip')
        
        # Save covariates
        if not covariate_df.empty:
            cov_file = os.path.join(self.results_dir, f"{qtl_type}_covariates.txt.gz")
            covariate_df.to_csv(cov_file, sep='\t', compression='gzip')
        else:
            cov_file = None
        
        # Create and save phenotype positions
        pheno_pos_file = os.path.join(self.results_dir, f"{qtl_type}_phenotype_positions.txt.gz")
        pheno_pos_df = self._create_phenotype_positions(normalized_df.columns, qtl_type)
        pheno_pos_df.to_csv(pheno_pos_file, sep='\t', compression='gzip')
        
        logger.info(f"Saved processed data: {pheno_file}")
        
        return {
            'phenotype_file': pheno_file,
            'phenotype_df': normalized_df,
            'phenotype_pos_file': pheno_pos_file,
            'phenotype_pos_df': pheno_pos_df,
            'covariate_file': cov_file,
            'covariate_df': covariate_df
        }

    def _create_phenotype_positions(self, feature_ids, qtl_type):
        """Create phenotype positions DataFrame"""
        annotation_file = self.config['input_files'].get('annotations')
        
        try:
            if annotation_file and os.path.exists(annotation_file):
                annot_df = pd.read_csv(annotation_file, sep='\t', comment='#')
            else:
                annot_df = pd.DataFrame()
                logger.warning("No annotation file found, creating default positions")
        except Exception as e:
            logger.warning(f"Could not read annotation file: {e}, creating default positions")
            annot_df = pd.DataFrame()
        
        positions_data = []
        
        for feature_id in feature_ids:
            if not annot_df.empty and 'gene_id' in annot_df.columns:
                feature_annot = annot_df[annot_df['gene_id'] == feature_id]
                if len(feature_annot) > 0:
                    feature_annot = feature_annot.iloc[0]
                    positions_data.append({
                        'phenotype_id': feature_id,
                        'chr': str(feature_annot.get('chr', '1')),
                        'start': int(feature_annot.get('start', 1)),
                        'end': int(feature_annot.get('end', 1000)),
                        'strand': feature_annot.get('strand', '+')
                    })
                else:
                    # Create default annotation if not found
                    positions_data.append({
                        'phenotype_id': feature_id,
                        'chr': '1',
                        'start': 1,
                        'end': 1000,
                        'strand': '+'
                    })
            else:
                # Create default annotations
                positions_data.append({
                    'phenotype_id': feature_id,
                    'chr': '1',
                    'start': 1,
                    'end': 1000,
                    'strand': '+'
                })
        
        positions_df = pd.DataFrame(positions_data)
        positions_df = positions_df.set_index('phenotype_id')
        return positions_df
    
    def _map_qtl_type_to_config_key(self, qtl_type):
        """Map QTL type to config file key"""
        mapping = {'eqtl': 'expression', 'pqtl': 'protein', 'sqtl': 'splicing'}
        return mapping.get(qtl_type, qtl_type)

class GenotypeLoader:
    """Optimized genotype data loading for tensorQTL - NO MANUAL FILTERING"""
    
    def __init__(self, config):
        self.config = config
    
    def load_genotypes(self, genotype_file):
        """Load genotype data optimized for tensorQTL"""
        if not TENSORQTL_AVAILABLE:
            raise ImportError("tensorQTL is not available. Please install: pip install tensorqtl==1.0.10")
        
        logger.info(f"Loading genotype data: {genotype_file}")
        
        if genotype_file.endswith('.bed'):
            plink_prefix = genotype_file.replace('.bed', '')
        elif genotype_file.endswith('.pgen'):
            plink_prefix = genotype_file.replace('.pgen', '')
        else:
            raise ValueError(f"Unsupported genotype format: {genotype_file}")
        
        pr = genotypeio.read_plink(plink_prefix)
        
        # Handle different return formats
        if isinstance(pr, tuple):
            genotypes, variants, samples = pr
            logger.info(f"Genotype data: {genotypes.shape[0]} variants × {genotypes.shape[1]} samples")
            
            class GenotypeContainer:
                def __init__(self, genotypes, variants, samples):
                    self.genotypes = genotypes
                    self.variants = variants
                    self.samples = samples
            
            return GenotypeContainer(genotypes, variants, samples)
        else:
            logger.info(f"Genotype data: {pr.genotypes.shape[0]} variants × {pr.genotypes.shape[1]} samples")
            return pr

def prepare_genotypes(config, results_dir):
    """Prepare genotype data optimized for tensorQTL"""
    logger.info("Preparing genotype data for tensorQTL...")
    
    validator = FileFormatValidator(config)
    validation_results = validator.validate_all_input_files()
    
    if not validation_results['all_valid']:
        logger.error("File validation failed, cannot proceed with genotype preparation")
        return None
    
    genotype_validation = validation_results['details'].get('genotypes', {})
    genotype_file = genotype_validation.get('converted_file', genotype_validation.get('file'))
    
    logger.info(f"Genotype preparation completed: {genotype_file}")
    return genotype_file

def load_covariates(config, results_dir, qtl_type='eqtl'):
    """Load and prepare covariates for tensorQTL"""
    logger.info(f"Loading covariates for {qtl_type}...")
    
    # Try to load pre-processed covariates first
    cov_file = os.path.join(results_dir, f"{qtl_type}_covariates.txt.gz")
    if os.path.exists(cov_file):
        cov_df = pd.read_csv(cov_file, sep='\t', index_col=0)
        logger.info(f"Loaded pre-processed covariates: {cov_df.shape[1]} samples")
        return cov_df.T
    
    # Fallback to original covariate file or exp_covariate_design
    batch_config = config.get('batch_correction', {})
    exp_covariate_design_file = batch_config.get('exp_covariate_design')
    
    if exp_covariate_design_file and os.path.exists(exp_covariate_design_file):
        logger.info(f"Loading covariate design from: {exp_covariate_design_file}")
        try:
            cov_df = pd.read_csv(exp_covariate_design_file, sep='\t', index_col=0)
            logger.info(f"Loaded covariate design: {cov_df.shape[0]} samples, {cov_df.shape[1]} covariates")
            return cov_df.T
        except Exception as e:
            logger.warning(f"Failed to load exp_covariate_design file: {e}")
    
    # Final fallback to original covariates file
    covariates_file = config['input_files'].get('covariates')
    if not covariates_file or not os.path.exists(covariates_file):
        logger.warning("No covariate file found")
        return None
    
    # Load with dynamic format handling
    processor = PhenotypeProcessor(config, results_dir)
    cov_df = processor._load_data_file(covariates_file, "covariates")
    
    if cov_df.empty:
        logger.warning("Covariate data is empty")
        return None
    
    # Transpose for tensorQTL and clean
    cov_df = cov_df.T
    cov_df = cov_df.apply(pd.to_numeric, errors='coerce').fillna(cov_df.mean())
    
    logger.info(f"Loaded covariates: {cov_df.shape[1]} covariates")
    return cov_df

def run_qtl_analysis(config, genotype_file, qtl_type, results_dir, analysis_mode='cis'):
    """Run QTL analysis using tensorQTL"""
    if not TENSORQTL_AVAILABLE:
        raise ImportError("tensorQTL is not available. Please install: pip install tensorqtl==1.0.10")
    
    logger.info(f"Running {qtl_type} {analysis_mode}-QTL analysis...")
    
    try:
        # Validate files
        validator = FileFormatValidator(config)
        if not validator.validate_all_input_files()['all_valid']:
            return {'status': 'failed', 'error': 'File validation failed'}
        
        # Setup hardware
        hardware_optimizer = HardwareOptimizer(config)
        device, device_info = hardware_optimizer.setup_hardware()
        
        if device_info['gpu_available']:
            logger.info(f"Using GPU for {qtl_type} {analysis_mode}-QTL analysis")
        else:
            logger.info(f"Using CPU for {qtl_type} {analysis_mode}-QTL analysis")
        
        # Load genotype data
        genotype_loader = GenotypeLoader(config)
        pr = genotype_loader.load_genotypes(genotype_file)
        
        # Extract samples
        if hasattr(pr, 'samples'):
            genotype_samples = [str(sample) for sample in pr.samples]
        else:
            genotype_samples = [str(sample) for sample in pr.genotypes.columns.tolist()]
        
        # Prepare phenotype data
        pheno_processor = PhenotypeProcessor(config, results_dir)
        pheno_data = pheno_processor.prepare_phenotype_data(qtl_type, genotype_samples)
        
        # Load covariates
        covariates_df = load_covariates(config, results_dir, qtl_type)
        
        # Prepare data for tensorQTL
        phenotype_df_t = pheno_data['phenotype_df']
        phenotype_df_t.index = [str(idx) for idx in phenotype_df_t.index]
        phenotype_df_t.columns = [str(col) for col in phenotype_df_t.columns]
        
        # Run analysis
        analysis_config = config.get('tensorqtl', {})
        
        if analysis_mode == 'cis':
            phenotype_pos_df = pheno_data.get('phenotype_pos_df')
            cis_df = cis.map_cis(
                pr,
                phenotype_df_t, 
                phenotype_pos_df,
                covariates_df=covariates_df,
                window=analysis_config.get('cis_window', 1000000),
                seed=analysis_config.get('seed', 12345)
            )
            
            result_file = os.path.join(results_dir, f"{qtl_type}_cis.cis_qtl.txt.gz")
            cis_df.to_csv(result_file, sep='\t', compression='gzip')
            significant_count = len(cis_df[cis_df['pval_perm'] < analysis_config.get('fdr_threshold', 0.05)]) if 'pval_perm' in cis_df.columns else 0
            
        else:  # trans
            trans_df = trans.map_trans(
                pr,
                phenotype_df_t,
                covariates_df=covariates_df,
                batch_size=analysis_config.get('batch_size', 20000),
                return_sparse=True
            )
            
            result_file = os.path.join(results_dir, f"{qtl_type}_trans.trans_qtl.txt.gz")
            if trans_df is not None and len(trans_df) > 0:
                if 'pval' in trans_df.columns:
                    fdr = calculate_qvalues(trans_df['pval'])
                    trans_df['fdr'] = fdr
                    significant_count = (fdr < analysis_config.get('fdr_threshold', 0.05)).sum()
                trans_df.to_csv(result_file, sep='\t', compression='gzip')
            else:
                significant_count = 0
                pd.DataFrame().to_csv(result_file, sep='\t', compression='gzip')
        
        logger.info(f"{qtl_type} {analysis_mode}: Found {significant_count} significant associations")
        
        # Clean up GPU memory
        if device_info['gpu_available']:
            torch.cuda.empty_cache()
        
        return {
            'result_file': result_file,
            'significant_count': significant_count,
            'status': 'completed',
            'hardware_used': 'GPU' if device_info['gpu_available'] else 'CPU'
        }
        
    except Exception as e:
        logger.error(f"{analysis_mode}-QTL analysis failed for {qtl_type}: {e}")
        return {'status': 'failed', 'error': str(e)}

def run_command(cmd, description, config, check=True):
    """Run shell command with error handling"""
    logger.info(f"Executing: {description}")
    
    timeout = config.get('large_data', {}).get('command_timeout', 7200)
    
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True,
            executable='/bin/bash', timeout=timeout
        )
        
        if result.returncode == 0:
            logger.info(f"{description} completed successfully")
        else:
            logger.warning(f"{description} completed with exit code {result.returncode}")
            if result.stderr.strip():
                logger.error(f"Error: {result.stderr.strip()}")
            
        return result
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"{description} failed: {e}")
        if check:
            raise
        return None

# Backward compatibility functions
def prepare_phenotype_data(config, qtl_type, results_dir):
    """Prepare phenotype data for tensorQTL - Compatibility wrapper"""
    processor = PhenotypeProcessor(config, results_dir)
    return processor.prepare_phenotype_data(qtl_type)

def process_expression_data(config, results_dir=None):
    """Process expression data for modular pipeline"""
    if results_dir is None:
        results_dir = config.get('results_dir', 'results')
        logger.warning(f"results_dir not provided, using config value: {results_dir}")
    
    return prepare_phenotype_data(config, 'eqtl', results_dir)

def run_cis_analysis(config, genotype_file, qtl_type, results_dir):
    """Run cis-QTL analysis - Compatibility wrapper"""
    return run_qtl_analysis(config, genotype_file, qtl_type, results_dir, 'cis')

def run_trans_analysis(config, genotype_file, qtl_type, results_dir):
    """Run trans-QTL analysis - Compatibility wrapper"""
    return run_qtl_analysis(config, genotype_file, qtl_type, results_dir, 'trans')

def run_qtl_mapping(config, genotype_file, qtl_type, results_dir, analysis_mode='cis'):
    """Unified QTL mapping function for modular pipeline"""
    if analysis_mode == 'cis':
        return run_cis_analysis(config, genotype_file, qtl_type, results_dir)
    elif analysis_mode == 'trans':
        return run_trans_analysis(config, genotype_file, qtl_type, results_dir)
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")

def validate_and_prepare_tensorqtl_inputs(config, qtl_type='eqtl'):
    """High-level function to validate and prepare all tensorQTL inputs"""
    logger.info("Validating and preparing tensorQTL inputs...")
    
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    validator = FileFormatValidator(config)
    validation_results = validator.validate_all_input_files()
    
    if not validation_results['all_valid']:
        logger.error("File validation failed")
        for file_type, result in validation_results['details'].items():
            if not result.get('valid', False):
                logger.error(f"   - {file_type}: {result.get('error', 'Unknown error')}")
        return None
    
    logger.info("All input files validated successfully")
    
    # Prepare phenotype data to get all required files
    pheno_processor = PhenotypeProcessor(config, results_dir)
    pheno_data = pheno_processor.prepare_phenotype_data(qtl_type)
    
    # Get genotype file
    genotype_validation = validation_results['details'].get('genotypes', {})
    genotype_file = genotype_validation.get('converted_file', genotype_validation.get('file'))
    
    tensorqtl_inputs = {
        'expression_bed': os.path.join(results_dir, f"{qtl_type}_expression.bed"),
        'covariates': os.path.join(results_dir, f"{qtl_type}_covariates.txt"),
        'genotypes': genotype_file,
        'validation': validation_results
    }
    
    logger.info("TensorQTL inputs prepared successfully")
    return tensorqtl_inputs

if __name__ == "__main__":
    """Standalone QTL analysis script"""
    import yaml
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/config.yaml"
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate and prepare inputs
        tensorqtl_inputs = validate_and_prepare_tensorqtl_inputs(config)
        
        if not tensorqtl_inputs:
            logger.error("Input validation failed, exiting")
            sys.exit(1)
        
        genotype_file = tensorqtl_inputs['genotypes']
        qtl_types = config['analysis'].get('qtl_types', ['eqtl'])
        
        if isinstance(qtl_types, str) and qtl_types != 'all':
            qtl_types = [qtl_types]
        
        analysis_mode = config['analysis'].get('qtl_mode', 'cis')
        
        for qtl_type in qtl_types:
            if analysis_mode in ['cis', 'both']:
                run_cis_analysis(config, genotype_file, qtl_type, config['results_dir'])
            if analysis_mode in ['trans', 'both']:
                run_trans_analysis(config, genotype_file, qtl_type, config['results_dir'])
        
        logger.info("QTL analysis completed successfully")
        
    except Exception as e:
        logger.error(f"QTL analysis failed: {e}")
        sys.exit(1)