#!/usr/bin/env python3
"""
Enhanced GWAS analysis utilities with comprehensive error handling and performance optimizations
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with parallel processing, memory optimization, comprehensive reporting, and dynamic data handling.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import gc
import psutil

warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

try:
    from scripts.utils.directory_manager import get_directory_manager, get_module_directories
except ImportError as e:
    logger.warning(f"Directory manager not available: {e}")
    # Fallback directory structure
    class DirectoryManager:
        def __init__(self, results_dir):
            self.results_dir = Path(results_dir)
        
        def get_directory(self, category, subcategory=None, create=True):
            if subcategory:
                path = self.results_dir / category / subcategory
            else:
                path = self.results_dir / category
            
            if create:
                path.mkdir(parents=True, exist_ok=True)
            return path

class DynamicDataHandler:
    """Enhanced handler for dynamic covariate and phenotype data"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data_handling', {})
        
    def validate_and_prepare_data(self, genotype_samples, phenotype_file, covariate_file):
        """Validate and prepare phenotype and covariate data with dynamic handling"""
        logger.info("🔍 Validating and preparing dynamic data...")
        
        # Load and validate phenotype data
        pheno_data = self._load_phenotype_data(phenotype_file)
        if pheno_data.empty:
            raise ValueError("Phenotype data is empty or invalid")
        
        # Load and validate covariate data
        cov_data = self._load_covariate_data(covariate_file)
        
        # Align samples across all datasets
        aligned_data = self._align_samples(genotype_samples, pheno_data, cov_data)
        
        # Generate enhanced covariates if needed
        if self.data_config.get('generate_enhanced_covariates', True):
            aligned_data['enhanced_covariates'] = self._generate_enhanced_covariates(
                aligned_data['phenotype'], aligned_data['covariates']
            )
        
        logger.info(f"✅ Data prepared: {aligned_data['phenotype'].shape[1]} phenotypes, "
                   f"{aligned_data['covariates'].shape[1]} covariates, "
                   f"{len(aligned_data['common_samples'])} common samples")
        
        return aligned_data
    
    def _load_phenotype_data(self, phenotype_file):
        """Load phenotype data with flexible format handling"""
        logger.info(f"📊 Loading phenotype data: {phenotype_file}")
        
        if not os.path.exists(phenotype_file):
            raise FileNotFoundError(f"Phenotype file not found: {phenotype_file}")
        
        try:
            # Try different formats and separators
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(phenotype_file, sep=sep, index_col=0)
                    if not df.empty:
                        logger.info(f"✅ Phenotype data loaded with separator '{sep}': {df.shape}")
                        return df
                except Exception as e:
                    continue
            
            # If standard separators fail, try with header detection
            df = pd.read_csv(phenotype_file, index_col=0)
            if df.empty:
                raise ValueError("Could not load phenotype data with any standard separator")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading phenotype data: {e}")
            raise
    
    def _load_covariate_data(self, covariate_file):
        """Load covariate data with dynamic column handling"""
        logger.info(f"📊 Loading covariate data: {covariate_file}")
        
        if not os.path.exists(covariate_file):
            logger.warning(f"⚠️ Covariate file not found: {covariate_file}")
            return pd.DataFrame()
        
        try:
            # Try different formats and separators
            for sep in ['\t', ',', ' ']:
                try:
                    df = pd.read_csv(covariate_file, sep=sep, index_col=0)
                    if not df.empty:
                        logger.info(f"✅ Covariate data loaded with separator '{sep}': {df.shape}")
                        return df
                except Exception as e:
                    continue
            
            # If standard separators fail, try with header detection
            df = pd.read_csv(covariate_file, index_col=0)
            if df.empty:
                logger.warning("⚠️ Covariate data is empty")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading covariate data: {e}")
            return pd.DataFrame()
    
    def _align_samples(self, genotype_samples, phenotype_df, covariate_df):
        """Align samples across genotype, phenotype, and covariate data"""
        # Convert genotype samples to set for fast lookup
        genotype_sample_set = set(genotype_samples)
        
        # Get phenotype samples
        phenotype_samples = set(phenotype_df.columns)
        
        # Get covariate samples
        if not covariate_df.empty:
            covariate_samples = set(covariate_df.columns)
        else:
            covariate_samples = set()
        
        # Find common samples
        if covariate_samples:
            common_samples = genotype_sample_set & phenotype_samples & covariate_samples
        else:
            common_samples = genotype_sample_set & phenotype_samples
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between genotype, phenotype, and covariate data")
        
        common_samples = sorted(common_samples)
        
        # Subset data to common samples
        aligned_phenotype = phenotype_df[common_samples]
        
        if not covariate_df.empty:
            aligned_covariates = covariate_df[common_samples]
        else:
            aligned_covariates = pd.DataFrame()
        
        logger.info(f"🔧 Sample alignment: {len(common_samples)} common samples "
                   f"(genotype: {len(genotype_samples)}, "
                   f"phenotype: {len(phenotype_samples)}, "
                   f"covariates: {len(covariate_samples) if covariate_samples else 0})")
        
        return {
            'phenotype': aligned_phenotype,
            'covariates': aligned_covariates,
            'common_samples': common_samples
        }
    
    def _generate_enhanced_covariates(self, phenotype_df, existing_covariates):
        """Generate enhanced covariates including PCA and other derived features"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            logger.info("🔧 Generating enhanced covariates...")
            
            # Prepare phenotype data for PCA (transpose to samples x features)
            pheno_for_pca = phenotype_df.T.fillna(phenotype_df.T.mean())
            
            # Remove constant features
            constant_mask = pheno_for_pca.std() == 0
            if constant_mask.any():
                pheno_for_pca = pheno_for_pca.loc[:, ~constant_mask]
            
            if pheno_for_pca.shape[1] < 2:
                logger.warning("⚠️ Insufficient features for PCA")
                return existing_covariates
            
            # Standardize data
            scaler = StandardScaler()
            pheno_scaled = scaler.fit_transform(pheno_for_pca)
            
            # Determine number of PC components
            n_components = min(10, pheno_scaled.shape[1], pheno_scaled.shape[0] - 1)
            if n_components < 1:
                return existing_covariates
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pc_components = pca.fit_transform(pheno_scaled)
            
            # Create PCA covariate DataFrame
            pc_columns = [f'PC{i+1}' for i in range(n_components)]
            pc_df = pd.DataFrame(pc_components, index=pheno_for_pca.index, columns=pc_columns)
            pc_df = pc_df.T  # Transpose to covariates x samples
            
            # Combine with existing covariates
            if not existing_covariates.empty:
                enhanced_covariates = pd.concat([existing_covariates, pc_df])
            else:
                enhanced_covariates = pc_df
            
            # Log explained variance
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"✅ Enhanced covariates: {n_components} PC components "
                       f"(explained variance: {explained_variance:.3f})")
            
            return enhanced_covariates
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced covariate generation failed: {e}")
            return existing_covariates

class GWASAnalyzer:
    """Enhanced GWAS analysis with performance optimizations and comprehensive reporting"""
    
    def __init__(self, config):
        self.config = config
        self.gwas_config = config.get('gwas', {})
        self.performance_config = config.get('performance', {})
        self.parallel_processing = self.performance_config.get('parallel_gwas', True)
        self.num_workers = min(4, self.performance_config.get('num_threads', 4))
        self.data_handler = DynamicDataHandler(config)
        
    def run_gwas_analysis(self, genotype_file, results_dir):
        """Run comprehensive GWAS analysis with enhanced performance"""
        logger.info("📊 Running enhanced GWAS analysis...")
        
        try:
            # Initialize directory manager for organized directory structure
            dir_manager = get_directory_manager(results_dir)
            
            # Setup GWAS-specific directories
            gwas_dirs = get_module_directories(
                'gwas_analysis',
                [
                    'analysis_results',
                    {'analysis_results': ['gwas_results']},
                    'processed_data',
                    {'processed_data': ['genotypes']},
                    'visualization',
                    {'visualization': ['manhattan_plots', 'qq_plots']},
                    'reports',
                    {'reports': ['analysis_reports']},
                    'processed_data',
                    {'processed_data': ['quality_control']},
                    'system',
                    {'system': ['temporary_files']}
                ],
                results_dir
            )
            
            # Get genotype samples
            genotype_samples = self._get_genotype_samples(genotype_file)
            if not genotype_samples:
                raise ValueError("Could not extract genotype samples")
            
            # Prepare GWAS data with dynamic handling
            gwas_data = self.prepare_gwas_data(dir_manager, genotype_samples)
            if not gwas_data:
                raise ValueError("GWAS data preparation failed")
            
            # Run GWAS using PLINK with optimizations
            gwas_results = self.run_plink_gwas_optimized(genotype_file, gwas_data, dir_manager)
            
            # Count significant associations
            significant_count = self.count_significant_gwas(gwas_results['result_file'])
            logger.info(f"✅ Found {significant_count} significant GWAS associations")
            
            # Run comprehensive GWAS QC
            gwas_qc_results = self.run_comprehensive_gwas_qc(gwas_results['result_file'], dir_manager)
            
            # Generate GWAS-specific reports and plots
            self.generate_gwas_reports(gwas_results, gwas_qc_results, dir_manager)
            
            return {
                'result_file': gwas_results['result_file'],
                'significant_count': significant_count,
                'method': self.gwas_config.get('method', 'linear'),
                'qc_results': gwas_qc_results,
                'individual_files': gwas_results.get('individual_files', []),
                'manhattan_plot': gwas_results.get('manhattan_plot'),
                'qq_plot': gwas_results.get('qq_plot'),
                'summary_stats': gwas_results.get('summary_stats', {}),
                'phenotype_count': len(gwas_data['phenotype_cols']),
                'covariate_count': gwas_data.get('covariate_count', 0),
                'sample_count': len(gwas_data['common_samples']),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"❌ GWAS analysis failed: {e}")
            return {
                'result_file': "",
                'significant_count': 0,
                'method': self.gwas_config.get('method', 'linear'),
                'qc_results': {},
                'status': 'failed',
                'error': str(e)
            }
    
    def _get_genotype_samples(self, genotype_file):
        """Extract sample IDs from genotype file"""
        logger.info("🔍 Extracting genotype samples...")
        
        try:
            if genotype_file.endswith('.vcf.gz') or genotype_file.endswith('.vcf'):
                # Use bcftools to get samples from VCF
                cmd = f"bcftools query -l {genotype_file}"
                result = self.run_command(cmd, "Extracting VCF samples", check=False)
                if result.returncode == 0:
                    samples = [s.strip() for s in result.stdout.split('\n') if s.strip()]
                    logger.info(f"✅ Extracted {len(samples)} samples from VCF")
                    return samples
            elif genotype_file.endswith('.bed'):
                # Read PLINK fam file
                fam_file = genotype_file.replace('.bed', '.fam')
                if os.path.exists(fam_file):
                    fam_df = pd.read_csv(fam_file, sep='\s+', header=None)
                    samples = fam_df[1].tolist()  # Second column is sample ID
                    logger.info(f"✅ Extracted {len(samples)} samples from PLINK")
                    return samples
            
            logger.warning("⚠️ Could not extract genotype samples, will rely on file alignment")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting genotype samples: {e}")
            return None
    
    def prepare_gwas_data(self, dir_manager, genotype_samples=None):
        """Prepare GWAS phenotype and covariate data with enhanced handling and validation"""
        logger.info("🔧 Preparing enhanced GWAS data...")
        
        # Read GWAS phenotype file
        gwas_file = (self.config['input_files'].get('gwas_phenotype') or 
                    self.config['analysis'].get('gwas_phenotype'))
        
        if not gwas_file or not os.path.exists(gwas_file):
            logger.error(f"GWAS phenotype file not found: {gwas_file}")
            return None
        
        # Read covariates
        covariates_file = self.config['input_files'].get('covariates')
        if not covariates_file or not os.path.exists(covariates_file):
            logger.warning("⚠️ Covariate file not found or not specified, proceeding without covariates")
            covariates_file = None
        
        try:
            # Use dynamic data handler
            if genotype_samples is not None:
                aligned_data = self.data_handler.validate_and_prepare_data(
                    genotype_samples, gwas_file, covariates_file
                )
                phenotype_df = aligned_data['phenotype']
                covariates_df = aligned_data['covariates']
                common_samples = aligned_data['common_samples']
            else:
                # Load data without alignment
                phenotype_df = self.data_handler._load_phenotype_data(gwas_file)
                covariates_df = self.data_handler._load_covariate_data(covariates_file) if covariates_file else pd.DataFrame()
                common_samples = phenotype_df.columns.tolist()
            
            logger.info(f"📊 Loaded GWAS data: {phenotype_df.shape[0]} features, {len(common_samples)} samples")
            logger.info(f"📊 Covariates: {covariates_df.shape[0]} covariates" if not covariates_df.empty else "📊 No covariates used")
            
            # Identify phenotype columns
            phenotype_cols = phenotype_df.index.tolist()
            if not phenotype_cols:
                raise ValueError("No phenotype features found in GWAS file")
            
            # Apply enhanced phenotype QC
            phenotype_df = self.apply_enhanced_phenotype_qc(phenotype_df, phenotype_cols)
            
            # Create PLINK compatible files in temporary_files directory
            temp_dir = dir_manager.get_directory('system', 'temporary_files')
            plink_pheno_file = temp_dir / "gwas_phenotype.txt"
            plink_cov_file = temp_dir / "gwas_covariates.txt"
            
            # Prepare phenotype file for PLINK
            pheno_output = phenotype_df.T.reset_index()
            pheno_output.columns = ['sample_id'] + phenotype_cols
            pheno_output.to_csv(str(plink_pheno_file), sep='\t', index=False)
            logger.info(f"💾 Saved PLINK phenotype file: {plink_pheno_file}")
            
            # Prepare covariate file for PLINK if covariates exist
            if not covariates_df.empty:
                cov_output = covariates_df.T.reset_index()
                cov_output.columns = ['sample_id'] + covariates_df.index.tolist()
                cov_output.to_csv(str(plink_cov_file), sep='\t', index=False)
                logger.info(f"💾 Saved PLINK covariate file: {plink_cov_file}")
            else:
                plink_cov_file = None
                logger.info("ℹ️ No covariate file created")
            
            return {
                'phenotype_file': str(plink_pheno_file),
                'covariate_file': str(plink_cov_file) if plink_cov_file else None,
                'phenotype_cols': phenotype_cols,
                'sample_count': len(common_samples),
                'phenotype_count': len(phenotype_cols),
                'covariate_count': covariates_df.shape[0] if not covariates_df.empty else 0,
                'common_samples': common_samples,
                'original_sample_count': phenotype_df.shape[1]
            }
            
        except Exception as e:
            logger.error(f"❌ Error preparing GWAS data: {e}")
            return None
    
    def apply_enhanced_phenotype_qc(self, pheno_df, phenotype_cols):
        """Apply comprehensive quality control to GWAS phenotypes"""
        logger.info("🔧 Applying enhanced GWAS phenotype QC...")
        
        original_shape = pheno_df.shape
        samples_removed = 0
        
        # Remove samples with missing sample IDs
        pheno_df = pheno_df.dropna(subset=['sample_id'])
        samples_removed += original_shape[0] - pheno_df.shape[0]
        
        # Check for missing values in phenotypes
        missing_report = {}
        for pheno in phenotype_cols:
            missing_count = pheno_df[pheno].isna().sum()
            if missing_count > 0:
                missing_report[pheno] = missing_count
                logger.warning(f"⚠️ Phenotype {pheno} has {missing_count} missing values")
        
        # Remove extreme outliers using multiple methods
        outlier_report = {}
        for pheno in phenotype_cols:
            if pheno_df[pheno].dtype in [np.float64, np.int64]:
                # Method 1: Z-score outliers
                z_scores = np.abs((pheno_df[pheno] - pheno_df[pheno].mean()) / pheno_df[pheno].std())
                z_outliers = z_scores > 5
                
                # Method 2: IQR outliers
                Q1 = pheno_df[pheno].quantile(0.25)
                Q3 = pheno_df[pheno].quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = (pheno_df[pheno] < (Q1 - 3 * IQR)) | (pheno_df[pheno] > (Q3 + 3 * IQR))
                
                # Combine methods
                extreme_outliers = z_outliers | iqr_outliers
                if extreme_outliers.sum() > 0:
                    outlier_report[pheno] = extreme_outliers.sum()
                    pheno_df = pheno_df[~extreme_outliers]
        
        samples_removed += original_shape[0] - pheno_df.shape[0] - samples_removed
        
        logger.info(f"🔧 Phenotype QC: {original_shape[0]} → {pheno_df.shape[0]} samples "
                   f"({samples_removed} removed)")
        
        if missing_report:
            logger.info(f"🔧 Missing values: {missing_report}")
        if outlier_report:
            logger.info(f"🔧 Outliers removed: {outlier_report}")
        
        return pheno_df
    
    def run_plink_gwas_optimized(self, genotype_file, gwas_data, dir_manager):
        """Run optimized GWAS using PLINK with parallel processing"""
        logger.info("🔧 Running optimized PLINK GWAS...")
        
        method = self.gwas_config.get('method', 'linear')
        
        # Convert VCF to PLINK format if needed - store in genotypes directory
        genotypes_dir = dir_manager.get_directory('processed_data', 'genotypes')
        plink_base = genotypes_dir / "genotypes"
        
        if not os.path.exists(str(plink_base) + ".bed"):
            logger.info("🔄 Converting VCF to PLINK format...")
            self.run_command(
                f"{self.config['paths']['plink']} --vcf {genotype_file} --make-bed --out {plink_base}",
                "Converting VCF to PLINK format"
            )
        
        # Run GWAS for each phenotype with optimizations
        if self.parallel_processing and len(gwas_data['phenotype_cols']) > 1:
            return self._run_parallel_gwas(str(plink_base), gwas_data, method, dir_manager)
        else:
            return self._run_sequential_gwas(str(plink_base), gwas_data, method, dir_manager)
    
    def _run_parallel_gwas(self, plink_base, gwas_data, method, dir_manager):
        """Run GWAS for multiple phenotypes in parallel"""
        all_results = []
        individual_files = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_phenotype = {}
            
            for i, phenotype in enumerate(gwas_data['phenotype_cols']):
                future = executor.submit(
                    self._run_single_gwas,
                    plink_base, phenotype, i, method, gwas_data, dir_manager
                )
                future_to_phenotype[future] = (phenotype, i)
            
            # Collect results as they complete
            for future in as_completed(future_to_phenotype):
                phenotype, i = future_to_phenotype[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        individual_files.append(result['result_file'])
                        logger.info(f"✅ GWAS completed for {phenotype}")
                except Exception as e:
                    logger.error(f"❌ GWAS failed for {phenotype}: {e}")
        
        return self._combine_gwas_results(all_results, individual_files, dir_manager)
    
    def _run_sequential_gwas(self, plink_base, gwas_data, method, dir_manager):
        """Run GWAS for multiple phenotypes sequentially"""
        all_results = []
        individual_files = []
        
        for i, phenotype in enumerate(gwas_data['phenotype_cols']):
            logger.info(f"🔍 Running GWAS for phenotype: {phenotype} (column {i+1})")
            
            try:
                result = self._run_single_gwas(plink_base, phenotype, i, method, gwas_data, dir_manager)
                if result:
                    all_results.append(result)
                    individual_files.append(result['result_file'])
                    logger.info(f"✅ Processed GWAS results for {phenotype}: {len(result.get('results_df', pd.DataFrame()))} variants")
            except Exception as e:
                logger.error(f"❌ GWAS failed for {phenotype}: {e}")
        
        return self._combine_gwas_results(all_results, individual_files, dir_manager)
    
    def _run_single_gwas(self, plink_base, phenotype, pheno_index, method, gwas_data, dir_manager):
        """Run GWAS for a single phenotype"""
        # Use gwas_results directory for output files
        gwas_results_dir = dir_manager.get_directory('analysis_results', 'gwas_results')
        output_prefix = gwas_results_dir / f"gwas_{phenotype}"
        
        # Build optimized PLINK command
        cmd = f"{self.config['paths']['plink']} --bfile {plink_base} --pheno {gwas_data['phenotype_file']} --mpheno {pheno_index + 1}"
        
        if self.gwas_config.get('covariates', True) and gwas_data.get('covariate_file'):
            cmd += f" --covar {gwas_data['covariate_file']}"
        
        if method == 'linear':
            cmd += " --linear --ci 0.95"
        elif method == 'logistic':
            cmd += " --logistic --ci 0.95"
        else:
            raise ValueError(f"❌ Unsupported GWAS method: {method}")
        
        # Add optimized filters
        cmd += f" --maf {self.gwas_config.get('maf_threshold', 0.01)}"
        cmd += f" --geno {self.gwas_config.get('geno_threshold', 0.05)}"
        cmd += f" --hwe {self.gwas_config.get('hwe_threshold', 1e-6)}"
        cmd += f" --out {output_prefix}"
        
        # Add performance options
        if self.performance_config.get('num_threads', 1) > 1:
            cmd += f" --threads {self.performance_config['num_threads']}"
        
        # Execute command
        self.run_command(cmd, f"GWAS for {phenotype}")
        
        # Process results
        result_file = f"{output_prefix}.assoc.{method}"
        if os.path.exists(result_file):
            try:
                results_df = pd.read_csv(result_file, delim_whitespace=True)
                results_df['PHENOTYPE'] = phenotype
                
                # Add additional computed columns
                if 'P' in results_df.columns:
                    results_df['-log10p'] = -np.log10(results_df['P'])
                
                # Generate plots for this phenotype in appropriate directories
                manhattan_plot = self._create_manhattan_plot(results_df, phenotype, dir_manager)
                qq_plot = self._create_qq_plot(results_df, phenotype, dir_manager)
                
                return {
                    'results_df': results_df,
                    'result_file': result_file,
                    'manhattan_plot': manhattan_plot,
                    'qq_plot': qq_plot,
                    'phenotype': phenotype
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not process GWAS results for {phenotype}: {e}")
                return None
        else:
            logger.warning(f"⚠️ No GWAS results file created for {phenotype}")
            return None
    
    def _combine_gwas_results(self, all_results, individual_files, dir_manager):
        """Combine results from all phenotypes"""
        if all_results:
            # Combine all results
            combined_dfs = [result['results_df'] for result in all_results if 'results_df' in result]
            if combined_dfs:
                combined_results = pd.concat(combined_dfs, ignore_index=True)
                gwas_results_dir = dir_manager.get_directory('analysis_results', 'gwas_results')
                combined_file = gwas_results_dir / "gwas_combined_results.txt"
                combined_results.to_csv(str(combined_file), sep='\t', index=False)
                logger.info(f"💾 Combined GWAS results saved: {combined_file}")
                
                # Generate summary statistics
                summary_stats = self._calculate_gwas_summary_stats(combined_results)
                
                # Create overall Manhattan and QQ plots in appropriate directories
                overall_manhattan = self._create_manhattan_plot(combined_results, "combined", dir_manager)
                overall_qq = self._create_qq_plot(combined_results, "combined", dir_manager)
                
                return {
                    'result_file': str(combined_file),
                    'individual_files': individual_files,
                    'manhattan_plot': overall_manhattan,
                    'qq_plot': overall_qq,
                    'summary_stats': summary_stats,
                    'all_results': all_results
                }
        
        logger.warning("❌ No GWAS results were generated")
        return {
            'result_file': "",
            'individual_files': [],
            'manhattan_plot': "",
            'qq_plot': "",
            'summary_stats': {},
            'all_results': []
        }
    
    def _calculate_gwas_summary_stats(self, gwas_results):
        """Calculate comprehensive GWAS summary statistics"""
        summary = {
            'total_variants': len(gwas_results),
            'total_phenotypes': gwas_results['PHENOTYPE'].nunique(),
            'significant_variants': {}
        }
        
        # Calculate significance by phenotype
        for phenotype in gwas_results['PHENOTYPE'].unique():
            pheno_results = gwas_results[gwas_results['PHENOTYPE'] == phenotype]
            
            if 'P' in pheno_results.columns:
                sig_5e8 = len(pheno_results[pheno_results['P'] < 5e-8])
                sig_1e5 = len(pheno_results[pheno_results['P'] < 1e-5])
                sig_1e3 = len(pheno_results[pheno_results['P'] < 1e-3])
                
                summary['significant_variants'][phenotype] = {
                    'p_5e-8': sig_5e8,
                    'p_1e-5': sig_1e5,
                    'p_1e-3': sig_1e3
                }
        
        # Calculate genomic control lambda
        if 'P' in gwas_results.columns:
            p_values = gwas_results['P'].dropna()
            if len(p_values) > 0:
                summary['lambda_gc'] = self.calculate_lambda_gc(p_values)
        
        # Effect size statistics
        if 'BETA' in gwas_results.columns:
            betas = gwas_results['BETA'].dropna()
            summary['effect_size_stats'] = {
                'mean': float(betas.mean()),
                'std': float(betas.std()),
                'min': float(betas.min()),
                'max': float(betas.max()),
                'median': float(betas.median())
            }
        
        return summary
    
    def _create_manhattan_plot(self, gwas_results, phenotype, dir_manager):
        """Create Manhattan plot for GWAS results in appropriate directory"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8')
            
            # Prepare data for Manhattan plot
            if 'CHR' in gwas_results.columns and 'BP' in gwas_results.columns and 'P' in gwas_results.columns:
                # Sample data if too large for performance
                if len(gwas_results) > 1000000:
                    plot_data = gwas_results.sample(n=1000000, random_state=42)
                else:
                    plot_data = gwas_results.copy()
                
                # Create Manhattan plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Color points by chromosome
                colors = ['#2E86AB', '#A23B72']
                chroms = sorted(plot_data['CHR'].unique())
                
                for i, chrom in enumerate(chroms):
                    chrom_data = plot_data[plot_data['CHR'] == chrom]
                    color = colors[i % len(colors)]
                    
                    ax.scatter(chrom_data['BP'], -np.log10(chrom_data['P']), 
                              color=color, s=1, alpha=0.6)
                
                # Add significance lines
                ax.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.8, label='p = 5e-8')
                ax.axhline(y=-np.log10(1e-5), color='orange', linestyle='--', alpha=0.8, label='p = 1e-5')
                
                ax.set_xlabel('Chromosomal Position')
                ax.set_ylabel('-log10(p-value)')
                ax.set_title(f'Manhattan Plot - {phenotype}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Save in manhattan_plots directory
                manhattan_dir = dir_manager.get_directory('visualization', 'manhattan_plots')
                plot_file = manhattan_dir / f"gwas_manhattan_{phenotype}.png"
                plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
                plt.close()
                
                return str(plot_file)
            
        except Exception as e:
            logger.warning(f"Could not create Manhattan plot: {e}")
        
        return ""
    
    def _create_qq_plot(self, gwas_results, phenotype, dir_manager):
        """Create QQ plot for GWAS results in appropriate directory"""
        try:
            import matplotlib.pyplot as plt
            
            if 'P' in gwas_results.columns:
                p_values = gwas_results['P'].dropna()
                p_values = p_values[(p_values > 0) & (p_values <= 1)]
                
                if len(p_values) > 0:
                    # Sample for performance
                    if len(p_values) > 100000:
                        p_values = np.random.choice(p_values, 100000, replace=False)
                    
                    observed = -np.log10(np.sort(p_values))
                    expected = -np.log10(np.linspace(1/len(observed), 1, len(observed)))
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(expected, observed, alpha=0.6, s=10)
                    
                    # Add diagonal line
                    min_val = min(expected.min(), observed.min())
                    max_val = max(expected.max(), observed.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    ax.set_xlabel('Expected -log10(p)')
                    ax.set_ylabel('Observed -log10(p)')
                    ax.set_title(f'QQ Plot - {phenotype}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add lambda GC
                    lambda_gc = self.calculate_lambda_gc(p_values)
                    ax.text(0.05, 0.95, f'λ = {lambda_gc:.3f}', 
                           transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    # Save in qq_plots directory
                    qq_dir = dir_manager.get_directory('visualization', 'qq_plots')
                    plot_file = qq_dir / f"gwas_qq_{phenotype}.png"
                    plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    return str(plot_file)
            
        except Exception as e:
            logger.warning(f"Could not create QQ plot: {e}")
        
        return ""
    
    def count_significant_gwas(self, result_file, pval_threshold=5e-8):
        """Count significant GWAS hits with enhanced reporting"""
        if not os.path.exists(result_file):
            logger.warning(f"⚠️ GWAS results file not found: {result_file}")
            return 0
        
        try:
            results_df = pd.read_csv(result_file, sep='\t')
            
            significant_count = 0
            suggestive_count = 0
            
            if 'P' in results_df.columns:
                significant_count = len(results_df[results_df['P'] < pval_threshold])
                suggestive_count = len(results_df[(results_df['P'] >= pval_threshold) & (results_df['P'] < 1e-5)])
                
                logger.info(f"📊 GWAS significant hits: {significant_count} (p < {pval_threshold})")
                logger.info(f"📊 GWAS suggestive hits: {suggestive_count} (p < 1e-5)")
                
                # Report top hits by phenotype
                phenotypes = results_df['PHENOTYPE'].unique() if 'PHENOTYPE' in results_df.columns else ['unknown']
                for pheno in phenotypes:
                    pheno_data = results_df[results_df['PHENOTYPE'] == pheno] if 'PHENOTYPE' in results_df.columns else results_df
                    top_hits = pheno_data.nsmallest(3, 'P')
                    if not top_hits.empty:
                        logger.info(f"🏆 Top hits for {pheno}:")
                        for _, hit in top_hits.iterrows():
                            variant = hit.get('SNP', 'unknown')
                            pval = hit['P']
                            beta = hit.get('BETA', 'N/A')
                            logger.info(f"   {variant}: p = {pval:.2e}, beta = {beta}")
                    
            else:
                logger.warning("⚠️ No P-value column in GWAS results")
                
            return significant_count
            
        except Exception as e:
            logger.warning(f"⚠️ Could not count significant GWAS hits: {e}")
            return 0
    
    def run_comprehensive_gwas_qc(self, result_file, dir_manager):
        """Run comprehensive GWAS QC with enhanced metrics"""
        logger.info("🔧 Running comprehensive GWAS QC...")
        
        qc_results = {}
        
        try:
            if not os.path.exists(result_file):
                return qc_results
            
            results_df = pd.read_csv(result_file, sep='\t')
            
            if len(results_df) == 0:
                return qc_results
            
            # Calculate lambda GC
            p_values = results_df['P'].dropna()
            p_values = p_values[(p_values > 0) & (p_values <= 1)]
            
            if len(p_values) > 0:
                lambda_gc = self.calculate_lambda_gc(p_values)
                qc_results['lambda_gc'] = lambda_gc
                logger.info(f"📊 GWAS genomic control lambda: {lambda_gc:.3f}")
            
            # Check effect size distribution
            if 'BETA' in results_df.columns:
                betas = results_df['BETA'].dropna()
                qc_results['effect_size_stats'] = {
                    'mean': float(betas.mean()),
                    'std': float(betas.std()),
                    'min': float(betas.min()),
                    'max': float(betas.max()),
                    'median': float(betas.median())
                }
            
            # Check missingness
            total_variants = len(results_df)
            missing_p = results_df['P'].isna().sum()
            qc_results['missingness'] = {
                'missing_p': int(missing_p),
                'missing_p_pct': float((missing_p / total_variants) * 100)
            }
            
            # Stratified QC by phenotype
            if 'PHENOTYPE' in results_df.columns:
                pheno_qc = {}
                for pheno in results_df['PHENOTYPE'].unique():
                    pheno_data = results_df[results_df['PHENOTYPE'] == pheno]
                    pheno_p = pheno_data['P'].dropna()
                    
                    if len(pheno_p) > 0:
                        pheno_lambda = self.calculate_lambda_gc(pheno_p)
                        pheno_sig = len(pheno_data[pheno_data['P'] < 5e-8])
                        
                        pheno_qc[pheno] = {
                            'lambda_gc': pheno_lambda,
                            'significant_hits': pheno_sig,
                            'total_variants': len(pheno_data)
                        }
                
                qc_results['phenotype_stratified'] = pheno_qc
            
            # Create comprehensive QC report in quality_control directory
            self._create_gwas_qc_report(qc_results, dir_manager)
            
            logger.info("✅ Comprehensive GWAS QC completed")
            
        except Exception as e:
            logger.warning(f"⚠️ GWAS QC failed: {e}")
        
        return qc_results
    
    def calculate_lambda_gc(self, p_values):
        """Calculate genomic control lambda with robust error handling"""
        try:
            # Remove extreme p-values that can cause numerical issues
            p_values = p_values[(p_values > 1e-300) & (p_values <= 1)]
            
            if len(p_values) == 0:
                return 1.0
            
            chi_squared = stats.chi2.ppf(1 - p_values, 1)
            lambda_gc = np.median(chi_squared) / 0.4549364  # Median of chi-squared with 1 df
            
            return float(lambda_gc)
        except:
            return 1.0
    
    def _create_gwas_qc_report(self, qc_results, dir_manager):
        """Create comprehensive GWAS QC report in quality_control directory"""
        try:
            qc_dir = dir_manager.get_directory('processed_data', 'quality_control')
            report_file = qc_dir / "gwas_qc_report.txt"
            
            with open(str(report_file), 'w') as f:
                f.write("GWAS Quality Control Report\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Genomic control lambda: {qc_results.get('lambda_gc', 'N/A'):.3f}\n")
                
                if 'missingness' in qc_results:
                    missing = qc_results['missingness']
                    f.write(f"Missing P-values: {missing.get('missing_p', 'N/A')} ({missing.get('missing_p_pct', 'N/A'):.2f}%)\n")
                
                if 'effect_size_stats' in qc_results:
                    stats = qc_results['effect_size_stats']
                    f.write(f"Effect size - Mean: {stats.get('mean', 'N/A'):.4f}, "
                           f"Std: {stats.get('std', 'N/A'):.4f}, "
                           f"Range: [{stats.get('min', 'N/A'):.4f}, {stats.get('max', 'N/A'):.4f}]\n")
                
                if 'phenotype_stratified' in qc_results:
                    f.write("\nStratified by Phenotype:\n")
                    f.write("-" * 40 + "\n")
                    for pheno, pheno_qc in qc_results['phenotype_stratified'].items():
                        f.write(f"{pheno}:\n")
                        f.write(f"  Lambda: {pheno_qc.get('lambda_gc', 'N/A'):.3f}\n")
                        f.write(f"  Significant hits: {pheno_qc.get('significant_hits', 'N/A')}\n")
                        f.write(f"  Total variants: {pheno_qc.get('total_variants', 'N/A')}\n")
                
                f.write("\nQC Interpretation:\n")
                f.write("-" * 40 + "\n")
                lambda_gc = qc_results.get('lambda_gc', 1.0)
                if lambda_gc < 0.95:
                    f.write("WARNING: Lambda < 0.95 may indicate conservative test statistics\n")
                elif lambda_gc > 1.05:
                    f.write("WARNING: Lambda > 1.05 may indicate genomic inflation\n")
                else:
                    f.write("GOOD: Lambda between 0.95-1.05 suggests well-controlled statistics\n")
            
            logger.info(f"💾 GWAS QC report saved: {report_file}")
            
        except Exception as e:
            logger.warning(f"Could not create GWAS QC report: {e}")
    
    def generate_gwas_reports(self, gwas_results, qc_results, dir_manager):
        """Generate comprehensive GWAS reports and summaries in appropriate directories"""
        try:
            # Create summary statistics file in analysis_reports directory
            reports_dir = dir_manager.get_directory('reports', 'analysis_reports')
            summary_file = reports_dir / "gwas_summary_statistics.txt"
            
            with open(str(summary_file), 'w') as f:
                f.write("GWAS Summary Statistics\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total phenotypes analyzed: {len(gwas_results.get('individual_files', []))}\n")
                f.write(f"Total significant associations (p < 5e-8): {gwas_results.get('significant_count', 0)}\n")
                f.write(f"Genomic control lambda: {qc_results.get('lambda_gc', 'N/A'):.3f}\n")
                f.write(f"Analysis method: {gwas_results.get('method', 'N/A')}\n")
                
                if 'summary_stats' in gwas_results:
                    stats = gwas_results['summary_stats']
                    f.write(f"Total variants tested: {stats.get('total_variants', 'N/A')}\n")
                    
                    if 'significant_variants' in stats:
                        f.write("\nSignificant variants by phenotype:\n")
                        for pheno, sig_stats in stats['significant_variants'].items():
                            f.write(f"  {pheno}: {sig_stats.get('p_5e-8', 0)} (p<5e-8), "
                                   f"{sig_stats.get('p_1e-5', 0)} (p<1e-5)\n")
            
            logger.info(f"💾 GWAS summary statistics saved: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate GWAS reports: {e}")
    
    def run_command(self, cmd, description, check=True):
        """Run shell command with comprehensive error handling and timeout"""
        logger.info(f"Executing: {description}")
        logger.debug(f"Command: {cmd}")
        
        # Set timeout from config
        timeout = self.config.get('large_data', {}).get('command_timeout', 7200)
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                capture_output=True, 
                text=True,
                executable='/bin/bash',
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
            else:
                logger.warning(f"⚠️ {description} completed with exit code {result.returncode}")
                if result.stderr:
                    logger.warning(f"Stderr: {result.stderr[:500]}...")
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed with exit code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Command: {e.cmd}")
            if check:
                raise RuntimeError(f"Command failed: {description}") from e
            return e
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after {timeout} seconds")
            if check:
                raise RuntimeError(f"Command timed out: {description}")
            return None
        except Exception as e:
            logger.error(f"❌ {description} failed with unexpected error: {e}")
            if check:
                raise
            return None

# Modular pipeline function
def run_gwas_analysis(config, genotype_file, results_dir):
    """
    Main function for GWAS analysis module in the modular pipeline
    Returns: dict (gwas results)
    """
    try:
        logger.info("🚀 Starting enhanced GWAS analysis module...")
        
        # Initialize analyzer
        analyzer = GWASAnalyzer(config)
        
        # Run GWAS analysis
        gwas_results = analyzer.run_gwas_analysis(genotype_file, results_dir)
        
        if gwas_results and gwas_results.get('status') == 'completed':
            logger.info(f"✅ GWAS analysis completed successfully: {gwas_results.get('significant_count', 0)} significant hits")
            return gwas_results
        else:
            logger.error("❌ GWAS analysis failed")
            return {}
            
    except Exception as e:
        logger.error(f"❌ GWAS analysis module failed: {e}")
        return {}

# Backward compatibility function
def run_gwas_analysis_legacy(config, vcf_gz, results_dir):
    """Legacy function for backward compatibility"""
    return run_gwas_analysis(config, vcf_gz, results_dir)

# Maintain backward compatibility
if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Run GWAS analysis
    results = run_gwas_analysis(config, "test_data/genotypes.vcf.gz", "test_results")
    print(f"GWAS completed: {results.get('significant_count', 0)} significant hits")