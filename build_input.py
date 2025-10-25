#!/usr/bin/env python3
"""
Enhanced Pipeline script to prepare QTL input files optimized for tensorQTL:
1. Filter RNA-seq count matrix to samples present in metadata, rename to IID.
2. Build covariate.tsv with user-specified covariates from metadata.
3. Write samples_to_keep.txt and filter VCF to those samples (autosomes only, chr1–chr22).
4. Create phenotype data from metadata for GWAS analysis.
5. Generate annotations.bed with protein_coding genes only.
6. Generate chromosome_map_chr.tsv (1..22 → chr1..chr22).
7. Generate chromosome_map_nochr.txt (chr1..chr22, chrX, chrY, chrM).

Optimized for tensorQTL compatibility and performance with dynamic configuration.
"""

import pandas as pd
import os
import subprocess
import shutil
import logging
from pathlib import Path
import numpy as np
import yaml
import argparse
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BuildInput')

class InputBuilder:
    """Enhanced input builder for tensorQTL with dynamic configuration"""
    
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.setup_paths()
        self.sample_mapping = {}
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_file}")
        return config
    
    def setup_paths(self):
        """Setup input and output paths"""
        # Input files
        self.count_file = self.config['input_files']['count_file']
        self.meta_file = self.config['input_files']['meta_file']
        self.pca_file = self.config['input_files'].get('pca_file')
        self.vcf_input = self.config['input_files']['vcf_input']
        self.gtf_table = self.config['input_files']['gtf_table']
        
        # Output directory
        self.out_dir = self.config['output']['output_dir']
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Output files
        self.count_output = os.path.join(self.out_dir, "rnseq_count.tsv")
        self.covar_output = os.path.join(self.out_dir, "covariate.tsv")
        self.phenotype_output = os.path.join(self.out_dir, "phenotype.tsv")
        self.samples_list = os.path.join(self.out_dir, "samples_to_keep.txt")
        self.vcf_output = os.path.join(self.out_dir, "cohort.filtered.samples.vcf.gz")
        self.plink_output = os.path.join(self.out_dir, "cohort.filtered.samples")  # PLINK prefix
        self.bed_output = os.path.join(self.out_dir, "annotations.bed")
        self.chrom_map_chr_output = os.path.join(self.out_dir, "chromosome_map_chr.tsv")
        self.chrom_map_nochr_output = os.path.join(self.out_dir, "chromosome_map_nochr.txt")
        self.sample_mapping_output = os.path.join(self.out_dir, "sample_mapping.txt")
        self.qtl_config_output = os.path.join(self.out_dir, "qtl_analysis_config.yaml")
        
        logger.info(f"Output directory: {self.out_dir}")
    
    def run_command(self, cmd, description):
        """Run shell command with enhanced error handling"""
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ {description} completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def normalize_ids(self, s: pd.Series) -> pd.Series:
        """Normalize sample IDs for consistency"""
        return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    
    def validate_input_files(self):
        """Validate all input files exist and are accessible"""
        logger.info("Validating input files...")
        
        required_files = {
            'Count file': self.count_file,
            'Metadata file': self.meta_file,
            'VCF file': self.vcf_input,
            'GTF annotation': self.gtf_table
        }
        
        missing_files = []
        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")
                logger.error(f"❌ {file_type} not found: {file_path}")
            else:
                logger.info(f"✅ {file_type}: {file_path}")
        
        if self.pca_file and not os.path.exists(self.pca_file):
            logger.warning(f"⚠️ PCA file not found: {self.pca_file}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files))
        
        # Check for required tools
        self.bcftools_path = shutil.which("bcftools")
        self.plink_path = shutil.which("plink")
        
        if not self.bcftools_path:
            logger.warning("⚠️ bcftools not found in PATH - VCF processing may fail")
        
        logger.info("✅ All input files validated")
    
    def process_expression_data(self):
        """Process RNA-seq count data with enhanced tensorQTL compatibility"""
        logger.info("Step 1: Processing RNA-seq count data...")
        
        # Load count data
        count_df = pd.read_csv(self.count_file, sep="\t")
        meta_df = pd.read_csv(self.meta_file)
        
        # Validate count matrix structure
        logger.info(f"Count matrix shape: {count_df.shape}")
        if 'gene_id' not in count_df.columns:
            raise ValueError("Count matrix must contain 'gene_id' column")
        
        count_samples = [c for c in count_df.columns if c != "gene_id"]
        logger.info(f"Found {len(count_samples)} samples in count matrix")
        
        # Normalize IDs in metadata
        metadata_config = self.config['metadata_columns']
        rnaseq_lib_col = metadata_config['rnaseq_library']
        wgs_lib_col = metadata_config['wgs_library']
        
        for col in [rnaseq_lib_col, wgs_lib_col]:
            if col in meta_df.columns:
                meta_df[col] = self.normalize_ids(meta_df[col])
        
        # Build mapping RNASeq_Library -> IID
        meta_map = meta_df[[rnaseq_lib_col, wgs_lib_col]].dropna()
        meta_map = meta_map.rename(columns={wgs_lib_col: "IID"})
        
        rna_libs = set(meta_map[rnaseq_lib_col])
        matched_rna_libs = [c for c in count_samples if c in rna_libs]
        
        rename_dict = (
            meta_map.set_index(rnaseq_lib_col)["IID"]
            .reindex(matched_rna_libs)
            .to_dict()
        )
        rename_dict = {k: v for k, v in rename_dict.items() if isinstance(v, str) and len(v) > 0}
        
        logger.info(f"Matched {len(rename_dict)} samples between count matrix and metadata")
        
        if len(rename_dict) == 0:
            raise ValueError("No sample matches found between count matrix and metadata")
        
        # Filter count matrix and rename to IID
        count_keep_cols = ["gene_id"] + list(rename_dict.keys())
        filtered_count_df = count_df[count_keep_cols].copy()
        filtered_count_df = filtered_count_df.rename(columns=rename_dict)
        
        # Remove genes with all zeros or low expression
        original_gene_count = filtered_count_df.shape[0]
        
        if self.config['expression_processing'].get('remove_zero_genes', True):
            filtered_count_df = filtered_count_df[filtered_count_df.iloc[:, 1:].sum(axis=1) > 0]
        
        # Remove lowly expressed genes
        expression_threshold = self.config['expression_processing'].get('expression_threshold', 0.1)
        if expression_threshold > 0:
            mean_expression = filtered_count_df.iloc[:, 1:].mean(axis=1)
            filtered_count_df = filtered_count_df[mean_expression > expression_threshold]
        
        filtered_gene_count = filtered_count_df.shape[0]
        logger.info(f"Filtered genes: {filtered_gene_count}/{original_gene_count} retained "
                   f"({original_gene_count - filtered_gene_count} removed)")
        
        # Save count data
        filtered_count_df.to_csv(self.count_output, sep="\t", index=False)
        logger.info(f"✅ Saved filtered count data: {filtered_count_df.shape}")
        
        self.expression_samples = [c for c in filtered_count_df.columns if c != "gene_id"]
        self.filtered_count_df = filtered_count_df
        return filtered_count_df
    
    def build_covariates(self):
        """Build covariate matrix with user-specified columns"""
        logger.info("Step 2: Building covariate file...")
        
        meta_df = pd.read_csv(self.meta_file)
        metadata_config = self.config['metadata_columns']
        wgs_lib_col = metadata_config['wgs_library']
        
        # Normalize WGS library IDs
        meta_df[wgs_lib_col] = self.normalize_ids(meta_df[wgs_lib_col])
        
        # Load PCA data if available
        pca_data = None
        if self.pca_file and os.path.exists(self.pca_file):
            try:
                pca_cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 41)]
                pca_df = pd.read_csv(self.pca_file, sep=r"\s+", comment="#", names=pca_cols)
                pca_df["IID"] = self.normalize_ids(pca_df["IID"])
                
                # Use specified number of PCs
                pca_count = self.config['covariates'].get('pca_count', 5)
                pca_columns = [f"PC{i}" for i in range(1, pca_count + 1)]
                pca_data = pca_df[["IID"] + pca_columns]
                logger.info(f"Loaded {pca_count} PCA components")
                
            except Exception as e:
                logger.warning(f"Could not load PCA file: {e}")
                pca_data = None
        
        # Get user-specified covariate columns
        covariate_columns = self.config['covariates']['include']
        available_columns = [col for col in covariate_columns if col in meta_df.columns]
        missing_columns = set(covariate_columns) - set(available_columns)
        
        if missing_columns:
            logger.warning(f"Missing covariate columns: {missing_columns}")
        
        # Prepare base covariates from metadata
        base_covariates = meta_df[[wgs_lib_col] + available_columns].copy()
        base_covariates = base_covariates.rename(columns={wgs_lib_col: "IID"})
        
        # Merge with PCA data if available
        if pca_data is not None:
            base_covariates = base_covariates.merge(pca_data, on="IID", how="left")
            available_columns.extend(pca_columns)
        
        # Restrict to samples present in expression data
        base_covariates = base_covariates[base_covariates["IID"].isin(self.expression_samples)]
        base_covariates = base_covariates.drop_duplicates(subset=["IID"])
        
        # Handle categorical variables
        categorical_columns = self.config['covariates'].get('categorical_columns', [])
        for col in available_columns:
            if col in base_covariates.columns:
                if col in categorical_columns:
                    # Convert categorical to numeric codes
                    base_covariates[col] = pd.Categorical(base_covariates[col]).codes
                else:
                    # Ensure numeric columns are properly typed
                    try:
                        base_covariates[col] = pd.to_numeric(base_covariates[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert column '{col}' to numeric")
        
        # Handle missing values
        missing_strategy = self.config['covariates'].get('missing_value_strategy', 'drop')
        if missing_strategy == 'drop':
            base_covariates = base_covariates.dropna()
        elif missing_strategy == 'mean':
            base_covariates = base_covariates.fillna(base_covariates.mean(numeric_only=True))
        
        # Pivot: rows = covariates, columns = IID (tensorQTL format)
        covar_matrix = base_covariates.set_index("IID").T
        covar_matrix.index.name = "covariate"
        
        # Add intercept term (recommended for tensorQTL)
        if self.config['covariates'].get('add_intercept', True):
            covar_matrix.loc['intercept'] = 1.0
        
        # Remove constant covariates
        constant_covariates = covar_matrix.index[covar_matrix.nunique(axis=1) <= 1]
        if len(constant_covariates) > 0:
            logger.warning(f"Removing {len(constant_covariates)} constant covariates")
            covar_matrix = covar_matrix.drop(constant_covariates)
        
        covar_matrix.to_csv(self.covar_output, sep="\t")
        logger.info(f"✅ Saved covariates: {covar_matrix.shape[1]} samples, {covar_matrix.shape[0]} covariates")
        
        self.covariate_samples = covar_matrix.columns.tolist()
        self.covar_matrix = covar_matrix
        return covar_matrix

    def create_phenotype_data(self):
        """Create phenotype data from metadata for tensorQTL GWAS analysis"""
        logger.info("Step 3: Creating phenotype data from metadata...")
        
        meta_df = pd.read_csv(self.meta_file)
        metadata_config = self.config['metadata_columns']
        wgs_lib_col = metadata_config['wgs_library']
        
        # Normalize WGS library IDs
        meta_df[wgs_lib_col] = self.normalize_ids(meta_df[wgs_lib_col])
        
        # Get phenotype configuration
        phenotype_config = self.config.get('phenotypes', {})
        phenotype_columns = phenotype_config.get('include', [])
        
        if not phenotype_columns:
            logger.info("No phenotype columns specified in config, skipping phenotype creation")
            return None
        
        # Filter to available phenotype columns
        available_phenotypes = [col for col in phenotype_columns if col in meta_df.columns]
        missing_phenotypes = set(phenotype_columns) - set(available_phenotypes)
        
        if missing_phenotypes:
            logger.warning(f"Missing phenotype columns: {missing_phenotypes}")
        
        if not available_phenotypes:
            logger.warning("No valid phenotype columns found")
            return None
        
        # Create phenotype dataframe
        phenotype_df = meta_df[[wgs_lib_col] + available_phenotypes].copy()
        phenotype_df = phenotype_df.rename(columns={wgs_lib_col: "IID"})
        
        # Filter to common samples (if available at this stage)
        if hasattr(self, 'common_samples'):
            phenotype_df = phenotype_df[phenotype_df["IID"].isin(self.common_samples)]
        elif hasattr(self, 'expression_samples'):
            phenotype_df = phenotype_df[phenotype_df["IID"].isin(self.expression_samples)]
        
        phenotype_df = phenotype_df.drop_duplicates(subset=["IID"])
        
        # Handle data types and missing values
        for col in available_phenotypes:
            if col in phenotype_df.columns:
                # Convert categorical phenotypes to numeric codes
                if phenotype_config.get('categorical_phenotypes', {}).get(col):
                    phenotype_df[col] = pd.Categorical(phenotype_df[col]).codes
                else:
                    # Try to convert to numeric, keep as is if not possible
                    try:
                        phenotype_df[col] = pd.to_numeric(phenotype_df[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert phenotype '{col}' to numeric")
        
        # Handle missing values
        missing_strategy = phenotype_config.get('missing_value_strategy', 'drop')
        if missing_strategy == 'drop':
            phenotype_df = phenotype_df.dropna()
        elif missing_strategy == 'mean':
            phenotype_df = phenotype_df.fillna(phenotype_df.mean(numeric_only=True))
        
        # Apply normalization if specified
        normalization = phenotype_config.get('normalization', 'none')
        if normalization == 'standardize':
            for col in available_phenotypes:
                if col in phenotype_df.columns and phenotype_df[col].dtype in [np.float64, np.int64]:
                    if phenotype_df[col].std() > 0:
                        phenotype_df[col] = (phenotype_df[col] - phenotype_df[col].mean()) / phenotype_df[col].std()
        elif normalization == 'quantile':
            for col in available_phenotypes:
                if col in phenotype_df.columns and phenotype_df[col].dtype in [np.float64, np.int64]:
                    phenotype_df[col] = pd.qcut(phenotype_df[col], q=10, labels=False, duplicates='drop')
        
        # Set IID as index and transpose for tensorQTL format
        phenotype_matrix = phenotype_df.set_index("IID").T
        phenotype_matrix.index.name = "phenotype"
        
        # Remove constant phenotypes
        constant_phenotypes = phenotype_matrix.index[phenotype_matrix.nunique(axis=1) <= 1]
        if len(constant_phenotypes) > 0:
            logger.warning(f"Removing {len(constant_phenotypes)} constant phenotypes")
            phenotype_matrix = phenotype_matrix.drop(constant_phenotypes)
        
        # Save phenotype data
        phenotype_matrix.to_csv(self.phenotype_output, sep="\t")
        
        logger.info(f"✅ Phenotype data created: {phenotype_matrix.shape[1]} samples, {phenotype_matrix.shape[0]} phenotypes")
        
        self.phenotype_matrix = phenotype_matrix
        self.phenotype_samples = phenotype_matrix.columns.tolist()
        return phenotype_matrix
    
    def process_genotypes(self):
        """Process genotype data with enhanced tensorQTL optimization"""
        logger.info("Step 4: Processing genotype data...")
        
        if not self.bcftools_path:
            raise RuntimeError("bcftools is required for VCF processing")
        
        # Determine common samples across all data types
        expression_set = set(self.expression_samples)
        covariate_set = set(self.covariate_samples)
        phenotype_set = set(getattr(self, 'phenotype_samples', []))
        
        # Start with expression samples and intersect with available data
        common_samples = expression_set
        
        if len(covariate_set) > 0:
            common_samples = common_samples & covariate_set
        
        if len(phenotype_set) > 0:
            common_samples = common_samples & phenotype_set
        
        self.common_samples = list(common_samples)
        
        logger.info(f"Common samples: {len(self.common_samples)} "
                   f"(Expression: {len(self.expression_samples)}, "
                   f"Covariates: {len(self.covariate_samples)}, "
                   f"Phenotypes: {len(phenotype_set)})")
        
        if len(self.common_samples) == 0:
            raise ValueError("No common samples found across expression, covariates, and genotypes")
        
        # Write sample list for VCF filtering
        with open(self.samples_list, "w") as f:
            for sample in self.common_samples:
                f.write(f"{sample}\n")
        
        logger.info(f"Sample list created with {len(self.common_samples)} samples")
        
        # VCF filtering configuration
        vcf_config = self.config['vcf_processing']
        regions = vcf_config.get('regions', [f"chr{i}" for i in range(1, 23)])
        
        logger.info("Filtering VCF to target samples and regions...")
        
        # Build bcftools command
        cmd_filter = [self.bcftools_path, "view", "-S", self.samples_list]
        
        # Add region filtering if specified
        if regions:
            cmd_filter.extend(["-r", ",".join(regions)])
        
        # Add variant filtering options
        if vcf_config.get('biallelic_only', True):
            cmd_filter.extend(["-m2", "-M2"])  # Biallelic only
        
        variant_type = vcf_config.get('variant_type', 'snps')
        if variant_type != 'all':
            cmd_filter.extend(["-v", variant_type])
        
        # Add quality filters if specified
        if vcf_config.get('min_quality'):
            cmd_filter.extend(["-i", f"QUAL>={vcf_config['min_quality']}"])
        
        cmd_filter.extend([self.vcf_input, "-Oz", "-o", self.vcf_output])
        
        try:
            self.run_command(cmd_filter, "Filtering VCF")
            
            # Index the filtered VCF
            cmd_index = [self.bcftools_path, "index", "-t", self.vcf_output]
            self.run_command(cmd_index, "Indexing filtered VCF")
            
        except Exception as e:
            logger.error(f"VCF filtering failed: {e}")
            raise
        
        # Convert to PLINK format for better tensorQTL performance
        if self.plink_path and self.config['vcf_processing'].get('convert_to_plink', True):
            self.convert_to_plink()
        
        return self.common_samples
    
    def convert_to_plink(self):
        """Convert VCF to PLINK format for optimal tensorQTL performance"""
        logger.info("Converting VCF to PLINK format for tensorQTL...")
        
        plink_config = self.config['vcf_processing'].get('plink_settings', {})
        
        cmd_plink = [
            self.plink_path,
            "--vcf", self.vcf_output,
            "--make-bed",
            "--out", self.plink_output,
            "--memory", str(plink_config.get('memory', 4000)),
            "--threads", str(plink_config.get('threads', 2))
        ]
        
        # Add additional PLINK options
        if plink_config.get('maf_threshold'):
            cmd_plink.extend(["--maf", str(plink_config['maf_threshold'])])
        
        if plink_config.get('geno_threshold'):
            cmd_plink.extend(["--geno", str(plink_config['geno_threshold'])])
        
        if plink_config.get('mind_threshold'):
            cmd_plink.extend(["--mind", str(plink_config['mind_threshold'])])
        
        try:
            self.run_command(cmd_plink, "Converting VCF to PLINK format")
            logger.info(f"✅ PLINK files created: {self.plink_output}.bed/bim/fam")
        except Exception as e:
            logger.warning(f"PLINK conversion failed: {e}. tensorQTL can use VCF directly.")
    
    def create_annotations(self):
        """Create annotation BED file with enhanced gene filtering"""
        logger.info("Step 5: Creating annotation BED file...")
        
        annotation_config = self.config['annotations']
        
        try:
            gtf_df = pd.read_csv(self.gtf_table, sep="\t")
            
            # Filter genes based on configuration
            genes = gtf_df[gtf_df["feature"] == annotation_config.get('feature_type', 'gene')].copy()
            
            # Filter by gene type
            gene_types = annotation_config.get('gene_types', ['protein_coding'])
            if gene_types:
                genes = genes[genes["gene_type"].isin(gene_types)]
            
            # Clean gene_id (remove version suffix)
            genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
            
            # Select and sort columns
            bed_columns = ["chr", "start", "end", "gene_id", "strand"]
            additional_columns = annotation_config.get('additional_columns', [])
            bed_columns.extend([col for col in additional_columns if col in genes.columns])
            
            bed_df = genes[bed_columns].copy()
            bed_df = bed_df.sort_values(by=["chr", "start"])
            
            # Ensure chromosome names are consistent
            bed_df['chr'] = bed_df['chr'].astype(str)
            
            # Filter to genes present in count data
            count_genes = set(self.filtered_count_df['gene_id'])
            bed_df = bed_df[bed_df['gene_id'].isin(count_genes)]
            
            bed_df.to_csv(self.bed_output, sep="\t", index=False, header=True)
            logger.info(f"✅ Annotation BED created with {bed_df.shape[0]} genes")
            
            return bed_df
            
        except Exception as e:
            logger.error(f"Failed to create annotation BED: {e}")
            # Create minimal BED file as fallback
            logger.info("Creating minimal BED file from count data...")
            bed_fallback = pd.DataFrame({
                'chr': ['chr1'] * len(self.filtered_count_df),
                'start': range(1, len(self.filtered_count_df) + 1),
                'end': range(1001, len(self.filtered_count_df) + 1001),
                'gene_id': self.filtered_count_df['gene_id'],
                'strand': ['+'] * len(self.filtered_count_df)
            })
            bed_fallback.to_csv(self.bed_output, sep="\t", index=False, header=True)
            logger.warning("Created fallback BED file with placeholder positions")
            return bed_fallback
    
    def create_chromosome_maps(self):
        """Create chromosome mapping files for tensorQTL compatibility"""
        logger.info("Step 6: Creating chromosome maps...")
        
        # chromosome_map_chr.tsv (1..22 → chr1..chr22)
        chrom_map = pd.DataFrame({
            "num": list(range(1, 23)),
            "chr": [f"chr{i}" for i in range(1, 23)]
        })
        chrom_map.to_csv(self.chrom_map_chr_output, sep="\t", index=False, header=False)
        
        # chromosome_map_nochr.txt (chr1..chr22, chrX, chrY, chrM)
        chrom_map_nochr = pd.DataFrame({
            "chr": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"],
            "num": [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
        })
        chrom_map_nochr.to_csv(self.chrom_map_nochr_output, sep="\t", index=False, header=False)
        
        logger.info("✅ Chromosome maps created")
    
    def create_sample_mapping(self):
        """Create comprehensive sample mapping file"""
        logger.info("Step 7: Creating sample mapping...")
        
        sample_mapping_data = []
        for sample in self.common_samples:
            sample_info = {
                'sample_id': sample,
                'data_types': [],
                'has_covariates': sample in self.covariate_samples,
                'has_expression': sample in self.expression_samples,
                'has_genotype': True
            }
            
            # Build data types list
            if sample in self.expression_samples:
                sample_info['data_types'].append('expression')
            if sample in self.covariate_samples:
                sample_info['data_types'].append('covariates')
            if hasattr(self, 'phenotype_samples') and sample in self.phenotype_samples:
                sample_info['data_types'].append('phenotypes')
            
            sample_info['data_types'] = ','.join(sample_info['data_types'])
            sample_mapping_data.append(sample_info)
        
        sample_mapping_df = pd.DataFrame(sample_mapping_data)
        sample_mapping_df.to_csv(self.sample_mapping_output, sep="\t", index=False)
        logger.info(f"✅ Sample mapping created for {len(sample_mapping_data)} samples")
    
    def generate_qtl_config(self):
        """Generate tensorQTL configuration file based on built inputs"""
        logger.info("Step 8: Generating QTL analysis configuration...")
        
        qtl_config = {
            'results_dir': os.path.join(self.out_dir, "qtl_results"),
            'input_files': {
                'genotypes': self.vcf_output,
                'expression': self.count_output,
                'covariates': self.covar_output,
                'annotations': self.bed_output
            },
            'analysis': {
                'qtl_types': 'eqtl',
                'qtl_mode': 'cis'
            },
            'tensorqtl': {
                'cis_window': 1000000,
                'maf_threshold': 0.05,
                'fdr_threshold': 0.05,
                'num_permutations': 1000,
                'use_gpu': False
            }
        }
        
        # Add phenotype file if created
        if hasattr(self, 'phenotype_matrix') and os.path.exists(self.phenotype_output):
            qtl_config['input_files']['phenotype'] = self.phenotype_output
            qtl_config['analysis']['gwas_phenotypes'] = True
        
        # Add PLINK path if available
        if os.path.exists(self.plink_output + ".bed"):
            qtl_config['input_files']['genotypes'] = self.plink_output + ".bed"
        
        with open(self.qtl_config_output, 'w') as f:
            yaml.dump(qtl_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ QTL configuration saved: {self.qtl_config_output}")
    
    def validate_outputs(self):
        """Validate all output files and tensorQTL compatibility"""
        logger.info("Step 9: Validating outputs and tensorQTL compatibility...")
        
        # Check file existence and sizes
        output_files = [
            (self.count_output, "Expression data"),
            (self.covar_output, "Covariate data"),
            (self.phenotype_output, "Phenotype data"),
            (self.vcf_output, "Genotype VCF"),
            (self.bed_output, "Annotation BED"),
            (self.samples_list, "Sample list"),
            (self.sample_mapping_output, "Sample mapping")
        ]
        
        for file_path, description in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024**2)  # MB
                logger.info(f"  {description}: {file_size:.1f} MB")
            else:
                logger.warning(f"  {description}: File not found")
        
        # Check sample consistency
        logger.info(f"Sample counts - Expression: {len(self.expression_samples)}, "
                   f"Covariates: {len(self.covariate_samples)}, "
                   f"Phenotypes: {len(getattr(self, 'phenotype_samples', []))}, "
                   f"Common: {len(self.common_samples)}")
        
        if len(self.common_samples) < 10:
            logger.warning("Very small sample size for QTL analysis")
        
        # Check gene count
        logger.info(f"Gene count: {self.filtered_count_df.shape[0]}")
        
        if self.filtered_count_df.shape[0] < 100:
            logger.warning("Low gene count for meaningful QTL analysis")
    
    def run_pipeline(self):
        """Execute the complete input building pipeline"""
        logger.info("Starting QTL input building pipeline...")
        
        try:
            self.validate_input_files()
            self.process_expression_data()
            self.build_covariates()
            self.create_phenotype_data()  # New step for phenotype creation
            self.process_genotypes()
            self.create_annotations()
            self.create_chromosome_maps()
            self.create_sample_mapping()
            self.generate_qtl_config()
            self.validate_outputs()
            
            self.generate_summary()
            logger.info("🎉 Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def generate_summary(self):
        """Generate comprehensive pipeline summary"""
        logger.info("\n" + "="*60)
        logger.info("=== QTL INPUT BUILDING PIPELINE SUMMARY ===")
        logger.info("="*60)
        
        logger.info(f"📊 EXPRESSION DATA")
        logger.info(f"  Input: {self.count_file}")
        logger.info(f"  Output: {self.count_output}")
        logger.info(f"  Samples: {len(self.expression_samples)}")
        logger.info(f"  Genes: {self.filtered_count_df.shape[0]}")
        
        logger.info(f"📈 COVARIATES")
        logger.info(f"  Output: {self.covar_output}")
        logger.info(f"  Samples: {len(self.covariate_samples)}")
        logger.info(f"  Covariates: {self.covar_matrix.shape[0]}")
        logger.info(f"  Included: {', '.join(self.covar_matrix.index.tolist())}")
        
        logger.info(f"🏥 PHENOTYPES")
        if hasattr(self, 'phenotype_matrix'):
            logger.info(f"  Output: {self.phenotype_output}")
            logger.info(f"  Samples: {len(self.phenotype_samples)}")
            logger.info(f"  Phenotypes: {self.phenotype_matrix.shape[0]}")
            logger.info(f"  Included: {', '.join(self.phenotype_matrix.index.tolist())}")
        else:
            logger.info(f"  No phenotype data created")
        
        logger.info(f"🧬 GENOTYPES")
        logger.info(f"  Input: {self.vcf_input}")
        logger.info(f"  Output: {self.vcf_output}")
        if os.path.exists(self.plink_output + ".bed"):
            logger.info(f"  PLINK: {self.plink_output}.bed")
        logger.info(f"  Samples: {len(self.common_samples)}")
        
        logger.info(f"📝 ANNOTATIONS")
        logger.info(f"  Output: {self.bed_output}")
        
        logger.info(f"⚙️  CONFIGURATION")
        logger.info(f"  QTL config: {self.qtl_config_output}")
        
        logger.info("="*60)
        logger.info("=== TENSORQTL USAGE ===")
        logger.info(f"python qtl_analysis.py --config {self.qtl_config_output}")
        logger.info("="*60)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Build QTL input files for tensorQTL')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        builder = InputBuilder(args.config)
        builder.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()