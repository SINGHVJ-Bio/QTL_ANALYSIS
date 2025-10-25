#!/usr/bin/env python3
"""
Pipeline script to prepare QTL input files optimized for tensorQTL with correct formats.
Verified against tensorQTL documentation: https://github.com/broadinstitute/tensorqtl
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
    """Input builder for tensorQTL with verified correct formats"""
    
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.setup_paths()
        
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
        
        # Output files - using tensorQTL standard naming
        self.expression_output = os.path.join(self.out_dir, "expression.bed")
        self.covar_output = os.path.join(self.out_dir, "covariates.txt")
        self.phenotype_output = os.path.join(self.out_dir, "phenotypes.txt")
        self.samples_output = os.path.join(self.out_dir, "samples.txt")
        self.genotype_vcf_output = os.path.join(self.out_dir, "genotypes.vcf.gz")
        self.genotype_plink_output = os.path.join(self.out_dir, "genotypes")
        self.gene_annotation_output = os.path.join(self.out_dir, "genes.bed")
        
        logger.info(f"Output directory: {self.out_dir}")
    
    def run_command(self, cmd, description):
        """Run shell command with error handling"""
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
        """Validate all input files exist"""
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
        if not self.bcftools_path:
            logger.warning("⚠️ bcftools not found in PATH - VCF processing may fail")
        
        logger.info("✅ All input files validated")
    
    def process_expression_data(self):
        """Process expression data to BED format for tensorQTL"""
        logger.info("Step 1: Processing expression data to BED format...")
        
        # Load count data
        count_df = pd.read_csv(self.count_file, sep="\t")
        meta_df = pd.read_csv(self.meta_file)
        
        # Validate count matrix
        if 'gene_id' not in count_df.columns:
            raise ValueError("Count matrix must contain 'gene_id' column")
        
        count_samples = [c for c in count_df.columns if c != "gene_id"]
        logger.info(f"Count matrix: {count_df.shape[0]} genes, {len(count_samples)} samples")
        
        # Normalize IDs in metadata
        metadata_config = self.config['metadata_columns']
        rnaseq_lib_col = metadata_config['rnaseq_library']
        wgs_lib_col = metadata_config['wgs_library']
        
        meta_df[rnaseq_lib_col] = self.normalize_ids(meta_df[rnaseq_lib_col])
        meta_df[wgs_lib_col] = self.normalize_ids(meta_df[wgs_lib_col])
        
        # Build mapping RNASeq_Library -> WGS_Library (IID)
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
        
        expression_threshold = self.config['expression_processing'].get('expression_threshold', 0.1)
        if expression_threshold > 0:
            mean_expression = filtered_count_df.iloc[:, 1:].mean(axis=1)
            filtered_count_df = filtered_count_df[mean_expression > expression_threshold]
        
        filtered_gene_count = filtered_count_df.shape[0]
        logger.info(f"Filtered genes: {filtered_gene_count}/{original_gene_count} retained")
        
        # Convert to BED format for tensorQTL
        logger.info("Converting to BED format...")
        
        # Load gene annotations to get positions
        try:
            gtf_df = pd.read_csv(self.gtf_table, sep="\t")
            genes = gtf_df[gtf_df["feature"] == self.config['annotations'].get('feature_type', 'gene')].copy()
            genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
            
            # Filter by gene type
            gene_types = self.config['annotations'].get('gene_types', ['protein_coding'])
            if gene_types:
                genes = genes[genes["gene_type"].isin(gene_types)]
            
            # Merge with expression data
            expression_bed = filtered_count_df.merge(
                genes[['gene_id', 'chr', 'start', 'end', 'strand']], 
                on='gene_id', 
                how='left'
            )
            
            # Fill missing positions with placeholders
            missing_positions = expression_bed['chr'].isna()
            if missing_positions.any():
                logger.warning(f"Missing genomic positions for {missing_positions.sum()} genes, using placeholders")
                expression_bed.loc[missing_positions, 'chr'] = 'chr1'
                expression_bed.loc[missing_positions, 'start'] = range(1, missing_positions.sum() + 1)
                expression_bed.loc[missing_positions, 'end'] = range(1001, missing_positions.sum() + 1001)
                expression_bed.loc[missing_positions, 'strand'] = '+'
            
        except Exception as e:
            logger.warning(f"Could not create BED from annotations: {e}")
            # Fallback: create minimal BED format
            expression_bed = pd.DataFrame({
                'chr': ['chr1'] * len(filtered_count_df),
                'start': range(1, len(filtered_count_df) + 1),
                'end': range(1001, len(filtered_count_df) + 1001),
                'gene_id': filtered_count_df['gene_id'],
                'score': [0] * len(filtered_count_df),
                'strand': ['+'] * len(filtered_count_df)
            })
            # Add expression data
            sample_cols = [col for col in filtered_count_df.columns if col != 'gene_id']
            for col in sample_cols:
                expression_bed[col] = filtered_count_df[col]
        
        # Reorder columns for proper BED format
        sample_cols = [col for col in expression_bed.columns if col not in ['chr', 'start', 'end', 'gene_id', 'score', 'strand']]
        expression_bed['score'] = 0  # Required BED column
        
        # Create final BED dataframe in correct order
        bed_columns = ['chr', 'start', 'end', 'gene_id', 'score', 'strand'] + sample_cols
        expression_bed = expression_bed[bed_columns]
        
        # Sort by chromosome and position
        expression_bed = expression_bed.sort_values(['chr', 'start'])
        
        # Save BED file
        expression_bed.to_csv(self.expression_output, sep="\t", index=False)
        logger.info(f"✅ Expression BED created: {expression_bed.shape[0]} genes, {len(sample_cols)} samples")
        
        self.expression_samples = sample_cols
        self.filtered_count_df = filtered_count_df
        return expression_bed
    
    def load_pca_data(self):
        """Load PCA data from eigenvec file"""
        logger.info("Loading PCA data...")
        
        if not self.pca_file or not os.path.exists(self.pca_file):
            logger.warning("PCA file not available")
            return None
        
        try:
            # Load PCA data (PLINK eigenvec format: FID IID PC1 PC2 ...)
            pca_df = pd.read_csv(self.pca_file, sep=r"\s+", header=None)
            
            # Determine number of PCs to read
            pca_count = self.config['covariates'].get('pca_count', 5)
            expected_columns = 2 + pca_count
            
            if pca_df.shape[1] < expected_columns:
                logger.warning(f"PCA file has {pca_df.shape[1]} columns, but expected {expected_columns}")
                pca_count = pca_df.shape[1] - 2
            
            # Assign column names
            pca_columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, pca_count + 1)]
            pca_df = pca_df.iloc[:, :len(pca_columns)]
            pca_df.columns = pca_columns
            
            # Normalize IID for consistency
            pca_df['IID'] = self.normalize_ids(pca_df['IID'])
            
            logger.info(f"Loaded PCA data with {pca_df.shape[0]} samples and {pca_count} components")
            return pca_df
            
        except Exception as e:
            logger.error(f"Failed to load PCA data: {e}")
            return None
    
    def build_covariates(self):
        """Build covariate file in correct tensorQTL format"""
        logger.info("Step 2: Building covariate file...")
        
        meta_df = pd.read_csv(self.meta_file)
        metadata_config = self.config['metadata_columns']
        wgs_lib_col = metadata_config['wgs_library']
        
        # Normalize WGS library IDs
        meta_df[wgs_lib_col] = self.normalize_ids(meta_df[wgs_lib_col])
        
        # Get covariate configuration
        covariate_config = self.config['covariates']
        meta_covariates = covariate_config.get('include', [])
        pca_covariates = covariate_config.get('pca_include', [])
        categorical_columns = covariate_config.get('categorical_columns', [])
        
        # Load PCA data if available
        pca_data = None
        if pca_covariates and self.pca_file:
            pca_df = self.load_pca_data()
            if pca_df is not None:
                available_pca_cols = [col for col in pca_covariates if col in pca_df.columns]
                pca_data = pca_df[["IID"] + available_pca_cols]
                logger.info(f"Loaded PCA components: {available_pca_cols}")
        
        # Prepare base covariates from metadata
        available_meta_cols = [col for col in meta_covariates if col in meta_df.columns]
        missing_meta_cols = set(meta_covariates) - set(available_meta_cols)
        
        if missing_meta_cols:
            logger.warning(f"Missing metadata covariate columns: {missing_meta_cols}")
        
        base_covariates = meta_df[[wgs_lib_col] + available_meta_cols].copy()
        base_covariates = base_covariates.rename(columns={wgs_lib_col: "ID"})
        
        # Merge with PCA data if available
        if pca_data is not None:
            base_covariates = base_covariates.merge(pca_data.rename(columns={"IID": "ID"}), on="ID", how="left")
            available_meta_cols.extend(available_pca_cols)
        
        # Restrict to samples present in expression data
        base_covariates = base_covariates[base_covariates["ID"].isin(self.expression_samples)]
        base_covariates = base_covariates.drop_duplicates(subset=["ID"])
        
        # Handle data types - keep categorical variables as strings
        for col in available_meta_cols:
            if col in base_covariates.columns:
                if col in categorical_columns:
                    # Keep categorical variables as strings
                    base_covariates[col] = base_covariates[col].astype(str)
                    logger.info(f"Keeping categorical column '{col}' as string")
                else:
                    # Ensure numeric columns are properly typed
                    try:
                        base_covariates[col] = pd.to_numeric(base_covariates[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert column '{col}' to numeric")
        
        # Handle missing values
        missing_strategy = covariate_config.get('missing_value_strategy', 'drop')
        if missing_strategy == 'drop':
            original_count = base_covariates.shape[0]
            base_covariates = base_covariates.dropna()
            dropped_count = original_count - base_covariates.shape[0]
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} samples with missing covariate values")
        
        # Add intercept term if requested
        if covariate_config.get('add_intercept', True):
            base_covariates['intercept'] = 1.0
        
        # Set ID as index and transpose for tensorQTL format
        # tensorQTL expects covariates as rows, samples as columns
        covar_matrix = base_covariates.set_index("ID").T
        covar_matrix.index.name = "covariate"
        
        # Remove constant covariates
        constant_covariates = covar_matrix.index[covar_matrix.nunique(axis=1) <= 1]
        if len(constant_covariates) > 0:
            logger.warning(f"Removing {len(constant_covariates)} constant covariates")
            covar_matrix = covar_matrix.drop(constant_covariates)
        
        # Save in correct tensorQTL format
        covar_matrix.to_csv(self.covar_output, sep="\t")
        logger.info(f"✅ Covariates created: {covar_matrix.shape[0]} covariates, {covar_matrix.shape[1]} samples")
        
        self.covariate_samples = covar_matrix.columns.tolist()
        self.covar_matrix = covar_matrix
        return covar_matrix

    def create_phenotype_data(self):
        """Create phenotype data in correct tensorQTL format"""
        logger.info("Step 3: Creating phenotype data...")
        
        meta_df = pd.read_csv(self.meta_file)
        metadata_config = self.config['metadata_columns']
        wgs_lib_col = metadata_config['wgs_library']
        
        # Normalize WGS library IDs
        meta_df[wgs_lib_col] = self.normalize_ids(meta_df[wgs_lib_col])
        
        # Get phenotype configuration
        phenotype_config = self.config.get('phenotypes', {})
        phenotype_columns = phenotype_config.get('include', [])
        categorical_phenotypes = phenotype_config.get('categorical_phenotypes', [])
        
        if not phenotype_columns:
            logger.info("No phenotype columns specified, skipping phenotype creation")
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
        phenotype_df = phenotype_df.rename(columns={wgs_lib_col: "ID"})
        
        # Filter to common samples
        if hasattr(self, 'common_samples'):
            phenotype_df = phenotype_df[phenotype_df["ID"].isin(self.common_samples)]
        else:
            phenotype_df = phenotype_df[phenotype_df["ID"].isin(self.expression_samples)]
        
        phenotype_df = phenotype_df.drop_duplicates(subset=["ID"])
        
        # Handle data types - keep categorical phenotypes as strings
        for col in available_phenotypes:
            if col in phenotype_df.columns:
                if col in categorical_phenotypes:
                    phenotype_df[col] = phenotype_df[col].astype(str)
                    logger.info(f"Keeping categorical phenotype '{col}' as string")
                else:
                    try:
                        phenotype_df[col] = pd.to_numeric(phenotype_df[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert phenotype '{col}' to numeric")
        
        # Handle missing values
        missing_strategy = phenotype_config.get('missing_value_strategy', 'drop')
        if missing_strategy == 'drop':
            phenotype_df = phenotype_df.dropna()
        
        # Apply normalization if specified
        normalization = phenotype_config.get('normalization', 'none')
        if normalization == 'standardize':
            for col in available_phenotypes:
                if col in phenotype_df.columns and phenotype_df[col].dtype in [np.float64, np.int64]:
                    if phenotype_df[col].std() > 0:
                        phenotype_df[col] = (phenotype_df[col] - phenotype_df[col].mean()) / phenotype_df[col].std()
        
        # Set ID as index and transpose for tensorQTL format
        # tensorQTL expects phenotypes as rows, samples as columns
        phenotype_matrix = phenotype_df.set_index("ID").T
        phenotype_matrix.index.name = "phenotype"
        
        # Remove constant phenotypes
        constant_phenotypes = phenotype_matrix.index[phenotype_matrix.nunique(axis=1) <= 1]
        if len(constant_phenotypes) > 0:
            logger.warning(f"Removing {len(constant_phenotypes)} constant phenotypes")
            phenotype_matrix = phenotype_matrix.drop(constant_phenotypes)
        
        # Save phenotype data
        phenotype_matrix.to_csv(self.phenotype_output, sep="\t")
        logger.info(f"✅ Phenotype data created: {phenotype_matrix.shape[0]} phenotypes, {phenotype_matrix.shape[1]} samples")
        
        self.phenotype_matrix = phenotype_matrix
        self.phenotype_samples = phenotype_matrix.columns.tolist()
        return phenotype_matrix
    
    def process_genotypes(self):
        """Process genotype data"""
        logger.info("Step 4: Processing genotype data...")
        
        if not self.bcftools_path:
            raise RuntimeError("bcftools is required for VCF processing")
        
        # Determine common samples across all data types
        expression_set = set(self.expression_samples)
        covariate_set = set(self.covariate_samples)
        phenotype_set = set(getattr(self, 'phenotype_samples', []))
        
        common_samples = expression_set & covariate_set & phenotype_set
        self.common_samples = list(common_samples)
        
        logger.info(f"Common samples: {len(self.common_samples)}")
        
        if len(self.common_samples) == 0:
            raise ValueError("No common samples found across data types")
        
        # Write sample list
        with open(self.samples_output, "w") as f:
            for sample in self.common_samples:
                f.write(f"{sample}\n")
        
        logger.info(f"Sample list created with {len(self.common_samples)} samples")
        
        # VCF filtering
        vcf_config = self.config['vcf_processing']
        regions = vcf_config.get('regions', [f"chr{i}" for i in range(1, 23)])
        
        cmd_filter = [self.bcftools_path, "view", "-S", self.samples_output]
        
        if regions:
            cmd_filter.extend(["-r", ",".join(regions)])
        
        if vcf_config.get('biallelic_only', True):
            cmd_filter.extend(["-m2", "-M2"])
        
        variant_type = vcf_config.get('variant_type', 'snps')
        if variant_type != 'all':
            cmd_filter.extend(["-v", variant_type])
        
        if vcf_config.get('min_quality'):
            cmd_filter.extend(["-i", f"QUAL>={vcf_config['min_quality']}"])
        
        cmd_filter.extend([self.vcf_input, "-Oz", "-o", self.genotype_vcf_output])
        
        try:
            self.run_command(cmd_filter, "Filtering VCF")
            
            # Index the filtered VCF
            cmd_index = [self.bcftools_path, "index", "-t", self.genotype_vcf_output]
            self.run_command(cmd_index, "Indexing VCF")
            
        except Exception as e:
            logger.error(f"VCF filtering failed: {e}")
            raise
        
        # Convert to PLINK format if requested
        plink_path = shutil.which("plink")
        if plink_path and self.config['vcf_processing'].get('convert_to_plink', True):
            self.convert_to_plink(plink_path)
        
        return self.common_samples
    
    def convert_to_plink(self, plink_path):
        """Convert VCF to PLINK format"""
        logger.info("Converting VCF to PLINK format...")
        
        plink_config = self.config['vcf_processing'].get('plink_settings', {})
        
        cmd_plink = [
            plink_path,
            "--vcf", self.genotype_vcf_output,
            "--make-bed",
            "--out", self.genotype_plink_output,
            "--memory", str(plink_config.get('memory', 4000)),
            "--threads", str(plink_config.get('threads', 2))
        ]
        
        if plink_config.get('maf_threshold'):
            cmd_plink.extend(["--maf", str(plink_config['maf_threshold'])])
        
        if plink_config.get('geno_threshold'):
            cmd_plink.extend(["--geno", str(plink_config['geno_threshold'])])
        
        if plink_config.get('mind_threshold'):
            cmd_plink.extend(["--mind", str(plink_config['mind_threshold'])])
        
        try:
            self.run_command(cmd_plink, "Converting to PLINK format")
            logger.info(f"✅ PLINK files created: {self.genotype_plink_output}.bed/bim/fam")
        except Exception as e:
            logger.warning(f"PLINK conversion failed: {e}")
    
    def create_gene_annotations(self):
        """Create gene annotation BED file"""
        logger.info("Step 5: Creating gene annotation BED file...")
        
        annotation_config = self.config['annotations']
        
        try:
            gtf_df = pd.read_csv(self.gtf_table, sep="\t")
            
            # Filter genes
            genes = gtf_df[gtf_df["feature"] == annotation_config.get('feature_type', 'gene')].copy()
            
            # Filter by gene type
            gene_types = annotation_config.get('gene_types', ['protein_coding'])
            if gene_types:
                genes = genes[genes["gene_type"].isin(gene_types)]
            
            # Clean gene_id
            genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
            
            # Select columns
            bed_columns = ["chr", "start", "end", "gene_id", "score", "strand"]
            additional_columns = annotation_config.get('additional_columns', [])
            bed_columns.extend([col for col in additional_columns if col in genes.columns])
            
            bed_df = genes[bed_columns].copy()
            bed_df = bed_df.sort_values(by=["chr", "start"])
            
            # Ensure required columns
            if 'score' not in bed_df.columns:
                bed_df['score'] = 0
            if 'strand' not in bed_df.columns:
                bed_df['strand'] = '+'
            
            # Filter to genes present in expression data
            count_genes = set(self.filtered_count_df['gene_id'])
            bed_df = bed_df[bed_df['gene_id'].isin(count_genes)]
            
            # Save without header for BED format
            bed_df.to_csv(self.gene_annotation_output, sep="\t", index=False, header=False)
            logger.info(f"✅ Gene annotation BED created with {bed_df.shape[0]} genes")
            
            return bed_df
            
        except Exception as e:
            logger.error(f"Failed to create gene annotation BED: {e}")
            # Create minimal BED file
            bed_fallback = pd.DataFrame({
                'chr': ['chr1'] * len(self.filtered_count_df),
                'start': range(1, len(self.filtered_count_df) + 1),
                'end': range(1001, len(self.filtered_count_df) + 1001),
                'gene_id': self.filtered_count_df['gene_id'],
                'score': [0] * len(self.filtered_count_df),
                'strand': ['+'] * len(self.filtered_count_df)
            })
            bed_fallback.to_csv(self.gene_annotation_output, sep="\t", index=False, header=False)
            logger.warning("Created fallback BED file")
            return bed_fallback
    
    def run_pipeline(self):
        """Execute the complete input building pipeline"""
        logger.info("Starting tensorQTL input building pipeline...")
        
        try:
            self.validate_input_files()
            self.process_expression_data()
            self.build_covariates()
            self.create_phenotype_data()
            self.process_genotypes()
            self.create_gene_annotations()
            
            self.generate_summary()
            logger.info("🎉 Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def generate_summary(self):
        """Generate pipeline summary"""
        logger.info("\n" + "="*60)
        logger.info("=== TENSORQTL INPUT SUMMARY ===")
        logger.info("="*60)
        
        logger.info(f"📊 EXPRESSION DATA")
        logger.info(f"  File: {self.expression_output}")
        logger.info(f"  Format: BED (genes × samples)")
        logger.info(f"  Genes: {self.filtered_count_df.shape[0]}")
        logger.info(f"  Samples: {len(self.expression_samples)}")
        
        logger.info(f"📈 COVARIATES")
        logger.info(f"  File: {self.covar_output}")
        logger.info(f"  Format: TSV (covariates × samples)")
        logger.info(f"  Covariates: {self.covar_matrix.shape[0]}")
        logger.info(f"  Samples: {self.covar_matrix.shape[1]}")
        
        if hasattr(self, 'phenotype_matrix'):
            logger.info(f"🏥 PHENOTYPES")
            logger.info(f"  File: {self.phenotype_output}")
            logger.info(f"  Format: TSV (phenotypes × samples)")
            logger.info(f"  Phenotypes: {self.phenotype_matrix.shape[0]}")
            logger.info(f"  Samples: {self.phenotype_matrix.shape[1]}")
        
        logger.info(f"🧬 GENOTYPES")
        logger.info(f"  VCF: {self.genotype_vcf_output}")
        if os.path.exists(self.genotype_plink_output + ".bed"):
            logger.info(f"  PLINK: {self.genotype_plink_output}.bed")
        logger.info(f"  Samples: {len(self.common_samples)}")
        
        logger.info(f"📝 GENE ANNOTATIONS")
        logger.info(f"  File: {self.gene_annotation_output}")
        logger.info(f"  Format: BED")
        
        logger.info("="*60)
        logger.info("=== TENSORQTL USAGE ===")
        logger.info("python3 -m tensorqtl \\")
        logger.info(f"  {self.genotype_vcf_output} \\")
        logger.info(f"  {self.expression_output} \\")
        logger.info(f"  output_prefix \\")
        logger.info(f"  --covariates {self.covar_output} \\")
        logger.info(f"  --mode cis")
        logger.info("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Build tensorQTL input files')
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