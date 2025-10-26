#!/usr/bin/env python3
"""
Final optimized pipeline to prepare QTL input files for tensorQTL
Generates both BED and TSV expression files, BED + TSV phenotype files, and annotations BED
"""

import pandas as pd
import os
import subprocess
import logging
from pathlib import Path
import yaml
import argparse
import sys
import tempfile
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BuildInput')

class FinalInputBuilder:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.setup_paths()
        self.setup_parallel_config()
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
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
        self.expression_bed_output = os.path.join(self.out_dir, "expression.bed")
        self.expression_tsv_output = os.path.join(self.out_dir, "expression.tsv")
        self.covar_output = os.path.join(self.out_dir, "covariates.txt")
        self.phenotype_bed_output = os.path.join(self.out_dir, "phenotypes.bed")
        self.phenotype_tsv_output = os.path.join(self.out_dir, "phenotype_data.tsv")
        self.samples_output = os.path.join(self.out_dir, "samples.txt")
        self.genotype_vcf_output = os.path.join(self.out_dir, "genotypes.vcf.gz")
        self.mapping_output = os.path.join(self.out_dir, "sample_mapping.tsv")
        self.annotation_bed_output = os.path.join(self.out_dir, "annotations.bed")
        
        logger.info(f"Output directory: {self.out_dir}")
    
    def setup_parallel_config(self):
        """Setup parallel processing configuration"""
        vcf_config = self.config['vcf_processing']
        
        # Determine number of threads
        self.threads = vcf_config.get('threads', 1)
        if self.threads == 0:
            self.threads = multiprocessing.cpu_count()
        
        self.memory = vcf_config.get('memory', 4000)
        
        logger.info(f"Parallel processing: {self.threads} threads, {self.memory}MB memory")
    
    def run_command(self, cmd, description):
        """Run shell command with error handling and timing"""
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        import time
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            logger.info(f"✅ {description} completed in {elapsed_time:.1f}s")
            return result
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ {description} failed after {elapsed_time:.1f}s: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def normalize_ids(self, s):
        """Normalize sample IDs for consistency"""
        return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    
    def validate_input_files(self):
        """Validate all input files exist"""
        logger.info("Validating input files...")
        
        files_to_check = {
            'Count file': self.count_file,
            'Metadata file': self.meta_file,
            'VCF file': self.vcf_input,
            'GTF annotation': self.gtf_table
        }
        
        for file_type, file_path in files_to_check.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_type} not found: {file_path}")
            logger.info(f"✅ {file_type}: {file_path}")
        
        if self.pca_file and not os.path.exists(self.pca_file):
            logger.warning(f"⚠️ PCA file not found: {self.pca_file}")
        
        logger.info("✅ All input files validated")
    
    def create_annotation_bed(self):
        """Create annotations.bed file with gene information"""
        logger.info("Creating annotations.bed file...")
        
        try:
            # Load GTF data
            gtf_df = pd.read_csv(self.gtf_table, sep="\t")
            
            # Get annotation configuration
            annotation_config = self.config['annotations']
            feature_type = annotation_config.get('feature_type', 'gene')
            gene_types = annotation_config.get('gene_types', ['protein_coding'])
            additional_columns = annotation_config.get('additional_columns', ['gene_name', 'gene_type'])
            
            # Filter genes by feature type and gene type
            genes = gtf_df[gtf_df["feature"] == feature_type].copy()
            
            if gene_types:
                genes = genes[genes["gene_type"].isin(gene_types)]
            
            # Clean gene_id (remove version numbers)
            genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
            
            # Select required columns
            required_columns = ["chr", "start", "end", "gene_id", "strand"]
            available_columns = required_columns + [col for col in additional_columns if col in genes.columns]
            
            # Create annotations BED
            annotation_bed = genes[available_columns].copy()
            
            # Ensure proper data types
            annotation_bed['start'] = annotation_bed['start'].astype(int)
            annotation_bed['end'] = annotation_bed['end'].astype(int)
            
            # Sort by chromosome and position
            annotation_bed = annotation_bed.sort_values(['chr', 'start'])
            
            # Save annotations BED file
            annotation_bed.to_csv(self.annotation_bed_output, sep="\t", index=False)
            logger.info(f"✅ Annotations BED created: {annotation_bed.shape[0]} genes")
            
            self.annotation_bed = annotation_bed
            return annotation_bed
            
        except Exception as e:
            logger.error(f"Failed to create annotations BED: {e}")
            raise
    
    def process_expression_data(self):
        """Process expression data - creates both BED and TSV files"""
        logger.info("Processing expression data...")
        
        # Load files in parallel
        def load_count_data():
            return pd.read_csv(self.count_file, sep="\t")
        
        def load_meta_data():
            return pd.read_csv(self.meta_file)
        
        def load_gtf_data():
            return pd.read_csv(self.gtf_table, sep="\t")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            count_future = executor.submit(load_count_data)
            meta_future = executor.submit(load_meta_data)
            gtf_future = executor.submit(load_gtf_data)
            
            count_df = count_future.result()
            meta_df = meta_future.result()
            gtf_df = gtf_future.result()
        
        if 'gene_id' not in count_df.columns:
            raise ValueError("Count matrix must contain 'gene_id' column")
        
        count_samples = [c for c in count_df.columns if c != "gene_id"]
        logger.info(f"Count matrix: {count_df.shape[0]} genes, {len(count_samples)} samples")
        
        # Build mapping RNASeq_Library -> WGS_Library
        metadata_config = self.config['metadata_columns']
        rnaseq_col = metadata_config['rnaseq_library']
        wgs_col = metadata_config['wgs_library']
        
        # Normalize IDs
        meta_df[rnaseq_col] = self.normalize_ids(meta_df[rnaseq_col])
        meta_df[wgs_col] = self.normalize_ids(meta_df[wgs_col])
        
        meta_map = meta_df[[rnaseq_col, wgs_col]].dropna()
        meta_map = meta_map.rename(columns={wgs_col: "IID"})
        
        # Create mapping dictionary
        rename_dict = meta_map.set_index(rnaseq_col)["IID"].to_dict()
        
        # Filter to valid mappings
        valid_samples = [c for c in count_samples if c in rename_dict and pd.notna(rename_dict[c]) and len(str(rename_dict[c])) > 0]
        rename_dict = {k: v for k, v in rename_dict.items() if k in valid_samples}
        
        logger.info(f"Matched {len(rename_dict)} samples between count matrix and metadata")
        
        if len(rename_dict) == 0:
            raise ValueError("No sample matches found between count matrix and metadata")
        
        # Filter count matrix and rename to IID
        count_keep_cols = ["gene_id"] + list(rename_dict.keys())
        filtered_count_df = count_df[count_keep_cols].copy()
        filtered_count_df = filtered_count_df.rename(columns=rename_dict)
        
        # Save TSV file (gene_id + samples only, no genomic info)
        filtered_count_df.to_csv(self.expression_tsv_output, sep="\t", index=False)
        logger.info(f"✅ Expression TSV created: {self.expression_tsv_output}")
        
        # Create BED format with genomic coordinates
        logger.info("Creating BED format expression file...")
        
        # Filter genes
        genes = gtf_df[gtf_df["feature"] == "gene"].copy()
        genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        
        # Merge with expression data
        expression_bed = filtered_count_df.merge(
            genes[['gene_id', 'chr', 'start', 'end', 'strand']], 
            on='gene_id', 
            how='left'
        )
        
        # Handle missing positions
        missing_positions = expression_bed['chr'].isna()
        if missing_positions.any():
            missing_count = missing_positions.sum()
            logger.warning(f"Missing genomic positions for {missing_count} genes")
            
            expression_bed.loc[missing_positions, 'chr'] = 'chr1'
            expression_bed.loc[missing_positions, 'start'] = range(1, missing_count + 1)
            expression_bed.loc[missing_positions, 'end'] = range(1001, missing_count + 1001)
            expression_bed.loc[missing_positions, 'strand'] = '+'
        
        # Prepare BED columns
        sample_cols = [col for col in filtered_count_df.columns if col != 'gene_id']
        expression_bed['score'] = 0
        
        # Ensure proper data types
        expression_bed['start'] = expression_bed['start'].astype(int)
        expression_bed['end'] = expression_bed['end'].astype(int)
        
        # Create final BED dataframe
        bed_columns = ['chr', 'start', 'end', 'gene_id', 'score', 'strand'] + sample_cols
        expression_bed = expression_bed[bed_columns].sort_values(['chr', 'start'])
        
        # Save BED file
        expression_bed.to_csv(self.expression_bed_output, sep="\t", index=False)
        logger.info(f"✅ Expression BED created: {expression_bed.shape[0]} genes, {len(sample_cols)} samples")
        
        self.expression_samples = sample_cols
        self.expression_bed = expression_bed
        self.filtered_count_df = filtered_count_df
        return expression_bed
    
    def load_pca_data(self):
        """Load PCA data from eigenvec file"""
        if not self.pca_file or not os.path.exists(self.pca_file):
            return None
        
        try:
            pca_df = pd.read_csv(self.pca_file, sep=r"\s+", header=None, engine='python')
            pca_count = self.config['covariates'].get('pca_count', 5)
            
            # Assign column names
            pca_columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, pca_count + 1)]
            pca_df = pca_df.iloc[:, :len(pca_columns)]
            pca_df.columns = pca_columns
            
            pca_df['IID'] = self.normalize_ids(pca_df['IID'])
            logger.info(f"Loaded PCA data with {pca_df.shape[0]} samples")
            return pca_df
            
        except Exception as e:
            logger.error(f"Failed to load PCA data: {e}")
            return None
    
    def build_covariates(self):
        """Build covariate file with correct samples"""
        logger.info("Building covariate file...")
        
        meta_df = pd.read_csv(self.meta_file, low_memory=False)
        metadata_config = self.config['metadata_columns']
        wgs_col = metadata_config['wgs_library']
        
        # Normalize IDs
        meta_df[wgs_col] = self.normalize_ids(meta_df[wgs_col])
        
        # Get covariate columns
        covariate_config = self.config['covariates']
        meta_covariates = covariate_config.get('include', [])
        pca_covariates = covariate_config.get('pca_include', [])
        
        # Load PCA data
        pca_data = None
        if pca_covariates and self.pca_file:
            pca_df = self.load_pca_data()
            if pca_df is not None:
                available_pca_cols = [col for col in pca_covariates if col in pca_df.columns]
                pca_data = pca_df[["IID"] + available_pca_cols]
        
        # Prepare base covariates
        available_meta_cols = [col for col in meta_covariates if col in meta_df.columns]
        base_covariates = meta_df[[wgs_col] + available_meta_cols].copy()
        base_covariates = base_covariates.rename(columns={wgs_col: "ID"})
        
        # Merge with PCA data
        if pca_data is not None:
            base_covariates = base_covariates.merge(pca_data.rename(columns={"IID": "ID"}), on="ID", how="left")
        
        # Filter to expression samples and remove duplicates
        base_covariates = base_covariates[base_covariates["ID"].isin(self.expression_samples)]
        base_covariates = base_covariates.drop_duplicates(subset=["ID"])
        
        # Handle missing values by dropping
        original_count = base_covariates.shape[0]
        base_covariates = base_covariates.dropna()
        if original_count - base_covariates.shape[0] > 0:
            logger.warning(f"Dropped {original_count - base_covariates.shape[0]} samples with missing covariate values")
        
        # Add intercept
        base_covariates['intercept'] = 1.0
        
        # Transpose for tensorQTL format
        covar_matrix = base_covariates.set_index("ID").T
        covar_matrix.index.name = "covariate"
        
        # Save covariate file
        covar_matrix.to_csv(self.covar_output, sep="\t")
        logger.info(f"✅ Covariates created: {covar_matrix.shape[0]} covariates, {covar_matrix.shape[1]} samples")
        
        self.covariate_samples = covar_matrix.columns.tolist()
        self.covar_matrix = covar_matrix
        return covar_matrix

    def create_phenotype_data(self):
        """Create phenotype data in BED format for tensorQTL and TSV format for analysis"""
        logger.info("Creating phenotype data...")
        
        meta_df = pd.read_csv(self.meta_file, low_memory=False)
        metadata_config = self.config['metadata_columns']
        wgs_col = metadata_config['wgs_library']
        
        meta_df[wgs_col] = self.normalize_ids(meta_df[wgs_col])
        
        # Get phenotype columns
        phenotype_config = self.config.get('phenotypes', {})
        phenotype_columns = phenotype_config.get('include', [])
        
        if not phenotype_columns:
            logger.info("No phenotype columns specified, skipping")
            return None
        
        # Filter to available phenotypes
        available_phenotypes = [col for col in phenotype_columns if col in meta_df.columns]
        
        if not available_phenotypes:
            logger.warning("No valid phenotype columns found")
            return None
        
        # Create phenotype dataframe
        phenotype_df = meta_df[[wgs_col] + available_phenotypes].copy()
        phenotype_df = phenotype_df.rename(columns={wgs_col: "ID"})
        
        # Filter to common samples and remove duplicates
        common_samples = set(self.expression_samples) & set(self.covariate_samples)
        phenotype_df = phenotype_df[phenotype_df["ID"].isin(common_samples)]
        phenotype_df = phenotype_df.drop_duplicates(subset=["ID"])
        
        # Handle missing values
        phenotype_df = phenotype_df.dropna()
        
        # Save TSV format (samples as rows, phenotypes as columns)
        phenotype_tsv = phenotype_df.set_index("ID")
        phenotype_tsv.to_csv(self.phenotype_tsv_output, sep="\t")
        logger.info(f"✅ Phenotype TSV created: {phenotype_tsv.shape[1]} phenotypes, {phenotype_tsv.shape[0]} samples")
        
        # Transpose to get phenotypes as rows, samples as columns for BED format
        phenotype_matrix = phenotype_df.set_index("ID").T
        phenotype_matrix.index.name = "phenotype_id"
        
        # Convert to BED format for tensorQTL
        bed_data = []
        start_pos = 1000
        
        for i, phenotype_id in enumerate(phenotype_matrix.index):
            bed_row = {
                'chr': 'chr1',
                'start': start_pos + (i * 1000),
                'end': start_pos + (i * 1000) + 1000,
                'phenotype_id': phenotype_id,
                'score': 0,
                'strand': '+'
            }
            # Add sample values
            for sample in phenotype_matrix.columns:
                bed_row[sample] = phenotype_matrix.loc[phenotype_id, sample]
            
            bed_data.append(bed_row)
        
        # Create BED dataframe
        bed_columns = ['chr', 'start', 'end', 'phenotype_id', 'score', 'strand'] + phenotype_matrix.columns.tolist()
        phenotype_bed = pd.DataFrame(bed_data, columns=bed_columns)
        
        # Ensure proper data types
        phenotype_bed['start'] = phenotype_bed['start'].astype(int)
        phenotype_bed['end'] = phenotype_bed['end'].astype(int)
        
        # Save in BED format
        phenotype_bed.to_csv(self.phenotype_bed_output, sep="\t", index=False)
        logger.info(f"✅ Phenotype BED created: {phenotype_bed.shape[0]} phenotypes, {phenotype_bed.shape[1]-6} samples")
        
        self.phenotype_samples = phenotype_matrix.columns.tolist()
        self.phenotype_bed = phenotype_bed
        self.phenotype_tsv = phenotype_tsv
        return phenotype_bed
    
    def create_sample_mapping(self):
        """Create sample mapping file for cross-checking"""
        logger.info("Creating sample mapping file...")
        
        meta_df = pd.read_csv(self.meta_file, low_memory=False)
        metadata_config = self.config['metadata_columns']
        rnaseq_col = metadata_config['rnaseq_library']
        wgs_col = metadata_config['wgs_library']
        
        # Normalize IDs
        meta_df[rnaseq_col] = self.normalize_ids(meta_df[rnaseq_col])
        meta_df[wgs_col] = self.normalize_ids(meta_df[wgs_col])
        
        # Get common samples across all data types
        common_samples = set(self.expression_samples) & set(self.covariate_samples)
        if hasattr(self, 'phenotype_samples'):
            common_samples &= set(self.phenotype_samples)
        
        # Create mapping file
        mapping_df = meta_df[meta_df[wgs_col].isin(common_samples)][[rnaseq_col, wgs_col]].copy()
        mapping_df = mapping_df.rename(columns={rnaseq_col: "RNASeq_Library", wgs_col: "WGS_Library"})
        
        # Add data presence flags
        mapping_df['in_expression'] = True
        mapping_df['in_covariates'] = mapping_df['WGS_Library'].isin(self.covariate_samples)
        if hasattr(self, 'phenotype_samples'):
            mapping_df['in_phenotypes'] = mapping_df['WGS_Library'].isin(self.phenotype_samples)
        
        mapping_df.to_csv(self.mapping_output, sep="\t", index=False)
        logger.info(f"✅ Sample mapping created: {mapping_df.shape[0]} samples")
        
        return mapping_df
    
    def process_genotypes_parallel(self):
        """Process genotype data with multi-threading for maximum speed"""
        logger.info("Processing genotype data with parallel processing...")
        
        # Find bcftools
        bcftools_path = subprocess.run(["which", "bcftools"], capture_output=True, text=True)
        if bcftools_path.returncode != 0:
            raise RuntimeError("bcftools not found in PATH")
        
        bcftools_path = bcftools_path.stdout.strip()
        
        # Get common samples
        common_samples = set(self.expression_samples) & set(self.covariate_samples)
        if hasattr(self, 'phenotype_samples'):
            common_samples &= set(self.phenotype_samples)
        
        self.common_samples = list(common_samples)
        logger.info(f"Common samples: {len(self.common_samples)}")
        
        if len(self.common_samples) == 0:
            raise ValueError("No common samples found across data types")
        
        # Write sample list
        with open(self.samples_output, "w") as f:
            for sample in self.common_samples:
                f.write(f"{sample}\n")
        
        # VCF processing configuration
        vcf_config = self.config['vcf_processing']
        regions = vcf_config.get('regions', [f"chr{i}" for i in range(1, 23)])
        biallelic_only = vcf_config.get('biallelic_only', True)
        variant_type = vcf_config.get('variant_type', 'snps')
        
        logger.info(f"VCF processing: biallelic_only={biallelic_only}, variant_type={variant_type}")
        logger.info(f"Using {self.threads} threads for VCF processing")
        
        if biallelic_only:
            # Simple case: only keep biallelic SNPs with parallel processing
            # Note: --threads flag goes after the subcommand, not as global option
            cmd_filter = [
                bcftools_path,
                "view", 
                "--threads", str(self.threads),  # Threads flag after subcommand
                "-S", self.samples_output,      # Filter samples
                "-r", ",".join(regions),        # Filter regions
                "-m2", "-M2",                   # Biallelic only
                "-v", variant_type,             # SNP/indel filter
                self.vcf_input, 
                "-Oz", "-o", self.genotype_vcf_output
            ]
            self.run_command(cmd_filter, "Filtering VCF for biallelic sites (parallel)")
            
        else:
            # Complex case: handle multiallelic sites by splitting them with parallel processing
            logger.info("Processing multiallelic sites with parallel splitting...")
            
            # Use temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_filtered_vcf = os.path.join(temp_dir, "temp_filtered.vcf.gz")
                
                # First filter: samples, regions, variant type
                cmd_first_filter = [
                    bcftools_path,
                    "view",
                    "--threads", str(self.threads),  # Threads flag after subcommand
                    "-S", self.samples_output,      # Filter samples
                    "-r", ",".join(regions),        # Filter regions
                    "-v", variant_type,             # SNP/indel filter
                    self.vcf_input,
                    "-Oz", "-o", temp_filtered_vcf
                ]
                self.run_command(cmd_first_filter, "Initial VCF filtering (parallel)")
                
                # Index the temporary VCF
                cmd_index_temp = [bcftools_path, "index", "-t", temp_filtered_vcf]
                self.run_command(cmd_index_temp, "Indexing temporary VCF")
                
                # Split multiallelic sites into biallelic records with parallel processing
                cmd_split = [
                    bcftools_path,
                    "norm",
                    "--threads", str(self.threads),  # Threads flag after subcommand
                    "-m", "-any",                   # Split multiallelic sites
                    temp_filtered_vcf,
                    "-Oz", "-o", self.genotype_vcf_output
                ]
                self.run_command(cmd_split, "Splitting multiallelic sites (parallel)")
        
        # Index the final VCF
        cmd_index = [bcftools_path, "index", "-t", self.genotype_vcf_output]
        self.run_command(cmd_index, "Indexing final VCF")
        
        logger.info(f"✅ Genotype VCF created: {self.genotype_vcf_output}")
        return self.common_samples
    
    def run_pipeline(self):
        """Execute the complete input building pipeline"""
        logger.info("Starting final tensorQTL input building pipeline...")
        
        try:
            self.validate_input_files()
            self.create_annotation_bed()     # Create annotations first
            self.process_expression_data()   # Creates both BED and TSV
            self.build_covariates()
            self.create_phenotype_data()     # Creates both BED and TSV
            self.process_genotypes_parallel()
            self.create_sample_mapping()
            
            self.generate_summary()
            logger.info("🎉 Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def generate_summary(self):
        """Generate pipeline summary"""
        logger.info("\n" + "="*60)
        logger.info("=== FINAL PIPELINE SUMMARY ===")
        logger.info("="*60)
        
        logger.info(f"📊 EXPRESSION DATA")
        logger.info(f"  BED file (tensorQTL): {self.expression_bed_output}")
        logger.info(f"  TSV file (raw counts): {self.expression_tsv_output}")
        logger.info(f"  Genes: {self.expression_bed.shape[0]}")
        logger.info(f"  Samples: {len(self.expression_samples)}")
        
        logger.info(f"📈 COVARIATES")
        logger.info(f"  File: {self.covar_output}")
        logger.info(f"  Format: TSV (covariates × samples)")
        logger.info(f"  Covariates: {self.covar_matrix.shape[0]}")
        logger.info(f"  Samples: {self.covar_matrix.shape[1]}")
        
        if hasattr(self, 'phenotype_bed'):
            logger.info(f"🏥 PHENOTYPES")
            logger.info(f"  BED file (tensorQTL): {self.phenotype_bed_output}")
            logger.info(f"  TSV file (analysis): {self.phenotype_tsv_output}")
            logger.info(f"  Phenotypes: {self.phenotype_bed.shape[0]}")
            logger.info(f"  Samples: {len(self.phenotype_samples)}")
        
        logger.info(f"🧬 GENOTYPES")
        logger.info(f"  VCF: {self.genotype_vcf_output}")
        vcf_config = self.config['vcf_processing']
        logger.info(f"  Biallelic only: {vcf_config.get('biallelic_only', True)}")
        logger.info(f"  Variant type: {vcf_config.get('variant_type', 'snps')}")
        logger.info(f"  Threads used: {self.threads}")
        logger.info(f"  Samples: {len(self.common_samples)}")
        
        logger.info(f"📝 ANNOTATIONS")
        logger.info(f"  File: {self.annotation_bed_output}")
        logger.info(f"  Genes: {self.annotation_bed.shape[0]}")
        
        logger.info(f"📋 SAMPLE MAPPING")
        logger.info(f"  File: {self.mapping_output}")
        
        logger.info("="*60)
        logger.info("=== TENSORQTL USAGE ===")
        logger.info("python3 -m tensorqtl \\")
        logger.info(f"  {self.genotype_vcf_output} \\")
        logger.info(f"  {self.expression_bed_output} \\")
        logger.info(f"  output_prefix \\")
        logger.info(f"  --covariates {self.covar_output} \\")
        logger.info(f"  --phenotypes {self.phenotype_bed_output} \\")
        logger.info(f"  --mode cis")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='Build final tensorQTL input files with parallel processing')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--threads', type=int, help='Override number of threads (0=all cores)')
    
    args = parser.parse_args()
    
    try:
        builder = FinalInputBuilder(args.config)
        
        # Override threads if provided
        if args.threads is not None:
            builder.threads = args.threads
            logger.info(f"Overriding threads to: {builder.threads}")
        
        builder.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()