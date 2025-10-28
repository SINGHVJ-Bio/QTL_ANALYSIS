#!/usr/bin/env python3
"""
Optimized pipeline to prepare QTL input files for tensorQTL
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
import numpy as np
from sklearn.preprocessing import StandardScaler
pd.set_option('future.no_silent_downcasting', True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BuildInput')

class FinalInputBuilder:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.setup_paths()
        
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_paths(self):
        # Input files
        self.count_file = self.config['input_files']['count_file']
        self.meta_file = self.config['input_files']['meta_file']
        self.pca_file = self.config['input_files'].get('pca_file')
        self.vcf_input = self.config['input_files']['vcf_input']
        self.gtf_file = self.config['input_files']['gtf']
        
        # File separators
        self.count_file_sep = self.config['input_files'].get('count_file_sep', '\t')
        self.meta_file_sep = self.config['input_files'].get('meta_file_sep', ',')
        self.pca_file_sep = self.config['input_files'].get('pca_file_sep', ' ')
        
        # Count transformation
        self.add_value = self.config.get('count_transformation', {}).get('add_value', 1)
        self.round_counts = self.config.get('count_transformation', {}).get('round_counts', True)
        
        # Sample name optimization - read from output section
        self.sample_name_optimization = self.config['output'].get('sample_name_optimization', False)
        
        # Output directory
        self.out_dir = self.config['output']['output_dir']
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Output files
        self.expression_bed_output = os.path.join(self.out_dir, "expression.bed")
        self.expression_tsv_output = os.path.join(self.out_dir, "expression.tsv")
        self.covar_output = os.path.join(self.out_dir, "covariates.txt")
        self.covar_raw_output = os.path.join(self.out_dir, "covariate_raw.txt")
        self.samples_output = os.path.join(self.out_dir, "samples.txt")
        self.genotype_vcf_output = os.path.join(self.out_dir, "genotypes.vcf.gz")
        self.annotation_bed_output = os.path.join(self.out_dir, "annotations.bed")
        self.sample_mapping_file = os.path.join(self.out_dir, "sample_mapping.txt")
        
        logger.info(f"Output directory: {self.out_dir}")
        if self.sample_name_optimization:
            logger.info("Sample name optimization: ENABLED")
    
    def run_command(self, cmd, description):
        logger.info(f"Running: {description}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ {description} completed")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
    
    def optimize_sample_name(self, sample_id):
        """Extract only the first part before any underscore"""
        if pd.isna(sample_id):
            return sample_id
        return str(sample_id).split('_')[0]
    
    def normalize_sample_name(self, sample_id):
        """Apply both normalization and optimization"""
        normalized = str(sample_id).strip().replace('.0', '')
        if self.sample_name_optimization:
            normalized = self.optimize_sample_name(normalized)
        return normalized
    
    def apply_sample_mapping(self, df, sample_column):
        """Apply sample mapping to a dataframe column"""
        if sample_column in df.columns:
            df[sample_column] = df[sample_column].apply(self.normalize_sample_name)
        return df
    
    def read_file(self, file_path, sep, **kwargs):
        """Read file with encoding fallback and error handling"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding, **kwargs)
                return df
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        return pd.read_csv(file_path, sep=sep, encoding='utf-8', **kwargs)
    
    def get_vcf_samples(self):
        """Extract sample IDs from VCF file"""
        bcftools_path = subprocess.run(["which", "bcftools"], capture_output=True, text=True).stdout.strip()
        
        # Check if VCF file is valid and indexed
        cmd_check = [bcftools_path, "index", "-s", self.vcf_input]
        try:
            subprocess.run(cmd_check, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("VCF file might not be indexed, trying to index...")
            cmd_index = [bcftools_path, "index", self.vcf_input]
            subprocess.run(cmd_index, check=True, capture_output=True)
        
        # Extract samples
        cmd = [bcftools_path, "query", "-l", self.vcf_input]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        vcf_samples = [sample.strip() for sample in result.stdout.strip().split('\n') if sample.strip()]
        
        logger.info(f"Found {len(vcf_samples)} samples in VCF")
        return set(vcf_samples)
    
    def create_sample_mapping(self, vcf_samples):
        """Create mapping from original VCF samples to optimized names"""
        sample_mapping = {}
        
        for sample in vcf_samples:
            optimized_sample = self.normalize_sample_name(sample)
            sample_mapping[sample] = optimized_sample
        
        # Write mapping file for bcftools reheader
        with open(self.sample_mapping_file, 'w') as f:
            for old_name, new_name in sample_mapping.items():
                f.write(f"{old_name}\t{new_name}\n")
        
        logger.info(f"Created sample mapping for {len(sample_mapping)} samples")
        return sample_mapping
    
    def rename_vcf_samples(self):
        """Rename VCF samples using bcftools reheader"""
        logger.info("Renaming VCF samples...")
        
        bcftools_path = subprocess.run(["which", "bcftools"], capture_output=True, text=True).stdout.strip()
        
        # Create temporary VCF with renamed samples
        temp_vcf = os.path.join(self.out_dir, "genotypes_renamed.vcf.gz")
        
        cmd = [
            bcftools_path, "reheader",
            "-s", self.sample_mapping_file,
            self.vcf_input,
            "-o", temp_vcf
        ]
        
        self.run_command(cmd, "Renaming VCF samples with bcftools reheader")
        
        # Index the renamed VCF
        cmd_index = [bcftools_path, "index", "-t", temp_vcf]
        self.run_command(cmd_index, "Indexing renamed VCF")
        
        return temp_vcf
    
    def check_sample_overlap(self, vcf_samples, expression_samples):
        """Check if samples match between VCF and expression data"""
        vcf_set = set(vcf_samples)
        expr_set = set(expression_samples)
        
        common_samples = vcf_set & expr_set
        vcf_only = vcf_set - expr_set
        expr_only = expr_set - vcf_set
        
        logger.info(f"Sample overlap: {len(common_samples)} common, {len(vcf_only)} VCF-only, {len(expr_only)} expression-only")
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between VCF and expression data")
        
        return common_samples, vcf_only, expr_only
    
    def subset_vcf_if_needed(self, vcf_file, samples_to_keep):
        """Subset VCF only if necessary"""
        bcftools_path = subprocess.run(["which", "bcftools"], capture_output=True, text=True).stdout.strip()
        
        # Get current samples in VCF
        cmd = [bcftools_path, "query", "-l", vcf_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        current_samples = set([sample.strip() for sample in result.stdout.strip().split('\n') if sample.strip()])
        
        # Check if subsetting is needed
        if current_samples == samples_to_keep:
            logger.info("VCF samples already match target samples - no subsetting needed")
            return vcf_file
        
        logger.info(f"Subsetting VCF from {len(current_samples)} to {len(samples_to_keep)} samples")
        
        # Write samples to keep
        samples_file = os.path.join(self.out_dir, "samples_to_keep.txt")
        with open(samples_file, 'w') as f:
            for sample in samples_to_keep:
                f.write(f"{sample}\n")
        
        # Subset VCF
        cmd_subset = [
            bcftools_path, "view",
            "-S", samples_file,
            vcf_file,
            "-Oz", "-o", self.genotype_vcf_output
        ]
        
        self.run_command(cmd_subset, "Subsetting VCF to common samples")
        
        # Index the subsetted VCF
        cmd_index = [bcftools_path, "index", "-t", self.genotype_vcf_output]
        self.run_command(cmd_index, "Indexing subsetted VCF")
        
        # Clean up temporary files
        if vcf_file != self.vcf_input:  # Don't delete original input
            os.remove(vcf_file)
            os.remove(vcf_file + '.tbi')
        os.remove(samples_file)
        
        return self.genotype_vcf_output
    
    def transform_count_matrix(self, count_df):
        sample_cols = [col for col in count_df.columns if col != 'gene_id']
        transformed_df = count_df.copy()
        
        transformed_df[sample_cols] = transformed_df[sample_cols] + self.add_value
        if self.round_counts:
            transformed_df[sample_cols] = np.round(transformed_df[sample_cols]).astype(int)
        
        logger.info("Count transformation applied")
        return transformed_df
    
    def parse_gtf_for_gene_coordinates(self, gtf_file, gene_types=None):
        gene_coords = {}
        
        with open(gtf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 9 or fields[2] != 'gene':
                    continue
                    
                attributes = {}
                for attr in fields[8].split(';'):
                    attr = attr.strip()
                    if ' ' in attr:
                        key, value = attr.split(' ', 1)
                        attributes[key] = value.strip('"')
                
                gene_id = attributes.get('gene_id', '').split('.')[0]
                gene_type = attributes.get('gene_type', '')
                
                if gene_types and gene_type not in gene_types:
                    continue
                    
                if gene_id:
                    chrom = fields[0]
                    start = int(fields[3])
                    end = int(fields[4])
                    strand = fields[6]
                    
                    tss = start if strand == '+' else end
                    bed_start = tss
                    bed_end = tss + 1
                    
                    gene_coords[gene_id] = {
                        'chr': chrom, 'start': bed_start, 'end': bed_end, 'strand': strand,
                        'gene_name': attributes.get('gene_name', gene_id),
                        'gene_type': gene_type
                    }
        
        logger.info(f"Found coordinates for {len(gene_coords)} genes")
        return gene_coords

    def create_annotation_bed(self):
        logger.info("Creating annotations.bed file...")
        
        annotation_config = self.config['annotations']
        gene_types = annotation_config.get('gene_types', ['protein_coding'])
        additional_columns = annotation_config.get('additional_columns', ['gene_name', 'gene_type'])
        
        gene_coords = self.parse_gtf_for_gene_coordinates(self.gtf_file, gene_types)
        
        annotation_data = []
        for gene_id, coords in gene_coords.items():
            row = {
                'chr': coords['chr'], 'start': coords['start'], 'end': coords['end'],
                'gene_id': gene_id, 'strand': coords['strand']
            }
            for col in additional_columns:
                if col in coords:
                    row[col] = coords[col]
            annotation_data.append(row)
        
        annotation_bed = pd.DataFrame(annotation_data)
        annotation_bed['start'] = annotation_bed['start'].astype(int)
        annotation_bed['end'] = annotation_bed['end'].astype(int)
        
        # Sort by chromosome and position
        chrom_order = {f'chr{i}': i for i in range(1, 23)}
        chrom_order.update({'chrX': 23, 'chrY': 24, 'chrM': 25})
        annotation_bed['chrom_num'] = annotation_bed['chr'].map(chrom_order)
        annotation_bed = annotation_bed.sort_values(['chrom_num', 'start']).drop('chrom_num', axis=1)
        
        annotation_bed.to_csv(self.annotation_bed_output, sep="\t", index=False)
        logger.info(f"✅ Annotations BED created: {annotation_bed.shape[0]} genes")
        
        self.annotation_bed = annotation_bed
        self.gene_coords = gene_coords
        return annotation_bed
    
    def process_expression_data(self):
        logger.info("Processing expression data...")
        
        count_df = self.read_file(self.count_file, self.count_file_sep)
        
        if 'gene_id' not in count_df.columns:
            raise ValueError("Count matrix must contain 'gene_id' column")
        
        count_df['gene_id'] = count_df['gene_id'].astype(str).str.replace(r"\.\d+$", "", regex=True)
        count_samples = [c for c in count_df.columns if c != "gene_id"]
        
        # Apply count transformation
        count_df = self.transform_count_matrix(count_df)
        
        # Build sample mapping
        metadata_config = self.config['metadata_columns']
        rnaseq_col = metadata_config['rnaseq_library']
        wgs_col = metadata_config['wgs_library']
        
        meta_df = self.read_file(self.meta_file, self.meta_file_sep)
        
        # Apply sample name normalization to metadata
        meta_df[rnaseq_col] = meta_df[rnaseq_col].apply(self.normalize_sample_name)
        meta_df[wgs_col] = meta_df[wgs_col].apply(self.normalize_sample_name)
        
        meta_map = meta_df[[rnaseq_col, wgs_col]].dropna().rename(columns={wgs_col: "IID"})
        rename_dict = meta_map.set_index(rnaseq_col)["IID"].to_dict()
        
        # Apply sample name normalization to count matrix column names
        count_df_renamed = count_df.copy()
        count_df_renamed.columns = [self.normalize_sample_name(col) if col != 'gene_id' else col for col in count_df.columns]
        
        # Filter count matrix and rename to IID
        count_keep_cols = ["gene_id"] + [c for c in count_df_renamed.columns if c in rename_dict and c != 'gene_id']
        filtered_count_df = count_df_renamed[count_keep_cols].copy().rename(columns=rename_dict)
        filtered_count_df.to_csv(self.expression_tsv_output, sep="\t", index=False)
        
        # Create BED format
        if not hasattr(self, 'gene_coords'):
            gene_types = self.config['annotations'].get('gene_types', ['protein_coding'])
            self.gene_coords = self.parse_gtf_for_gene_coordinates(self.gtf_file, gene_types)
        
        bed_data = []
        for _, row in filtered_count_df.iterrows():
            gene_id = row['gene_id']
            if gene_id in self.gene_coords:
                coords = self.gene_coords[gene_id]
                bed_row = {
                    '#chr': coords['chr'], 'start': coords['start'], 'end': coords['end'],
                    'gene_id': gene_id, 'score': 0, 'strand': coords['strand']
                }
                for sample in filtered_count_df.columns:
                    if sample != 'gene_id':
                        bed_row[sample] = row[sample]
                bed_data.append(bed_row)
        
        expression_bed = pd.DataFrame(bed_data)
        sample_cols = [col for col in filtered_count_df.columns if col != 'gene_id']
        bed_columns = ['#chr', 'start', 'end', 'gene_id', 'score', 'strand'] + sample_cols
        expression_bed = expression_bed[bed_columns]
        
        # Sort by chromosomal coordinates
        chrom_order = {f'chr{i}': i for i in range(1, 23)}
        chrom_order.update({'chrX': 23, 'chrY': 24, 'chrM': 25})
        expression_bed['chrom_num'] = expression_bed['#chr'].map(chrom_order)
        expression_bed = expression_bed.sort_values(['chrom_num', 'start']).drop('chrom_num', axis=1)
        
        expression_bed.to_csv(self.expression_bed_output, sep="\t", index=False)
        logger.info(f"✅ Expression BED created: {expression_bed.shape[0]} genes, {len(sample_cols)} samples")
        
        self.expression_samples = sample_cols
        self.expression_bed = expression_bed
        return expression_bed
    
    def load_pca_data(self):
        if not self.pca_file or not os.path.exists(self.pca_file):
            return None
        
        try:
            # Try reading with different approaches for PCA files
            logger.info(f"Reading PCA file: {self.pca_file}")
            
            # Approach 1: Try reading with tab separator (most common for eigenvec)
            try:
                pca_df = pd.read_csv(self.pca_file, sep='\t', engine='python')
                logger.info("Successfully read PCA file with tab separator")
            except:
                # Approach 2: Try reading with space separator
                try:
                    pca_df = pd.read_csv(self.pca_file, sep='\\s+', engine='python')
                    logger.info("Successfully read PCA file with space separator")
                except:
                    # Approach 3: Try reading with any whitespace
                    pca_df = pd.read_csv(self.pca_file, delim_whitespace=True, engine='python')
                    logger.info("Successfully read PCA file with whitespace delimiter")
            
            # Check and clean column names
            pca_df.columns = [col.strip().replace('#', '') for col in pca_df.columns]
            
            # Debug: log the columns found
            logger.info(f"PCA file columns are like: {list(pca_df.columns[:5])}")
            logger.info(f"PCA file shape: {pca_df.shape}")
            
            pca_count = self.config['covariates'].get('pca_count', 5)
            
            # Check if we have PC columns
            available_pcs = [col for col in pca_df.columns if col.startswith('PC')]
            logger.info(f"Available PC columns are like: {available_pcs[:5]}")
            
            if len(available_pcs) < pca_count:
                logger.warning(f"Requested {pca_count} PCs but only {len(available_pcs)} available in PCA file")
                pca_count = len(available_pcs)
            
            if pca_count == 0:
                logger.error("No PC columns found in PCA file")
                return None
            
            # Select the required columns: FID, IID and the first pca_count PCs
            required_columns = []
            
            # Add FID and IID if they exist
            if 'FID' in pca_df.columns:
                required_columns.append('FID')
            if 'IID' in pca_df.columns:
                required_columns.append('IID')
            
            # Add the PC columns
            pc_columns = [f'PC{i}' for i in range(1, pca_count + 1)]
            required_columns.extend(pc_columns)
            
            # Check if all required columns exist
            missing_columns = set(required_columns) - set(pca_df.columns)
            if missing_columns:
                logger.warning(f"Missing columns in PCA file: {missing_columns}")
                # Use only available columns
                required_columns = [col for col in required_columns if col in pca_df.columns]
            
            pca_df = pca_df[required_columns]
            
            # Apply sample name normalization to PCA data
            if 'IID' in pca_df.columns:
                pca_df['IID'] = pca_df['IID'].apply(self.normalize_sample_name)
                logger.info(f"Loaded PCA data with {pca_df.shape[0]} samples, {pca_count} PCs")
                return pca_df
            else:
                logger.error("No IID column found in PCA file after processing")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load PCA data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def build_covariates(self):
        logger.info("Building covariate files...")
        
        meta_df = self.read_file(self.meta_file, self.meta_file_sep, low_memory=False)
        metadata_config = self.config['metadata_columns']
        wgs_col = metadata_config['wgs_library']
        
        # Apply sample name normalization to metadata
        meta_df[wgs_col] = meta_df[wgs_col].apply(self.normalize_sample_name)
        
        covariate_config = self.config['covariates']
        meta_covariates = covariate_config.get('include', [])
        pca_covariates = covariate_config.get('pca_include', [])
        
        # Load PCA data
        pca_data = None
        if pca_covariates and self.pca_file:
            pca_df = self.load_pca_data()
            if pca_df is not None:
                available_pca_cols = [col for col in pca_covariates if col in pca_df.columns]
                if available_pca_cols:
                    pca_data = pca_df[["IID"] + available_pca_cols]
        
        # Prepare base covariates
        available_meta_cols = [col for col in meta_covariates if col in meta_df.columns]
        base_covariates = meta_df[[wgs_col] + available_meta_cols].copy().rename(columns={wgs_col: "ID"})
        
        # Filter to expression samples
        base_covariates = base_covariates[base_covariates["ID"].isin(self.expression_samples)].drop_duplicates(subset=["ID"]).dropna()
        
        # Create RAW covariates file
        raw_covariates = base_covariates.copy()
        if pca_data is not None:
            raw_covariates = raw_covariates.merge(pca_data.rename(columns={"IID": "ID"}), on="ID", how="left")
        
        raw_covar_matrix = raw_covariates.set_index("ID").T
        raw_covar_matrix.index.name = "covariate"
        raw_covar_matrix.to_csv(self.covar_raw_output, sep="\t", index=True)
        logger.info(f"✅ Raw covariates created: {raw_covar_matrix.shape[0]} covariates, {raw_covar_matrix.shape[1]} samples")
        
        # Create PROCESSED covariates file
        processed_covariates = base_covariates.copy()
        
        # One-hot encode categorical variables
        categorical_cols = [col for col in available_meta_cols if processed_covariates[col].dtype == 'object' or processed_covariates[col].nunique() < 10]
        if categorical_cols:
            processed_covariates = pd.get_dummies(processed_covariates, columns=categorical_cols, drop_first=True)
        
        if pca_data is not None:
            processed_covariates = processed_covariates.merge(pca_data.rename(columns={"IID": "ID"}), on="ID", how="left")
        
        # Standardize continuous covariates
        cols_to_standardize = []
        for col in processed_covariates.columns:
            if col != 'ID' and processed_covariates[col].dtype in ['float64', 'int64'] and processed_covariates[col].nunique() > 2:
                cols_to_standardize.append(col)
        
        if cols_to_standardize:
            scaler = StandardScaler()
            processed_covariates[cols_to_standardize] = scaler.fit_transform(processed_covariates[cols_to_standardize])
        
        processed_covar_matrix = processed_covariates.set_index("ID").T
        processed_covar_matrix.index.name = "id"

        # Convert any True/False values to 1/0
        processed_covar_matrix = processed_covar_matrix.replace({True: 1, False: 0})
        # Save processed covariate file in proper TensorQTL format
        processed_covar_matrix.to_csv(self.covar_output, sep="\t", index=True)
        logger.info(f"✅ Processed covariates created: {processed_covar_matrix.shape[0]} covariates, {processed_covar_matrix.shape[1]} samples")
        
        self.covariate_samples = processed_covar_matrix.columns.tolist()
        return processed_covar_matrix
    
    def process_genotypes(self):
        """Optimized genotype processing with sample renaming first"""
        logger.info("Processing genotype data...")
        
        # Get VCF samples
        vcf_samples = self.get_vcf_samples()
        
        # Create sample mapping and rename VCF samples
        sample_mapping = self.create_sample_mapping(vcf_samples)
        renamed_vcf = self.rename_vcf_samples()
        
        # Get the renamed samples (normalized names)
        renamed_vcf_samples = set(sample_mapping.values())
        
        # Check sample overlap
        common_samples, vcf_only, expr_only = self.check_sample_overlap(renamed_vcf_samples, self.expression_samples)
        
        # Write final sample list
        with open(self.samples_output, 'w') as f:
            for sample in common_samples:
                f.write(f"{sample}\n")
        
        # Subset VCF only if needed
        final_vcf = self.subset_vcf_if_needed(renamed_vcf, common_samples)
        
        if final_vcf != self.genotype_vcf_output:
            # If we didn't create a new file, copy the renamed VCF to final location
            if final_vcf == renamed_vcf and renamed_vcf != self.vcf_input:
                os.rename(renamed_vcf, self.genotype_vcf_output)
                os.rename(renamed_vcf + '.tbi', self.genotype_vcf_output + '.tbi')
        
        logger.info(f"✅ Genotype VCF created: {self.genotype_vcf_output}")
        self.common_samples = list(common_samples)
        return self.common_samples
    
    def run_pipeline(self):
        logger.info("Starting tensorQTL input building pipeline...")
        
        try:
            self.create_annotation_bed()
            self.process_expression_data()
            self.build_covariates()
            self.process_genotypes()
            
            logger.info("🎉 Pipeline completed successfully!")
            self.generate_summary()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def generate_summary(self):
        logger.info("\n" + "="*50)
        logger.info("=== PIPELINE SUMMARY ===")
        logger.info("="*50)
        logger.info(f"Expression: {self.expression_bed.shape[0]} genes, {len(self.expression_samples)} samples")
        logger.info(f"Covariates: {len(self.covariate_samples)} samples")
        logger.info(f"Genotypes: {len(self.common_samples)} samples")
        logger.info(f"Sample ID optimization: {'ENABLED' if self.sample_name_optimization else 'DISABLED'}")
        logger.info("="*50)

def main():
    parser = argparse.ArgumentParser(description='Build tensorQTL input files')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    try:
        builder = FinalInputBuilder(args.config)
        builder.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()