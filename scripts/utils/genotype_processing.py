#!/usr/bin/env python3
"""
Comprehensive genotype processing utilities
Handles VCF, VCF.GZ, BCF, and other formats with pre-processing
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import tempfile
import re

logger = logging.getLogger('QTLPipeline')

class GenotypeProcessor:
    def __init__(self, config):
        self.config = config
        self.processing_config = config.get('genotype_processing', {})
        
    def process_genotypes(self, input_file, output_dir):
        """Main genotype processing function"""
        logger.info(f"Processing genotype file: {input_file}")
        
        # Create processing directory
        process_dir = os.path.join(output_dir, "genotype_processing")
        Path(process_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect input format
        input_format = self.detect_input_format(input_file)
        logger.info(f"Detected input format: {input_format}")
        
        # Step 1: Convert to standard format if needed
        standardized_file = self.standardize_format(input_file, process_dir, input_format)
        
        # Step 2: Apply pre-processing steps
        processed_file = self.apply_preprocessing(standardized_file, process_dir)
        
        # Step 3: Compress and index final file
        final_file = self.prepare_final_file(processed_file, process_dir)
        
        logger.info(f"Genotype processing completed: {final_file}")
        return final_file
    
    def detect_input_format(self, input_file):
        """Detect the format of the input genotype file"""
        if input_file.endswith('.vcf.gz'):
            return 'vcf.gz'
        elif input_file.endswith('.vcf'):
            return 'vcf'
        elif input_file.endswith('.bcf'):
            return 'bcf'
        elif input_file.endswith('.bed'):
            return 'plink_bed'
        elif input_file.endswith('.h5') or input_file.endswith('.hdf5'):
            return 'hdf5'
        else:
            # Try to detect by content
            return self.detect_format_by_content(input_file)
    
    def detect_format_by_content(self, input_file):
        """Detect file format by examining content"""
        try:
            if input_file.endswith('.gz'):
                import gzip
                with gzip.open(input_file, 'rt') as f:
                    first_line = f.readline()
            else:
                with open(input_file, 'r') as f:
                    first_line = f.readline()
            
            if first_line.startswith('##fileformat=VCF'):
                return 'vcf'
            elif first_line.startswith('#CHROM'):
                return 'vcf'
            elif 'BED' in first_line:
                return 'plink_bed'
            else:
                logger.warning(f"Could not detect format for {input_file}, assuming VCF")
                return 'vcf'
        except Exception as e:
            logger.warning(f"Format detection failed: {e}, assuming VCF")
            return 'vcf'
    
    def standardize_format(self, input_file, output_dir, input_format):
        """Convert various formats to standard VCF"""
        output_file = os.path.join(output_dir, "standardized.vcf")
        
        if input_format in ['vcf', 'vcf.gz']:
            # Already in VCF format, just copy/link
            if input_file.endswith('.gz'):
                return input_file
            else:
                # Compress if not already compressed
                self.run_command(
                    f"{self.config['paths']['bgzip']} -c {input_file} > {output_file}.gz",
                    "Compressing VCF file"
                )
                return f"{output_file}.gz"
                
        elif input_format == 'bcf':
            # Convert BCF to VCF
            self.run_command(
                f"{self.config['paths']['bcftools']} view {input_file} -Ov -o {output_file}",
                "Converting BCF to VCF"
            )
            self.run_command(
                f"{self.config['paths']['bgzip']} -c {output_file} > {output_file}.gz",
                "Compressing VCF file"
            )
            return f"{output_file}.gz"
            
        elif input_format == 'plink_bed':
            # Convert PLINK to VCF
            base_name = input_file.replace('.bed', '')
            self.run_command(
                f"{self.config['paths']['plink']} --bfile {base_name} --recode vcf --out {output_file}.plink",
                "Converting PLINK to VCF"
            )
            self.run_command(
                f"{self.config['paths']['bgzip']} -c {output_file}.plink.vcf > {output_file}.gz",
                "Compressing VCF file"
            )
            return f"{output_file}.gz"
            
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
    
    def apply_preprocessing(self, input_file, output_dir):
        """Apply all configured pre-processing steps"""
        current_file = input_file
        
        # Step 1: Filter variants
        if self.processing_config.get('filter_variants', True):
            current_file = self.filter_variants(current_file, output_dir)
        
        # Step 2: Handle multi-allelic sites
        if self.processing_config.get('handle_multiallelic', True):
            current_file = self.handle_multiallelic(current_file, output_dir)
        
        # Step 3: Normalize chromosomes
        if self.processing_config.get('normalize_chromosomes', True):
            current_file = self.normalize_chromosomes(current_file, output_dir)
        
        # Step 4: Remove phasing
        if self.processing_config.get('remove_phasing', True):
            current_file = self.remove_phasing(current_file, output_dir)
        
        # Step 5: Normalize indels
        if self.processing_config.get('normalize_indels', True):
            current_file = self.normalize_indels(current_file, output_dir)
        
        # Step 6: Remove duplicates
        if self.processing_config.get('remove_duplicates', True):
            current_file = self.remove_duplicates(current_file, output_dir)
        
        # Step 7: Left align variants
        if self.processing_config.get('left_align_variants', True):
            current_file = self.left_align_variants(current_file, output_dir)
        
        return current_file
    
    def filter_variants(self, input_file, output_dir):
        """Apply variant filtering"""
        output_file = os.path.join(output_dir, "filtered.vcf.gz")
        
        filter_expr = []
        
        # MAF filter
        min_maf = self.processing_config.get('min_maf', 0.01)
        filter_expr.append(f"MAF > {min_maf}")
        
        # Call rate filter
        min_call_rate = self.processing_config.get('min_call_rate', 0.95)
        filter_expr.append(f"F_MISSING < {1 - min_call_rate}")
        
        # Quality filter
        qual_threshold = self.processing_config.get('quality_threshold', 30)
        filter_expr.append(f"QUAL > {qual_threshold}")
        
        filter_string = " && ".join(filter_expr)
        
        self.run_command(
            f"{self.config['paths']['bcftools']} view {input_file} "
            f"-i '{filter_string}' -Oz -o {output_file}",
            "Filtering variants"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing filtered VCF"
        )
        
        return output_file
    
    def handle_multiallelic(self, input_file, output_dir):
        """Handle multi-allelic sites"""
        output_file = os.path.join(output_dir, "biallelic.vcf.gz")
        action = self.processing_config.get('multiallelic_action', 'split')
        
        if action == 'split':
            # Split multi-allelic sites into multiple biallelic records
            self.run_command(
                f"{self.config['paths']['bcftools']} norm {input_file} "
                f"-m- -Oz -o {output_file}",
                "Splitting multi-allelic sites"
            )
        elif action == 'drop':
            # Drop multi-allelic sites
            self.run_command(
                f"{self.config['paths']['bcftools']} view {input_file} "
                f"--max-alleles 2 -Oz -o {output_file}",
                "Dropping multi-allelic sites"
            )
        elif action == 'first_alt':
            # Keep only first alternate allele
            self.run_command(
                f"{self.config['paths']['bcftools']} view {input_file} "
                f"--min-alleles 2 --max-alleles 2 -Oz -o {output_file}",
                "Keeping first alternate allele only"
            )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing biallelic VCF"
        )
        
        return output_file
    
    def normalize_chromosomes(self, input_file, output_dir):
        """Normalize chromosome naming convention"""
        output_file = os.path.join(output_dir, "chrom_normalized.vcf.gz")
        prefix_config = self.processing_config.get('chromosome_prefix', 'auto')
        
        if prefix_config == 'auto':
            # Detect current format and standardize
            current_file = input_file
        elif prefix_config == 'chr':
            # Ensure chromosomes have 'chr' prefix
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs data/chromosome_map_chr.txt -Oz -o {output_file}",
                "Adding 'chr' prefix to chromosomes"
            )
            current_file = output_file
        elif prefix_config == 'none':
            # Remove 'chr' prefix if present
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs data/chromosome_map_nochr.txt -Oz -o {output_file}",
                "Removing 'chr' prefix from chromosomes"
            )
            current_file = output_file
        else:
            # Use custom mapping file
            if os.path.exists(prefix_config):
                self.run_command(
                    f"{self.config['paths']['bcftools']} annotate {input_file} "
                    f"--rename-chrs {prefix_config} -Oz -o {output_file}",
                    "Applying custom chromosome mapping"
                )
                current_file = output_file
            else:
                logger.warning(f"Chromosome mapping file not found: {prefix_config}")
                current_file = input_file
        
        if current_file != input_file:
            self.run_command(
                f"{self.config['paths']['tabix']} -p vcf {current_file}",
                "Indexing chromosome-normalized VCF"
            )
        
        return current_file
    
    def remove_phasing(self, input_file, output_dir):
        """Remove phasing information from genotypes"""
        output_file = os.path.join(output_dir, "unphased.vcf.gz")
        
        self.run_command(
            f"{self.config['paths']['bcftools']} view {input_file} -U -Oz -o {output_file}",
            "Removing phasing information"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing unphased VCF"
        )
        
        return output_file
    
    def normalize_indels(self, input_file, output_dir):
        """Normalize INDEL representations"""
        output_file = os.path.join(output_dir, "indel_normalized.vcf.gz")
        
        self.run_command(
            f"{self.config['paths']['bcftools']} norm {input_file} "
            f"-f reference.fa -Oz -o {output_file}",
            "Normalizing INDELs"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing INDEL-normalized VCF"
        )
        
        return output_file
    
    def remove_duplicates(self, input_file, output_dir):
        """Remove duplicate variants"""
        output_file = os.path.join(output_dir, "deduplicated.vcf.gz")
        
        self.run_command(
            f"{self.config['paths']['bcftools']} norm {input_file} "
            f"-d all -Oz -o {output_file}",
            "Removing duplicate variants"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing deduplicated VCF"
        )
        
        return output_file
    
    def left_align_variants(self, input_file, output_dir):
        """Left-align variants"""
        output_file = os.path.join(output_dir, "left_aligned.vcf.gz")
        
        self.run_command(
            f"{self.config['paths']['bcftools']} norm {input_file} "
            f"-f reference.fa -c s -Oz -o {output_file}",
            "Left-aligning variants"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing left-aligned VCF"
        )
        
        return output_file
    
    def prepare_final_file(self, input_file, output_dir):
        """Prepare final genotype file for analysis"""
        final_file = os.path.join(output_dir, "final_genotypes.vcf.gz")
        
        # Copy to final location and ensure proper indexing
        self.run_command(
            f"cp {input_file} {final_file}",
            "Creating final genotype file"
        )
        
        self.run_command(
            f"cp {input_file}.tbi {final_file}.tbi",
            "Copying index file"
        )
        
        # Validate final file
        self.validate_vcf(final_file)
        
        return final_file
    
    def validate_vcf(self, vcf_file):
        """Validate VCF file structure"""
        logger.info("Validating VCF file structure...")
        
        # Check if VCF is valid
        result = self.run_command(
            f"{self.config['paths']['bcftools']} view {vcf_file} -h",
            "Validating VCF header",
            check=False
        )
        
        if result.returncode != 0:
            raise ValueError(f"VCF file validation failed: {vcf_file}")
        
        # Check for required fields
        result = self.run_command(
            f"{self.config['paths']['bcftools']} view {vcf_file} | head -100 | grep -v '^#'",
            "Checking VCF content",
            check=False
        )
        
        if "GT" not in result.stdout:
            logger.warning("GT field not found in VCF - genotypes may not be properly encoded")
        
        logger.info("VCF validation completed successfully")
    
    def run_command(self, cmd, description, check=True):
        """Run shell command with error handling"""
        logger.info(f"Executing: {description}")
        logger.debug(f"Command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                capture_output=True, 
                text=True,
                executable='/bin/bash'
            )
            if check:
                logger.info(f"✓ {description} completed successfully")
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed with exit code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Command: {e.cmd}")
            if check:
                raise
            return e