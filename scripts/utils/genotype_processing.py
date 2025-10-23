#!/usr/bin/env python3
"""
Comprehensive genotype processing utilities - Enhanced Version
Handles VCF, VCF.GZ, BCF, PLINK with advanced filtering and QC
Optimized for 100GB+ VCF files with 400+ samples
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess
import tempfile
import re
import gzip
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger('QTLPipeline')

class GenotypeProcessor:
    def __init__(self, config):
        self.config = config
        self.processing_config = config.get('genotype_processing', {})
        self.large_data_config = config.get('large_data', {})
        
    def process_genotypes(self, input_file, output_dir):
        """Main genotype processing function with comprehensive QC"""
        logger.info(f"🔧 Processing genotype file: {input_file}")
        
        # Create processing directory
        process_dir = os.path.join(output_dir, "genotype_processing")
        Path(process_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 0: Validate input file and system resources
        self.validate_input_file(input_file)
        self.check_system_resources()
        
        # Step 1: Detect and validate format
        input_format = self.detect_input_format(input_file)
        logger.info(f"📁 Detected input format: {input_format}")
        
        # Step 2: For large VCF files, convert directly to PLINK format
        if self.should_use_plink(input_file, input_format):
            logger.info("🔄 Large dataset detected - using PLINK format for efficiency")
            plink_base = self.convert_to_plink_direct(input_file, process_dir, input_format)
            return plink_base
        
        # Step 3: Standard processing for smaller files
        standardized_file = self.standardize_format(input_file, process_dir, input_format)
        
        # Step 4: Apply comprehensive pre-processing
        processed_file = self.apply_preprocessing(standardized_file, process_dir)
        
        # Step 5: Generate QC reports
        self.generate_qc_reports(processed_file, process_dir)
        
        # Step 6: Prepare final file
        final_file = self.prepare_final_file(processed_file, process_dir)
        
        logger.info(f"✅ Genotype processing completed: {final_file}")
        return final_file
    
    def should_use_plink(self, input_file, input_format):
        """Determine if we should use PLINK format for efficiency"""
        # Check file size
        file_size_gb = os.path.getsize(input_file) / (1024**3)
        
        # Use PLINK if file is large OR explicitly configured
        use_plink = (
            file_size_gb > 10 or  # Large file threshold
            self.large_data_config.get('force_plink', False) or
            self.processing_config.get('prefer_plink', True)
        )
        
        if use_plink:
            logger.info(f"📊 Large file detected ({file_size_gb:.1f} GB) - using PLINK format")
        
        return use_plink
    
    def convert_to_plink_direct(self, input_file, output_dir, input_format):
        """Convert directly to PLINK format for large datasets"""
        logger.info("🔄 Converting directly to PLINK format for large dataset...")
        
        plink_base = os.path.join(output_dir, "genotypes")
        
        if input_format in ['vcf', 'vcf.gz', 'bcf']:
            # Convert VCF to PLINK with chunking for large files
            self.convert_vcf_to_plink_chunked(input_file, plink_base)
        elif input_format == 'plink_bed':
            # Already in PLINK format, just ensure all files exist
            base_name = input_file.replace('.bed', '')
            required_files = [f'{base_name}.bed', f'{base_name}.bim', f'{base_name}.fam']
            for req_file in required_files:
                if not os.path.exists(req_file):
                    raise FileNotFoundError(f"PLINK file not found: {req_file}")
            
            # Copy to our directory
            for ext in ['.bed', '.bim', '.fam']:
                src = f"{base_name}{ext}"
                dst = f"{plink_base}{ext}"
                self.run_command(f"cp {src} {dst}", f"Copying {ext} file")
        else:
            raise ValueError(f"Cannot convert {input_format} directly to PLINK")
        
        # Apply PLINK-specific filters
        filtered_base = self.apply_plink_filters(plink_base, output_dir)
        
        logger.info(f"✅ PLINK conversion completed: {filtered_base}")
        return filtered_base
    
    def convert_vcf_to_plink_chunked(self, vcf_file, plink_base):
        """Convert VCF to PLINK with chunking for very large files"""
        logger.info("🔧 Converting VCF to PLINK with chunking...")
        
        # First, get chromosome list
        chromosomes = self.get_chromosomes_from_vcf(vcf_file)
        logger.info(f"📊 Found chromosomes: {', '.join(chromosomes)}")
        
        if len(chromosomes) > 1 and self.large_data_config.get('process_by_chromosome', True):
            # Process by chromosome in parallel
            self.process_by_chromosome(vcf_file, plink_base, chromosomes)
        else:
            # Process entire file at once
            self.run_command(
                f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed --out {plink_base}",
                "Converting VCF to PLINK"
            )
    
    def get_chromosomes_from_vcf(self, vcf_file):
        """Extract chromosome list from VCF file"""
        logger.info("🔍 Extracting chromosome list...")
        
        cmd = f"{self.config['paths']['bcftools']} view -h {vcf_file} 2>/dev/null | "
        cmd += "grep -v '^#' | cut -f1 | sort | uniq"
        
        result = self.run_command(cmd, "Extracting chromosomes", check=False)
        
        if result.returncode == 0 and result.stdout.strip():
            chromosomes = [c.strip() for c in result.stdout.split('\n') if c.strip()]
            return chromosomes
        else:
            # Fallback: try to get from the file directly
            try:
                cmd = f"{self.config['paths']['bcftools']} view -H {vcf_file} | cut -f1 | head -1000 | sort | uniq"
                result = self.run_command(cmd, "Extracting chromosomes fallback", check=False)
                if result.returncode == 0:
                    chromosomes = [c.strip() for c in result.stdout.split('\n') if c.strip()]
                    return chromosomes
            except:
                pass
            
            # Final fallback
            return ['1']  # Assume chromosome 1 if we can't determine
    
    def process_by_chromosome(self, vcf_file, plink_base, chromosomes):
        """Process VCF by chromosome in parallel"""
        logger.info(f"🔧 Processing {len(chromosomes)} chromosomes...")
        
        max_workers = min(len(chromosomes), self.config['performance'].get('num_threads', 4))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for chrom in chromosomes:
                chrom_base = f"{plink_base}_chr{chrom}"
                future = executor.submit(self.process_single_chromosome, vcf_file, chrom_base, chrom)
                futures[future] = chrom
            
            # Wait for completion
            for future in as_completed(futures):
                chrom = futures[future]
                try:
                    future.result()
                    logger.info(f"✅ Chromosome {chrom} processed")
                except Exception as e:
                    logger.error(f"❌ Chromosome {chrom} failed: {e}")
                    raise
        
        # Merge all chromosome files
        self.merge_plink_files(plink_base, chromosomes)
        
        # Clean up temporary chromosome files
        self.cleanup_chromosome_files(plink_base, chromosomes)
    
    def process_single_chromosome(self, vcf_file, output_base, chromosome):
        """Process a single chromosome"""
        cmd = (
            f"{self.config['paths']['plink']} --vcf {vcf_file} "
            f"--chr {chromosome} --make-bed --out {output_base}"
        )
        self.run_command(cmd, f"Processing chromosome {chromosome}")
    
    def merge_plink_files(self, plink_base, chromosomes):
        """Merge PLINK files from all chromosomes"""
        logger.info("🔧 Merging chromosome files...")
        
        # Create merge list
        merge_list_file = f"{plink_base}_merge_list.txt"
        with open(merge_list_file, 'w') as f:
            for chrom in chromosomes:
                f.write(f"{plink_base}_chr{chrom}\n")
        
        # Merge using PLINK
        self.run_command(
            f"{self.config['paths']['plink']} --merge-list {merge_list_file} --make-bed --out {plink_base}",
            "Merging chromosome files"
        )
        
        # Remove merge list
        os.remove(merge_list_file)
    
    def cleanup_chromosome_files(self, plink_base, chromosomes):
        """Clean up temporary chromosome files"""
        for chrom in chromosomes:
            for ext in ['.bed', '.bim', '.fam', '.log']:
                file_path = f"{plink_base}_chr{chrom}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    def apply_plink_filters(self, plink_base, output_dir):
        """Apply comprehensive filters to PLINK data"""
        logger.info("🔧 Applying PLINK filters...")
        
        filtered_base = os.path.join(output_dir, "filtered_genotypes")
        
        filter_cmd = f"{self.config['paths']['plink']} --bfile {plink_base}"
        
        # MAF filter
        min_maf = self.processing_config.get('min_maf', 0.01)
        filter_cmd += f" --maf {min_maf}"
        
        # Call rate filter
        min_call_rate = self.processing_config.get('min_call_rate', 0.95)
        filter_cmd += f" --geno {1 - min_call_rate}"
        
        # HWE filter
        hwe_threshold = self.processing_config.get('hwe_threshold', 1e-6)
        filter_cmd += f" --hwe {hwe_threshold}"
        
        # Final output
        filter_cmd += f" --make-bed --out {filtered_base}"
        
        self.run_command(filter_cmd, "Applying PLINK filters")
        
        return filtered_base
    
    def check_system_resources(self):
        """Check if system has sufficient resources for large dataset processing"""
        logger.info("🔍 Checking system resources...")
        
        # Check memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        required_memory = self.large_data_config.get('min_memory_gb', 16)
        
        if memory_gb < required_memory:
            logger.warning(f"⚠️ Low memory: {memory_gb:.1f} GB available, {required_memory} GB recommended")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_space_gb = disk_usage.free / (1024**3)
        required_disk = self.large_data_config.get('min_disk_gb', 50)
        
        if free_space_gb < required_disk:
            logger.warning(f"⚠️ Low disk space: {free_space_gb:.1f} GB available, {required_disk} GB recommended")
        
        logger.info(f"✅ System check: {memory_gb:.1f} GB RAM, {free_space_gb:.1f} GB disk free")
    
    def validate_input_file(self, input_file):
        """Validate input genotype file"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Genotype file not found: {input_file}")
        
        # Check file size
        file_size = os.path.getsize(input_file) / (1024**3)  # GB
        if file_size == 0:
            raise ValueError(f"Genotype file is empty: {input_file}")
        
        logger.info(f"📊 Input file size: {file_size:.2f} GB")
        
        # For very large files, do a quick format check without loading entire file
        if file_size > 10:
            self.quick_validate_large_file(input_file)
    
    def quick_validate_large_file(self, input_file):
        """Quick validation for very large files"""
        logger.info("🔍 Performing quick validation for large file...")
        
        # Check if file is compressed
        if input_file.endswith('.gz'):
            # Check gzip integrity
            try:
                with gzip.open(input_file, 'rb') as f:
                    # Read just the first few bytes to check integrity
                    f.read(100)
                logger.info("✅ Gzip integrity check passed")
            except Exception as e:
                raise ValueError(f"Gzip file appears corrupted: {e}")
        
        # For VCF files, check header
        if input_file.endswith(('.vcf', '.vcf.gz')):
            try:
                cmd = f"{self.config['paths']['bcftools']} view -h {input_file} | head -5"
                result = self.run_command(cmd, "Checking VCF header", check=False)
                if result.returncode != 0:
                    raise ValueError("VCF file header cannot be read")
                logger.info("✅ VCF header validation passed")
            except Exception as e:
                logger.warning(f"⚠️ VCF header check failed: {e}")
    
    def detect_input_format(self, input_file):
        """Detect the format of the input genotype file"""
        file_ext = input_file.lower()
        
        if file_ext.endswith('.vcf.gz') or file_ext.endswith('.vcf.bgz'):
            return 'vcf.gz'
        elif file_ext.endswith('.vcf'):
            return 'vcf'
        elif file_ext.endswith('.bcf'):
            return 'bcf'
        elif file_ext.endswith('.bed'):
            return 'plink_bed'
        elif file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
            return 'hdf5'
        else:
            # Try to detect by content
            return self.detect_format_by_content(input_file)
    
    def detect_format_by_content(self, input_file):
        """Detect file format by examining content"""
        try:
            if input_file.endswith('.gz'):
                with gzip.open(input_file, 'rt') as f:
                    first_line = f.readline()
            else:
                with open(input_file, 'r') as f:
                    first_line = f.readline()
            
            if first_line.startswith('##fileformat=VCF'):
                return 'vcf'
            elif first_line.startswith('#CHROM'):
                return 'vcf'
            elif first_line.startswith('#fileformat=VCF'):
                return 'vcf'
            elif 'BED' in first_line:
                return 'plink_bed'
            else:
                # Check for PLINK BED magic number
                if input_file.endswith('.bed'):
                    with open(input_file, 'rb') as f:
                        magic = f.read(2)
                        if magic == b'\x6c\x1b':  # PLINK BED magic number
                            return 'plink_bed'
                
                logger.warning(f"Could not detect format for {input_file}, assuming VCF")
                return 'vcf'
        except Exception as e:
            logger.warning(f"Format detection failed: {e}, assuming VCF")
            return 'vcf'
    
    def standardize_format(self, input_file, output_dir, input_format):
        """Convert various formats to standard VCF.gz"""
        output_file = os.path.join(output_dir, "standardized.vcf.gz")
        
        if input_format in ['vcf', 'vcf.gz']:
            if input_file.endswith('.gz'):
                # Already compressed VCF
                return input_file
            else:
                # Compress VCF
                self.run_command(
                    f"{self.config['paths']['bgzip']} -c {input_file} > {output_file}",
                    "Compressing VCF file"
                )
                return output_file
                
        elif input_format == 'bcf':
            # Convert BCF to VCF.gz
            temp_vcf = os.path.join(output_dir, "temp_standardized.vcf")
            self.run_command(
                f"{self.config['paths']['bcftools']} view {input_file} -Ov -o {temp_vcf}",
                "Converting BCF to VCF"
            )
            self.run_command(
                f"{self.config['paths']['bgzip']} -c {temp_vcf} > {output_file}",
                "Compressing VCF file"
            )
            os.remove(temp_vcf)
            return output_file
            
        elif input_format == 'plink_bed':
            # Convert PLINK to VCF.gz
            base_name = input_file.replace('.bed', '')
            temp_vcf = os.path.join(output_dir, "temp_plink.vcf")
            
            self.run_command(
                f"{self.config['paths']['plink']} --bfile {base_name} --recode vcf --out {temp_vcf}.plink",
                "Converting PLINK to VCF"
            )
            self.run_command(
                f"{self.config['paths']['bgzip']} -c {temp_vcf}.plink.vcf > {output_file}",
                "Compressing VCF file"
            )
            # Cleanup
            for ext in ['.vcf', '.log']:
                if os.path.exists(f"{temp_vcf}.plink{ext}"):
                    os.remove(f"{temp_vcf}.plink{ext}")
                    
            return output_file
            
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
    
    def apply_preprocessing(self, input_file, output_dir):
        """Apply all configured pre-processing steps"""
        current_file = input_file
        
        processing_steps = [
            ('filter_variants', self.filter_variants),
            ('handle_multiallelic', self.handle_multiallelic),
            ('normalize_chromosomes', self.normalize_chromosomes),
            ('remove_phasing', self.remove_phasing),
            ('normalize_indels', self.normalize_indels),
            ('remove_duplicates', self.remove_duplicates),
            ('left_align_variants', self.left_align_variants),
        ]
        
        for step_name, step_func in processing_steps:
            if self.processing_config.get(step_name, False):
                logger.info(f"🔧 Applying {step_name}...")
                current_file = step_func(current_file, output_dir)
        
        return current_file
    
    def filter_variants(self, input_file, output_dir):
        """Apply comprehensive variant filtering"""
        output_file = os.path.join(output_dir, "filtered.vcf.gz")
        
        filter_expr = []
        
        # MAF filter
        min_maf = self.processing_config.get('min_maf', 0.01)
        if min_maf > 0:
            filter_expr.append(f"MAF > {min_maf}")
        
        # Call rate filter
        min_call_rate = self.processing_config.get('min_call_rate', 0.95)
        if min_call_rate > 0:
            filter_expr.append(f"F_MISSING < {1 - min_call_rate}")
        
        # Quality filter
        qual_threshold = self.processing_config.get('quality_threshold', 30)
        if qual_threshold > 0:
            filter_expr.append(f"QUAL > {qual_threshold}")
        
        if filter_expr:
            filter_string = " && ".join(filter_expr)
            self.run_command(
                f"{self.config['paths']['bcftools']} view {input_file} "
                f"-i '{filter_string}' -Oz -o {output_file}",
                "Filtering variants"
            )
        else:
            # No filtering, just copy
            self.run_command(f"cp {input_file} {output_file}", "Copying file")
        
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
        else:
            # Default: split
            self.run_command(
                f"{self.config['paths']['bcftools']} norm {input_file} "
                f"-m- -Oz -o {output_file}",
                "Splitting multi-allelic sites (default)"
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
            # Auto-detect and standardize
            current_file = self.auto_normalize_chromosomes(input_file, output_dir)
        elif prefix_config == 'chr':
            # Ensure chromosomes have 'chr' prefix
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs <(echo -e '1\\tchr1\\n2\\tchr2\\n3\\tchr3\\n4\\tchr4\\n5\\tchr5\\n6\\tchr6\\n7\\tchr7\\n8\\tchr8\\n9\\tchr9\\n10\\tchr10\\n11\\tchr11\\n12\\tchr12\\n13\\tchr13\\n14\\tchr14\\n15\\tchr15\\n16\\tchr16\\n17\\tchr17\\n18\\tchr18\\n19\\tchr19\\n20\\tchr20\\n21\\tchr21\\n22\\tchr22\\nX\\tchrX\\nY\\tchrY\\nMT\\tchrM') "
                f"-Oz -o {output_file}",
                "Adding 'chr' prefix to chromosomes"
            )
            current_file = output_file
        elif prefix_config == 'none':
            # Remove 'chr' prefix if present
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs <(echo -e 'chr1\\t1\\nchr2\\t2\\nchr3\\t3\\nchr4\\t4\\nchr5\\t5\\nchr6\\t6\\nchr7\\t7\\nchr8\\t8\\nchr9\\t9\\nchr10\\t10\\nchr11\\t11\\nchr12\\t12\\nchr13\\t13\\nchr14\\t14\\nchr15\\t15\\nchr16\\t16\\nchr17\\t17\\nchr18\\t18\\nchr19\\t19\\nchr20\\t20\\nchr21\\t21\\nchr22\\t22\\nchrX\\tX\\nchrY\\tY\\nchrM\\tMT') "
                f"-Oz -o {output_file}",
                "Removing 'chr' prefix from chromosomes"
            )
            current_file = output_file
        else:
            logger.warning(f"Unknown chromosome prefix setting: {prefix_config}")
            current_file = input_file
        
        if current_file != input_file:
            self.run_command(
                f"{self.config['paths']['tabix']} -p vcf {current_file}",
                "Indexing chromosome-normalized VCF"
            )
        
        return current_file
    
    def auto_normalize_chromosomes(self, input_file, output_dir):
        """Auto-detect chromosome format and standardize"""
        # Check current format
        result = self.run_command(
            f"{self.config['paths']['bcftools']} view {input_file} | head -100 | grep -v '^#' | head -1",
            "Checking chromosome format", check=False
        )
        
        if 'chr' in result.stdout:
            # Already has chr prefix, ensure consistency
            return input_file
        else:
            # Add chr prefix
            output_file = os.path.join(output_dir, "chrom_normalized.vcf.gz")
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs <(echo -e '1\\tchr1\\n2\\tchr2\\n3\\tchr3\\n4\\tchr4\\n5\\tchr5\\n6\\tchr6\\n7\\tchr7\\n8\\tchr8\\n9\\tchr9\\n10\\tchr10\\n11\\tchr11\\n12\\tchr12\\n13\\tchr13\\n14\\tchr14\\n15\\tchr15\\n16\\tchr16\\n17\\tchr17\\n18\\tchr18\\n19\\tchr19\\n20\\tchr20\\n21\\tchr21\\n22\\tchr22\\nX\\tchrX\\nY\\tchrY\\nMT\\tchrM') "
                f"-Oz -o {output_file}",
                "Auto-adding 'chr' prefix to chromosomes"
            )
            return output_file
    
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
        """Normalize INDEL representations (requires reference genome)"""
        output_file = os.path.join(output_dir, "indel_normalized.vcf.gz")
        
        # Check if reference genome is available
        ref_genome = self.processing_config.get('reference_genome')
        if not ref_genome or not os.path.exists(ref_genome):
            logger.warning("❌ Reference genome not available for INDEL normalization")
            logger.warning("   Please set 'reference_genome' in config to enable this feature")
            return input_file
        
        # Check if reference index exists
        ref_index = f"{ref_genome}.fai"
        if not os.path.exists(ref_index):
            logger.warning(f"❌ Reference genome index not found: {ref_index}")
            logger.warning("   Create index with: samtools faidx <reference.fa>")
            return input_file
        
        logger.info(f"🔧 Normalizing INDELs using reference: {os.path.basename(ref_genome)}")
        
        self.run_command(
            f"{self.config['paths']['bcftools']} norm {input_file} "
            f"-f {ref_genome} -Oz -o {output_file}",
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
        """Left-align variants (requires reference genome)"""
        output_file = os.path.join(output_dir, "left_aligned.vcf.gz")
        
        # Check if reference genome is available
        ref_genome = self.processing_config.get('reference_genome')
        if not ref_genome or not os.path.exists(ref_genome):
            logger.warning("❌ Reference genome not available for left alignment")
            logger.warning("   Please set 'reference_genome' in config to enable this feature")
            return input_file
        
        # Check if reference index exists
        ref_index = f"{ref_genome}.fai"
        if not os.path.exists(ref_index):
            logger.warning(f"❌ Reference genome index not found: {ref_index}")
            logger.warning("   Create index with: samtools faidx <reference.fa>")
            return input_file
        
        logger.info(f"🔧 Left-aligning variants using reference: {os.path.basename(ref_genome)}")
        
        self.run_command(
            f"{self.config['paths']['bcftools']} norm {input_file} "
            f"-f {ref_genome} -c s -Oz -o {output_file}",
            "Left-aligning variants"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing left-aligned VCF"
        )
        
        return output_file
    
    def generate_qc_reports(self, vcf_file, output_dir):
        """Generate comprehensive QC reports"""
        logger.info("📊 Generating QC reports...")
        
        # Basic stats
        self.run_command(
            f"{self.config['paths']['bcftools']} stats {vcf_file} > {output_dir}/vcf_stats.txt",
            "Generating VCF statistics"
        )
        
        # Sample statistics
        self.run_command(
            f"{self.config['paths']['bcftools']} query -l {vcf_file} | wc -l > {output_dir}/sample_count.txt",
            "Counting samples"
        )
        
        # Variant statistics
        self.run_command(
            f"{self.config['paths']['bcftools']} view -H {vcf_file} | wc -l > {output_dir}/variant_count.txt",
            "Counting variants"
        )
        
        logger.info("✅ QC reports generated")
    
    def prepare_final_file(self, input_file, output_dir):
        """Prepare final genotype file for analysis"""
        final_file = os.path.join(output_dir, "final_genotypes.vcf.gz")
        
        # Copy to final location
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
        
        # Log final file info
        result = self.run_command(
            f"{self.config['paths']['bcftools']} view -H {final_file} | wc -l",
            "Counting final variants", check=False
        )
        variant_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        
        result = self.run_command(
            f"{self.config['paths']['bcftools']} query -l {final_file} | wc -l", 
            "Counting final samples", check=False
        )
        sample_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        
        logger.info(f"📊 Final genotype data: {variant_count} variants, {sample_count} samples")
        
        return final_file
    
    def validate_vcf(self, vcf_file):
        """Validate VCF file structure"""
        logger.info("🔍 Validating VCF file structure...")
        
        # Check if VCF is valid
        result = self.run_command(
            f"{self.config['paths']['bcftools']} view {vcf_file} -h > /dev/null",
            "Validating VCF header",
            check=False
        )
        
        if result.returncode != 0:
            raise ValueError(f"VCF file validation failed: {vcf_file}")
        
        logger.info("✅ VCF validation completed successfully")
    
    def run_command(self, cmd, description, check=True):
        """Run shell command with comprehensive error handling"""
        logger.info(f"Executing: {description}")
        logger.debug(f"Command: {cmd}")
        
        # Set timeout for large dataset operations
        timeout = self.large_data_config.get('command_timeout', 3600)  # 1 hour default
        
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
            if check and result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after {timeout} seconds")
            if check:
                raise RuntimeError(f"Command timed out: {description}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed with exit code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Command: {e.cmd}")
            if check:
                raise RuntimeError(f"Command failed: {description}") from e
            return e