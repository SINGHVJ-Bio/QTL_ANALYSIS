#!/usr/bin/env python3
"""
Comprehensive genotype processing utilities - Enhanced Version
Handles VCF, VCF.GZ, BCF, PLINK with advanced filtering and QC
Optimized for 100GB+ VCF files with 400+ samples
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with modular pipeline support and function exports.
Updated for tensorQTL compatibility with proper sample alignment and format handling.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import traceback, logging, sys
import subprocess
import tempfile
import re
import gzip
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger('QTLPipeline')

try:
    from scripts.utils.directory_manager import get_directory_manager, get_module_directories
except ImportError as e:
    logger.error(f"Directory manager import error: {e}")
    raise

class GenotypeProcessor:
    def __init__(self, config):
        self.config = config
        self.processing_config = config.get('genotype_processing', {})
        self.large_data_config = config.get('large_data', {})
        self.tensorqtl_config = config.get('tensorqtl', {})
        self.results_dir = Path(config.get('results_dir', 'results'))
        
        # Initialize directory manager
        self.dm = get_directory_manager(self.results_dir)
        
    def process_genotypes(self, input_file, output_dir=None):
        """Main genotype processing function with comprehensive QC and tensorQTL optimization"""
        logger.info(f"🔧 Processing genotype file: {input_file}")
        
        # Use directory manager for processing directory
        process_dir = self.dm.get_directory('processed_data', 'genotypes/processing')
        
        # Step 0: Validate input file and system resources
        self.validate_input_file(input_file)
        self.check_system_resources()
        
        # Step 1: Detect and validate format
        input_format = self.detect_input_format(input_file)
        logger.info(f"📁 Detected input format: {input_format}")
        
        # Step 2: For tensorQTL, always prefer PLINK format for optimal performance
        if self.should_use_plink(input_file, input_format):
            logger.info("🎯 Using PLINK format for optimal tensorQTL performance")
            plink_base = self.convert_to_plink_direct(input_file, process_dir, input_format)
            
            # Apply tensorQTL-optimized preprocessing
            processed_base = self.apply_tensorqtl_optimized_processing(plink_base, process_dir)
            
            # Generate QC reports with tensorQTL compatibility info
            self.generate_tensorqtl_qc_reports(processed_base, process_dir)
            
            # Prepare final file for tensorQTL
            final_file = self.prepare_tensorqtl_final_file(processed_base, process_dir)
        else:
            # Step 3: Standard processing for smaller files (maintain backward compatibility)
            logger.info("🔧 Using standard processing for smaller files")
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
        """Determine if we should use PLINK format for efficiency - Enhanced for tensorQTL"""
        # For tensorQTL, always prefer PLINK format for optimal performance
        use_plink = (
            self.large_data_config.get('force_plink', True) or  # Changed default to True
            self.processing_config.get('prefer_plink', True) or
            self.tensorqtl_config.get('prefer_plink', True) or
            True  # Always use PLINK for tensorQTL for best performance
        )
        
        if use_plink:
            file_size_gb = os.path.getsize(input_file) / (1024**3)
            logger.info(f"📊 Using PLINK format for optimal tensorQTL performance ({file_size_gb:.1f} GB)")
        
        return use_plink
    
    def convert_to_plink_direct(self, input_file, output_dir, input_format):
        """Convert directly to PLINK format for large datasets - Enhanced for tensorQTL"""
        logger.info("🔄 Converting directly to PLINK format for tensorQTL optimization...")
        
        plink_base = output_dir / "genotypes"
        
        if input_format in ['vcf', 'vcf.gz', 'bcf']:
            # Convert VCF to PLINK with tensorQTL optimization
            self.convert_vcf_to_plink_tensorqtl(input_file, str(plink_base))
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
            raise ValueError(f"Cannot convert {input_format} directly to PLINK for tensorQTL")
        
        logger.info(f"✅ PLINK conversion completed for tensorQTL: {plink_base}")
        return str(plink_base)
    
    def convert_vcf_to_plink_tensorqtl(self, vcf_file, plink_base):
        """Convert VCF to PLINK with tensorQTL-optimized parameters"""
        logger.info("🔧 Converting VCF to PLINK with tensorQTL optimization...")
        
        # First, get chromosome list for parallel processing
        chromosomes = self.get_chromosomes_from_vcf(vcf_file)
        logger.info(f"📊 Found chromosomes: {', '.join(chromosomes)}")
        
        if len(chromosomes) > 1 and self.large_data_config.get('process_by_chromosome', True):
            # Process by chromosome in parallel for large datasets
            self.process_by_chromosome_tensorqtl(vcf_file, plink_base, chromosomes)
        else:
            # Process entire file at once with tensorQTL-optimized parameters
            plink_threads = self.processing_config.get('plink_threads', 4)
            self.run_command(
                f"{self.config['paths']['plink']} --vcf {vcf_file} --make-bed "
                f"--out {plink_base} --threads {plink_threads}",
                "Converting VCF to PLINK for tensorQTL"
            )
    
    def process_by_chromosome_tensorqtl(self, vcf_file, plink_base, chromosomes):
        """Process VCF by chromosome in parallel with tensorQTL optimization"""
        logger.info(f"🔧 Processing {len(chromosomes)} chromosomes for tensorQTL...")
        
        max_workers = min(len(chromosomes), self.config['performance'].get('num_threads', 4))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for chrom in chromosomes:
                chrom_base = f"{plink_base}_chr{chrom}"
                future = executor.submit(self.process_single_chromosome_tensorqtl, vcf_file, chrom_base, chrom)
                futures[future] = chrom
            
            # Wait for completion
            for future in as_completed(futures):
                chrom = futures[future]
                try:
                    future.result()
                    logger.info(f"✅ Chromosome {chrom} processed for tensorQTL")
                except Exception as e:
                    logger.error(f"❌ Chromosome {chrom} failed: {e}")
                    raise
        
        # Merge all chromosome files with tensorQTL-optimized parameters
        self.merge_plink_files_tensorqtl(plink_base, chromosomes)
        
        # Clean up temporary chromosome files
        self.cleanup_chromosome_files(plink_base, chromosomes)
    
    def process_single_chromosome_tensorqtl(self, vcf_file, output_base, chromosome):
        """Process a single chromosome with tensorQTL optimization"""
        plink_threads = self.processing_config.get('plink_threads', 2)
        cmd = (
            f"{self.config['paths']['plink']} --vcf {vcf_file} "
            f"--chr {chromosome} --make-bed --out {output_base} --threads {plink_threads}"
        )
        self.run_command(cmd, f"Processing chromosome {chromosome} for tensorQTL")
    
    def merge_plink_files_tensorqtl(self, plink_base, chromosomes):
        """Merge PLINK files from all chromosomes with tensorQTL optimization"""
        logger.info("🔧 Merging chromosome files for tensorQTL...")
        
        # Create merge list
        merge_list_file = f"{plink_base}_merge_list.txt"
        with open(merge_list_file, 'w') as f:
            for chrom in chromosomes:
                f.write(f"{plink_base}_chr{chrom}\n")
        
        # Merge using PLINK with optimized parameters
        plink_threads = self.processing_config.get('plink_threads', 4)
        self.run_command(
            f"{self.config['paths']['plink']} --merge-list {merge_list_file} "
            f"--make-bed --out {plink_base} --threads {plink_threads}",
            "Merging chromosome files for tensorQTL"
        )
        
        # Remove merge list
        os.remove(merge_list_file)
    
    def apply_tensorqtl_optimized_processing(self, plink_base, output_dir):
        """Apply tensorQTL-optimized processing and filtering"""
        logger.info("🔧 Applying tensorQTL-optimized processing...")
        
        filtered_base = output_dir / "filtered_genotypes_tensorqtl"
        
        # Start with basic filter command
        filter_cmd = f"{self.config['paths']['plink']} --bfile {plink_base}"
        
        # Apply tensorQTL-recommended filters
        # MAF filter - tensorQTL default is 0.05, but we use config value
        min_maf = self.tensorqtl_config.get('min_maf', self.processing_config.get('min_maf', 0.01))
        filter_cmd += f" --maf {min_maf}"
        
        # Call rate filter
        min_call_rate = self.processing_config.get('min_call_rate', 0.95)
        filter_cmd += f" --geno {1 - min_call_rate}"
        
        # HWE filter - important for tensorQTL quality
        hwe_threshold = self.processing_config.get('hwe_threshold', 1e-6)
        filter_cmd += f" --hwe {hwe_threshold}"
        
        # Sample missingness filter
        sample_missing_threshold = self.processing_config.get('max_missing', 0.1)
        filter_cmd += f" --mind {sample_missing_threshold}"
        
        # Final output
        filter_cmd += f" --make-bed --out {filtered_base}"
        
        # Add threads for performance
        plink_threads = self.processing_config.get('plink_threads', 4)
        filter_cmd += f" --threads {plink_threads}"
        
        self.run_command(filter_cmd, "Applying tensorQTL-optimized filters")
        
        # Extract and save sample list for tensorQTL alignment
        self.extract_tensorqtl_sample_list(str(filtered_base), output_dir)
        
        return str(filtered_base)
    
    def extract_tensorqtl_sample_list(self, plink_base, output_dir):
        """Extract and save sample list for tensorQTL sample alignment"""
        try:
            # Read FAM file to get sample list
            fam_file = f"{plink_base}.fam"
            if os.path.exists(fam_file):
                fam_df = pd.read_csv(fam_file, sep='\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Pheno'])
                samples = fam_df['IID'].tolist()
                
                # Save sample list for tensorQTL using directory manager
                sample_file = self.dm.get_directory('processed_data', 'quality_control/sample_lists') / 'tensorqtl_samples.txt'
                with open(sample_file, 'w') as f:
                    for sample in samples:
                        f.write(f"{sample}\n")
                
                logger.info(f"✅ Extracted {len(samples)} samples for tensorQTL alignment")
                return samples
            else:
                logger.warning("❌ FAM file not found for sample extraction")
                return []
        except Exception as e:
            logger.warning(f"❌ Could not extract tensorQTL sample list: {e}")
            return []
    
    def generate_tensorqtl_qc_reports(self, plink_base, output_dir):
        """Generate comprehensive QC reports with tensorQTL compatibility info"""
        logger.info("📊 Generating tensorQTL QC reports...")
        
        # Use directory manager for QC reports
        qc_dir = self.dm.get_directory('reports', 'qc_reports/tensorqtl_genotype_qc')
        
        # Basic stats using PLINK
        self.run_command(
            f"{self.config['paths']['plink']} --bfile {plink_base} --freq --out {plink_base}_maf",
            "Generating MAF statistics"
        )
        
        self.run_command(
            f"{self.config['paths']['plink']} --bfile {plink_base} --missing --out {plink_base}_missing",
            "Generating missingness statistics"
        )
        
        self.run_command(
            f"{self.config['paths']['plink']} --bfile {plink_base} --hardy --out {plink_base}_hwe",
            "Generating HWE statistics"
        )
        
        # Generate tensorQTL compatibility report
        self.generate_tensorqtl_compatibility_report(plink_base, str(qc_dir))
        
        logger.info("✅ TensorQTL QC reports generated")
    
    def generate_tensorqtl_compatibility_report(self, plink_base, output_dir):
        """Generate tensorQTL-specific compatibility report"""
        try:
            report_file = Path(output_dir) / "tensorqtl_compatibility_report.html"
            
            # Gather statistics
            stats = self.gather_tensorqtl_stats(plink_base)
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>TensorQTL Genotype Compatibility Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                    .good {{ color: green; }}
                    .warning {{ color: orange; }}
                    .error {{ color: red; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>TensorQTL Genotype Compatibility Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Compatibility Summary</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th><th>Status</th><th>Recommendation</th></tr>
                        <tr>
                            <td>Format</td>
                            <td>PLINK BED</td>
                            <td class="good">✅ Optimal</td>
                            <td>Perfect for tensorQTL</td>
                        </tr>
                        <tr>
                            <td>Sample Count</td>
                            <td>{stats.get('samples', 'N/A')}</td>
                            <td class="{'good' if stats.get('samples', 0) > 50 else 'warning'}">
                                {'✅ Sufficient' if stats.get('samples', 0) > 50 else '⚠️ Low'}
                            </td>
                            <td>{'Adequate for QTL analysis' if stats.get('samples', 0) > 50 else 'Consider increasing sample size'}</td>
                        </tr>
                        <tr>
                            <td>Variant Count</td>
                            <td>{stats.get('variants', 'N/A')}</td>
                            <td class="{'good' if stats.get('variants', 0) > 100000 else 'warning'}">
                                {'✅ Sufficient' if stats.get('variants', 0) > 100000 else '⚠️ Low'}
                            </td>
                            <td>{'Good variant density' if stats.get('variants', 0) > 100000 else 'Consider imputation'}</td>
                        </tr>
                        <tr>
                            <td>MAF Distribution</td>
                            <td>Mean: {stats.get('mean_maf', 'N/A')}</td>
                            <td class="{'good' if stats.get('mean_maf', 0) > 0.05 else 'warning'}">
                                {'✅ Good' if stats.get('mean_maf', 0) > 0.05 else '⚠️ Low diversity'}
                            </td>
                            <td>{'Good genetic diversity' if stats.get('mean_maf', 0) > 0.05 else 'Common in specific populations'}</td>
                        </tr>
                        <tr>
                            <td>Missingness</td>
                            <td>Max: {stats.get('max_missing', 'N/A')}</td>
                            <td class="{'good' if stats.get('max_missing', 1) < 0.1 else 'warning'}">
                                {'✅ Good' if stats.get('max_missing', 1) < 0.1 else '⚠️ High'}
                            </td>
                            <td>{'Good data quality' if stats.get('max_missing', 1) < 0.1 else 'Consider stricter filtering'}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>TensorQTL Recommendations</h2>
                    <ul>
                        <li><strong>Format:</strong> PLINK format is optimal for tensorQTL performance</li>
                        <li><strong>Sample Alignment:</strong> Ensure sample IDs match phenotype and covariate files</li>
                        <li><strong>MAF Threshold:</strong> Current setting: {self.tensorqtl_config.get('maf_threshold', 0.05)}</li>
                        <li><strong>Cis Window:</strong> Current setting: {self.tensorqtl_config.get('cis_window', 1000000)} bp</li>
                        <li><strong>Performance:</strong> Using {self.config['performance'].get('num_threads', 4)} CPU threads</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"✅ TensorQTL compatibility report generated: {report_file}")
            
        except Exception as e:
            logger.warning(f"❌ Could not generate tensorQTL compatibility report: {e}")
    
    def gather_tensorqtl_stats(self, plink_base):
        """Gather statistics for tensorQTL compatibility report"""
        stats = {}
        
        try:
            # Get sample count from FAM file
            fam_file = f"{plink_base}.fam"
            if os.path.exists(fam_file):
                fam_df = pd.read_csv(fam_file, sep='\s+', header=None)
                stats['samples'] = len(fam_df)
            
            # Get variant count from BIM file
            bim_file = f"{plink_base}.bim"
            if os.path.exists(bim_file):
                bim_df = pd.read_csv(bim_file, sep='\s+', header=None)
                stats['variants'] = len(bim_df)
            
            # Get MAF statistics
            maf_file = f"{plink_base}_maf.frq"
            if os.path.exists(maf_file):
                maf_df = pd.read_csv(maf_file, sep='\s+')
                stats['mean_maf'] = maf_df['MAF'].mean() if 'MAF' in maf_df.columns else 'N/A'
            
            # Get missingness statistics
            missing_file = f"{plink_base}_missing.lmiss"
            if os.path.exists(missing_file):
                missing_df = pd.read_csv(missing_file, sep='\s+')
                stats['max_missing'] = missing_df['F_MISS'].max() if 'F_MISS' in missing_df.columns else 'N/A'
            
        except Exception as e:
            logger.warning(f"❌ Could not gather tensorQTL statistics: {e}")
        
        return stats
    
    def prepare_tensorqtl_final_file(self, plink_base, output_dir):
        """Prepare final genotype file optimized for tensorQTL"""
        # Use directory manager for final genotypes location
        final_dir = self.dm.get_directory('processed_data', 'genotypes/final')
        final_base = final_dir / "genotypes_tensorqtl"
        
        # Copy to final location with tensorQTL-optimized name
        for ext in ['.bed', '.bim', '.fam']:
            src = f"{plink_base}{ext}"
            dst = f"{final_base}{ext}"
            if os.path.exists(src):
                self.run_command(f"cp {src} {dst}", f"Creating final {ext} file")
        
        # Validate final files
        self.validate_tensorqtl_files(str(final_base))
        
        # Log final statistics
        self.log_tensorqtl_final_stats(str(final_base))
        
        return str(final_base) + ".bed"
    
    def validate_tensorqtl_files(self, plink_base):
        """Validate that all PLINK files exist and are readable for tensorQTL"""
        logger.info("🔍 Validating tensorQTL genotype files...")
        
        required_files = [f"{plink_base}.bed", f"{plink_base}.bim", f"{plink_base}.fam"]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required tensorQTL file not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"TensorQTL file is empty: {file_path}")
        
        # Check if we can read the files
        try:
            # Read FAM file to check samples
            fam_df = pd.read_csv(f"{plink_base}.fam", sep='\s+', header=None)
            logger.info(f"✅ TensorQTL validation: {len(fam_df)} samples confirmed")
            
            # Read BIM file to check variants
            bim_df = pd.read_csv(f"{plink_base}.bim", sep='\s+', header=None)
            logger.info(f"✅ TensorQTL validation: {len(bim_df)} variants confirmed")
            
        except Exception as e:
            raise ValueError(f"TensorQTL file validation failed: {e}")
        
        logger.info("✅ All tensorQTL genotype files validated successfully")
    
    def log_tensorqtl_final_stats(self, plink_base):
        """Log final statistics for tensorQTL analysis"""
        try:
            # Read FAM file for sample count
            fam_df = pd.read_csv(f"{plink_base}.fam", sep='\s+', header=None)
            sample_count = len(fam_df)
            
            # Read BIM file for variant count
            bim_df = pd.read_csv(f"{plink_base}.bim", sep='\s+', header=None)
            variant_count = len(bim_df)
            
            # Calculate approximate file size
            total_size = sum(os.path.getsize(f"{plink_base}{ext}") for ext in ['.bed', '.bim', '.fam'])
            total_size_gb = total_size / (1024**3)
            
            logger.info(f"📊 TensorQTL Final Statistics:")
            logger.info(f"   🧬 Samples: {sample_count}")
            logger.info(f"   🧬 Variants: {variant_count}")
            logger.info(f"   💾 Total size: {total_size_gb:.2f} GB")
            logger.info(f"   🔧 Format: PLINK (optimized for tensorQTL)")
            logger.info(f"   ⚡ Ready for tensorQTL analysis")
            
        except Exception as e:
            logger.warning(f"❌ Could not log final tensorQTL statistics: {e}")
    
    # ORIGINAL FUNCTIONS - MAINTAINED FOR BACKWARD COMPATIBILITY
    
    def convert_vcf_to_plink_chunked(self, vcf_file, plink_base):
        """Convert VCF to PLINK with chunking for very large files - ORIGINAL FUNCTION"""
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
        """Extract chromosome list from VCF file - ORIGINAL FUNCTION"""
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
        """Process VCF by chromosome in parallel - ORIGINAL FUNCTION"""
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
        """Process a single chromosome - ORIGINAL FUNCTION"""
        cmd = (
            f"{self.config['paths']['plink']} --vcf {vcf_file} "
            f"--chr {chromosome} --make-bed --out {output_base}"
        )
        self.run_command(cmd, f"Processing chromosome {chromosome}")
    
    def merge_plink_files(self, plink_base, chromosomes):
        """Merge PLINK files from all chromosomes - ORIGINAL FUNCTION"""
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
        """Clean up temporary chromosome files - ORIGINAL FUNCTION"""
        for chrom in chromosomes:
            for ext in ['.bed', '.bim', '.fam', '.log']:
                file_path = f"{plink_base}_chr{chrom}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    def apply_plink_filters(self, plink_base, output_dir):
        """Apply comprehensive filters to PLINK data - ORIGINAL FUNCTION"""
        logger.info("🔧 Applying PLINK filters...")
        
        filtered_base = output_dir / "filtered_genotypes"
        
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
        
        return str(filtered_base)
    
    def check_system_resources(self):
        """Check if system has sufficient resources for large dataset processing - ORIGINAL FUNCTION"""
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
        """Validate input genotype file - ENHANCED for tensorQTL"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Genotype file not found: {input_file}")
        
        # Check file size
        file_size = os.path.getsize(input_file) / (1024**3)  # GB
        if file_size == 0:
            raise ValueError(f"Genotype file is empty: {input_file}")
        
        logger.info(f"📊 Input file size: {file_size:.2f} GB")
        
        # Check format compatibility for tensorQTL
        input_format = self.detect_input_format(input_file)
        compatible_formats = ['vcf', 'vcf.gz', 'bcf', 'plink_bed']
        if input_format not in compatible_formats:
            logger.warning(f"⚠️ Input format {input_format} may not be optimal for tensorQTL")
        
        # For very large files, do a quick format check without loading entire file
        if file_size > 10:
            self.quick_validate_large_file(input_file)
    
    def quick_validate_large_file(self, input_file):
        """Quick validation for very large files - ENHANCED for tensorQTL"""
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
        
        # For VCF files, check header compatibility for tensorQTL
        if input_file.endswith(('.vcf', '.vcf.gz')):
            try:
                cmd = f"{self.config['paths']['bcftools']} view -h {input_file} | head -5"
                result = self.run_command(cmd, "Checking VCF header for tensorQTL", check=False)
                if result.returncode != 0:
                    raise ValueError("VCF file header cannot be read - tensorQTL may fail")
                if '#CHROM' not in result.stdout:
                    raise ValueError("VCF file missing #CHROM line - invalid format for tensorQTL")
                logger.info("✅ VCF header validation passed for tensorQTL")
            except Exception as e:
                logger.warning(f"⚠️ VCF header check failed: {e}")
    
    def detect_input_format(self, input_file):
        """Detect the format of the input genotype file - ORIGINAL FUNCTION"""
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
        """Detect file format by examining content - ORIGINAL FUNCTION"""
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
        """Convert various formats to standard VCF.gz - ORIGINAL FUNCTION"""
        output_file = output_dir / "standardized.vcf.gz"
        
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
                return str(output_file)
                
        elif input_format == 'bcf':
            # Convert BCF to VCF.gz
            temp_vcf = output_dir / "temp_standardized.vcf"
            self.run_command(
                f"{self.config['paths']['bcftools']} view {input_file} -Ov -o {temp_vcf}",
                "Converting BCF to VCF"
            )
            self.run_command(
                f"{self.config['paths']['bgzip']} -c {temp_vcf} > {output_file}",
                "Compressing VCF file"
            )
            os.remove(temp_vcf)
            return str(output_file)
            
        elif input_format == 'plink_bed':
            # Convert PLINK to VCF.gz
            base_name = input_file.replace('.bed', '')
            temp_vcf = output_dir / "temp_plink.vcf"
            
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
                    
            return str(output_file)
            
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
    
    def apply_preprocessing(self, input_file, output_dir):
        """Apply all configured pre-processing steps - ORIGINAL FUNCTION"""
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
        """Apply comprehensive variant filtering - ORIGINAL FUNCTION"""
        output_file = output_dir / "filtered.vcf.gz"
        
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
        
        return str(output_file)
    
    def handle_multiallelic(self, input_file, output_dir):
        """Handle multi-allelic sites - ORIGINAL FUNCTION"""
        output_file = output_dir / "biallelic.vcf.gz"
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
        
        return str(output_file)
    
    def normalize_chromosomes(self, input_file, output_dir):
        """Normalize chromosome naming convention - ORIGINAL FUNCTION"""
        output_file = output_dir / "chrom_normalized.vcf.gz"
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
            current_file = str(output_file)
        elif prefix_config == 'none':
            # Remove 'chr' prefix if present
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs <(echo -e 'chr1\\t1\\nchr2\\t2\\nchr3\\t3\\nchr4\\t4\\nchr5\\t5\\nchr6\\t6\\nchr7\\t7\\nchr8\\t8\\nchr9\\t9\\nchr10\\t10\\nchr11\\t11\\nchr12\\t12\\nchr13\\t13\\nchr14\\t14\\nchr15\\t15\\nchr16\\t16\\nchr17\\t17\\nchr18\\t18\\nchr19\\t19\\nchr20\\t20\\nchr21\\t21\\nchr22\\t22\\nchrX\\tX\\nchrY\\tY\\nchrM\\tMT') "
                f"-Oz -o {output_file}",
                "Removing 'chr' prefix from chromosomes"
            )
            current_file = str(output_file)
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
        """Auto-detect chromosome format and standardize - ORIGINAL FUNCTION"""
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
            output_file = output_dir / "chrom_normalized.vcf.gz"
            self.run_command(
                f"{self.config['paths']['bcftools']} annotate {input_file} "
                f"--rename-chrs <(echo -e '1\\tchr1\\n2\\tchr2\\n3\\tchr3\\n4\\tchr4\\n5\\tchr5\\n6\\tchr6\\n7\\tchr7\\n8\\tchr8\\n9\\tchr9\\n10\\tchr10\\n11\\tchr11\\n12\\tchr12\\n13\\tchr13\\n14\\tchr14\\n15\\tchr15\\n16\\tchr16\\n17\\tchr17\\n18\\tchr18\\n19\\tchr19\\n20\\tchr20\\n21\\tchr21\\n22\\tchr22\\nX\\tchrX\\nY\\tchrY\\nMT\\tchrM') "
                f"-Oz -o {output_file}",
                "Auto-adding 'chr' prefix to chromosomes"
            )
            return str(output_file)
    
    def remove_phasing(self, input_file, output_dir):
        """Remove phasing information from genotypes - ORIGINAL FUNCTION"""
        output_file = output_dir / "unphased.vcf.gz"
        
        self.run_command(
            f"{self.config['paths']['bcftools']} view {input_file} -U -Oz -o {output_file}",
            "Removing phasing information"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing unphased VCF"
        )
        
        return str(output_file)
    
    def normalize_indels(self, input_file, output_dir):
        """Normalize INDEL representations (requires reference genome) - ORIGINAL FUNCTION"""
        output_file = output_dir / "indel_normalized.vcf.gz"
        
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
        
        return str(output_file)
    
    def remove_duplicates(self, input_file, output_dir):
        """Remove duplicate variants - ORIGINAL FUNCTION"""
        output_file = output_dir / "deduplicated.vcf.gz"
        
        self.run_command(
            f"{self.config['paths']['bcftools']} norm {input_file} "
            f"-d all -Oz -o {output_file}",
            "Removing duplicate variants"
        )
        
        self.run_command(
            f"{self.config['paths']['tabix']} -p vcf {output_file}",
            "Indexing deduplicated VCF"
        )
        
        return str(output_file)
    
    def left_align_variants(self, input_file, output_dir):
        """Left-align variants (requires reference genome) - ORIGINAL FUNCTION"""
        output_file = output_dir / "left_aligned.vcf.gz"
        
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
        
        return str(output_file)
    
    def generate_qc_reports(self, vcf_file, output_dir):
        """Generate comprehensive QC reports - ORIGINAL FUNCTION"""
        logger.info("📊 Generating QC reports...")
        
        # Use directory manager for QC reports
        qc_dir = self.dm.get_directory('reports', 'qc_reports/genotype_qc')
        
        # Basic stats
        self.run_command(
            f"{self.config['paths']['bcftools']} stats {vcf_file} > {qc_dir}/vcf_stats.txt",
            "Generating VCF statistics"
        )
        
        # Sample statistics
        self.run_command(
            f"{self.config['paths']['bcftools']} query -l {vcf_file} | wc -l > {qc_dir}/sample_count.txt",
            "Counting samples"
        )
        
        # Variant statistics
        self.run_command(
            f"{self.config['paths']['bcftools']} view -H {vcf_file} | wc -l > {qc_dir}/variant_count.txt",
            "Counting variants"
        )
        
        logger.info("✅ QC reports generated")
    
    def prepare_final_file(self, input_file, output_dir):
        """Prepare final genotype file for analysis - ORIGINAL FUNCTION"""
        # Use directory manager for final genotypes location
        final_dir = self.dm.get_directory('processed_data', 'genotypes/final')
        final_file = final_dir / "final_genotypes.vcf.gz"
        
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
        self.validate_vcf(str(final_file))
        
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
        
        return str(final_file)
    
    def validate_vcf(self, vcf_file):
        """Validate VCF file structure - ORIGINAL FUNCTION"""
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
        """Run shell command with comprehensive error handling - ORIGINAL FUNCTION"""
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

# Modular pipeline function
def process_genotypes(config):
    """
    Main function for genotype processing module in the modular pipeline
    Returns: bool (success)
    """
    try:
        logger.info("🚀 Starting genotype processing module...")
        
        # Get input parameters from config
        input_file = config['input_files']['genotypes']
        output_dir = config['results_dir']
        
        # Initialize processor
        processor = GenotypeProcessor(config)
        
        # Process genotypes
        result_file = processor.process_genotypes(input_file, output_dir)
        
        if result_file and os.path.exists(result_file):
            logger.info(f"✅ Genotype processing completed successfully: {result_file}")
            return True
        else:
            logger.error("❌ Genotype processing failed - no output file generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Genotype processing module failed: {e}")
        logger.debug(traceback.format_exc())
        return False

# Additional tensorQTL-specific utility functions
def validate_tensorqtl_genotypes(genotype_file):
    """Validate that genotype files are ready for tensorQTL analysis"""
    logger.info(f"🔍 Validating tensorQTL genotype files: {genotype_file}")
    
    try:
        if genotype_file.endswith('.bed'):
            base_name = genotype_file.replace('.bed', '')
            required_files = [f'{base_name}.bed', f'{base_name}.bim', f'{base_name}.fam']
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.error(f"❌ Missing tensorQTL file: {file_path}")
                    return False
            
            # Check if files are readable
            try:
                fam_df = pd.read_csv(f'{base_name}.fam', sep='\s+', header=None)
                bim_df = pd.read_csv(f'{base_name}.bim', sep='\s+', header=None)
                
                logger.info(f"✅ TensorQTL validation: {len(fam_df)} samples, {len(bim_df)} variants")
                return True
                
            except Exception as e:
                logger.error(f"❌ TensorQTL file reading failed: {e}")
                return False
        else:
            logger.warning(f"⚠️ Non-PLINK format may have reduced tensorQTL performance: {genotype_file}")
            return True
            
    except Exception as e:
        logger.error(f"❌ TensorQTL genotype validation failed: {e}")
        return False

def get_tensorqtl_sample_list(genotype_file):
    """Extract sample list from genotype file for tensorQTL alignment"""
    try:
        if genotype_file.endswith('.bed'):
            base_name = genotype_file.replace('.bed', '')
            fam_file = f'{base_name}.fam'
            
            if os.path.exists(fam_file):
                fam_df = pd.read_csv(fam_file, sep='\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Pheno'])
                return fam_df['IID'].tolist()
        
        # For VCF files, use bcftools
        elif genotype_file.endswith(('.vcf', '.vcf.gz', '.bcf')):
            cmd = f"bcftools query -l {genotype_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return [s.strip() for s in result.stdout.split('\n') if s.strip()]
        
        logger.warning("❌ Could not extract sample list from genotype file")
        return []
        
    except Exception as e:
        logger.error(f"❌ Error extracting tensorQTL sample list: {e}")
        return []

# Maintain backward compatibility
if __name__ == "__main__":
    # Load config and run as standalone script
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    success = process_genotypes(config)
    sys.exit(0 if success else 1)