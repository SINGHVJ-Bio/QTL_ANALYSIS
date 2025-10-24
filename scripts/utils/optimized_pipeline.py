#!/usr/bin/env python3
"""
Optimized QTL Pipeline for Multi-Core CPU Systems
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger('QTLPipeline')

class OptimizedQTLAnalysis:
    def __init__(self, config):
        self.config = config
        self.num_cores = config['performance']['num_threads']
        self.setup_parallel_environment()
        
    def setup_parallel_environment(self):
        """Setup optimal parallel processing environment"""
        # Set environment variables for external tools
        os.environ['OMP_NUM_THREADS'] = str(self.num_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.num_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.num_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.num_cores)
        
        logger.info(f"🔧 Parallel environment setup: {self.num_cores} cores")
        
    def parallel_chromosome_processing(self, function, data, description=""):
        """Process data in parallel by chromosomes"""
        chromosomes = self.get_chromosomes()
        max_workers = min(self.config['performance'].get('max_concurrent_chromosomes', 4), 
                         len(chromosomes), self.num_cores)
        
        logger.info(f"🔄 Parallel {description} across {len(chromosomes)} chromosomes using {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for chrom in chromosomes:
                future = executor.submit(function, chrom, data)
                futures[future] = chrom
            
            results = {}
            for future in futures:
                chrom = futures[future]
                try:
                    results[chrom] = future.result(timeout=3600)  # 1 hour timeout
                    logger.info(f"✅ Completed chromosome {chrom}")
                except Exception as e:
                    logger.error(f"❌ Failed chromosome {chrom}: {e}")
                    results[chrom] = None
        
        return results
    
    def get_chromosomes(self):
        """Get list of chromosomes to process"""
        # Auto-detect chromosomes from genotype file or use standard set
        standard_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        return standard_chromosomes
    
    def optimized_tensorqtl_analysis(self, genotype_file, phenotype_file, covariates_file, output_dir):
        """Run tensorQTL analysis with CPU optimizations"""
        import tensorqtl
        from tensorqtl import genotypeio, cis, trans
        import torch
        
        # Set PyTorch for CPU optimization
        torch.set_num_threads(self.num_cores)
        device = torch.device("cpu")
        
        logger.info(f"🧬 Running tensorQTL on CPU with {self.num_cores} threads")
        
        # Load data with optimizations
        phenotype_df = self.load_phenotypes_optimized(phenotype_file)
        covariates_df = self.load_covariates_optimized(covariates_file)
        
        # Process genotypes in chunks for memory efficiency
        genotype_loader = self.create_optimized_genotype_loader(genotype_file)
        
        # Run cis-QTL analysis with optimizations
        cis_results = self.run_optimized_cis_analysis(
            genotype_loader, phenotype_df, covariates_df, output_dir, device
        )
        
        return cis_results
    
    def load_phenotypes_optimized(self, phenotype_file):
        """Load phenotypes with memory optimizations"""
        # Use efficient data types
        dtype_dict = {f'sample_{i}': np.float32 for i in range(1000)}  # Adjust based on your data
        
        return pd.read_csv(phenotype_file, sep='\t', index_col=0, 
                          dtype=dtype_dict, engine='c')
    
    def load_covariates_optimized(self, covariates_file):
        """Load covariates with memory optimizations"""
        return pd.read_csv(covariates_file, sep='\t', index_col=0, 
                          dtype=np.float32, engine='c')
    
    def create_optimized_genotype_loader(self, genotype_file):
        """Create optimized genotype data loader"""
        # Implementation depends on your genotype format
        # This would interface with tensorQTL's genotypeio
        pass
    
    def run_optimized_cis_analysis(self, genotype_loader, phenotype_df, covariates_df, output_dir, device):
        """Run cis-QTL analysis with CPU optimizations"""
        import tensorqtl
        from tensorqtl import cis
        
        # Get optimization parameters from config
        batch_size = self.config['performance']['tensorqtl_batch_size']
        chunk_size = self.config['performance']['tensorqtl_chunk_size']
        
        # Run analysis with optimized parameters
        cis_df = cis.map_nominal(genotype_loader, phenotype_df, covariates_df,
                                output_dir=output_dir,
                                maf_threshold=self.config['tensorqtl']['maf_threshold'],
                                window=self.config['tensorqtl']['cis_window'],
                                batch_size=batch_size,
                                chunk_size=chunk_size,
                                device=device)
        
        return cis_df

def optimize_system_settings():
    """Apply system-level optimizations for QTL analysis"""
    
    # CPU affinity settings (Linux/Mac)
    if hasattr(os, 'sched_setaffinity'):
        try:
            # Use all available cores
            available_cores = list(range(mp.cpu_count()))
            os.sched_setaffinity(0, available_cores)
            logger.info(f"🔧 Set CPU affinity to use all {len(available_cores)} cores")
        except Exception as e:
            logger.warning(f"⚠️ Could not set CPU affinity: {e}")
    
    # Increase file descriptor limit (Unix systems)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard), hard))
        logger.info(f"🔧 Increased file descriptor limit to {min(8192, hard)}")
    except Exception as e:
        logger.warning(f"⚠️ Could not increase file descriptor limit: {e}")

# Enhanced main runner with multi-core optimizations
def run_optimized_pipeline(config_file):
    """Run the optimized multi-core pipeline"""
    
    # Apply system optimizations
    optimize_system_settings()
    
    # Load configuration
    with open(config_file, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Initialize optimized pipeline
    pipeline = OptimizedQTLAnalysis(config)
    
    # Run the pipeline
    logger.info("🚀 Starting Optimized Multi-Core QTL Pipeline")
    logger.info(f"💻 Using {pipeline.num_cores} CPU cores")
    
    # Your existing pipeline logic here, but using optimized methods
    # [Rest of your pipeline implementation...]