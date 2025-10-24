#!/usr/bin/env python3
"""
CPU-optimized tensorQTL wrapper for maximum multi-core performance
"""

import os
import torch
import pandas as pd
import numpy as np
import tensorqtl
from tensorqtl import genotypeio, cis, trans
import logging

logger = logging.getLogger('QTLPipeline')

class CPUOptimizedTensorQTL:
    def __init__(self, config):
        self.config = config
        self.num_threads = config['performance']['num_threads']
        self.setup_torch_optimizations()
        
    def setup_torch_optimizations(self):
        """Setup PyTorch for optimal CPU performance"""
        torch.set_num_threads(self.num_threads)
        torch.set_float32_matmul_precision('medium')  # Balance speed and precision
        
        # Enable other optimizations
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        
        logger.info(f"🔧 PyTorch optimized for {self.num_threads} CPU threads")
        
    def run_multi_chromosome_analysis(self, genotype_path, phenotype_df, covariates_df, output_dir):
        """Run QTL analysis across multiple chromosomes in parallel"""
        from concurrent.futures import ProcessPoolExecutor
        
        chromosomes = self.get_chromosome_list(genotype_path)
        max_workers = min(self.config['performance'].get('max_concurrent_chromosomes', 4), 
                         len(chromosomes))
        
        logger.info(f"🔄 Processing {len(chromosomes)} chromosomes with {max_workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chrom in chromosomes:
                future = executor.submit(
                    self.analyze_single_chromosome,
                    chrom, genotype_path, phenotype_df, covariates_df, output_dir
                )
                futures.append((future, chrom))
            
            # Collect results
            results = {}
            for future, chrom in futures:
                try:
                    results[chrom] = future.result(timeout=7200)  # 2 hour timeout per chromosome
                    logger.info(f"✅ Completed chromosome {chrom}")
                except Exception as e:
                    logger.error(f"❌ Chromosome {chrom} failed: {e}")
                    results[chrom] = None
        
        return results
    
    def analyze_single_chromosome(self, chrom, genotype_path, phenotype_df, covariates_df, output_dir):
        """Analyze a single chromosome with optimized settings"""
        # Set thread count for this process
        torch.set_num_threads(self.num_threads)
        
        # Load genotype data for this chromosome
        genotype_df = self.load_chromosome_genotypes(genotype_path, chrom)
        
        if genotype_df is None or len(genotype_df) == 0:
            logger.warning(f"⚠️ No variants found for chromosome {chrom}")
            return None
        
        # Run cis-QTL analysis
        cis_df = cis.map_nominal(
            genotype_df, 
            phenotype_df, 
            covariates_df,
            window=self.config['tensorqtl']['cis_window'],
            maf_threshold=self.config['tensorqtl']['maf_threshold'],
            batch_size=self.config['performance']['tensorqtl_batch_size'],
            chunk_size=self.config['performance']['tensorqtl_chunk_size']
        )
        
        # Save results
        output_file = os.path.join(output_dir, f"chr{chrom}_cis_qtl.txt")
        cis_df.to_csv(output_file, sep='\t')
        
        return {
            'chromosome': chrom,
            'output_file': output_file,
            'num_variants': len(genotype_df),
            'num_associations': len(cis_df)
        }
    
    def load_chromosome_genotypes(self, genotype_path, chrom):
        """Load genotypes for a specific chromosome"""
        # Implementation depends on your genotype format
        # This could use tensorQTL's genotypeio or your custom loader
        try:
            # Example for PLINK format
            plink_file = f"{genotype_path}.chr{chrom}"
            if os.path.exists(plink_file + ".bed"):
                genotype_df = genotypeio.load_genotype(plink_file)
                return genotype_df
        except Exception as e:
            logger.error(f"❌ Error loading genotypes for chr{chrom}: {e}")
        
        return None
    
    def get_chromosome_list(self, genotype_path):
        """Detect available chromosomes from genotype data"""
        # Auto-detect chromosomes or use predefined list
        chromosomes = []
        for chrom in [str(i) for i in range(1, 23)] + ['X', 'Y']:
            plink_file = f"{genotype_path}.chr{chrom}"
            if os.path.exists(plink_file + ".bed"):
                chromosomes.append(chrom)
        
        if not chromosomes:
            # Fallback to standard chromosomes
            chromosomes = [str(i) for i in range(1, 23)]
            
        logger.info(f"🔍 Found genotypes for chromosomes: {chromosomes}")
        return chromosomes