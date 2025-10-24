#!/usr/bin/env python3
"""
Fine-mapping utilities for identifying credible sets of causal variants
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import pandas as pd
import numpy as np
import logging
import subprocess
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

def map_qtl_type_to_config_key(qtl_type):
    """Map QTL analysis types to config file keys"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

class FineMapping:
    def __init__(self, config):
        self.config = config
        self.finemap_config = config.get('fine_mapping', {})
        
    def run_fine_mapping(self, qtl_results_file, vcf_file, output_dir, qtl_type):
        """Run fine-mapping for significant QTL regions for specific QTL type"""
        logger.info(f"🔍 Running {qtl_type} fine-mapping analysis...")
        
        if not self.finemap_config.get('enable', False):
            logger.info("ℹ️ Fine-mapping disabled in config")
            return {}
        
        try:
            # Read QTL results
            qtl_df = pd.read_csv(qtl_results_file, sep='\t')
            
            if qtl_df.empty:
                logger.warning(f"No {qtl_type} results for fine-mapping")
                return {}
            
            # Filter significant associations
            fdr_threshold = self.config['qtl'].get('fdr_threshold', 0.05)
            if 'fdr' in qtl_df.columns:
                significant_qtls = qtl_df[qtl_df['fdr'] < fdr_threshold]
            else:
                # Use p-value threshold if FDR not available
                p_threshold = 1e-5
                significant_qtls = qtl_df[qtl_df['p_value'] < p_threshold]
            
            if significant_qtls.empty:
                logger.warning(f"No significant {qtl_type} for fine-mapping")
                return {}
            
            logger.info(f"🔧 Fine-mapping {len(significant_qtls)} significant {qtl_type}")
            
            # Group by gene
            results = {}
            for gene_id, gene_qtls in significant_qtls.groupby('gene_id'):
                if len(gene_qtls) < 2:  # Need multiple variants for fine-mapping
                    continue
                
                logger.info(f"🔬 Fine-mapping {qtl_type} gene: {gene_id}")
                gene_results = self.fine_map_gene(gene_qtls, vcf_file, output_dir, gene_id, qtl_type)
                results[gene_id] = gene_results
            
            # Generate summary
            self.generate_finemap_summary(results, output_dir, qtl_type)
            
            logger.info(f"✅ {qtl_type} fine-mapping completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ {qtl_type} fine-mapping failed: {e}")
            return {}
    
    def fine_map_gene(self, gene_qtls, vcf_file, output_dir, gene_id, qtl_type):
        """Run fine-mapping for a single gene"""
        try:
            method = self.finemap_config.get('method', 'susie')
            credible_threshold = self.finemap_config.get('credible_set_threshold', 0.95)
            max_causal = self.finemap_config.get('max_causal_variants', 5)
            
            if method == 'susie':
                return self.run_susie_finemap(gene_qtls, credible_threshold, max_causal, gene_id, output_dir, qtl_type)
            elif method == 'finemap':
                return self.run_finemap(gene_qtls, credible_threshold, max_causal, gene_id, output_dir, qtl_type)
            else:
                logger.warning(f"Unknown fine-mapping method: {method}")
                return {'error': f'Unknown method: {method}'}
                
        except Exception as e:
            logger.error(f"❌ {qtl_type} fine-mapping failed for {gene_id}: {e}")
            return {'error': str(e)}
    
    def run_susie_finemap(self, gene_qtls, credible_threshold, max_causal, gene_id, output_dir, qtl_type):
        """Run SuSiE fine-mapping (simplified implementation)"""
        try:
            # This is a simplified mock implementation
            # In practice, you would use the susieR package or similar
            
            # Prepare data for fine-mapping
            variants = gene_qtls['variant_id'].tolist()
            p_values = gene_qtls['p_value'].tolist()
            betas = gene_qtls.get('beta', np.random.normal(0, 0.1, len(gene_qtls))).tolist()
            ses = gene_qtls.get('se', np.abs(betas) / stats.norm.ppf(1 - np.array(p_values)/2)).tolist()
            
            # Calculate Bayes factors (simplified)
            z_scores = np.abs(stats.norm.ppf(np.array(p_values)/2))
            bayes_factors = np.exp(z_scores**2 / 2)
            
            # Calculate posterior probabilities
            posterior_probs = bayes_factors / np.sum(bayes_factors)
            
            # Identify credible set
            sorted_indices = np.argsort(posterior_probs)[::-1]
            cumulative_prob = 0
            credible_set = []
            
            for idx in sorted_indices:
                cumulative_prob += posterior_probs[idx]
                credible_set.append({
                    'variant_id': variants[idx],
                    'posterior_prob': posterior_probs[idx],
                    'p_value': p_values[idx],
                    'beta': betas[idx],
                    'se': ses[idx],
                    'qtl_type': qtl_type
                })
                if cumulative_prob >= credible_threshold:
                    break
            
            results = {
                'gene_id': gene_id,
                'qtl_type': qtl_type,
                'method': 'susie',
                'credible_set_threshold': credible_threshold,
                'credible_set_size': len(credible_set),
                'credible_set_variants': credible_set,
                'total_variants': len(variants),
                'max_posterior_prob': max(posterior_probs) if posterior_probs else 0
            }
            
            # Save results
            output_file = os.path.join(output_dir, f"finemap_{qtl_type}_susie_{gene_id}.txt")
            self.save_finemap_results(results, output_file)
            results['results_file'] = output_file
            
            logger.info(f"✅ {qtl_type} {gene_id}: Credible set size = {len(credible_set)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {qtl_type} SuSiE fine-mapping failed for {gene_id}: {e}")
            return {'error': str(e)}
    
    def run_finemap(self, gene_qtls, credible_threshold, max_causal, gene_id, output_dir, qtl_type):
        """Run FINEMAP (simplified implementation)"""
        try:
            # Mock implementation - similar to SuSiE but with different assumptions
            
            variants = gene_qtls['variant_id'].tolist()
            p_values = gene_qtls['p_value'].tolist()
            
            # Simplified FINEMAP-like calculation
            # In practice, you would use the FINEMAP software
            log_bf = -np.log(p_values)  # Simplified Bayes factor
            
            # Normalize to posterior probabilities
            posterior_probs = np.exp(log_bf - np.max(log_bf))
            posterior_probs = posterior_probs / np.sum(posterior_probs)
            
            # Identify credible set
            sorted_indices = np.argsort(posterior_probs)[::-1]
            cumulative_prob = 0
            credible_set = []
            
            for idx in sorted_indices:
                cumulative_prob += posterior_probs[idx]
                credible_set.append({
                    'variant_id': variants[idx],
                    'posterior_prob': posterior_probs[idx],
                    'p_value': p_values[idx],
                    'qtl_type': qtl_type
                })
                if cumulative_prob >= credible_threshold:
                    break
            
            results = {
                'gene_id': gene_id,
                'qtl_type': qtl_type,
                'method': 'finemap',
                'credible_set_threshold': credible_threshold,
                'credible_set_size': len(credible_set),
                'credible_set_variants': credible_set,
                'total_variants': len(variants),
                'max_posterior_prob': max(posterior_probs) if posterior_probs else 0
            }
            
            # Save results
            output_file = os.path.join(output_dir, f"finemap_{qtl_type}_finemap_{gene_id}.txt")
            self.save_finemap_results(results, output_file)
            results['results_file'] = output_file
            
            logger.info(f"✅ {qtl_type} {gene_id}: Credible set size = {len(credible_set)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {qtl_type} FINEMAP failed for {gene_id}: {e}")
            return {'error': str(e)}
    
    def save_finemap_results(self, results, output_file):
        """Save fine-mapping results to file"""
        try:
            with open(output_file, 'w') as f:
                f.write(f"# Fine-mapping results for {results['gene_id']}\n")
                f.write(f"# QTL Type: {results['qtl_type']}\n")
                f.write(f"# Method: {results['method']}\n")
                f.write(f"# Credible set threshold: {results['credible_set_threshold']}\n")
                f.write(f"# Credible set size: {results['credible_set_size']}\n")
                f.write(f"# Total variants: {results['total_variants']}\n")
                f.write(f"# Max posterior probability: {results['max_posterior_prob']:.4f}\n\n")
                
                f.write("variant_id\tposterior_prob\tp_value\tbeta\tse\tqtl_type\n")
                for variant in results['credible_set_variants']:
                    f.write(f"{variant['variant_id']}\t{variant['posterior_prob']:.6f}\t")
                    f.write(f"{variant['p_value']:.2e}\t{variant.get('beta', 'NA')}\t{variant.get('se', 'NA')}\t{variant.get('qtl_type', 'NA')}\n")
                    
        except Exception as e:
            logger.error(f"❌ Error saving fine-mapping results: {e}")
    
    def generate_finemap_summary(self, results, output_dir, qtl_type):
        """Generate summary of fine-mapping results"""
        logger.info(f"📊 Generating {qtl_type} fine-mapping summary...")
        
        try:
            summary_file = os.path.join(output_dir, f"{qtl_type}_fine_mapping_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"{qtl_type.upper()} Fine-mapping Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                
                successful_genes = 0
                total_credible_sets = 0
                credible_set_sizes = []
                
                for gene_id, result in results.items():
                    if 'error' in result:
                        f.write(f"Gene: {gene_id}\n")
                        f.write(f"  Status: FAILED - {result['error']}\n\n")
                        continue
                    
                    successful_genes += 1
                    credible_set_size = result.get('credible_set_size', 0)
                    total_credible_sets += credible_set_size
                    credible_set_sizes.append(credible_set_size)
                    
                    f.write(f"Gene: {gene_id}\n")
                    f.write(f"  Method: {result.get('method', 'N/A')}\n")
                    f.write(f"  Credible set size: {credible_set_size}\n")
                    f.write(f"  Total variants: {result.get('total_variants', 0)}\n")
                    f.write(f"  Max posterior prob: {result.get('max_posterior_prob', 0):.4f}\n")
                    f.write(f"  Results file: {result.get('results_file', 'N/A')}\n\n")
                
                f.write("Overall Summary:\n")
                f.write(f"  Successfully fine-mapped genes: {successful_genes}\n")
                f.write(f"  Total credible set variants: {total_credible_sets}\n")
                if credible_set_sizes:
                    f.write(f"  Mean credible set size: {np.mean(credible_set_sizes):.2f}\n")
                    f.write(f"  Median credible set size: {np.median(credible_set_sizes):.2f}\n")
                    f.write(f"  Min credible set size: {min(credible_set_sizes)}\n")
                    f.write(f"  Max credible set size: {max(credible_set_sizes)}\n")
            
            logger.info(f"💾 {qtl_type} fine-mapping summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error generating {qtl_type} fine-mapping summary: {e}")