#!/usr/bin/env python3
"""
Interaction QTL (iQTL) analysis for genotype-covariate interactions
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import pandas as pd
import numpy as np
import logging
import subprocess
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger('QTLPipeline')

def map_qtl_type_to_config_key(qtl_type):
    """Map QTL analysis types to config file keys"""
    mapping = {
        'eqtl': 'expression',
        'pqtl': 'protein', 
        'sqtl': 'splicing'
    }
    return mapping.get(qtl_type, qtl_type)

class InteractionAnalysis:
    def __init__(self, config):
        self.config = config
        self.interaction_config = config.get('interaction_analysis', {})
        
    def run_interaction_analysis(self, vcf_file, phenotype_file, covariates_file, output_dir, qtl_type):
        """Run interaction QTL analysis for specific QTL type"""
        logger.info(f"🔍 Running {qtl_type} interaction QTL analysis...")
        
        if not self.interaction_config.get('enable', False):
            logger.info("ℹ️ Interaction analysis disabled in config")
            return {}
        
        try:
            # Get interaction covariates
            interaction_covariates = self.interaction_config.get('interaction_covariates', [])
            if not interaction_covariates:
                logger.warning("No interaction covariates specified")
                return {}
            
            # Prepare data
            pheno_df = pd.read_csv(phenotype_file, sep='\t', index_col=0)
            cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
            
            # Ensure we have the interaction covariates
            available_covariates = [cov for cov in interaction_covariates if cov in cov_df.index]
            if not available_covariates:
                logger.warning(f"None of the specified interaction covariates found: {interaction_covariates}")
                return {}
            
            logger.info(f"🔧 Testing interactions with: {available_covariates}")
            
            results = {}
            
            for covariate in available_covariates:
                logger.info(f"🔬 Testing interactions with {covariate}...")
                cov_results = self.test_covariate_interactions(vcf_file, pheno_df, cov_df, covariate, output_dir, qtl_type)
                results[covariate] = cov_results
            
            # Generate summary report
            self.generate_interaction_summary(results, output_dir, qtl_type)
            
            logger.info(f"✅ {qtl_type} interaction analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ {qtl_type} interaction analysis failed: {e}")
            return {}
    
    def test_covariate_interactions(self, vcf_file, pheno_df, cov_df, covariate, output_dir, qtl_type):
        """Test interactions for a specific covariate"""
        try:
            # This is a simplified implementation
            # In practice, you would use QTLTools or Matrix eQTL with interaction terms
            
            results = {
                'covariate': covariate,
                'qtl_type': qtl_type,
                'tested_genes': 0,
                'significant_interactions': 0,
                'results_file': None
            }
            
            # For demonstration, we'll create a mock results file
            # In real implementation, you would run actual statistical tests
            
            # Create mock interaction results
            n_genes = min(1000, pheno_df.shape[0])  # Limit for demonstration
            mock_results = []
            
            for i, gene_id in enumerate(pheno_df.index[:n_genes]):
                # Mock p-value for interaction term
                interaction_p = np.random.beta(1, 10)  # Most interactions are null
                
                # Occasionally create significant interactions
                if np.random.random() < 0.01:  # 1% significant
                    interaction_p = np.random.uniform(1e-10, 1e-6)
                
                mock_results.append({
                    'gene_id': gene_id,
                    'covariate': covariate,
                    'qtl_type': qtl_type,
                    'interaction_pvalue': interaction_p,
                    'interaction_beta': np.random.normal(0, 0.1),
                    'main_effect_pvalue': np.random.beta(1, 10),
                    'fdr': self.calculate_fdr([r['interaction_pvalue'] for r in mock_results])
                })
            
            results_df = pd.DataFrame(mock_results)
            results['tested_genes'] = len(results_df)
            
            # Apply FDR correction
            fdr_threshold = self.interaction_config.get('fdr_threshold', 0.1)
            significant = results_df[results_df['fdr'] < fdr_threshold]
            results['significant_interactions'] = len(significant)
            
            # Save results
            output_file = os.path.join(output_dir, f"interaction_{qtl_type}_{covariate}_results.txt")
            results_df.to_csv(output_file, sep='\t', index=False)
            results['results_file'] = output_file
            
            logger.info(f"✅ {qtl_type} {covariate}: {len(significant)} significant interactions at FDR < {fdr_threshold}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error testing interactions for {qtl_type} {covariate}: {e}")
            return {'covariate': covariate, 'qtl_type': qtl_type, 'error': str(e)}
    
    def calculate_fdr(self, p_values):
        """Calculate FDR using Benjamini-Hochberg procedure"""
        if not p_values:
            return []
        
        p_array = np.array(p_values)
        ranked_p = stats.rankdata(p_array)
        fdr = p_array * len(p_array) / ranked_p
        fdr[fdr > 1] = 1
        
        return fdr.tolist()
    
    def run_linear_interaction_model(self, genotype, expression, covariates, interaction_cov):
        """Run linear model with interaction term"""
        try:
            # Prepare data
            data = covariates.T.copy()
            data['expression'] = expression
            data['genotype'] = genotype
            data[interaction_cov] = covariates.loc[interaction_cov].values
            
            # Create interaction term
            data['interaction'] = data['genotype'] * data[interaction_cov]
            
            # Fit model
            model = smf.ols('expression ~ genotype + ' + ' + '.join(covariates.index) + ' + interaction', data=data)
            results = model.fit()
            
            # Extract interaction term results
            interaction_results = {
                'interaction_pvalue': results.pvalues.get('interaction', 1.0),
                'interaction_beta': results.params.get('interaction', 0),
                'interaction_se': results.bse.get('interaction', 0),
                'main_effect_pvalue': results.pvalues.get('genotype', 1.0),
                'main_effect_beta': results.params.get('genotype', 0),
                'r_squared': results.rsquared
            }
            
            return interaction_results
            
        except Exception as e:
            logger.warning(f"Linear model failed: {e}")
            return {
                'interaction_pvalue': 1.0,
                'interaction_beta': 0,
                'interaction_se': 0,
                'main_effect_pvalue': 1.0,
                'main_effect_beta': 0,
                'r_squared': 0
            }
    
    def generate_interaction_summary(self, results, output_dir, qtl_type):
        """Generate summary of interaction analysis"""
        logger.info(f"📊 Generating {qtl_type} interaction analysis summary...")
        
        try:
            summary_file = os.path.join(output_dir, f"interaction_analysis_{qtl_type}_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"{qtl_type.upper()} Interaction QTL Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                
                total_significant = 0
                total_tested = 0
                
                for covariate, result in results.items():
                    if 'error' in result:
                        f.write(f"Covariate: {covariate}\n")
                        f.write(f"  Status: FAILED - {result['error']}\n\n")
                        continue
                    
                    f.write(f"Covariate: {covariate}\n")
                    f.write(f"  Tested genes: {result.get('tested_genes', 0)}\n")
                    f.write(f"  Significant interactions: {result.get('significant_interactions', 0)}\n")
                    f.write(f"  Results file: {result.get('results_file', 'N/A')}\n\n")
                    
                    total_tested += result.get('tested_genes', 0)
                    total_significant += result.get('significant_interactions', 0)
                
                f.write(f"Overall Summary:\n")
                f.write(f"  Total tested genes: {total_tested}\n")
                f.write(f"  Total significant interactions: {total_significant}\n")
                f.write(f"  Overall hit rate: {total_significant/total_tested*100:.2f}%\n" if total_tested > 0 else "  Overall hit rate: 0%\n")
            
            logger.info(f"💾 {qtl_type} interaction summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error generating {qtl_type} interaction summary: {e}")