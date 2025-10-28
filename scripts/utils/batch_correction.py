#!/usr/bin/env python3
"""
Batch Correction Module for QTL Analysis
Handles different confounders for RNA-seq and protein data using enhanced configuration
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import os

logger = logging.getLogger('QTLPipeline')

def remove_batch_effect_custom(x, batches=None, covariates=None, design=None):
    """
    Remove batch effects using linear regression (Python implementation of R's removeBatchEffect)
    
    Parameters:
    -----------
    x : pandas DataFrame
        Expression data (features x samples)
    batches : list of arrays or None
        List of batch variables, each should be same length as samples
    covariates : pandas DataFrame or None
        Covariates matrix (samples x covariates)
    design : numpy array or None
        Design matrix (samples x design_variables)
    
    Returns:
    --------
    corrected_data : pandas DataFrame
        Batch-corrected expression data
    """
    try:
        if batches is None and covariates is None:
            logger.info("No batches or covariates provided, returning original data")
            return x.copy()
        
        # Convert to numpy for processing
        x_matrix = x.values.T  # Convert to samples x features
        n_samples, n_features = x_matrix.shape
        
        # Create design matrix if not provided
        if design is None:
            design = np.ones((n_samples, 1))
        
        # Process batches
        X_batch_list = []
        if batches is not None:
            for batch in batches:
                batch = np.array(batch)
                if len(batch) != n_samples:
                    raise ValueError(f"Batch length {len(batch)} doesn't match samples {n_samples}")
                
                # Create contrast matrix (similar to contr.sum in R)
                unique_batches = np.unique(batch)
                if len(unique_batches) > 1:
                    batch_matrix = np.zeros((n_samples, len(unique_batches) - 1))
                    for i, batch_level in enumerate(unique_batches[:-1]):
                        batch_matrix[:, i] = (batch == batch_level).astype(float)
                    # Set reference level (last level) to -1
                    for i in range(len(unique_batches) - 1):
                        batch_matrix[batch == unique_batches[-1], i] = -1
                    X_batch_list.append(batch_matrix)
        
        # Combine batch matrices
        if X_batch_list:
            X_batch = np.column_stack(X_batch_list)
        else:
            X_batch = np.empty((n_samples, 0))
        
        # Add covariates
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                covariates_matrix = covariates.values
            else:
                covariates_matrix = np.array(covariates)
            
            if covariates_matrix.shape[0] != n_samples:
                raise ValueError(f"Covariates rows {covariates_matrix.shape[0]} don't match samples {n_samples}")
            
            X_batch = np.column_stack([X_batch, covariates_matrix]) if X_batch.size > 0 else covariates_matrix
        
        if X_batch.size == 0:
            logger.info("No batch/covariate effects to remove")
            return x.copy()
        
        # Full design matrix
        X_full = np.column_stack([design, X_batch])
        
        # Perform batch correction
        corrected_matrix = np.zeros_like(x_matrix)
        
        for i in range(n_features):
            y = x_matrix[:, i]
            
            # Fit linear model
            try:
                model = LinearRegression(fit_intercept=False)
                model.fit(X_full, y)
                
                # Get coefficients for batch effects
                n_design = design.shape[1]
                beta_batch = model.coef_[n_design:]
                
                # Remove batch effects
                if X_batch.size > 0:
                    batch_effect = X_batch @ beta_batch
                    corrected_matrix[:, i] = y - batch_effect
                else:
                    corrected_matrix[:, i] = y
                    
            except Exception as e:
                logger.warning(f"Could not correct feature {i}: {e}")
                corrected_matrix[:, i] = y
        
        # Convert back to DataFrame
        corrected_data = pd.DataFrame(
            corrected_matrix.T,
            index=x.index,
            columns=x.columns
        )
        
        logger.info(f"✅ Batch correction completed: {x.shape} -> {corrected_data.shape}")
        return corrected_data
        
    except Exception as e:
        logger.error(f"❌ Error in batch correction: {e}")
        raise

def load_enhanced_confounders(config, sample_ids, qtl_type):
    """
    Load confounders using the enhanced configuration structure
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    sample_ids : list
        List of sample IDs to subset
    qtl_type : str
        Type of QTL analysis ('eqtl', 'pqtl', 'sqtl')
    
    Returns:
    --------
    batches : list of arrays
        List of batch variables (categorical)
    covariates : pandas DataFrame
        Covariates matrix (linear)
    sample_mapping : dict
        Mapping for batch variables
    """
    try:
        batch_config = config.get('batch_correction', {})
        exp_covariates = batch_config.get('exp_covariates', {})
        covariate_design_file = batch_config.get('exp_covariate_design')
        
        if not covariate_design_file:
            logger.warning(f"⏭️ No covariate design file specified in config")
            return [], None, {}
        
        if not os.path.exists(covariate_design_file):
            logger.warning(f"⏭️ Covariate design file not found: {covariate_design_file}")
            return [], None, {}
        
        logger.info(f"📁 Loading enhanced confounders for {qtl_type} from: {covariate_design_file}")
        
        # Load covariate design matrix
        covariate_design = pd.read_csv(covariate_design_file, sep='\t', index_col=0)
        
        # Subset to requested samples
        available_samples = [s for s in sample_ids if s in covariate_design.index]
        if len(available_samples) != len(sample_ids):
            logger.warning(f"Only {len(available_samples)}/{len(sample_ids)} samples found in covariate design")
        
        if len(available_samples) == 0:
            logger.error("❌ No overlapping samples found between expression data and covariate design")
            return [], None, {}
        
        covariate_design = covariate_design.loc[available_samples]
        
        # Get categorical and linear covariates from config
        categorical_covariates = exp_covariates.get('categorical', [])
        linear_covariates = exp_covariates.get('linear', [])
        
        logger.info(f"🔧 Enhanced confounders - Categorical: {categorical_covariates}")
        logger.info(f"🔧 Enhanced confounders - Linear: {linear_covariates}")
        
        # Check if all specified covariates exist in the design matrix
        all_covariates = categorical_covariates + linear_covariates
        missing_covariates = [cov for cov in all_covariates if cov not in covariate_design.columns]
        
        if missing_covariates:
            logger.warning(f"⚠️ Missing covariates in design matrix: {missing_covariates}")
            # Remove missing covariates from lists
            categorical_covariates = [cov for cov in categorical_covariates if cov in covariate_design.columns]
            linear_covariates = [cov for cov in linear_covariates if cov in covariate_design.columns]
        
        # Prepare batches (categorical covariates)
        batches = []
        for covariate in categorical_covariates:
            if covariate in covariate_design.columns:
                batch_data = covariate_design[covariate].values
                batches.append(batch_data)
                logger.info(f"  - Categorical: {covariate}: {len(np.unique(batch_data))} levels")
        
        # Prepare covariates (linear covariates)
        covariates = None
        if linear_covariates:
            covariates = covariate_design[linear_covariates].copy()
            # Convert to numeric, coerce errors to NaN
            for col in linear_covariates:
                covariates[col] = pd.to_numeric(covariates[col], errors='coerce')
                # Fill NaN values with mean for linear covariates
                if covariates[col].isna().any():
                    covariates[col] = covariates[col].fillna(covariates[col].mean())
            logger.info(f"  - Linear covariates: {linear_covariates}")
        
        # Create sample mapping for reporting
        sample_mapping = {
            'covariate_design_file': covariate_design_file,
            'categorical_covariates': categorical_covariates,
            'linear_covariates': linear_covariates,
            'samples_used': available_samples,
            'covariate_design_shape': covariate_design.shape,
            'qtl_type': qtl_type,
            'confounders_source': 'enhanced'
        }
        
        return batches, covariates, sample_mapping
        
    except Exception as e:
        logger.error(f"❌ Error loading enhanced confounders for {qtl_type}: {e}")
        raise

def get_recommended_batch_correction(normalization_method, qtl_type):
    """
    Get recommended batch correction approach based on normalization method
    """
    recommendations = {
        'vst': {
            'method': 'custom_linear_regression',
            'reason': 'VST normalized data works well with linear regression batch correction',
            'strength': 'Removes technical variation while preserving biological signals'
        },
        'log2': {
            'method': 'custom_linear_regression',
            'reason': 'Log2 normalized data works well with linear regression batch correction',
            'strength': 'Handles batch effects well in log-transformed data'
        },
        'raw': {
            'method': 'custom_linear_regression', 
            'reason': 'Raw count data needs careful batch correction',
            'strength': 'Preserves count distribution while removing biases'
        }
    }
    
    method_info = recommendations.get(normalization_method, recommendations['vst'])
    
    logger.info(f"🎯 Recommended batch correction for {normalization_method.upper()}: {method_info['method']}")
    logger.info(f"📋 Reason: {method_info['reason']}")
    
    return method_info

def run_batch_correction_pipeline(normalized_data, qtl_type, config):
    """
    Main function to run batch correction pipeline for specific QTL type
    
    Parameters:
    -----------
    normalized_data : pandas DataFrame
        Normalized expression data (features x samples)
    qtl_type : str
        Type of QTL analysis ('eqtl', 'pqtl', 'sqtl')
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    corrected_data : pandas DataFrame
        Batch-corrected expression data
    correction_info : dict
        Information about the correction process
    """
    try:
        logger.info(f"🚀 Starting batch correction pipeline for {qtl_type}...")
        
        # Skip batch correction for splicing if disabled
        if qtl_type == 'sqtl':
            batch_enabled = config.get('batch_correction', {}).get('enabled', {}).get('sqtl', False)
            if not batch_enabled:
                logger.info("⏭️ Skipping batch correction for sQTL analysis (disabled in config)")
                return normalized_data, {
                    'batch_correction_skipped': True, 
                    'reason': 'sQTL batch correction disabled in config',
                    'batch_correction_applied': False
                }
        
        # Check if batch correction is enabled for this QTL type
        batch_enabled = config.get('batch_correction', {}).get('enabled', {}).get(qtl_type, True)
        if not batch_enabled:
            logger.info(f"⏭️ Batch correction disabled for {qtl_type} in config")
            return normalized_data, {
                'batch_correction_skipped': True,
                'reason': f'Batch correction disabled for {qtl_type}',
                'batch_correction_applied': False
            }
        
        # Use enhanced confounders configuration
        logger.info("🔧 Using enhanced confounders configuration")
        batches, covariates, sample_mapping = load_enhanced_confounders(
            config, 
            sample_ids=normalized_data.columns.tolist(),
            qtl_type=qtl_type
        )
        
        # Skip if no batches or covariates found
        if not batches and covariates is None:
            logger.info(f"⏭️ No batch/covariate effects found for {qtl_type}, skipping batch correction")
            return normalized_data, {
                'batch_correction_skipped': True, 
                'reason': 'No batch/covariate effects found in enhanced confounders',
                'batch_correction_applied': False
            }
        
        # Subset normalized data to available samples with covariates
        available_samples = sample_mapping.get('samples_used', normalized_data.columns.tolist())
        normalized_data_subset = normalized_data[available_samples]
        
        # Perform batch correction
        corrected_data = remove_batch_effect_custom(
            x=normalized_data_subset,
            batches=batches,
            covariates=covariates,
            design=None  # Using default intercept design
        )
        
        # Prepare correction info
        correction_info = {
            'input_shape': normalized_data.shape,
            'output_shape': corrected_data.shape,
            'correction_method': 'linear_regression',
            'qtl_type': qtl_type,
            'batch_correction_applied': True,
            'samples_used': available_samples,
            'confounders_source': 'enhanced',
            'covariate_design_file': sample_mapping.get('covariate_design_file'),
            'categorical_covariates': sample_mapping.get('categorical_covariates', []),
            'linear_covariates': sample_mapping.get('linear_covariates', [])
        }
        
        logger.info(f"✅ Batch correction pipeline completed for {qtl_type}")
        return corrected_data, correction_info
        
    except Exception as e:
        logger.error(f"❌ Batch correction pipeline failed for {qtl_type}: {e}")
        raise

def run_enhanced_batch_correction(normalized_data, qtl_type, config):
    """
    Enhanced batch correction with explicit parameter passing for direct calls
    
    Parameters:
    -----------
    normalized_data : pandas DataFrame
        Normalized expression data (features x samples)
    qtl_type : str
        Type of QTL analysis ('eqtl', 'pqtl', 'sqtl')
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    corrected_data : pandas DataFrame
        Batch-corrected expression data
    correction_info : dict
        Information about the correction process
    """
    try:
        logger.info(f"🚀 Starting enhanced batch correction for {qtl_type}...")
        
        # Get batch correction configuration
        batch_config = config.get('batch_correction', {})
        exp_covariates = batch_config.get('exp_covariates', {})
        covariate_design_file = batch_config.get('exp_covariate_design')
        
        if not covariate_design_file:
            logger.warning(f"⏭️ No covariate design file specified in config")
            return normalized_data, {
                'batch_correction_skipped': True,
                'reason': 'No covariate design file specified',
                'batch_correction_applied': False
            }
        
        if not os.path.exists(covariate_design_file):
            logger.warning(f"⏭️ Covariate design file not found: {covariate_design_file}")
            return normalized_data, {
                'batch_correction_skipped': True,
                'reason': f'Covariate design file not found: {covariate_design_file}',
                'batch_correction_applied': False
            }
        
        # Load covariate design matrix
        logger.info(f"📁 Loading covariate design from: {covariate_design_file}")
        covariate_design = pd.read_csv(covariate_design_file, sep='\t', index_col=0)
        
        # Get sample IDs from normalized data
        sample_ids = normalized_data.columns.tolist()
        
        # Subset covariate design to available samples
        available_samples = [s for s in sample_ids if s in covariate_design.index]
        if len(available_samples) != len(sample_ids):
            logger.warning(f"Only {len(available_samples)}/{len(sample_ids)} samples found in covariate design")
        
        if len(available_samples) == 0:
            logger.error("❌ No overlapping samples found between expression data and covariate design")
            return normalized_data, {
                'batch_correction_skipped': True,
                'reason': 'No overlapping samples found',
                'batch_correction_applied': False
            }
        
        covariate_design = covariate_design.loc[available_samples]
        
        # Get categorical and linear covariates from config
        categorical_covariates = exp_covariates.get('categorical', [])
        linear_covariates = exp_covariates.get('linear', [])
        
        logger.info(f"🔧 Categorical covariates: {categorical_covariates}")
        logger.info(f"🔧 Linear covariates: {linear_covariates}")
        
        # Check if all specified covariates exist in the design matrix
        all_covariates = categorical_covariates + linear_covariates
        missing_covariates = [cov for cov in all_covariates if cov not in covariate_design.columns]
        
        if missing_covariates:
            logger.warning(f"⚠️ Missing covariates in design matrix: {missing_covariates}")
            # Remove missing covariates from lists
            categorical_covariates = [cov for cov in categorical_covariates if cov in covariate_design.columns]
            linear_covariates = [cov for cov in linear_covariates if cov in covariate_design.columns]
        
        # Prepare batches (categorical covariates)
        batches = []
        for covariate in categorical_covariates:
            if covariate in covariate_design.columns:
                batch_data = covariate_design[covariate].values
                batches.append(batch_data)
                logger.info(f"  - Categorical: {covariate}: {len(np.unique(batch_data))} levels")
        
        # Prepare covariates (linear covariates)
        covariates = None
        if linear_covariates:
            covariates = covariate_design[linear_covariates].copy()
            # Convert to numeric, coerce errors to NaN
            for col in linear_covariates:
                covariates[col] = pd.to_numeric(covariates[col], errors='coerce')
                # Fill NaN values with mean for linear covariates
                if covariates[col].isna().any():
                    covariates[col] = covariates[col].fillna(covariates[col].mean())
            logger.info(f"  - Linear covariates: {linear_covariates}")
        
        # Skip if no batches or covariates found
        if not batches and covariates is None:
            logger.info(f"⏭️ No batch/covariate effects found for {qtl_type}, skipping batch correction")
            return normalized_data, {
                'batch_correction_skipped': True,
                'reason': 'No batch/covariate effects found',
                'batch_correction_applied': False
            }
        
        # Subset normalized data to available samples
        normalized_data_subset = normalized_data[available_samples]
        
        # Perform batch correction
        corrected_data = remove_batch_effect_custom(
            x=normalized_data_subset,
            batches=batches,
            covariates=covariates,
            design=None  # Using default intercept design
        )
        
        # Prepare correction info
        correction_info = {
            'covariate_design_file': covariate_design_file,
            'samples_used': available_samples,
            'categorical_covariates': categorical_covariates,
            'linear_covariates': linear_covariates,
            'input_shape': normalized_data.shape,
            'output_shape': corrected_data.shape,
            'correction_method': 'linear_regression',
            'qtl_type': qtl_type,
            'batch_correction_applied': True
        }
        
        logger.info(f"✅ Enhanced batch correction completed for {qtl_type}")
        return corrected_data, correction_info
        
    except Exception as e:
        logger.error(f"❌ Enhanced batch correction failed for {qtl_type}: {e}")
        raise

# Backward compatibility wrapper
def run_batch_correction(normalized_data, qtl_type, config):
    """
    Backward compatibility wrapper for run_batch_correction_pipeline
    """
    return run_batch_correction_pipeline(normalized_data, qtl_type, config)

if __name__ == "__main__":
    """Test the batch correction module"""
    import yaml
    
    # Example usage
    config = {
        'batch_correction': {
            'enabled': {
                'eqtl': True,
                'pqtl': True,
                'sqtl': False
            },
            'exp_covariates': {
                'categorical': ['Batch', 'Sex'],
                'linear': ['Age', 'RIN', 'PC1']
            },
            'exp_covariate_design': '/home/ubuntu/covariate_matrix.tsv'
        }
    }
    
    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame(
        np.random.randn(100, 50),
        index=[f'gene_{i}' for i in range(100)],
        columns=[f'sample_{i}' for i in range(50)]
    )
    
    try:
        corrected_data, correction_info = run_batch_correction_pipeline(
            test_data, 'eqtl', config
        )
        print(f"Batch correction completed: {correction_info}")
    except Exception as e:
        print(f"Batch correction failed: {e}")