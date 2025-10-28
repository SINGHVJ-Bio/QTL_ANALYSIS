# [file name]: batch_correction.py
# [file content begin]
#!/usr/bin/env python3
"""
Batch Correction Module for QTL Analysis
Handles different confounders for RNA-seq and protein data
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

def load_confounders_for_qtl_type(confounders_file, sample_ids, qtl_type):
    """
    Load and prepare confounders data for specific QTL type
    
    Parameters:
    -----------
    confounders_file : str
        Path to confounders TSV file
    sample_ids : list
        List of sample IDs to subset
    qtl_type : str
        Type of QTL analysis ('eqtl', 'pqtl', 'sqtl')
    
    Returns:
    --------
    batches : list of arrays
        List of batch variables
    covariates : pandas DataFrame
        Covariates matrix
    sample_mapping : dict
        Mapping for batch variables
    """
    try:
        logger.info(f"📁 Loading confounders for {qtl_type} from: {confounders_file}")
        
        if not os.path.exists(confounders_file):
            raise FileNotFoundError(f"Confounders file not found: {confounders_file}")
        
        confounders = pd.read_csv(confounders_file, sep='\t', index_col=0)
        
        # Subset to requested samples
        available_samples = [s for s in sample_ids if s in confounders.index]
        if len(available_samples) != len(sample_ids):
            logger.warning(f"Only {len(available_samples)}/{len(sample_ids)} samples found in confounders")
        
        confounders = confounders.loc[available_samples]
        
        # Different handling for different QTL types
        if qtl_type == 'eqtl':
            # For RNA-seq: use batch and technical covariates
            batch_columns = [col for col in confounders.columns if 'batch' in col.lower() or 'plate' in col.lower()]
            covariate_columns = [col for col in confounders.columns if col not in batch_columns]
            
        elif qtl_type == 'pqtl':
            # For protein: use specific protein batch columns if available
            batch_columns = [col for col in confounders.columns if any(x in col.lower() for x in ['batch', 'plate', 'run'])]
            covariate_columns = [col for col in confounders.columns if col not in batch_columns]
            
        elif qtl_type == 'sqtl':
            # For splicing: no batch correction typically needed
            logger.info("sQTL analysis - skipping batch correction as typically not required")
            return [], None, {}
        
        else:
            logger.warning(f"Unknown QTL type {qtl_type}, using default batch correction")
            batch_columns = [col for col in confounders.columns if 'batch' in col.lower()]
            covariate_columns = [col for col in confounders.columns if col not in batch_columns]
        
        logger.info(f"🔧 {qtl_type.upper()} - Batch columns: {batch_columns}")
        logger.info(f"🔧 {qtl_type.upper()} - Covariate columns: {covariate_columns}")
        
        # Prepare batches
        batches = []
        for batch_col in batch_columns:
            batch_data = confounders[batch_col].values
            batches.append(batch_data)
            logger.info(f"  - {batch_col}: {len(np.unique(batch_data))} levels")
        
        # Prepare covariates
        covariates = None
        if covariate_columns:
            covariates = confounders[covariate_columns]
            # Convert categorical variables to numeric if needed
            for col in covariates.columns:
                if covariates[col].dtype == 'object':
                    covariates[col] = pd.Categorical(covariates[col]).codes
            logger.info(f"  - Covariates: {covariates.shape[1]} variables")
        
        # Create sample mapping for reporting
        sample_mapping = {
            'batch_columns': batch_columns,
            'covariate_columns': covariate_columns,
            'samples_used': confounders.index.tolist(),
            'confounders_shape': confounders.shape,
            'qtl_type': qtl_type
        }
        
        return batches, covariates, sample_mapping
        
    except Exception as e:
        logger.error(f"❌ Error loading confounders for {qtl_type}: {e}")
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
            'method': 'combat',  # If available, otherwise use custom
            'reason': 'Log2 normalized data benefits from ComBat for strong batch effects',
            'strength': 'Handles strong batch effects well in log-transformed data'
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
        
        # Skip batch correction for splicing if no confounders specified
        if qtl_type == 'sqtl':
            logger.info("⏭️  Skipping batch correction for sQTL analysis")
            return normalized_data, {'batch_correction_skipped': True, 'reason': 'sQTL typically does not require batch correction'}
        
        # Get appropriate confounders file based on QTL type
        input_files = config.get('input_files', {})
        
        if qtl_type == 'eqtl':
            confounders_file = input_files.get('rnaseq_confounders')
        elif qtl_type == 'pqtl':
            # Try protein-specific confounders, fall back to RNA-seq confounders
            confounders_file = input_files.get('protein_confounders', input_files.get('rnaseq_confounders'))
        else:
            confounders_file = input_files.get('rnaseq_confounders')
        
        if not confounders_file or not os.path.exists(confounders_file):
            logger.warning(f"⏭️  No confounders file found for {qtl_type}, skipping batch correction")
            return normalized_data, {'batch_correction_skipped': True, 'reason': f'No confounders file found for {qtl_type}'}
        
        # Load confounders
        batches, covariates, sample_mapping = load_confounders_for_qtl_type(
            confounders_file, 
            sample_ids=normalized_data.columns.tolist(),
            qtl_type=qtl_type
        )
        
        # Skip if no batches or covariates found
        if not batches and covariates is None:
            logger.info(f"⏭️  No batch/covariate effects found for {qtl_type}, skipping batch correction")
            return normalized_data, {'batch_correction_skipped': True, 'reason': 'No batch/covariate effects found'}
        
        # Perform batch correction
        corrected_data = remove_batch_effect_custom(
            x=normalized_data,
            batches=batches,
            covariates=covariates,
            design=None  # Using default intercept design
        )
        
        # Prepare correction info
        correction_info = {
            'confounders_file': confounders_file,
            'samples_used': sample_mapping['samples_used'],
            'batch_columns': sample_mapping['batch_columns'],
            'covariate_columns': sample_mapping['covariate_columns'],
            'input_shape': normalized_data.shape,
            'output_shape': corrected_data.shape,
            'correction_method': 'linear_regression',
            'qtl_type': qtl_type,
            'batch_correction_applied': True
        }
        
        logger.info(f"✅ Batch correction pipeline completed for {qtl_type}")
        return corrected_data, correction_info
        
    except Exception as e:
        logger.error(f"❌ Batch correction pipeline failed for {qtl_type}: {e}")
        raise
# [file content end]