#!/usr/bin/env python3
"""
Enhanced Batch Correction Module for QTL Analysis
Optimized for Python with proper handling of categorical and linear covariates
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import logging
import os

logger = logging.getLogger('QTLPipeline')

def remove_batch_effect_optimized(x, categorical_covariates=None, linear_covariates=None, design_matrix=None):
    """
    Optimized batch correction for Python that properly handles categorical and linear covariates
    
    Parameters:
    -----------
    x : pandas DataFrame
        Expression data (features x samples)
    categorical_covariates : pandas DataFrame or None
        Categorical covariates (samples x categorical_variables)
    linear_covariates : pandas DataFrame or None  
        Linear covariates (samples x linear_variables)
    design_matrix : numpy array or None
        Optional design matrix for conditions of interest
    
    Returns:
    --------
    corrected_data : pandas DataFrame
        Batch-corrected expression data
    """
    try:
        if categorical_covariates is None and linear_covariates is None:
            logger.info("No covariates provided, returning original data")
            return x.copy()
        
        # Convert to numpy for processing
        x_matrix = x.values.T  # Convert to samples x features for processing
        n_samples, n_features = x_matrix.shape
        
        # Create default design matrix (intercept only) if not provided
        if design_matrix is None:
            design_matrix = np.ones((n_samples, 1))
        
        # Process categorical covariates using proper encoding
        X_categorical = None
        if categorical_covariates is not None and not categorical_covariates.empty:
            X_categorical_list = []
            
            for col in categorical_covariates.columns:
                cat_data = categorical_covariates[col].values
                
                # Use proper categorical encoding (one-hot with drop first to avoid collinearity)
                unique_cats = np.unique(cat_data)
                if len(unique_cats) > 1:  # Only encode if more than one category
                    # Create one-hot encoding, dropping first category to avoid collinearity
                    for i, cat in enumerate(unique_cats[1:]):  # Skip first category
                        indicator = (cat_data == cat).astype(float)
                        X_categorical_list.append(indicator.reshape(-1, 1))
            
            if X_categorical_list:
                X_categorical = np.column_stack(X_categorical_list)
                logger.info(f"Encoded {X_categorical.shape[1]} categorical variables")
        
        # Process linear covariates
        X_linear = None
        if linear_covariates is not None and not linear_covariates.empty:
            # Ensure numeric and handle missing values
            linear_processed = linear_covariates.copy()
            for col in linear_processed.columns:
                linear_processed[col] = pd.to_numeric(linear_processed[col], errors='coerce')
                # Fill missing with mean
                if linear_processed[col].isna().any():
                    linear_processed[col] = linear_processed[col].fillna(linear_processed[col].mean())
            
            X_linear = linear_processed.values
            logger.info(f"Processed {X_linear.shape[1]} linear covariates")
        
        # Combine all covariates
        X_covariates_list = []
        if X_categorical is not None:
            X_covariates_list.append(X_categorical)
        if X_linear is not None:
            X_covariates_list.append(X_linear)
        
        if not X_covariates_list:
            logger.info("No valid covariates after processing")
            return x.copy()
        
        X_covariates = np.column_stack(X_covariates_list)
        
        # Full design matrix: [design_matrix, X_covariates]
        X_full = np.column_stack([design_matrix, X_covariates])
        
        # Perform batch correction using vectorized operations for better performance
        corrected_matrix = np.zeros_like(x_matrix)
        
        for i in range(n_features):
            y = x_matrix[:, i]
            
            # Skip if all values are the same
            if np.all(y == y[0]):
                corrected_matrix[:, i] = y
                continue
                
            try:
                # Fit linear model
                model = LinearRegression(fit_intercept=False)
                model.fit(X_full, y)
                
                # Remove only the covariate effects (keep design matrix effects)
                n_design = design_matrix.shape[1]
                beta_covariates = model.coef_[n_design:]
                
                # Remove covariate effects: y - X_covariates @ beta_covariates
                if X_covariates.size > 0:
                    covariate_effects = X_covariates @ beta_covariates
                    corrected_matrix[:, i] = y - covariate_effects
                else:
                    corrected_matrix[:, i] = y
                    
            except Exception as e:
                logger.warning(f"Could not correct feature {i}: {e}")
                corrected_matrix[:, i] = y
        
        # Convert back to DataFrame (features x samples)
        corrected_data = pd.DataFrame(
            corrected_matrix.T,
            index=x.index,
            columns=x.columns
        )
        
        logger.info(f"✅ Optimized batch correction completed: {x.shape} -> {corrected_data.shape}")
        return corrected_data
        
    except Exception as e:
        logger.error(f"❌ Error in optimized batch correction: {e}")
        raise

def prepare_covariates_from_config(config, sample_ids, qtl_type):
    """
    Prepare covariates based on configuration for specific QTL type
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    sample_ids : list
        List of sample IDs
    qtl_type : str
        Type of QTL analysis
    
    Returns:
    --------
    categorical_covariates : pandas DataFrame
        Processed categorical covariates
    linear_covariates : pandas DataFrame  
        Processed linear covariates
    sample_mapping : dict
        Information about the processing
    """
    try:
        batch_config = config.get('batch_correction', {})
        covariate_design_file = batch_config.get('exp_covariate_design')
        
        if not covariate_design_file or not os.path.exists(covariate_design_file):
            logger.warning(f"Covariate design file not found: {covariate_design_file}")
            return pd.DataFrame(), pd.DataFrame(), {}
        
        logger.info(f"Loading covariate design from: {covariate_design_file}")
        covariate_design = pd.read_csv(covariate_design_file, sep='\t', index_col=0)
        
        # Subset to available samples
        available_samples = [s for s in sample_ids if s in covariate_design.index]
        if len(available_samples) != len(sample_ids):
            logger.warning(f"Only {len(available_samples)}/{len(sample_ids)} samples found in covariate design")
        
        if len(available_samples) == 0:
            logger.error("No overlapping samples found")
            return pd.DataFrame(), pd.DataFrame(), {}
        
        covariate_design = covariate_design.loc[available_samples]
        
        # Get categorical and linear covariates from config
        categorical_vars = batch_config.get('exp_covariates', {}).get('categorical', [])
        linear_vars = batch_config.get('exp_covariates', {}).get('linear', [])
        
        logger.info(f"Categorical covariates: {categorical_vars}")
        logger.info(f"Linear covariates: {linear_vars}")
        
        # Filter to available columns
        categorical_vars = [var for var in categorical_vars if var in covariate_design.columns]
        linear_vars = [var for var in linear_vars if var in covariate_design.columns]
        
        missing_categorical = set(batch_config.get('exp_covariates', {}).get('categorical', [])) - set(categorical_vars)
        missing_linear = set(batch_config.get('exp_covariates', {}).get('linear', [])) - set(linear_vars)
        
        if missing_categorical:
            logger.warning(f"Missing categorical covariates: {missing_categorical}")
        if missing_linear:
            logger.warning(f"Missing linear covariates: {missing_linear}")
        
        # Prepare categorical covariates
        categorical_covariates = pd.DataFrame()
        if categorical_vars:
            categorical_covariates = covariate_design[categorical_vars].copy()
            # Ensure string type for categorical variables
            for col in categorical_vars:
                categorical_covariates[col] = categorical_covariates[col].astype(str)
        
        # Prepare linear covariates  
        linear_covariates = pd.DataFrame()
        if linear_vars:
            linear_covariates = covariate_design[linear_vars].copy()
            # Convert to numeric and handle missing values
            for col in linear_vars:
                linear_covariates[col] = pd.to_numeric(linear_covariates[col], errors='coerce')
                if linear_covariates[col].isna().any():
                    col_mean = linear_covariates[col].mean()
                    linear_covariates[col] = linear_covariates[col].fillna(col_mean)
                    logger.info(f"Filled missing values in {col} with mean: {col_mean:.3f}")
        
        sample_mapping = {
            'covariate_design_file': covariate_design_file,
            'categorical_vars': categorical_vars,
            'linear_vars': linear_vars,
            'samples_used': available_samples,
            'original_samples': sample_ids,
            'qtl_type': qtl_type
        }
        
        logger.info(f"Prepared {len(categorical_vars)} categorical and {len(linear_vars)} linear covariates")
        return categorical_covariates, linear_covariates, sample_mapping
        
    except Exception as e:
        logger.error(f"Error preparing covariates: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

def run_optimized_batch_correction(normalized_data, qtl_type, config, design_variables=None):
    """
    Optimized batch correction pipeline for Python
    
    Parameters:
    -----------
    normalized_data : pandas DataFrame
        Normalized expression data (features x samples)
    qtl_type : str
        Type of QTL analysis
    config : dict
        Configuration dictionary
    design_variables : list, optional
        Variables to include in design matrix
    
    Returns:
    --------
    corrected_data : pandas DataFrame
        Batch-corrected expression data
    correction_info : dict
        Information about correction process
    """
    try:
        logger.info(f"🚀 Starting optimized batch correction for {qtl_type}...")
        
        # Check if batch correction is enabled for this QTL type
        batch_enabled = config.get('batch_correction', {}).get('enabled', {}).get(qtl_type, True)
        if not batch_enabled:
            logger.info(f"⏭️ Batch correction disabled for {qtl_type}")
            return normalized_data, {
                'batch_correction_applied': False,
                'reason': f'Batch correction disabled for {qtl_type}'
            }
        
        # Prepare covariates from config
        sample_ids = normalized_data.columns.tolist()
        categorical_covariates, linear_covariates, sample_mapping = prepare_covariates_from_config(
            config, sample_ids, qtl_type
        )
        
        # Check if we have any covariates to correct for
        has_categorical = categorical_covariates is not None and not categorical_covariates.empty
        has_linear = linear_covariates is not None and not linear_covariates.empty
        
        if not has_categorical and not has_linear:
            logger.info(f"⏭️ No valid covariates found for {qtl_type}, skipping batch correction")
            return normalized_data, {
                'batch_correction_applied': False,
                'reason': 'No valid covariates found'
            }
        
        # Create design matrix if design variables specified
        design_matrix = None
        if design_variables:
            # This would require the full covariate design matrix
            # For now, we'll use intercept-only design
            logger.info(f"Using design variables: {design_variables}")
            # In a more advanced implementation, we could create a proper design matrix here
        
        # Subset normalized data to samples with covariates
        available_samples = sample_mapping.get('samples_used', sample_ids)
        normalized_data_subset = normalized_data[available_samples]
        
        # Apply optimized batch correction
        corrected_data = remove_batch_effect_optimized(
            x=normalized_data_subset,
            categorical_covariates=categorical_covariates,
            linear_covariates=linear_covariates,
            design_matrix=design_matrix
        )
        
        # Prepare comprehensive correction info
        correction_info = {
            'batch_correction_applied': True,
            'correction_method': 'optimized_linear_regression',
            'qtl_type': qtl_type,
            'input_shape': normalized_data.shape,
            'output_shape': corrected_data.shape,
            'samples_used': available_samples,
            'categorical_covariates': sample_mapping.get('categorical_vars', []),
            'linear_covariates': sample_mapping.get('linear_vars', []),
            'covariate_design_file': sample_mapping.get('covariate_design_file'),
            'design_variables_used': design_variables
        }
        
        # Log summary
        cat_count = len(correction_info['categorical_covariates'])
        lin_count = len(correction_info['linear_covariates'])
        logger.info(f"✅ Optimized batch correction completed for {qtl_type}")
        logger.info(f"   Used {cat_count} categorical and {lin_count} linear covariates")
        logger.info(f"   Corrected {corrected_data.shape[0]} features across {len(available_samples)} samples")
        
        return corrected_data, correction_info
        
    except Exception as e:
        logger.error(f"❌ Optimized batch correction failed for {qtl_type}: {e}")
        raise

# COMPATIBILITY: Keep the original function for backward compatibility
def remove_batch_effect_custom(x, batches=None, covariates=None, design=None):
    """
    Original batch correction function maintained for backward compatibility
    Now enhanced to properly handle categorical and linear covariates
    """
    try:
        logger.info("Using enhanced version of remove_batch_effect_custom")
        
        # Convert old-style parameters to new format
        categorical_covariates = None
        linear_covariates = None
        
        if batches is not None:
            # Convert batches list to DataFrame
            batch_data = {}
            for i, batch in enumerate(batches):
                batch_data[f'batch_{i}'] = batch
            categorical_covariates = pd.DataFrame(batch_data)
        
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                linear_covariates = covariates
            else:
                linear_covariates = pd.DataFrame(covariates)
        
        # Use the optimized function
        return remove_batch_effect_optimized(
            x=x,
            categorical_covariates=categorical_covariates,
            linear_covariates=linear_covariates,
            design_matrix=design
        )
        
    except Exception as e:
        logger.error(f"Error in compatibility batch correction: {e}")
        raise

# ENHANCED: Updated main pipeline function that accepts batch_covariates parameter
def run_batch_correction_pipeline(normalized_data, qtl_type, config, batch_covariates=None):
    """
    Enhanced batch correction pipeline that accepts batch_covariates parameter
    for compatibility with qtl_analysis.py
    
    Parameters:
    -----------
    normalized_data : pandas DataFrame
        Normalized expression data (features x samples)
    qtl_type : str
        Type of QTL analysis ('eqtl', 'pqtl', 'sqtl')
    config : dict
        Configuration dictionary
    batch_covariates : pandas DataFrame, optional
        Pre-loaded batch covariates from qtl_analysis.py
    
    Returns:
    --------
    corrected_data : pandas DataFrame
        Batch-corrected expression data
    correction_info : dict
        Information about the correction process
    """
    try:
        logger.info(f"🚀 Starting batch correction pipeline for {qtl_type}...")
        
        # Skip if disabled for this QTL type
        if qtl_type == 'sqtl':
            batch_enabled = config.get('batch_correction', {}).get('enabled', {}).get('sqtl', False)
            if not batch_enabled:
                logger.info("⏭️ Skipping batch correction for sQTL analysis")
                return normalized_data, {
                    'batch_correction_skipped': True, 
                    'reason': 'sQTL batch correction disabled',
                    'batch_correction_applied': False
                }
        
        batch_enabled = config.get('batch_correction', {}).get('enabled', {}).get(qtl_type, True)
        if not batch_enabled:
            logger.info(f"⏭️ Batch correction disabled for {qtl_type}")
            return normalized_data, {
                'batch_correction_skipped': True,
                'reason': f'Batch correction disabled for {qtl_type}',
                'batch_correction_applied': False
            }
        
        # If batch_covariates is provided from qtl_analysis.py, use it
        if batch_covariates is not None and not batch_covariates.empty:
            logger.info("Using batch covariates provided from qtl_analysis.py")
            logger.info(f"📊 Batch covariates shape: {batch_covariates.shape}")
            logger.info(f"📊 Batch covariates index (samples): {batch_covariates.index.tolist()[:3]}...")
            logger.info(f"📊 Batch covariates columns (covariates): {batch_covariates.columns.tolist()}")
            
            # Convert the provided covariates to our categorical/linear format
            batch_config = config.get('batch_correction', {})
            categorical_vars = batch_config.get('exp_covariates', {}).get('categorical', [])
            linear_vars = batch_config.get('exp_covariates', {}).get('linear', [])
            
            # Filter to available columns in the provided covariates
            available_categorical = [var for var in categorical_vars if var in batch_covariates.columns]
            available_linear = [var for var in linear_vars if var in batch_covariates.columns]
            
            logger.info(f"🔧 Available categorical covariates: {available_categorical}")
            logger.info(f"🔧 Available linear covariates: {available_linear}")
            
            # Align samples - batch_covariates has samples as INDEX (correct format)
            sample_ids = normalized_data.columns.tolist()
            available_samples = [s for s in sample_ids if s in batch_covariates.index]
            
            logger.info(f"🔍 Sample matching: {len(available_samples)}/{len(sample_ids)} samples found in batch covariates")
            
            if len(available_samples) != len(sample_ids):
                logger.warning(f"⚠️ Only {len(available_samples)}/{len(sample_ids)} samples found in batch covariates")
                logger.info(f"🔍 First 5 expression samples: {sample_ids[:5]}")
                logger.info(f"🔍 First 5 batch covariate samples: {batch_covariates.index.tolist()[:5]}")
            
            if len(available_samples) == 0:
                logger.error("❌ No overlapping samples found between expression data and batch covariates")
                logger.error("💡 Check if sample IDs match exactly between expression data and batch covariates file")
                return normalized_data, {
                    'batch_correction_skipped': True,
                    'reason': 'No overlapping samples with batch covariates',
                    'batch_correction_applied': False
                }
            
            # CRITICAL: Ensure samples are in the SAME ORDER for both expression and covariates
            # Sort the available samples to maintain consistency
            available_samples_sorted = sorted(available_samples)
            
            # Subset and REORDER data to ensure consistent sample order
            normalized_data_subset = normalized_data[available_samples_sorted]
            
            # Subset and REORDER covariates to match expression data order
            categorical_covariates = batch_covariates.loc[available_samples_sorted, available_categorical] if available_categorical else pd.DataFrame()
            linear_covariates = batch_covariates.loc[available_samples_sorted, available_linear] if available_linear else pd.DataFrame()
            
            logger.info(f"✅ Data subset: {normalized_data_subset.shape}")
            logger.info(f"✅ Categorical covariates: {categorical_covariates.shape}")
            logger.info(f"✅ Linear covariates: {linear_covariates.shape}")
            
            # Verify sample alignment
            expr_samples = normalized_data_subset.columns.tolist()
            covar_samples = categorical_covariates.index.tolist() if not categorical_covariates.empty else linear_covariates.index.tolist()
            
            if expr_samples != covar_samples:
                logger.error("❌ CRITICAL: Sample order mismatch between expression data and covariates!")
                logger.error(f"💡 First 3 expression samples: {expr_samples[:3]}")
                logger.error(f"💡 First 3 covariate samples: {covar_samples[:3]}")
                raise ValueError("Sample order mismatch between expression data and covariates")
            else:
                logger.info("✅ Sample order verified: expression data and covariates are aligned")
            
            # Ensure proper data types
            for col in categorical_covariates.columns:
                categorical_covariates[col] = categorical_covariates[col].astype(str)
            
            for col in linear_covariates.columns:
                linear_covariates[col] = pd.to_numeric(linear_covariates[col], errors='coerce')
                if linear_covariates[col].isna().any():
                    linear_covariates[col] = linear_covariates[col].fillna(linear_covariates[col].mean())
            
            # Apply batch correction
            corrected_data = remove_batch_effect_optimized(
                x=normalized_data_subset,
                categorical_covariates=categorical_covariates,
                linear_covariates=linear_covariates,
                design_matrix=None
            )
            
            correction_info = {
                'batch_correction_applied': True,
                'correction_method': 'optimized_linear_regression',
                'qtl_type': qtl_type,
                'input_shape': normalized_data.shape,
                'output_shape': corrected_data.shape,
                'samples_used': available_samples_sorted,
                'categorical_covariates': available_categorical,
                'linear_covariates': available_linear,
                'covariate_source': 'provided_from_qtl_analysis'
            }
            
        else:
            # Use the config-based approach
            corrected_data, correction_info = run_optimized_batch_correction(
                normalized_data, qtl_type, config
            )
        
        return corrected_data, correction_info
        
    except Exception as e:
        logger.error(f"❌ Batch correction pipeline failed for {qtl_type}: {e}")
        raise

# Backward compatibility functions
def run_batch_correction(normalized_data, qtl_type, config):
    """Backward compatibility wrapper"""
    return run_batch_correction_pipeline(normalized_data, qtl_type, config)

def run_enhanced_batch_correction(normalized_data, qtl_type, config):
    """Enhanced batch correction - alias for optimized version"""
    return run_optimized_batch_correction(normalized_data, qtl_type, config)

def get_recommended_batch_correction(normalization_method, qtl_type):
    """
    Get recommended batch correction approach based on normalization method
    """
    recommendations = {
        'vst': {
            'method': 'optimized_linear_regression',
            'reason': 'VST normalized data works well with linear regression batch correction',
            'strength': 'Removes technical variation while preserving biological signals'
        },
        'log2': {
            'method': 'optimized_linear_regression', 
            'reason': 'Log2 normalized data works well with linear regression batch correction',
            'strength': 'Handles batch effects well in log-transformed data'
        },
        'raw': {
            'method': 'optimized_linear_regression',
            'reason': 'Raw count data needs careful batch correction',
            'strength': 'Preserves count distribution while removing biases'
        }
    }
    
    method_info = recommendations.get(normalization_method, recommendations['vst'])
    
    logger.info(f"🎯 Recommended batch correction for {normalization_method.upper()}: {method_info['method']}")
    logger.info(f"📋 Reason: {method_info['reason']}")
    
    return method_info

if __name__ == "__main__":
    """Test the optimized batch correction"""
    import yaml
    
    # Example config matching your structure
    config = {
        'batch_correction': {
            'enabled': {
                'eqtl': True,
                'pqtl': False,  
                'sqtl': False
            },
            'exp_covariates': {
                'categorical': ["Batch", "Sex"],
                'linear': ["Age", "RIN", "PC1", "PC2"]
            },
            'exp_covariate_design': "/Users/singhvj/QTL_ANALYSIS_OUTPUT/confounders_matrix.tsv"
        }
    }
    
    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame(
        np.random.randn(100, 50),
        index=[f'gene_{i}' for i in range(100)],
        columns=[f'sample_{i}' for i in range(50)]
    )
    
    print("Testing optimized batch correction...")
    try:
        corrected_data, correction_info = run_optimized_batch_correction(
            test_data, 'eqtl', config
        )
        print(f"✅ Test completed: {correction_info}")
    except Exception as e:
        print(f"❌ Test failed: {e}")