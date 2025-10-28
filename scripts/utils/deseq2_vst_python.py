# deseq2_vst_python.py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger('QTLPipeline')

def deseq2_vst_python(counts_df, blind=True, fit_type='parametric'):
    """
    Python implementation of DESeq2's Variance Stabilizing Transformation (VST)
    
    Parameters:
    -----------
    counts_df : pandas DataFrame
        Raw count data (genes x samples)
    blind : bool
        Whether to blind the transformation to the experimental design
    fit_type : str
        Type of fit for dispersion trend ('parametric', 'local')
    
    Returns:
    --------
    vst_df : pandas DataFrame
        VST normalized expression data
    """
    try:
        logger.info("🔬 Starting DESeq2 VST normalization (Python implementation)...")
        
        # Convert to numpy array for processing
        counts = counts_df.values.astype(float)
        genes, samples = counts.shape
        
        logger.info(f"📊 Input data: {genes} genes, {samples} samples")
        
        # Step 1: Calculate size factors (like DESeq2's estimateSizeFactors)
        size_factors = calculate_size_factors(counts)
        logger.info(f"✅ Calculated size factors for {samples} samples")
        
        # Step 2: Estimate dispersions (like DESeq2's estimateDispersions)
        dispersions = estimate_dispersions(counts, size_factors, fit_type=fit_type)
        logger.info(f"✅ Estimated dispersions for {genes} genes")
        
        # Step 3: Apply VST transformation
        vst_matrix = apply_vst_transform(counts, size_factors, dispersions, blind=blind)
        
        # Convert back to DataFrame
        vst_df = pd.DataFrame(
            vst_matrix,
            index=counts_df.index,
            columns=counts_df.columns
        )
        
        logger.info(f"✅ VST normalization completed: {vst_df.shape}")
        logger.info(f"📈 VST stats - Mean: {vst_matrix.mean():.2f}, Std: {vst_matrix.std():.2f}")
        
        return vst_df
        
    except Exception as e:
        logger.error(f"❌ DESeq2 VST normalization failed: {e}")
        raise

def calculate_size_factors(counts):
    """Calculate size factors using DESeq2's median ratio method"""
    # Filter out genes with all zeros
    non_zero_genes = np.any(counts > 0, axis=1)
    counts_filtered = counts[non_zero_genes]
    
    if counts_filtered.shape[0] == 0:
        return np.ones(counts.shape[1])
    
    # Calculate geometric means
    log_counts = np.log(counts_filtered + 1)  # Add pseudocount to avoid log(0)
    geometric_means = np.exp(np.mean(log_counts, axis=1))
    
    # Calculate size factors using median ratio
    size_factors = []
    for sample_idx in range(counts_filtered.shape[1]):
        ratios = counts_filtered[:, sample_idx] / geometric_means
        # Remove infinite and NaN values
        ratios = ratios[np.isfinite(ratios)]
        if len(ratios) > 0:
            size_factor = np.median(ratios)
            size_factors.append(size_factor)
        else:
            size_factors.append(1.0)
    
    size_factors = np.array(size_factors)
    
    # Handle edge cases
    size_factors[size_factors == 0] = 1.0
    size_factors = size_factors / np.mean(size_factors)  # Normalize
    
    return size_factors

def estimate_dispersions(counts, size_factors, fit_type='parametric'):
    """Estimate gene dispersions using DESeq2-like method"""
    genes, samples = counts.shape
    
    if genes == 0 or samples == 0:
        return np.array([])
    
    # Calculate normalized counts
    normalized_counts = counts / size_factors
    
    # Calculate means and variances
    means = np.mean(normalized_counts, axis=1)
    variances = np.var(normalized_counts, axis=1, ddof=1)
    
    # Handle zero means
    means[means == 0] = np.min(means[means > 0]) if np.any(means > 0) else 1.0
    
    # Calculate raw dispersions
    raw_dispersions = variances / (means ** 2)
    
    # Filter extreme values
    valid_mask = (means > 0) & (variances > 0) & np.isfinite(raw_dispersions)
    means_valid = means[valid_mask]
    dispersions_valid = raw_dispersions[valid_mask]
    
    if len(means_valid) == 0:
        return np.ones(genes) * 0.1  # Default dispersion
    
    # Fit dispersion trend
    if fit_type == 'parametric':
        dispersions_fitted = fit_parametric_dispersion(means_valid, dispersions_valid)
    elif fit_type == 'local':
        dispersions_fitted = fit_local_dispersion(means_valid, dispersions_valid)
    else:
        dispersions_fitted = fit_parametric_dispersion(means_valid, dispersions_valid)
    
    # Create final dispersion array
    final_dispersions = np.ones(genes) * np.median(dispersions_fitted)
    
    # Use fitted dispersions for valid genes, median for others
    final_dispersions[valid_mask] = dispersions_fitted
    
    # Ensure dispersions are positive
    final_dispersions = np.maximum(final_dispersions, 1e-8)
    
    return final_dispersions

def fit_parametric_dispersion(means, dispersions):
    """Fit parametric curve to dispersion-mean relationship"""
    try:
        # Log transform for stability
        log_means = np.log(means + 1)
        log_dispersions = np.log(dispersions + 1e-8)
        
        # Fit polynomial (simplified version of DESeq2's fit)
        coeffs = np.polyfit(log_means, log_dispersions, deg=2)
        fitted_log_dispersions = np.polyval(coeffs, log_means)
        
        return np.exp(fitted_log_dispersions)
    except:
        # Fallback: use loess-like smoothing
        return fit_local_dispersion(means, dispersions)

def fit_local_dispersion(means, dispersions):
    """Fit local regression to dispersion-mean relationship"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Use polynomial features for local fitting
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(means.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_poly, dispersions)
        
        return model.predict(X_poly)
    except:
        # Ultimate fallback: use mean dispersion
        return np.ones_like(dispersions) * np.mean(dispersions)

def apply_vst_transform(counts, size_factors, dispersions, blind=True):
    """Apply variance stabilizing transformation"""
    genes, samples = counts.shape
    
    if genes == 0 or samples == 0:
        return counts
    
    # Calculate normalized counts
    normalized_counts = counts / size_factors
    
    # Calculate means
    means = np.mean(normalized_counts, axis=1)
    
    # Apply VST formula: asinh(sqrt(dispersion) * (count - mean) / sqrt(mean)) / sqrt(dispersion)
    vst_matrix = np.zeros_like(counts)
    
    for i in range(genes):
        if dispersions[i] > 0:
            sqrt_disp = np.sqrt(dispersions[i])
            if means[i] > 0:
                # DESeq2-like VST transformation
                vst_values = np.arcsinh(sqrt_disp * (counts[i, :] - means[i]) / np.sqrt(means[i])) / sqrt_disp
            else:
                vst_values = np.arcsinh(np.sqrt(counts[i, :] + 0.5))  # Fallback for zero means
        else:
            vst_values = np.arcsinh(np.sqrt(counts[i, :] + 0.5))  # Fallback for zero dispersion
        
        vst_matrix[i, :] = vst_values
    
    # If blind=False, we would adjust for experimental design here
    # For simplicity, we're implementing the basic blind transformation
    
    return vst_matrix

def simple_vst_fallback(counts_df):
    """
    Simple VST fallback using arcsinh transformation
    Good approximation when full DESeq2 implementation is not feasible
    """
    logger.info("🔄 Using simplified VST (arcsinh transformation)...")
    
    counts = counts_df.values.astype(float)
    
    # Simple size factor normalization
    size_factors = np.sum(counts, axis=0)
    size_factors = size_factors / np.median(size_factors)
    normalized_counts = counts / size_factors
    
    # Arcsinh transformation (good for variance stabilization)
    vst_matrix = np.arcsinh(normalized_counts)
    
    vst_df = pd.DataFrame(
        vst_matrix,
        index=counts_df.index,
        columns=counts_df.columns
    )
    
    return vst_df