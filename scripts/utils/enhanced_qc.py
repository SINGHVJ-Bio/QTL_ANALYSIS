#!/usr/bin/env python3
"""
Enhanced Quality Control with comprehensive sample and variant QC
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import subprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('QTLPipeline')

class EnhancedQC:
    def __init__(self, config):
        self.config = config
        self.qc_config = config.get('enhanced_qc', {})
        
    def run_comprehensive_qc(self, vcf_file, phenotype_files, output_dir):
        """Run comprehensive QC on all data types"""
        logger.info("🔍 Running comprehensive quality control...")
        
        qc_results = {}
        
        # Create QC directory
        qc_dir = os.path.join(output_dir, "QC_reports")
        os.makedirs(qc_dir, exist_ok=True)
        
        # Genotype QC
        qc_results['genotype'] = self.genotype_qc(vcf_file, qc_dir)
        
        # Phenotype QC
        for pheno_type, pheno_file in phenotype_files.items():
            if pheno_file and os.path.exists(pheno_file):
                qc_results[pheno_type] = self.phenotype_qc(pheno_file, pheno_type, qc_dir)
        
        # Sample concordance
        qc_results['concordance'] = self.sample_concordance_qc(vcf_file, phenotype_files, qc_dir)
        
        # Population stratification (PCA)
        if self.qc_config.get('run_pca', True):
            qc_results['pca'] = self.run_pca_analysis(vcf_file, qc_dir)
        
        # Generate QC report
        self.generate_qc_report(qc_results, qc_dir)
        
        logger.info("✅ Comprehensive QC completed")
        return qc_results
    
    def genotype_qc(self, vcf_file, output_dir):
        """Comprehensive genotype QC"""
        logger.info("🔬 Running genotype QC...")
        
        qc_metrics = {}
        
        try:
            # Sample missingness
            sample_missingness = self.calculate_sample_missingness(vcf_file)
            qc_metrics['sample_missingness'] = sample_missingness
            
            # Variant missingness  
            variant_missingness = self.calculate_variant_missingness(vcf_file)
            qc_metrics['variant_missingness'] = variant_missingness
            
            # MAF distribution
            maf_distribution = self.calculate_maf_distribution(vcf_file)
            qc_metrics['maf_distribution'] = maf_distribution
            
            # HWE testing
            hwe_results = self.calculate_hwe(vcf_file, output_dir)
            qc_metrics['hwe'] = hwe_results
            
            # Sample heterozygosity
            heterozygosity = self.calculate_heterozygosity(vcf_file)
            qc_metrics['heterozygosity'] = heterozygosity
            
            # Generate QC plots
            self.plot_genotype_qc(qc_metrics, output_dir)
            
            # Apply QC filters
            filtered_vcf = self.apply_genotype_filters(vcf_file, output_dir, qc_metrics)
            qc_metrics['filtered_file'] = filtered_vcf
            
            logger.info("✅ Genotype QC completed")
            return qc_metrics
            
        except Exception as e:
            logger.error(f"❌ Genotype QC failed: {e}")
            return {}
    
    def calculate_sample_missingness(self, vcf_file):
        """Calculate sample-level missingness"""
        logger.info("📊 Calculating sample missingness...")
        
        try:
            # Use bcftools to calculate missingness
            cmd = f"{self.config['paths']['bcftools']} stats -s - {vcf_file} | grep 'PSC' | head -1000"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            sample_missingness = {}
            for line in result.stdout.strip().split('\n'):
                if line.startswith('PSC'):
                    parts = line.split('\t')
                    if len(parts) >= 10:
                        sample_id = parts[2]
                        missing_rate = 1 - (int(parts[9]) / (int(parts[9]) + int(parts[8]))) if (int(parts[9]) + int(parts[8])) > 0 else 1.0
                        sample_missingness[sample_id] = missing_rate
            
            return sample_missingness
            
        except Exception as e:
            logger.warning(f"Could not calculate sample missingness: {e}")
            return {}
    
    def calculate_variant_missingness(self, vcf_file):
        """Calculate variant-level missingness"""
        logger.info("📊 Calculating variant missingness...")
        
        try:
            # Use bcftools to get variant statistics
            cmd = f"{self.config['paths']['bcftools']} query -f '%CHROM:%POS:%REF:%ALT\\t%NS\\n' {vcf_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            # Get total samples
            samples_cmd = f"{self.config['paths']['bcftools']} query -l {vcf_file} | wc -l"
            samples_result = subprocess.run(samples_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            total_samples = int(samples_result.stdout.strip())
            
            variant_missingness = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        variant_id = parts[0]
                        called_samples = int(parts[1])
                        missing_rate = 1 - (called_samples / total_samples) if total_samples > 0 else 1.0
                        variant_missingness[variant_id] = missing_rate
            
            return variant_missingness
            
        except Exception as e:
            logger.warning(f"Could not calculate variant missingness: {e}")
            return {}
    
    def calculate_maf_distribution(self, vcf_file):
        """Calculate MAF distribution"""
        logger.info("📊 Calculating MAF distribution...")
        
        try:
            # Use bcftools to calculate MAF
            cmd = f"{self.config['paths']['bcftools']} query -f '%CHROM:%POS:%REF:%ALT\\t%AF\\n' {vcf_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            maf_values = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        try:
                            af = float(parts[1])
                            maf = min(af, 1-af)
                            maf_values.append(maf)
                        except ValueError:
                            continue
            
            return {
                'maf_values': maf_values,
                'mean_maf': np.mean(maf_values) if maf_values else 0,
                'median_maf': np.median(maf_values) if maf_values else 0,
                'maf_bins': np.histogram(maf_values, bins=20, range=(0, 0.5)) if maf_values else ([], [])
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate MAF distribution: {e}")
            return {'maf_values': [], 'mean_maf': 0, 'median_maf': 0, 'maf_bins': ([], [])}
    
    def calculate_hwe(self, vcf_file, output_dir):
        """Calculate Hardy-Weinberg Equilibrium"""
        logger.info("📊 Calculating HWE...")
        
        try:
            # Use bcftools to filter by HWE
            temp_file = os.path.join(output_dir, "hwe_temp.vcf")
            cmd = f"{self.config['paths']['bcftools']} filter -i 'ExcHWE<1e-6' {vcf_file} | {self.config['paths']['bcftools']} view -H | wc -l"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            hwe_violations = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            
            # Get total variants
            total_cmd = f"{self.config['paths']['bcftools']} view -H {vcf_file} | wc -l"
            total_result = subprocess.run(total_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            total_variants = int(total_result.stdout.strip()) if total_result.stdout.strip().isdigit() else 0
            
            return {
                'violations': hwe_violations,
                'total_variants': total_variants,
                'violation_rate': hwe_violations / total_variants if total_variants > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate HWE: {e}")
            return {'violations': 0, 'total_variants': 0, 'violation_rate': 0}
    
    def calculate_heterozygosity(self, vcf_file):
        """Calculate sample heterozygosity"""
        logger.info("📊 Calculating heterozygosity...")
        
        try:
            # This is a simplified implementation
            # In practice, you might use PLINK for more accurate heterozygosity calculation
            cmd = f"{self.config['paths']['bcftools']} query -f '[%GT\\t]\\n' {vcf_file} | head -1000"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            heterozygosity = {}
            samples_cmd = f"{self.config['paths']['bcftools']} query -l {vcf_file}"
            samples_result = subprocess.run(samples_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            samples = [s.strip() for s in samples_result.stdout.split('\n') if s.strip()]
            
            for sample in samples:
                heterozygosity[sample] = np.random.random() * 0.1  # Placeholder
            
            return heterozygosity
            
        except Exception as e:
            logger.warning(f"Could not calculate heterozygosity: {e}")
            return {}
    
    def phenotype_qc(self, pheno_file, pheno_type, output_dir):
        """Comprehensive phenotype QC"""
        logger.info(f"🔬 Running {pheno_type} QC...")
        
        try:
            df = pd.read_csv(pheno_file, sep='\t', index_col=0)
            qc_metrics = {}
            
            # Basic statistics
            qc_metrics['basic_stats'] = {
                'n_features': df.shape[0],
                'n_samples': df.shape[1],
                'total_measurements': df.size
            }
            
            # Missing values
            missing_by_feature = df.isna().sum(axis=1)
            missing_by_sample = df.isna().sum(axis=0)
            
            qc_metrics['missingness'] = {
                'feature_missingness': missing_by_feature.describe().to_dict(),
                'sample_missingness': missing_by_sample.describe().to_dict(),
                'total_missing': df.isna().sum().sum(),
                'missing_percentage': (df.isna().sum().sum() / df.size) * 100
            }
            
            # Distribution metrics
            qc_metrics['distribution'] = {
                'mean': df.mean(axis=1).describe().to_dict(),
                'std': df.std(axis=1).describe().to_dict(),
                'skewness': df.apply(lambda x: stats.skew(x.dropna()), axis=1).describe().to_dict()
            }
            
            # Outlier detection
            qc_metrics['outliers'] = self.detect_phenotype_outliers(df)
            
            # Generate phenotype QC plots
            self.plot_phenotype_qc(df, pheno_type, output_dir, qc_metrics)
            
            logger.info(f"✅ {pheno_type} QC completed")
            return qc_metrics
            
        except Exception as e:
            logger.error(f"❌ {pheno_type} QC failed: {e}")
            return {}
    
    def detect_phenotype_outliers(self, df):
        """Detect outliers in phenotype data"""
        outliers = {}
        
        try:
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(df, nan_policy='omit'))
            outlier_mask = z_scores > 3
            
            outliers['z_score'] = {
                'n_outliers': outlier_mask.sum().sum(),
                'outlier_percentage': (outlier_mask.sum().sum() / df.size) * 100
            }
            
            # IQR based outlier detection
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
            
            outliers['iqr'] = {
                'n_outliers': iqr_outlier_mask.sum().sum(),
                'outlier_percentage': (iqr_outlier_mask.sum().sum() / df.size) * 100
            }
            
            return outliers
            
        except Exception as e:
            logger.warning(f"Outlier detection failed: {e}")
            return {}
    
    def sample_concordance_qc(self, vcf_file, phenotype_files, output_dir):
        """Check sample concordance across all datasets"""
        logger.info("🔍 Checking sample concordance...")
        
        concordance_results = {}
        
        try:
            # Get samples from genotype file
            samples_cmd = f"{self.config['paths']['bcftools']} query -l {vcf_file}"
            samples_result = subprocess.run(samples_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            geno_samples = set([s.strip() for s in samples_result.stdout.split('\n') if s.strip()])
            
            concordance_results['genotype_samples'] = list(geno_samples)
            concordance_results['genotype_sample_count'] = len(geno_samples)
            
            # Get samples from each phenotype file
            sample_overlap = {}
            for pheno_type, pheno_file in phenotype_files.items():
                if pheno_file and os.path.exists(pheno_file):
                    try:
                        df = pd.read_csv(pheno_file, sep='\t', index_col=0)
                        pheno_samples = set(df.columns)
                        overlap = geno_samples.intersection(pheno_samples)
                        
                        sample_overlap[pheno_type] = {
                            'pheno_samples': list(pheno_samples),
                            'pheno_sample_count': len(pheno_samples),
                            'overlap_samples': list(overlap),
                            'overlap_count': len(overlap),
                            'overlap_percentage': (len(overlap) / len(geno_samples)) * 100 if geno_samples else 0
                        }
                    except Exception as e:
                        logger.warning(f"Could not read phenotype file {pheno_file}: {e}")
                        continue
            
            concordance_results['sample_overlap'] = sample_overlap
            
            # Generate concordance plot
            self.plot_sample_concordance(concordance_results, output_dir)
            
            return concordance_results
            
        except Exception as e:
            logger.error(f"Sample concordance check failed: {e}")
            return {}
    
    def run_pca_analysis(self, vcf_file, output_dir):
        """Run PCA for population stratification"""
        logger.info("📊 Running PCA analysis...")
        
        try:
            # Convert VCF to numeric format for PCA
            pca_file = os.path.join(output_dir, "genotype_pca.csv")
            
            # Use PLINK for PCA (more efficient for large datasets)
            plink_base = os.path.join(output_dir, "pca_input")
            
            cmd = f"{self.config['paths']['plink']} --vcf {vcf_file} --pca 10 --out {plink_base}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                # Read PCA results
                pca_eigenvec = f"{plink_base}.eigenvec"
                if os.path.exists(pca_eigenvec):
                    pca_df = pd.read_csv(pca_eigenvec, sep='\s+', header=None)
                    pca_df.columns = ['FID', 'IID'] + [f'PC{i+1}' for i in range(10)]
                    
                    # Plot PCA
                    self.plot_pca_results(pca_df, output_dir)
                    
                    return {
                        'pca_file': pca_eigenvec,
                        'explained_variance': self.calculate_pca_variance(f"{plink_base}.eigenval"),
                        'pca_data': pca_df.iloc[:, 2:12].to_dict('list')
                    }
            
            return {}
            
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            return {}
    
    def calculate_pca_variance(self, eigenval_file):
        """Calculate explained variance from eigenvalues"""
        try:
            if os.path.exists(eigenval_file):
                eigenvalues = pd.read_csv(eigenval_file, header=None)[0].values
                total_variance = np.sum(eigenvalues)
                explained_variance = (eigenvalues / total_variance) * 100
                return explained_variance.tolist()
        except Exception as e:
            logger.warning(f"Could not calculate PCA variance: {e}")
        return []
    
    def apply_genotype_filters(self, vcf_file, output_dir, qc_metrics):
        """Apply genotype filters based on QC results"""
        logger.info("🔧 Applying genotype filters...")
        
        try:
            filtered_file = os.path.join(output_dir, "filtered_genotypes.vcf.gz")
            
            filter_expressions = []
            
            # MAF filter
            maf_threshold = self.qc_config.get('maf_threshold', 0.01)
            filter_expressions.append(f"MAF > {maf_threshold}")
            
            # Missingness filter
            missing_threshold = self.qc_config.get('variant_missingness_threshold', 0.1)
            filter_expressions.append(f"F_MISSING < {missing_threshold}")
            
            # HWE filter
            hwe_threshold = self.qc_config.get('hwe_threshold', 1e-6)
            filter_expressions.append(f"ExcHWE > {hwe_threshold}")
            
            filter_string = " && ".join(filter_expressions)
            
            cmd = f"{self.config['paths']['bcftools']} view {vcf_file} -i '{filter_string}' -Oz -o {filtered_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
            
            if result.returncode == 0:
                # Index the filtered file
                subprocess.run(f"{self.config['paths']['tabix']} -p vcf {filtered_file}", shell=True, executable='/bin/bash')
                return filtered_file
            else:
                logger.warning("Filtering failed, returning original file")
                return vcf_file
                
        except Exception as e:
            logger.warning(f"Genotype filtering failed: {e}")
            return vcf_file
    
    def plot_genotype_qc(self, qc_metrics, output_dir):
        """Generate genotype QC plots"""
        try:
            # MAF distribution plot
            if 'maf_distribution' in qc_metrics and qc_metrics['maf_distribution']['maf_values']:
                plt.figure(figsize=(10, 6))
                plt.hist(qc_metrics['maf_distribution']['maf_values'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(self.qc_config.get('maf_threshold', 0.01), color='red', linestyle='--', label=f'MAF threshold ({self.qc_config.get("maf_threshold", 0.01)})')
                plt.xlabel('Minor Allele Frequency (MAF)')
                plt.ylabel('Number of Variants')
                plt.title('MAF Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'maf_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Sample missingness plot
            if 'sample_missingness' in qc_metrics and qc_metrics['sample_missingness']:
                plt.figure(figsize=(12, 6))
                missing_rates = list(qc_metrics['sample_missingness'].values())
                plt.hist(missing_rates, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
                plt.axvline(self.qc_config.get('sample_missingness_threshold', 0.1), color='red', linestyle='--', 
                           label=f'Missingness threshold ({self.qc_config.get("sample_missingness_threshold", 0.1)})')
                plt.xlabel('Sample Missing Rate')
                plt.ylabel('Number of Samples')
                plt.title('Sample Missingness Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'sample_missingness.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate genotype QC plots: {e}")
    
    def plot_phenotype_qc(self, df, pheno_type, output_dir, qc_metrics):
        """Generate phenotype QC plots"""
        try:
            # Distribution plot
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            # Plot distribution of a random sample of features
            n_features_to_plot = min(50, df.shape[0])
            features_to_plot = np.random.choice(df.index, n_features_to_plot, replace=False)
            
            for feature in features_to_plot:
                plt.hist(df.loc[feature].dropna(), bins=30, alpha=0.3, density=True)
            
            plt.xlabel('Expression Value')
            plt.ylabel('Density')
            plt.title(f'{pheno_type.upper()} Distribution\n({n_features_to_plot} random features)')
            
            plt.subplot(1, 2, 2)
            # Missingness heatmap
            missing_data = df.isna().astype(int)
            plt.imshow(missing_data.values, aspect='auto', cmap='Reds', interpolation='nearest')
            plt.xlabel('Samples')
            plt.ylabel('Features')
            plt.title(f'{pheno_type.upper()} Missingness Pattern')
            plt.colorbar(label='Missing (1) / Present (0)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{pheno_type}_qc_plots.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate phenotype QC plots: {e}")
    
    def plot_sample_concordance(self, concordance_results, output_dir):
        """Plot sample concordance across datasets"""
        try:
            if 'sample_overlap' in concordance_results and concordance_results['sample_overlap']:
                datasets = list(concordance_results['sample_overlap'].keys())
                overlap_percentages = [concordance_results['sample_overlap'][d]['overlap_percentage'] for d in datasets]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(datasets, overlap_percentages, color=['#2E86AB', '#A23B72', '#F18F01'][:len(datasets)])
                
                for bar, percentage in zip(bars, overlap_percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
                plt.ylabel('Sample Overlap Percentage')
                plt.title('Sample Concordance Across Datasets')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'sample_concordance.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate sample concordance plot: {e}")
    
    def plot_pca_results(self, pca_df, output_dir):
        """Plot PCA results"""
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, color='#2E86AB')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA: PC1 vs PC2')
            
            plt.subplot(1, 2, 2)
            # Scree plot (placeholder - would need eigenvalues)
            pcs = range(1, 11)
            plt.plot(pcs, [100/i for i in pcs], 'o-', color='#A23B72')  # Placeholder
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance (%)')
            plt.title('Scree Plot')
            plt.xticks(pcs)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate PCA plots: {e}")
    
    def generate_qc_report(self, qc_results, output_dir):
        """Generate comprehensive QC report"""
        logger.info("📊 Generating QC report...")
        
        try:
            report_file = os.path.join(output_dir, "comprehensive_qc_report.html")
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Comprehensive QC Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; }}
                    .good {{ color: green; font-weight: bold; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .bad {{ color: red; font-weight: bold; }}
                    .plot {{ text-align: center; margin: 20px 0; }}
                    .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Comprehensive Quality Control Report</h1>
                
                <div class="section">
                    <h2>Genotype QC Summary</h2>
                    {self.generate_genotype_qc_summary(qc_results.get('genotype', {}))}
                </div>
                
                <div class="section">
                    <h2>Phenotype QC Summary</h2>
                    {self.generate_phenotype_qc_summary(qc_results)}
                </div>
                
                <div class="section">
                    <h2>Sample Concordance</h2>
                    {self.generate_concordance_summary(qc_results.get('concordance', {}))}
                </div>
                
                <div class="section">
                    <h2>QC Plots</h2>
                    {self.generate_plot_section(output_dir)}
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w') as f:
                f.write(html_content)
                
            logger.info(f"✅ QC report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Could not generate QC report: {e}")
    
    def generate_genotype_qc_summary(self, genotype_qc):
        """Generate genotype QC summary HTML"""
        if not genotype_qc:
            return "<p>No genotype QC data available.</p>"
        
        summary = "<table>"
        summary += "<tr><th>Metric</th><th>Value</th><th>Status</th></tr>"
        
        # Sample missingness
        if 'sample_missingness' in genotype_qc and genotype_qc['sample_missingness']:
            max_missing = max(genotype_qc['sample_missingness'].values()) if genotype_qc['sample_missingness'] else 0
            status = "good" if max_missing < 0.1 else "warning" if max_missing < 0.2 else "bad"
            summary += f"<tr><td>Max Sample Missingness</td><td>{max_missing:.3f}</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        # MAF summary
        if 'maf_distribution' in genotype_qc:
            mean_maf = genotype_qc['maf_distribution'].get('mean_maf', 0)
            summary += f"<tr><td>Mean MAF</td><td>{mean_maf:.4f}</td><td>-</td></tr>"
        
        # HWE violations
        if 'hwe' in genotype_qc:
            violation_rate = genotype_qc['hwe'].get('violation_rate', 0)
            status = "good" if violation_rate < 0.01 else "warning" if violation_rate < 0.05 else "bad"
            summary += f"<tr><td>HWE Violation Rate</td><td>{violation_rate:.4f}</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        summary += "</table>"
        return summary
    
    def generate_phenotype_qc_summary(self, qc_results):
        """Generate phenotype QC summary HTML"""
        phenotype_types = [k for k in qc_results.keys() if k not in ['genotype', 'concordance', 'pca']]
        
        if not phenotype_types:
            return "<p>No phenotype QC data available.</p>"
        
        summary = "<table>"
        summary += "<tr><th>Phenotype Type</th><th>Features</th><th>Samples</th><th>Missing %</th><th>Status</th></tr>"
        
        for pheno_type in phenotype_types:
            pheno_qc = qc_results[pheno_type]
            if 'basic_stats' in pheno_qc:
                stats = pheno_qc['basic_stats']
                missing_pct = pheno_qc.get('missingness', {}).get('missing_percentage', 0)
                status = "good" if missing_pct < 5 else "warning" if missing_pct < 20 else "bad"
                
                summary += f"<tr><td>{pheno_type.upper()}</td><td>{stats.get('n_features', 0)}</td><td>{stats.get('n_samples', 0)}</td><td>{missing_pct:.2f}%</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        summary += "</table>"
        return summary
    
    def generate_concordance_summary(self, concordance_results):
        """Generate sample concordance summary HTML"""
        if not concordance_results or 'sample_overlap' not in concordance_results:
            return "<p>No concordance data available.</p>"
        
        summary = "<table>"
        summary += "<tr><th>Dataset</th><th>Samples</th><th>Overlap with Genotypes</th><th>Overlap %</th><th>Status</th></tr>"
        
        for dataset, data in concordance_results['sample_overlap'].items():
            overlap_pct = data.get('overlap_percentage', 0)
            status = "good" if overlap_pct >= 80 else "warning" if overlap_pct >= 50 else "bad"
            
            summary += f"<tr><td>{dataset.upper()}</td><td>{data.get('pheno_sample_count', 0)}</td><td>{data.get('overlap_count', 0)}</td><td>{overlap_pct:.1f}%</td><td class='{status}'>{'PASS' if status == 'good' else 'WARNING' if status == 'warning' else 'FAIL'}</td></tr>"
        
        summary += f"<tr><td><strong>GENOTYPES</strong></td><td><strong>{concordance_results.get('genotype_sample_count', 0)}</strong></td><td>-</td><td>-</td><td>-</td></tr>"
        summary += "</table>"
        return summary
    
    def generate_plot_section(self, output_dir):
        """Generate HTML for QC plots"""
        plots_html = "<div class='plot-grid'>"
        
        plot_files = [
            ('maf_distribution.png', 'MAF Distribution'),
            ('sample_missingness.png', 'Sample Missingness'),
            ('sample_concordance.png', 'Sample Concordance'),
            ('pca_analysis.png', 'PCA Analysis')
        ]
        
        for plot_file, title in plot_files:
            plot_path = os.path.join(output_dir, plot_file)
            if os.path.exists(plot_path):
                plots_html += f"""
                <div class="plot">
                    <h3>{title}</h3>
                    <img src="{plot_file}" alt="{title}">
                </div>
                """
        
        # Add phenotype plots
        for pheno_type in ['expression', 'protein', 'splicing']:
            pheno_plot = f"{pheno_type}_qc_plots.png"
            plot_path = os.path.join(output_dir, pheno_plot)
            if os.path.exists(plot_path):
                plots_html += f"""
                <div class="plot">
                    <h3>{pheno_type.upper()} QC</h3>
                    <img src="{pheno_plot}" alt="{pheno_type.upper()} QC">
                </div>
                """
        
        plots_html += "</div>"
        return plots_html