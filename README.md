# QTL Analysis Pipeline

**Author**: Dr. Vijay Singh  
**Email**: vijay.s.gautam@gmail.com  
**Version**: 1.0  

---

## 📖 Overview

The **Enhanced QTL Analysis Pipeline** is a comprehensive, production-ready framework for performing **cis/trans quantitative trait locus (QTL) mapping**. It includes advanced features such as interaction analysis, fine-mapping, and extensive quality control.  

The pipeline supports multiple molecular phenotypes (**expression, protein, splicing**) and provides rich **visualization and reporting** capabilities.

---

## 📁 Project Structure

QTL_ANALYSIS
├── config
│   ├── config.yaml
│   └── test_config.yaml
├── create_sample_data.py
├── data
│   ├── annotations.bed
│   ├── chromosome_map_chr.txt
│   ├── chromosome_map_nochr.txt
│   ├── covariates.txt
│   ├── expression.txt
│   ├── genotypes.vcf
│   ├── gwas_phenotype.txt
│   ├── protein.txt
│   └── splicing.txt
├── documentation
│   ├── QTL Analysis Pipeline - Complete Documentation.pdf
│   └── QTL Analysis Pipeline - Use Cases Guide.pdf
├── environment.yml
├── fix_sample_data.py
├── README_QTL_ANALYSIS.pdf
├── requirements.txt
├── run_QTLPipeline.py
├── scripts
│   ├── __pycache__
│   │   └── main.cpython-313.pyc
│   ├── analysis
│   │   ├── __init__.py
│   │   ├── fine_mapping.py
│   │   └── interaction_analysis.py
│   ├── main.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-313.pyc
│       │   ├── genotype_processing.cpython-313.pyc
│       │   ├── gwas_analysis.cpython-313.pyc
│       │   ├── plotting.cpython-313.pyc
│       │   ├── qtl_analysis.cpython-313.pyc
│       │   ├── report_generator.cpython-313.pyc
│       │   └── validation.cpython-313.pyc
│       ├── advanced_plotting.py
│       ├── enhanced_qc.py
│       ├── genotype_processing.py
│       ├── gwas_analysis.py
│       ├── plotting.py
│       ├── qtl_analysis.py
│       ├── report_generator.py
│       └── validation.py
└── test_pipeline.py



---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Using conda
conda env create -f environment.yml
conda activate qtl_analysis

# Or using pip
pip install -r requirements.txt

# Validate Environment
python validate_environment.py

# Run Quick Test
python quick_test.py

# Run Analysis
# Basic analysis
python run_QTLPipeline.py --config config/config.yaml

# With advanced features
python run_QTLPipeline.py --config config/config.yaml --enhanced-qc --interaction-analysis

# Performance optimized
python run_QTLPipeline.py --config config/config.yaml --threads 8 --memory 16

# ⚙️ Configuration (Edit config/config.yaml to match your dataset.)
input_files:
  genotypes: "data/genotypes.vcf"
  covariates: "data/covariates.txt"
  annotations: "data/annotations.bed"
  expression: "data/expression.txt"
  protein: "data/protein.txt"
  splicing: "data/splicing.txt"

analysis:
  qtl_types: "all"        # eqtl, pqtl, sqtl, or all
  qtl_mode: "cis"         # cis, trans, or both
  run_gwas: false         # Enable GWAS analysis

qtl:
  cis_window: 1000000     # cis-window size in bp
  permutations: 1000      # Permutations for FDR
  fdr_threshold: 0.05     # False discovery rate

# 📊 Supported Analyses
Core QTL Mapping
    cis-QTL: Local genetic associations within a defined window
    trans-QTL: Distant genetic associations across the genome
    Multiple Phenotypes: eQTL, pQTL, sQTL

Advanced Features
    Interaction Analysis: genotype × covariate interactions
    Fine-mapping: credible set identification (SuSiE/FINEMAP)
    Enhanced QC: PCA-based quality control
    GWAS Integration: optional genome-wide association studies

Visualization & Reporting
    Interactive Manhattan, QQ, and volcano plots
    Comprehensive HTML reports with methodology sections
    QC dashboards and summary statistics

# 🔧 Input File Formats
Genotype Data
    Supported: VCF, VCF.GZ, BCF, PLINK BED
    Automatically detected and processed
    
Phenotype Data
Tab-separated, features as rows, samples as columns:

gene_id    sample1    sample2    sample3
gene1      10.5       11.2       9.8
gene2      8.7        9.1        8.9


Covariates
covariate    sample1    sample2    sample3
age          45         52         38
sex          1          2          1
PC1          0.01       -0.02      0.03


Annotations (BED format)
chr    start    end    gene_id    score    strand
chr1   1000     2000   gene1      0        +
chr1   5000     6000   gene2      0        -


# 🎯 Command Line Options
# Basic usage
python run_QTLPipeline.py --config config/config.yaml

# Custom analysis
python run_QTLPipeline.py --config config/config.yaml --qtl-types eqtl,pqtl --qtl-mode both

# Enable advanced features
python run_QTLPipeline.py --config config/config.yaml --enhanced-qc --interaction-analysis --fine-mapping

# Performance tuning
python run_QTLPipeline.py --config config/config.yaml --threads 8 --memory 16 --output-dir my_results

# Validation only
python run_QTLPipeline.py --config config/config.yaml --validate-only

# Debug mode
python run_QTLPipeline.py --config config/config.yaml --debug


# 📈 Output Structure
results/
├── QTL_results/              # QTL analysis results
│   ├── eqtl_cis_significant.txt
│   ├── eqtl_cis_nominals.txt
│   ├── pqtl_trans_significant.txt
│   └── ...
├── GWAS_results/             # GWAS results (if enabled)
├── plots/
│   ├── static/               # PNG/PDF plots
│   └── interactive/          # HTML interactive plots
├── reports/
│   └── analysis_report.html  # Main HTML report
├── QC_reports/               # Quality control reports
├── genotype_processing/      # Processed genotype files
├── interaction_results/      # Interaction analysis
├── fine_mapping_results/     # Fine-mapping results
├── logs/                     # Execution logs
└── pipeline_summary.txt      # Summary report


# 🔍 Quality Control
The pipeline performs QC at multiple levels:

Sample-level: missingness, heterozygosity, relatedness
Variant-level: MAF, call rate, HWE, quality scores
Phenotype-level: missing values, distribution, outliers
Population stratification: PCA analysis
Sample concordance: cross-dataset matching

# 🐛 Troubleshooting
# Missing Dependencies
python validate_environment.py

#File Path Errors
Verify paths in config/config.yaml
Ensure files exist in data/

# Memory Issues
python run_QTLPipeline.py --config config/config.yaml --memory 16

# Sample Mismatch

Use --enhanced-qc
Review QC_reports/sample_concordance.html

# Getting Help

Check logs in logs/
Run with --debug
Use --validate-only

Contact: vijay.s.gautam@gmail.com
Documentation
Full documentation: documentation/
Configuration guide: config/
Example data generation scripts included

# 🤝 Citation
# If you use this pipeline in your research, please cite:
Singh V. QTL Analysis Pipeline. 2024.
Maintainer: Dr. Vijay Singh 📧 Contact: vijay.s.gautam@gmail.com 📅 Last Updated: 2024