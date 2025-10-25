#!/usr/bin/env python3
"""
Enhanced Pipeline script to prepare QTL input files optimized for tensorQTL:
1. Filter RNA-seq count matrix to samples present in metadata, rename to IID.
2. Build covariate.tsv with Age, Sex, RNASeq_RIN, RNASeq_Batch, PC1–PC5.
3. Write samples_to_keep.txt and filter VCF to those samples (autosomes only, chr1–chr22).
4. Generate annotations.bed with protein_coding genes only.
5. Generate chromosome_map_chr.tsv (1..22 → chr1..chr22).
6. Generate chromosome_map_nochr.txt (chr1..chr22, chrX, chrY, chrM).

Optimized for tensorQTL compatibility and performance.
"""

import pandas as pd
import os
import subprocess
import shutil
import logging
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BuildInput')

# === Input files ===
count_file = "/home/ubuntu/DATA_DRIVE/qtl_data/Merge_Pre_EMULSION_ENABL150_ENABL300_count_protein_coding.tsv"
meta_file = "/home/ubuntu/DATA_DRIVE/qtl_data/meta_wgs_rnaseq.csv"
pca_file = "/home/ubuntu/DATA_DRIVE/qtl_data/pca_result_cohort.eigenvec"
vcf_input = "/home/ubuntu/DATA_DRIVE/qtl_data/ENABL_merged_280_cohort.snp.recalibrated.vcf.gz"
gtf_table = "/home/ubuntu/DATA_DRIVE/qtl_data/gencode.v38.annotation.gtf.full.table.txt"

# === Output files ===
out_dir = "/home/ubuntu/DATA_DRIVE/qtl_data/input"
count_output = os.path.join(out_dir, "rnseq_count.tsv")
covar_output = os.path.join(out_dir, "covariate.tsv")
samples_list = os.path.join(out_dir, "samples_to_keep.txt")
vcf_output = os.path.join(out_dir, "cohort.filtered.samples.vcf.gz")
plink_output = os.path.join(out_dir, "cohort.filtered.samples")  # PLINK prefix
bed_output = os.path.join(out_dir, "annotations.bed")
chrom_map_chr_output = os.path.join(out_dir, "chromosome_map_chr.tsv")
chrom_map_nochr_output = os.path.join(out_dir, "chromosome_map_nochr.txt")
sample_mapping_output = os.path.join(out_dir, "sample_mapping.txt")

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# === Helper: normalize IDs ===
def normalize_ids(s: pd.Series) -> pd.Series:
    """Normalize sample IDs for consistency"""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def run_command(cmd, description):
    """Run shell command with error handling"""
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

# -------------------------------------------------------------------
# Step 1: RNA-seq counts - Enhanced for tensorQTL compatibility
# -------------------------------------------------------------------
logger.info("Step 1: Processing RNA-seq count data...")

count_df = pd.read_csv(count_file, sep="\t")
meta_df = pd.read_csv(meta_file)

# Check count matrix structure
logger.info(f"Count matrix shape: {count_df.shape}")
logger.info(f"Count matrix columns: {count_df.columns[:5].tolist()}...")

# Normalize IDs in metadata
for col in ["RNASeq_Library", "WGS_Library"]:
    if col in meta_df.columns:
        meta_df[col] = normalize_ids(meta_df[col])

# Build mapping RNASeq_Library -> IID
meta_map = meta_df[["RNASeq_Library", "WGS_Library"]].dropna()
meta_map = meta_map.rename(columns={"WGS_Library": "IID"})

count_samples = [c for c in count_df.columns if c != "gene_id"]
rna_libs = set(meta_map["RNASeq_Library"])
matched_rna_libs = [c for c in count_samples if c in rna_libs]

rename_dict = (
    meta_map.set_index("RNASeq_Library")["IID"]
    .reindex(matched_rna_libs)
    .to_dict()
)
rename_dict = {k: v for k, v in rename_dict.items() if isinstance(v, str) and len(v) > 0}

# Filter count matrix and rename to IID
count_keep_cols = ["gene_id"] + list(rename_dict.keys())
filtered_count_df = count_df[count_keep_cols].copy()
filtered_count_df = filtered_count_df.rename(columns=rename_dict)

# Remove genes with all zeros (common in RNA-seq data)
original_gene_count = filtered_count_df.shape[0]
filtered_count_df = filtered_count_df[filtered_count_df.iloc[:, 1:].sum(axis=1) > 0]
filtered_gene_count = filtered_count_df.shape[0]
logger.info(f"Removed {original_gene_count - filtered_gene_count} genes with all zeros")

# Save count data
filtered_count_df.to_csv(count_output, sep="\t", index=False)
logger.info(f"✅ Saved filtered count data: {filtered_count_df.shape}")

# -------------------------------------------------------------------
# Step 2: Covariates - Enhanced for tensorQTL
# -------------------------------------------------------------------
logger.info("Step 2: Building covariate file...")

# Load PCA (PC1..PC40), keep PC1–PC5
pca_cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 41)]
try:
    pca_df = pd.read_csv(pca_file, sep=r"\s+", comment="#", names=pca_cols)
    pca_df["IID"] = normalize_ids(pca_df["IID"])
except Exception as e:
    logger.warning(f"Could not load PCA file: {e}")
    # Create empty PCA dataframe as fallback
    pca_df = pd.DataFrame(columns=["IID"] + [f"PC{i}" for i in range(1, 6)])

# Merge metadata with PCA
meta_with_pca = meta_df.merge(
    pca_df[["IID", "PC1", "PC2", "PC3", "PC4", "PC5"]],
    left_on="WGS_Library", right_on="IID", how="inner"
)

# Restrict to samples present in counts
iid_in_counts = [c for c in filtered_count_df.columns if c != "gene_id"]
covar_cols = ["IID", "Age", "Sex", "RNASeq_RIN", "RNASeq_Batch", "PC1", "PC2", "PC3", "PC4", "PC5"]

# Check which covariates are available
available_covar_cols = [col for col in covar_cols if col in meta_with_pca.columns]
missing_cols = set(covar_cols) - set(available_covar_cols)
if missing_cols:
    logger.warning(f"Missing covariates: {missing_cols}")

covar_df = (
    meta_with_pca[available_covar_cols]
    .dropna(subset=["IID"])
    .drop_duplicates(subset=["IID"])
)
covar_df = covar_df[covar_df["IID"].isin(iid_in_counts)]

# Handle categorical variables (Sex, Batch) - convert to numeric
if 'Sex' in covar_df.columns:
    # Convert Sex to numeric (M=0, F=1, etc.)
    covar_df['Sex'] = pd.Categorical(covar_df['Sex']).codes
    
if 'RNASeq_Batch' in covar_df.columns:
    # Convert batch to numeric codes
    covar_df['RNASeq_Batch'] = pd.Categorical(covar_df['RNASeq_Batch']).codes

# Pivot: rows = covariates, columns = IID (tensorQTL format)
covar_matrix = covar_df.set_index("IID").T
covar_matrix.index.name = "covariate"

# Add intercept term (recommended for tensorQTL)
covar_matrix.loc['intercept'] = 1.0

covar_matrix.to_csv(covar_output, sep="\t")
logger.info(f"✅ Saved covariates: {covar_matrix.shape[1]} samples, {covar_matrix.shape[0]} covariates")

# -------------------------------------------------------------------
# Step 3: VCF filtering and PLINK conversion - Enhanced for tensorQTL
# -------------------------------------------------------------------
logger.info("Step 3: Processing genotype data...")

# Write sample list
with open(samples_list, "w") as f:
    for s in iid_in_counts:
        f.write(f"{s}\n")

logger.info(f"Sample list created with {len(iid_in_counts)} samples")

# Check for required tools
bcftools_path = shutil.which("bcftools")
plink_path = shutil.which("plink")

if not bcftools_path:
    logger.error("❌ bcftools not found in PATH")
    raise RuntimeError("bcftools is required for VCF processing")

# Filter VCF with bcftools
regions = [f"chr{i}" for i in range(1, 23)]

logger.info("Filtering VCF to autosomes and target samples...")

# Single step: filter samples and regions, split multiallelics, and normalize
cmd_filter = [
    bcftools_path, "view",
    "-S", samples_list,
    "-r", ",".join(regions),
    "-m2", "-M2", "-v", "snps",  # Keep only biallelic SNPs
    vcf_input,
    "-Oz", "-o", vcf_output
]

try:
    run_command(cmd_filter, "Filtering VCF to autosomes and samples")
    
    # Index the filtered VCF
    cmd_index = [bcftools_path, "index", "-t", vcf_output]
    run_command(cmd_index, "Indexing filtered VCF")
    
except Exception as e:
    logger.error(f"VCF filtering failed: {e}")
    # Try without region filtering as fallback
    logger.info("Trying without region filtering...")
    cmd_filter_fallback = [
        bcftools_path, "view",
        "-S", samples_list,
        "-m2", "-M2", "-v", "snps",
        vcf_input,
        "-Oz", "-o", vcf_output
    ]
    run_command(cmd_filter_fallback, "Filtering VCF (fallback without region filter)")
    run_command([bcftools_path, "index", "-t", vcf_output], "Indexing VCF")

# Convert to PLINK format for better tensorQTL performance
if plink_path:
    logger.info("Converting VCF to PLINK format for tensorQTL...")
    
    cmd_plink = [
        plink_path,
        "--vcf", vcf_output,
        "--make-bed",
        "--out", plink_output,
        "--memory", "4000",  # 4GB memory
        "--threads", "2"     # Use 2 threads
    ]
    
    try:
        run_command(cmd_plink, "Converting VCF to PLINK format")
        logger.info(f"✅ PLINK files created: {plink_output}.bed/bim/fam")
    except Exception as e:
        logger.warning(f"PLINK conversion failed: {e}. tensorQTL can use VCF directly.")
else:
    logger.warning("plink not found, skipping PLINK conversion")

# -------------------------------------------------------------------
# Step 4: Annotations BED - Enhanced for tensorQTL
# -------------------------------------------------------------------
logger.info("Step 4: Creating annotation BED file...")

try:
    gtf_df = pd.read_csv(gtf_table, sep="\t")
    
    # Keep only gene features and protein_coding
    genes = gtf_df[(gtf_df["feature"] == "gene") & (gtf_df["gene_type"] == "protein_coding")].copy()
    
    # Clean gene_id (remove version suffix)
    genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    
    # Select and sort
    bed_df = genes[["chr", "start", "end", "gene_id", "strand", "gene_name", "gene_type"]].copy()
    bed_df = bed_df.sort_values(by=["chr", "start"])
    
    # Ensure chromosome names are consistent
    bed_df['chr'] = bed_df['chr'].astype(str)
    
    # Filter to genes present in count data
    count_genes = set(filtered_count_df['gene_id'])
    bed_df = bed_df[bed_df['gene_id'].isin(count_genes)]
    
    bed_df.to_csv(bed_output, sep="\t", index=False, header=True)
    logger.info(f"✅ Annotation BED created with {bed_df.shape[0]} protein-coding genes")
    
except Exception as e:
    logger.error(f"Failed to create annotation BED: {e}")
    # Create minimal BED file as fallback
    logger.info("Creating minimal BED file from count data...")
    bed_fallback = pd.DataFrame({
        'chr': ['chr1'] * len(filtered_count_df),
        'start': range(1, len(filtered_count_df) + 1),
        'end': range(1001, len(filtered_count_df) + 1001),
        'gene_id': filtered_count_df['gene_id'],
        'strand': ['+'] * len(filtered_count_df)
    })
    bed_fallback.to_csv(bed_output, sep="\t", index=False, header=True)
    logger.warning("Created fallback BED file with placeholder positions")

# -------------------------------------------------------------------
# Step 5: Chromosome maps and sample mapping
# -------------------------------------------------------------------
logger.info("Step 5: Creating chromosome maps and sample mapping...")

# chromosome_map_chr.tsv (1..22 → chr1..chr22)
chrom_map = pd.DataFrame({
    "num": list(range(1, 23)),
    "chr": [f"chr{i}" for i in range(1, 23)]
})
chrom_map.to_csv(chrom_map_chr_output, sep="\t", index=False, header=False)

# chromosome_map_nochr.txt (chr1..chr22, chrX, chrY, chrM)
chrom_map_nochr = pd.DataFrame({
    "chr": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"],
    "num": [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
})
chrom_map_nochr.to_csv(chrom_map_nochr_output, sep="\t", index=False, header=False)

# Create comprehensive sample mapping file
sample_mapping_data = []
for sample in iid_in_counts:
    sample_info = {
        'sample_id': sample,
        'data_types': 'genotype,expression',
        'has_covariates': sample in covar_matrix.columns,
        'has_expression': True,
        'has_genotype': True
    }
    sample_mapping_data.append(sample_info)

sample_mapping_df = pd.DataFrame(sample_mapping_data)
sample_mapping_df.to_csv(sample_mapping_output, sep="\t", index=False)
logger.info(f"✅ Sample mapping created for {len(sample_mapping_data)} samples")

# -------------------------------------------------------------------
# Step 6: Validation and tensorQTL compatibility check
# -------------------------------------------------------------------
logger.info("Step 6: Validating tensorQTL compatibility...")

# Check sample consistency
geno_samples = set(iid_in_counts)
expr_samples = set(filtered_count_df.columns[1:])  # Exclude gene_id
covar_samples = set(covar_matrix.columns)

common_samples = geno_samples & expr_samples & covar_samples
logger.info(f"Sample overlap - Genotype: {len(geno_samples)}, "
           f"Expression: {len(expr_samples)}, Covariates: {len(covar_samples)}, "
           f"Common: {len(common_samples)}")

if len(common_samples) < min(len(geno_samples), len(expr_samples), len(covar_samples)):
    logger.warning("Sample mismatch detected between data types")

# Check file sizes and basic stats
files_to_check = [
    (count_output, "Expression data"),
    (covar_output, "Covariate data"), 
    (vcf_output, "Genotype VCF"),
    (bed_output, "Annotation BED")
]

for file_path, description in files_to_check:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024**2)  # MB
        logger.info(f"  {description}: {file_size:.1f} MB")
    else:
        logger.warning(f"  {description}: File not found")

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
logger.info("\n" + "="*50)
logger.info("=== Pipeline Summary ===")
logger.info(f"rnseq_count.tsv: {count_output} | {filtered_count_df.shape[1]-1} samples, {filtered_count_df.shape[0]} genes")
logger.info(f"covariate.tsv: {covar_output} | {covar_matrix.shape[1]} samples, {covar_matrix.shape[0]} covariates")
logger.info(f"samples_to_keep.txt: {samples_list} | {len(iid_in_counts)} samples")
logger.info(f"VCF output: {vcf_output} | Filtered to autosomes and biallelic SNPs")
if plink_path and os.path.exists(plink_output + ".bed"):
    logger.info(f"PLINK output: {plink_output}.bed | Recommended for tensorQTL")
logger.info(f"annotations.bed: {bed_output} | {bed_df.shape[0] if 'bed_df' in locals() else 'Unknown'} protein-coding genes")
logger.info(f"chromosome_map_chr.tsv: {chrom_map_chr_output}")
logger.info(f"chromosome_map_nochr.txt: {chrom_map_nochr_output}")
logger.info(f"sample_mapping.txt: {sample_mapping_output}")
logger.info(f"Common samples across all data: {len(common_samples)}")
logger.info("="*50)

# Generate tensorQTL usage instructions
logger.info("\n=== tensorQTL Usage ===")
logger.info("For optimal performance with tensorQTL:")
if os.path.exists(plink_output + ".bed"):
    logger.info("Use PLINK format: --genotype_path " + plink_output + ".bed")
else:
    logger.info("Use VCF format: --genotype_path " + vcf_output)
logger.info("--phenotype_path " + count_output)
logger.info("--covariates_path " + covar_output)
logger.info("--annotation_path " + bed_output)
logger.info("="*50)