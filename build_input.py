#!/usr/bin/env python3
"""
Pipeline script to prepare QTL input files:
1. Filter RNA-seq count matrix to samples present in metadata, rename to IID.
2. Build covariate.tsv with Age, Sex, RNASeq_RIN, RNASeq_Batch, PC1–PC5.
3. Write samples_to_keep.txt and filter VCF to those samples (autosomes only, chr1–chr22).
4. Generate annotations.bed with protein_coding genes only.
5. Generate chromosome_map_chr.tsv (1..22 → chr1..chr22).
6. Generate chromosome_map_nochr.txt (chr1..chr22, chrX, chrY, chrM).
"""

import pandas as pd
import os
import subprocess
import shutil

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
vcf_output = os.path.join(out_dir, "cohort.filtered.samples_split.vcf.gz")
bed_output = os.path.join(out_dir, "annotations.bed")
chrom_map_chr_output = os.path.join(out_dir, "chromosome_map_chr.tsv")
chrom_map_nochr_output = os.path.join(out_dir, "chromosome_map_nochr.txt")

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# === Helper: normalize IDs ===
def normalize_ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

# -------------------------------------------------------------------
# Step 1: RNA-seq counts
# -------------------------------------------------------------------
count_df = pd.read_csv(count_file, sep="\t")
meta_df = pd.read_csv(meta_file)

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
filtered_count_df.to_csv(count_output, sep="\t", index=False)

# -------------------------------------------------------------------
# Step 2: Covariates
# -------------------------------------------------------------------
# Load PCA (PC1..PC40), keep PC1–PC5
pca_cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 41)]
pca_df = pd.read_csv(pca_file, sep=r"\s+", comment="#", names=pca_cols)
pca_df["IID"] = normalize_ids(pca_df["IID"])

# Merge metadata with PCA
meta_with_pca = meta_df.merge(
    pca_df[["IID", "PC1", "PC2", "PC3", "PC4", "PC5"]],
    left_on="WGS_Library", right_on="IID", how="inner"
)

# Restrict to samples present in counts
iid_in_counts = [c for c in filtered_count_df.columns if c != "gene_id"]
covar_cols = ["IID", "Age", "Sex", "RNASeq_RIN", "RNASeq_Batch", "PC1", "PC2", "PC3", "PC4", "PC5"]

covar_df = (
    meta_with_pca[covar_cols]
    .dropna(subset=["IID"])
    .drop_duplicates(subset=["IID"])
)
covar_df = covar_df[covar_df["IID"].isin(iid_in_counts)]

# Pivot: rows = covariates, columns = IID
covar_matrix = covar_df.set_index("IID").T
covar_matrix.index.name = "ID"
covar_matrix.to_csv(covar_output, sep="\t")

# -------------------------------------------------------------------
# Step 3: VCF filtering (autosomes only) with multiallelic handling
# -------------------------------------------------------------------
# Write sample list
with open(samples_list, "w") as f:
    for s in iid_in_counts:
        f.write(f"{s}\n")

# Filter VCF with bcftools if available
bcftools_path = shutil.which("bcftools")
if bcftools_path:
    # Restrict to chr1–chr22 only and split multiallelic variants
    regions = [f"chr{i}" for i in range(1, 23)]
    
    # Split multiallelic variants first, then filter samples and regions
    temp_vcf = os.path.join(out_dir, "temp_split_multiallelics.vcf.gz")
    
    # Step 1: Split multiallelic variants
    cmd_split = [
        bcftools_path, "norm",
        "-m", "-any",  # Split multiallelic sites
        vcf_input,
        "-Oz", "-o", temp_vcf
    ]
    print("Splitting multiallelic variants:", " ".join(cmd_split))
    subprocess.run(cmd_split, check=True)
    
    # Step 2: Index the split VCF
    cmd_index_temp = [bcftools_path, "index", "-t", temp_vcf]
    subprocess.run(cmd_index_temp, check=True)
    
    # Step 3: Filter to samples and regions
    cmd_view = [
        bcftools_path, "view",
        "-S", samples_list,
        "-r", ",".join(regions),
        "-Oz", "-o", vcf_output,
        temp_vcf
    ]
    print("Filtering to samples and regions:", " ".join(cmd_view))
    subprocess.run(cmd_view, check=True)
    
    # Step 4: Index final VCF
    cmd_index = [bcftools_path, "index", "-t", vcf_output]
    print("Indexing final VCF:", " ".join(cmd_index))
    subprocess.run(cmd_index, check=True)
    
    # Clean up temporary file
    if os.path.exists(temp_vcf):
        os.remove(temp_vcf)
        os.remove(temp_vcf + ".tbi")
    
    vcf_status = "VCF filtered to autosomes (chr1–chr22), multiallelic variants split, and indexed."
else:
    vcf_status = f"bcftools not found. Use {samples_list} and restrict to chr1–chr22 manually."

# -------------------------------------------------------------------
# Step 4: Annotations BED
# -------------------------------------------------------------------
gtf_df = pd.read_csv(gtf_table, sep="\t")

# Keep only gene features and protein_coding
genes = gtf_df[(gtf_df["feature"] == "gene") & (gtf_df["gene_type"] == "protein_coding")].copy()

# Clean gene_id (remove version suffix)
genes["gene_id"] = genes["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)

# Select and sort
bed_df = genes[["chr", "start", "end", "gene_id", "strand", "gene_name", "gene_type"]].copy()
bed_df = bed_df.sort_values(by=["chr", "start"])
bed_df.to_csv(bed_output, sep="\t", index=False, header=True)

# -------------------------------------------------------------------
# Step 5: Chromosome maps
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n=== Pipeline Summary ===")
print(f"rnseq_count.tsv: {count_output} | shape {filtered_count_df.shape}")
print(f"covariate.tsv: {covar_output} | samples {covar_matrix.shape[1]}, covariates {covar_matrix.shape[0]}")
print(f"samples_to_keep.txt: {samples_list} | {len(iid_in_counts)} samples")
print(f"VCF output: {vcf_output} | {vcf_status}")
print(f"annotations.bed: {bed_output} | protein_coding genes {bed_df.shape[0]}")
print(f"chromosome_map_chr.tsv: {chrom_map_chr_output} | {chrom_map.shape[0]} entries")
print(f"chromosome_map_nochr.txt: {chrom_map_nochr_output} | {chrom_map_nochr.shape[0]} entries")