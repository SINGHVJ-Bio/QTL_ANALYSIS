#!/usr/bin/env Rscript
#
# DESeq2 VST Normalization Script for eQTL Analysis
# Author: Dr. Vijay Singh
# Email: vijay.s.gautam@gmail.com
#
# This script performs VST normalization using DESeq2 for RNA-seq count data
# Usage: Rscript deseq2_vst.R <input_file> <output_file> [blind=TRUE] [fit_type=parametric]

# Load required libraries
suppressPackageStartupMessages({
  library(DESeq2)
  library(readr)
  library(dplyr)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript deseq2_vst.R <input_file> <output_file> [blind=TRUE] [fit_type=parametric]")
}

input_file <- args[1]
output_file <- args[2]
blind <- ifelse(length(args) > 2, as.logical(args[3]), TRUE)
fit_type <- ifelse(length(args) > 3, args[4], "parametric")

cat("🔧 DESeq2 VST Normalization Parameters:\n")
cat("  Input file:", input_file, "\n")
cat("  Output file:", output_file, "\n")
cat("  Blind transformation:", blind, "\n")
cat("  Fit type:", fit_type, "\n")

# Read input data
cat("📊 Reading input data...\n")
tryCatch({
  # Read tab-separated file with gene IDs as row names
  count_data <- read_tsv(input_file, show_col_types = FALSE)
  gene_ids <- count_data[[1]]  # First column contains gene IDs
  count_matrix <- as.matrix(count_data[, -1])
  rownames(count_matrix) <- gene_ids
  
  cat("  Loaded", nrow(count_matrix), "genes and", ncol(count_matrix), "samples\n")
  
  # Check if data contains negative values (not appropriate for count data)
  if (any(count_matrix < 0, na.rm = TRUE)) {
    warning("Input data contains negative values. DESeq2 expects count data (non-negative integers).")
  }
  
  # Check if data contains non-integer values
  if (any(count_matrix != floor(count_matrix), na.rm = TRUE)) {
    cat("⚠️  Input data contains non-integer values. Converting to integers...\n")
    count_matrix <- round(count_matrix)
  }
  
  # Remove genes with all zeros or constant expression
  non_zero_genes <- rowSums(count_matrix > 0) > 0
  count_matrix <- count_matrix[non_zero_genes, ]
  cat("  After filtering zero genes:", nrow(count_matrix), "genes remaining\n")
  
  # Remove genes with too many zeros (>80% samples)
  zero_threshold <- 0.8 * ncol(count_matrix)
  low_zero_genes <- rowSums(count_matrix == 0) < zero_threshold
  count_matrix <- count_matrix[low_zero_genes, ]
  cat("  After filtering high-zero genes:", nrow(count_matrix), "genes remaining\n")
  
  # Create DESeq2 dataset
  cat("🔬 Creating DESeq2 dataset...\n")
  col_data <- data.frame(sample = colnames(count_matrix))
  rownames(col_data) <- colnames(count_matrix)
  
  dds <- DESeqDataSetFromMatrix(
    countData = count_matrix,
    colData = col_data,
    design = ~ 1  # No design matrix for basic normalization
  )
  
  # Filter lowly expressed genes (DESeq2 recommendation)
  cat("🔍 Filtering lowly expressed genes...\n")
  keep <- rowSums(counts(dds)) >= 10
  dds <- dds[keep, ]
  cat("  After DESeq2 filtering:", nrow(dds), "genes remaining\n")
  
  # Perform VST normalization
  cat("🔄 Performing VST normalization...\n")
  vsd <- vst(dds, blind = blind, fitType = fit_type)
  
  # Extract normalized counts
  cat("💾 Extracting normalized expression values...\n")
  vst_normalized <- assay(vsd)
  
  # Add gene IDs back
  vst_df <- as.data.frame(vst_normalized)
  vst_df$gene_id <- rownames(vst_normalized)
  
  # Reorder columns to have gene_id first
  vst_df <- vst_df[, c("gene_id", setdiff(colnames(vst_df), "gene_id"))]
  
  # Write output
  cat("📁 Writing normalized data to:", output_file, "\n")
  write_tsv(vst_df, output_file)
  
  # Generate summary statistics
  cat("\n📈 VST Normalization Summary:\n")
  cat("  Input genes:", length(gene_ids), "\n")
  cat("  Final genes:", nrow(vst_df), "\n")
  cat("  Samples:", ncol(vst_normalized), "\n")
  cat("  Mean expression (VST):", round(mean(vst_normalized), 2), "\n")
  cat("  SD expression (VST):", round(sd(vst_normalized), 2), "\n")
  cat("  Range (VST):", round(range(vst_normalized), 2), "\n")
  
  cat("✅ DESeq2 VST normalization completed successfully!\n")
  
}, error = function(e) {
  cat("❌ Error in DESeq2 VST normalization:", e$message, "\n")
  quit(status = 1)
})