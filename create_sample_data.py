#!/usr/bin/env python3
"""
Script to create sample NAFLD test data for QTL pipeline testing
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create all sample data files for testing"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("🧬 Creating NAFLD sample test data...")
    
    # Sample IDs
    samples = [f"NAFLD_{i:03d}" for i in range(1, 11)]
    
    # Create genotypes.vcf
    create_genotypes_file(data_dir / "genotypes.vcf", samples)
    
    # Create covariates.txt
    create_covariates_file(data_dir / "covariates.txt", samples)
    
    # Create annotations.bed
    create_annotations_file(data_dir / "annotations.bed")
    
    # Create expression.txt
    create_expression_file(data_dir / "expression.txt", samples)
    
    # Create protein.txt
    create_protein_file(data_dir / "protein.txt", samples)
    
    # Create splicing.txt
    create_splicing_file(data_dir / "splicing.txt", samples)
    
    # Create gwas_phenotype.txt
    create_gwas_phenotype_file(data_dir / "gwas_phenotype.txt", samples)
    
    print("✅ All sample data files created successfully!")
    print(f"📁 Data directory: {data_dir.absolute()}")
    
    # Display file sizes
    print("\n📊 File sizes:")
    for file in data_dir.glob("*"):
        size_kb = file.stat().st_size / 1024
        print(f"  {file.name}: {size_kb:.1f} KB")

def create_genotypes_file(file_path, samples):
    """Create sample VCF file with NAFLD-related variants"""
    
    vcf_header = """##fileformat=VCFv4.2
##fileDate=20240115
##source=QTLPipeline_TestData
##reference=GRCh38
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##INFO=<ID=MAF,Number=1,Type=Float,Description="Minor Allele Frequency">
##INFO=<ID=R2,Number=1,Type=Float,Description="Imputation Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">
##contig=<ID=chr19,length=58617616>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	{samples}"""
    
    # NAFLD-related SNPs on chromosome 19
    variants = [
        ("chr19", 1050000, "rs123456", "A", "G", "AF=0.35;MAF=0.35;R2=0.98"),
        ("chr19", 1050500, "rs234567", "C", "T", "AF=0.42;MAF=0.42;R2=0.97"),
        ("chr19", 1051000, "rs345678", "G", "A", "AF=0.28;MAF=0.28;R2=0.99"),
        ("chr19", 1051500, "rs456789", "T", "C", "AF=0.31;MAF=0.31;R2=0.96"),
        ("chr19", 1052000, "rs567890", "A", "G", "AF=0.25;MAF=0.25;R2=0.98"),
        ("chr19", 1052500, "rs678901", "C", "T", "AF=0.38;MAF=0.38;R2=0.97"),
        ("chr19", 1053000, "rs789012", "G", "A", "AF=0.33;MAF=0.33;R2=0.99"),
        ("chr19", 1053500, "rs890123", "T", "C", "AF=0.29;MAF=0.29;R2=0.98"),
        ("chr19", 1054000, "rs901234", "A", "G", "AF=0.36;MAF=0.36;R2=0.97"),
        ("chr19", 1054500, "rs012345", "C", "T", "AF=0.27;MAF=0.27;R2=0.98")
    ]
    
    # Generate realistic genotype data
    np.random.seed(42)  # For reproducible results
    
    with open(file_path, 'w') as f:
        # Write header
        f.write(vcf_header.format(samples="\t".join(samples)) + "\n")
        
        # Write variants
        for chrom, pos, variant_id, ref, alt, info in variants:
            # Generate genotype data for each sample
            genotypes = []
            for i in range(len(samples)):
                # Create realistic genotype distribution
                if i % 3 == 0:
                    gt = "0|0"
                    ds = np.random.uniform(0.01, 0.05)
                elif i % 3 == 1:
                    gt = "0|1" 
                    ds = np.random.uniform(0.43, 0.53)
                else:
                    gt = "1|1"
                    ds = np.random.uniform(0.96, 0.99)
                
                genotypes.append(f"{gt}:{ds:.2f}")
            
            line = f"{chrom}\t{pos}\t{variant_id}\t{ref}\t{alt}\t100\tPASS\t{info}\tGT:DS\t" + "\t".join(genotypes)
            f.write(line + "\n")

def create_covariates_file(file_path, samples):
    """Create covariates file with NAFLD-relevant covariates"""
    
    np.random.seed(42)
    
    data = {
        'ID': ['age', 'sex', 'bmi', 'batch', 'PC1', 'PC2', 'PC3', 'insulin_resistance', 'liver_fat_content', 'fibrosis_stage'],
        # Age: 35-65 years
        'age': np.random.randint(35, 66, len(samples)),
        # Sex: 1=male, 2=female
        'sex': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        # BMI: 25-35 kg/m²
        'bmi': np.random.uniform(25, 35, len(samples)),
        # Batch effect
        'batch': [1, 1, 2, 2, 1, 2, 1, 2, 1, 2],
        # Principal components
        'PC1': np.random.uniform(-0.03, 0.03, len(samples)),
        'PC2': np.random.uniform(-0.03, 0.03, len(samples)),
        'PC3': np.random.uniform(-0.03, 0.03, len(samples)),
        # NAFLD-specific covariates
        'insulin_resistance': np.random.uniform(1.0, 4.5, len(samples)),
        'liver_fat_content': np.random.uniform(8.0, 23.0, len(samples)),
        'fibrosis_stage': np.random.randint(0, 4, len(samples))
    }
    
    df = pd.DataFrame(data)
    df.set_index('ID', inplace=True)
    df.columns = samples
    
    # Format to reasonable precision
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].round(2)
    
    df.to_csv(file_path, sep='\t', float_format='%.2f')

def create_annotations_file(file_path):
    """Create annotations BED file with NAFLD-related genes"""
    
    # NAFLD-related genes on chromosome 19
    genes = [
        ("chr19", 1045000, 1055000, "ENSG00000130032", "+", "PNPLA3", "protein_coding"),
        ("chr19", 1058000, 1068000, "ENSG00000130024", "-", "TM6SF2", "protein_coding"),
        ("chr19", 1072000, 1082000, "ENSG00000130018", "+", "MBOAT7", "protein_coding"),
        ("chr19", 1086000, 1096000, "ENSG00000130011", "-", "HSD17B13", "protein_coding"),
        ("chr19", 1100000, 1110000, "ENSG00000130005", "+", "GCKR", "protein_coding"),
        ("chr19", 1114000, 1124000, "ENSG00000129998", "-", "LYPLAL1", "protein_coding"),
        ("chr19", 1128000, 1138000, "ENSG00000129991", "+", "PPP1R3B", "protein_coding"),
        ("chr19", 1142000, 1152000, "ENSG00000129984", "-", "FDFT1", "protein_coding"),
        ("chr19", 1156000, 1166000, "ENSG00000129977", "+", "SAMM50", "protein_coding"),
        ("chr19", 1170000, 1180000, "ENSG00000129970", "-", "PARVB", "protein_coding")
    ]
    
    with open(file_path, 'w') as f:
        f.write("#chr\tstart\tend\tgene_id\tstrand\tgene_name\tgene_type\n")
        for gene in genes:
            f.write("\t".join(gene) + "\n")

def create_expression_file(file_path, samples):
    """Create expression data for NAFLD-related genes"""
    
    np.random.seed(42)
    
    gene_ids = [f"ENSG000001300{i:02d}" for i in range(32, 22, -1)]
    gene_names = ["PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", "PPP1R3B", "FDFT1", "SAMM50", "PARVB"]
    
    data = {}
    for i, gene_id in enumerate(gene_ids):
        # Create expression values with some structure
        base_level = np.random.uniform(5, 15)
        variation = np.random.uniform(2, 4, len(samples))
        data[gene_id] = base_level + variation
    
    df = pd.DataFrame(data, index=samples).T
    df.index.name = 'gene_id'
    
    # Add some correlation with genotypes for realistic eQTLs
    df = df.round(2)
    df.to_csv(file_path, sep='\t')

def create_protein_file(file_path, samples):
    """Create protein abundance data"""
    
    np.random.seed(42)
    
    protein_ids = ["PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", "PPP1R3B", "FDFT1", "SAMM50", "PARVB"]
    
    data = {}
    for protein_id in protein_ids:
        base_level = np.random.uniform(35, 55)
        variation = np.random.uniform(3, 8, len(samples))
        data[protein_id] = base_level + variation
    
    df = pd.DataFrame(data, index=samples).T
    df.index.name = 'protein_id'
    df = df.round(2)
    df.to_csv(file_path, sep='\t')

def create_splicing_file(file_path, samples):
    """Create splicing data (PSI values)"""
    
    np.random.seed(42)
    
    event_ids = [f"{gene}_AS1" for gene in ["PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", "PPP1R3B", "FDFT1", "SAMM50", "PARVB"]]
    
    data = {}
    for event_id in event_ids:
        base_level = np.random.uniform(0.6, 0.9)
        variation = np.random.uniform(0.05, 0.15, len(samples))
        data[event_id] = base_level + variation
    
    df = pd.DataFrame(data, index=samples).T
    df.index.name = 'event_id'
    df = df.round(2)
    df.to_csv(file_path, sep='\t')

def create_gwas_phenotype_file(file_path, samples):
    """Create GWAS phenotype data for NAFLD traits"""
    
    np.random.seed(42)
    
    data = {
        'sample_id': samples,
        'liver_fat': np.random.uniform(8, 23, len(samples)),
        'alt_levels': np.random.uniform(30, 90, len(samples)),
        'ast_levels': np.random.uniform(25, 70, len(samples)),
        'fibrosis_score': np.random.randint(0, 4, len(samples)),
        'nafld_activity_score': np.random.randint(2, 8, len(samples))
    }
    
    df = pd.DataFrame(data)
    df = df.round(1)
    df.to_csv(file_path, sep='\t', index=False)

if __name__ == "__main__":
    create_sample_data()