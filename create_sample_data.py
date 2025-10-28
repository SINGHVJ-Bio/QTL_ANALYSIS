#!/usr/bin/env python3
"""
Script to create realistic NAFLD test data for QTL pipeline testing
Enhanced version with stronger associations for meaningful results
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create all sample data files for testing with realistic associations"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("🧬 Creating realistic NAFLD sample test data...")
    
    # Increased sample size for better power
    samples = [f"NAFLD_{i:03d}" for i in range(1, 101)]  # 100 samples
    
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
    """Create sample VCF file with NAFLD-related variants and realistic MAF"""
    
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
    
    # Create more variants (50 variants) around NAFLD genes
    variants = []
    base_pos = 1040000
    for i in range(50):
        chrom = "chr19"
        pos = base_pos + i * 2000  # Spread variants every 2kb
        variant_id = f"rs{i+100000}"
        ref = np.random.choice(["A", "C", "G", "T"])
        alt = np.random.choice([base for base in ["A", "C", "G", "T"] if base != ref])
        maf = np.random.uniform(0.1, 0.4)  # Realistic MAF
        af = max(maf, 1-maf)
        info = f"AF={af:.3f};MAF={maf:.3f};R2=0.98"
        variants.append((chrom, pos, variant_id, ref, alt, info, maf))
    
    # Generate realistic genotype data
    np.random.seed(42)
    
    with open(file_path, 'w') as f:
        # Write header
        f.write(vcf_header.format(samples="\t".join(samples)) + "\n")
        
        # Write variants
        for chrom, pos, variant_id, ref, alt, info, maf in variants:
            genotypes = []
            dosages = []
            
            # Generate genotypes based on MAF
            for i in range(len(samples)):
                rand_val = np.random.random()
                if rand_val < maf**2:  # Homozygous minor
                    gt = "1|1"
                    ds = np.random.uniform(1.8, 2.0)
                elif rand_val < maf**2 + 2*maf*(1-maf):  # Heterozygous
                    gt = "0|1"
                    ds = np.random.uniform(0.8, 1.2)
                else:  # Homozygous major
                    gt = "0|0"
                    ds = np.random.uniform(0.0, 0.2)
                
                genotypes.append(f"{gt}:{ds:.3f}")
            
            line = f"{chrom}\t{pos}\t{variant_id}\t{ref}\t{alt}\t100\tPASS\t{info}\tGT:DS\t" + "\t".join(genotypes)
            f.write(line + "\n")
    
    print(f"✅ Created genotypes.vcf with {len(variants)} variants")

def create_covariates_file(file_path, samples):
    """Create covariates file with NAFLD-relevant covariates"""
    
    np.random.seed(42)
    
    n_samples = len(samples)
    
    # Create data with consistent array lengths
    covariate_data = {
        'age': np.random.randint(35, 66, n_samples),
        'sex': [1 if i % 2 == 0 else 2 for i in range(n_samples)],
        'bmi': np.random.uniform(25, 35, n_samples),
        'batch': [1 if i < n_samples/2 else 2 for i in range(n_samples)],
        'PC1': np.random.normal(0, 0.02, n_samples),
        'PC2': np.random.normal(0, 0.02, n_samples),
        'PC3': np.random.normal(0, 0.02, n_samples),
        'insulin_resistance': np.random.normal(2.5, 0.8, n_samples),
        'liver_fat_content': np.random.normal(15.0, 4.0, n_samples),
        'fibrosis_stage': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    }
    
    # Create DataFrame with samples as columns and covariates as rows
    covariate_names = ['age', 'sex', 'bmi', 'batch', 'PC1', 'PC2', 'PC3', 
                      'insulin_resistance', 'liver_fat_content', 'fibrosis_stage']
    
    # Build the data in the correct orientation
    data_dict = {}
    for covariate in covariate_names:
        data_dict[covariate] = covariate_data[covariate]
    
    # Create DataFrame with covariates as rows and samples as columns
    df = pd.DataFrame(data_dict, index=samples).T
    
    # Format to reasonable precision
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].round(3)
    
    df.to_csv(file_path, sep='\t', float_format='%.3f')
    print(f"✅ Created covariates.txt with {len(df)} covariates")

def create_annotations_file(file_path):
    """Create annotations BED file with NAFLD-related genes"""
    
    # NAFLD-related genes on chromosome 19 - expanded list
    genes = []
    base_start = 1040000
    
    gene_names = [
        "PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", 
        "PPP1R3B", "FDFT1", "SAMM50", "PARVB", "SREBF1", "MTOR", "INSIG1",
        "SCAP", "FASN", "ACACA", "SCD", "DGAT2", "AGPAT2", "LPIN1",
        "MLXIPL", "NR1H3", "NR1H2", "PPARA", "PPARD", "PPARG", "RXRA",
        "RXRB", "RXRG", "SREBF2"
    ]
    
    for i, gene_name in enumerate(gene_names):
        start = base_start + i * 50000  # 50kb spacing
        end = start + 10000  # 10kb gene length
        gene_id = f"ENSG000001{30000 + i:05d}"
        strand = "+" if i % 2 == 0 else "-"
        genes.append(("chr19", start, end, gene_id, strand, gene_name, "protein_coding"))
    
    with open(file_path, 'w') as f:
        f.write("chr\tstart\tend\tgene_id\tstrand\tgene_name\tgene_type\n")
        for gene in genes:
            gene_str = [str(element) for element in gene]
            f.write("\t".join(gene_str) + "\n")
    
    print(f"✅ Created annotations.bed with {len(genes)} genes")

def create_expression_file(file_path, samples):
    """Create expression data with strong genetic associations"""
    
    np.random.seed(42)
    
    # Create gene list matching annotations
    gene_ids = [f"ENSG000001{30000 + i:05d}" for i in range(30)]
    gene_names = [
        "PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", 
        "PPP1R3B", "FDFT1", "SAMM50", "PARVB", "SREBF1", "MTOR", "INSIG1",
        "SCAP", "FASN", "ACACA", "SCD", "DGAT2", "AGPAT2", "LPIN1",
        "MLXIPL", "NR1H3", "NR1H2", "PPARA", "PPARD", "PPARG", "RXRA",
        "RXRB", "RXRG", "SREBF2"
    ]
    
    data = {}
    
    # Read genotype data to create realistic associations
    genotype_patterns = {}
    try:
        with open("data/genotypes.vcf", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                variant_id = fields[2]
                # Get genotype data for this variant
                genotypes = fields[9:]
                dosage_values = []
                for gt in genotypes:
                    dosage = float(gt.split(':')[1])
                    dosage_values.append(dosage)
                genotype_patterns[variant_id] = dosage_values
    except FileNotFoundError:
        print("⚠️  genotypes.vcf not found, creating expression without genetic effects")
        # Create random genotype patterns as fallback
        for i in range(10):
            variant_id = f"rs{i+100000}"
            dosage_values = np.random.choice([0, 1, 2], len(samples), p=[0.25, 0.5, 0.25])
            genotype_patterns[variant_id] = dosage_values
    
    for i, (gene_id, gene_name) in enumerate(zip(gene_ids, gene_names)):
        # Base expression level
        base_level = np.random.uniform(8, 12)
        
        # Add biological noise
        biological_noise = np.random.normal(0, 1, len(samples))
        
        # Create genetic effects for some genes
        genetic_effect = np.zeros(len(samples))
        
        # Strong genetic effect for first 10 genes
        if i < 10 and genotype_patterns:
            # Use nearby variants (every 5th variant)
            variant_idx = min(i * 5, len(genotype_patterns) - 1)
            variant_key = list(genotype_patterns.keys())[variant_idx]
            dosage = genotype_patterns[variant_key]
            
            # Create strong linear effect
            effect_size = np.random.uniform(2.0, 4.0)  # Strong effect
            genetic_effect = np.array(dosage) * effect_size
            
            # Add some noise to the genetic effect
            genetic_effect += np.random.normal(0, 0.5, len(samples))
        
        # Combine components
        expr_values = base_level + biological_noise + genetic_effect
        
        # Ensure positive values
        expr_values = np.maximum(expr_values, 1.0)
        
        data[gene_id] = expr_values
    
    df = pd.DataFrame(data, index=samples).T
    df.index.name = 'gene_id'
    df = df.round(3)
    df.to_csv(file_path, sep='\t')
    print(f"✅ Created expression.txt with {len(df)} genes and strong genetic effects")

def create_protein_file(file_path, samples):
    """Create protein abundance data with associations"""
    
    np.random.seed(43)  # Different seed from expression
    
    protein_ids = [
        "PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", 
        "PPP1R3B", "FDFT1", "SAMM50", "PARVB", "SREBF1", "MTOR", "INSIG1",
        "SCAP", "FASN", "ACACA", "SCD", "DGAT2", "AGPAT2", "LPIN1"
    ]
    
    # Read genotype data
    genotype_patterns = {}
    try:
        with open("data/genotypes.vcf", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                variant_id = fields[2]
                genotypes = fields[9:]
                dosage_values = []
                for gt in genotypes:
                    dosage = float(gt.split(':')[1])
                    dosage_values.append(dosage)
                genotype_patterns[variant_id] = dosage_values
    except FileNotFoundError:
        print("⚠️  genotypes.vcf not found, creating protein data without genetic effects")
        for i in range(10):
            variant_id = f"rs{i+100000}"
            dosage_values = np.random.choice([0, 1, 2], len(samples), p=[0.25, 0.5, 0.25])
            genotype_patterns[variant_id] = dosage_values
    
    data = {}
    for i, protein_id in enumerate(protein_ids):
        base_level = np.random.uniform(40, 60)
        technical_noise = np.random.normal(0, 3, len(samples))
        
        # Genetic effects for some proteins
        genetic_effect = np.zeros(len(samples))
        if i < 8 and genotype_patterns:  # Genetic effects for first 8 proteins
            variant_idx = min(i * 3, len(genotype_patterns) - 1)
            variant_key = list(genotype_patterns.keys())[variant_idx]
            dosage = genotype_patterns[variant_key]
            
            effect_size = np.random.uniform(5.0, 8.0)  # Protein effects can be stronger
            genetic_effect = np.array(dosage) * effect_size
        
        protein_values = base_level + technical_noise + genetic_effect
        protein_values = np.maximum(protein_values, 10.0)  # Ensure positive
        
        data[protein_id] = protein_values
    
    df = pd.DataFrame(data, index=samples).T
    df.index.name = 'protein_id'
    df = df.round(2)
    df.to_csv(file_path, sep='\t')
    print(f"✅ Created protein.txt with {len(df)} proteins")

def create_splicing_file(file_path, samples):
    """Create splicing data (PSI values) with genetic effects"""
    
    np.random.seed(44)  # Different seed
    
    event_ids = []
    for gene in ["PNPLA3", "TM6SF2", "MBOAT7", "HSD17B13", "GCKR", "LYPLAL1", 
                "PPP1R3B", "FDFT1", "SAMM50", "PARVB", "SREBF1", "MTOR"]:
        for event_type in ["SE", "A5SS", "A3SS"]:
            event_ids.append(f"{gene}_{event_type}_1")
    
    # Read genotype data
    genotype_patterns = {}
    try:
        with open("data/genotypes.vcf", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                variant_id = fields[2]
                genotypes = fields[9:]
                dosage_values = []
                for gt in genotypes:
                    dosage = float(gt.split(':')[1])
                    dosage_values.append(dosage)
                genotype_patterns[variant_id] = dosage_values
    except FileNotFoundError:
        print("⚠️  genotypes.vcf not found, creating splicing data without genetic effects")
        for i in range(10):
            variant_id = f"rs{i+100000}"
            dosage_values = np.random.choice([0, 1, 2], len(samples), p=[0.25, 0.5, 0.25])
            genotype_patterns[variant_id] = dosage_values
    
    data = {}
    for i, event_id in enumerate(event_ids):
        base_level = np.random.uniform(0.4, 0.8)
        noise = np.random.normal(0, 0.1, len(samples))
        
        # Genetic effects for some splicing events
        genetic_effect = np.zeros(len(samples))
        if i < 15 and genotype_patterns:  # Genetic effects for first 15 events
            variant_idx = min(i * 2, len(genotype_patterns) - 1)
            variant_key = list(genotype_patterns.keys())[variant_idx]
            dosage = genotype_patterns[variant_key]
            
            effect_size = np.random.uniform(0.1, 0.2)
            genetic_effect = np.array(dosage) * effect_size
        
        psi_values = base_level + noise + genetic_effect
        # Clamp to [0, 1] range
        psi_values = np.clip(psi_values, 0.1, 0.9)
        
        data[event_id] = psi_values
    
    df = pd.DataFrame(data, index=samples).T
    df.index.name = 'event_id'
    df = df.round(3)
    df.to_csv(file_path, sep='\t')
    print(f"✅ Created splicing.txt with {len(df)} splicing events")

def create_gwas_phenotype_file(file_path, samples):
    """Create GWAS phenotype data for NAFLD traits with genetic components"""
    
    np.random.seed(45)
    
    n_samples = len(samples)
    
    # Create phenotypes with genetic components
    liver_fat = np.random.normal(15.0, 4.0, n_samples)
    alt_levels = np.random.normal(60.0, 15.0, n_samples)
    
    # Try to read genotype data for genetic effects
    try:
        with open("data/genotypes.vcf", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                variant_id = fields[2]
                genotypes = fields[9:]
                dosage_values = []
                for gt in genotypes:
                    dosage = float(gt.split(':')[1])
                    dosage_values.append(dosage)
                # Add genetic effects to liver fat using first variant
                liver_fat += np.array(dosage_values) * 3.0  # Genetic effect
                break  # Only use first variant
    except FileNotFoundError:
        print("⚠️  genotypes.vcf not found, creating GWAS data without genetic effects")
    
    data = {
        'sample_id': samples,
        'liver_fat': liver_fat,
        'alt_levels': alt_levels,
        'ast_levels': np.random.normal(45.0, 12.0, n_samples),
        'fibrosis_score': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'nafld_activity_score': np.random.choice([2, 3, 4, 5, 6, 7], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"✅ Created gwas_phenotype.txt with {len(df)} samples")

if __name__ == "__main__":
    create_sample_data()