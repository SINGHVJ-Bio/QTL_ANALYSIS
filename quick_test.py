#!/usr/bin/env python3
"""
Enhanced QTL Pipeline - Quick Test and Validation Script
Runs a quick validation and test to ensure the pipeline is working correctly

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import yaml

def create_test_data():
    """Create minimal test data for validation using existing structure"""
    test_dir = Path("quick_test_data")
    test_dir.mkdir(exist_ok=True)
    
    print("📁 Creating test data...")
    
    # Create sample data (10 samples, 50 genes, 100 variants)
    samples = [f"sample_{i}" for i in range(1, 11)]
    genes = [f"gene_{i}" for i in range(1, 51)]
    variants = [f"var_{i}" for i in range(1, 101)]
    
    # Create genotype data (mini VCF)
    vcf_content = """##fileformat=VCFv4.2
##source=QTLPipeline_QuickTest
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t""" + "\t".join(samples) + "\n"
    
    for i, var in enumerate(variants):
        chrom = f"chr{(i % 5) + 1}"  # Use chr1-chr5 for quick test
        pos = (i + 1) * 1000
        ref = "A"
        alt = "G"
        # Create random genotypes
        gts = []
        for _ in samples:
            gt_val = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            gts.append(f"{gt_val}")
        
        vcf_content += f"{chrom}\t{pos}\t{var}\t{ref}\t{alt}\t100\tPASS\t.\tGT\t" + "\t".join(gts) + "\n"
    
    with open(test_dir / "test_genotypes.vcf", "w") as f:
        f.write(vcf_content)
    
    # Create expression data
    expr_data = pd.DataFrame(
        np.random.normal(0, 1, (len(genes), len(samples))),
        index=genes,
        columns=samples
    )
    expr_data.to_csv(test_dir / "test_expression.txt", sep="\t")
    
    # Create covariates
    cov_data = pd.DataFrame({
        'covariate': ['age', 'sex', 'PC1'],
        **{sample: [
            np.random.randint(20, 60),  # age
            np.random.randint(1, 3),    # sex
            np.random.normal(0, 1)      # PC1
        ] for sample in samples}
    })
    cov_data.set_index('covariate', inplace=True)
    cov_data.to_csv(test_dir / "test_covariates.txt", sep="\t")
    
    # Create annotations
    annot_data = []
    for i, gene in enumerate(genes):
        chrom = f"chr{(i % 5) + 1}"
        start = (i + 1) * 10000
        end = start + 1000
        strand = "+" if i % 2 == 0 else "-"
        annot_data.append([chrom, start, end, gene, 0, strand])
    
    annot_df = pd.DataFrame(annot_data, columns=['chr', 'start', 'end', 'gene_id', 'score', 'strand'])
    annot_df.to_csv(test_dir / "test_annotations.bed", sep="\t", index=False)
    
    print(f"✅ Created test data in {test_dir}/")
    return test_dir

def create_test_config(test_dir):
    """Create test configuration for quick validation"""
    config = {
        'results_dir': "quick_test_results",
        'input_files': {
            'genotypes': str(test_dir / "test_genotypes.vcf"),
            'covariates': str(test_dir / "test_covariates.txt"),
            'annotations': str(test_dir / "test_annotations.bed"),
            'expression': str(test_dir / "test_expression.txt")
        },
        'genotype_processing': {
            'auto_detect_format': True,
            'filter_variants': True,
            'min_maf': 0.05,
            'min_call_rate': 0.8,
            'normalize_chromosomes': True,
            'handle_multiallelic': True,
            'multiallelic_action': 'split'
        },
        'analysis': {
            'qtl_types': 'eqtl',
            'qtl_mode': 'cis',
            'run_gwas': False
        },
        'qtl': {
            'cis_window': 50000,  # Smaller window for quick test
            'permutations': 10,   # Minimal permutations for speed
            'fdr_threshold': 0.1,
            'maf_threshold': 0.05,
            'min_maf': 0.05,
            'min_call_rate': 0.8
        },
        'enhanced_qc': {
            'enable': False  # Disable for speed
        },
        'interaction_analysis': {
            'enable': False
        },
        'fine_mapping': {
            'enable': False
        },
        'plotting': {
            'enabled': True,
            'format': 'png',
            'dpi': 100,
            'plot_types': ['manhattan', 'qq']
        },
        'output': {
            'generate_report': True,
            'save_plots': True,
            'remove_intermediate': False,
            'compression': False
        },
        'performance': {
            'num_threads': 2,
            'memory_gb': 2,
            'chunk_size': 5,
            'temp_dir': 'temp_quick_test',
            'cleanup_temp': True
        },
        'qc': {
            'check_sample_concordance': True,
            'filter_low_expressed': False,  # Disable for small test data
            'check_maf_distribution': True,
            'check_missingness': True
        },
        'paths': {
            'qtltools': 'qtltools',
            'bcftools': 'bcftools',
            'bgzip': 'bgzip',
            'tabix': 'tabix',
            'plink': 'plink',
            'python': 'python3',
            'R': 'R'
        }
    }
    
    config_file = "quick_test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created test configuration: {config_file}")
    return config_file

def run_quick_test():
    """Run quick test of the pipeline"""
    print("🚀 QTL PIPELINE QUICK TEST")
    print("=" * 60)
    print("Author: Dr. Vijay Singh")
    print("Email: vijay.s.gautam@gmail.com")
    print("=" * 60)
    
    # Step 1: Create test data
    test_dir = create_test_data()
    
    # Step 2: Create test config
    config_file = create_test_config(test_dir)
    
    # Step 3: Run validation
    print("\n🔍 Running input validation...")
    try:
        result = subprocess.run([
            sys.executable, 'run_QTLPipeline.py', 
            '--config', config_file,
            '--validate-only'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Input validation passed!")
        else:
            print("❌ Input validation failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Step 4: Run quick analysis
    print("\n🔬 Running quick QTL analysis...")
    try:
        # Use subprocess to run the pipeline
        process = subprocess.Popen([
            sys.executable, 'run_QTLPipeline.py',
            '--config', config_file,
            '--threads', '2'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        stderr = process.stderr.read()
        if stderr:
            print("STDERR:", stderr)
        
        if process.returncode == 0:
            print("✅ Quick test analysis completed successfully!")
            
            # Check generated files
            print("\n📊 Generated output structure:")
            results_dir = "quick_test_results"
            if os.path.exists(results_dir):
                for root, dirs, files in os.walk(results_dir):
                    level = root.replace(results_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f'{indent}📁 {os.path.basename(root)}/')
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Show first 5 files
                        if not file.startswith('.'):
                            file_size = os.path.getsize(os.path.join(root, file))
                            print(f'{subindent}📄 {file} ({file_size} bytes)')
                    if len(files) > 5:
                        print(f'{subindent}... and {len(files) - 5} more files')
            
            return True
        else:
            print("❌ Quick test analysis failed:")
            print(f"Return code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    files_to_remove = [
        "quick_test_config.yaml",
        "quick_test_data",
        "quick_test_results",
        "temp_quick_test"
    ]
    
    for item in files_to_remove:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"✅ Removed directory: {item}")
            else:
                os.remove(item)
                print(f"✅ Removed file: {item}")

def main():
    """Main quick test function"""
    print("🔬 ENHANCED QTL PIPELINE - QUICK TEST")
    print("=" * 60)
    print("This script will:")
    print("  1. Create minimal test data")
    print("  2. Validate the pipeline setup") 
    print("  3. Run a quick QTL analysis")
    print("  4. Generate test results")
    print("  5. Clean up test files")
    print("=" * 60)
    
    try:
        # Run the quick test
        success = run_quick_test()
        
        # Cleanup
        cleanup_test_files()
        
        # Final message
        print("\n" + "=" * 60)
        if success:
            print("🎉 QUICK TEST COMPLETED SUCCESSFULLY!")
            print("   Your QTL pipeline is working correctly!")
            print("\n💡 Next steps:")
            print("   1. Configure config/config.yaml with your real data paths")
            print("   2. Run: python run_QTLPipeline.py --config config/config.yaml")
            print("   3. Check results in the specified output directory")
            print("   4. Contact: vijay.s.gautam@gmail.com for support")
        else:
            print("❌ QUICK TEST FAILED")
            print("   Please check the error messages above")
            print("   Make sure all dependencies are installed correctly")
            print("   Run: python validate_environment.py to check requirements")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Quick test interrupted by user")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Quick test failed with error: {e}")
        cleanup_test_files()
        sys.exit(1)

if __name__ == "__main__":
    main()