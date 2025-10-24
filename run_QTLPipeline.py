#!/usr/bin/env python3
"""
QTL Analysis Pipeline - Main Runner Script - Enhanced Version
Complete pipeline for cis/trans QTL analysis with comprehensive reporting
With proper normalization strategies per QTL type
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup basic logging for the runner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    """Main runner function"""
    parser = argparse.ArgumentParser(
        description='Enhanced QTL Analysis Pipeline Runner - Complete cis/trans QTL analysis with advanced features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default config
  python run_QTLPipeline.py --config config/config.yaml

  # Run specific QTL types only
  python run_QTLPipeline.py --config config/config.yaml --qtl-types eqtl,pqtl

  # Run only cis-QTL analysis
  python run_QTLPipeline.py --config config/config.yaml --qtl-mode cis

  # Run only trans-QTL analysis  
  python run_QTLPipeline.py --config config/config.yaml --qtl-mode trans

  # Run both cis and trans
  python run_QTLPipeline.py --config config/config.yaml --qtl-mode both

  # Enable enhanced features
  python run_QTLPipeline.py --config config/config.yaml --enhanced-qc --interaction-analysis --fine-mapping

  # Validate inputs only
  python run_QTLPipeline.py --config config/config.yaml --validate-only

  # Run with custom output directory
  python run_QTLPipeline.py --config config/config.yaml --output-dir my_results

  # Run with performance tuning
  python run_QTLPipeline.py --config config/config.yaml --threads 8 --memory 16

  # Override normalization methods
  python run_QTLPipeline.py --config config/config.yaml --eqtl-norm vst --pqtl-norm log2 --sqtl-norm log2
        """
    )
    parser.add_argument('--config', required=True, 
                       help='Path to configuration file (YAML format)')
    parser.add_argument('--qtl-types', 
                       help='Override QTL types from config (comma-separated: eqtl,pqtl,sqtl)')
    parser.add_argument('--qtl-mode', choices=['cis', 'trans', 'both'],
                       help='Override QTL mapping mode from config')
    parser.add_argument('--output-dir',
                       help='Override results directory from config')
    parser.add_argument('--run-gwas', action='store_true', 
                       help='Enable GWAS analysis (overrides config setting)')
    parser.add_argument('--enhanced-qc', action='store_true',
                       help='Enable enhanced QC (overrides config setting)')
    parser.add_argument('--interaction-analysis', action='store_true',
                       help='Enable interaction analysis (overrides config setting)')
    parser.add_argument('--fine-mapping', action='store_true',
                       help='Enable fine-mapping (overrides config setting)')
    parser.add_argument('--threads', type=int,
                       help='Number of threads for parallel processing')
    parser.add_argument('--memory', type=int,
                       help='Memory allocation in GB')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate inputs, do not run analysis')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    
    # Normalization override arguments
    parser.add_argument('--eqtl-norm', choices=['vst', 'log2', 'quantile', 'tpm', 'raw'],
                       help='Override eQTL normalization method')
    parser.add_argument('--pqtl-norm', choices=['log2', 'quantile', 'zscore', 'raw'],
                       help='Override pQTL normalization method')
    parser.add_argument('--sqtl-norm', choices=['log2', 'arcsinh', 'zscore', 'raw'],
                       help='Override sQTL normalization method')
    
    # Command line arguments for multi-core optimization
    parser.add_argument('--optimize-cpu', action='store_true',
                    help='Enable CPU optimization mode')
    parser.add_argument('--chromosome-parallel', action='store_true',
                    help='Process chromosomes in parallel')
    parser.add_argument('--monitor-performance', action='store_true', default=True,
                    help='Monitor CPU and memory usage during analysis')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Add scripts directory to Python path
    script_dir = Path(__file__).parent / "scripts"
    sys.path.insert(0, str(script_dir))
    
    try:
        # Import after adding scripts to path
        from scripts.main import QTLPipeline
        
        # Initialize pipeline
        pipeline = QTLPipeline(args.config)
        
        # Override settings from command line
        if args.qtl_types:
            analysis_types = [atype.strip() for atype in args.qtl_types.split(',')]
            pipeline.config['analysis']['qtl_types'] = analysis_types

        if args.qtl_mode:
            pipeline.config['analysis']['qtl_mode'] = args.qtl_mode
            
        if args.output_dir:
            pipeline.config['results_dir'] = args.output_dir
            # Recreate directories with new path
            pipeline.setup_directories()
            
        if args.run_gwas:
            pipeline.config['analysis']['run_gwas'] = True
            
        if args.enhanced_qc:
            pipeline.config['enhanced_qc'] = {'enable': True}
            
        if args.interaction_analysis:
            pipeline.config['interaction_analysis'] = {'enable': True}
            
        if args.fine_mapping:
            pipeline.config['fine_mapping'] = {'enable': True}
            
        if args.threads:
            pipeline.config['performance']['num_threads'] = args.threads
            # Also update related thread settings for optimal performance
            pipeline.config['performance']['max_workers'] = min(args.threads, 16)  # Cap at 16 to avoid overhead
            pipeline.config['genotype_processing']['plink_threads'] = args.threads
            pipeline.config['genotype_processing']['bcftools_threads'] = args.threads
            print(f"✅ Setting CPU threads to {args.threads} for all operations")
            
        if args.memory:
            pipeline.config['performance']['memory_gb'] = args.memory
            # Adjust batch sizes based on available memory
            if args.memory >= 64:
                pipeline.config['performance']['tensorqtl_batch_size'] = 30000
                pipeline.config['performance']['tensorqtl_chunk_size'] = 300
            elif args.memory >= 32:
                pipeline.config['performance']['tensorqtl_batch_size'] = 20000
                pipeline.config['performance']['tensorqtl_chunk_size'] = 200
            print(f"✅ Setting memory to {args.memory}GB with optimized batch sizes")

        # New multi-core optimization arguments
        if hasattr(args, 'optimize_cpu') and args.optimize_cpu:
            pipeline.config['tensorqtl']['use_gpu'] = False
            pipeline.config['performance']['process_by_chromosome'] = True
            pipeline.config['large_data']['force_plink'] = True
            pipeline.config['performance']['parallel_processing'] = True
            print("✅ Enabled CPU optimization mode")

        if hasattr(args, 'chromosome_parallel') and args.chromosome_parallel:
            pipeline.config['performance']['process_by_chromosome'] = True
            max_chromosomes = min(8, pipeline.config['performance']['num_threads'] // 2)
            pipeline.config['performance']['max_concurrent_chromosomes'] = max(1, max_chromosomes)
            pipeline.config['large_data']['max_concurrent_chromosomes'] = max(1, max_chromosomes)
            print(f"✅ Enabled chromosome-level parallel processing ({max_chromosomes} concurrent chromosomes)")

        # Override normalization methods if specified
        if args.eqtl_norm:
            pipeline.config['normalization']['eqtl']['method'] = args.eqtl_norm
            print(f"✅ Overriding eQTL normalization: {args.eqtl_norm}")
            
        if args.pqtl_norm:
            pipeline.config['normalization']['pqtl']['method'] = args.pqtl_norm
            print(f"✅ Overriding pQTL normalization: {args.pqtl_norm}")
            
        if args.sqtl_norm:
            pipeline.config['normalization']['sqtl']['method'] = args.sqtl_norm
            print(f"✅ Overriding sQTL normalization: {args.sqtl_norm}")
            
        if args.validate_only:
            from scripts.utils.validation import validate_inputs
            print("🔍 Running comprehensive input validation...")
            validate_inputs(pipeline.config)
            print("✅ All inputs validated successfully!")
            return

        # Print system configuration
        print("\n🔧 System Configuration:")
        print("=" * 50)
        print(f"   CPU Threads:       {pipeline.config['performance']['num_threads']}")
        print(f"   Memory Allocation: {pipeline.config['performance']['memory_gb']}GB")
        print(f"   Chromosome Parallel: {pipeline.config['performance'].get('process_by_chromosome', False)}")
        print(f"   Max Workers:       {pipeline.config['performance'].get('max_workers', 4)}")
        print(f"   TensorQTL Batch:   {pipeline.config['performance'].get('tensorqtl_batch_size', 10000)}")

        # Print normalization settings
        print("\n🔧 Normalization Configuration:")
        print("=" * 40)
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            if qtl_type in pipeline.config['normalization']:
                method = pipeline.config['normalization'][qtl_type]['method']
                print(f"   {qtl_type.upper():<6}: {method}")
        print("=" * 40)
        
        # Run the complete pipeline
        print("🚀 Starting Enhanced QTL Analysis Pipeline...")
        print("=" * 60)
        print("🔧 Features Enabled:")
        if pipeline.config.get('enhanced_qc', {}).get('enable', False):
            print("   ✅ Enhanced Quality Control")
        if pipeline.config.get('interaction_analysis', {}).get('enable', False):
            print("   ✅ Interaction Analysis")
        if pipeline.config.get('fine_mapping', {}).get('enable', False):
            print("   ✅ Fine-mapping")
        if pipeline.config['analysis'].get('run_gwas', False):
            print("   ✅ GWAS Analysis")
        print("=" * 60)
        
        # Start performance monitoring
        performance_monitor = None
        if not args.validate_only:
            try:
                from scripts.utils.performance_monitor import PerformanceMonitor
                performance_monitor = PerformanceMonitor(
                    interval=60, 
                    log_file=os.path.join(pipeline.logs_dir, "performance_metrics.csv")
                )
                monitor_data = performance_monitor.start()
                print("🔍 Started performance monitoring")
            except ImportError as e:
                print(f"⚠️  Could not start performance monitoring: {e}")

        # Run the complete pipeline
        print("🚀 Starting Enhanced QTL Analysis Pipeline...")
        results = pipeline.run_pipeline()

        # Stop performance monitoring
        if performance_monitor:
            performance_monitor.stop()
            print("📈 Performance monitoring completed")
            
        # Print comprehensive success summary
        print("\n" + "=" * 80)
        print("🎉 ENHANCED QTL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Results Directory: {pipeline.results_dir}")
        print(f"📊 Analysis Mode:     {pipeline.config['analysis'].get('qtl_mode', 'cis')}")
        print(f"⏱️  Total Runtime:     {datetime.now() - pipeline.start_time}")
        
        # Print normalization summary
        print("\n🔧 Normalization Methods Used:")
        print("-" * 40)
        for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
            if qtl_type in pipeline.config['normalization']:
                method = pipeline.config['normalization'][qtl_type]['method']
                print(f"   {qtl_type.upper():<6}: {method}")
        
        # Print detailed analysis summary
        print("\n" + "📈 ANALYSIS SUMMARY")
        print("-" * 80)
        
        total_significant = 0
        if 'qtl' in results:
            print("QTL Results:")
            for qtl_type, result in results['qtl'].items():
                if 'cis' in result:
                    cis_status = "✅" if result['cis']['status'] == 'completed' else "❌"
                    cis_count = result['cis'].get('significant_count', 0) if result['cis']['status'] == 'completed' else 'N/A'
                    if isinstance(cis_count, int):
                        total_significant += cis_count
                    print(f"  {qtl_type.upper():<8} CIS:  {cis_status} Significant: {cis_count}")
                
                if 'trans' in result:
                    trans_status = "✅" if result['trans']['status'] == 'completed' else "❌"  
                    trans_count = result['trans'].get('significant_count', 0) if result['trans']['status'] == 'completed' else 'N/A'
                    if isinstance(trans_count, int):
                        total_significant += trans_count
                    print(f"  {qtl_type.upper():<8} TRANS: {trans_status} Significant: {trans_count}")
                
        if 'gwas' in results:
            gwas_result = results['gwas']
            status = "✅ COMPLETED" if gwas_result['status'] == 'completed' else "❌ FAILED"
            count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
            method = gwas_result.get('method', 'N/A')
            if isinstance(count, int):
                total_significant += count
            print(f"GWAS Analysis: {status} (Method: {method}, Significant: {count})")
            
        # Advanced analyses summary
        if 'advanced' in results:
            print("\nAdvanced Analyses:")
            for analysis_type, result in results['advanced'].items():
                if 'interaction' in analysis_type:
                    print(f"  🤝 Interaction Analysis: {analysis_type}")
                elif 'fine_mapping' in analysis_type:
                    print(f"  🎯 Fine-mapping: {analysis_type}")
        
        print(f"\n🏆 TOTAL SIGNIFICANT ASSOCIATIONS: {total_significant}")
        
        print("\n" + "📋 OUTPUT FILES")
        print("-" * 80)
        print(f"📄 HTML Report:      {pipeline.reports_dir}/analysis_report.html")
        print(f"📊 Plots Directory:  {pipeline.plots_dir}/")
        print(f"📈 Results:          {pipeline.qtl_results_dir}/")
        print(f"🔍 QC Reports:       {pipeline.qc_reports_dir}/")
        print(f"🤝 Interaction:      {pipeline.interaction_results_dir}/")
        print(f"🎯 Fine-mapping:     {pipeline.fine_mapping_results_dir}/")
        print(f"📝 Logs:             {pipeline.logs_dir}/")
        print(f"📋 Summary:          {pipeline.results_dir}/pipeline_summary.txt")
        
        print("\n" + "💡 NEXT STEPS")
        print("-" * 80)
        print("1. Review the HTML report for comprehensive results")
        print("2. Check generated plots in the plots directory") 
        print("3. Examine detailed results in respective directories")
        print("4. Review QC reports for data quality assessment")
        print("5. Explore advanced analyses results if enabled")
        print("6. Check logs for any warnings or additional information")
        print("=" * 80)
        
    except ImportError as e:
        logging.error(f"❌ Import error - make sure all dependencies are installed: {e}")
        print("\n💡 Install required packages: pip install -r requirements.txt")
        print("   Additional packages for enhanced features:")
        print("   - scikit-learn: for PCA analysis")
        print("   - plotly: for interactive plots")
        print("   - statsmodels: for statistical models")
        print("   - DESeq2 (R package): for VST normalization")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()