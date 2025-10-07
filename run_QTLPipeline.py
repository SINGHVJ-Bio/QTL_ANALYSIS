#!/usr/bin/env python3
"""
QTL Analysis Pipeline - Main Runner Script
Complete pipeline for cis/trans QTL analysis with comprehensive reporting
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

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
        description='QTL Analysis Pipeline Runner - Complete cis/trans QTL analysis',
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

  # Validate inputs only
  python run_QTLPipeline.py --config config/config.yaml --validate-only

  # Run with custom output directory
  python run_QTLPipeline.py --config config/config.yaml --output-dir my_results
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
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate inputs, do not run analysis')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    
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
            
        if args.validate_only:
            from scripts.utils.validation import validate_inputs
            print("🔍 Running comprehensive input validation...")
            validate_inputs(pipeline.config)
            print("✅ All inputs validated successfully!")
            return
        
        # Run the complete pipeline
        print("🚀 Starting QTL Analysis Pipeline...")
        results = pipeline.run_pipeline()
        
        # Print comprehensive success summary
        print("\n" + "="*80)
        print("🎉 QTL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Results Directory: {pipeline.results_dir}")
        print(f"📊 Analysis Mode:     {pipeline.config['analysis'].get('qtl_mode', 'cis')}")
        print(f"⏱️  Total Runtime:     {datetime.now() - pipeline.start_time}")
        
        # Print detailed analysis summary
        print("\n" + "📈 ANALYSIS SUMMARY")
        print("-" * 80)
        
        if 'qtl' in results:
            print("QTL Results:")
            for qtl_type, result in results['qtl'].items():
                if 'cis' in result:
                    cis_status = "✅" if result['cis']['status'] == 'completed' else "❌"
                    cis_count = result['cis'].get('significant_count', 0) if result['cis']['status'] == 'completed' else 'N/A'
                    print(f"  {qtl_type.upper():<8} CIS:  {cis_status} Significant: {cis_count}")
                
                if 'trans' in result:
                    trans_status = "✅" if result['trans']['status'] == 'completed' else "❌"  
                    trans_count = result['trans'].get('significant_count', 0) if result['trans']['status'] == 'completed' else 'N/A'
                    print(f"  {qtl_type.upper():<8} TRANS: {trans_status} Significant: {trans_count}")
                
        if 'gwas' in results:
            gwas_result = results['gwas']
            status = "✅ COMPLETED" if gwas_result['status'] == 'completed' else "❌ FAILED"
            count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
            method = gwas_result.get('method', 'N/A')
            print(f"GWAS Analysis: {status} (Method: {method}, Significant: {count})")
            
        print("\n" + "📋 OUTPUT FILES")
        print("-" * 80)
        print(f"📄 HTML Report:      {pipeline.reports_dir}/analysis_report.html")
        print(f"📊 Plots Directory:  {pipeline.plots_dir}/")
        print(f"📈 Results:          {pipeline.qtl_results_dir}/")
        print(f"📝 Logs:             {pipeline.logs_dir}/")
        print(f"📋 Summary:          {pipeline.results_dir}/pipeline_summary.txt")
        
        print("\n" + "💡 NEXT STEPS")
        print("-" * 80)
        print("1. Review the HTML report for comprehensive results")
        print("2. Check generated plots in the plots directory") 
        print("3. Examine detailed results in QTL_results directory")
        print("4. Review logs for any warnings or additional information")
        print("="*80)
        
    except ImportError as e:
        logging.error(f"❌ Import error - make sure all dependencies are installed: {e}")
        print("\n💡 Install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()