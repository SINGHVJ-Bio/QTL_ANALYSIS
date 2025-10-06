#!/usr/bin/env python3
"""
QTL Analysis Pipeline - Main Runner Script
Located in root directory for easy access
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
        description='QTL Analysis Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default config
  python run_QTLPipeline.py --config config/config.yaml

  # Run specific QTL types only
  python run_QTLPipeline.py --config config/config.yaml --analysis-types eqtl,pqtl

  # Run with GWAS analysis enabled
  python run_QTLPipeline.py --config config/config.yaml --run-gwas

  # Run only eQTL analysis
  python run_QTLPipeline.py --config config/config.yaml --analysis-types eqtl

  # Validate inputs only without running analysis
  python run_QTLPipeline.py --config config/config.yaml --validate-only
        """
    )
    parser.add_argument('--config', required=True, 
                       help='Path to configuration file (YAML format)')
    parser.add_argument('--analysis-types', 
                       help='Override analysis types from config (comma-separated: eqtl,pqtl,sqtl)')
    parser.add_argument('--run-gwas', action='store_true', 
                       help='Enable GWAS analysis (overrides config setting)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate inputs, do not run analysis')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Add scripts directory to Python path
    script_dir = Path(__file__).parent / "scripts"
    sys.path.insert(0, str(script_dir))
    
    try:
        from main import QTLPipeline
        
        # Initialize pipeline
        pipeline = QTLPipeline(args.config)
        
        # Override settings from command line
        if args.analysis_types:
            pipeline.config['analysis']['qtl_types'] = args.analysis_types
        
        if args.run_gwas:
            pipeline.config['analysis']['run_gwas'] = True
            
        if args.validate_only:
            # Only run validation
            print("🔍 Running input validation only...")
            from utils.validation import validate_inputs
            validate_inputs(pipeline.config)
            print("✅ Validation completed successfully!")
            return
        
        # Run the complete pipeline
        results = pipeline.run_pipeline()
        
        # Print success summary
        print("\n" + "="*70)
        print("🎉 QTL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Results Directory: {pipeline.results_dir}")
        print(f"📊 Plots Directory:   {pipeline.plots_dir}")
        print(f"📋 Reports Directory: {pipeline.reports_dir}")
        print(f"📝 Logs Directory:    {pipeline.logs_dir}")
        print(f"⏱️  Total Runtime:    {datetime.now() - pipeline.start_time}")
        
        # Print analysis summary
        print("\n" + "📈 ANALYSIS SUMMARY")
        print("-" * 70)
        
        if 'qtl' in results:
            print("QTL Results:")
            for qtl_type, result in results['qtl'].items():
                status = "✅ COMPLETED" if result['status'] == 'completed' else "❌ FAILED"
                count = result.get('significant_count', 0) if result['status'] == 'completed' else 'N/A'
                print(f"  {qtl_type.upper():<8} {status:<15} Significant: {count}")
                
        if 'gwas' in results:
            gwas_result = results['gwas']
            status = "✅ COMPLETED" if gwas_result['status'] == 'completed' else "❌ FAILED"
            count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
            method = gwas_result.get('method', 'N/A')
            print(f"GWAS Analysis: {status} (Method: {method}, Significant: {count})")
            
        print("\n" + "📋 NEXT STEPS")
        print("-" * 70)
        print(f"1. View the HTML report: {pipeline.reports_dir}/analysis_report.html")
        print(f"2. Check generated plots: {pipeline.plots_dir}/")
        print(f"3. Review detailed logs: {pipeline.logs_dir}/")
        print("="*70)
        
    except Exception as e:
        logging.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()