#!/usr/bin/env python3
"""
Modular QTL Pipeline Runner - Enhanced Version
Allows running modules separately, in combination, or all together
with dependency checking and helpful messages.

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with comprehensive logging, dependency tracking, and backward compatibility.
"""

import argparse
import sys
import os
import yaml
import subprocess
from pathlib import Path
import logging
import traceback
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qtl_pipeline.log')
    ]
)
logger = logging.getLogger('ModularQTLPipeline')

class ModularQTLPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
        
        # Define modules with dependencies and descriptions
        self.modules = {
            'data_preparation': {
                'function': self.run_data_preparation,
                'dependencies': [],
                'description': 'Prepare and validate input data',
                'output_files': ['data/processed/input_validation_report.txt']
            },
            'genotype_processing': {
                'function': self.run_genotype_processing,
                'dependencies': ['data_preparation'],
                'description': 'Process genotype data (VCF/PLINK formats)',
                'output_files': ['data/processed/genotypes/final_genotypes.vcf.gz']
            },
            'expression_processing': {
                'function': self.run_expression_processing,
                'dependencies': ['data_preparation'],
                'description': 'Process expression data (normalization, QC)',
                'output_files': ['data/processed/expression/expression_processed.txt']
            },
            'quality_control': {
                'function': self.run_quality_control,
                'dependencies': ['genotype_processing', 'expression_processing'],
                'description': 'Perform comprehensive quality control',
                'output_files': ['data/processed/qc/qc_report.html']
            },
            'qtl_mapping': {
                'function': self.run_qtl_mapping,
                'dependencies': ['quality_control'],
                'description': 'Perform cis/trans QTL mapping analysis',
                'output_files': ['results/qtl/cis_qtl.txt', 'results/qtl/trans_qtl.txt']
            },
            'fine_mapping': {
                'function': self.run_fine_mapping,
                'dependencies': ['qtl_mapping'],
                'description': 'Fine mapping of QTL signals',
                'output_files': ['results/fine_mapping/fine_mapped_variants.txt']
            },
            'interaction_analysis': {
                'function': self.run_interaction_analysis,
                'dependencies': ['qtl_mapping'],
                'description': 'Covariate interaction analysis',
                'output_files': ['results/interaction/interaction_results.txt']
            },
            'visualization': {
                'function': self.run_visualization,
                'dependencies': ['qtl_mapping', 'fine_mapping'],
                'description': 'Generate plots and visualizations',
                'output_files': ['results/plots/manhattan_plot.png', 'results/plots/qq_plot.png']
            },
            'report_generation': {
                'function': self.run_report_generation,
                'dependencies': ['qtl_mapping', 'fine_mapping', 'visualization'],
                'description': 'Generate comprehensive HTML reports',
                'output_files': ['results/reports/final_report.html']
            }
        }
        
        # Track which modules have been run in this session
        self.completed_modules = set()
        
        # Track module execution times
        self.execution_times = {}

    def load_config(self):
        """Load configuration from YAML file with enhanced error handling"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"✅ Configuration loaded from: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {self.config_path}")
            logger.info("💡 Available config files:")
            config_dir = Path("config")
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    logger.info(f"   - {config_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error loading config file: {e}")
            sys.exit(1)

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            'data/processed/genotypes',
            'data/processed/expression', 
            'data/processed/qc',
            'results/qtl',
            'results/fine_mapping',
            'results/interaction',
            'results/plots',
            'results/reports',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Directory structure verified")

    def check_dependencies(self, module_name):
        """Check if all dependencies for a module are satisfied with detailed reporting"""
        dependencies = self.modules[module_name]['dependencies']
        missing_deps = [dep for dep in dependencies if dep not in self.completed_modules]
        
        if missing_deps:
            logger.error(f"❌ Module '{module_name}' requires the following modules to be run first:")
            for dep in missing_deps:
                dep_info = self.modules[dep]
                logger.error(f"   📍 {dep}: {dep_info['description']}")
                
                # Check if output files exist (might have been run in previous session)
                output_exists = self.check_module_outputs(dep)
                if output_exists:
                    logger.info(f"   💡 Output files found for '{dep}'. You can skip dependency with --force")
            
            logger.info(f"\n💡 Suggested command to run dependencies:")
            dep_list = " ".join(missing_deps)
            logger.info(f"   python run_QTLPipeline.py --modules {dep_list}")
            
            logger.info(f"\n💡 Or run all required dependencies automatically:")
            logger.info(f"   python run_QTLPipeline.py --modules {module_name} --auto-deps")
            
            return False
        return True

    def check_module_outputs(self, module_name):
        """Check if a module's output files already exist"""
        output_files = self.modules[module_name].get('output_files', [])
        existing_files = []
        
        for file_pattern in output_files:
            if Path(file_pattern).exists():
                existing_files.append(file_pattern)
        
        if existing_files:
            logger.info(f"   📁 Found existing output files for '{module_name}':")
            for file_path in existing_files:
                logger.info(f"      - {file_path}")
            return True
        return False

    def mark_module_completed(self, module_name, execution_time):
        """Mark a module as completed and record execution time"""
        self.completed_modules.add(module_name)
        self.execution_times[module_name] = execution_time
        logger.info(f"✅ {module_name} completed successfully in {execution_time:.2f} seconds")

    def run_data_preparation(self):
        """Run data preparation module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING DATA PREPARATION MODULE")
        logger.info("="*60)
        
        try:
            # Import utility modules
            try:
                from scripts.utils import validation
                from scripts.utils.enhanced_qc import EnhancedQC
            except ImportError:
                from scripts.utils import validation
                from scripts.utils.enhanced_qc import EnhancedQC
            
            # Step 1: Comprehensive input validation
            logger.info("📋 Validating inputs...")
            validation.validate_inputs(self.config)
            
            # Step 2: Enhanced Quality Control
            logger.info("🔍 Running enhanced QC...")
            qc_processor = EnhancedQC(self.config)
            qc_results = qc_processor.run_data_preparation()
            
            if qc_results:
                self.mark_module_completed('data_preparation', (datetime.now() - start_time).total_seconds())
                return True
            else:
                logger.error("❌ Data preparation failed")
                return False
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_genotype_processing(self):
        """Run genotype processing module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING GENOTYPE PROCESSING MODULE")
        logger.info("="*60)
        
        try:
            from scripts.utils.genotype_processing import process_genotypes
            success = process_genotypes(self.config)
            
            if success:
                self.mark_module_completed('genotype_processing', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            logger.error(f"❌ Genotype processing failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_expression_processing(self):
        """Run expression processing module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING EXPRESSION PROCESSING MODULE")
        logger.info("="*60)
        
        try:
            from scripts.utils.qtl_analysis import process_expression_data
            success = process_expression_data(self.config)
            
            if success:
                self.mark_module_completed('expression_processing', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            logger.error(f"❌ Expression processing failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_quality_control(self):
        """Run quality control module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING QUALITY CONTROL MODULE")
        logger.info("="*60)
        
        try:
            from scripts.utils.enhanced_qc import EnhancedQC
            qc = EnhancedQC(self.config)
            success = qc.run_quality_control()
            
            if success:
                self.mark_module_completed('quality_control', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            logger.error(f"❌ Quality control failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_qtl_mapping(self):
        """Run QTL mapping module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING QTL MAPPING MODULE")
        logger.info("="*60)
        
        try:
            # Import and use the original main pipeline for QTL mapping to preserve all functionality
            from scripts.main import QTLPipeline
            
            # Create a temporary pipeline instance for QTL mapping
            temp_pipeline = QTLPipeline(self.config_path)
            temp_pipeline.setup_directories()
            
            # Prepare genotype data
            logger.info("🧬 Preparing genotype data...")
            from scripts.utils.qtl_analysis import prepare_genotypes
            genotype_file = prepare_genotypes(self.config, temp_pipeline.results_dir)
            
            # Run QTL analyses
            logger.info("🔍 Running QTL analyses...")
            qtl_results = temp_pipeline.run_qtl_analyses(genotype_file)
            
            if qtl_results and any('cis' in result or 'trans' in result for result in qtl_results.values()):
                self.mark_module_completed('qtl_mapping', (datetime.now() - start_time).total_seconds())
                return True
            else:
                logger.error("❌ QTL mapping failed to produce results")
                return False
            
        except Exception as e:
            logger.error(f"❌ QTL mapping failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_fine_mapping(self):
        """Run fine mapping module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING FINE MAPPING MODULE")
        logger.info("="*60)
        
        try:
            from scripts.analysis.fine_mapping import run_fine_mapping
            success = run_fine_mapping(self.config)
            
            if success:
                self.mark_module_completed('fine_mapping', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            logger.error(f"❌ Fine mapping failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_interaction_analysis(self):
        """Run interaction analysis module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING INTERACTION ANALYSIS MODULE")
        logger.info("="*60)
        
        try:
            from scripts.analysis.interaction_analysis import run_interaction_analysis
            success = run_interaction_analysis(self.config)
            
            if success:
                self.mark_module_completed('interaction_analysis', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            logger.error(f"❌ Interaction analysis failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_visualization(self):
        """Run visualization module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING VISUALIZATION MODULE")
        logger.info("="*60)
        
        try:
            # Import and use the original main pipeline for visualization to preserve all functionality
            from scripts.main import QTLPipeline
            
            # Create a temporary pipeline instance for visualization
            temp_pipeline = QTLPipeline(self.config_path)
            temp_pipeline.setup_directories()
            
            # Load existing results for visualization
            import json
            metadata_file = os.path.join(temp_pipeline.results_dir, "results_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    temp_pipeline.results = json.load(f)
            else:
                logger.warning("📊 No existing results found for visualization. Running QTL mapping first...")
                from scripts.utils.qtl_analysis import prepare_genotypes
                genotype_file = prepare_genotypes(self.config, temp_pipeline.results_dir)
                temp_pipeline.results['qtl'] = temp_pipeline.run_qtl_analyses(genotype_file)
            
            # Generate plots using original pipeline
            if temp_pipeline.config['plotting'].get('enabled', True):
                logger.info("📈 Generating plots...")
                temp_pipeline.generate_plots()
                self.mark_module_completed('visualization', (datetime.now() - start_time).total_seconds())
                return True
            else:
                logger.info("📈 Plotting disabled in config")
                return True
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_report_generation(self):
        """Run report generation module"""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("🚀 RUNNING REPORT GENERATION MODULE")
        logger.info("="*60)
        
        try:
            # Import and use the original main pipeline for report generation to preserve all functionality
            from scripts.main import QTLPipeline
            
            # Create a temporary pipeline instance for reports
            temp_pipeline = QTLPipeline(self.config_path)
            temp_pipeline.setup_directories()
            
            # Load existing results for reporting
            import json
            metadata_file = os.path.join(temp_pipeline.results_dir, "results_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    temp_pipeline.results = json.load(f)
            else:
                logger.warning("📝 No existing results found for reporting. Running QTL mapping first...")
                from scripts.utils.qtl_analysis import prepare_genotypes
                genotype_file = prepare_genotypes(self.config, temp_pipeline.results_dir)
                temp_pipeline.results['qtl'] = temp_pipeline.run_qtl_analyses(genotype_file)
            
            # Generate reports using original pipeline
            logger.info("📝 Generating reports...")
            temp_pipeline.generate_reports()
            
            self.mark_module_completed('report_generation', (datetime.now() - start_time).total_seconds())
            return True
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def run_modules(self, module_list, force=False, auto_deps=False):
        """Run specified modules in correct order with enhanced dependency handling"""
        # Define execution order based on dependencies
        execution_order = [
            'data_preparation',
            'genotype_processing', 
            'expression_processing',
            'quality_control',
            'qtl_mapping',
            'fine_mapping',
            'interaction_analysis',
            'visualization',
            'report_generation'
        ]
        
        # Filter modules to run and maintain order
        modules_to_run = [mod for mod in execution_order if mod in module_list]
        
        logger.info(f"🎯 Target modules: {', '.join(modules_to_run)}")
        
        # Auto-add dependencies if requested
        if auto_deps:
            all_required = self.get_all_dependencies(modules_to_run)
            modules_to_run = [mod for mod in execution_order if mod in all_required]
            logger.info(f"🔧 Auto-added dependencies: {', '.join(modules_to_run)}")
        
        success_count = 0
        total_modules = len(modules_to_run)
        
        for i, module_name in enumerate(modules_to_run, 1):
            logger.info(f"\n📦 Progress: [{i}/{total_modules}] - {module_name}")
            
            if not force and not self.check_dependencies(module_name):
                if auto_deps:
                    logger.error(f"❌ Cannot resolve dependencies for {module_name}")
                    return False
                else:
                    logger.error(f"❌ Skipping {module_name} due to missing dependencies")
                    continue
            
            logger.info(f"▶️  Starting module: {module_name}")
            success = self.modules[module_name]['function']()
            
            if success:
                success_count += 1
                logger.info(f"✅ {module_name} completed successfully")
            else:
                logger.error(f"❌ {module_name} failed!")
                if not force:
                    logger.info("💡 Use --force to continue with other modules")
                    return False
        
        logger.info(f"\n🎉 Pipeline execution summary: {success_count}/{total_modules} modules completed successfully")
        return success_count == total_modules

    def get_all_dependencies(self, target_modules):
        """Get all dependencies for target modules (including transitive dependencies)"""
        all_required = set(target_modules)
        
        for module in target_modules:
            dependencies = self.modules[module]['dependencies']
            for dep in dependencies:
                all_required.add(dep)
                # Recursively get dependencies of dependencies
                all_required.update(self.get_all_dependencies([dep]))
        
        return all_required

    def list_modules(self):
        """List all available modules with detailed descriptions"""
        logger.info("\n" + "="*80)
        logger.info("🔧 QTL PIPELINE - AVAILABLE MODULES")
        logger.info("="*80)
        
        for i, (module_name, module_info) in enumerate(self.modules.items(), 1):
            deps = ", ".join(module_info['dependencies']) if module_info['dependencies'] else "None"
            logger.info(f"{i:2d}. {module_name:25} - {module_info['description']}")
            logger.info(f"     📍 Dependencies: {deps}")
            
            # Show output files
            output_files = module_info.get('output_files', [])
            if output_files:
                logger.info(f"     📁 Outputs: {', '.join([os.path.basename(f) for f in output_files[:2]])}")
                if len(output_files) > 2:
                    logger.info(f"          ... and {len(output_files) - 2} more files")
        
        logger.info(f"\n📊 Total modules available: {len(self.modules)}")
        
        # Show usage examples
        logger.info("\n💡 USAGE EXAMPLES:")
        logger.info("  Run all modules: python run_QTLPipeline.py --all")
        logger.info("  Run specific:    python run_QTLPipeline.py --modules qtl_mapping fine_mapping")
        logger.info("  With auto-deps:  python run_QTLPipeline.py --modules visualization --auto-deps")
        logger.info("  Force run:       python run_QTLPipeline.py --modules qtl_mapping --force")

    def print_execution_summary(self):
        """Print summary of module execution times"""
        if self.execution_times:
            logger.info("\n" + "="*60)
            logger.info("⏱️  EXECUTION TIME SUMMARY")
            logger.info("="*60)
            
            total_time = sum(self.execution_times.values())
            for module, time_taken in self.execution_times.items():
                logger.info(f"  {module:25} - {time_taken:8.2f}s ({time_taken/60:6.2f}m)")
            
            logger.info(f"  {'Total':25} - {total_time:8.2f}s ({total_time/60:6.2f}m)")

def main():
    parser = argparse.ArgumentParser(
        description='Modular QTL Analysis Pipeline - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
📚 COMPREHENSIVE USAGE GUIDE:

BASIC USAGE:
  Run all modules:
    python {__file__} --all

  Run specific modules:
    python {__file__} --modules genotype_processing expression_processing
  
  Run QTL mapping and downstream analysis:
    python {__file__} --modules qtl_mapping fine_mapping visualization

  List available modules:
    python {__file__} --list

ADVANCED OPTIONS:
  Auto-resolve dependencies:
    python {__file__} --modules visualization --auto-deps

  Force run (ignore dependency checks):
    python {__file__} --modules qtl_mapping --force

  Custom configuration:
    python {__file__} --config config/test_config.yaml --modules quality_control

  Run with specific log level:
    python {__file__} --modules data_preparation --log-level DEBUG

MODULE COMBINATIONS:
  Data processing only:
    --modules data_preparation genotype_processing expression_processing quality_control

  QTL analysis only:
    --modules qtl_mapping fine_mapping interaction_analysis

  Visualization and reporting:
    --modules visualization report_generation

  Full analysis without reports:
    --modules data_preparation genotype_processing expression_processing quality_control qtl_mapping fine_mapping

COMPATIBILITY:
  This modular pipeline works alongside the original main.py:
    Original full pipeline: python main.py [config_file]
    Modular pipeline:       python run_QTLPipeline.py --modules [module_list]
        '''
    )
    
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file (default: config/config.yaml)')
    
    parser.add_argument('--modules', nargs='+', 
                       choices=['data_preparation', 'genotype_processing', 'expression_processing',
                               'quality_control', 'qtl_mapping', 'fine_mapping', 
                               'interaction_analysis', 'visualization', 'report_generation'],
                       help='Run specific modules')
    
    parser.add_argument('--all', action='store_true',
                       help='Run all modules in pipeline')
    
    parser.add_argument('--list', action='store_true',
                       help='List all available modules with descriptions')
    
    parser.add_argument('--force', action='store_true',
                       help='Force run even if dependencies are missing')
    
    parser.add_argument('--auto-deps', action='store_true',
                       help='Automatically resolve and run all dependencies')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize pipeline
    pipeline = ModularQTLPipeline(args.config)
    
    if args.list:
        pipeline.list_modules()
        return
    
    # Validate module arguments
    if not args.all and not args.modules:
        logger.error("❌ Please specify either --all, --modules, or --list")
        parser.print_help()
        sys.exit(1)
    
    # Run pipeline
    start_time = datetime.now()
    logger.info(f"🚀 Starting Modular QTL Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.all:
            # Run all modules
            all_modules = list(pipeline.modules.keys())
            success = pipeline.run_modules(all_modules, args.force, args.auto_deps)
        else:
            # Run specific modules
            success = pipeline.run_modules(args.modules, args.force, args.auto_deps)
        
        # Print execution summary
        pipeline.print_execution_summary()
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n⏱️  Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}m)")
        
        if success:
            logger.info("🎉 Modular pipeline completed successfully!")
            logger.info("💡 You can also use the original pipeline: python main.py")
        else:
            logger.error("❌ Modular pipeline completed with errors")
            logger.info("💡 Check the logs for detailed error information")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed with unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()