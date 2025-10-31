#!/usr/bin/env python3
"""
Modular QTL Pipeline Runner - Enhanced Version with Consistent Directory Management
Allows running modules separately, in combination, or all together
with dependency checking and helpful messages.

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with comprehensive logging, dependency tracking, and consistent directory management.
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

try:
    from scripts.utils.enhanced_qc import EnhancedQC
    from scripts.utils.genotype_processing import process_genotypes
    from scripts.utils.qtl_analysis import prepare_genotypes, process_expression_data
    from scripts.analysis.fine_mapping import run_fine_mapping
    from scripts.analysis.interaction_analysis import run_interaction_analysis
    from scripts.utils.directory_manager import get_directory_manager, get_module_directories
except ImportError as e:
    logging.error(f"Import error in modular pipeline: {e}")
    raise

warnings.filterwarnings('ignore')

class ModularQTLPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results_dir = Path(self.config['results_dir'])
        
        # REORDERED: Setup logging FIRST
        self.setup_logging()
        self.setup_directories()
        
        # Define modules with dependencies and descriptions
        self.modules = {
            'data_preparation': {
                'function': self.run_data_preparation,
                'dependencies': [],
                'description': 'Prepare and validate input data',
                'output_files': [self.results_dir / 'reports' / 'input_validation_report.txt']
            },
            'genotype_processing': {
                'function': self.run_genotype_processing,
                'dependencies': ['data_preparation'],
                'description': 'Process genotype data (VCF/PLINK formats)',
                'output_files': [self.results_dir / 'processed_data' / 'genotypes' / 'final_genotypes.vcf.gz']
            },
            'expression_processing': {
                'function': self.run_expression_processing,
                'dependencies': ['data_preparation'],
                'description': 'Process expression data (normalization, QC)',
                'output_files': [self.results_dir / 'processed_data' / 'expression' / 'expression_processed.txt']
            },
            'quality_control': {
                'function': self.run_quality_control,
                'dependencies': ['genotype_processing', 'expression_processing'],
                'description': 'Perform comprehensive quality control',
                'output_files': [self.results_dir / 'processed_data' / 'quality_control' / 'qc_report.html']
            },
            'qtl_mapping': {
                'function': self.run_qtl_mapping,
                'dependencies': ['quality_control'],
                'description': 'Perform cis/trans QTL mapping analysis',
                'output_files': [
                    self.results_dir / 'analysis_results' / 'qtl_mapping' / 'cis_qtl.txt',
                    self.results_dir / 'analysis_results' / 'qtl_mapping' / 'trans_qtl.txt'
                ]
            },
            'fine_mapping': {
                'function': self.run_fine_mapping,
                'dependencies': ['qtl_mapping'],
                'description': 'Fine mapping of QTL signals',
                'output_files': [self.results_dir / 'analysis_results' / 'fine_mapping' / 'fine_mapped_variants.txt']
            },
            'interaction_analysis': {
                'function': self.run_interaction_analysis,
                'dependencies': ['qtl_mapping'],
                'description': 'Covariate interaction analysis',
                'output_files': [self.results_dir / 'analysis_results' / 'interaction_analysis' / 'interaction_results.txt']
            },
            'visualization': {
                'function': self.run_visualization,
                'dependencies': ['qtl_mapping', 'fine_mapping'],
                'description': 'Generate plots and visualizations',
                'output_files': [
                    self.results_dir / 'visualization' / 'manhattan_plots' / 'manhattan_plot.png',
                    self.results_dir / 'visualization' / 'qq_plots' / 'qq_plot.png'
                ]
            },
            'report_generation': {
                'function': self.run_report_generation,
                'dependencies': ['qtl_mapping', 'fine_mapping', 'visualization'],
                'description': 'Generate comprehensive HTML reports',
                'output_files': [self.results_dir / 'reports' / 'pipeline_reports' / 'final_report.html']
            }
        }
        
        # Track which modules have been run in this session
        self.completed_modules = set()
        self.execution_times = {}
        self.directory_manager = None

    def setup_logging(self):
        """Setup logging to use results directory"""
        # Get logs directory from directory manager
        if self.directory_manager:
            logs_dir = self.directory_manager.get_directory('system', 'logs')
        else:
            logs_dir = self.results_dir / 'system' / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / 'qtl_pipeline.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger('ModularQTLPipeline')
        self.logger.info(f"✅ Logging to: {log_file}")

    def load_config(self):
        """Load configuration from YAML file with enhanced error handling"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logging.info(f"✅ Configuration loaded from: {self.config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"❌ Config file not found: {self.config_path}")
            logging.info("💡 Available config files:")
            config_dir = Path("config")
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    logging.info(f"   - {config_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"❌ Error loading config file: {e}")
            sys.exit(1)

    def setup_directories(self):
        """Initialize only essential directories using directory manager"""
        try:
            from scripts.utils.directory_manager import get_directory_manager
            self.directory_manager = get_directory_manager(self.results_dir)
            self.directory_manager.initialize_pipeline_directories()
            
            self.logger.info(f"✅ Directory structure initialized in: {self.results_dir}")
            self.logger.info("📁 Subdirectories will be created by processing modules as needed")
            
        except Exception as e:
            self.logger.error(f"❌ Directory setup failed: {e}")
            # Fallback: create minimal directory structure
            (self.results_dir / 'system' / 'logs').mkdir(parents=True, exist_ok=True)
            self.logger.warning("⚠️ Using fallback directory structure")

    def check_dependencies(self, module_name):
        """Check if all dependencies for a module are satisfied with detailed reporting"""
        dependencies = self.modules[module_name]['dependencies']
        missing_deps = [dep for dep in dependencies if dep not in self.completed_modules]
        
        if missing_deps:
            self.logger.error(f"❌ Module '{module_name}' requires the following modules to be run first:")
            for dep in missing_deps:
                dep_info = self.modules[dep]
                self.logger.error(f"   📍 {dep}: {dep_info['description']}")
                
                output_exists = self.check_module_outputs(dep)
                if output_exists:
                    self.logger.info(f"   💡 Output files found for '{dep}'. You can skip dependency with --force")
            
            self.logger.info(f"\n💡 Suggested command to run dependencies:")
            dep_list = " ".join(missing_deps)
            self.logger.info(f"   python run_QTLPipeline.py --modules {dep_list}")
            
            self.logger.info(f"\n💡 Or run all required dependencies automatically:")
            self.logger.info(f"   python run_QTLPipeline.py --modules {module_name} --auto-deps")
            
            return False
        return True

    def check_module_outputs(self, module_name):
        """Check if a module's output files already exist"""
        output_files = self.modules[module_name].get('output_files', [])
        existing_files = []
        
        for file_path in output_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
        
        if existing_files:
            self.logger.info(f"   📁 Found existing output files for '{module_name}':")
            for file_path in existing_files:
                self.logger.info(f"      - {file_path}")
            return True
        return False

    def mark_module_completed(self, module_name, execution_time):
        """Mark a module as completed and record execution time"""
        self.completed_modules.add(module_name)
        self.execution_times[module_name] = execution_time
        self.logger.info(f"✅ {module_name} completed successfully in {execution_time:.2f} seconds")

    def run_data_preparation(self):
        """Run data preparation module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING DATA PREPARATION MODULE")
        self.logger.info("="*60)
        
        try:
            from scripts.utils import validation
            from scripts.utils.enhanced_qc import EnhancedQC
            
            self.logger.info("📋 Validating inputs...")
            validation.validate_inputs(self.config)
            
            self.logger.info("🔍 Running enhanced QC...")
            qc_processor = EnhancedQC(self.config)
            qc_results = qc_processor.run_data_preparation()
            
            if qc_results:
                self.mark_module_completed('data_preparation', (datetime.now() - start_time).total_seconds())
                return True
            else:
                self.logger.error("❌ Data preparation failed")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_genotype_processing(self):
        """Run genotype processing module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING GENOTYPE PROCESSING MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            processing_dirs = get_module_directories(
                'genotype_processing',
                [
                    'processed_data',
                    {'processed_data': ['genotypes']}
                ],
                str(self.results_dir)
            )
            
            from scripts.utils.genotype_processing import process_genotypes
            success = process_genotypes(self.config)
            
            if success:
                self.mark_module_completed('genotype_processing', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Genotype processing failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_expression_processing(self):
        """Run expression processing module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING EXPRESSION PROCESSING MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            processing_dirs = get_module_directories(
                'expression_processing',
                [
                    'processed_data',
                    {'processed_data': ['expression']},
                    'comparison_analysis'
                ],
                str(self.results_dir)
            )
            
            from scripts.utils.qtl_analysis import process_expression_data
            success = process_expression_data(self.config)
            
            if success:
                self.mark_module_completed('expression_processing', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Expression processing failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_quality_control(self):
        """Run quality control module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING QUALITY CONTROL MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            qc_dirs = get_module_directories(
                'quality_control',
                [
                    'processed_data',
                    {'processed_data': ['quality_control']},
                    'reports',
                    {'reports': ['qc_reports']}
                ],
                str(self.results_dir)
            )
            
            from scripts.utils.enhanced_qc import EnhancedQC
            qc = EnhancedQC(self.config)
            success = qc.run_quality_control()
            
            if success:
                self.mark_module_completed('quality_control', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Quality control failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_qtl_mapping(self):
        """Run QTL mapping module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING QTL MAPPING MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            qtl_dirs = get_module_directories(
                'qtl_mapping',
                [
                    'analysis_results',
                    {'analysis_results': ['qtl_mapping']}
                ],
                str(self.results_dir)
            )
            
            from scripts.main import QTLPipeline
            
            temp_pipeline = QTLPipeline(self.config_path)
            temp_pipeline.setup_directories()
            
            self.logger.info("🧬 Preparing genotype data...")
            from scripts.utils.qtl_analysis import prepare_genotypes
            genotype_file = prepare_genotypes(self.config, str(self.results_dir))
            
            self.logger.info("🔍 Running QTL analyses...")
            qtl_results = temp_pipeline.run_qtl_analyses(genotype_file)
            
            if qtl_results and any('cis' in result or 'trans' in result for result in qtl_results.values()):
                self.mark_module_completed('qtl_mapping', (datetime.now() - start_time).total_seconds())
                return True
            else:
                self.logger.error("❌ QTL mapping failed to produce results")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ QTL mapping failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_fine_mapping(self):
        """Run fine mapping module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING FINE MAPPING MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            fine_mapping_dirs = get_module_directories(
                'fine_mapping',
                [
                    'analysis_results',
                    {'analysis_results': ['fine_mapping']}
                ],
                str(self.results_dir)
            )
            
            from scripts.analysis.fine_mapping import run_fine_mapping
            success = run_fine_mapping(self.config)
            
            if success:
                self.mark_module_completed('fine_mapping', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Fine mapping failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_interaction_analysis(self):
        """Run interaction analysis module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING INTERACTION ANALYSIS MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            interaction_dirs = get_module_directories(
                'interaction_analysis',
                [
                    'analysis_results',
                    {'analysis_results': ['interaction_analysis']}
                ],
                str(self.results_dir)
            )
            
            from scripts.analysis.interaction_analysis import run_interaction_analysis
            success = run_interaction_analysis(self.config)
            
            if success:
                self.mark_module_completed('interaction_analysis', (datetime.now() - start_time).total_seconds())
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Interaction analysis failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_visualization(self):
        """Run visualization module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING VISUALIZATION MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            viz_dirs = get_module_directories(
                'visualization',
                [
                    'visualization',
                    {'visualization': ['summary_plots', 'interactive_plots', 'manhattan_plots', 'qq_plots']}
                ],
                str(self.results_dir)
            )
            
            from scripts.main import QTLPipeline
            
            temp_pipeline = QTLPipeline(self.config_path)
            temp_pipeline.setup_directories()
            
            import json
            metadata_file = self.results_dir / "system" / "logs" / "results_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    temp_pipeline.results = json.load(f)
            else:
                self.logger.warning("📊 No existing results found for visualization. Running QTL mapping first...")
                from scripts.utils.qtl_analysis import prepare_genotypes
                genotype_file = prepare_genotypes(self.config, str(self.results_dir))
                temp_pipeline.results['qtl'] = temp_pipeline.run_qtl_analyses(genotype_file)
            
            if temp_pipeline.config['plotting'].get('enabled', True):
                self.logger.info("📈 Generating plots...")
                temp_pipeline.generate_plots()
                self.mark_module_completed('visualization', (datetime.now() - start_time).total_seconds())
                return True
            else:
                self.logger.info("📈 Plotting disabled in config")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Visualization failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_report_generation(self):
        """Run report generation module"""
        start_time = datetime.now()
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 RUNNING REPORT GENERATION MODULE")
        self.logger.info("="*60)
        
        try:
            # Setup module-specific directories
            report_dirs = get_module_directories(
                'report_generation',
                [
                    'reports',
                    {'reports': ['pipeline_reports', 'analysis_reports']}
                ],
                str(self.results_dir)
            )
            
            from scripts.main import QTLPipeline
            
            temp_pipeline = QTLPipeline(self.config_path)
            temp_pipeline.setup_directories()
            
            import json
            metadata_file = self.results_dir / "system" / "logs" / "results_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    temp_pipeline.results = json.load(f)
            else:
                self.logger.warning("📝 No existing results found for reporting. Running QTL mapping first...")
                from scripts.utils.qtl_analysis import prepare_genotypes
                genotype_file = prepare_genotypes(self.config, str(self.results_dir))
                temp_pipeline.results['qtl'] = temp_pipeline.run_qtl_analyses(genotype_file)
            
            self.logger.info("📝 Generating reports...")
            temp_pipeline.generate_reports()
            
            self.mark_module_completed('report_generation', (datetime.now() - start_time).total_seconds())
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def run_modules(self, module_list, force=False, auto_deps=False):
        """Run specified modules in correct order with enhanced dependency handling"""
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
        
        modules_to_run = [mod for mod in execution_order if mod in module_list]
        
        self.logger.info(f"🎯 Target modules: {', '.join(modules_to_run)}")
        
        if auto_deps:
            all_required = self.get_all_dependencies(modules_to_run)
            modules_to_run = [mod for mod in execution_order if mod in all_required]
            self.logger.info(f"🔧 Auto-added dependencies: {', '.join(modules_to_run)}")
        
        success_count = 0
        total_modules = len(modules_to_run)
        
        for i, module_name in enumerate(modules_to_run, 1):
            self.logger.info(f"\n📦 Progress: [{i}/{total_modules}] - {module_name}")
            
            if not force and not self.check_dependencies(module_name):
                if auto_deps:
                    self.logger.error(f"❌ Cannot resolve dependencies for {module_name}")
                    return False
                else:
                    self.logger.error(f"❌ Skipping {module_name} due to missing dependencies")
                    continue
            
            self.logger.info(f"▶️  Starting module: {module_name}")
            success = self.modules[module_name]['function']()
            
            if success:
                success_count += 1
                self.logger.info(f"✅ {module_name} completed successfully")
            else:
                self.logger.error(f"❌ {module_name} failed!")
                if not force:
                    self.logger.info("💡 Use --force to continue with other modules")
                    return False
        
        self.logger.info(f"\n🎉 Pipeline execution summary: {success_count}/{total_modules} modules completed successfully")
        return success_count == total_modules

    def get_all_dependencies(self, target_modules):
        """Get all dependencies for target modules (including transitive dependencies)"""
        all_required = set(target_modules)
        
        for module in target_modules:
            dependencies = self.modules[module]['dependencies']
            for dep in dependencies:
                all_required.add(dep)
                all_required.update(self.get_all_dependencies([dep]))
        
        return all_required

    def list_modules(self):
        """List all available modules with detailed descriptions"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🔧 QTL PIPELINE - AVAILABLE MODULES")
        self.logger.info("="*80)
        
        for i, (module_name, module_info) in enumerate(self.modules.items(), 1):
            deps = ", ".join(module_info['dependencies']) if module_info['dependencies'] else "None"
            self.logger.info(f"{i:2d}. {module_name:25} - {module_info['description']}")
            self.logger.info(f"     📍 Dependencies: {deps}")
            
            output_files = module_info.get('output_files', [])
            if output_files:
                self.logger.info(f"     📁 Outputs: {', '.join([os.path.basename(str(f)) for f in output_files[:2]])}")
                if len(output_files) > 2:
                    self.logger.info(f"          ... and {len(output_files) - 2} more files")
        
        self.logger.info(f"\n📊 Total modules available: {len(self.modules)}")
        
        self.logger.info("\n💡 USAGE EXAMPLES:")
        self.logger.info("  Run all modules: python run_QTLPipeline.py --all")
        self.logger.info("  Run specific:    python run_QTLPipeline.py --modules qtl_mapping fine_mapping")
        self.logger.info("  With auto-deps:  python run_QTLPipeline.py --modules visualization --auto-deps")
        self.logger.info("  Force run:       python run_QTLPipeline.py --modules qtl_mapping --force")

    def print_execution_summary(self):
        """Print summary of module execution times"""
        if self.execution_times:
            self.logger.info("\n" + "="*60)
            self.logger.info("⏱️  EXECUTION TIME SUMMARY")
            self.logger.info("="*60)
            
            total_time = sum(self.execution_times.values())
            for module, time_taken in self.execution_times.items():
                self.logger.info(f"  {module:25} - {time_taken:8.2f}s ({time_taken/60:6.2f}m)")
            
            self.logger.info(f"  {'Total':25} - {total_time:8.2f}s ({total_time/60:6.2f}m)")

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
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    pipeline = ModularQTLPipeline(args.config)
    
    if args.list:
        pipeline.list_modules()
        return
    
    if not args.all and not args.modules:
        logging.error("❌ Please specify either --all, --modules, or --list")
        parser.print_help()
        sys.exit(1)
    
    start_time = datetime.now()
    pipeline.logger.info(f"🚀 Starting Modular QTL Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.all:
            all_modules = list(pipeline.modules.keys())
            success = pipeline.run_modules(all_modules, args.force, args.auto_deps)
        else:
            success = pipeline.run_modules(args.modules, args.force, args.auto_deps)
        
        pipeline.print_execution_summary()
        
        total_time = (datetime.now() - start_time).total_seconds()
        pipeline.logger.info(f"\n⏱️  Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}m)")
        
        if success:
            pipeline.logger.info("🎉 Modular pipeline completed successfully!")
            pipeline.logger.info("💡 You can also use the original pipeline: python main.py")
        else:
            pipeline.logger.error("❌ Modular pipeline completed with errors")
            pipeline.logger.info("💡 Check the logs for detailed error information")
            sys.exit(1)
            
    except KeyboardInterrupt:
        pipeline.logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        pipeline.logger.error(f"❌ Pipeline failed with unexpected error: {e}")
        pipeline.logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()