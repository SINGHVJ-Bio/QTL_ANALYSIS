#!/usr/bin/env python3
"""
Enhanced Main QTL Analysis Pipeline with cis/trans capabilities - Enhanced Version
Optimized for large datasets with 100GB+ VCF files
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with modular pipeline compatibility, consistent directory management, and additional features.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
import warnings
import psutil

warnings.filterwarnings('ignore')

# Import utility modules with better error handling
try:
    from scripts.utils import validation, plotting, qtl_analysis, gwas_analysis, report_generator
    from scripts.utils.enhanced_qc import EnhancedQC
    from scripts.utils.advanced_plotting import AdvancedPlotter
    from scripts.utils.normalization_comparison import NormalizationComparison
    from scripts.utils.genotype_processing import process_genotypes
    from scripts.utils.batch_correction import run_batch_correction_pipeline
    from scripts.utils.deseq2_vst_python import deseq2_vst_python, simple_vst_fallback
    from scripts.utils.directory_manager import get_directory_manager, get_module_directories
except ImportError as e:
    # Setup basic logging first for import errors
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error(f"Import error: {e}")
    logging.error("Please ensure all utility modules are available")
    raise

# Import analysis modules
try:
    from scripts.analysis.interaction_analysis import InteractionAnalysis
    from scripts.analysis.fine_mapping import FineMapping, run_fine_mapping
except ImportError:
    InteractionAnalysis = None
    FineMapping = None
    run_fine_mapping = None
    logging.warning("Interaction analysis and fine-mapping modules not available")

class QTLPipeline:
    def __init__(self, config_file=None):
        self.start_time = datetime.now()
        
        # Step 1: Setup basic logging immediately for initialization phase
        self._setup_basic_logging()
        
        # Step 2: Load config (will use basic logger)
        self.config = self.load_config(config_file)
        self.results_dir = Path(self.config['results_dir'])
        
        # Step 3: Create essential directories for proper logging
        self._create_essential_directories()
        
        # Step 4: Setup proper logging with file handler
        self.setup_logging()
        
        # Step 5: Setup full directory structure
        self.setup_directories()
        
        self.results = {}
        self.monitor_resources = self.config.get('large_data', {}).get('monitor_resources', True)
        self.directory_manager = None
        
    def _setup_basic_logging(self):
        """Setup basic logging for initialization phase before directories exist"""
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self._basic_logger = logging.getLogger('QTLPipeline_Basic')
        
    def _create_essential_directories(self):
        """Create only essential directories needed for logging"""
        # Create minimal directory structure for logging
        self.logs_dir = self.results_dir / 'system' / 'logs'
        self.temp_dir = self.results_dir / 'system' / 'temporary_files'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, '_basic_logger'):
            self._basic_logger.info(f"📁 Created essential directories in: {self.results_dir}")
        
    def setup_logging(self):
        """Setup proper logging with file handler"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Use directory manager to get logs directory if available, otherwise use basic structure
        if hasattr(self, 'directory_manager') and self.directory_manager:
            logs_dir = self.directory_manager.get_directory('system', 'logs')
        else:
            logs_dir = self.logs_dir
            
        log_file = logs_dir / f"pipeline_{timestamp}.log"
        
        # Remove existing handlers and setup new ones
        logger = logging.getLogger('QTLPipeline')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        self.logger = logger
        self.logger.info("🚀 Enhanced QTL Analysis Pipeline Started")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        self.logger.info(f"🔧 Analysis mode: {self.config['analysis'].get('qtl_mode', 'cis')}")
        self.logger.info(f"🔍 Enhanced QC: {self.config.get('enhanced_qc', {}).get('enable', False)}")
        self.logger.info(f"🤝 Interaction analysis: {self.config.get('interaction_analysis', {}).get('enable', False)}")
        self.logger.info(f"🎯 Fine-mapping: {self.config.get('fine_mapping', {}).get('enable', False)}")
        self.logger.info(f"💾 Large data optimizations: {self.config.get('large_data', {}).get('force_plink', False)}")
        self.logger.info(f"🔄 Modular pipeline also available: python run_QTLPipeline.py --list")
        
    def load_config(self, config_file):
        """Load configuration from YAML file with validation"""
        if not config_file:
            config_file = "config/config.yaml"
            
        if not os.path.exists(config_file):
            error_msg = f"Config file not found: {config_file}"
            if hasattr(self, '_basic_logger'):
                self._basic_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            mandatory_fields = ['results_dir', 'input_files']
            for field in mandatory_fields:
                if field not in config:
                    error_msg = f"'{field}' must be specified in the config file"
                    if hasattr(self, '_basic_logger'):
                        self._basic_logger.error(error_msg)
                    raise ValueError(error_msg)
            
            mandatory_inputs = ['genotypes', 'covariates', 'annotations']
            for input_file in mandatory_inputs:
                if input_file not in config['input_files']:
                    error_msg = f"'{input_file}' must be specified in 'input_files'"
                    if hasattr(self, '_basic_logger'):
                        self._basic_logger.error(error_msg)
                    raise ValueError(error_msg)
            
            config.setdefault('genotype_processing', {})
            processing_defaults = {
                'auto_detect_format': True,
                'filter_variants': True,
                'normalize_chromosomes': True,
                'handle_multiallelic': True,
                'remove_phasing': True,
                'min_maf': 0.01,
                'min_call_rate': 0.95,
                'prefer_plink': True
            }
            for key, value in processing_defaults.items():
                config['genotype_processing'].setdefault(key, value)
            
            config.setdefault('large_data', {})
            large_data_defaults = {
                'min_memory_gb': 16,
                'min_disk_gb': 50,
                'command_timeout': 3600,
                'process_by_chromosome': True,
                'force_plink': False,
                'monitor_resources': True
            }
            for key, value in large_data_defaults.items():
                config['large_data'].setdefault(key, value)
            
            config.setdefault('analysis', {})
            config['analysis'].setdefault('qtl_types', 'all')
            config['analysis'].setdefault('qtl_mode', 'cis')
            config['analysis'].setdefault('run_gwas', False)
            
            config.setdefault('qtl', {})
            qtl_defaults = {
                'cis_window': 1000000,
                'permutations': 1000,
                'fdr_threshold': 0.05,
                'maf_threshold': 0.05,
                'min_maf': 0.01,
                'min_call_rate': 0.95,
                'chunk_genes': 100,
                'chunk_variants': 10000
            }
            for key, value in qtl_defaults.items():
                config['qtl'].setdefault(key, value)
            
            config.setdefault('enhanced_qc', {'enable': True})
            config.setdefault('interaction_analysis', {'enable': False})
            config.setdefault('fine_mapping', {'enable': False})
            
            config.setdefault('performance', {})
            performance_defaults = {
                'num_threads': 4,
                'memory_gb': 8,
                'chunk_size': 100,
                'max_chunk_memory': 4
            }
            for key, value in performance_defaults.items():
                config['performance'].setdefault(key, value)
            
            config.setdefault('plotting', {'enabled': True})
            config.setdefault('output', {'generate_report': True})
            config.setdefault('qc', {'check_sample_concordance': True})
            
            config.setdefault('paths', {})
            tool_defaults = {
                'qtltools': 'qtltools',
                'bcftools': 'bcftools', 
                'bgzip': 'bgzip',
                'tabix': 'tabix',
                'python': 'python3',
                'plink': 'plink',
                'R': 'R'
            }
            for key, value in tool_defaults.items():
                config['paths'].setdefault(key, value)
            
            if hasattr(self, '_basic_logger'):
                self._basic_logger.info(f"✅ Configuration loaded from: {config_file}")
            
            return config
            
        except Exception as e:
            error_msg = f"Error loading config file: {e}"
            if hasattr(self, '_basic_logger'):
                self._basic_logger.error(error_msg)
            raise
            
    def setup_directories(self):
        """Create only essential directories using directory manager"""
        try:
            self.directory_manager = get_directory_manager(self.results_dir)
            self.directory_manager.initialize_pipeline_directories()
            
            # Update directory attributes with directory manager paths
            self.logs_dir = self.directory_manager.get_directory('system', 'logs')
            self.temp_dir = self.directory_manager.get_directory('system', 'temporary_files')
            
            self.logger.info(f"✅ Essential directories created in: {self.results_dir}")
            self.logger.info("📁 Analysis directories will be created by processing modules as needed")
            
        except Exception as e:
            self.logger.error(f"❌ Directory setup failed: {e}")
            # Fallback: use already created minimal directory structure
            self.logger.warning("⚠️ Using fallback directory structure")
            
    def run_pipeline(self):
        """Execute the complete analysis pipeline with resource monitoring"""
        try:
            if self.monitor_resources:
                self.monitor_system_resources()
            
            self.logger.info("📋 Step 1: Validating inputs...")
            validation.validate_inputs(self.config)
            
            if self.config.get('enhanced_qc', {}).get('enable', False):
                self.logger.info("🔍 Step 2: Running enhanced QC...")
                qc_results = self.run_enhanced_qc()
                self.results['qc'] = qc_results
            
            if self.config.get('enhanced_qc', {}).get('generate_normalization_plots', True):
                self.logger.info("📊 Generating normalization comparison reports...")
                try:
                    # Setup directories for normalization comparison
                    norm_dirs = get_module_directories(
                        'normalization_comparison',
                        ['comparison_analysis'],
                        str(self.results_dir)
                    )
                    comparison = NormalizationComparison(self.config, str(self.results_dir))
                    self.logger.info("✅ Normalization comparison setup completed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Normalization comparison setup failed: {e}")
                    
            self.logger.info("🧬 Step 3: Preparing genotype data...")
            genotype_file = qtl_analysis.prepare_genotypes(self.config, str(self.results_dir))
            
            try:
                from scripts.utils.genotype_processing import process_genotypes
                self.logger.info("🔄 Using modular genotype processing...")
                genotype_file = process_genotypes(self.config)
            except ImportError:
                self.logger.info("🔧 Using standard genotype processing...")
                genotype_file = qtl_analysis.prepare_genotypes(self.config, str(self.results_dir))
            
            self.logger.info("🔍 Step 4: Running QTL analyses...")
            qtl_results = self.run_qtl_analyses(genotype_file)
            self.results['qtl'] = qtl_results
            
            if self.config['analysis'].get('run_gwas', False):
                self.logger.info("📊 Step 5: Running GWAS analysis...")
                gwas_results = self.run_gwas_analysis(genotype_file)
                self.results['gwas'] = gwas_results
            
            self.logger.info("🔬 Step 6: Running advanced analyses...")
            advanced_results = self.run_advanced_analyses(genotype_file)
            self.results['advanced'] = advanced_results
            
            if self.config['plotting'].get('enabled', True):
                self.logger.info("📈 Step 7: Generating plots...")
                self.generate_plots()
            
            self.logger.info("📝 Step 8: Generating reports...")
            self.generate_reports()
            
            self.cleanup_temp_files()
            
            runtime = datetime.now() - self.start_time
            self.logger.info(f"✅ Pipeline completed successfully in {runtime}")
            self.logger.info(f"💡 You can also run individual modules using: python run_QTLPipeline.py --list")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.cleanup_temp_files()
            raise
            
    def monitor_system_resources(self):
        """Monitor system resources throughout pipeline execution"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.logger.info(f"💻 System Resources:")
        self.logger.info(f"   Memory: {memory.percent:.1f}% used ({memory.available / (1024**3):.1f} GB available)")
        self.logger.info(f"   Disk: {disk.percent:.1f}% used ({disk.free / (1024**3):.1f} GB free)")
        
        if memory.percent > 90:
            self.logger.warning("⚠️  High memory usage detected!")
        if disk.percent > 90:
            self.logger.warning("⚠️  Low disk space detected!")
    
    def cleanup_temp_files(self):
        """Clean up temporary files to free disk space"""
        self.logger.info("🧹 Cleaning up temporary files...")
        
        temp_dirs = [self.temp_dir]
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith(('.tmp', '.temp', '_temp')):
                                file_path = os.path.join(root, file)
                                try:
                                    os.remove(file_path)
                                except:
                                    pass
                    self.logger.info(f"✅ Cleaned up temporary files in {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not clean up {temp_dir}: {e}")
    
    def run_enhanced_qc(self):
        """Run enhanced quality control with proper mapping"""
        try:
            # Setup QC directories
            qc_dirs = get_module_directories(
                'enhanced_qc',
                [
                    'processed_data',
                    {'processed_data': ['quality_control']},
                    'reports',
                    {'reports': ['qc_reports']}
                ],
                str(self.results_dir)
            )
            
            qc_processor = EnhancedQC(self.config)
            
            qtl_types = self.get_qtl_types()
            
            phenotype_files = {}
            for qtl_type in qtl_types:
                config_key = self.map_qtl_type_to_config_key(qtl_type)
                phenotype_files[qtl_type] = self.config['input_files'].get(config_key)
            
            qc_results = qc_processor.run_comprehensive_qc(
                self.config['input_files']['genotypes'],
                phenotype_files,
                str(self.results_dir)
            )
            
            return qc_results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced QC failed: {e}")
            return {}
            
    def run_qtl_analyses(self, genotype_file):
        """Run QTL analyses based on configuration with proper mapping"""
        qtl_types = self.get_qtl_types()
        qtl_mode = self.config['analysis'].get('qtl_mode', 'cis')
        results = {}
        
        # Setup QTL analysis directories
        qtl_dirs = get_module_directories(
            'qtl_analysis',
            [
                'analysis_results',
                {'analysis_results': ['qtl_mapping']}
            ],
            str(self.results_dir)
        )
        
        for qtl_type in qtl_types:
            self.logger.info(f"🔬 Running {qtl_type.upper()} analysis ({qtl_mode})...")
            results[qtl_type] = {}
            
            try:
                config_key = self.map_qtl_type_to_config_key(qtl_type)
                phenotype_file = self.config['input_files'].get(config_key)
                
                if not phenotype_file or not os.path.exists(phenotype_file):
                    self.logger.error(f"❌ Phenotype file not found for {qtl_type}: {phenotype_file}")
                    continue
                
                if qtl_mode in ['cis', 'both']:
                    cis_result = qtl_analysis.run_cis_analysis(
                        self.config, genotype_file, qtl_type, str(self.results_dir)
                    )
                    results[qtl_type]['cis'] = cis_result
                    status_msg = "✅" if cis_result['status'] == 'completed' else "❌"
                    self.logger.info(f"{status_msg} {qtl_type.upper()} cis: {cis_result.get('significant_count', 0)} significant")
                
                if qtl_mode in ['trans', 'both']:
                    trans_result = qtl_analysis.run_trans_analysis(
                        self.config, genotype_file, qtl_type, str(self.results_dir)
                    )
                    results[qtl_type]['trans'] = trans_result
                    status_msg = "✅" if trans_result['status'] == 'completed' else "❌"
                    self.logger.info(f"{status_msg} {qtl_type.upper()} trans: {trans_result.get('significant_count', 0)} significant")
                    
            except Exception as e:
                self.logger.error(f"❌ {qtl_type.upper()} failed: {e}")
                if 'cis' not in results[qtl_type]:
                    results[qtl_type]['cis'] = {
                        'status': 'failed', 
                        'error': str(e),
                        'significant_count': 0
                    }
                if 'trans' not in results[qtl_type] and qtl_mode in ['trans', 'both']:
                    results[qtl_type]['trans'] = {
                        'status': 'failed', 
                        'error': str(e),
                        'significant_count': 0
                    }
                    
        return results
        
    def run_gwas_analysis(self, genotype_file):
        """Run GWAS analysis if requested"""
        try:
            # Setup GWAS directories
            gwas_dirs = get_module_directories(
                'gwas_analysis',
                [
                    'analysis_results',
                    {'analysis_results': ['gwas_results']}
                ],
                str(self.results_dir)
            )
            
            result = gwas_analysis.run_gwas_analysis(
                self.config, genotype_file, str(self.results_dir)
            )
            return result
        except Exception as e:
            self.logger.error(f"❌ GWAS analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def run_advanced_analyses(self, genotype_file):
        """Run advanced analyses like interaction and fine-mapping with proper mapping"""
        advanced_results = {}
        
        if self.config.get('interaction_analysis', {}).get('enable', False) and InteractionAnalysis:
            try:
                # Setup interaction analysis directories
                interaction_dirs = get_module_directories(
                    'interaction_analysis',
                    [
                        'analysis_results',
                        {'analysis_results': ['interaction_analysis']}
                    ],
                    str(self.results_dir)
                )
                
                self.logger.info("🤝 Running interaction analysis...")
                interaction_analyzer = InteractionAnalysis(self.config)
                
                for qtl_type in self.get_qtl_types():
                    config_key = self.map_qtl_type_to_config_key(qtl_type)
                    phenotype_file = self.config['input_files'].get(config_key)
                    if phenotype_file and os.path.exists(phenotype_file):
                        interaction_results = interaction_analyzer.run_interaction_analysis(
                            genotype_file, phenotype_file, 
                            self.config['input_files']['covariates'],
                            str(self.results_dir),
                            qtl_type
                        )
                        advanced_results[f'interaction_{qtl_type}'] = interaction_results
                        
            except Exception as e:
                self.logger.error(f"❌ Interaction analysis failed: {e}")
                advanced_results['interaction'] = {'status': 'failed', 'error': str(e)}
        
        if self.config.get('fine_mapping', {}).get('enable', False) and FineMapping:
            try:
                # Setup fine-mapping directories
                fine_mapping_dirs = get_module_directories(
                    'fine_mapping',
                    [
                        'analysis_results',
                        {'analysis_results': ['fine_mapping']}
                    ],
                    str(self.results_dir)
                )
                
                self.logger.info("🎯 Running fine-mapping...")
                
                if 'qtl' in self.results:
                    for qtl_type, result in self.results['qtl'].items():
                        if 'cis' in result and result['cis']['status'] == 'completed':
                            result_file = result['cis'].get('result_file')
                            if result_file and os.path.exists(result_file):
                                finemap_results = run_fine_mapping(
                                    self.config,
                                    result_file,
                                    genotype_file,
                                    str(self.results_dir),
                                    qtl_type
                                )
                                advanced_results[f'fine_mapping_{qtl_type}_cis'] = finemap_results
                                
            except Exception as e:
                self.logger.error(f"❌ Fine-mapping failed: {e}")
                advanced_results['fine_mapping'] = {'status': 'failed', 'error': str(e)}
        
        return advanced_results
            
    def get_qtl_types(self):
        """Parse and validate QTL types from config with proper mapping"""
        config_types = self.config['analysis']['qtl_types']
        
        if config_types == 'all':
            available_types = []
            for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
                config_key = self.map_qtl_type_to_config_key(qtl_type)
                if (config_key in self.config['input_files'] and 
                    self.config['input_files'][config_key] and 
                    os.path.exists(self.config['input_files'][config_key])):
                    available_types.append(qtl_type)
            return available_types
        elif isinstance(config_types, str):
            return [t.strip() for t in config_types.split(',')]
        elif isinstance(config_types, list):
            return config_types
        else:
            raise ValueError(f"Invalid qtl_types configuration: {config_types}")
            
    def map_qtl_type_to_config_key(self, qtl_type):
        """Map QTL analysis types to config file keys"""
        mapping = {
            'eqtl': 'expression',
            'pqtl': 'protein', 
            'sqtl': 'splicing'
        }
        return mapping.get(qtl_type, qtl_type)
        
    def generate_plots(self):
        """Generate all requested plots"""
        # Setup visualization directories
        viz_dirs = get_module_directories(
            'visualization',
            [
                'visualization',
                {'visualization': ['summary_plots', 'interactive_plots', 'manhattan_plots', 'qq_plots']}
            ],
            str(self.results_dir)
        )
        
        plotter = plotting.QTLPlotter(self.config, self.results, str(self.results_dir))
        
        if 'qtl' in self.results:
            for qtl_type, result in self.results['qtl'].items():
                if 'cis' in result and result['cis']['status'] == 'completed':
                    plotter.create_cis_plots(qtl_type, result['cis'])
                if 'trans' in result and result['trans']['status'] == 'completed':
                    plotter.create_trans_plots(qtl_type, result['trans'])
        
        if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
            plotter.create_gwas_plots(self.results['gwas'])
            
        plotter.create_summary_plots()
        
        if self.config['plotting'].get('plot_types', []):
            if 'interactive' in self.config['plotting']['plot_types']:
                try:
                    advanced_plotter = AdvancedPlotter(self.config, str(self.results_dir))
                    
                    if 'qtl' in self.results:
                        for qtl_type, result in self.results['qtl'].items():
                            if 'cis' in result and result['cis']['status'] == 'completed':
                                nominals_file = result['cis'].get('nominals_file')
                                if nominals_file and os.path.exists(nominals_file):
                                    advanced_plotter.create_interactive_manhattan(
                                        nominals_file, 
                                        f"{qtl_type}_cis",
                                        f"{qtl_type.upper()} cis-QTL Manhattan Plot"
                                    )
                    
                    advanced_plotter.create_multi_panel_figure(self.results, "comprehensive_summary")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Advanced plotting failed: {e}")
        
    def generate_reports(self):
        """Generate comprehensive reports"""
        # Setup report directories
        report_dirs = get_module_directories(
            'report_generation',
            [
                'reports',
                {'reports': ['pipeline_reports', 'analysis_reports']}
            ],
            str(self.results_dir)
        )
        
        report_data = {
            'results': self.results,
            'config': self.config,
            'runtime': str(datetime.now() - self.start_time),
            'timestamp': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'results_dir': str(self.results_dir)
        }
        
        if self.config['output'].get('generate_report', True):
            report_file = self.results_dir / "reports" / "pipeline_reports" / "analysis_report.html"
            report_generator.generate_html_report(report_data, str(report_file))
            self.logger.info(f"📄 HTML report generated: {report_file}")
        
        summary_file = self.results_dir / "reports" / "analysis_reports" / "pipeline_summary.txt"
        report_generator.generate_summary_report(report_data, str(summary_file))
        self.logger.info(f"📋 Summary report generated: {summary_file}")
        
        metadata_file = self.results_dir / "system" / "logs" / "results_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"💾 Results metadata saved: {metadata_file}")
        
        if 'qc' in self.results and self.results['qc']:
            qc_report_dir = self.results_dir / "reports" / "qc_reports"
            self.logger.info(f"🔍 Enhanced QC report available in: {qc_report_dir}")


if __name__ == "__main__":
    # Setup basic logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = None
    
    try:
        pipeline = QTLPipeline(config_file)
        results = pipeline.run_pipeline()
        print(f"✅ Pipeline completed successfully!")
        print(f"💡 You can also run individual modules using: python run_QTLPipeline.py --list")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)