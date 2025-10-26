#!/usr/bin/env python3
"""
Enhanced Main QTL Analysis Pipeline with cis/trans capabilities - Enhanced Version
Optimized for large datasets with 100GB+ VCF files
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

Enhanced with modular pipeline compatibility and additional features.
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

# Import utility modules
try:
    from utils import validation, plotting, qtl_analysis, gwas_analysis, report_generator
    from utils.enhanced_qc import EnhancedQC
    from utils.advanced_plotting import AdvancedPlotter
    from utils.normalization_comparison import NormalizationComparison
    from utils.genotype_processing import process_genotypes
except ImportError:
    # Try alternative import paths for script organization
    try:
        from scripts.utils import validation, plotting, qtl_analysis, gwas_analysis, report_generator
        from scripts.utils.enhanced_qc import EnhancedQC
        from scripts.utils.advanced_plotting import AdvancedPlotter
        from scripts.utils.normalization_comparison import NormalizationComparison
        from scripts.utils.genotype_processing import process_genotypes
    except ImportError as e:
        logging.error(f"Import error: {e}")
        logging.error("Please ensure all utility modules are available")
        raise

# Import analysis modules
try:
    from analysis.interaction_analysis import InteractionAnalysis
    from analysis.fine_mapping import FineMapping
except ImportError:
    try:
        from scripts.analysis.interaction_analysis import InteractionAnalysis
        from scripts.analysis.fine_mapping import FineMapping
    except ImportError:
        InteractionAnalysis = None
        FineMapping = None
        logging.warning("Interaction analysis and fine-mapping modules not available")

logger = logging.getLogger('QTLPipeline')

class QTLPipeline:
    def __init__(self, config_file=None):
        self.start_time = datetime.now()
        self.config = self.load_config(config_file)
        self.setup_directories()
        self.setup_logging()
        self.results = {}
        self.monitor_resources = self.config.get('large_data', {}).get('monitor_resources', True)
        
    def load_config(self, config_file):
        """Load configuration from YAML file with validation"""
        if not config_file:
            config_file = "config/config.yaml"
            
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate mandatory fields
            mandatory_fields = ['results_dir', 'input_files']
            for field in mandatory_fields:
                if field not in config:
                    raise ValueError(f"'{field}' must be specified in the config file")
            
            mandatory_inputs = ['genotypes', 'covariates', 'annotations']
            for input_file in mandatory_inputs:
                if input_file not in config['input_files']:
                    raise ValueError(f"'{input_file}' must be specified in 'input_files'")
            
            # Set comprehensive defaults including large data defaults
            config.setdefault('genotype_processing', {})
            processing_defaults = {
                'auto_detect_format': True,
                'filter_variants': True,
                'normalize_chromosomes': True,
                'handle_multiallelic': True,
                'remove_phasing': True,
                'min_maf': 0.01,
                'min_call_rate': 0.95,
                'prefer_plink': True  # New: Prefer PLINK for large datasets
            }
            for key, value in processing_defaults.items():
                config['genotype_processing'].setdefault(key, value)
            
            # Large data defaults
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
            
            # Analysis defaults
            config.setdefault('analysis', {})
            config['analysis'].setdefault('qtl_types', 'all')
            config['analysis'].setdefault('qtl_mode', 'cis')
            config['analysis'].setdefault('run_gwas', False)
            
            # QTL defaults
            config.setdefault('qtl', {})
            qtl_defaults = {
                'cis_window': 1000000,
                'permutations': 1000,
                'fdr_threshold': 0.05,
                'maf_threshold': 0.05,
                'min_maf': 0.01,
                'min_call_rate': 0.95,
                'chunk_genes': 100,  # New: Memory optimization
                'chunk_variants': 10000  # New: Memory optimization
            }
            for key, value in qtl_defaults.items():
                config['qtl'].setdefault(key, value)
            
            # Enhanced features defaults
            config.setdefault('enhanced_qc', {'enable': True})
            config.setdefault('interaction_analysis', {'enable': False})
            config.setdefault('fine_mapping', {'enable': False})
            
            # Performance defaults
            config.setdefault('performance', {})
            performance_defaults = {
                'num_threads': 4,
                'memory_gb': 8,
                'chunk_size': 100,
                'max_chunk_memory': 4
            }
            for key, value in performance_defaults.items():
                config['performance'].setdefault(key, value)
            
            # Other section defaults
            config.setdefault('plotting', {'enabled': True})
            config.setdefault('output', {'generate_report': True})
            config.setdefault('qc', {'check_sample_concordance': True})
            
            # Tool paths
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
            
            return config
            
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            raise
            
    def setup_directories(self):
        """Create comprehensive directory structure"""
        self.results_dir = self.config['results_dir']
        
        directories = {
            'logs': "logs",
            'temp': "temp", 
            'plots': "plots",
            'qtl_results': "QTL_results",
            'gwas_results': "GWAS_results",
            'reports': "reports",
            'genotype_processing': "genotype_processing",
            'qc_reports': "QC_reports",
            'interaction_results': "interaction_results",
            'fine_mapping_results': "fine_mapping_results",
            'advanced_plots': "plots/interactive",
            'large_data_cache': "cache/large_data"  # New: Cache for large datasets
        }
        
        for name, path in directories.items():
            full_path = os.path.join(self.results_dir, path)
            setattr(self, f"{name}_dir", full_path)
            Path(full_path).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"pipeline_{timestamp}.log")
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('QTLPipeline')
        self.logger.info("🚀 Enhanced QTL Analysis Pipeline Started")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        self.logger.info(f"🔧 Analysis mode: {self.config['analysis'].get('qtl_mode', 'cis')}")
        self.logger.info(f"🔍 Enhanced QC: {self.config.get('enhanced_qc', {}).get('enable', False)}")
        self.logger.info(f"🤝 Interaction analysis: {self.config.get('interaction_analysis', {}).get('enable', False)}")
        self.logger.info(f"🎯 Fine-mapping: {self.config.get('fine_mapping', {}).get('enable', False)}")
        self.logger.info(f"💾 Large data optimizations: {self.config.get('large_data', {}).get('force_plink', False)}")
        self.logger.info(f"🔄 Modular pipeline also available: python run_QTLPipeline.py --list")
        
    def map_qtl_type_to_config_key(self, qtl_type):
        """Map QTL analysis types to config file keys"""
        mapping = {
            'eqtl': 'expression',
            'pqtl': 'protein', 
            'sqtl': 'splicing'
        }
        return mapping.get(qtl_type, qtl_type)
        
    def run_pipeline(self):
        """Execute the complete analysis pipeline with resource monitoring"""
        try:
            # Monitor system resources
            if self.monitor_resources:
                self.monitor_system_resources()
            
            # Step 1: Comprehensive input validation
            self.logger.info("📋 Step 1: Validating inputs...")
            validation.validate_inputs(self.config)
            
            # Step 2: Enhanced Quality Control
            if self.config.get('enhanced_qc', {}).get('enable', False):
                self.logger.info("🔍 Step 2: Running enhanced QC...")
                qc_results = self.run_enhanced_qc()
                self.results['qc'] = qc_results
            
            # Step 2.5: Generate normalization comparison reports if enabled
            if self.config.get('enhanced_qc', {}).get('generate_normalization_plots', True):
                self.logger.info("📊 Generating normalization comparison reports...")
                try:
                    comparison = NormalizationComparison(self.config, self.results_dir)
                    # This will be called later during phenotype preparation, but we initialize here
                    self.logger.info("✅ Normalization comparison setup completed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Normalization comparison setup failed: {e}")
                    
            # Step 3: Prepare genotype data with enhanced processing
            self.logger.info("🧬 Step 3: Preparing genotype data...")
            genotype_file = qtl_analysis.prepare_genotypes(self.config, self.results_dir)
            
            # Alternative: Use modular genotype processing if available
            try:
                from scripts.utils.genotype_processing import process_genotypes
                self.logger.info("🔄 Using modular genotype processing...")
                genotype_file = process_genotypes(self.config)
            except ImportError:
                self.logger.info("🔧 Using standard genotype processing...")
                genotype_file = qtl_analysis.prepare_genotypes(self.config, self.results_dir)
            
            # Step 4: Run QTL analyses (cis/trans/both)
            self.logger.info("🔍 Step 4: Running QTL analyses...")
            qtl_results = self.run_qtl_analyses(genotype_file)
            self.results['qtl'] = qtl_results
            
            # Step 5: Run GWAS analysis if requested
            if self.config['analysis'].get('run_gwas', False):
                self.logger.info("📊 Step 5: Running GWAS analysis...")
                gwas_results = self.run_gwas_analysis(genotype_file)
                self.results['gwas'] = gwas_results
            
            # Step 6: Run advanced analyses
            self.logger.info("🔬 Step 6: Running advanced analyses...")
            advanced_results = self.run_advanced_analyses(genotype_file)
            self.results['advanced'] = advanced_results
            
            # Step 7: Generate comprehensive plots
            if self.config['plotting'].get('enabled', True):
                self.logger.info("📈 Step 7: Generating plots...")
                self.generate_plots()
            
            # Step 8: Generate detailed reports
            self.logger.info("📝 Step 8: Generating reports...")
            self.generate_reports()
            
            # Clean up temporary files
            self.cleanup_temp_files()
            
            # Calculate and log runtime
            runtime = datetime.now() - self.start_time
            self.logger.info(f"✅ Pipeline completed successfully in {runtime}")
            self.logger.info(f"💡 You can also run individual modules using: python run_QTLPipeline.py --list")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            # Clean up on failure
            self.cleanup_temp_files()
            raise
            
    def monitor_system_resources(self):
        """Monitor system resources throughout pipeline execution"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.logger.info(f"💻 System Resources:")
        self.logger.info(f"   Memory: {memory.percent:.1f}% used ({memory.available / (1024**3):.1f} GB available)")
        self.logger.info(f"   Disk: {disk.percent:.1f}% used ({disk.free / (1024**3):.1f} GB free)")
        
        # Warn if resources are low
        if memory.percent > 90:
            self.logger.warning("⚠️  High memory usage detected!")
        if disk.percent > 90:
            self.logger.warning("⚠️  Low disk space detected!")
    
    def cleanup_temp_files(self):
        """Clean up temporary files to free disk space"""
        self.logger.info("🧹 Cleaning up temporary files...")
        
        temp_dirs = [self.temp_dir, self.large_data_cache_dir]
        
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
            qc_processor = EnhancedQC(self.config)
            
            # Get QTL types for QC
            qtl_types = self.get_qtl_types()
            
            # Build phenotype_files dictionary for backward compatibility
            phenotype_files = {}
            for qtl_type in qtl_types:
                config_key = self.map_qtl_type_to_config_key(qtl_type)
                phenotype_files[qtl_type] = self.config['input_files'].get(config_key)
            
            # Run comprehensive QC - FIXED: Pass phenotype_files dictionary
            qc_results = qc_processor.run_comprehensive_qc(
                self.config['input_files']['genotypes'],
                phenotype_files,  # FIX: Pass phenotype_files dict for backward compatibility
                self.results_dir
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
        
        for qtl_type in qtl_types:
            self.logger.info(f"🔬 Running {qtl_type.upper()} analysis ({qtl_mode})...")
            results[qtl_type] = {}
            
            try:
                # Get phenotype file with proper mapping
                config_key = self.map_qtl_type_to_config_key(qtl_type)
                phenotype_file = self.config['input_files'].get(config_key)
                
                if not phenotype_file or not os.path.exists(phenotype_file):
                    self.logger.error(f"❌ Phenotype file not found for {qtl_type}: {phenotype_file}")
                    continue
                
                # Run cis-QTL if requested
                if qtl_mode in ['cis', 'both']:
                    cis_result = qtl_analysis.run_cis_analysis(
                        self.config, genotype_file, qtl_type, self.qtl_results_dir
                    )
                    results[qtl_type]['cis'] = cis_result
                    status_msg = "✅" if cis_result['status'] == 'completed' else "❌"
                    self.logger.info(f"{status_msg} {qtl_type.upper()} cis: {cis_result.get('significant_count', 0)} significant")
                
                # Run trans-QTL if requested  
                if qtl_mode in ['trans', 'both']:
                    trans_result = qtl_analysis.run_trans_analysis(
                        self.config, genotype_file, qtl_type, self.qtl_results_dir
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
            result = gwas_analysis.run_gwas_analysis(
                self.config, genotype_file, self.gwas_results_dir
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
        
        # Interaction analysis
        if self.config.get('interaction_analysis', {}).get('enable', False) and InteractionAnalysis:
            try:
                self.logger.info("🤝 Running interaction analysis...")
                interaction_analyzer = InteractionAnalysis(self.config)
                
                # Run interaction analysis for each QTL type with proper mapping
                for qtl_type in self.get_qtl_types():
                    config_key = self.map_qtl_type_to_config_key(qtl_type)
                    phenotype_file = self.config['input_files'].get(config_key)
                    if phenotype_file and os.path.exists(phenotype_file):
                        # FIXED: Pass qtl_type parameter to interaction analysis
                        interaction_results = interaction_analyzer.run_interaction_analysis(
                            genotype_file, phenotype_file, 
                            self.config['input_files']['covariates'],
                            self.interaction_results_dir,
                            qtl_type  # FIX: Add qtl_type parameter
                        )
                        advanced_results[f'interaction_{qtl_type}'] = interaction_results
                        
            except Exception as e:
                self.logger.error(f"❌ Interaction analysis failed: {e}")
                advanced_results['interaction'] = {'status': 'failed', 'error': str(e)}
        
        # Fine-mapping
        if self.config.get('fine_mapping', {}).get('enable', False) and FineMapping:
            try:
                self.logger.info("🎯 Running fine-mapping...")
                fine_mapper = FineMapping(self.config)
                
                # Run fine-mapping for significant QTL results
                if 'qtl' in self.results:
                    for qtl_type, result in self.results['qtl'].items():
                        if 'cis' in result and result['cis']['status'] == 'completed':
                            result_file = result['cis'].get('result_file')
                            if result_file and os.path.exists(result_file):
                                # FIXED: Pass qtl_type parameter to fine-mapping
                                finemap_results = fine_mapper.run_fine_mapping(
                                    result_file, genotype_file, self.fine_mapping_results_dir, qtl_type
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
            
    def generate_plots(self):
        """Generate all requested plots"""
        # Basic plotting
        plotter = plotting.QTLPlotter(self.config, self.results, self.plots_dir)
        
        # Generate QTL plots
        if 'qtl' in self.results:
            for qtl_type, result in self.results['qtl'].items():
                if 'cis' in result and result['cis']['status'] == 'completed':
                    plotter.create_cis_plots(qtl_type, result['cis'])
                if 'trans' in result and result['trans']['status'] == 'completed':
                    plotter.create_trans_plots(qtl_type, result['trans'])
        
        # Generate GWAS plots
        if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
            plotter.create_gwas_plots(self.results['gwas'])
            
        # Generate summary plots
        plotter.create_summary_plots()
        
        # Advanced plotting with interactive features
        if self.config['plotting'].get('plot_types', []):
            if 'interactive' in self.config['plotting']['plot_types']:
                try:
                    advanced_plotter = AdvancedPlotter(self.config, self.results_dir)
                    
                    # Create interactive plots for significant results
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
                    
                    # Create multi-panel summary
                    advanced_plotter.create_multi_panel_figure(self.results, "comprehensive_summary")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Advanced plotting failed: {e}")
        
    def generate_reports(self):
        """Generate comprehensive reports"""
        report_data = {
            'results': self.results,
            'config': self.config,
            'runtime': str(datetime.now() - self.start_time),
            'timestamp': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'results_dir': self.results_dir
        }
        
        # Generate HTML report
        if self.config['output'].get('generate_report', True):
            report_file = os.path.join(self.reports_dir, "analysis_report.html")
            report_generator.generate_html_report(report_data, report_file)
            self.logger.info(f"📄 HTML report generated: {report_file}")
        
        # Generate summary report
        summary_file = os.path.join(self.results_dir, "pipeline_summary.txt")
        report_generator.generate_summary_report(report_data, summary_file)
        self.logger.info(f"📋 Summary report generated: {summary_file}")
        
        # Save results metadata
        metadata_file = os.path.join(self.results_dir, "results_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"💾 Results metadata saved: {metadata_file}")
        
        # Generate enhanced QC report if available
        if 'qc' in self.results and self.results['qc']:
            qc_report_file = os.path.join(self.qc_reports_dir, "enhanced_qc_report.html")
            # This would call a method from EnhancedQC to generate a detailed report
            self.logger.info(f"🔍 Enhanced QC report available in: {self.qc_reports_dir}")


if __name__ == "__main__":
    """Main entry point for the pipeline"""
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