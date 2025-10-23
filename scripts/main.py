#!/usr/bin/env python3
"""
Enhanced Main QTL Analysis Pipeline with cis/trans capabilities - Enhanced Version
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com

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
warnings.filterwarnings('ignore')

# Import utility modules
from utils import validation, plotting, qtl_analysis, gwas_analysis, report_generator
from utils.enhanced_qc import EnhancedQC
from utils.advanced_plotting import AdvancedPlotter

# Import analysis modules
try:
    from analysis.interaction_analysis import InteractionAnalysis
    from analysis.fine_mapping import FineMapping
except ImportError:
    InteractionAnalysis = None
    FineMapping = None

logger = logging.getLogger('QTLPipeline')

class QTLPipeline:
    def __init__(self, config_file=None):
        self.start_time = datetime.now()
        self.config = self.load_config(config_file)
        self.setup_directories()
        self.setup_logging()
        self.results = {}
        
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
            
            # Set comprehensive defaults
            config.setdefault('genotype_processing', {})
            processing_defaults = {
                'auto_detect_format': True,
                'filter_variants': True,
                'normalize_chromosomes': True,
                'handle_multiallelic': True,
                'remove_phasing': True,
                'min_maf': 0.01,
                'min_call_rate': 0.95
            }
            for key, value in processing_defaults.items():
                config['genotype_processing'].setdefault(key, value)
            
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
                'min_call_rate': 0.95
            }
            for key, value in qtl_defaults.items():
                config['qtl'].setdefault(key, value)
            
            # Enhanced features defaults
            config.setdefault('enhanced_qc', {'enable': True})
            config.setdefault('interaction_analysis', {'enable': False})
            config.setdefault('fine_mapping', {'enable': False})
            
            # Other section defaults
            config.setdefault('plotting', {'enabled': True})
            config.setdefault('output', {'generate_report': True})
            config.setdefault('qc', {'check_sample_concordance': True})
            config.setdefault('performance', {'num_threads': 4})
            
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
            'advanced_plots': "plots/interactive"
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
        
    def run_pipeline(self):
        """Execute the complete analysis pipeline"""
        try:
            # Step 1: Comprehensive input validation
            self.logger.info("📋 Step 1: Validating inputs...")
            validation.validate_inputs(self.config)
            
            # Step 2: Enhanced Quality Control
            if self.config.get('enhanced_qc', {}).get('enable', False):
                self.logger.info("🔍 Step 2: Running enhanced QC...")
                qc_results = self.run_enhanced_qc()
                self.results['qc'] = qc_results
            
            # Step 3: Prepare genotype data with enhanced processing
            self.logger.info("🧬 Step 3: Preparing genotype data...")
            vcf_gz = qtl_analysis.prepare_genotypes(self.config, self.results_dir)
            
            # Step 4: Run QTL analyses (cis/trans/both)
            self.logger.info("🔍 Step 4: Running QTL analyses...")
            qtl_results = self.run_qtl_analyses(vcf_gz)
            self.results['qtl'] = qtl_results
            
            # Step 5: Run GWAS analysis if requested
            if self.config['analysis'].get('run_gwas', False):
                self.logger.info("📊 Step 5: Running GWAS analysis...")
                gwas_results = self.run_gwas_analysis(vcf_gz)
                self.results['gwas'] = gwas_results
            
            # Step 6: Run advanced analyses
            self.logger.info("🔬 Step 6: Running advanced analyses...")
            advanced_results = self.run_advanced_analyses(vcf_gz)
            self.results['advanced'] = advanced_results
            
            # Step 7: Generate comprehensive plots
            if self.config['plotting'].get('enabled', True):
                self.logger.info("📈 Step 7: Generating plots...")
                self.generate_plots()
            
            # Step 8: Generate detailed reports
            self.logger.info("📝 Step 8: Generating reports...")
            self.generate_reports()
            
            # Calculate and log runtime
            runtime = datetime.now() - self.start_time
            self.logger.info(f"✅ Pipeline completed successfully in {runtime}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise
            
    def run_enhanced_qc(self):
        """Run enhanced quality control"""
        try:
            qc_processor = EnhancedQC(self.config)
            
            # Get phenotype files
            phenotype_files = {}
            qtl_types = self.get_qtl_types()
            for qtl_type in qtl_types:
                phenotype_files[qtl_type] = self.config['input_files'].get(qtl_type)
            
            # Run comprehensive QC
            qc_results = qc_processor.run_comprehensive_qc(
                self.config['input_files']['genotypes'],
                phenotype_files,
                self.results_dir
            )
            
            return qc_results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced QC failed: {e}")
            return {}
            
    def run_qtl_analyses(self, vcf_gz):
        """Run QTL analyses based on configuration"""
        qtl_types = self.get_qtl_types()
        qtl_mode = self.config['analysis'].get('qtl_mode', 'cis')
        results = {}
        
        for qtl_type in qtl_types:
            self.logger.info(f"🔬 Running {qtl_type.upper()} analysis ({qtl_mode})...")
            results[qtl_type] = {}
            
            try:
                # Run cis-QTL if requested
                if qtl_mode in ['cis', 'both']:
                    cis_result = qtl_analysis.run_cis_analysis(
                        self.config, vcf_gz, qtl_type, self.qtl_results_dir
                    )
                    results[qtl_type]['cis'] = cis_result
                    status_msg = "✅" if cis_result['status'] == 'completed' else "❌"
                    self.logger.info(f"{status_msg} {qtl_type.upper()} cis: {cis_result.get('significant_count', 0)} significant")
                
                # Run trans-QTL if requested  
                if qtl_mode in ['trans', 'both']:
                    trans_result = qtl_analysis.run_trans_analysis(
                        self.config, vcf_gz, qtl_type, self.qtl_results_dir
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
        
    def run_gwas_analysis(self, vcf_gz):
        """Run GWAS analysis if requested"""
        try:
            result = gwas_analysis.run_gwas_analysis(
                self.config, vcf_gz, self.gwas_results_dir
            )
            return result
        except Exception as e:
            self.logger.error(f"❌ GWAS analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def run_advanced_analyses(self, vcf_gz):
        """Run advanced analyses like interaction and fine-mapping"""
        advanced_results = {}
        
        # Interaction analysis
        if self.config.get('interaction_analysis', {}).get('enable', False) and InteractionAnalysis:
            try:
                self.logger.info("🤝 Running interaction analysis...")
                interaction_analyzer = InteractionAnalysis(self.config)
                
                # Run interaction analysis for each QTL type
                for qtl_type in self.get_qtl_types():
                    phenotype_file = self.config['input_files'].get(qtl_type)
                    if phenotype_file and os.path.exists(phenotype_file):
                        interaction_results = interaction_analyzer.run_interaction_analysis(
                            vcf_gz, phenotype_file, 
                            self.config['input_files']['covariates'],
                            self.interaction_results_dir
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
                                finemap_results = fine_mapper.run_fine_mapping(
                                    result_file, vcf_gz, self.fine_mapping_results_dir
                                )
                                advanced_results[f'fine_mapping_{qtl_type}_cis'] = finemap_results
                                
            except Exception as e:
                self.logger.error(f"❌ Fine-mapping failed: {e}")
                advanced_results['fine_mapping'] = {'status': 'failed', 'error': str(e)}
        
        return advanced_results
            
    def get_qtl_types(self):
        """Parse and validate QTL types from config"""
        config_types = self.config['analysis']['qtl_types']
        
        if config_types == 'all':
            available_types = []
            for qtl_type in ['eqtl', 'pqtl', 'sqtl']:
                if qtl_type in self.config['input_files'] and self.config['input_files'][qtl_type]:
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