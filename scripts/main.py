#!/usr/bin/env python3
"""
Enhanced Main QTL Analysis Pipeline with comprehensive genotype processing
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

# Import utility modules
from utils import validation, plotting, qtl_analysis, gwas_analysis, report_generator

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
            if 'results_dir' not in config:
                raise ValueError("'results_dir' must be specified in the config file")
            
            if 'input_files' not in config:
                raise ValueError("'input_files' must be specified in the config file")
            
            mandatory_inputs = ['genotypes', 'covariates', 'annotations']
            for input_file in mandatory_inputs:
                if input_file not in config['input_files']:
                    raise ValueError(f"'{input_file}' must be specified in 'input_files'")
            
            # Set defaults for genotype processing
            config.setdefault('genotype_processing', {})
            config['genotype_processing'].setdefault('auto_detect_format', True)
            config['genotype_processing'].setdefault('filter_variants', True)
            config['genotype_processing'].setdefault('normalize_chromosomes', True)
            config['genotype_processing'].setdefault('handle_multiallelic', True)
            config['genotype_processing'].setdefault('remove_phasing', True)
            
            # Set defaults for other sections
            config.setdefault('analysis', {})
            config['analysis'].setdefault('qtl_types', 'all')
            config['analysis'].setdefault('run_gwas', False)
            
            config.setdefault('qtl', {})
            config['qtl'].setdefault('cis_window', 1000000)
            config['qtl'].setdefault('permutations', 1000)
            config['qtl'].setdefault('fdr_threshold', 0.05)
            
            config.setdefault('plotting', {})
            config['plotting'].setdefault('enabled', True)
            
            config.setdefault('output', {})
            config['output'].setdefault('generate_report', True)
            
            # Set default tool paths
            config.setdefault('paths', {})
            config['paths'].setdefault('qtltools', 'qtltools')
            config['paths'].setdefault('bcftools', 'bcftools')
            config['paths'].setdefault('bgzip', 'bgzip')
            config['paths'].setdefault('tabix', 'tabix')
            config['paths'].setdefault('python', 'python3')
            config['paths'].setdefault('plink', 'plink')
            
            return config
            
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            raise
            
    def setup_directories(self):
        """Create directory structure inside results_dir"""
        self.results_dir = self.config['results_dir']
        directories = {
            'logs': os.path.join(self.results_dir, "logs"),
            'temp': os.path.join(self.results_dir, "temp"),
            'plots': os.path.join(self.results_dir, "plots"),
            'qtl_results': os.path.join(self.results_dir, "qtl_results"),
            'gwas_results': os.path.join(self.results_dir, "gwas_results"),
            'reports': os.path.join(self.results_dir, "reports"),
            'genotype_processing': os.path.join(self.results_dir, "genotype_processing")
        }
        
        for name, path in directories.items():
            setattr(self, f"{name}_dir", path)
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"pipeline_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('QTLPipeline')
        self.logger.info(f"🚀 Pipeline started at {self.start_time}")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        
    def run_pipeline(self):
        """Execute the complete analysis pipeline"""
        try:
            # Step 1: Validate inputs
            self.logger.info("📋 Step 1: Validating inputs...")
            validation.validate_inputs(self.config)
            
            # Step 2: Prepare genotype data with pre-processing
            self.logger.info("🧬 Step 2: Preparing genotype data with pre-processing...")
            vcf_gz = qtl_analysis.prepare_genotypes(self.config, self.results_dir)
            
            # Step 3: Run QTL analyses
            self.logger.info("🔍 Step 3: Running QTL analyses...")
            qtl_results = self.run_qtl_analyses(vcf_gz)
            self.results['qtl'] = qtl_results
            
            # Step 4: Run GWAS analysis if requested
            if self.config['analysis'].get('run_gwas', False):
                self.logger.info("📊 Step 4: Running GWAS analysis...")
                gwas_results = self.run_gwas_analysis(vcf_gz)
                self.results['gwas'] = gwas_results
            
            # Step 5: Generate plots
            if self.config['plotting'].get('enabled', True):
                self.logger.info("📈 Step 5: Generating plots...")
                self.generate_plots()
            
            # Step 6: Generate reports
            self.logger.info("📝 Step 6: Generating reports...")
            self.generate_reports()
            
            # Calculate runtime
            runtime = datetime.now() - self.start_time
            self.logger.info(f"✅ Pipeline completed successfully in {runtime}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise
            
    def run_qtl_analyses(self, vcf_gz):
        """Run all specified QTL analyses"""
        qtl_types = self.get_qtl_types()
        results = {}
        
        for qtl_type in qtl_types:
            self.logger.info(f"🔬 Running {qtl_type.upper()} analysis...")
            try:
                result = qtl_analysis.run_qtl_analysis(
                    self.config, vcf_gz, qtl_type, self.qtl_results_dir
                )
                results[qtl_type] = {
                    'status': 'completed',
                    'result_file': result['result_file'],
                    'nominals_file': result['nominals_file'],
                    'significant_count': result['significant_count']
                }
                self.logger.info(f"✅ {qtl_type.upper()} completed: {result['significant_count']} significant associations")
                
            except Exception as e:
                self.logger.error(f"❌ {qtl_type.upper()} failed: {e}")
                results[qtl_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
        return results
        
    def run_gwas_analysis(self, vcf_gz):
        """Run GWAS analysis if requested"""
        try:
            result = gwas_analysis.run_gwas_analysis(
                self.config, vcf_gz, self.gwas_results_dir
            )
            return {
                'status': 'completed',
                'result_file': result['result_file'],
                'significant_count': result['significant_count'],
                'method': result['method']
            }
        except Exception as e:
            self.logger.error(f"❌ GWAS analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def get_qtl_types(self):
        """Parse and validate QTL types"""
        config_types = self.config['analysis']['qtl_types']
        
        if config_types == 'all':
            return ['eqtl', 'pqtl', 'sqtl']
        elif isinstance(config_types, str):
            return [t.strip() for t in config_types.split(',')]
        elif isinstance(config_types, list):
            return config_types
        else:
            raise ValueError(f"Invalid qtl_types configuration: {config_types}")
            
    def generate_plots(self):
        """Generate all requested plots"""
        plotter = plotting.QTLPlotter(self.config, self.results, self.plots_dir)
        
        # Generate QTL plots
        if 'qtl' in self.results:
            for qtl_type, result in self.results['qtl'].items():
                if result['status'] == 'completed':
                    plotter.create_qtl_plots(qtl_type, result)
        
        # Generate GWAS plots
        if 'gwas' in self.results and self.results['gwas']['status'] == 'completed':
            plotter.create_gwas_plots(self.results['gwas'])
            
        # Generate summary plots
        plotter.create_summary_plots()
        
    def generate_reports(self):
        """Generate comprehensive reports"""
        report_data = {
            'results': self.results,
            'config': self.config,
            'runtime': str(datetime.now() - self.start_time),
            'timestamp': self.start_time.strftime("%Y-%m-%d %H:%M:%S")
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
            json.dump(self.results, f, indent=2)
        self.logger.info(f"💾 Results metadata saved: {metadata_file}")