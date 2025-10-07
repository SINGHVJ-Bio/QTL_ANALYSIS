#!/usr/bin/env python3
"""
Enhanced Main QTL Analysis Pipeline with cis/trans capabilities
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
            'qc_reports': "QC_reports"
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
        self.logger.info("🚀 QTL Analysis Pipeline Started")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        self.logger.info(f"🔧 Analysis mode: {self.config['analysis'].get('qtl_mode', 'cis')}")
        
    def run_pipeline(self):
        """Execute the complete analysis pipeline"""
        try:
            # Step 1: Comprehensive input validation
            self.logger.info("📋 Step 1: Validating inputs...")
            validation.validate_inputs(self.config)
            
            # Step 2: Prepare genotype data with enhanced processing
            self.logger.info("🧬 Step 2: Preparing genotype data...")
            vcf_gz = qtl_analysis.prepare_genotypes(self.config, self.results_dir)
            
            # Step 3: Run QTL analyses (cis/trans/both)
            self.logger.info("🔍 Step 3: Running QTL analyses...")
            qtl_results = self.run_qtl_analyses(vcf_gz)
            self.results['qtl'] = qtl_results
            
            # Step 4: Run GWAS analysis if requested
            if self.config['analysis'].get('run_gwas', False):
                self.logger.info("📊 Step 4: Running GWAS analysis...")
                gwas_results = self.run_gwas_analysis(vcf_gz)
                self.results['gwas'] = gwas_results
            
            # Step 5: Generate comprehensive plots
            if self.config['plotting'].get('enabled', True):
                self.logger.info("📈 Step 5: Generating plots...")
                self.generate_plots()
            
            # Step 6: Generate detailed reports
            self.logger.info("📝 Step 6: Generating reports...")
            self.generate_reports()
            
            # Calculate and log runtime
            runtime = datetime.now() - self.start_time
            self.logger.info(f"✅ Pipeline completed successfully in {runtime}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise
            
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
                    results[qtl_type]['cis'] = {
                        'status': 'completed',
                        'result_file': cis_result['result_file'],
                        'nominals_file': cis_result['nominals_file'],
                        'significant_count': cis_result['significant_count']
                    }
                    self.logger.info(f"✅ {qtl_type.upper()} cis completed: {cis_result['significant_count']} significant")
                
                # Run trans-QTL if requested
                if qtl_mode in ['trans', 'both']:
                    trans_result = qtl_analysis.run_trans_analysis(
                        self.config, vcf_gz, qtl_type, self.qtl_results_dir
                    )
                    results[qtl_type]['trans'] = {
                        'status': 'completed', 
                        'result_file': trans_result['result_file'],
                        'significant_count': trans_result['significant_count']
                    }
                    self.logger.info(f"✅ {qtl_type.upper()} trans completed: {trans_result['significant_count']} significant")
                    
            except Exception as e:
                self.logger.error(f"❌ {qtl_type.upper()} failed: {e}")
                if 'cis' not in results[qtl_type]:
                    results[qtl_type]['cis'] = {'status': 'failed', 'error': str(e)}
                if 'trans' not in results[qtl_type] and qtl_mode in ['trans', 'both']:
                    results[qtl_type]['trans'] = {'status': 'failed', 'error': str(e)}
                
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