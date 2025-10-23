#!/usr/bin/env python3
"""
Enhanced QTL Pipeline - Environment Validation Script
Comprehensive validation of all dependencies, tools, and system requirements

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import os
import sys
import subprocess
import importlib
import shutil
import platform
import psutil
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for validation script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

class EnvironmentValidator:
    def __init__(self):
        self.system_info = {}
        self.validation_results = {
            'system': {'status': 'unknown', 'issues': [], 'suggestions': []},
            'python_packages': {'status': 'unknown', 'issues': [], 'suggestions': []},
            'bioinformatics_tools': {'status': 'unknown', 'issues': [], 'suggestions': []},
            'file_permissions': {'status': 'unknown', 'issues': [], 'suggestions': []},
            'project_structure': {'status': 'unknown', 'issues': [], 'suggestions': []},
            'overall': {'status': 'unknown'}
        }
        
    def collect_system_info(self):
        """Collect comprehensive system information"""
        logging.info("🖥️  Collecting system information...")
        
        try:
            self.system_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 1),
                'disk_usage': psutil.disk_usage('/')
            }
            
            logging.info(f"   • System: {self.system_info['platform']} {self.system_info['platform_release']}")
            logging.info(f"   • Architecture: {self.system_info['architecture'][0]}")
            logging.info(f"   • Python: {self.system_info['python_version']}")
            logging.info(f"   • Memory: {self.system_info['total_memory_gb']} GB total, {self.system_info['available_memory_gb']} GB available")
            logging.info(f"   • Disk: {self.system_info['disk_usage'].free // (1024**3)} GB free")
            
        except Exception as e:
            logging.warning(f"   ⚠️ Could not collect all system information: {e}")
    
    def validate_system_requirements(self):
        """Validate system requirements"""
        logging.info("\n🔍 Validating system requirements...")
        issues = []
        suggestions = []
        
        # Check Python version
        python_version = tuple(map(int, self.system_info['python_version'].split('.')[:2]))
        if python_version < (3, 7):
            issues.append(f"Python version {self.system_info['python_version']} is below minimum required 3.7")
            suggestions.append("Upgrade to Python 3.7 or higher")
        else:
            logging.info("   ✅ Python version: OK")
        
        # Check memory
        if self.system_info['total_memory_gb'] < 8:
            issues.append(f"System memory ({self.system_info['total_memory_gb']} GB) is below recommended 8 GB")
            suggestions.append("Consider running on a system with more RAM or use smaller datasets")
        else:
            logging.info("   ✅ System memory: OK")
        
        # Check disk space
        free_disk_gb = self.system_info['disk_usage'].free // (1024**3)
        if free_disk_gb < 10:
            issues.append(f"Low disk space ({free_disk_gb} GB free)")
            suggestions.append("Free up disk space before running large analyses")
        else:
            logging.info(f"   ✅ Disk space: {free_disk_gb} GB free")
        
        # Check operating system
        if self.system_info['platform'] not in ['Linux', 'Darwin']:
            issues.append(f"Untested operating system: {self.system_info['platform']}")
            suggestions.append("Pipeline is tested on Linux and macOS. Windows may require WSL")
        else:
            logging.info("   ✅ Operating system: Supported")
        
        # Update validation results
        self.validation_results['system']['issues'] = issues
        self.validation_results['system']['suggestions'] = suggestions
        self.validation_results['system']['status'] = 'pass' if not issues else 'fail'
    
    def validate_python_packages(self):
        """Validate all required Python packages"""
        logging.info("\n🐍 Validating Python packages...")
        
        required_packages = {
            'pandas': '1.3.0',
            'numpy': '1.21.0',
            'scipy': '1.7.0',
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'PyYAML': '6.0',
            'scikit-learn': '1.0.0',
            'statsmodels': '0.13.0',
            'plotly': '5.0.0'
        }
        
        optional_packages = {
            'rpy2': '3.5.0',      # For R integration
            'pysam': '0.19.0',    # For advanced VCF handling
            'psutil': ''          # For system monitoring (already imported)
        }
        
        issues = []
        suggestions = []
        missing_optional = []
        
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package)
                if min_version:
                    actual_version = getattr(module, '__version__', 'unknown')
                    if actual_version != 'unknown':
                        # Simple version comparison
                        actual_parts = list(map(int, actual_version.split('.')[:3]))
                        min_parts = list(map(int, min_version.split('.')[:3]))
                        
                        if actual_parts < min_parts:
                            issues.append(f"{package} version {actual_version} is below required {min_version}")
                            suggestions.append(f"Upgrade {package}: pip install --upgrade {package}")
                        else:
                            logging.info(f"   ✅ {package}: {actual_version} (>= {min_version})")
                    else:
                        logging.info(f"   ✅ {package}: installed (version unknown)")
                else:
                    logging.info(f"   ✅ {package}: installed")
                    
            except ImportError:
                issues.append(f"Required package not found: {package}")
                suggestions.append(f"Install {package}: pip install {package}>={min_version}")
        
        # Check optional packages
        for package, min_version in optional_packages.items():
            try:
                importlib.import_module(package)
                logging.info(f"   ✅ {package}: installed (optional)")
            except ImportError:
                missing_optional.append(package)
        
        if missing_optional:
            logging.info(f"   ⚠️  Optional packages not installed: {', '.join(missing_optional)}")
            suggestions.append(f"Install optional packages: pip install {' '.join(missing_optional)}")
        
        self.validation_results['python_packages']['issues'] = issues
        self.validation_results['python_packages']['suggestions'] = suggestions
        self.validation_results['python_packages']['status'] = 'pass' if not issues else 'fail'
    
    def validate_bioinformatics_tools(self):
        """Validate required bioinformatics tools"""
        logging.info("\n🔧 Validating bioinformatics tools...")
        
        required_tools = {
            'qtltools': {
                'test_cmd': 'qtltools --version',
                'install_url': 'https://qtltools.github.io/qtltools/',
                'description': 'QTL mapping software'
            },
            'bcftools': {
                'test_cmd': 'bcftools --version',
                'install_url': 'https://www.htslib.org/',
                'description': 'VCF/BCF manipulation'
            },
            'bgzip': {
                'test_cmd': 'bgzip --version 2>/dev/null || which bgzip',
                'install_url': 'https://www.htslib.org/',
                'description': 'File compression'
            },
            'tabix': {
                'test_cmd': 'tabix --version 2>/dev/null || which tabix',
                'install_url': 'https://www.htslib.org/',
                'description': 'File indexing'
            }
        }
        
        optional_tools = {
            'plink': {
                'test_cmd': 'plink --version',
                'install_url': 'https://www.cog-genomics.org/plink/',
                'description': 'GWAS analysis'
            },
            'R': {
                'test_cmd': 'R --version',
                'install_url': 'https://www.r-project.org/',
                'description': 'Statistical computing'
            }
        }
        
        issues = []
        suggestions = []
        missing_optional = []
        
        # Check required tools
        for tool, info in required_tools.items():
            try:
                result = subprocess.run(
                    info['test_cmd'], 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    executable='/bin/bash'
                )
                
                if result.returncode == 0:
                    # Try to extract version
                    version_line = result.stdout.split('\n')[0] if result.stdout else 'found'
                    logging.info(f"   ✅ {tool}: {version_line}")
                else:
                    issues.append(f"Required tool not found: {tool} ({info['description']})")
                    suggestions.append(f"Install {tool} from: {info['install_url']}")
                    
            except Exception as e:
                issues.append(f"Error checking {tool}: {e}")
                suggestions.append(f"Install {tool} from: {info['install_url']}")
        
        # Check optional tools
        for tool, info in optional_tools.items():
            try:
                result = subprocess.run(
                    info['test_cmd'], 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    executable='/bin/bash'
                )
                
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0] if result.stdout else 'found'
                    logging.info(f"   ✅ {tool}: {version_line} (optional)")
                else:
                    missing_optional.append(tool)
                    
            except Exception:
                missing_optional.append(tool)
        
        if missing_optional:
            logging.info(f"   ⚠️  Optional tools not found: {', '.join(missing_optional)}")
            for tool in missing_optional:
                suggestions.append(f"Optional: Install {tool} from: {optional_tools[tool]['install_url']}")
        
        self.validation_results['bioinformatics_tools']['issues'] = issues
        self.validation_results['bioinformatics_tools']['suggestions'] = suggestions
        self.validation_results['bioinformatics_tools']['status'] = 'pass' if not issues else 'fail'
    
    def validate_file_permissions(self):
        """Validate file permissions and directory access"""
        logging.info("\n📁 Validating file permissions...")
        
        issues = []
        suggestions = []
        
        # Check current directory permissions
        current_dir = Path.cwd()
        try:
            test_file = current_dir / '.permission_test'
            test_file.touch()
            test_file.unlink()
            logging.info("   ✅ Current directory: writable")
        except Exception:
            issues.append("Current directory is not writable")
            suggestions.append("Run from a directory with write permissions")
        
        # Check if we can create results directory
        results_dir = current_dir / 'validation_test_results'
        try:
            results_dir.mkdir(exist_ok=True)
            (results_dir / 'test.txt').write_text('test')
            shutil.rmtree(results_dir)
            logging.info("   ✅ Directory creation: OK")
        except Exception:
            issues.append("Cannot create directories or write files")
            suggestions.append("Check directory permissions and disk space")
        
        # Check if we can execute shell commands
        try:
            result = subprocess.run('echo "test"', shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("   ✅ Shell execution: OK")
            else:
                issues.append("Shell command execution failed")
                suggestions.append("Check system permissions and shell availability")
        except Exception:
            issues.append("Cannot execute shell commands")
            suggestions.append("System may not support required shell operations")
        
        self.validation_results['file_permissions']['issues'] = issues
        self.validation_results['file_permissions']['suggestions'] = suggestions
        self.validation_results['file_permissions']['status'] = 'pass' if not issues else 'fail'
    
    def validate_project_structure(self):
        """Validate QTL_ANALYSIS project structure"""
        logging.info("\n📋 Validating project structure...")
        
        issues = []
        suggestions = []
        
        # Check main directory structure
        required_dirs = [
            'config',
            'data', 
            'scripts',
            'scripts/analysis',
            'scripts/utils'
        ]
        
        required_files = [
            'run_QTLPipeline.py',
            'scripts/main.py',
            'config/config.yaml',
            'requirements.txt'
        ]
        
        required_scripts = [
            'scripts/utils/genotype_processing.py',
            'scripts/utils/qtl_analysis.py',
            'scripts/utils/gwas_analysis.py',
            'scripts/utils/enhanced_qc.py',
            'scripts/utils/validation.py',
            'scripts/utils/report_generator.py',
            'scripts/analysis/fine_mapping.py',
            'scripts/analysis/interaction_analysis.py'
        ]
        
        # Check directories
        for directory in required_dirs:
            if not os.path.isdir(directory):
                issues.append(f"Missing directory: {directory}")
                suggestions.append(f"Create directory: {directory}")
            else:
                logging.info(f"   ✅ {directory}/: found")
        
        # Check main files
        for file in required_files:
            if not os.path.exists(file):
                issues.append(f"Missing file: {file}")
                suggestions.append(f"Ensure {file} exists in project root")
            else:
                logging.info(f"   ✅ {file}: found")
        
        # Check script files
        for script in required_scripts:
            if not os.path.exists(script):
                issues.append(f"Missing script: {script}")
                suggestions.append(f"Ensure {script} exists in project structure")
            else:
                logging.info(f"   ✅ {script}: found")
        
        # Check data files (warn if missing, but don't fail)
        expected_data_files = [
            'data/genotypes.vcf',
            'data/expression.txt', 
            'data/covariates.txt',
            'data/annotations.bed'
        ]
        
        missing_data = []
        for data_file in expected_data_files:
            if not os.path.exists(data_file):
                missing_data.append(data_file)
            else:
                logging.info(f"   ✅ {data_file}: found")
        
        if missing_data:
            logging.info(f"   ⚠️  Expected data files not found: {', '.join(missing_data)}")
            suggestions.append("Run create_sample_data.py to generate test data")
        
        self.validation_results['project_structure']['issues'] = issues
        self.validation_results['project_structure']['suggestions'] = suggestions
        self.validation_results['project_structure']['status'] = 'pass' if not issues else 'fail'
        
        return len(issues) == 0
    
    def run_comprehensive_validation(self):
        """Run all validation checks"""
        logging.info("=" * 70)
        logging.info("🔬 QTL ANALYSIS PIPELINE - COMPREHENSIVE ENVIRONMENT VALIDATION")
        logging.info("=" * 70)
        logging.info("Author: Dr. Vijay Singh (vijay.s.gautam@gmail.com)")
        logging.info("=" * 70)
        
        # Collect system information
        self.collect_system_info()
        
        # Run all validations
        self.validate_system_requirements()
        self.validate_python_packages()
        self.validate_bioinformatics_tools()
        self.validate_file_permissions()
        structure_ok = self.validate_project_structure()
        
        # Determine overall status
        all_checks = [
            self.validation_results['system']['status'],
            self.validation_results['python_packages']['status'],
            self.validation_results['bioinformatics_tools']['status'],
            self.validation_results['file_permissions']['status'],
            self.validation_results['project_structure']['status']
        ]
        
        if all(status == 'pass' for status in all_checks) and structure_ok:
            self.validation_results['overall']['status'] = 'pass'
        else:
            self.validation_results['overall']['status'] = 'fail'
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        logging.info("\n" + "=" * 70)
        logging.info("📊 VALIDATION REPORT SUMMARY")
        logging.info("=" * 70)
        
        # Overall status
        if self.validation_results['overall']['status'] == 'pass':
            logging.info("🎉 ENVIRONMENT VALIDATION: PASSED ✅")
            logging.info("   Your QTL analysis pipeline is ready to use!")
        else:
            logging.info("❌ ENVIRONMENT VALIDATION: FAILED")
            logging.info("   Please address the issues below before running the pipeline.")
        
        # Detailed issues and suggestions
        all_issues = []
        all_suggestions = []
        
        for category, results in self.validation_results.items():
            if category == 'overall':
                continue
                
            if results['issues']:
                logging.info(f"\n📋 {category.upper().replace('_', ' ')} ISSUES:")
                for issue in results['issues']:
                    logging.info(f"   • {issue}")
                    all_issues.append(issue)
                
                if results['suggestions']:
                    logging.info(f"   💡 SUGGESTIONS:")
                    for suggestion in results['suggestions']:
                        logging.info(f"     - {suggestion}")
                        all_suggestions.append(suggestion)
        
        if not all_issues:
            logging.info("\n✅ No critical issues found!")
        
        # Next steps
        logging.info("\n🎯 NEXT STEPS:")
        if self.validation_results['overall']['status'] == 'pass':
            logging.info("   1. Review configuration: config/config.yaml")
            logging.info("   2. Run quick test: python quick_test.py")
            logging.info("   3. Run full analysis: python run_QTLPipeline.py --config config/config.yaml")
        else:
            logging.info("   1. Address all critical issues listed above")
            logging.info("   2. Re-run validation: python validate_environment.py")
            logging.info("   3. Contact: vijay.s.gautam@gmail.com for support")
        
        logging.info("\n" + "=" * 70)
        
        return self.validation_results['overall']['status'] == 'pass'

def main():
    """Main validation function"""
    validator = EnvironmentValidator()
    validator.run_comprehensive_validation()
    success = validator.generate_report()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    setup_logging()
    main()