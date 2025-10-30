# [file name]: scripts/utils/directory_manager.py
#!/usr/bin/env python3
"""
Central Directory Management for QTL Pipeline
Ensures consistent directory structure across all modules

Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

from pathlib import Path
import logging
import os
from typing import Dict, List, Optional, Union
import json

logger = logging.getLogger('QTLPipeline')

class DirectoryManager:
    """Manages directory creation and consistency across the pipeline"""
    
    # Define all directory types with clear purposes
    DIRECTORY_STRUCTURE = {
        # Core analysis output directories
        'processed_data': {
            'genotypes': 'Processed and quality-controlled genotype data',
            'expression': 'Processed and normalized expression data',
            'proteomics': 'Processed protein data', 
            'splicing': 'Processed splicing data',
            'quality_control': 'Quality control reports and metrics'
        },
        'analysis_results': {
            'qtl_mapping': 'QTL mapping results (cis/trans)',
            'fine_mapping': 'Fine-mapping results',
            'interaction_analysis': 'Covariate interaction results',
            'gwas_results': 'GWAS analysis results'
        },
        'visualization': {
            'summary_plots': 'Summary and overview plots',
            'interactive_plots': 'Interactive visualizations',
            'manhattan_plots': 'Manhattan plots',
            'qq_plots': 'Q-Q plots'
        },
        'reports': {
            'pipeline_reports': 'Pipeline execution reports',
            'qc_reports': 'Quality control reports',
            'analysis_reports': 'Analysis summary reports'
        },
        'system': {
            'logs': 'Pipeline execution logs',
            'temporary_files': 'Temporary processing files',
            'cache': 'Cached data for performance'
        },
        'comparison_analysis': {
            'normalization_comparison': 'Normalization method comparisons',
            'batch_correction': 'Batch correction results'
        }
    }
    
    def __init__(self, results_dir: Union[str, Path]):
        self.results_dir = Path(results_dir)
        self.created_dirs = set()
        self.directory_paths = {}
        
    def initialize_pipeline_directories(self) -> Dict[str, Path]:
        """Initialize only essential directories needed for pipeline startup"""
        essential_dirs = {
            'system_logs': self.results_dir / 'system' / 'logs',
            'system_temporary': self.results_dir / 'system' / 'temporary_files'
        }
        
        for name, path in essential_dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.add(str(path))
            self.directory_paths[name] = path
            
        logger.info(f"✅ Pipeline directories initialized in: {self.results_dir}")
        return essential_dirs
    
    def get_directory(self, category: str, subcategory: Optional[str] = None, 
                     create: bool = True) -> Path:
        """Get directory path and optionally create it"""
        if subcategory:
            dir_path = self.results_dir / category / subcategory
        else:
            dir_path = self.results_dir / category
            
        dir_key = f"{category}_{subcategory}" if subcategory else category
        
        if create and str(dir_path) not in self.created_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.add(str(dir_path))
                self.directory_paths[dir_key] = dir_path
                logger.debug(f"📁 Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create directory {dir_path}: {e}")
                raise
                
        return dir_path
    
    def setup_module_directories(self, module_name: str, 
                               required_directories: List[Union[str, Dict]]) -> Dict[str, Path]:
        """Setup directories required for a specific module"""
        module_dirs = {}
        
        for dir_spec in required_directories:
            if isinstance(dir_spec, str):
                # Simple directory path
                dir_path = self.get_directory(dir_spec)
                module_dirs[dir_spec] = dir_path
                
            elif isinstance(dir_spec, dict):
                # Nested directory structure {category: subcategory}
                for category, subcategory in dir_spec.items():
                    if isinstance(subcategory, list):
                        # Multiple subcategories
                        for sub in subcategory:
                            dir_path = self.get_directory(category, sub)
                            key = f"{category}_{sub}"
                            module_dirs[key] = dir_path
                    else:
                        # Single subcategory
                        dir_path = self.get_directory(category, subcategory)
                        key = f"{category}_{subcategory}" if subcategory else category
                        module_dirs[key] = dir_path
        
        logger.info(f"📁 Setup {len(module_dirs)} directories for module: {module_name}")
        return module_dirs
    
    def validate_directory_structure(self) -> Dict[str, bool]:
        """Validate that all expected directories exist and are accessible"""
        validation_results = {}
        
        for category, subcategories in self.DIRECTORY_STRUCTURE.items():
            if isinstance(subcategories, dict):
                for subcategory, description in subcategories.items():
                    dir_path = self.results_dir / category / subcategory
                    exists = dir_path.exists() and dir_path.is_dir()
                    writable = os.access(dir_path, os.W_OK) if exists else False
                    validation_results[f"{category}/{subcategory}"] = exists and writable
                    
                    if not exists:
                        logger.warning(f"⚠️ Missing directory: {dir_path}")
                    elif not writable:
                        logger.warning(f"⚠️ Directory not writable: {dir_path}")
            else:
                dir_path = self.results_dir / category
                exists = dir_path.exists() and dir_path.is_dir()
                writable = os.access(dir_path, os.W_OK) if exists else False
                validation_results[category] = exists and writable
                
                if not exists:
                    logger.warning(f"⚠️ Missing directory: {dir_path}")
                elif not writable:
                    logger.warning(f"⚠️ Directory not writable: {dir_path}")
        
        return validation_results
    
    def get_directory_summary(self) -> Dict[str, Union[str, List]]:
        """Get summary of all directories and their purposes"""
        summary = {
            'results_root': str(self.results_dir),
            'total_created': len(self.created_dirs),
            'directories': {}
        }
        
        for category, subcategories in self.DIRECTORY_STRUCTURE.items():
            if isinstance(subcategories, dict):
                summary['directories'][category] = {}
                for subcategory, description in subcategories.items():
                    dir_path = self.results_dir / category / subcategory
                    summary['directories'][category][subcategory] = {
                        'path': str(dir_path),
                        'purpose': description,
                        'exists': dir_path.exists(),
                        'writable': os.access(dir_path, os.W_OK) if dir_path.exists() else False
                    }
            else:
                dir_path = self.results_dir / category
                summary['directories'][category] = {
                    'path': str(dir_path),
                    'purpose': subcategories,
                    'exists': dir_path.exists(),
                    'writable': os.access(dir_path, os.W_OK) if dir_path.exists() else False
                }
        
        return summary
    
    def save_directory_structure(self, output_file: Optional[str] = None) -> str:
        """Save directory structure to JSON file"""
        if output_file is None:
            output_file = self.results_dir / 'system' / 'logs' / 'directory_structure.json'
        
        summary = self.get_directory_summary()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Directory structure saved to: {output_file}")
        return str(output_file)
    
    def cleanup_empty_directories(self) -> List[str]:
        """Remove empty directories that were created but not used"""
        removed_dirs = []
        
        for dir_path_str in list(self.created_dirs):
            dir_path = Path(dir_path_str)
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # Check if directory is empty
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        self.created_dirs.remove(dir_path_str)
                        removed_dirs.append(dir_path_str)
                        logger.debug(f"🧹 Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not remove directory {dir_path}: {e}")
        
        if removed_dirs:
            logger.info(f"🧹 Cleaned up {len(removed_dirs)} empty directories")
        
        return removed_dirs

# Global directory manager instance for module-level access
_global_directory_manager = None

def get_directory_manager(results_dir: Union[str, Path] = None) -> DirectoryManager:
    """Get or create global directory manager instance"""
    global _global_directory_manager
    
    if _global_directory_manager is None:
        if results_dir is None:
            raise ValueError("Results directory must be provided for first initialization")
        _global_directory_manager = DirectoryManager(results_dir)
        _global_directory_manager.initialize_pipeline_directories()
    
    return _global_directory_manager

def initialize_directories(results_dir: Union[str, Path]) -> DirectoryManager:
    """Initialize directories for a new pipeline run"""
    global _global_directory_manager
    _global_directory_manager = DirectoryManager(results_dir)
    return _global_directory_manager.initialize_pipeline_directories()

def get_module_directories(module_name: str, 
                          required_dirs: List[Union[str, Dict]],
                          results_dir: Union[str, Path] = None) -> Dict[str, Path]:
    """Convenience function for modules to get their required directories"""
    dm = get_directory_manager(results_dir)
    return dm.setup_module_directories(module_name, required_dirs)

# Convenience functions for common directory patterns
def get_analysis_directories(results_dir: Union[str, Path] = None) -> Dict[str, Path]:
    """Get directories for analysis modules"""
    required_dirs = [
        'analysis_results',
        {'analysis_results': ['qtl_mapping', 'fine_mapping', 'interaction_analysis', 'gwas_results']}
    ]
    return get_module_directories('analysis', required_dirs, results_dir)

def get_visualization_directories(results_dir: Union[str, Path] = None) -> Dict[str, Path]:
    """Get directories for visualization modules"""
    required_dirs = [
        'visualization',
        {'visualization': ['summary_plots', 'interactive_plots', 'manhattan_plots', 'qq_plots']}
    ]
    return get_module_directories('visualization', required_dirs, results_dir)

def get_processing_directories(results_dir: Union[str, Path] = None) -> Dict[str, Path]:
    """Get directories for data processing modules"""
    required_dirs = [
        'processed_data',
        {'processed_data': ['genotypes', 'expression', 'proteomics', 'splicing', 'quality_control']},
        'comparison_analysis'
    ]
    return get_module_directories('data_processing', required_dirs, results_dir)

if __name__ == "__main__":
    # Test the directory manager
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dm = DirectoryManager(temp_dir)
        dm.initialize_pipeline_directories()
        
        # Test module directory setup
        analysis_dirs = dm.setup_module_directories('test_module', [
            'analysis_results',
            {'analysis_results': ['qtl_mapping', 'fine_mapping']},
            'visualization'
        ])
        
        print("✅ Directory manager test completed successfully")
        print(f"Created directories: {list(analysis_dirs.keys())}")
        
        # Save structure
        dm.save_directory_structure()