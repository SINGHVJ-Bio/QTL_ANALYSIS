#!/usr/bin/env python3
"""
GWAS Phenotype Collector - Download and process GWAS phenotype data from public databases
"""

import pandas as pd
import requests
import json
import os
import logging
from pathlib import Path
import argparse
import gzip
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GWASPhenotypeCollector')

class GWASPhenotypeCollector:
    """Collect GWAS phenotype data from various public databases"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def search_gwas_catalog(self, query, pvalue_threshold=1e-5):
        """Search GWAS Catalog for specific phenotypes"""
        logger.info(f"Searching GWAS Catalog for: {query}")
        
        base_url = "https://www.ebi.ac.uk/gwas/rest/api"
        
        try:
            # Search for studies
            search_url = f"{base_url}/studies/search/findByDiseaseTrait"
            params = {
                'trait': query,
                'size': 100
            }
            
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            studies = response.json()['_embedded']['studies']
            
            logger.info(f"Found {len(studies)} studies for '{query}'")
            
            study_data = []
            for study in studies:
                study_info = {
                    'study_id': study.get('accessionId', ''),
                    'trait': study.get('diseaseTrait', {}).get('trait', ''),
                    'pubmed_id': study.get('publicationInfo', {}).get('pubmedId', ''),
                    'initial_sample': study.get('initialSampleSize', ''),
                    'replication_sample': study.get('replicationSampleSize', ''),
                    'platform': study.get('platform', [])
                }
                study_data.append(study_info)
            
            return study_data
            
        except Exception as e:
            logger.error(f"Error searching GWAS Catalog: {e}")
            return []
    
    def download_ukb_phenotypes(self, phenotype_codes, output_file):
        """Download UK Biobank phenotype data (requires access)"""
        logger.info(f"Processing UK Biobank phenotypes: {phenotype_codes}")
        
        # This is a template - actual implementation requires UK Biobank access
        # and proper data handling procedures
        
        template_data = {
            'sample_id': [],
            'phenotype_code': [],
            'value': [],
            'description': []
        }
        
        for code in phenotype_codes:
            # Placeholder for actual UK Biobank data extraction
            template_data['phenotype_code'].extend([code] * 10)
            template_data['description'].extend([f"UKB Field {code}"] * 10)
        
        df = pd.DataFrame(template_data)
        df.to_csv(output_file, sep="\t", index=False)
        logger.info(f"UK Biobank phenotype template saved: {output_file}")
        
        return df
    
    def get_ieu_gwas_data(self, gwas_id, output_file):
        """Download GWAS summary statistics from IEU OpenGWAS"""
        logger.info(f"Downloading IEU GWAS data for: {gwas_id}")
        
        base_url = "https://gwas.mrcieu.ac.uk/files"
        
        try:
            # Download GWAS summary statistics
            download_url = f"{base_url}/{gwas_id}/{gwas_id}.vcf.gz"
            local_file = self.output_dir / f"{gwas_id}.vcf.gz"
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(local_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract and convert to manageable format
            self.process_gwas_summary_stats(local_file, output_file)
            
            logger.info(f"GWAS data downloaded and processed: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading IEU GWAS data: {e}")
            return False
    
    def process_gwas_summary_stats(self, vcf_file, output_file):
        """Process GWAS summary statistics VCF file"""
        logger.info(f"Processing GWAS summary statistics: {vcf_file}")
        
        try:
            # This is a simplified processing step
            # In practice, you'd want to extract relevant fields and format properly
            
            cmd = [
                'bcftools', 'query',
                '-f', '%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%INFO/BETA\\t%INFO/SE\\t%INFO/P\\n',
                str(vcf_file)
            ]
            
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Create proper GWAS summary stats format
            lines = result.stdout.strip().split('\n')
            gwas_data = []
            
            for line in lines:
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 8:
                        gwas_data.append({
                            'CHROM': parts[0],
                            'POS': parts[1],
                            'SNP': parts[2],
                            'A1': parts[4],  # ALT
                            'A2': parts[3],  # REF
                            'BETA': parts[5] if parts[5] != '.' else '0',
                            'SE': parts[6] if parts[6] != '.' else '0',
                            'P': parts[7] if parts[7] != '.' else '1'
                        })
            
            df = pd.DataFrame(gwas_data)
            df.to_csv(output_file, sep="\t", index=False)
            
            # Clean up
            os.remove(vcf_file)
            
        except Exception as e:
            logger.error(f"Error processing GWAS summary stats: {e}")
    
    def create_phenotype_manifest(self, traits_of_interest):
        """Create a manifest of available phenotypes"""
        logger.info("Creating phenotype manifest...")
        
        manifest_data = []
        
        # Common complex traits
        common_traits = [
            'body mass index', 'height', 'type 2 diabetes', 
            'coronary artery disease', 'blood pressure',
            'cholesterol', 'asthma', 'depression', 'schizophrenia'
        ]
        
        for trait in common_traits:
            if any(t in trait.lower() for t in traits_of_interest):
                studies = self.search_gwas_catalog(trait)
                for study in studies:
                    manifest_data.append({
                        'trait': trait,
                        'study_id': study['study_id'],
                        'pubmed_id': study['pubmed_id'],
                        'sample_size': study['initial_sample']
                    })
        
        manifest_df = pd.DataFrame(manifest_data)
        manifest_file = self.output_dir / "phenotype_manifest.tsv"
        manifest_df.to_csv(manifest_file, sep="\t", index=False)
        
        logger.info(f"Phenotype manifest created: {manifest_file}")
        return manifest_df
    
    def format_for_tensorqtl(self, gwas_file, output_file):
        """Format GWAS summary statistics for tensorQTL compatibility"""
        logger.info(f"Formatting {gwas_file} for tensorQTL...")
        
        try:
            df = pd.read_csv(gwas_file, sep="\t")
            
            # Standardize column names
            column_mapping = {
                'CHROM': 'chromosome',
                'POS': 'position', 
                'SNP': 'variant_id',
                'A1': 'effect_allele',
                'A2': 'other_allele',
                'BETA': 'beta',
                'SE': 'standard_error',
                'P': 'p_value'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Ensure required columns
            required_cols = ['variant_id', 'chromosome', 'position', 'p_value']
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            
            # Add Z-score if beta and SE are available
            if 'beta' in df.columns and 'standard_error' in df.columns:
                df['z_score'] = df['beta'] / df['standard_error']
            
            df.to_csv(output_file, sep="\t", index=False)
            logger.info(f"Formatted GWAS data saved: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error formatting GWAS data: {e}")
            return None

def main():
    """Main function for GWAS phenotype collection"""
    parser = argparse.ArgumentParser(description='Collect GWAS phenotype data from public databases')
    parser.add_argument('--output-dir', required=True, help='Output directory for phenotype data')
    parser.add_argument('--traits', nargs='+', help='Traits of interest', 
                       default=['body mass index', 'height', 'type 2 diabetes'])
    parser.add_argument('--download-gwas', action='store_true', help='Download GWAS summary statistics')
    parser.add_argument('--gwas-ids', nargs='+', help='Specific GWAS IDs to download')
    
    args = parser.parse_args()
    
    collector = GWASPhenotypeCollector(args.output_dir)
    
    # Create phenotype manifest
    manifest = collector.create_phenotype_manifest(args.traits)
    logger.info(f"Found {len(manifest)} studies for specified traits")
    
    # Download GWAS data if requested
    if args.download_gwas and args.gwas_ids:
        for gwas_id in args.gwas_ids:
            output_file = collector.output_dir / f"{gwas_id}_gwas.tsv"
            if collector.get_ieu_gwas_data(gwas_id, output_file):
                # Format for tensorQTL
                formatted_file = collector.output_dir / f"{gwas_id}_formatted.tsv"
                collector.format_for_tensorqtl(output_file, formatted_file)
    
    logger.info("GWAS phenotype collection completed!")

if __name__ == "__main__":
    main()