# Run input data build pipeline
# --------------------------- #
python build_input.py --config input_build_config.yaml

# The pipeline will now create:
# - phenotype.tsv: Phenotype data for GWAS analysis
# - Updated QTL config that includes phenotype file path
# - Enhanced sample mapping with phenotype information


# For GWAS phenotype collection
# --------------------------- #
# Search for traits and create manifest
python gwas_phenotype_collector.py --output-dir /path/to/gwas_data --traits "body mass index" "height"

# Download specific GWAS studies
python gwas_phenotype_collector.py --output-dir /path/to/gwas_data --download-gwas --gwas-id "ieu-a-2" "ieu-a-7"