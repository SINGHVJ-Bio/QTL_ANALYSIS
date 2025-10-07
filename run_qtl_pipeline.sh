#!/bin/bash

# Enhanced QTL Analysis Pipeline Shell Script
# Complete cis/trans QTL analysis with comprehensive reporting
# Located in root directory for easy access
# Usage: ./run_qtl_pipeline.sh [options]

set -euo pipefail

# Initialize variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DATA_DIR=""
OUTPUT_DIR=""
CONFIG_FILE=""
ANALYSIS_TYPES=""
QTL_MODE=""
FORCE_OVERWRITE=false
VALIDATE_ONLY=false
RUN_GWAS=false
DEBUG_MODE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_debug() { echo -e "${PURPLE}[DEBUG]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Print banner
print_banner() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   QTL Analysis Pipeline                      ║"
    echo "║                 Complete cis/trans Mapping                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Enhanced QTL Analysis Pipeline - Complete cis/trans QTL mapping

OPTIONS:
    -d, --data-dir DIR        Directory containing input data (required)
    -o, --output-dir DIR      Output directory (default: results_<timestamp>)
    -c, --config FILE         Configuration file (default: config/pipeline_config.yaml)
    -a, --analysis-types TYPES Comma-separated list: eqtl,pqtl,sqtl (default: all)
    -m, --qtl-mode MODE       QTL mapping mode: cis, trans, both (default: cis)
    -f, --force               Overwrite existing output directory
    -v, --validate-only       Only validate inputs, don't run analysis
    -g, --run-gwas            Enable GWAS analysis
    --debug                   Enable debug mode with detailed logging
    -h, --help                Show this help message

EXAMPLES:
    # Run complete analysis with default config
    $0 -d data -o results

    # Run specific QTL types only
    $0 -d data -o results -a eqtl,pqtl

    # Run only cis-QTL analysis
    $0 -d data -o results -m cis

    # Run only trans-QTL analysis  
    $0 -d data -o results -m trans

    # Run both cis and trans QTL
    $0 -d data -o results -m both

    # Run with GWAS analysis enabled
    $0 -d data -o results -g

    # Validate inputs only
    $0 -d data --validate-only

    # Run with custom config and debug mode
    $0 -d data -c my_config.yaml --debug
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -a|--analysis-types)
            ANALYSIS_TYPES="$2"
            shift 2
            ;;
        -m|--qtl-mode)
            QTL_MODE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_OVERWRITE=true
            shift
            ;;
        -v|--validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        -g|--run-gwas)
            RUN_GWAS=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set default output directory if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${PROJECT_ROOT}/results_${TIMESTAMP}"
fi

# Set default config if not provided
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/config/pipeline_config.yaml"
fi

# Validate required arguments
if [[ -z "$DATA_DIR" ]]; then
    log_error "Data directory is required"
    usage
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    log_error "Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if output directory exists
if [[ -d "$OUTPUT_DIR" ]]; then
    if [[ "$FORCE_OVERWRITE" == true ]]; then
        log_warn "Overwriting existing output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    else
        log_error "Output directory already exists: $OUTPUT_DIR"
        log_error "Use -f to overwrite or specify a different directory"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
log_success "Output directory created: $OUTPUT_DIR"

# Set up logging
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Function to check required tools
check_requirements() {
    log_step "Checking system requirements..."
    
    local tools=("qtltools" "bgzip" "tabix" "bcftools" "python3")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        log_error "You can use conda: conda install -c bioconda ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All required tools are available"
    
    # Display tool versions
    log_info "Tool versions:"
    for tool in "${tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            version=$($tool --version 2>/dev/null | head -1 || echo "version unknown")
            log_info "  $tool: $version"
        fi
    done
}

# Function to validate Python dependencies
check_python_deps() {
    log_step "Checking Python dependencies..."
    
    local deps=("pandas" "numpy" "yaml" "matplotlib" "seaborn" "scipy")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! python3 -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing Python dependencies: ${missing_deps[*]}"
        log_error "Install with: pip install ${missing_deps[*]}"
        log_error "Or install all requirements: pip install -r requirements.txt"
        exit 1
    fi
    
    log_success "All Python dependencies are available"
}

# Function to run input validation
run_validation() {
    log_step "Running comprehensive input validation..."
    
    local python_cmd=(
        python3 "${PROJECT_ROOT}/run_QTLPipeline.py"
        --config "$CONFIG_FILE"
        --validate-only
    )
    
    if [[ "$DEBUG_MODE" == true ]]; then
        python_cmd+=(--debug)
    fi
    
    "${python_cmd[@]}"
    
    if [[ $? -eq 0 ]]; then
        log_success "Input validation completed successfully"
    else
        log_error "Input validation failed"
        exit 1
    fi
}

# Function to setup data directory structure
setup_data_directory() {
    log_step "Setting up data directory structure..."
    
    # Check if data directory exists in project root, if not create symlink
    if [[ "$DATA_DIR" != "${PROJECT_ROOT}/data" ]] && [[ ! -d "${PROJECT_ROOT}/data" ]]; then
        log_info "Creating data directory symlink..."
        ln -sf "$DATA_DIR" "${PROJECT_ROOT}/data"
        log_success "Created symlink: ${PROJECT_ROOT}/data -> $DATA_DIR"
    fi
    
    # Verify data files exist
    log_info "Checking data files..."
    local data_files=("genotypes" "covariates" "annotations")
    for file_type in "${data_files[@]}"; do
        file_path="${DATA_DIR}/${file_type}.*"
        if ls $file_path 1> /dev/null 2>&1; then
            log_info "✅ Found ${file_type} files"
        else
            log_warn "⚠️  No ${file_type} files found in $DATA_DIR"
        fi
    done
}

# Function to display configuration summary
display_config_summary() {
    log_step "Configuration Summary:"
    log_info "Project root:    $PROJECT_ROOT"
    log_info "Data directory:  $DATA_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Config file:     $CONFIG_FILE"
    log_info "Analysis types:  ${ANALYSIS_TYPES:-all}"
    log_info "QTL mode:        ${QTL_MODE:-cis}"
    log_info "GWAS analysis:   $RUN_GWAS"
    log_info "Validate only:   $VALIDATE_ONLY"
    log_info "Debug mode:      $DEBUG_MODE"
    log_info "Date:            $(date)"
}

# Function to run the main pipeline
run_pipeline() {
    log_step "Starting QTL analysis pipeline..."
    
    local python_cmd=(
        python3 "${PROJECT_ROOT}/run_QTLPipeline.py"
        --config "$CONFIG_FILE"
        --output-dir "$OUTPUT_DIR"
    )
    
    # Add optional arguments if provided
    if [[ -n "$ANALYSIS_TYPES" ]]; then
        python_cmd+=(--analysis-types "$ANALYSIS_TYPES")
    fi
    
    if [[ -n "$QTL_MODE" ]]; then
        python_cmd+=(--qtl-mode "$QTL_MODE")
    fi
    
    if [[ "$RUN_GWAS" == true ]]; then
        python_cmd+=(--run-gwas)
    fi
    
    if [[ "$DEBUG_MODE" == true ]]; then
        python_cmd+=(--debug)
    fi
    
    log_info "Running: ${python_cmd[*]}"
    "${python_cmd[@]}"
}

# Function to generate final report
generate_final_report() {
    log_step "Generating final reports..."
    
    if [[ -f "${OUTPUT_DIR}/reports/analysis_report.html" ]]; then
        log_success "HTML report: ${OUTPUT_DIR}/reports/analysis_report.html"
    else
        log_warn "HTML report not found"
    fi
    
    if [[ -f "${OUTPUT_DIR}/pipeline_summary.txt" ]]; then
        log_success "Summary report: ${OUTPUT_DIR}/pipeline_summary.txt"
    fi
    
    # Display directory structure
    log_info "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree -L 2 "$OUTPUT_DIR" || find "$OUTPUT_DIR" -maxdepth 2 -type d | head -20
    else
        find "$OUTPUT_DIR" -maxdepth 2 -type d | head -20
    fi
}

# Main execution function
main() {
    print_banner
    log_step "🚀 Starting Enhanced QTL Analysis Pipeline"
    
    # Display configuration
    display_config_summary
    
    # Check requirements
    check_requirements
    check_python_deps
    
    # Setup data directory
    setup_data_directory
    
    # Run validation
    run_validation
    
    if [[ "$VALIDATE_ONLY" == true ]]; then
        log_success "✅ Validation only mode completed successfully"
        log_info "All inputs are valid and ready for analysis"
        exit 0
    fi
    
    # Run the pipeline
    if run_pipeline; then
        log_success "✅ QTL analysis pipeline completed successfully"
        generate_final_report
        
        log_step "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
        log_info "Results available in: $OUTPUT_DIR"
        log_info "Log file: $LOG_FILE"
        log_info "HTML report: $OUTPUT_DIR/reports/analysis_report.html"
        log_info "Summary: $OUTPUT_DIR/pipeline_summary.txt"
        
        # Display next steps
        echo
        log_step "📋 NEXT STEPS:"
        log_info "1. Review the HTML report: $OUTPUT_DIR/reports/analysis_report.html"
        log_info "2. Check generated plots: $OUTPUT_DIR/plots/"
        log_info "3. Examine detailed results: $OUTPUT_DIR/QTL_results/"
        log_info "4. Review logs for any warnings: $OUTPUT_DIR/logs/"
        
    else
        log_error "❌ QTL analysis pipeline failed"
        log_error "Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    
    # Remove temporary symlink if created
    if [[ -L "${PROJECT_ROOT}/data" ]] && [[ "$(readlink "${PROJECT_ROOT}/data")" == "$DATA_DIR" ]]; then
        rm -f "${PROJECT_ROOT}/data"
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Pipeline completed successfully"
    else
        log_error "Pipeline failed with exit code $exit_code"
        log_error "Check the log file: $LOG_FILE"
    fi
    
    exit $exit_code
}

# Set traps
trap cleanup EXIT
trap 'log_error "Script interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"