#!/bin/bash

# Enhanced QTL Analysis Pipeline Shell Script
# Usage: ./run_qtl_pipeline.sh [options]

set -euo pipefail

# Initialize variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR=""
OUTPUT_DIR=""
CONFIG_FILE=""
ANALYSIS_TYPES=""
FORCE_OVERWRITE=false
VALIDATE_ONLY=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Enhanced QTL Analysis Pipeline

OPTIONS:
    -d, --data-dir DIR        Directory containing input data (required)
    -o, --output-dir DIR      Output directory (default: results_<timestamp>)
    -c, --config FILE         Configuration file (default: config/config.yaml)
    -a, --analysis-types TYPES Comma-separated list: eqtl,pqtl,sqtl (default: all)
    -f, --force               Overwrite existing output directory
    -v, --validate-only       Only validate inputs, don't run analysis
    -h, --help                Show this help message

EXAMPLES:
    $0 -d data -o results -a eqtl,pqtl
    $0 -d /path/to/data -c my_config.yaml -a all
    $0 -d data --validate-only
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
        -f|--force)
            FORCE_OVERWRITE=true
            shift
            ;;
        -v|--validate-only)
            VALIDATE_ONLY=true
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
    CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"
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
log_info "Output directory created: $OUTPUT_DIR"

# Set up logging
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Function to check required tools
check_requirements() {
    log_info "Checking system requirements..."
    
    local tools=("qtltools" "bgzip" "tabix" "bcftools" "python3")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All required tools are available"
}

# Function to validate Python dependencies
check_python_deps() {
    log_info "Checking Python dependencies..."
    
    local deps=("pandas" "numpy" "yaml")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! python3 -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing Python dependencies: ${missing_deps[*]}"
        log_error "Install with: pip install ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All Python dependencies are available"
}

# Function to run input validation
run_validation() {
    log_info "Running input validation..."
    
    python3 "${SCRIPT_DIR}/validate_inputs.py" \
        --data-dir "$DATA_DIR" \
        --config "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR"
    
    if [[ $? -eq 0 ]]; then
        log_success "Input validation completed successfully"
    else
        log_error "Input validation failed"
        exit 1
    fi
}

# Main execution function
main() {
    log_info "Starting Enhanced QTL Analysis Pipeline"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Data directory: $DATA_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Config file: $CONFIG_FILE"
    log_info "Analysis types: ${ANALYSIS_TYPES:-all}"
    log_info "Date: $(date)"
    
    # Check requirements
    check_requirements
    check_python_deps
    
    # Run validation
    run_validation
    
    if [[ "$VALIDATE_ONLY" == true ]]; then
        log_success "Validation only mode completed"
        exit 0
    fi
    
    # Run the pipeline
    log_info "Starting QTL analysis..."
    
    local python_cmd=(
        python3 "${SCRIPT_DIR}/qtl_pipeline.py"
        --data-dir "$DATA_DIR"
        --output-dir "$OUTPUT_DIR"
        --config "$CONFIG_FILE"
    )
    
    if [[ -n "$ANALYSIS_TYPES" ]]; then
        python_cmd+=(--analysis-types "$ANALYSIS_TYPES")
    fi
    
    "${python_cmd[@]}"
    
    if [[ $? -eq 0 ]]; then
        log_success "QTL analysis pipeline completed successfully"
        log_info "Results available in: $OUTPUT_DIR"
        log_info "Log file: $LOG_FILE"
    else
        log_error "QTL analysis pipeline failed"
        exit 1
    fi
}

# Trap for cleanup on exit
cleanup() {
    log_info "Cleaning up..."
    # Add any cleanup tasks here
}

trap cleanup EXIT

# Run main function
main "$@"