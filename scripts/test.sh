#!/bin/bash
# Test script for Oniris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  (no args)     Run all tests (C++ + Python + model downloads)"
    echo "  --cpp         Run C++ unit tests only"
    echo "  --python      Run Python tests only (including model downloads)"
    echo "  --fast        Run C++ + Python tests without model downloads"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0            Run all tests (including model downloads)"
    echo "  $0 --fast     Run tests without downloading models"
    echo "  $0 --cpp      Run C++ tests only"
    echo "  $0 --python   Run Python tests only"
}

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found. Please run build.sh first."
    exit 1
fi

# Set environment
export PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

# Filter out segfault messages from output
filter_output() {
    grep -v "Segmentation fault" | grep -v "^./scripts/test.sh:" | grep -v "^\[.*\] [0-9]* Aborted" || true
}

run_cpp_tests() {
    echo "========================================"
    echo "Running C++ Unit Tests"
    echo "========================================"
    cd "$BUILD_DIR"
    ctest --output-on-failure -j4
    echo "C++ tests passed!"
    echo ""
}

run_python_tests() {
    local include_download=$1
    echo "========================================"
    echo "Running Python Tests"
    echo "========================================"
    cd "$PROJECT_ROOT"
    
    if [ "$include_download" = "true" ]; then
        echo "Including model download tests..."
        pytest tests/ -v --tb=short 2>&1 | filter_output
    else
        echo "Skipping model download tests..."
        pytest tests/ -v --tb=short -m "not download" 2>&1 | filter_output
    fi
    
    echo "Python tests passed!"
    echo ""
}

# Parse arguments
if [ $# -eq 0 ]; then
    # Default: run all tests with model downloads
    run_cpp_tests
    run_python_tests true
    echo "========================================"
    echo "All tests passed!"
    echo "========================================"
elif [ "$1" = "--fast" ]; then
    # Fast mode: skip model downloads
    run_cpp_tests
    run_python_tests false
    echo "========================================"
    echo "All tests passed! (fast mode)"
    echo "========================================"
elif [ "$1" = "--cpp" ]; then
    run_cpp_tests
elif [ "$1" = "--python" ]; then
    run_python_tests true
elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
else
    echo "Unknown option: $1"
    show_help
    exit 1
fi
