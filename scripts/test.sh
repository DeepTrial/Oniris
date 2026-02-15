#!/bin/bash
# Test script for Oniris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

show_help() {
    echo "Usage: $0 [options] [test_type]"
    echo ""
    echo "Test Types:"
    echo "  all         Run all tests (default)"
    echo "  cpp         Run C++ unit tests only"
    echo "  python      Run Python tests only"
    echo "  system      Run system tests (downloads real models)"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --verbose       Verbose output"
    echo "  -c, --coverage      Generate coverage report"
    echo ""
    echo "Examples:"
    echo "  $0                  Run all tests"
    echo "  $0 cpp              Run C++ tests only"
    echo "  $0 python -v        Run Python tests with verbose output"
    echo "  $0 all -c           Run all tests with coverage"
}

# Parse arguments
VERBOSE=""
COVERAGE=0
TEST_TYPE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE=1
            shift
            ;;
        all|cpp|python|system)
            TEST_TYPE="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found. Please run build.sh first."
    exit 1
fi

echo "========================================"
echo "Running Oniris Tests"
echo "========================================"
echo "Test type: $TEST_TYPE"
echo ""

run_cpp_tests() {
    echo "Running C++ unit tests..."
    cd "$BUILD_DIR"
    if [ -n "$VERBOSE" ]; then
        ctest --output-on-failure $VERBOSE
    else
        ctest --output-on-failure
    fi
    echo "C++ tests passed!"
    echo ""
}

run_python_tests() {
    echo "Running Python tests..."
    cd "$PROJECT_ROOT"
    
    if [ "$COVERAGE" -eq 1 ]; then
        pytest tests/unit $VERBOSE --cov=python/oniris --cov-report=html --cov-report=term
    else
        pytest tests/unit $VERBOSE
    fi
    echo "Python tests passed!"
    echo ""
}

run_system_tests() {
    echo "Running system tests..."
    echo "Note: This may download models from the internet"
    cd "$PROJECT_ROOT"
    pytest tests/system $VERBOSE
    echo "System tests passed!"
    echo ""
}

# Run tests based on type
case $TEST_TYPE in
    all)
        run_cpp_tests
        run_python_tests
        echo "All tests passed!"
        ;;
    cpp)
        run_cpp_tests
        ;;
    python)
        run_python_tests
        ;;
    system)
        run_system_tests
        ;;
esac

echo ""
echo "========================================"
echo "Test run completed!"
echo "========================================"
