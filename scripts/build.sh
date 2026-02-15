#!/bin/bash
# Build script for Oniris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --debug         Build in debug mode"
    echo "  -r, --release       Build in release mode (default)"
    echo "  -c, --clean         Clean build directory before building"
    echo "  -j, --jobs N        Number of parallel jobs (default: $JOBS)"
    echo "  --no-python         Skip Python bindings"
    echo "  --no-tests          Skip building tests"
    echo ""
    echo "Examples:"
    echo "  $0                  Build release version"
    echo "  $0 -d               Build debug version"
    echo "  $0 -c -j8           Clean build with 8 parallel jobs"
}

# Parse arguments
CLEAN=0
PYTHON_BINDINGS=ON
BUILD_TESTS=ON

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -c|--clean)
            CLEAN=1
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --no-python)
            PYTHON_BINDINGS=OFF
            shift
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Building Oniris"
echo "========================================"
echo "Build type: $BUILD_TYPE"
echo "Jobs: $JOBS"
echo "Python bindings: $PYTHON_BINDINGS"
echo "Tests: $BUILD_TESTS"
echo ""

# Clean if requested
if [ "$CLEAN" -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring with CMake..."
cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_PYTHON_BINDINGS="$PYTHON_BINDINGS" \
    -DBUILD_TESTS="$BUILD_TESTS"

# Build
echo ""
echo "Building..."
cmake --build . --parallel "$JOBS"

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo "Build directory: $BUILD_DIR"
