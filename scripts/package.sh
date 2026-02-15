#!/bin/bash
# Package script for Oniris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
DIST_DIR="${PROJECT_ROOT}/dist"

show_help() {
    echo "Usage: $0 [options] [target]"
    echo ""
    echo "Targets:"
    echo "  all         Package everything (default)"
    echo "  python      Package Python wheel only"
    echo "  source      Create source distribution"
    echo "  clean       Clean package artifacts"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  --skip-build        Skip build step (use existing build)"
    echo ""
    echo "Examples:"
    echo "  $0                  Package everything"
    echo "  $0 python           Package Python wheel only"
    echo "  $0 clean            Clean all package artifacts"
}

# Parse arguments
SKIP_BUILD=0
TARGET="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        all|python|source|clean)
            TARGET="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

clean_packages() {
    echo "Cleaning package artifacts..."
    rm -rf "$DIST_DIR"
    rm -rf "${PROJECT_ROOT}/build/lib"
    rm -rf "${PROJECT_ROOT}/build/bdist"
    rm -rf "${PROJECT_ROOT}/*.egg-info"
    find "$PROJECT_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    echo "Clean completed!"
}

build_wheel() {
    echo "Building Python wheel..."
    cd "$PROJECT_ROOT"
    
    # Ensure we have a clean build
    rm -rf build/lib build/bdist
    
    # Build wheel
    pip wheel . --wheel-dir="$DIST_DIR" --no-deps
    
    echo "Wheel built successfully!"
    ls -lh "$DIST_DIR"/*.whl
}

build_source_dist() {
    echo "Building source distribution..."
    cd "$PROJECT_ROOT"
    
    python setup.py sdist --dist-dir="$DIST_DIR"
    
    echo "Source distribution built successfully!"
    ls -lh "$DIST_DIR"/*.tar.gz
}

# Main execution
echo "========================================"
echo "Packaging Oniris"
echo "========================================"
echo "Target: $TARGET"
echo ""

# Clean if requested
if [ "$TARGET" = "clean" ]; then
    clean_packages
    exit 0
fi

# Create dist directory
mkdir -p "$DIST_DIR"

# Build first if needed
if [ "$SKIP_BUILD" -eq 0 ] && [ "$TARGET" != "source" ]; then
    echo "Building project first..."
    "$SCRIPT_DIR/build.sh" --release
    echo ""
fi

# Package based on target
case $TARGET in
    all)
        build_wheel
        build_source_dist
        ;;
    python)
        build_wheel
        ;;
    source)
        build_source_dist
        ;;
esac

echo ""
echo "========================================"
echo "Packaging completed!"
echo "========================================"
echo "Artifacts location: $DIST_DIR"
echo ""
ls -lh "$DIST_DIR/"
