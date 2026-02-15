#!/bin/bash
# Setup script for Oniris development environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  --system-packages   Allow system packages"
    echo ""
    echo "Examples:"
    echo "  $0                  Setup development environment"
}

USE_SYSTEM_PACKAGES=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --system-packages)
            USE_SYSTEM_PACKAGES=1
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
echo "Setting up Oniris Development Environment"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check CMake
echo ""
echo "Checking CMake..."
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.14 or later."
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
echo "Found CMake $CMAKE_VERSION"

# Check compiler
echo ""
echo "Checking C++ compiler..."
if command -v g++ &> /dev/null; then
    CXX_VERSION=$(g++ --version | head -n1)
    echo "Found: $CXX_VERSION"
elif command -v clang++ &> /dev/null; then
    CXX_VERSION=$(clang++ --version | head -n1)
    echo "Found: $CXX_VERSION"
else
    echo "Warning: No C++ compiler found. Please install GCC 7+ or Clang 5+."
fi

# Note: pybind11 will be automatically fetched by CMake if not present in third_party/

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
cd "$PROJECT_ROOT"

if [ "$USE_SYSTEM_PACKAGES" -eq 1 ]; then
    pip install -r requirements.txt
else
    pip install --user -r requirements.txt 2>/dev/null || pip install -r requirements.txt
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p "$PROJECT_ROOT/build"

# Build the project
echo ""
echo "Building project..."
"$SCRIPT_DIR/build.sh" --release

# Install in development mode
echo ""
echo "Installing in development mode..."
"$SCRIPT_DIR/install.sh" --dev

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "You can now:"
echo "  - Run tests:       ./scripts/test.sh"
echo "  - Build project:   ./scripts/build.sh"
echo "  - Install package: ./scripts/install.sh"
echo "  - Create package:  ./scripts/package.sh"
echo ""
echo "To get started, see docs/QUICKSTART.md"
