#!/bin/bash
# Install script for Oniris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
INSTALL_MODE="${INSTALL_MODE:-dev}"  # dev or system

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --dev           Install in development mode (editable) - default"
    echo "  -s, --system        Install system-wide (requires sudo)"
    echo "  -u, --user          Install in user directory"
    echo "  --prefix PATH       Installation prefix for system install"
    echo ""
    echo "Examples:"
    echo "  $0                  Install in development mode"
    echo "  $0 -s               Install system-wide"
    echo "  $0 -u               Install in user directory (~/.local)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dev)
            INSTALL_MODE="dev"
            shift
            ;;
        -s|--system)
            INSTALL_MODE="system"
            shift
            ;;
        -u|--user)
            INSTALL_MODE="user"
            shift
            ;;
        --prefix)
            CMAKE_INSTALL_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Installing Oniris"
echo "========================================"
echo "Mode: $INSTALL_MODE"
echo ""

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Building first..."
    "$SCRIPT_DIR/build.sh"
fi

install_dev() {
    echo "Installing in development mode..."
    cd "$PROJECT_ROOT"
    pip install -e .
    echo "Development install completed!"
}

install_user() {
    echo "Installing in user directory..."
    cd "$PROJECT_ROOT"
    pip install --user -e .
    echo "User install completed!"
}

install_system() {
    echo "Installing system-wide..."
    
    # Install C++ library
    cd "$BUILD_DIR"
    if [ -n "$CMAKE_INSTALL_PREFIX" ]; then
        cmake --install . --prefix "$CMAKE_INSTALL_PREFIX"
    else
        sudo cmake --install .
    fi
    
    # Install Python package
    cd "$PROJECT_ROOT"
    pip install .
    
    echo "System install completed!"
}

# Run installation based on mode
case $INSTALL_MODE in
    dev)
        install_dev
        ;;
    user)
        install_user
        ;;
    system)
        install_system
        ;;
esac

echo ""
echo "========================================"
echo "Installation completed!"
echo "========================================"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import oniris; print(f'Oniris version: {oniris.__version__}')" || {
    echo "Warning: Could not import oniris. Please check your Python path."
}
