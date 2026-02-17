#!/bin/bash
#
# Start Oniris Web Visualizer
#
# Usage:
#   ./scripts/start_web.sh              # Start server on port 5000
#   ./scripts/start_web.sh --port 8080  # Use custom port

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_ROOT/third_party/web"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}  Oniris Web Visualizer${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Check if web directory exists
if [ ! -d "$WEB_DIR" ]; then
    echo -e "${RED}Error: Web visualizer not found at $WEB_DIR${NC}"
    exit 1
fi

# Parse arguments
PORT="5000"
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --host|-H)
            HOST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port, -p PORT     Port to run on (default: 5000)"
            echo "  --host, -H HOST     Host to bind to (default: 0.0.0.0)"
            echo "  --help, -h          Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check Python
echo -e "${BLUE}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $PYTHON_VERSION"

# Check dependencies
echo ""
echo -e "${BLUE}Checking dependencies...${NC}"

cd "$WEB_DIR"

MISSING_DEPS=""

python3 -c "import onnx" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS onnx"
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS numpy"

if [ -n "$MISSING_DEPS" ]; then
    echo -e "${YELLOW}Missing dependencies: $MISSING_DEPS${NC}"
    echo "Installing..."
    pip install $MISSING_DEPS 2>/dev/null || pip install --user $MISSING_DEPS
fi

echo -e "${GREEN}  Dependencies OK${NC}"

# Check if port is already in use
echo ""
echo -e "${BLUE}Checking port $PORT...${NC}"

PID=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo -e "${YELLOW}  Port $PORT is in use, stopping existing process...${NC}"
    kill -9 $PID 2>/dev/null || true
    sleep 1
fi

echo -e "${GREEN}  Port $PORT is available${NC}"

# Update server.py port
sed -i "s/^PORT = [0-9]*/PORT = $PORT/" "$WEB_DIR/server.py"

# Start server
echo ""
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}  Starting Web Visualizer${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo "  📁 Web directory: $WEB_DIR"
echo "  🌐 URL: http://$HOST:$PORT"
echo ""
echo -e "${BLUE}Starting server...${NC}"
echo "  Press Ctrl+C to stop"
echo ""

cd "$WEB_DIR"
exec python3 server.py
