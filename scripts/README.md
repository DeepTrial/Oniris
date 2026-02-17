# Oniris Build Scripts

This directory contains build automation scripts for Oniris.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `setup.sh` | Initialize development environment |
| `build.sh` | Build the C++ library and Python bindings |
| `test.sh` | Run tests (C++ and/or Python) |
| `install.sh` | Install the package |
| `package.sh` | Create distribution packages |
| `lint.sh` | Run linters and formatters |
| `start_web.sh` | Start the web visualizer server |

## Usage

### Initial Setup

```bash
./scripts/setup.sh
```

This will:
1. Check system dependencies (Python, CMake, C++ compiler)
2. Install Python dependencies
3. Initialize git submodules
4. Build the project
5. Install in development mode

### Build

```bash
# Build release version (default)
./scripts/build.sh

# Build debug version
./scripts/build.sh --debug

# Clean and rebuild
./scripts/build.sh --clean

# Build with specific parallel jobs
./scripts/build.sh --jobs 8
```

### Test

```bash
# Run all tests
./scripts/test.sh

# Run C++ tests only
./scripts/test.sh cpp

# Run Python tests only
./scripts/test.sh python

# Run with coverage
./scripts/test.sh --coverage

# Run system tests (downloads real models)
./scripts/test.sh system
```

### Install

```bash
# Install in development mode (default)
./scripts/install.sh

# Install in user directory
./scripts/install.sh --user

# Install system-wide (requires sudo)
./scripts/install.sh --system
```

### Package

```bash
# Package everything (wheel + source)
./scripts/package.sh

# Package Python wheel only
./scripts/package.sh python

# Create source distribution
./scripts/package.sh source

# Clean package artifacts
./scripts/package.sh clean
```

### Lint/Format

```bash
# Run all linters
./scripts/lint.sh

# Format all code
./scripts/lint.sh format

# Lint C++ only
./scripts/lint.sh cpp

# Lint Python only
./scripts/lint.sh python

# Auto-fix Python issues
./scripts/lint.sh python fix
```

### Web Visualizer

```bash
# Start web visualizer (development mode)
./scripts/start_web.sh

# Production mode (requires gunicorn)
./scripts/start_web.sh --prod

# Custom port
./scripts/start_web.sh --port 8080

# Custom host and port
./scripts/start_web.sh --host 127.0.0.1 --port 3000

# Debug mode with auto-reload
./scripts/start_web.sh --debug

# Show all options
./scripts/start_web.sh --help
```

This will start the web visualizer at `http://localhost:5000` (or your specified port).

Features:
- Interactive ONNX model visualization
- Shape inference and model simplification
- Add/remove layers visually
- Export modified models

## Common Workflows

### Development Workflow

```bash
# 1. Initial setup
./scripts/setup.sh

# 2. Make changes to code...

# 3. Build
./scripts/build.sh

# 4. Test
./scripts/test.sh

# 5. Format code before committing
./scripts/lint.sh format
```

### Web Development Workflow

```bash
# 1. Start web server in one terminal
./scripts/start_web.sh --debug

# 2. Open http://localhost:5000 in browser

# 3. Make changes to web/frontend/ files
#    Changes are reflected immediately in debug mode

# 4. Test with real models
#    Upload .onnx files through the web interface
```

### Release Workflow

```bash
# 1. Run tests
./scripts/test.sh

# 2. Build release version
./scripts/build.sh --release --clean

# 3. Create packages
./scripts/package.sh

# 4. Packages will be in dist/ directory
ls dist/
```

## Script Options

All scripts support `--help` to see available options:

```bash
./scripts/build.sh --help
./scripts/test.sh --help
./scripts/install.sh --help
./scripts/package.sh --help
./scripts/lint.sh --help
./scripts/start_web.sh --help
```
