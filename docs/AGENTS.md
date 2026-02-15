# Oniris Project Guidelines

## Project Overview

Oniris is an ONNX compilation toolkit that provides model simplification and shape inference capabilities.

## Architecture

### Core Components

1. **IR (Intermediate Representation)**: In-memory representation of ONNX models
2. **Shape Inference Engine**: Handles both dynamic and static shape inference
3. **Model Simplifier**: Optimizes and simplifies ONNX models
4. **Custom Layer Registry**: Allows users to register custom layer handlers

### Directory Structure

- `src/core/`: Core utilities and data structures
- `src/ir/`: Intermediate representation classes
- `src/passes/`: Optimization passes (simplification, shape inference)
- `src/python/`: Python bindings (pybind11)
- `python/oniris/`: Python package code
- `tests/`: Test suites

## Coding Standards

### C++ Guidelines

- Use C++17 standard
- Follow Google C++ Style Guide conventions
- Use smart pointers for memory management
- Prefer `std::optional` over raw pointers for optional values
- Use `const` correctness

### Python Guidelines

- Follow PEP 8 style guide
- Type hints are required for all public functions
- Docstrings in Google style

### Comments

- All comments must be in English
- Use Doxygen-style comments for C++: `/** */` or `///`
- Use docstrings for Python: `""" """`

## Build System

- CMake 3.14+ for C++
- pybind11 for Python bindings (automatically fetched via CMake FetchContent)
- setuptools for Python packaging

## Testing

- Google Test for C++ unit tests
- pytest for Python tests
- System tests use real ONNX models from HuggingFace/ONNX Model Zoo
