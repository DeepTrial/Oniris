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

## Updating ONNX Proto Version

Oniris uses the official ONNX proto definition to generate C++ data type definitions. The proto file is automatically downloaded from GitHub during CMake configuration.

### How it works

1. **CMake downloads proto**: `CMakeLists.txt` downloads `onnx.proto` from GitHub:
   ```cmake
   file(DOWNLOAD https://raw.githubusercontent.com/onnx/onnx/v${ONNX_PROTO_VERSION}/onnx/onnx.proto ...)
   ```

2. **Code generation**: A custom CMake target runs `scripts/generate_from_proto.py` to parse the proto file and generate:
   - `src/core/types.hpp` - DataType enum
   - `src/core/types.cpp` - String conversion and type size functions

3. **Build dependency**: The `oniris_core` library depends on the `generate_types` target, ensuring types are regenerated when proto changes.

### Steps to update ONNX version

```bash
# Check latest ONNX release
curl -s https://api.github.com/repos/onnx/onnx/releases/latest | grep tag_name

# Option 1: Temporary update (for testing)
rm -rf build
mkdir build && cd build
cmake .. -DONNX_PROTO_VERSION="1.21.0"
make -j$(nproc)

# Option 2: Permanent update (for commit)
# Edit CMakeLists.txt:
#   set(ONNX_PROTO_VERSION "1.21.0" CACHE STRING "...")
# Then rebuild

# Option 3: Use local proto file
python3 scripts/generate_from_proto.py /path/to/custom/onnx.proto
make -j$(nproc)
```

### After updating

1. **Check generated types**:
   ```bash
   grep "kFloat.*= \|kInt.*= \|kUint.*= " src/core/types.hpp
   ```

2. **Add tests for new types** (if any): Edit `tests/unit/test_types.cpp`

3. **Run all tests**:
   ```bash
   cd build && ./tests/unit/oniris_tests
   ```

4. **Update documentation**: Update README.md version compatibility table

### Common issues

- **Download fails**: Check network connection or use local proto file
- **New types not recognized**: Regenerate types by touching `CMakeLists.txt` or deleting `build/third_party/onnx/onnx.proto`
- **Tests fail**: New types may need explicit test cases in `test_types.cpp`
