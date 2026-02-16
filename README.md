# Oniris - ONNX Compilation Toolkit

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Oniris is a high-performance ONNX model compilation and optimization toolkit written in C++ with Python bindings.

## Features

- **🔧 Model Simplification**: Simplify ONNX models similar to onnxsim, with graceful handling of unsupported layers
- **📐 Shape Inference**: Comprehensive shape inference supporting 120+ ONNX operators with both dynamic and static shapes
- **🔌 Extensible Architecture**: Plugin-based system for custom layers and operations
- **⚡ High Performance**: Core implementation in C++ with Python-friendly interfaces
- **✅ Production Ready**: Comprehensive test suite with real ONNX models

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Oniris

# Install dependencies
pip install -r requirements.txt

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
pip install -e .
```

### Simplify an ONNX Model

```python
import oniris

# Simplify a model
oniris.simplify('input.onnx', 'output.onnx')

# Or with custom options (e.g., disable fusion)
options = oniris.SimplifyOptions()
options.fuse_conv_bn = False  # Disable Conv+BN fusion
oniris.Simplifier.simplify(model, options)
```

### Shape Inference

```python
import oniris

model = oniris.load_model('model.onnx')
engine = oniris.ShapeInferenceEngine.get_instance()
engine.infer_model(model)

# Check inferred shapes
graph = model.get_graph()
for node in graph.get_nodes():
    if node.has_inferred_shapes():
        print(f"{node.get_name()}: shapes inferred")
```

### Custom Operators

```python
import oniris

def my_custom_infer(ctx):
    # Custom shape inference logic
    output_shape = oniris.Shape([...])
    return oniris.InferenceResult.Success([output_shape])

oniris.register_custom_shape_inference("MyCustomOp", my_custom_infer)
```

## Project Structure

```
Oniris/
├── src/              # C++ source code
│   ├── core/        # Core types and utilities
│   ├── ir/          # Intermediate Representation
│   ├── passes/      # Optimization passes
│   └── python/      # Python bindings
├── python/oniris/   # Python package
├── tests/           # Unit and system tests
├── examples/        # Usage examples
└── docs/            # Documentation
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started with Oniris
- [API Documentation](docs/API.md) - Complete API reference
- [Project Summary](docs/PROJECT_SUMMARY.md) - Architecture and design overview
- [Agent Guidelines](docs/AGENTS.md) - Development guidelines

## Testing

```bash
# Run all tests (using scripts)
./scripts/test.sh

# Or manually:
pytest tests/                    # Python tests
cd build && ctest --output-on-failure  # C++ tests
```

## Supported Operators

Oniris supports 130+ ONNX operators including:

- **Math (40+)**: Abs, Neg, Floor, Ceil, Exp, Log, Sin, Cos, Tanh, Erf, etc.
- **Element-wise (20+)**: Add, Sub, Mul, Div, Pow, Min, Max, etc.
- **Linear Algebra (3)**: MatMul, Gemm, QGemm (com.microsoft)
- **Convolution (10+)**: Conv, ConvTranspose, MaxPool, AveragePool, GlobalPool
- **Normalization (5+)**: BatchNormalization, LayerNormalization, GroupNormalization
- **Shape (20+)**: Reshape, Transpose, Concat, Split, Squeeze, Unsqueeze, Gather
- **Reduction (12)**: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ArgMax, ArgMin
- **Activation (20+)**: ReLU, GELU, Mish, HardSwish, Softmax, Sigmoid, etc.
- **Quantization (9)**: QuantizeLinear, QLinearConv, QLinearMatMul, QGemm, etc.

See [API Documentation](docs/API.md) for the full list.

## Building from Source

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+
- Python 3.8+

### Quick Build (using scripts)

```bash
# Setup development environment
./scripts/setup.sh

# Or manual steps:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
pip install -e .
```

See `scripts/README.md` for all available build scripts.

### CMake Options

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_PYTHON_BINDINGS=ON
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a new Pull Request

Please ensure:
- Code follows the existing style
- All tests pass
- New features include tests
- Documentation is updated

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- Built with [pybind11](https://github.com/pybind/pybind11)
- ONNX Model Zoo for test models
