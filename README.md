# Oniris - ONNX Compilation Toolkit

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Oniris is a high-performance ONNX model compilation and optimization toolkit written in C++ with Python bindings.

## Features

- **🌐 Web Visualizer**: Interactive web-based model visualization with pan, zoom, and editing
- **🔧 Model Simplification**: Simplify ONNX models similar to onnxsim, with graceful handling of unsupported layers
- **📐 Shape Inference**: Comprehensive shape inference supporting 165+ ONNX operators with both dynamic and static shapes
- **🔍 Subgraph Matching**: Find and match subgraph patterns with flexible topology and attribute constraints
- **🛠️ Model Editing**: Add/remove layers, modify shapes via web UI or Python API
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

**Simplification Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `skip_shape_inference` | Skip shape inference | `False` |
| `skip_constant_folding` | Skip constant folding | `False` |
| `skip_constant_to_initializer` | Skip converting Constant nodes to initializers | `False` |
| `skip_dead_node_elimination` | Skip dead node elimination | `False` |
| `skip_identity_elimination` | Skip Identity node elimination | `False` |
| `skip_transpose_elimination` | Skip nop Transpose elimination | `False` |
| `skip_reshape_elimination` | Skip nop Reshape elimination | `False` |
| `skip_pad_elimination` | Skip nop Pad elimination | `False` |
| `skip_slice_elimination` | Skip nop Slice elimination | `False` |
| `fuse_conv_bn` | Enable Conv+BatchNorm fusion | `True` |
| `fuse_conv_relu` | Enable Conv+ReLU fusion | `True` |
| `fuse_gemm_activation` | Enable Gemm+Activation fusion | `True` |
| `fuse_gemm_bias` | Enable Gemm+Bias fusion | `True` |
| `max_iterations` | Maximum optimization iterations | `10` |

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

### Web Visualizer

Launch the interactive web-based model visualizer:

```bash
# Using the convenience script (recommended)
./scripts/start_web.sh

# Or manually
cd third_party/web
pip install -r requirements.txt
python server.py

# Open http://localhost:5000 in your browser
```

Features:
- **Visualize** ONNX models with interactive graph rendering
- **Navigate** large models (LLM support) with pan/zoom and minimap
- **Edit** models visually - add/remove layers, run shape inference
- **Export** modified models

### Subgraph Matching

Find and match subgraph patterns in ONNX models using **tensor-flow based patterns** (inspired by onnx_matcher):

For activation fusion patterns and tensor-flow based matching:

```python
# Swish activation: x * sigmoid(x)
pattern = oniris.OnnxMatcherPattern.from_string("""
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
""")

matches = oniris.Matcher.find_all(model, pattern)
```

**Syntax:**
- `OpType(input, output)` - Basic node
- `Conv/Pool(?, c0)` - Match Conv OR Pool
- `?(?, ?)` - Wildcard for any op type
- `[a, b]` - Multi-input/output

See [ONNX Matcher Style](docs/ONNX_MATCHER_STYLE.md) for details.

### ONNX Tools (Model Modification)

```python
from third_party.onnx_tools import (
    modify_tensor_shape,
    replace_initializer,
    add_layer,
    add_conv,
    remove_node,
)
import onnx

# Load a model
model = onnx.load('model.onnx')

# Modify tensor shape
model = modify_tensor_shape(model, 'input', [1, 3, 224, 224])

# Replace weights
import numpy as np
new_weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
model = replace_initializer(model, 'conv1.weight', new_weights)

# Add a new Conv layer using generic API
model = add_layer(
    model, 
    op_type='Conv',
    inputs='input',
    outputs='conv_out',
    name='new_conv',
    kernel_size=3,
    in_channels=3,
    out_channels=64
)

# Or use convenience function
model = add_conv(
    model,
    input='input',
    output='conv_out',
    name='conv1',
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

# Remove a node
model = remove_node(model, 'dropout_1', reconnect_inputs=True)

# Save the modified model
onnx.save(model, 'modified.onnx')
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
├── src/                    # C++ source code
│   ├── core/              # Core types and utilities
│   │   ├── logger.cpp/hpp
│   │   └── types.cpp/hpp  # DataType, Shape, Dimension
│   ├── ir/                # Intermediate Representation
│   │   ├── node.cpp/hpp
│   │   ├── graph.cpp/hpp
│   │   ├── model.cpp/hpp
│   │   └── tensor.hpp
│   ├── passes/            # Optimization passes
│   │   ├── shape_inference.cpp/hpp  # 165+ operators
│   │   ├── simplifier.cpp/hpp       # Model simplification
│   │   └── onnx_matcher_style.cpp/hpp # Tensor-flow based pattern matching
│   ├── python/            # Python bindings (pybind11)
│   │   └── bindings.cpp
│   └── utils/             # ONNX utilities
│       └── onnx_utils.cpp/hpp
├── python/oniris/         # Python package
│   └── __init__.py        # Main API exports
├── third_party/           # Third-party tools and dependencies
│   ├── onnx_tools/        # ONNX model modification tools
│   │   ├── __init__.py    # Main exports
│   │   ├── model_modifier.py   # Tensor/node operations
│   │   ├── layer_builder.py    # Generic layer addition (164+ ops)
│   │   ├── cli.py         # Command-line interface
│   │   └── examples.py    # Usage examples
│   └── web/               # Web-based visualizer
│       ├── server.py      # Flask server entry point
│       ├── backend/       # Flask API server
│       │   ├── app.py     # REST API endpoints
│       │   └── spatial_index.py  # Spatial indexing for large models
│       ├── frontend/      # CSS/JS assets
│       │   ├── css/style.css
│       │   └── js/app.js  # Main frontend application
│       ├── templates/     # HTML templates
│       │   └── index.html
│       └── requirements.txt
├── scripts/               # Build and utility scripts
│   ├── setup.sh           # Initialize development environment
│   ├── build.sh           # Build C++ library and Python bindings
│   ├── test.sh            # Run tests
│   ├── install.sh         # Install package
│   ├── start_web.sh       # Start web visualizer
│   ├── lint.sh            # Run linters/formatters
│   ├── package.sh         # Create distribution packages
│   ├── generate_from_proto.py  # Generate types from ONNX proto
│   └── update_onnx_version.py  # Update ONNX version
├── tests/                 # Unit and system tests
│   ├── unit/              # C++ unit tests (CMake)
│   └── system/            # Python system tests
│       ├── test_models.py
│       ├── test_onnx_tools.py
│       └── test_onnx_tools_layer_builder.py
├── examples/              # Usage examples
│   ├── simple_example.py           # Basic API usage
│   ├── fusion_example.py           # Fusion control demo
│   └── onnx_matcher_style_example.py # ONNX matcher pattern matching demo
├── docs/                  # Documentation
│   ├── QUICKSTART.md      # Quick start guide
│   ├── API.md             # Complete API reference
│   ├── ONNX_MATCHER_STYLE.md # Tensor-flow based pattern matching spec
│   ├── PROJECT_SUMMARY.md # Architecture overview
│   └── AGENTS.md          # Development guidelines
├── requirements.txt       # Python dependencies
├── setup.py              # Python package setup
└── CMakeLists.txt        # CMake build configuration
```

## Third-Party Components

### 1. ONNX Tools (`third_party/onnx_tools/`)

A comprehensive Python toolkit for ONNX model modification, supporting **164+ operators** including standard ONNX and Microsoft domain ops.

**Features:**

| Feature | Description |
|---------|-------------|
| **Tensor Operations** | Modify tensor shapes, get/set tensor dimensions |
| **Initializer Operations** | Replace weights from numpy arrays or files |
| **Node Operations** | Remove nodes with automatic connection rewiring, insert nodes |
| **Rename Operations** | Rename nodes and tensors |
| **Generic Layer Addition** | Add any ONNX operator with simplified API |
| **Microsoft Domain Support** | Full support for `com.microsoft` operators (FusedConv, Attention, etc.) |

**Supported Operator Categories:**
- **Convolution**: Conv, ConvTranspose, ConvInteger
- **Linear**: Gemm, MatMul, MatMulInteger
- **Activations**: ReLU, Sigmoid, Tanh, GELU, LeakyReLU, etc.
- **Normalization**: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
- **Pooling**: MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool
- **Shape Ops**: Reshape, Transpose, Flatten, Squeeze, Unsqueeze
- **Microsoft Fused**: FusedConv, FusedGemm, Attention, QAttention, etc.

**CLI Usage:**
```bash
# Modify tensor shape
python -m third_party.onnx_tools.cli modify-shape model.onnx input "1,3,320,320" -o output.onnx

# Replace weights
python -m third_party.onnx_tools.cli replace-weight model.onnx conv1.weight new_weights.npy -o output.onnx

# Remove a node
python -m third_party.onnx_tools.cli remove-node model.onnx dropout_1 -o output.onnx

# Rename tensor
python -m third_party.onnx_tools.cli rename-tensor model.onnx input_0 image_input -o output.onnx

# Inspect model structure
python -m third_party.onnx_tools.cli inspect model.onnx
```

See `third_party/onnx_tools/__init__.py` for detailed API documentation.

### 2. Web Visualizer (`third_party/web/`)

A web-based ONNX model visualizer with integrated editing capabilities, inspired by Netron.

**Features:**

| Feature | Description |
|---------|-------------|
| **Visualization** | Interactive graph rendering using Netron's grapher.js |
| **Navigation** | Pan, zoom, minimap for large models (LLM support) |
| **Shape Inference** | Run shape inference on the model |
| **Simplification** | Simplify models by removing redundant operations |
| **Model Editing** | Add/remove layers, modify tensor shapes |
| **Export** | Download modified models |
| **Performance** | Viewport-based rendering for >500 node models |

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main visualization page |
| `/api/model/upload` | POST | Upload ONNX model |
| `/api/model/<id>/layout` | POST | Compute graph layout |
| `/api/model/<id>/shape_inference` | POST | Run shape inference |
| `/api/model/<id>/simplify` | POST | Simplify model |
| `/api/model/<id>/export` | GET | Export modified model |
| `/api/model/<id>/add_layer` | POST | Add a layer |
| `/api/model/<id>/remove_node` | POST | Remove a node |
| `/api/ops/schemas` | GET | List operator schemas |

See `third_party/web/README.md` for detailed documentation.

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started with Oniris
- [API Documentation](docs/API.md) - Complete API reference
- [ONNX Matcher Style](docs/ONNX_MATCHER_STYLE.md) - Tensor-flow based subgraph pattern matching
- [Project Summary](docs/PROJECT_SUMMARY.md) - Architecture and design overview
- [Agent Guidelines](docs/AGENTS.md) - Development guidelines
- [Build Scripts](scripts/README.md) - Build system documentation

## Testing

```bash
# Run all tests (using scripts)
./scripts/test.sh

# Or manually:
pytest tests/                    # Python tests
cd build && ctest --output-on-failure  # C++ tests

# System tests (downloads real models)
./scripts/test.sh system
```

## Supported Operators

Oniris supports 165+ ONNX operators including:

- **Math (40+)**: Abs, Neg, Floor, Ceil, Exp, Log, Sin, Cos, Tanh, Erf, etc.
- **Element-wise (20+)**: Add, Sub, Mul, Div, Pow, Min, Max, etc.
- **Linear Algebra (5+)**: MatMul, Gemm, QGemm, FusedGemm, FusedMatMul
- **Convolution (10+)**: Conv, ConvTranspose, MaxPool, AveragePool, GlobalPool, FusedConv
- **Normalization (8+)**: BatchNormalization, LayerNormalization, GroupNormalization, RmsNorm, SkipLayerNormalization
- **Shape (20+)**: Reshape, Transpose, Concat, Split, Squeeze, Unsqueeze, Gather
- **Reduction (12)**: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ArgMax, ArgMin
- **Activation (25+)**: ReLU, GELU, FastGelu, Mish, HardSwish, Softmax, Sigmoid, etc.
- **Quantization (15+)**: QuantizeLinear, QLinearConv, QLinearMatMul, QGemm, QLinearAdd, etc.
- **Attention (5+)**: Attention, MultiHeadAttention, DecoderAttention, EmbedLayerNormalization
- **Microsoft Domain (37+)**: Full support for com.microsoft domain operators

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

### Updating ONNX Proto Version

Oniris automatically downloads ONNX proto definition from GitHub during build. The proto file defines data types and IR format.

**Current ONNX Proto Version**: 1.20.1 (IR_VERSION 13)

**To update to a newer ONNX version:**

```bash
# Method 1: Using CMake option (recommended)
rm -rf build
mkdir build && cd build
cmake .. -DONNX_PROTO_VERSION="1.21.0"
make -j$(nproc)

# Method 2: Modifying CMakeLists.txt
# Edit CMakeLists.txt and change:
# set(ONNX_PROTO_VERSION "1.20.1" CACHE STRING "...")
# to desired version, then rebuild

# Method 3: Using local proto file
python3 scripts/generate_from_proto.py /path/to/onnx.proto
make -j$(nproc)
```

**Version Compatibility:**

| ONNX Version | IR_VERSION | New Data Types |
|--------------|------------|----------------|
| 1.14+        | 0x9        | FLOAT8E4M3FN, FLOAT8E5M2, etc. |
| 1.15+        | 0xA        | UINT4, INT4 |
| 1.17+        | 0xB        | FLOAT4E2M1 |
| 1.20+        | 0xD        | FLOAT8E8M0, UINT2, INT2 |

The generated `DataType` enum in `src/core/types.hpp` will automatically include all types from the specified ONNX version.

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
- Web visualizer uses [Netron](https://github.com/lutzroeder/netron) grapher code
- ONNX Model Zoo for test models
