# Oniris Project Summary

## Project Overview

**Oniris** is a high-performance ONNX model compilation and optimization toolkit developed in C++ with Python bindings. The project implements key ONNX model processing capabilities including model simplification (similar to onnxsim) and comprehensive shape inference with support for both static and dynamic shapes.

## Key Features

### 1. Comprehensive Shape Inference (120+ Operators)

Oniris supports shape inference for over 120 ONNX operators across all major categories:

| Category | Count | Operators |
|----------|-------|-----------|
| Math | 40+ | Abs, Neg, Floor, Ceil, Exp, Log, Sin, Cos, etc. |
| Activation | 20+ | ReLU, GELU, Mish, HardSwish, Softmax, etc. |
| Convolution | 10+ | Conv, ConvTranspose, Pooling, etc. |
| Normalization | 5+ | BatchNorm, LayerNorm, GroupNorm, etc. |
| Shape Ops | 20+ | Reshape, Transpose, Concat, Split, etc. |
| Reduction | 12 | ReduceSum, ReduceMean, ArgMax, etc. |
| Quantization | 8 | QuantizeLinear, QLinearConv, etc. |
| Control Flow | 3 | If, Loop, Scan |

### 2. Advanced Model Simplification (V2)

Feature-complete implementation matching onnx-simplifier:

**Fusion Passes:**
- Conv + BatchNormalization fusion
- Conv + ReLU fusion
- Gemm + Activation fusion

**Constant Folding:**
- Extended to 40+ operator types
- Shape computation folding
- Complex expression evaluation

**Nop Elimination:**
- Identity removal
- Nop transpose elimination
- Nop reshape elimination
- Nop pad/slice/resize elimination
- Single-input concat elimination

**Dead Code Elimination:**
- Unreachable node removal
- Unused constant cleanup

### 3. Dynamic Shape Support

- Symbolic dimension support (e.g., "batch", "seq_len")
- Dynamic shape propagation
- Conservative estimates for unknown dimensions

### 4. Extensible Architecture

**Custom Operator Registration:**
```cpp
// C++
engine.Register("CustomOp", [](const InferenceContext& ctx) {
    return InferenceResult::Success({output_shape});
});
```

```python
# Python
oniris.register_custom_shape_inference("CustomOp", custom_infer)
```

## Architecture

```
Oniris/
├── src/                      # Core C++ implementation
│   ├── core/                # Core utilities and types
│   │   ├── types.hpp/cpp    # Dimension, Shape, DataType
│   │   └── logger.hpp/cpp   # Logging utilities
│   ├── ir/                  # Intermediate Representation
│   │   ├── tensor.hpp       # Tensor and ValueInfo
│   │   ├── node.hpp/cpp     # Node (operation)
│   │   ├── graph.hpp/cpp    # Computation graph
│   │   └── model.hpp/cpp    # ONNX Model
│   ├── passes/              # Optimization passes
│   │   ├── shape_inference.hpp/cpp        # Shape inference (120+ operators)
│   │   └── simplifier.hpp/cpp             # Model simplification
│   ├── utils/               # Utilities
│   │   └── onnx_utils.hpp/cpp
│   └── python/              # Python bindings
│       └── bindings.cpp     # pybind11 bindings
├── python/oniris/           # Python package
├── tests/                   # Test suite
│   ├── unit/                # C++ and Python unit tests
│   │   ├── test_types.cpp
│   │   ├── test_ir.cpp
│   │   ├── test_shape_inference.cpp
│   │   └── test_simplifier.cpp
│   └── system/              # System tests with real models
├── docs/                    # Documentation
├── examples/                # Usage examples
└── Build files              # CMake, setup.py
```

## Implementation Details

### Shape Inference Engine

Registry-based architecture with 120+ handlers:

```cpp
// Registration example
void RegisterExtendedHandlers(ShapeInferenceEngine& engine) {
    // Math operators
    engine.Register("Abs", InferAbs);
    engine.Register("Exp", InferExp);
    // ... 120+ more
}
```

**Inference Context:**
- `input_shapes`: Vector of input Shape objects
- `input_dtypes`: Vector of input DataType objects
- `attributes`: Node attributes map
- `graph`: Pointer to containing graph

**Inference Result:**
- `output_shapes`: Vector of output shapes
- `output_dtypes`: Optional output data types
- `success`: Whether inference succeeded
- `error_msg`: Error message if failed

### Model Simplification

Iterative optimization until fixed point:

```cpp
int RunAllPasses(Graph& graph, const SimplifyOptions& options) {
    int total_changes = 0;
    
    // Identity eliminations
    total_changes += EliminateIdentity(graph);
    total_changes += EliminateNopTranspose(graph);
    total_changes += EliminateNopReshape(graph);
    
    // Fusion passes
    total_changes += FuseConvBN(graph);
    total_changes += FuseConvRelu(graph);
    
    // Constant folding
    total_changes += ConstantFoldingV2(graph);
    
    // Dead code elimination
    total_changes += EliminateDeadNodes(graph);
    
    return total_changes;
}
```

### Intermediate Representation

**ValueInfo**: Type and shape information
```cpp
struct ValueInfo {
    std::string name;
    Shape shape;
    DataType dtype;
};
```

**Node**: Operation with attributes
```cpp
class Node {
    std::string op_type_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, AttributeValue> attributes_;
};
```

**Graph**: Collection of nodes
```cpp
class Graph : public std::enable_shared_from_this<Graph> {
    std::vector<std::shared_ptr<Node>> nodes_;
    std::vector<ValueInfo> inputs_;
    std::vector<ValueInfo> outputs_;
    std::unordered_map<std::string, ConstantTensor> constants_;
};
```

## Testing Strategy

### Unit Tests

**Core Types:**
- Dimension (static/dynamic)
- Shape (broadcasting, validation)
- DataType conversions

**IR Classes:**
- Node creation and attributes
- Graph topology and validation
- Model serialization

**Shape Inference:**
- Individual operator tests (120+)
- Dynamic shape handling
- Broadcasting rules

**Simplification:**
- Fusion passes
- Nop elimination
- Constant folding

### System Tests

- Real ONNX model loading
- End-to-end simplification
- Shape inference validation
- Custom handler registration

## Performance Characteristics

### Time Complexity

- **Shape Inference**: O(N) where N = number of nodes
- **Simplification**: O(K*N) where K = iterations (typically 3-5)
- **Fusion**: O(E) where E = number of edges

### Memory Usage

- Lazy shape evaluation
- Shared pointer management
- Conservative copying

## Comparison with Other Tools

| Feature | Oniris | onnx-simplifier | ONNX Runtime |
|---------|--------|-----------------|--------------|
| Shape Inference | 120+ ops | ~50 ops | ~120 ops |
| Custom Ops | ✓ | ✗ | Limited |
| Fusion | ✓ | ✓ | ✓ |
| C++ API | ✓ | ✗ | ✓ |
| Python API | ✓ | ✓ | ✓ |
| Dynamic Shapes | Full | Limited | Full |

## Usage Examples

### Basic Usage

```python
import oniris

# Simplify a model
oniris.simplify('input.onnx', 'output.onnx')

# Disable specific fusion
options = oniris.SimplifyOptions()
options.fuse_conv_bn = False
oniris.Simplifier.simplify(model, options)
```

### Shape Inference

```python
model = oniris.load_model('model.onnx')
engine = oniris.ShapeInferenceEngine.get_instance()
engine.infer_model(model)
```

### Custom Operators

```python
def custom_infer(ctx):
    output_shape = oniris.Shape([...])
    return oniris.InferenceResult.Success([output_shape])

oniris.register_custom_shape_inference("MyOp", custom_infer)
```

## Build System

```bash
# Install dependencies
pip install -r requirements.txt

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Install Python package
pip install -e .

# Test
pytest tests/           # All tests
cd build && ctest      # C++ unit tests
pytest tests/unit      # Python unit tests
pytest tests/system    # System tests
```

## Documentation

- `../README.md` - Project overview
- `QUICKSTART.md` - Getting started guide
- `API.md` - Complete API reference
- `PROJECT_SUMMARY.md` - Architecture and design overview
- `AGENTS.md` - Development guidelines

## Code Statistics

- **C++ Files**: 25+
- **Core Classes**: 15+
- **Supported Operators**: 120+
- **Unit Tests**: 50+
- **Lines of Code**: ~8000+

## Future Roadmap

1. **More Operators**: Extend to 150+ operators
2. **Advanced Fusion**: Multi-branch fusion patterns
3. **Quantization**: INT8/FP16 optimization
4. **Profiling**: Performance analysis tools
5. **Visualization**: Graph visualization utilities

## License

MIT License

## Contributors

Oniris Team
