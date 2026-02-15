# Oniris Quick Start Guide

This guide will help you get started with Oniris in just a few minutes.

## Installation

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+
- Python 3.8+ (for Python bindings)

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd Oniris

# Install dependencies
pip install -r requirements.txt

# Build:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
pip install -e .
```

## Quick Examples

### 1. Simplify an ONNX Model

```python
import oniris

# Simplify a model file
result = oniris.simplify('input.onnx', 'output.onnx')
print(f"Simplified: {result.num_changes} changes")
```

### 2. Shape Inference

```python
import oniris

# Load a model
model = oniris.load_model('model.onnx')

# Run shape inference
engine = oniris.ShapeInferenceEngine.get_instance()
success = engine.infer_model(model, fail_on_unknown=False)

# Check results
graph = model.get_graph()
for node in graph.get_nodes():
    if node.has_inferred_shapes():
        print(f"{node.get_name()}: shapes inferred")
```

### 3. Create a Model Programmatically

```python
import oniris

# Create model
model = oniris.Model(8)
opset = oniris.OpsetImport()
opset.version = 13
model.add_opset_import(opset)

# Create graph
graph = model.create_graph("my_model")

# Add inputs
input_info = oniris.ValueInfo()
input_info.name = "input"
input_info.shape = oniris.Shape([1, 3, 224, 224])
input_info.dtype = oniris.DataType.FLOAT32
graph.add_input(input_info)

# Add nodes
conv = graph.create_node("Conv", "conv1")
conv.add_input("input")
conv.add_input("weight")
conv.add_output("output")
conv.set_attribute_ints("kernel_shape", [3, 3])
conv.set_attribute_ints("pads", [1, 1, 1, 1])

# Add outputs
output_info = oniris.ValueInfo()
output_info.name = "output"
graph.add_output(output_info)

# Validate
valid, msg = model.validate()
print(f"Valid: {valid}")

# Save
oniris.save_model(model, 'my_model.onnx')
```

### 4. Custom Shape Inference

```python
import oniris

# Define custom inference function
def my_custom_infer(ctx):
    # Input shapes are available in ctx.input_shapes
    input_shape = ctx.input_shapes[0]
    
    # Access attributes
    axis = ctx.get_attribute("axis")
    
    # Compute output shape
    output_shape = oniris.Shape([...])
    
    return oniris.InferenceResult.Success([output_shape])

# Register the handler
oniris.register_custom_shape_inference("MyCustomOp", my_custom_infer)

# Now MyCustomOp will use this function for shape inference
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run C++ unit tests only
cd build && ctest --output-on-failure

# Run Python tests only
pytest tests/unit -v

# Run system tests (downloads models)
pytest tests/system -v
```

## Command Line Usage

```bash
# Simplify a model
python -c "import oniris; oniris.simplify('in.onnx', 'out.onnx')"

# Print model info
python -c "import oniris; oniris.print_model_summary(oniris.load_model('model.onnx'))"
```

## Next Steps

- Read the [full API documentation](API.md)
- Check out the [examples](../examples/)
- Review the [system tests](../tests/system/) for more complex use cases

## Troubleshooting

### Build Issues

**CMake not found:**
```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake

# Windows (with chocolatey)
choco install cmake
```

**Compiler doesn't support C++17:**
- GCC 7+ or Clang 5+ required
- Check with: `g++ --version`

### Import Errors

**Module not found:**
```bash
# Make sure you installed the package
pip install -e .
```

**Symbol not found:**
- Rebuild the C++ extension: `python setup.py build_ext --inplace`

### Shape Inference Issues

**Unknown operators:**
```python
# Register a custom handler
oniris.register_custom_shape_inference("UnknownOp", custom_infer)

# Or skip unknown ops
engine.infer_model(model, fail_on_unknown=False)
```

## Getting Help

- Check the [documentation](../docs/)
- Review [test cases](../tests/) for examples
- Open an issue on GitHub
