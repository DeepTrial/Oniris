# Oniris API Documentation

## Overview

Oniris provides both C++ and Python APIs for ONNX model compilation and optimization.

## Table of Contents

- [Python API](#python-api)
  - [Core Types](#core-types)
  - [Intermediate Representation](#intermediate-representation)
  - [Shape Inference](#shape-inference)
  - [Model Simplification](#model-simplification)
  - [Utilities](#utilities)
- [C++ API](#c-api)
  - [Core Types](#c-core-types)
  - [IR Classes](#c-ir-classes)
  - [Passes](#c-passes)

---

## Python API

### Core Types

#### `DataType`

Enumeration of ONNX data types.

```python
import oniris

# Available data types
oniris.DataType.FLOAT32
oniris.DataType.INT64
oniris.DataType.UINT8
# ... etc
```

#### `Dimension`

Represents a tensor dimension that can be static or dynamic.

```python
# Static dimension
dim = oniris.Dimension(224)
print(dim.is_dynamic())  # False
print(dim.get_static_value())  # 224

# Dynamic dimension
dim = oniris.Dimension("batch_size")
print(dim.is_dynamic())  # True
print(dim.get_symbolic_name())  # "batch_size"
```

#### `Shape`

Multi-dimensional shape composed of `Dimension` objects.

```python
# Static shape
shape = oniris.Shape([1, 3, 224, 224])
print(shape.num_dims())  # 4
print(shape.is_static())  # True

# Dynamic shape
shape = oniris.Shape([oniris.Dimension("batch"), 3, 224, 224])
print(shape.is_dynamic())  # True
print(shape.get_dim(0).get_symbolic_name())  # "batch"

# Convert to/from list
shape_list = shape.to_list()  # ["batch", 3, 224, 224]
```

### Intermediate Representation

#### `Model`

Top-level ONNX model representation.

```python
# Create a new model
model = oniris.Model(ir_version=8)
model.set_producer_name("my_tool")
model.set_producer_version("1.0.0")

# Add opset
opset = oniris.OpsetImport()
opset.domain = ""
opset.version = 13
model.add_opset_import(opset)

# Create main graph
graph = model.create_graph("main")

# Validate
valid, error_msg = model.validate()
```

#### `Graph`

Computation graph containing nodes and tensors.

```python
graph = model.get_graph()

# Add nodes
conv = graph.create_node("Conv", "conv1")
conv.add_input("input")
conv.add_input("weight")
conv.add_output("output")
conv.set_attribute_ints("kernel_shape", [3, 3])
conv.set_attribute_ints("strides", [1, 1])

# Add inputs/outputs
input_info = oniris.ValueInfo()
input_info.name = "input"
input_info.shape = oniris.Shape([1, 3, 224, 224])
input_info.dtype = oniris.DataType.FLOAT32
graph.add_input(input_info)

# Topological sort
sorted_nodes = graph.topological_sort()

# Validate
valid, error_msg = graph.validate()
```

#### `Node`

Represents an operation in the graph.

```python
node = graph.create_node("Conv", "conv1")

# Inputs/outputs
node.add_input("x")
node.add_output("y")

# Attributes
node.set_attribute_int("axis", 1)
node.set_attribute_ints("kernel_shape", [3, 3])
node.set_attribute_float("alpha", 1.0)
node.set_attribute_string("auto_pad", "SAME")

# Check attributes
if node.has_attribute("axis"):
    value = node.get_attribute("axis")
```

### Shape Inference

#### `ShapeInferenceEngine`

Main interface for shape inference operations.

```python
engine = oniris.ShapeInferenceEngine.get_instance()

# Run inference on model
success = engine.infer_model(model, fail_on_unknown=False)

# Run inference on graph
success = engine.infer_graph(graph, fail_on_unknown=False)

# Check supported ops
supported_ops = engine.get_supported_ops()
print(f"Conv supported: {'Conv' in supported_ops}")

# Register custom handler
def my_custom_infer(context):
    # context.input_shapes: List of input Shape objects
    # context.attributes: Dict of node attributes
    output_shape = oniris.Shape([context.input_shapes[0].get_dim(0), 64])
    return oniris.InferenceResult.Success([output_shape])

oniris.register_custom_shape_inference("MyOp", my_custom_infer)
```

#### Custom Shape Inference

```python
import oniris

def custom_conv_infer(ctx):
    """Custom shape inference for a special Conv variant."""
    # Get input shape
    input_shape = ctx.input_shapes[0]
    
    # Get attributes
    kernel_shape = ctx.get_attribute("kernel_shape")
    strides = ctx.get_attribute("strides")
    
    # Calculate output shape
    # ... custom logic ...
    output_shape = oniris.Shape([...])
    
    return oniris.InferenceResult.Success([output_shape])

# Register the handler
oniris.register_custom_shape_inference("CustomConv", custom_conv_infer, domain="custom.domain")
```

### Model Simplification

#### `Simplifier`

Main interface for model simplification.

```python
# Simplify a model
options = oniris.SimplifyOptions()
options.skip_constant_folding = False
options.skip_dead_node_elimination = False
options.fail_on_unsupported = False
options.max_iterations = 10

result = oniris.Simplifier.simplify(model, options)

print(f"Success: {result.success}")
print(f"Changes: {result.num_changes}")
print(f"Iterations: {result.num_iterations}")

# Simplify a graph
result = oniris.Simplifier.simplify_graph(graph, options)
```

#### Convenience Function

```python
# Simplify ONNX file directly
result = oniris.simplify("input.onnx", "output.onnx")
```

### Utilities

```python
# Load/save models
model = oniris.load_model("model.onnx")
oniris.save_model(model, "output.onnx")

# Check if valid ONNX file
is_valid = oniris.is_valid_onnx_file("model.onnx")

# Get model info
info = oniris.get_model_info(model)
print(f"Nodes: {info.num_nodes}")
print(f"Inputs: {info.num_inputs}")
print(f"Ops used: {info.ops_used}")

# Print summary
oniris.print_model_summary(model)
```

---

## C++ API

### C++ Core Types

#### `Dimension`

```cpp
#include "core/types.hpp"

// Static dimension
oniris::Dimension dim(224);
bool is_dynamic = dim.IsDynamic();  // false
int64_t value = dim.GetStaticValue();  // 224

// Dynamic dimension
oniris::Dimension dim("batch_size");
std::string symbol = dim.GetSymbolicName();  // "batch_size"

// Modify
dim.SetStaticValue(10);
dim.SetDynamic("new_symbol");
```

#### `Shape`

```cpp
// Static shape
oniris::Shape shape({1, 3, 224, 224});
size_t rank = shape.NumDims();  // 4
bool is_static = shape.IsStatic();  // true

// Dynamic shape
oniris::Shape shape;
shape.AddDim(oniris::Dimension("batch"));
shape.AddDim(3);
shape.AddDim(224);
shape.AddDim(224);

// Access dimensions
const oniris::Dimension& dim = shape.GetDim(0);

// Get total size (if static)
auto total = shape.GetTotalSize();  // std::optional<int64_t>
```

### C++ IR Classes

#### `Model`

```cpp
#include "ir/model.hpp"

// Create model
auto model = std::make_shared<oniris::Model>(8);
model->SetProducerName("my_tool");
model->SetProducerVersion("1.0.0");

// Add opset
oniris::OpsetImport opset;
opset.domain = "";
opset.version = 13;
model->AddOpsetImport(opset);

// Create graph
auto graph = model->CreateGraph("main");

// Validate
std::string error_msg;
bool valid = model->Validate(&error_msg);
```

#### `Graph`

```cpp
#include "ir/graph.hpp"

auto graph = std::make_shared<oniris::Graph>("my_graph");

// Create nodes
auto node = graph->CreateNode("Conv", "conv1");
node->AddInput("x");
node->AddOutput("y");

// Set attributes
node->SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});
node->SetAttribute("strides", std::vector<int64_t>{1, 1});

// Add inputs/outputs
oniris::ValueInfo input_info;
input_info.name = "input";
input_info.shape = oniris::Shape({1, 3, 224, 224});
input_info.dtype = oniris::DataType::kFloat32;
graph->AddInput(input_info);

// Topological sort
auto sorted = graph->TopologicalSort();

// Remove dead nodes
graph->RemoveDeadNodes();

// Validate
std::string error_msg;
bool valid = graph->Validate(&error_msg);
```

#### `Node`

```cpp
#include "ir/node.hpp"

auto node = std::make_shared<oniris::Node>("Conv", "conv1");

// Set inputs/outputs
node->AddInput("x");
node->AddOutput("y");

// Attributes
node->SetAttribute("axis", static_cast<int64_t>(1));
node->SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});

// Check and get attributes
if (node->HasAttribute("axis")) {
    auto value = node->GetAttributeAs<int64_t>("axis");
    if (value.has_value()) {
        int64_t axis = *value;
    }
}

// Clone
auto cloned = node->Clone();
```

### C++ Passes

#### Shape Inference

```cpp
#include "passes/shape_inference.hpp"

auto& engine = oniris::passes::ShapeInferenceEngine::GetInstance();

// Register custom handler
engine.Register("CustomOp", [](const oniris::passes::InferenceContext& ctx) {
    // ctx.input_shapes: std::vector<oniris::Shape>
    // ctx.input_dtypes: std::vector<oniris::DataType>
    // ctx.attributes: pointer to attribute map
    // ctx.graph: pointer to containing graph
    
    oniris::Shape output_shape = ctx.input_shapes[0];
    return oniris::passes::InferenceResult::Success({output_shape});
});

// Run inference
bool success = engine.InferGraph(graph, fail_on_unknown);
success = engine.InferModel(model, fail_on_unknown);
```

#### Model Simplification

```cpp
#include "passes/simplifier.hpp"

oniris::passes::SimplifyOptions options;
options.skip_constant_folding = false;
options.skip_dead_node_elimination = false;
options.fail_on_unsupported = false;
options.max_iterations = 10;

auto result = oniris::passes::Simplifier::Simplify(model, options);
// or
auto result = oniris::passes::Simplifier::SimplifyGraph(graph, options);

if (result.success) {
    std::cout << "Changes: " << result.num_changes << std::endl;
    std::cout << "Iterations: " << result.num_iterations << std::endl;
}
```

---

## Best Practices

### 1. Error Handling

Always check return values and handle errors gracefully:

```python
# Python
result = oniris.Simplifier.simplify(model, options)
if not result.success:
    print(f"Simplification failed: {result.error_msg}")
    return
```

```cpp
// C++
std::string error_msg;
if (!model->Validate(&error_msg)) {
    std::cerr << "Validation failed: " << error_msg << std::endl;
    return;
}
```

### 2. Dynamic Shapes

Handle dynamic shapes properly:

```python
# Check if dimension is dynamic
if shape.get_dim(0).is_dynamic():
    # Use symbolic name or treat as unknown
    symbol = shape.get_dim(0).get_symbolic_name()
else:
    value = shape.get_dim(0).get_static_value()
```

### 3. Custom Operators

Register custom handlers early in your application:

```python
# Register at module level or in __init__
def setup_custom_ops():
    oniris.register_custom_shape_inference("MyOp1", infer_op1)
    oniris.register_custom_shape_inference("MyOp2", infer_op2)
```

### 4. Performance

- Run shape inference before simplification for better results
- Set `max_iterations` based on model complexity
- Skip unnecessary passes with `SimplifyOptions`

---

## Examples

See the `examples/` directory for complete working examples:

- `simple_example.py` - Basic API usage
- `custom_ops_example.py` - Custom operator handling
- `benchmark_example.py` - Performance benchmarking
