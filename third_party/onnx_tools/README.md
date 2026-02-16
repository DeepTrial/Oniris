# ONNX Model Modification Tools

A set of utilities for modifying ONNX models, providing both Python API and CLI interfaces.

## Features

- ✅ **Generic Layer Addition** - Add any ONNX operator or Microsoft domain operator via unified API
- ✅ **Flexible Initializer Specification** - Specify weights/constants as data, shapes, or auto-generate
- ✅ **Modify Tensor Shapes** - Change input/output tensor dimensions  
- ✅ **Replace Weights** - Swap initializers with numpy data
- ✅ **Remove Nodes** - Delete nodes and optionally reconnect inputs
- ✅ **Rename Operations** - Rename nodes and tensors
- ✅ **Chain Modifications** - Fluent API for multiple operations
- ✅ **CLI Interface** - Command-line tool for quick edits

## Installation

```bash
# Requirements
pip install onnx numpy
```

## Quick Start

### New Flexible API for Adding Layers

```python
from third_party.onnx_tools import add_layer
import numpy as np

# Simple - single input from another tensor
model = add_layer(model, "Relu", "input", "output", name="relu1")

# Multiple inputs from other tensors
model = add_layer(model, "Add", ["a", "b"], "output", name="add1")

# Conv with initializer specifications (NEW API)
model = add_layer(model, "Conv",
                  inputs={
                      "X": "input_data",  # from another node
                      "W": {"name": "weight", "data": weight_array},  # from numpy array
                      "B": {"name": "bias", "shape": [64], "type": np.float32}  # auto-generate
                  },
                  outputs="conv_out",
                  name="conv1",
                  attributes={"kernel_shape": [3, 3], "strides": [1, 1]})

# Elementwise with constant initializer
model = add_layer(model, "Mul",
                  inputs={
                      "A": "input",
                      "B": {"name": "scale", "data": np.array([0.5])}
                  },
                  outputs="scaled")
```

### Input Specification Formats

The `inputs` parameter accepts three formats:

1. **String** - Single input tensor name:
```python
add_layer(model, "Relu", "input", "output")
```

2. **List** - Multiple input tensor names (all from other nodes):
```python
add_layer(model, "Concat", ["a", "b", "c"], "output", attributes={"axis": 0})
```

3. **Dict** - Map formal names to specs (supports initializers):
```python
inputs={
    "X": "input_tensor",  # from another node
    "W": {"name": "weight", "data": weight_array},  # use existing array
    "B": {"name": "bias", "shape": [64]}  # auto-generate with shape
}
```

### Initializer Specification

When using dict format, each input can be:

```python
# Simple string - tensor from another node
"X": "input_data"

# Dict with existing data
"W": {"name": "weight", "data": np.random.randn(64, 3, 3, 3).astype(np.float32)}

# Dict with shape - auto-generate (weights use small random, bias use zeros)
"B": {"name": "bias", "shape": [64], "type": np.float32}

# Short form - just the numpy array (name auto-generated)
"W": {"data": weight_array}

# Short form - just the shape (name auto-generated)
"B": {"shape": [64]}
```

### Legacy API (Still Supported)

```python
# Auto-generate Conv weights
model = add_layer(model, "Conv", "input", "conv_out",
                  name="conv1",
                  kernel_size=3, in_channels=3, out_channels=64,
                  stride=1, padding=1, bias=True)

# Gemm with auto-generated weights
model = add_layer(model, "Gemm", "flatten", "fc_out",
                  name="fc1",
                  in_features=512, out_features=10, bias=True)
```

### Microsoft Domain Operators

```python
# FusedConv (Conv + ReLU)
model = add_layer(model, "FusedConv",
                  inputs={
                      "X": "input",
                      "W": {"shape": [64, 3, 3, 3]},
                      "B": {"shape": [64]}
                  },
                  outputs="fused_out",
                  name="fused1",
                  domain="com.microsoft",
                  attributes={"kernel_shape": [3, 3], "activation": "Relu"})

# Attention layer
model = add_layer(model, "Attention",
                  inputs={
                      "input": "query",
                      "weight": {"shape": [768, 768]},
                      "bias": {"shape": [768]}
                  },
                  outputs="attn_out",
                  name="attn1",
                  domain="com.microsoft",
                  attributes={"num_heads": 12})
```

### Chaining with ModelModifier

```python
from third_party.onnx_tools import ModelModifier

modifier = ModelModifier("model.onnx")
modifier \
    .add_layer("Conv",
               inputs={
                   "X": "input",
                   "W": {"shape": [32, 3, 3, 3]},
                   "B": {"shape": [32]}
               },
               outputs="c1",
               attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}) \
    .add_layer("Relu", "c1", "r1", name="relu1") \
    .add_layer("MaxPool", "r1", "p1", name="pool1",
               attributes={"kernel_shape": [2, 2], "strides": [2, 2]}) \
    .add_conv("conv2", "p1", "c2", 32, 64, 3, padding=1) \
    .save("modified.onnx")
```

## API Reference

### Core Function: `add_layer`

```python
def add_layer(
    model: onnx.ModelProto,
    op_type: str,                    # Operator type
    inputs: Union[str, List[str], Dict[str, Union[str, Dict]]],  # Input specification
    outputs: Union[str, List[str]],  # Output tensor name(s)
    name: Optional[str] = None,      # Node name
    domain: str = "",                # Operator domain
    attributes: Optional[Dict] = None,  # Operator attributes
    initializers: Optional[Dict[str, np.ndarray]] = None,  # Legacy initializers
    **kwargs                        # Layer-specific legacy parameters
) -> onnx.ModelProto:
```

### Input Specification Details

**Format 1: String (Single Input)**
```python
inputs="input_tensor"
# Equivalent to: inputs={"X": "input_tensor"}
```

**Format 2: List (Multiple Inputs from Tensors)**
```python
inputs=["tensor_a", "tensor_b", "tensor_c"]
# Maps to formal names: X=tensor_a, W=tensor_b, B=tensor_c, ...
```

**Format 3: Dict (Full Control with Initializers)**

Keys are formal input names from ONNX spec (X, W, B for Conv; A, B, C for Gemm, etc.)

```python
inputs={
    # String value: tensor from another node
    "X": "input_data",
    
    # Dict with 'data': use numpy array as initializer
    "W": {"name": "conv_w", "data": weight_array},
    
    # Dict with 'shape': auto-generate initializer
    "B": {"name": "conv_b", "shape": [64], "type": np.float32},
    
    # Minimal form: just the array
    "Z": {"data": np.array([1.0])},
    
    # Minimal form: just the shape
    "another": {"shape": [3, 3]}
}
```

### Convenience Functions

```python
# Conv with auto-generated weights
add_conv(model, name, input, output, in_ch, out_ch, kernel_size, ...)

# Linear/Gemm with auto-generated weights  
add_linear(model, name, input, output, in_feat, out_feat, bias=True, ...)

# Activations
add_activation(model, name, input, output, activation="relu", ...)

# Normalization
add_norm(model, name, input, output, norm_type="batchnorm", num_features=64)

# Pooling
add_pooling(model, name, input, output, pool_type="max", kernel_size=2, ...)

# Dropout
add_dropout(model, name, input, output, ratio=0.5)

# Shape manipulation (Flatten, Reshape, Transpose, Squeeze, Unsqueeze)
add_shape_manipulation(model, name, input, output, op_type="Flatten", ...)

# Microsoft fused ops
add_fused_conv(model, name, input, output, in_ch, out_ch, kernel_size, activation="")
add_fused_gemm(model, name, input, output, in_feat, out_feat, activation="")
add_attention(model, name, input, output, num_heads, domain="com.microsoft")
```

### Other Functions

```python
# Tensor operations
modify_tensor_shape(model, tensor_name, new_shape)
get_tensor_shape(model, tensor_name)
set_tensor_shape(model, tensor_name, new_shape)

# Initializer operations
replace_initializer(model, initializer_name, numpy_array, name=None)
replace_initializer_from_file(model, initializer_name, numpy_file, name=None)

# Node operations
remove_node(model, node_name, reconnect_inputs=True)
insert_node(model, node, before=None, after=None)
find_node_by_name(model, node_name)
find_nodes_by_op(model, op_type)

# Rename operations
rename_node(model, old_name, new_name)
rename_tensor(model, old_name, new_name)
```

## Complete Examples

### Build CNN with Manual Weights

```python
import numpy as np
from third_party.onnx_tools import add_layer

# Pre-defined weights
conv1_w = np.random.randn(32, 3, 3, 3).astype(np.float32) * 0.1
conv1_b = np.zeros(32, dtype=np.float32)
conv2_w = np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1
conv2_b = np.zeros(64, dtype=np.float32)

model = add_layer(model, "Conv",
                  inputs={
                      "X": "input",
                      "W": {"name": "conv1_W", "data": conv1_w},
                      "B": {"name": "conv1_B", "data": conv1_b}
                  },
                  outputs="c1",
                  name="conv1",
                  attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]})

model = add_layer(model, "Relu", "c1", "r1", name="relu1")
model = add_layer(model, "MaxPool", "r1", "p1", name="pool1",
                  attributes={"kernel_shape": [2, 2], "strides": [2, 2]})

model = add_layer(model, "Conv",
                  inputs={
                      "X": "p1",
                      "W": {"name": "conv2_W", "data": conv2_w},
                      "B": {"name": "conv2_B", "data": conv2_b}
                  },
                  outputs="c2",
                  name="conv2",
                  attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]})
```

### Build CNN with Auto-Generated Weights

```python
from third_party.onnx_tools import ModelModifier

modifier = ModelModifier("input.onnx")
modifier \
    .add_layer("Conv",
               inputs={
                   "X": "input",
                   "W": {"shape": [32, 3, 3, 3]},  # auto-generate
                   "B": {"shape": [32]}
               },
               outputs="c1",
               attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}) \
    .add_layer("Relu", "c1", "r1") \
    .add_layer("MaxPool", "r1", "p1",
               attributes={"kernel_shape": [2, 2], "strides": [2, 2]}) \
    .add_layer("Flatten", "p1", "f") \
    .add_layer("Gemm",
               inputs={
                   "A": "f",
                   "B": {"shape": [64*16*16, 10]},
                   "C": {"shape": [10]}
               },
               outputs="output",
               attributes={"transB": 1}) \
    .save("cnn.onnx")
```

### Mix Data Sources

```python
# Some weights from file, some auto-generated
pretrained_w = np.load("pretrained_conv.npy")

model = add_layer(model, "Conv",
                  inputs={
                      "X": "input",
                      "W": {"name": "conv_W", "data": pretrained_w},  # from file
                      "B": {"shape": [64]}  # auto-generate
                  },
                  outputs="conv_out",
                  name="conv1",
                  attributes={"kernel_shape": [3, 3]})
```

## Supported Operators

### Standard ONNX (125+)

Conv, ConvTranspose, Gemm, MatMul, BatchNormalization, InstanceNormalization, 
LayerNormalization, GroupNormalization, Relu, Sigmoid, Tanh, LeakyRelu, Elu, 
Selu, Softmax, LogSoftmax, Gelu, HardSigmoid, Mish, MaxPool, AveragePool, 
GlobalMaxPool, GlobalAveragePool, Flatten, Reshape, Transpose, Squeeze, 
Unsqueeze, Concat, Split, Add, Sub, Mul, Div, Pow, ReduceMean, ReduceSum, etc.

### Microsoft Domain (39+)

FusedConv, FusedGemm, FusedMatMul, Attention, MultiHeadAttention, 
SkipLayerNormalization, EmbeddingLayerNormalization, BiasGelu, FastGelu, 
GroupNorm, GroupQueryAttention, NhwcConv, NhwcMaxPool, QLinearConv, 
QLinearMatMul, RotaryEmbedding, DecoderMaskedSelfAttention, etc.

## CLI Reference

```bash
# Modify tensor shape
python -m third_party.onnx_tools.cli modify-shape model.onnx input "1,3,320,320" -o out.onnx

# Replace weights
python -m third_party.onnx_tools.cli replace-weight model.onnx conv1.weight weights.npy -o out.onnx

# Remove node
python -m third_party.onnx_tools.cli remove-node model.onnx dropout_1 -o out.onnx

# Rename tensor
python -m third_party.onnx_tools.cli rename-tensor model.onnx input_0 image -o out.onnx

# Inspect model
python -m third_party.onnx_tools.cli inspect model.onnx
```

## License

This is a third-party utility for the Oniris project.
