# ONNX Model Compiler

Oniris now includes a full-featured **ONNX Model Compiler** that implements a complete compilation pipeline:

```
Input ONNX Model → Optimization Pass → Shape Inference Pass → Pattern Matching → JSON Output
```

## Overview

The Model Compiler provides:

- **🔧 Optimization Passes**: Constant folding, dead code elimination, operator fusion
- **📐 Shape Inference**: Automatic shape propagation for 165+ operators
- **🔍 Pattern Matching**: User-defined pattern detection with tensor-flow based syntax
- **📊 JSON Output**: Structured compilation results for downstream tools

## Quick Start

### Basic Usage

```python
import oniris

# Create compiler and add patterns
compiler = oniris.ModelCompiler()
compiler.add_pattern("ConvRelu", """
    Conv(?, c0)
    Relu(c0, ?)
""")

# Configure options
options = oniris.CompilerOptions()
options.verbose = True

# Compile a model
result = compiler.compile("input.onnx", "output_optimized.onnx", options)

# Access results
print(f"Optimization changes: {result.optimization_stats.num_changes}")
print(f"Pattern matches: {result.pattern_matching_summary.total_matches}")

# Get JSON output
json_str = result.to_json(pretty=True)
print(json_str)
```

## API Reference

### ModelCompiler

Main compiler class that orchestrates the compilation pipeline.

```python
class ModelCompiler:
    def __init__(self)
    def add_pattern(self, name: str, pattern_string: str) -> bool
    def add_patterns(self, patterns: List[PatternDefinition])
    def clear_patterns(self)
    def get_pattern_count(self) -> int
    def get_pattern_names(self) -> List[str]
    def compile(self, input_path: str, output_path: str, options: CompilerOptions) -> CompilationResult
    def compile_model(self, model: Model, options: CompilerOptions) -> CompilationResult
    def run_pattern_matching(self, model: Model, match_type: PatternMatchType) -> PatternMatchingSummary
```

### PatternDefinition

Define patterns for subgraph matching.

```python
class PatternDefinition:
    def __init__(self, name: str, pattern_string: str)
    def parse(self) -> bool

# Example
pattern = oniris.PatternDefinition("Swish", """
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
""")
```

### CompilerOptions

Configure the compilation process.

```python
class CompilerOptions:
    simplify_options: SimplifyOptions      # Optimization options
    enable_optimization: bool = True       # Run optimization passes
    enable_shape_inference: bool = True    # Run shape inference
    fail_on_unknown_shape: bool = False    # Fail if shape inference fails
    enable_pattern_matching: bool = True   # Run pattern matching
    match_type: PatternMatchType = ALL     # FIRST, ALL, or COUNT_ONLY
    max_matches_per_pattern: int = 1000    # Limit matches per pattern
    pattern_match_before_opt: bool = False # Match before optimization
    save_optimized_model: bool = True      # Save optimized model
    save_json_result: bool = True          # Save JSON results
    json_output_path: str = ""             # JSON output file path
    verbose: bool = False                  # Verbose logging
```

**Note on Pattern Matching Timing:**
By default, pattern matching runs **after** optimization. This means if you have a `Conv`+`ReLU` pattern and `fuse_conv_relu` is enabled, the pattern won't be found because the nodes have been fused.

To find patterns in the **original** model (before optimization):
```python
options = oniris.CompilerOptions()
options.pattern_match_before_opt = True  # Match before optimization
```

### CompilationResult

Complete compilation results with JSON serialization.

```python
class CompilationResult:
    success: bool
    error_msg: str
    input_path: str
    output_path: str
    
    # Pipeline stage results
    model_info: ModelSummary
    optimization_stats: OptimizationStats
    shape_inference_stats: ShapeInferenceStats
    pattern_matching_summary: PatternMatchingSummary
    
    # Timing
    start_time: str
    end_time: str
    duration_ms: float
    
    def to_json(self, pretty: bool = True) -> str
    def save_json(self, filepath: str, pretty: bool = True) -> bool
```

## Pattern Syntax

Patterns use **ONNX Matcher Style** syntax (tensor-flow based):

```
OpType(input_tensors, output_tensors)
```

### Special Symbols

| Symbol | Meaning |
|--------|---------|
| `?` | Wildcard - matches any tensor or op type |
| `[a, b]` | List - matches multiple inputs/outputs |
| `/` | OR operator: `Conv/Pool` matches either |

### Pattern Examples

**Conv + ReLU:**
```python
pattern = """
    Conv(?, c0)
    Relu(c0, ?)
"""
```

**Swish Activation (x * sigmoid(x)):**
```python
pattern = """
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
"""
```

**Residual Connection:**
```python
pattern = """
    Conv(?, c0)
    Add([c0, ?], ?)
"""
```

**Multi-head Attention:**
```python
pattern = """
    MatMul(?, qk0)
    Softmax(qk0, softmax0)
    MatMul([softmax0, ?], ?)
"""
```

## Common Patterns

Use built-in common patterns for typical fusion opportunities:

```python
patterns = oniris.get_common_patterns()
# Returns:
# - ConvRelu: Conv followed by ReLU
# - ConvBnRelu: Conv + BatchNorm + ReLU
# - Swish: Swish activation pattern
# - Gelu: GELU activation pattern
# - Residual: Residual connection
# - Attention: Attention pattern
# - LayerNorm: Layer normalization pattern
```

## JSON Output Format

The compiler outputs structured JSON with complete compilation results:

```json
{
  "success": true,
  "input_path": "model.onnx",
  "output_path": "model_optimized.onnx",
  "timing": {
    "start_time": "2024-01-15 10:30:00",
    "end_time": "2024-01-15 10:30:05",
    "duration_ms": 5234.56
  },
  "model_info": {
    "producer_name": "pytorch",
    "ir_version": 8,
    "opset_version": 17,
    "num_nodes": 150,
    "num_initializers": 45,
    "num_inputs": 1,
    "num_outputs": 1,
    "op_types_used": ["Conv", "Relu", "BatchNormalization"],
    "op_type_counts": {"Conv": 50, "Relu": 50, "BatchNormalization": 20}
  },
  "optimization": {
    "success": true,
    "num_iterations": 3,
    "num_changes": 25,
    "pass_stats": {
      "constant_folding": 10,
      "dead_node_elimination": 5,
      "conv_bn_fusion": 10
    }
  },
  "shape_inference": {
    "success": true,
    "num_nodes_processed": 150,
    "num_nodes_failed": 0
  },
  "pattern_matching": {
    "total_patterns": 5,
    "patterns_with_matches": 3,
    "total_matches": 45,
    "match_counts": {
      "ConvRelu": 20,
      "ConvBn": 10,
      "Residual": 15
    },
    "results": {
      "ConvRelu": [
        {
          "match_id": 0,
          "tensor_bindings": {"c0": "conv1_out"},
          "nodes": [
            {
              "name": "conv1",
              "op_type": "Conv",
              "inputs": ["input", "weight1", "bias1"],
              "outputs": ["conv1_out"],
              "input_shapes": [[1, 3, 224, 224], [64, 3, 7, 7], [64]],
              "output_shapes": [[1, 64, 112, 112]]
            },
            {
              "name": "relu1",
              "op_type": "Relu",
              "inputs": ["conv1_out"],
              "outputs": ["relu1_out"],
              "input_shapes": [[1, 64, 112, 112]],
              "output_shapes": [[1, 64, 112, 112]]
            }
          ]
        }
      ]
    }
  }
}
```

## Advanced Usage

### Custom Optimization Options

```python
options = oniris.CompilerOptions()
options.simplify_options.fuse_conv_bn = True
options.simplify_options.fuse_conv_relu = True
options.simplify_options.max_iterations = 5
options.enable_shape_inference = True
options.fail_on_unknown_shape = False

result = compiler.compile("input.onnx", "output.onnx", options)
```

### Pattern Matching Only

```python
compiler = oniris.ModelCompiler()
compiler.add_patterns(oniris.get_common_patterns())

# Load model
model = oniris.load_model("model.onnx")

# Run only pattern matching
summary = compiler.run_pattern_matching(model, oniris.PatternMatchType.ALL)

print(f"Found {summary.total_matches} matches")
for pattern, count in summary.match_counts.items():
    print(f"  {pattern}: {count}")
```

### Processing JSON Results

```python
import json

result = compiler.compile("input.onnx", "output.onnx", options)

# Parse JSON
json_str = result.to_json()
data = json.loads(json_str)

# Access specific information
for pattern_name, matches in data['pattern_matching']['results'].items():
    print(f"Pattern: {pattern_name}")
    for match in matches:
        for node in match['nodes']:
            print(f"  Node: {node['name']} ({node['op_type']})")
            print(f"    Input shapes: {node['input_shapes']}")
            print(f"    Output shapes: {node['output_shapes']}")
```

## Command Line Usage

Create a simple CLI script:

```python
#!/usr/bin/env python3
"""ONNX Model Compiler CLI"""

import argparse
import oniris

def main():
    parser = argparse.ArgumentParser(description='ONNX Model Compiler')
    parser.add_argument('input', help='Input ONNX model')
    parser.add_argument('-o', '--output', help='Output optimized model')
    parser.add_argument('-j', '--json', help='JSON output file')
    parser.add_argument('-p', '--patterns', nargs='+', help='Pattern names to match')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    compiler = oniris.ModelCompiler()
    
    # Add patterns
    if args.patterns:
        for name in args.patterns:
            compiler.add_pattern(name, get_pattern_string(name))
    else:
        compiler.add_patterns(oniris.get_common_patterns())
    
    # Configure options
    options = oniris.CompilerOptions()
    options.verbose = args.verbose
    options.save_json_result = bool(args.json)
    options.json_output_path = args.json or ""
    
    # Compile
    result = compiler.compile(args.input, args.output or "", options)
    
    if result.success:
        print(f"Compilation successful!")
        print(f"Duration: {result.duration_ms:.2f} ms")
        print(f"Pattern matches: {result.pattern_matching_summary.total_matches}")
    else:
        print(f"Compilation failed: {result.error_msg}")

if __name__ == '__main__':
    main()
```

## Integration Examples

### With PyTorch

```python
import torch
import oniris

# Export PyTorch model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Compile with Oniris
compiler = oniris.ModelCompiler()
compiler.add_patterns(oniris.get_common_patterns())
result = compiler.compile("model.onnx", "model_optimized.onnx")

# Parse results for custom backend
json_result = json.loads(result.to_json())
```

### With TensorRT

```python
import oniris

# Compile and analyze model
compiler = oniris.ModelCompiler()
compiler.add_pattern("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)")
result = compiler.compile("model.onnx", "model_optimized.onnx")

# Use pattern matching results to guide TensorRT optimization
for pattern, matches in result.pattern_matching_summary.pattern_results.items():
    print(f"Pattern '{pattern}' can be fused: {len(matches)} occurrences")
```

## Error Handling

```python
result = compiler.compile("input.onnx", "output.onnx", options)

if not result.success:
    print(f"Compilation failed: {result.error_msg}")
    
    # Check specific stages
    if not result.optimization_stats.success:
        print(f"Optimization error: {result.optimization_stats.error_msg}")
    
    if not result.shape_inference_stats.success:
        print(f"Shape inference failed for nodes: {result.shape_inference_stats.failed_nodes}")
else:
    print("Compilation successful!")
```

## Performance Tips

1. **Limit pattern matches**: Set `max_matches_per_pattern` to avoid excessive memory usage
2. **Disable unused passes**: Set `enable_optimization=False` if you only need pattern matching
3. **Use COUNT_ONLY match type**: For quick analysis without storing all match details

```python
options = oniris.CompilerOptions()
options.max_matches_per_pattern = 100  # Limit matches
options.match_type = oniris.PatternMatchType.COUNT_ONLY  # Fast counting
```
