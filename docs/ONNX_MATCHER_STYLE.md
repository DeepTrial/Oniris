# ONNX Matcher Style Pattern Matching

This document describes the **tensor-flow based** pattern matching style inspired by [onnx_matcher](https://github.com/sesmfs/onnx_matcher).

## Overview

The ONNX Matcher style focuses on **tensor flow** between operations, making it: This makes it particularly suitable for:

- Matching activation fusion patterns (Swish, GELU, etc.)
- Finding subgraphs with specific data flow
- Replacing subgraphs with optimized implementations

## Syntax

### Basic Format

```
OpType(input_tensors, output_tensors)
```

### Special Symbols

| Symbol | Meaning |
|--------|---------|
| `?` | Wildcard - matches any tensor or op type |
| `[a, b]` | List - matches multiple inputs/outputs |
| `/` | OR operator for op types: `Conv/Pool` |

### Variable Binding

Variables (like `c0`, `s0`) bind tensors across nodes:

```
Conv(?, c0)       # Conv outputs to variable c0
Relu(c0, ?)       # Relu takes c0 as input
```

## Examples

### Example 1: Conv -> ReLU

```python
pattern = oniris.OnnxMatcherPattern.from_string("""
    Conv(?, c0)
    Relu(c0, ?)
""")

matches = oniris.OnnxMatcherStyleMatcher.find_all(model, pattern)
```

### Example 2: Swish Activation

Swish: `x * sigmoid(x)`

```python
pattern = oniris.OnnxMatcherPattern.from_string("""
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
""")
```

**Explanation:**
- `Conv` outputs to `c0`
- `Sigmoid` takes `c0` and outputs to `s0`
- `Mul` takes both `s0` AND `c0` as inputs (multi-input)

### Example 3: Conv or Pool followed by ReLU

```python
pattern = oniris.OnnxMatcherPattern.from_string("""
    Conv/MaxPool/AveragePool(?, c0)
    Relu(c0, ?)
""")
```

### Example 4: Wildcard Patterns

Match any op type:

```python
pattern = oniris.OnnxMatcherPattern.from_string("""
    ?(?, a0)
    Sigmoid(a0, ?)
""")
```

Match any tensor:

```python
pattern = oniris.OnnxMatcherPattern.from_string("""
    Conv(?, ?)
    ?(?, ?)
""")
```

### Example 5: Linear Chain

```python
pattern = oniris.OnnxMatcherPattern.from_string("""
    MatMul(?, a0)
    Add(a0, a1)
    Relu(a1, ?)
""")
```

## API Reference

### OnnxMatcherPattern

```python
# Parse pattern from string
pattern = oniris.OnnxMatcherPattern.from_string(pattern_str: str) -> Optional[OnnxMatcherPattern]
```

### OnnxMatcherStyleMatcher

```python
# Find all matches
matches = oniris.OnnxMatcherStyleMatcher.find_all(
    model: oniris.Model,
    pattern: OnnxMatcherPattern
) -> List[SubgraphMatch]

# Find first match
match = oniris.OnnxMatcherStyleMatcher.find_first(
    model: oniris.Model,
    pattern: OnnxMatcherPattern
) -> SubgraphMatch

# Check if pattern exists
has_match = oniris.OnnxMatcherStyleMatcher.has_match(
    model: oniris.Model,
    pattern: OnnxMatcherPattern
) -> bool
```

### Matcher (Convenience Alias)

```python
# Matcher is an alias for OnnxMatcherStyleMatcher
matches = oniris.Matcher.find_all(model, pattern)
```

## Match Result

```python
match = matches[0]

# Get matched nodes
for node in match.matched_nodes:
    print(f"{node.get_name()}: {node.get_op_type()}")

# Access by generated name
first_node = match.node_mapping["node0"]
second_node = match.node_mapping["node1"]
```

## Best Practices

### When to Use ONNX Matcher Style

| Use Case | Example Pattern |
|----------|-----------------|
| Activation fusions | Swish, GELU, Mish |
| Simple chains | Conv → ReLU → MaxPool |
| Multi-op matching | Conv/Pool → Activation |
| Data flow tracking | Identify where a tensor is used |

## Tips

1. **Use variables for tensor binding**: Variables like `c0`, `s0` must match across nodes
2. **Use `?` for flexibility**: Wildcards match anything
3. **Multi-optype for variations**: `Conv/Pool` matches either op type
4. **List syntax for multi-input**: `[s0, c0]` matches nodes with multiple inputs

## Limitations

1. Linear chain matching only (no complex branching yet)
2. No attribute constraints (e.g., `Conv[k=[3,3]]` not supported)
3. Single output nodes only in variable binding

## References

- Original implementation: [onnx_matcher](https://github.com/sesmfs/onnx_matcher)
