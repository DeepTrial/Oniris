#!/usr/bin/env python3
"""
ONNX Matcher Style Pattern Examples

This example demonstrates the onnx_matcher style pattern matching,
which uses tensor-flow based patterns instead of node-based patterns.

Syntax:
    OpType(input_tensors, output_tensors)
    OpType1/OpType2(input, output)
    ?(input, output)  # wildcard for any op type

Special:
    ?  - matches any tensor or op type
    [a, b] - list of tensors for multi-input/output nodes

Examples:
    # Conv -> ReLU
    Conv(?, c0)
    Relu(c0, ?)
    
    # Swish activation: Conv -> Sigmoid -> Mul
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
    
    # Conv or Pool followed by Relu
    Conv/MaxPool(?, c0)
    Relu(c0, ?)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import oniris


def example_1_basic_pattern():
    """Example 1: Basic Conv -> ReLU pattern."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Conv -> ReLU Pattern")
    print("=" * 60)
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("example1")
    
    # Build: input -> Conv -> ReLU -> output
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    graph.add_input(input_info)
    
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    output = oniris.ValueInfo()
    output.name = "output"
    graph.add_output(output)
    
    # Pattern: Conv -> ReLU
    # c0 is a variable that binds Conv's output to ReLU's input
    pattern = oniris.OnnxMatcherPattern.from_string("""
        Conv(?, c0)
        Relu(c0, ?)
    """)
    
    matches = oniris.OnnxMatcherStyleMatcher.find_all(model, pattern)
    
    print("Pattern:")
    print("  Conv(?, c0)")
    print("  Relu(c0, ?)")
    print(f"\nFound {len(matches)} match(es)")
    
    for i, match in enumerate(matches):
        print(f"\n  Match {i+1}:")
        for j, node in enumerate(match.matched_nodes):
            print(f"    [{j}] {node.get_name()}: {node.get_op_type()}")


def example_2_swish_activation():
    """Example 2: Swish activation pattern (Conv -> Sigmoid -> Mul)."""
    print("\n" + "=" * 60)
    print("Example 2: Swish Activation (Conv -> Sigmoid -> Mul)")
    print("=" * 60)
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("example2")
    
    # Build Swish: x * sigmoid(x)
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    graph.add_input(input_info)
    
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    
    sigmoid = graph.create_node("Sigmoid", "sigmoid1")
    sigmoid.add_input("conv_out")
    sigmoid.add_output("sig_out")
    
    # Mul takes both sigmoid output and conv output
    mul = graph.create_node("Mul", "mul1")
    mul.add_input("sig_out")
    mul.add_input("conv_out")
    mul.add_output("output")
    
    output = oniris.ValueInfo()
    output.name = "output"
    graph.add_output(output)
    
    # Pattern: Conv -> Sigmoid -> Mul
    # Note: [s0, c0] means Mul takes two inputs: s0 and c0
    pattern = oniris.OnnxMatcherPattern.from_string("""
        Conv(?, c0)
        Sigmoid(c0, s0)
        Mul([s0, c0], ?)
    """)
    
    matches = oniris.OnnxMatcherStyleMatcher.find_all(model, pattern)
    
    print("Pattern:")
    print("  Conv(?, c0)       # Conv output bound to c0")
    print("  Sigmoid(c0, s0)   # Takes c0, outputs s0")
    print("  Mul([s0, c0], ?)  # Takes both s0 and c0")
    print(f"\nFound {len(matches)} Swish pattern(s)")


def example_3_wildcard():
    """Example 3: Using wildcards."""
    print("\n" + "=" * 60)
    print("Example 3: Wildcard Patterns")
    print("=" * 60)
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("example3")
    
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    graph.add_input(input_info)
    
    # Relu -> Sigmoid
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("input")
    relu.add_output("relu_out")
    
    sigmoid = graph.create_node("Sigmoid", "sigmoid1")
    sigmoid.add_input("relu_out")
    sigmoid.add_output("output")
    
    output = oniris.ValueInfo()
    output.name = "output"
    graph.add_output(output)
    
    # Pattern 1: Match any op type
    print("\nPattern 1: ?(?, a0) followed by Sigmoid(a0, ?)")
    print("  (? means any op type)")
    
    pattern1 = oniris.OnnxMatcherPattern.from_string("""
        ?(?, a0)
        Sigmoid(a0, ?)
    """)
    
    matches1 = oniris.OnnxMatcherStyleMatcher.find_all(model, pattern1)
    print(f"  Found {len(matches1)} match(es)")
    
    # Pattern 2: Match any tensor
    print("\nPattern 2: Relu(?, ?) followed by ?(?, ?)")
    print("  (? means any tensor)")
    
    pattern2 = oniris.OnnxMatcherPattern.from_string("""
        Relu(?, ?)
        ?(?, ?)
    """)
    
    matches2 = oniris.OnnxMatcherStyleMatcher.find_all(model, pattern2)
    print(f"  Found {len(matches2)} match(es)")


def example_4_multi_optype():
    """Example 4: Multiple op types (Conv/Pool)."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Op Types (Conv/Pool)")
    print("=" * 60)
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("example4")
    
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    graph.add_input(input_info)
    
    # Conv branch
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("w1")
    conv.add_output("conv_out")
    
    relu1 = graph.create_node("Relu", "relu1")
    relu1.add_input("conv_out")
    relu1.add_output("out1")
    
    # MaxPool branch
    pool = graph.create_node("MaxPool", "pool1")
    pool.add_input("input")
    pool.add_output("pool_out")
    
    relu2 = graph.create_node("Relu", "relu2")
    relu2.add_input("pool_out")
    relu2.add_output("out2")
    
    # Pattern: Conv OR Pool followed by Relu
    pattern = oniris.OnnxMatcherPattern.from_string("""
        Conv/MaxPool(?, c0)
        Relu(c0, ?)
    """)
    
    matches = oniris.OnnxMatcherStyleMatcher.find_all(model, pattern)
    
    print("Pattern:")
    print("  Conv/MaxPool(?, c0)  # Match Conv OR MaxPool")
    print("  Relu(c0, ?)")
    print(f"\nFound {len(matches)} match(es)")
    print("  (Should find 2: Conv->Relu and MaxPool->Relu)")


def example_5_matcher_class():
    """Example 5: Using the Matcher convenience class."""
    print("\n" + "=" * 60)
    print("Example 5: Matcher Convenience Class")
    print("=" * 60)
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("example5")
    
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    graph.add_input(input_info)
    
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    
    bn = graph.create_node("BatchNormalization", "bn1")
    bn.add_input("conv_out")
    bn.add_input("scale")
    bn.add_input("bias")
    bn.add_input("mean")
    bn.add_input("var")
    bn.add_output("output")
    
    output = oniris.ValueInfo()
    output.name = "output"
    graph.add_output(output)
    
    # Using the Matcher alias (same as OnnxMatcherStyleMatcher)
    pattern = oniris.Matcher  # This is an alias for convenience
    
    pattern_str = oniris.OnnxMatcherPattern.from_string("""
        Conv(?, c0)
        BatchNormalization(c0, ?)
    """)
    
    matches = oniris.Matcher.find_all(model, pattern_str)
    
    print("Pattern:")
    print("  Conv(?, c0)")
    print("  BatchNormalization(c0, ?)")
    print(f"\nFound {len(matches)} Conv-BN pattern(s)")


if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Matcher Style Pattern Examples")
    print("=" * 60)
    
    example_1_basic_pattern()
    example_2_swish_activation()
    example_3_wildcard()
    example_4_multi_optype()
    example_5_matcher_class()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
