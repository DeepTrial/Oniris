#!/usr/bin/env python3
"""Example demonstrating fusion control in Oniris.

This example shows how to control fusion passes when simplifying ONNX models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import oniris


def test_fusion_control():
    """Test controlling fusion passes."""
    print("\n=== Fusion Control Demo ===")
    
    # Create a model with Conv+ReLU pattern
    model = oniris.Model(8)
    
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("fusion_demo")
    
    # Input
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([1, 3, 32, 32])
    graph.add_input(input_info)
    
    # Conv
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    conv.set_attribute_ints("kernel_shape", [3, 3])
    
    # ReLU
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    # Output
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    print(f"Original graph: {len(graph.get_nodes())} nodes")
    for node in graph.get_nodes():
        print(f"  - {node.get_name()}: {node.get_op_type()}")
    
    # Test 1: With fusion enabled (default)
    print("\n1. With fusion enabled:")
    model_copy = model.clone()
    options = oniris.SimplifyOptions()
    options.fuse_conv_relu = True
    options.skip_shape_inference = True
    
    result = oniris.Simplifier.simplify(model_copy, options)
    print(f"   Changes: {result.num_changes}")
    print(f"   Final nodes: {len(model_copy.get_graph().get_nodes())}")
    for node in model_copy.get_graph().get_nodes():
        print(f"     - {node.get_name()}: {node.get_op_type()}")
    
    # Test 2: With fusion disabled
    print("\n2. With fusion disabled:")
    model_copy2 = model.clone()
    options2 = oniris.SimplifyOptions()
    options2.fuse_conv_relu = False
    options2.skip_shape_inference = True
    
    result2 = oniris.Simplifier.simplify(model_copy2, options2)
    print(f"   Changes: {result2.num_changes}")
    print(f"   Final nodes: {len(model_copy2.get_graph().get_nodes())}")
    for node in model_copy2.get_graph().get_nodes():
        print(f"     - {node.get_name()}: {node.get_op_type()}")


def test_all_simplification_options():
    """Test all simplification options."""
    print("\n=== Simplification Options Demo ===")
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("options_demo")
    
    # Create various nop operations
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    graph.add_input(input_info)
    
    # Identity
    identity = graph.create_node("Identity", "id1")
    identity.add_input("input")
    identity.add_output("id_out")
    
    # Nop transpose
    transpose = graph.create_node("Transpose", "transpose1")
    transpose.add_input("id_out")
    transpose.add_output("t_out")
    transpose.set_attribute_ints("perm", [0, 1, 2])
    
    # ReLU
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("t_out")
    relu.add_output("output")
    
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    print(f"Original graph: {len(graph.get_nodes())} nodes")
    
    # Show all available options
    print("\nAvailable simplification options:")
    options = oniris.SimplifyOptions()
    
    boolean_options = [
        "skip_shape_inference",
        "skip_constant_folding",
        "skip_dead_node_elimination",
        "skip_identity_elimination",
        "skip_shape_ops_simplification",
        "skip_transpose_elimination",
        "skip_reshape_elimination",
        "skip_pad_elimination",
        "skip_slice_elimination",
        "fuse_conv_bn",
        "fuse_conv_relu",
        "fuse_gemm_activation",
        "fail_on_unsupported",
        "verbose",
    ]
    
    for opt_name in boolean_options:
        value = getattr(options, opt_name)
        print(f"  {opt_name}: {value}")
    
    print(f"  max_iterations: {options.max_iterations}")


def test_conv_bn_fusion():
    """Test Conv+BN fusion."""
    print("\n=== Conv+BN Fusion Demo ===")
    
    model = oniris.Model(8)
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("conv_bn_demo")
    
    # Input
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([1, 3, 32, 32])
    graph.add_input(input_info)
    
    # Conv
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    conv.set_attribute_ints("kernel_shape", [3, 3])
    
    # BN constants
    bn_scale = oniris.ConstantTensor()
    bn_scale.name = "bn_scale"
    bn_scale.shape = oniris.Shape([16])
    bn_scale.dtype = oniris.DataType.FLOAT32
    graph.add_constant("bn_scale", bn_scale)
    
    bn_bias = oniris.ConstantTensor()
    bn_bias.name = "bn_bias"
    bn_bias.shape = oniris.Shape([16])
    bn_bias.dtype = oniris.DataType.FLOAT32
    graph.add_constant("bn_bias", bn_bias)
    
    bn_mean = oniris.ConstantTensor()
    bn_mean.name = "bn_mean"
    bn_mean.shape = oniris.Shape([16])
    bn_mean.dtype = oniris.DataType.FLOAT32
    graph.add_constant("bn_mean", bn_mean)
    
    bn_var = oniris.ConstantTensor()
    bn_var.name = "bn_var"
    bn_var.shape = oniris.Shape([16])
    bn_var.dtype = oniris.DataType.FLOAT32
    graph.add_constant("bn_var", bn_var)
    
    # BatchNorm
    bn = graph.create_node("BatchNormalization", "bn1")
    bn.add_input("conv_out")
    bn.add_input("bn_scale")
    bn.add_input("bn_bias")
    bn.add_input("bn_mean")
    bn.add_input("bn_var")
    bn.add_output("output")
    
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    print(f"Original graph: {len(graph.get_nodes())} nodes")
    print("Nodes:")
    for node in graph.get_nodes():
        print(f"  - {node.get_name()}: {node.get_op_type()}")
    
    # Simplify with Conv+BN fusion
    options = oniris.SimplifyOptions()
    options.fuse_conv_bn = True
    options.skip_shape_inference = True
    
    result = oniris.Simplifier.simplify(model, options)
    
    print(f"\nAfter simplification:")
    print(f"  Changes: {result.num_changes}")
    print(f"  Final nodes: {len(graph.get_nodes())}")
    print("Nodes:")
    for node in graph.get_nodes():
        print(f"  - {node.get_name()}: {node.get_op_type()}")


def main():
    """Main example."""
    print("=" * 60)
    print("Oniris Fusion Control Example")
    print("=" * 60)
    
    test_fusion_control()
    test_all_simplification_options()
    test_conv_bn_fusion()
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
