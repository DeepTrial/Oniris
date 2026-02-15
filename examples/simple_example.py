#!/usr/bin/env python3
"""Simple example demonstrating Oniris API usage."""

import sys
from pathlib import Path

# Add the python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import oniris


def create_simple_model():
    """Create a simple Conv -> ReLU model programmatically."""
    # Create model
    model = oniris.Model(8)
    model.set_producer_name("oniris_example")
    model.set_producer_version("0.1.0")
    
    # Add opset
    opset = oniris.OpsetImport()
    opset.domain = ""
    opset.version = 13
    model.add_opset_import(opset)
    
    # Create graph
    graph = model.create_graph("simple_conv")
    
    # Add inputs
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([1, 3, 32, 32])
    input_info.dtype = oniris.DataType.FLOAT32
    graph.add_input(input_info)
    
    # Add weight (as value info for now)
    weight_info = oniris.ValueInfo()
    weight_info.name = "weight"
    weight_info.shape = oniris.Shape([16, 3, 3, 3])
    weight_info.dtype = oniris.DataType.FLOAT32
    graph.set_value_info("weight", weight_info)
    
    # Add Conv node
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    conv.set_attribute_ints("kernel_shape", [3, 3])
    conv.set_attribute_ints("strides", [1, 1])
    conv.set_attribute_ints("pads", [1, 1, 1, 1])
    
    # Add ReLU node
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    # Add output
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    output_info.dtype = oniris.DataType.FLOAT32
    graph.add_output(output_info)
    
    # Add value info for intermediate tensor
    conv_out_info = oniris.ValueInfo()
    conv_out_info.name = "conv_out"
    graph.set_value_info("conv_out", conv_out_info)
    
    return model


def test_shape_inference(model):
    """Test shape inference on a model."""
    print("\n=== Shape Inference ===")
    
    engine = oniris.ShapeInferenceEngine.get_instance()
    
    # Get supported ops
    supported_ops = engine.get_supported_ops()
    print(f"Number of supported ops: {len(supported_ops)}")
    print(f"Sample ops: {supported_ops[:5]}")
    
    # Run inference
    success = engine.infer_model(model, fail_on_unknown=False)
    print(f"Shape inference success: {success}")
    
    # Check inferred shapes
    graph = model.get_graph()
    for node in graph.get_nodes():
        print(f"\nNode: {node.get_name()} ({node.get_op_type()})")
        print(f"  Inputs: {list(node.get_inputs())}")
        print(f"  Outputs: {list(node.get_outputs())}")
        
        for i, output in enumerate(node.get_outputs()):
            vi = graph.get_value_info(output)
            if vi and vi.has_inferred_shape():
                print(f"  Output {i} shape: {vi.shape}")


def test_simplification(model):
    """Test model simplification."""
    print("\n=== Model Simplification ===")
    
    graph = model.get_graph()
    print(f"Original nodes: {len(graph.get_nodes())}")
    
    # Simplify
    options = oniris.SimplifyOptions()
    options.verbose = True
    
    result = oniris.Simplifier.simplify(model, options)
    
    print(f"Simplification success: {result.success}")
    print(f"Changes made: {result.num_changes}")
    print(f"Iterations: {result.num_iterations}")
    
    graph = model.get_graph()
    print(f"Final nodes: {len(graph.get_nodes())}")
    
    if result.unsupported_ops:
        print(f"Unsupported ops: {result.unsupported_ops}")


def test_custom_handler():
    """Test registering a custom shape inference handler."""
    print("\n=== Custom Handler ===")
    
    # Define custom inference function
    def custom_infer(ctx):
        # Double the first dimension
        input_shape = ctx.input_shapes[0]
        output_shape = oniris.Shape(list(input_shape.get_dims()))
        
        if not output_shape.get_dim(0).is_dynamic():
            new_dim = output_shape.get_dim(0).get_static_value() * 2
            output_shape.get_dim(0).set_static_value(new_dim)
        
        return oniris.InferenceResult.Success([output_shape])
    
    # Register handler
    oniris.register_custom_shape_inference("MyDoubleOp", custom_infer)
    
    # Create model with custom op
    model = oniris.Model(8)
    
    opset = oniris.OpsetImport()
    opset.version = 13
    model.add_opset_import(opset)
    
    graph = model.create_graph("custom_test")
    
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([4, 5])
    graph.add_input(input_info)
    
    custom_op = graph.create_node("MyDoubleOp", "custom1")
    custom_op.add_input("input")
    custom_op.add_output("output")
    
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    # Run inference
    engine = oniris.ShapeInferenceEngine.get_instance()
    engine.infer_model(model, fail_on_unknown=False)
    
    # Check result
    output_vi = graph.get_value_info("output")
    if output_vi:
        print(f"Input shape: [4, 5]")
        print(f"Output shape: {output_vi.shape}")
        assert output_vi.shape.get_dim(0).get_static_value() == 8
        print("Custom handler works correctly!")


def main():
    """Main example."""
    print("=" * 60)
    print("Oniris Example")
    print("=" * 60)
    
    print("\nOniris version:", oniris.__version__)
    print("ONNX version:", oniris.get_onnx_version())
    
    # Create a simple model
    print("\n=== Creating Model ===")
    model = create_simple_model()
    print(f"Model IR version: {model.get_ir_version()}")
    print(f"Model producer: {model.get_producer_name()}")
    
    # Validate model
    valid, error_msg = model.validate()
    print(f"Model valid: {valid}")
    if not valid:
        print(f"Validation error: {error_msg}")
    
    # Test shape inference
    test_shape_inference(model)
    
    # Test simplification
    test_simplification(model)
    
    # Test custom handler
    test_custom_handler()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
