"""
Examples of using ONNX model modification tools.
"""

import numpy as np
import onnx
from pathlib import Path


def example_modify_tensor_shape():
    """Example: Modify input tensor shape."""
    from third_party.onnx_tools import modify_tensor_shape
    
    # Load model
    model = onnx.load("model.onnx")
    
    # Change input shape from [1, 3, 224, 224] to [1, 3, 320, 320]
    model = modify_tensor_shape(model, "input", [1, 3, 320, 320])
    
    # Save modified model
    onnx.save(model, "model_320x320.onnx")
    print("✓ Modified input shape to [1, 3, 320, 320]")


def example_replace_initializer():
    """Example: Replace weights with numpy data."""
    from third_party.onnx_tools import replace_initializer, replace_initializer_from_file
    
    # Method 1: Using numpy array directly
    model = onnx.load("model.onnx")
    new_weights = np.random.randn(64, 3, 7, 7).astype(np.float32)
    model = replace_initializer(model, "conv1.weight", new_weights)
    onnx.save(model, "model_new_weights.onnx")
    
    # Method 2: Using numpy file
    model = onnx.load("model.onnx")
    model = replace_initializer_from_file(model, "fc1.weight", "new_weights.npy")
    onnx.save(model, "model_from_file.onnx")
    print("✓ Replaced initializers")


def example_remove_node():
    """Example: Remove dropout layer."""
    from third_party.onnx_tools import remove_node
    
    model = onnx.load("model.onnx")
    
    # Remove dropout node
    model = remove_node(model, "dropout_1", reconnect_inputs=True)
    
    onnx.save(model, "model_no_dropout.onnx")
    print("✓ Removed dropout node")


def example_rename():
    """Example: Rename nodes and tensors."""
    from third_party.onnx_tools import rename_node, rename_tensor
    
    model = onnx.load("model.onnx")
    
    # Rename a node
    model = rename_node(model, "conv1", "backbone_conv1")
    
    # Rename input tensor
    model = rename_tensor(model, "input_0", "image_input")
    
    # Rename output tensor
    model = rename_tensor(model, "output_0", "prediction")
    
    onnx.save(model, "model_renamed.onnx")
    print("✓ Renamed nodes and tensors")


def example_chain_operations():
    """Example: Chain multiple operations using ModelModifier."""
    from third_party.onnx_tools import ModelModifier
    
    # Chain multiple modifications
    modifier = ModelModifier("model.onnx")
    modifier \
        .modify_tensor_shape("input", [1, 3, 256, 256]) \
        .replace_initializer_from_file("conv1.weight", "new_conv1.npy") \
        .remove_node("dropout_1") \
        .rename_tensor("input", "image") \
        .rename_tensor("output", "prediction") \
        .save("model_modified.onnx")
    
    print("✓ Applied all modifications")


def example_batch_processing():
    """Example: Batch process multiple models."""
    from third_party.onnx_tools import ModelModifier
    from pathlib import Path
    
    input_dir = Path("models")
    output_dir = Path("modified_models")
    output_dir.mkdir(exist_ok=True)
    
    for model_path in input_dir.glob("*.onnx"):
        print(f"Processing {model_path.name}...")
        
        try:
            modifier = ModelModifier(model_path)
            modifier \
                .modify_tensor_shape("input", [1, 3, 224, 224]) \
                .remove_node("dropout") \
                .save(output_dir / model_path.name)
            
            print(f"  ✓ Saved to {output_dir / model_path.name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_inspect_model():
    """Example: Inspect model before modification."""
    import onnx
    
    model = onnx.load("model.onnx")
    
    print("=== Model Inputs ===")
    for input_proto in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                 for d in input_proto.type.tensor_type.shape.dim]
        print(f"  {input_proto.name}: {shape}")
    
    print("\n=== Model Outputs ===")
    for output_proto in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                 for d in output_proto.type.tensor_type.shape.dim]
        print(f"  {output_proto.name}: {shape}")
    
    print("\n=== Initializers ===")
    for init in model.graph.initializer:
        print(f"  {init.name}: {list(init.dims)} ({onnx.TensorProto.DataType.Name(init.data_type)})")
    
    print("\n=== Nodes ===")
    for node in model.graph.node[:10]:  # Show first 10 nodes
        print(f"  {node.name}: {node.op_type}")


if __name__ == "__main__":
    print("ONNX Model Modification Examples")
    print("================================\n")
    
    print("Available examples:")
    print("  1. Modify tensor shape")
    print("  2. Replace initializer")
    print("  3. Remove node")
    print("  4. Rename nodes/tensors")
    print("  5. Chain operations")
    print("  6. Batch processing")
    print("  7. Inspect model")
    print("\nRun: python -c \"from third_party.onnx_tools.examples import example_chain_operations; example_chain_operations()\"")
