"""
System tests for ONNX Model Modification Tools (third_party/onnx_tools).

This test suite validates the functionality of the ONNX tools package including:
- Tensor shape modification
- Initializer/weight replacement
- Node removal
- Node and tensor renaming
- ModelModifier chain operations

Requirements:
    pip install pytest onnx numpy
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from third_party.onnx_tools import (
    ModelModifier,
    modify_tensor_shape,
    get_tensor_shape,
    set_tensor_shape,
    replace_initializer,
    replace_initializer_from_file,
    remove_node,
    insert_node,
    find_node_by_name,
    find_nodes_by_op,
    rename_node,
    rename_tensor,
)


def create_simple_model():
    """Create a simple test model: Conv -> ReLU -> Dropout -> Identity."""
    # Input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    
    # Weights
    weight_data = np.random.randn(16, 3, 3, 3).astype(np.float32)
    weight_init = helper.make_tensor("weight", TensorProto.FLOAT, [16, 3, 3, 3], weight_data.flatten().tolist())
    
    bias_data = np.random.randn(16).astype(np.float32)
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [16], bias_data.flatten().tolist())
    
    # Nodes
    conv_node = helper.make_node("Conv", ["input", "weight", "bias"], ["conv_out"], 
                                  name="conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu1")
    dropout_node = helper.make_node("Dropout", ["relu_out"], ["dropout_out", "mask"], name="dropout1")
    identity_node = helper.make_node("Identity", ["dropout_out"], ["output"], name="identity1")
    
    # Output
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 32, 32])
    
    # Graph
    graph = helper.make_graph(
        [conv_node, relu_node, dropout_node, identity_node],
        "test_model",
        [input_tensor],
        [output_tensor],
        [weight_init, bias_init]
    )
    
    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    return model


def create_multi_input_model():
    """Create a model with multiple inputs for testing tensor renaming."""
    # Inputs
    input_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [1, 3, 32, 32])
    input_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 3, 32, 32])
    
    # Node: Add
    add_node = helper.make_node("Add", ["input_a", "input_b"], ["add_out"], name="add1")
    
    # Output
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
    
    # Graph
    graph = helper.make_graph(
        [add_node],
        "multi_input_model",
        [input_a, input_b],
        [output]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    return model


class TestTensorShapeModification(unittest.TestCase):
    """Test tensor shape modification functions."""
    
    def test_modify_input_shape(self):
        """Test modifying input tensor shape."""
        model = create_simple_model()
        
        # Modify input shape from [1, 3, 32, 32] to [1, 3, 64, 64]
        modified = modify_tensor_shape(model, "input", [1, 3, 64, 64])
        
        # Verify
        input_proto = modified.graph.input[0]
        shape = [d.dim_value for d in input_proto.type.tensor_type.shape.dim]
        self.assertEqual(shape, [1, 3, 64, 64])
    
    def test_modify_output_shape(self):
        """Test modifying output tensor shape."""
        model = create_simple_model()
        
        # Modify output shape
        modified = modify_tensor_shape(model, "output", [1, 16, 64, 64])
        
        # Verify
        output_proto = modified.graph.output[0]
        shape = [d.dim_value for d in output_proto.type.tensor_type.shape.dim]
        self.assertEqual(shape, [1, 16, 64, 64])
    
    def test_modify_with_dynamic_dimension(self):
        """Test modifying shape with dynamic (-1) dimension."""
        model = create_simple_model()
        
        # Modify with dynamic batch size
        modified = modify_tensor_shape(model, "input", [-1, 3, 32, 32])
        
        # Verify
        input_proto = modified.graph.input[0]
        dim = input_proto.type.tensor_type.shape.dim[0]
        self.assertEqual(dim.dim_param, "dynamic")
    
    def test_get_tensor_shape(self):
        """Test getting tensor shape."""
        model = create_simple_model()
        
        shape = get_tensor_shape(model, "input")
        self.assertEqual(shape, [1, 3, 32, 32])
    
    def test_set_tensor_shape_alias(self):
        """Test set_tensor_shape is alias for modify_tensor_shape."""
        model = create_simple_model()
        
        modified = set_tensor_shape(model, "input", [2, 3, 32, 32])
        shape = get_tensor_shape(modified, "input")
        self.assertEqual(shape, [2, 3, 32, 32])
    
    def test_modify_nonexistent_tensor(self):
        """Test modifying non-existent tensor raises error."""
        model = create_simple_model()
        
        with self.assertRaises(ValueError):
            modify_tensor_shape(model, "nonexistent", [1, 2, 3])


class TestInitializerReplacement(unittest.TestCase):
    """Test initializer/weight replacement functions."""
    
    def test_replace_initializer_with_array(self):
        """Test replacing initializer with numpy array."""
        model = create_simple_model()
        
        # Create new weights with same total elements but different shape
        new_weights = np.random.randn(8, 6, 3, 3).astype(np.float32)
        
        modified = replace_initializer(model, "weight", new_weights)
        
        # Verify
        init = None
        for i in modified.graph.initializer:
            if i.name == "weight":
                init = i
                break
        
        self.assertIsNotNone(init)
        self.assertEqual(list(init.dims), [8, 6, 3, 3])
    
    def test_replace_initializer_with_rename(self):
        """Test replacing initializer and renaming it."""
        model = create_simple_model()
        
        new_weights = np.random.randn(16, 3, 3, 3).astype(np.float32)
        modified = replace_initializer(model, "weight", new_weights, name="new_weight")
        
        # Verify old name is gone
        old_init = None
        for i in modified.graph.initializer:
            if i.name == "weight":
                old_init = i
                break
        self.assertIsNone(old_init)
        
        # Verify new name exists
        new_init = None
        for i in modified.graph.initializer:
            if i.name == "new_weight":
                new_init = i
                break
        self.assertIsNotNone(new_init)
        
        # Verify node input reference is updated
        conv_node = None
        for n in modified.graph.node:
            if n.name == "conv1":
                conv_node = n
                break
        self.assertIn("new_weight", conv_node.input)
    
    def test_replace_initializer_from_file(self):
        """Test replacing initializer from numpy file."""
        model = create_simple_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = Path(tmpdir) / "new_weights.npy"
            new_weights = np.random.randn(16, 3, 3, 3).astype(np.float32)
            np.save(npy_path, new_weights)
            
            modified = replace_initializer_from_file(model, "weight", str(npy_path))
            
            # Verify
            init = None
            for i in modified.graph.initializer:
                if i.name == "weight":
                    init = i
                    break
            
            self.assertIsNotNone(init)
    
    def test_replace_nonexistent_initializer(self):
        """Test replacing non-existent initializer raises error."""
        model = create_simple_model()
        
        with self.assertRaises(ValueError):
            replace_initializer(model, "nonexistent", np.array([1, 2, 3]))


class TestNodeRemoval(unittest.TestCase):
    """Test node removal functions."""
    
    def test_remove_node_with_reconnect(self):
        """Test removing a node and reconnecting inputs."""
        model = create_simple_model()
        
        # Remove dropout node (which has 1 input and 2 outputs)
        modified = remove_node(model, "dropout1", reconnect_inputs=True)
        
        # Verify dropout is removed
        dropout_node = None
        for n in modified.graph.node:
            if n.name == "dropout1":
                dropout_node = n
                break
        self.assertIsNone(dropout_node)
        
        # Verify reconnection: identity node should now take input from relu
        identity_node = None
        for n in modified.graph.node:
            if n.name == "identity1":
                identity_node = n
                break
        self.assertIn("relu_out", identity_node.input)
    
    def test_remove_node_without_reconnect(self):
        """Test removing a node without reconnecting."""
        model = create_simple_model()
        
        # Remove dropout without reconnection
        modified = remove_node(model, "dropout1", reconnect_inputs=False)
        
        # Verify dropout is removed
        dropout_node = None
        for n in modified.graph.node:
            if n.name == "dropout1":
                dropout_node = n
                break
        self.assertIsNone(dropout_node)
    
    def test_remove_nonexistent_node(self):
        """Test removing non-existent node raises error."""
        model = create_simple_model()
        
        with self.assertRaises(ValueError):
            remove_node(model, "nonexistent")


class TestNodeInsertion(unittest.TestCase):
    """Test node insertion functions."""
    
    def test_insert_node_at_end(self):
        """Test inserting node at end of graph."""
        model = create_simple_model()
        
        new_node = helper.make_node("Relu", ["output"], ["final_out"], name="final_relu")
        
        modified = insert_node(model, new_node)
        
        # Verify node is in graph
        found = False
        for n in modified.graph.node:
            if n.name == "final_relu":
                found = True
                break
        self.assertTrue(found)
    
    def test_insert_node_before(self):
        """Test inserting node before another node."""
        model = create_simple_model()
        
        new_node = helper.make_node("Identity", ["conv_out"], ["identity_conv"], name="id_before_relu")
        
        modified = insert_node(model, new_node, before="relu1")
        
        # Verify insertion order
        node_names = [n.name for n in modified.graph.node]
        self.assertIn("id_before_relu", node_names)
        self.assertLess(node_names.index("id_before_relu"), node_names.index("relu1"))
    
    def test_insert_node_after(self):
        """Test inserting node after another node."""
        model = create_simple_model()
        
        new_node = helper.make_node("Identity", ["conv_out"], ["identity_conv"], name="id_after_conv")
        
        modified = insert_node(model, new_node, after="conv1")
        
        # Verify insertion order
        node_names = [n.name for n in modified.graph.node]
        self.assertIn("id_after_conv", node_names)
        self.assertGreater(node_names.index("id_after_conv"), node_names.index("conv1"))


class TestNodeFinding(unittest.TestCase):
    """Test node finding functions."""
    
    def test_find_node_by_name(self):
        """Test finding node by name."""
        model = create_simple_model()
        
        node = find_node_by_name(model, "relu1")
        
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "relu1")
        self.assertEqual(node.op_type, "Relu")
    
    def test_find_node_by_name_not_found(self):
        """Test finding non-existent node returns None."""
        model = create_simple_model()
        
        node = find_node_by_name(model, "nonexistent")
        
        self.assertIsNone(node)
    
    def test_find_nodes_by_op(self):
        """Test finding nodes by operation type."""
        model = create_simple_model()
        
        relu_nodes = find_nodes_by_op(model, "Relu")
        
        self.assertEqual(len(relu_nodes), 1)
        self.assertEqual(relu_nodes[0].name, "relu1")
    
    def test_find_nodes_by_op_not_found(self):
        """Test finding non-existent operation type."""
        model = create_simple_model()
        
        nodes = find_nodes_by_op(model, "NonExistentOp")
        
        self.assertEqual(len(nodes), 0)


class TestRenaming(unittest.TestCase):
    """Test node and tensor renaming functions."""
    
    def test_rename_node(self):
        """Test renaming a node."""
        model = create_simple_model()
        
        modified = rename_node(model, "relu1", "activation_relu")
        
        # Verify old name is gone
        old_node = find_node_by_name(modified, "relu1")
        self.assertIsNone(old_node)
        
        # Verify new name exists
        new_node = find_node_by_name(modified, "activation_relu")
        self.assertIsNotNone(new_node)
        self.assertEqual(new_node.op_type, "Relu")
    
    def test_rename_nonexistent_node(self):
        """Test renaming non-existent node raises error."""
        model = create_simple_model()
        
        with self.assertRaises(ValueError):
            rename_node(model, "nonexistent", "new_name")
    
    def test_rename_input_tensor(self):
        """Test renaming an input tensor."""
        model = create_simple_model()
        
        modified = rename_tensor(model, "input", "image_input")
        
        # Verify input name changed
        input_names = [i.name for i in modified.graph.input]
        self.assertNotIn("input", input_names)
        self.assertIn("image_input", input_names)
        
        # Verify node reference updated
        conv_node = find_node_by_name(modified, "conv1")
        self.assertIn("image_input", conv_node.input)
    
    def test_rename_output_tensor(self):
        """Test renaming an output tensor."""
        model = create_simple_model()
        
        modified = rename_tensor(model, "output", "prediction")
        
        # Verify output name changed
        output_names = [o.name for o in modified.graph.output]
        self.assertNotIn("output", output_names)
        self.assertIn("prediction", output_names)
        
        # Verify node reference updated
        identity_node = find_node_by_name(modified, "identity1")
        self.assertIn("prediction", identity_node.output)
    
    def test_rename_initializer_tensor(self):
        """Test renaming an initializer tensor."""
        model = create_simple_model()
        
        modified = rename_tensor(model, "weight", "conv_weight")
        
        # Verify initializer name changed
        init_names = [i.name for i in modified.graph.initializer]
        self.assertNotIn("weight", init_names)
        self.assertIn("conv_weight", init_names)
        
        # Verify node reference updated
        conv_node = find_node_by_name(modified, "conv1")
        self.assertIn("conv_weight", conv_node.input)
    
    def test_rename_nonexistent_tensor(self):
        """Test renaming non-existent tensor raises error."""
        model = create_simple_model()
        
        with self.assertRaises(ValueError):
            rename_tensor(model, "nonexistent", "new_name")
    
    def test_rename_multiple_inputs(self):
        """Test renaming tensors in multi-input model."""
        model = create_multi_input_model()
        
        modified = rename_tensor(model, "input_a", "first_input")
        modified = rename_tensor(modified, "input_b", "second_input")
        
        # Verify both renamed
        input_names = [i.name for i in modified.graph.input]
        self.assertIn("first_input", input_names)
        self.assertIn("second_input", input_names)
        
        # Verify node references
        add_node = find_node_by_name(modified, "add1")
        self.assertIn("first_input", add_node.input)
        self.assertIn("second_input", add_node.input)


class TestModelModifier(unittest.TestCase):
    """Test ModelModifier class for chain operations."""
    
    def test_load_from_file(self):
        """Test loading model from file."""
        model = create_simple_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.onnx"
            onnx.save(model, str(model_path))
            
            modifier = ModelModifier(str(model_path))
            self.assertIsNotNone(modifier.get_model())
    
    def test_load_from_model(self):
        """Test loading from ModelProto directly."""
        model = create_simple_model()
        
        modifier = ModelModifier(model)
        self.assertIsNotNone(modifier.get_model())
    
    def test_chain_operations(self):
        """Test chaining multiple operations."""
        model = create_simple_model()
        
        modifier = ModelModifier(model)
        modifier \
            .modify_tensor_shape("input", [2, 3, 64, 64]) \
            .rename_node("relu1", "activation") \
            .rename_tensor("output", "prediction")
        
        modified = modifier.get_model()
        
        # Verify all changes applied
        shape = get_tensor_shape(modified, "input")
        self.assertEqual(shape, [2, 3, 64, 64])
        
        node = find_node_by_name(modified, "activation")
        self.assertIsNotNone(node)
        
        output_names = [o.name for o in modified.graph.output]
        self.assertIn("prediction", output_names)
    
    def test_save_model(self):
        """Test saving modified model."""
        model = create_simple_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "modified.onnx"
            
            modifier = ModelModifier(model)
            modifier.modify_tensor_shape("input", [4, 3, 32, 32])
            modifier.save(str(output_path))
            
            # Verify file exists and can be loaded
            self.assertTrue(output_path.exists())
            loaded = onnx.load(str(output_path))
            shape = get_tensor_shape(loaded, "input")
            self.assertEqual(shape, [4, 3, 32, 32])
    
    def test_replace_initializer_chain(self):
        """Test replace initializer in chain."""
        model = create_simple_model()
        
        new_weights = np.random.randn(16, 3, 3, 3).astype(np.float32)
        
        modifier = ModelModifier(model)
        modifier.replace_initializer("weight", new_weights)
        
        modified = modifier.get_model()
        init = None
        for i in modified.graph.initializer:
            if i.name == "weight":
                init = i
                break
        
        self.assertEqual(list(init.dims), [16, 3, 3, 3])
    
    def test_remove_node_chain(self):
        """Test remove node in chain."""
        model = create_simple_model()
        
        modifier = ModelModifier(model)
        modifier.remove_node("dropout1")
        
        modified = modifier.get_model()
        node = find_node_by_name(modified, "dropout1")
        self.assertIsNone(node)


class TestModelRoundtrip(unittest.TestCase):
    """Test that modifications preserve model validity."""
    
    def test_shape_modification_roundtrip(self):
        """Test shape changes are preserved through save/load."""
        model = create_simple_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Modify and save
            modified = modify_tensor_shape(model, "input", [8, 3, 128, 128])
            output_path = Path(tmpdir) / "modified.onnx"
            onnx.save(modified, str(output_path))
            
            # Reload and verify
            loaded = onnx.load(str(output_path))
            shape = get_tensor_shape(loaded, "input")
            self.assertEqual(shape, [8, 3, 128, 128])
            
            # Check model is still valid
            onnx.checker.check_model(loaded)
    
    def test_renaming_roundtrip(self):
        """Test renaming is preserved through save/load."""
        model = create_simple_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Rename and save
            modified = rename_tensor(model, "input", "image")
            output_path = Path(tmpdir) / "renamed.onnx"
            onnx.save(modified, str(output_path))
            
            # Reload and verify
            loaded = onnx.load(str(output_path))
            input_names = [i.name for i in loaded.graph.input]
            self.assertIn("image", input_names)
            
            # Check model is still valid
            onnx.checker.check_model(loaded)
    
    def test_full_modification_roundtrip(self):
        """Test complex modifications preserve validity."""
        model = create_simple_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Apply multiple modifications
            modifier = ModelModifier(model)
            modifier \
                .modify_tensor_shape("input", [1, 3, 64, 64]) \
                .rename_node("conv1", "feature_conv") \
                .rename_tensor("conv_out", "features") \
                .remove_node("dropout1")
            
            output_path = Path(tmpdir) / "full_modified.onnx"
            modifier.save(str(output_path))
            
            # Reload and verify
            loaded = onnx.load(str(output_path))
            
            # Verify shape
            shape = get_tensor_shape(loaded, "input")
            self.assertEqual(shape, [1, 3, 64, 64])
            
            # Verify node renamed
            self.assertIsNone(find_node_by_name(loaded, "conv1"))
            self.assertIsNotNone(find_node_by_name(loaded, "feature_conv"))
            
            # Verify tensor renamed
            conv_node = find_node_by_name(loaded, "feature_conv")
            self.assertIn("features", conv_node.output)
            
            # Verify node removed
            self.assertIsNone(find_node_by_name(loaded, "dropout1"))
            
            # Check model is still valid
            onnx.checker.check_model(loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
