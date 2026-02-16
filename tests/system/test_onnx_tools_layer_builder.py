"""
System tests for ONNX Model Modification Tools - Generic Layer Builder.

This test suite validates the generic add_layer API including:
- Generic add_layer function
- Convenience wrappers (add_conv, add_linear, etc.)
- Support for all ONNX standard ops
- Support for Microsoft domain ops
- Schema definitions

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
from onnx import helper, TensorProto, numpy_helper

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from third_party.onnx_tools import (
    add_layer,
    add_conv,
    add_linear,
    add_activation,
    add_norm,
    add_pooling,
    add_dropout,
    add_shape_manipulation,
    add_fused_conv,
    add_fused_gemm,
    add_attention,
    get_op_schema,
    ALL_OP_SCHEMAS,
    ONNX_OP_SCHEMAS,
    MICROSOFT_OP_SCHEMAS,
    ModelModifier,
)


def create_simple_model():
    """Create a simple test model."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    
    identity_node = helper.make_node("Identity", ["input"], ["identity_out"], name="identity1")
    
    graph = helper.make_graph(
        [identity_node],
        "test_model",
        [input_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    return model


def find_node_by_name(model, name):
    """Find node by name."""
    for node in model.graph.node:
        if node.name == name:
            return node
    return None


def count_nodes(model):
    """Count nodes in model."""
    return len(model.graph.node)


def count_initializers(model):
    """Count initializers in model."""
    return len(model.graph.initializer)


class TestGenericAddLayer(unittest.TestCase):
    """Test generic add_layer function."""
    
    def test_add_layer_basic(self):
        """Test basic add_layer with Conv."""
        model = create_simple_model()
        initial_nodes = count_nodes(model)
        
        modified = add_layer(
            model, "Conv", "input", "conv_out", name="conv1",
            kernel_size=[3, 3],
            attributes={
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "pads": [1, 1, 1, 1]
            },
            initializers={
                "conv1_W": np.random.randn(64, 3, 3, 3).astype(np.float32),
                "conv1_B": np.zeros(64).astype(np.float32)
            }
        )
        
        self.assertEqual(count_nodes(modified), initial_nodes + 1)
        conv_node = find_node_by_name(modified, "conv1")
        self.assertIsNotNone(conv_node)
        self.assertEqual(conv_node.op_type, "Conv")
    
    def test_add_layer_with_initializers_dict(self):
        """Test add_layer with initializers dict."""
        model = create_simple_model()
        
        weight = np.random.randn(10, 100).astype(np.float32) * 0.01
        bias = np.zeros(10).astype(np.float32)
        
        modified = add_layer(
            model, "Gemm", "input", "output", name="fc1",
            attributes={"transB": 1},
            initializers={
                "fc1_W": weight,
                "fc1_B": bias
            }
        )
        
        # Check initializers were added
        init_names = [i.name for i in modified.graph.initializer]
        self.assertIn("fc1_W", init_names)
        self.assertIn("fc1_B", init_names)
    
    def test_add_layer_multiple_inputs_outputs(self):
        """Test add_layer with multiple inputs/outputs."""
        model = create_simple_model()
        
        modified = add_layer(
            model, "Concat", ["a", "b", "c"], "concat_out", name="cat1",
            attributes={"axis": 1}
        )
        
        cat_node = find_node_by_name(modified, "cat1")
        self.assertEqual(list(cat_node.input), ["a", "b", "c"])
        self.assertEqual(list(cat_node.output), ["concat_out"])
    
    def test_add_layer_with_domain(self):
        """Test add_layer with custom domain."""
        model = create_simple_model()
        
        # Add a custom domain op (just for testing - won't validate)
        modified = add_layer(
            model, "CustomOp", "input", "output",
            name="custom1",
            domain="com.example",
            attributes={"attr1": 10}
        )
        
        custom_node = find_node_by_name(modified, "custom1")
        self.assertEqual(custom_node.domain, "com.example")
    
    def test_add_layer_activation(self):
        """Test adding activation via add_layer."""
        model = create_simple_model()
        
        modified = add_layer(model, "Relu", "input", "relu_out", name="relu1")
        
        relu_node = find_node_by_name(modified, "relu1")
        self.assertIsNotNone(relu_node)
        self.assertEqual(relu_node.op_type, "Relu")
    
    def test_add_layer_leaky_relu_with_alpha(self):
        """Test LeakyRelu with alpha attribute."""
        model = create_simple_model()
        
        modified = add_layer(
            model, "LeakyRelu", "input", "out", name="lrelu1",
            attributes={"alpha": 0.1}
        )
        
        lrelu_node = find_node_by_name(modified, "lrelu1")
        alpha_attr = [a for a in lrelu_node.attribute if a.name == "alpha"][0]
        self.assertAlmostEqual(alpha_attr.f, 0.1, places=5)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience wrapper functions."""
    
    def test_add_conv(self):
        """Test add_conv convenience function."""
        model = create_simple_model()
        
        modified = add_conv(
            model, "conv1", "input", "conv_out",
            in_channels=3, out_channels=64, kernel_size=3,
            stride=2, padding=1, bias=True
        )
        
        conv_node = find_node_by_name(modified, "conv1")
        self.assertIsNotNone(conv_node)
        self.assertEqual(conv_node.op_type, "Conv")
        
        # Check initializers were created
        init_names = [i.name for i in modified.graph.initializer]
        self.assertIn("conv1_W", init_names)
        self.assertIn("conv1_B", init_names)
        
        # Check attributes
        attrs = {a.name: a for a in conv_node.attribute}
        self.assertIn("kernel_shape", attrs)
        self.assertEqual(list(attrs["kernel_shape"].ints), [3, 3])
    
    def test_add_conv_no_bias(self):
        """Test add_conv without bias."""
        model = create_simple_model()
        
        modified = add_conv(
            model, "conv1", "input", "conv_out",
            in_channels=3, out_channels=64, kernel_size=3,
            bias=False
        )
        
        # Should only have weight, no bias
        conv_node = find_node_by_name(modified, "conv1")
        self.assertEqual(len(conv_node.input), 2)  # input + weight
    
    def test_add_linear(self):
        """Test add_linear convenience function."""
        model = create_simple_model()
        
        modified = add_linear(
            model, "fc1", "flatten", "fc_out",
            in_features=512, out_features=10, bias=True
        )
        
        gemm_node = find_node_by_name(modified, "fc1")
        self.assertIsNotNone(gemm_node)
        self.assertEqual(gemm_node.op_type, "Gemm")
        
        # Check transB attribute
        attrs = {a.name: a for a in gemm_node.attribute}
        self.assertEqual(attrs["transB"].i, 1)
    
    def test_add_activation_relu(self):
        """Test add_activation with ReLU."""
        model = create_simple_model()
        
        modified = add_activation(model, "relu1", "in", "out", "relu")
        
        relu_node = find_node_by_name(modified, "relu1")
        self.assertEqual(relu_node.op_type, "Relu")
    
    def test_add_activation_sigmoid(self):
        """Test add_activation with Sigmoid."""
        model = create_simple_model()
        
        modified = add_activation(model, "sig1", "in", "out", "sigmoid")
        
        sig_node = find_node_by_name(modified, "sig1")
        self.assertEqual(sig_node.op_type, "Sigmoid")
    
    def test_add_activation_leaky_relu(self):
        """Test add_activation with LeakyRelu."""
        model = create_simple_model()
        
        modified = add_activation(model, "lrelu1", "in", "out", "leaky_relu", alpha=0.1)
        
        lrelu_node = find_node_by_name(modified, "lrelu1")
        self.assertEqual(lrelu_node.op_type, "LeakyRelu")
    
    def test_add_activation_alias_map(self):
        """Test that activation aliases work."""
        model = create_simple_model()
        
        # Test various aliases
        for alias, expected in [
            ("relu", "Relu"),
            ("sigmoid", "Sigmoid"),
            ("tanh", "Tanh"),
            ("leakyrelu", "LeakyRelu"),
            ("softmax", "Softmax"),
            ("gelu", "Gelu"),
        ]:
            modified = add_activation(model, f"act_{alias}", "in", "out", alias)
            node = find_node_by_name(modified, f"act_{alias}")
            self.assertEqual(node.op_type, expected, f"Alias '{alias}' failed")
    
    def test_add_norm_batchnorm(self):
        """Test add_norm with BatchNorm."""
        model = create_simple_model()
        
        modified = add_norm(
            model, "bn1", "conv_out", "bn_out",
            norm_type="batchnorm", num_features=64
        )
        
        bn_node = find_node_by_name(modified, "bn1")
        self.assertEqual(bn_node.op_type, "BatchNormalization")
        
        # Check all 4 initializers
        init_names = [i.name for i in modified.graph.initializer]
        self.assertIn("bn1_scale", init_names)
        self.assertIn("bn1_bias", init_names)
        self.assertIn("bn1_mean", init_names)
        self.assertIn("bn1_var", init_names)
    
    def test_add_norm_layernorm(self):
        """Test add_norm with LayerNorm."""
        model = create_simple_model()
        
        modified = add_norm(
            model, "ln1", "input", "ln_out",
            norm_type="layernorm", num_features=256
        )
        
        ln_node = find_node_by_name(modified, "ln1")
        self.assertEqual(ln_node.op_type, "LayerNormalization")
    
    def test_add_pooling_max(self):
        """Test add_pooling with MaxPool."""
        model = create_simple_model()
        
        modified = add_pooling(
            model, "pool1", "conv_out", "pool_out",
            pool_type="max", kernel_size=2, stride=2
        )
        
        pool_node = find_node_by_name(modified, "pool1")
        self.assertEqual(pool_node.op_type, "MaxPool")
    
    def test_add_pooling_global_avg(self):
        """Test add_pooling with GlobalAveragePool."""
        model = create_simple_model()
        
        modified = add_pooling(
            model, "gap", "conv_out", "gap_out",
            pool_type="global_avg"
        )
        
        gap_node = find_node_by_name(modified, "gap")
        self.assertEqual(gap_node.op_type, "GlobalAveragePool")
    
    def test_add_dropout(self):
        """Test add_dropout."""
        model = create_simple_model()
        
        modified = add_dropout(model, "drop1", "fc_out", "drop_out", ratio=0.3)
        
        drop_node = find_node_by_name(modified, "drop1")
        self.assertEqual(drop_node.op_type, "Dropout")
        
        # Check ratio initializer
        init_names = [i.name for i in modified.graph.initializer]
        self.assertIn("drop1_ratio", init_names)
    
    def test_add_shape_manipulation_flatten(self):
        """Test add_shape_manipulation with Flatten."""
        model = create_simple_model()
        
        modified = add_shape_manipulation(
            model, "flat1", "conv_out", "flat_out", "Flatten", axis=1
        )
        
        flat_node = find_node_by_name(modified, "flat1")
        self.assertEqual(flat_node.op_type, "Flatten")
    
    def test_add_shape_manipulation_reshape(self):
        """Test add_shape_manipulation with Reshape."""
        model = create_simple_model()
        
        modified = add_shape_manipulation(
            model, "reshape1", "input", "out", "Reshape", shape=[1, -1]
        )
        
        reshape_node = find_node_by_name(modified, "reshape1")
        self.assertEqual(reshape_node.op_type, "Reshape")
        
        # Check shape initializer
        init_names = [i.name for i in modified.graph.initializer]
        self.assertIn("reshape1_shape", init_names)


class TestMicrosoftDomainOps(unittest.TestCase):
    """Test Microsoft domain operators."""
    
    def test_add_fused_conv(self):
        """Test add_fused_conv with ReLU activation."""
        model = create_simple_model()
        
        modified = add_fused_conv(
            model, "fused_conv1", "input", "fused_out",
            in_channels=3, out_channels=64, kernel_size=3,
            activation="Relu"
        )
        
        fused_node = find_node_by_name(modified, "fused_conv1")
        self.assertIsNotNone(fused_node)
        self.assertEqual(fused_node.op_type, "FusedConv")
        self.assertEqual(fused_node.domain, "com.microsoft")
        
        # Check activation attribute
        attrs = {a.name: a for a in fused_node.attribute}
        self.assertEqual(attrs["activation"].s, b"Relu")
    
    def test_add_fused_gemm(self):
        """Test add_fused_gemm."""
        model = create_simple_model()
        
        modified = add_fused_gemm(
            model, "fused_fc1", "input", "fused_out",
            in_features=100, out_features=10,
            activation="Relu"
        )
        
        fused_node = find_node_by_name(modified, "fused_fc1")
        self.assertIsNotNone(fused_node)
        self.assertEqual(fused_node.op_type, "FusedGemm")
        self.assertEqual(fused_node.domain, "com.microsoft")
    
    def test_add_attention(self):
        """Test add_attention."""
        model = create_simple_model()
        
        modified = add_attention(
            model, "attn1", "query", "attn_out",
            num_heads=8
        )
        
        attn_node = find_node_by_name(modified, "attn1")
        self.assertIsNotNone(attn_node)
        self.assertEqual(attn_node.op_type, "Attention")
        self.assertEqual(attn_node.domain, "com.microsoft")
        
        # Check num_heads attribute
        attrs = {a.name: a for a in attn_node.attribute}
        self.assertEqual(attrs["num_heads"].i, 8)
    
    def test_microsoft_op_with_add_layer(self):
        """Test adding Microsoft op directly via add_layer."""
        model = create_simple_model()
        
        modified = add_layer(
            model, "Gelu", "input", "out",
            name="gelu1",
            domain="com.microsoft"
        )
        
        gelu_node = find_node_by_name(modified, "gelu1")
        self.assertIsNotNone(gelu_node)
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertEqual(gelu_node.domain, "com.microsoft")


class TestOpSchemas(unittest.TestCase):
    """Test operator schema definitions."""
    
    def test_get_op_schema_standard(self):
        """Test get_op_schema for standard ops."""
        schema = get_op_schema("Conv")
        self.assertIsNotNone(schema)
        self.assertIn("attributes", schema)
        self.assertIn("kernel_shape", schema["attributes"])
    
    def test_get_op_schema_microsoft(self):
        """Test get_op_schema for Microsoft ops."""
        schema = get_op_schema("FusedConv", domain="com.microsoft")
        self.assertIsNotNone(schema)
        self.assertIn("activation", schema["attributes"])
    
    def test_get_op_schema_not_found(self):
        """Test get_op_schema returns None for unknown ops."""
        schema = get_op_schema("UnknownOp")
        self.assertIsNone(schema)
    
    def test_all_schemas_available(self):
        """Test that major ops are in schemas."""
        # Standard ops
        for op in ["Conv", "Gemm", "Relu", "BatchNormalization", "MaxPool", "Dropout"]:
            self.assertIn(op, ONNX_OP_SCHEMAS, f"{op} not in ONNX_OP_SCHEMAS")
        
        # Microsoft ops
        for op in ["FusedConv", "FusedGemm", "Attention", "Gelu"]:
            full_name = f"com.microsoft.{op}"
            self.assertIn(full_name, MICROSOFT_OP_SCHEMAS, f"{full_name} not in MICROSOFT_OP_SCHEMAS")


class TestModelModifierIntegration(unittest.TestCase):
    """Test ModelModifier.add_layer integration."""
    
    def test_model_modifier_add_layer_chain(self):
        """Test ModelModifier.add_layer chain method."""
        model = create_simple_model()
        
        modifier = ModelModifier(model)
        modifier.add_layer("Conv", "input", "c1", name="conv1",
                          kernel_size=3, in_channels=3, out_channels=32)
        
        modified = modifier.get_model()
        conv_node = find_node_by_name(modified, "conv1")
        self.assertIsNotNone(conv_node)
    
    def test_model_modifier_chain_multiple_layers(self):
        """Test chaining multiple add_layer calls."""
        model = create_simple_model()
        
        modifier = ModelModifier(model)
        modifier \
            .add_layer("Conv", "input", "c1", name="conv1",
                      kernel_size=3, in_channels=3, out_channels=32) \
            .add_layer("Relu", "c1", "r1", name="relu1") \
            .add_layer("MaxPool", "r1", "p1", name="pool1",
                      attributes={"kernel_shape": [2, 2], "strides": [2, 2]})
        
        modified = modifier.get_model()
        
        # All nodes should exist
        self.assertIsNotNone(find_node_by_name(modified, "conv1"))
        self.assertIsNotNone(find_node_by_name(modified, "relu1"))
        self.assertIsNotNone(find_node_by_name(modified, "pool1"))


class TestModelValidity(unittest.TestCase):
    """Test that modified models remain valid."""
    
    def test_conv_model_valid(self):
        """Test Conv model passes ONNX checker."""
        model = create_simple_model()
        
        modified = add_conv(
            model, "conv1", "input", "conv_out",
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        
        onnx.checker.check_model(modified)
    
    def test_linear_model_valid(self):
        """Test Linear model passes ONNX checker."""
        model = create_simple_model()
        
        modified = add_linear(
            model, "fc1", "input", "fc_out",
            in_features=100, out_features=10
        )
        
        onnx.checker.check_model(modified)
    
    def test_save_and_reload(self):
        """Test modified models can be saved and reloaded."""
        model = create_simple_model()
        
        modified = add_conv(
            model, "conv1", "input", "conv_out",
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )
        modified = add_activation(modified, "relu1", "conv_out", "out", "relu")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.onnx"
            onnx.save(modified, str(output_path))
            
            loaded = onnx.load(str(output_path))
            onnx.checker.check_model(loaded)


class TestAdvancedUsage(unittest.TestCase):
    """Test advanced usage patterns."""
    
    def test_build_simple_cnn(self):
        """Test building a simple CNN."""
        model = create_simple_model()
        
        # Build: Conv -> ReLU -> MaxPool -> Flatten -> Linear
        model = add_conv(model, "conv1", "input", "c1", 3, 32, 3, padding=1)
        model = add_activation(model, "relu1", "c1", "r1", "relu")
        model = add_pooling(model, "pool1", "r1", "p1", "max", 2, stride=2)
        model = add_shape_manipulation(model, "flat", "p1", "f", "Flatten")
        model = add_linear(model, "fc1", "f", "output", 32*16*16, 10)
        
        # Verify structure
        self.assertIsNotNone(find_node_by_name(model, "conv1"))
        self.assertIsNotNone(find_node_by_name(model, "relu1"))
        self.assertIsNotNone(find_node_by_name(model, "pool1"))
        self.assertIsNotNone(find_node_by_name(model, "flat"))
        self.assertIsNotNone(find_node_by_name(model, "fc1"))
        
        onnx.checker.check_model(model)
    
    def test_build_with_microsoft_ops(self):
        """Test building model with Microsoft fused ops."""
        model = create_simple_model()
        
        # FusedConv+ReLU -> FusedGemm+ReLU
        model = add_fused_conv(
            model, "fused1", "input", "f1", 3, 32, 3,
            padding=1, activation="Relu"
        )
        model = add_shape_manipulation(model, "flat", "f1", "f", "Flatten")
        model = add_fused_gemm(
            model, "fused_fc", "f", "output", 32*32*32, 10,
            activation=""
        )
        
        # Note: Microsoft ops won't pass standard ONNX checker
        # but should be structurally correct
        self.assertIsNotNone(find_node_by_name(model, "fused1"))
        self.assertIsNotNone(find_node_by_name(model, "fused_fc"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
