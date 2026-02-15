"""System tests for Oniris using real ONNX models.

This test suite downloads and tests Oniris against real ONNX models from:
- ONNX Model Zoo (https://github.com/onnx/models)
- HuggingFace model hub
- Custom test models

Requirements:
    pip install pytest pytest-cov onnx transformers torch
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

# Add the python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import oniris


# =============================================================================
# Test Model URLs and Configurations
# =============================================================================

# Models from ONNX Model Zoo that we test against
ONNX_MODEL_ZOO_MODELS = {
    "resnet50": {
        "url": "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx",
        "input_shapes": [[1, 3, 224, 224]],
        "opset": 7,
    },
    "mobilenetv2": {
        "url": "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "input_shapes": [[1, 3, 224, 224]],
        "opset": 7,
    },
    "squeezenet": {
        "url": "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
        "input_shapes": [[1, 3, 224, 224]],
        "opset": 9,
    },
}


def download_model(url: str, cache_dir: Optional[str] = None) -> str:
    """Download a model from URL to cache directory.
    
    Args:
        url: URL to download from
        cache_dir: Directory to cache models (default: ~/.cache/oniris/models)
        
    Returns:
        Path to downloaded model file
    """
    import urllib.request
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "oniris" / "models"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split("/")[-1]
    filepath = cache_dir / filename
    
    if filepath.exists():
        return str(filepath)
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    
    return str(filepath)


# =============================================================================
# System Test Base Class
# =============================================================================

class ModelTestBase(unittest.TestCase):
    """Base class for model tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def load_and_verify(self, model_path: str) -> oniris.Model:
        """Load a model and verify it's valid.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Loaded model
        """
        self.assertTrue(oniris.is_valid_onnx_file(model_path))
        
        model = oniris.load_model(model_path)
        self.assertIsNotNone(model)
        
        # Verify model structure
        valid, error_msg = model.validate()
        self.assertTrue(valid, f"Model validation failed: {error_msg}")
        
        return model
    
    def test_shape_inference(self, model: oniris.Model) -> None:
        """Test shape inference on a model.
        
        Args:
            model: Model to test
        """
        engine = oniris.ShapeInferenceEngine.get_instance()
        
        # Should not fail on unknown ops (skip gracefully)
        success = engine.infer_model(model, fail_on_unknown=False)
        self.assertTrue(success)
        
        # Get graph and check that outputs have shapes
        graph = model.get_graph()
        for node in graph.get_nodes():
            # Most nodes should have inferred shapes
            if node.get_op_type() in engine.get_supported_ops():
                self.assertTrue(
                    node.has_inferred_shapes() or len(node.get_outputs()) == 0,
                    f"Node {node.get_name()} ({node.get_op_type()}) missing inferred shapes"
                )
    
    def test_simplification(self, model: oniris.Model) -> oniris.Model:
        """Test model simplification.
        
        Args:
            model: Model to simplify
            
        Returns:
            Simplified model
        """
        options = oniris.SimplifyOptions()
        options.fail_on_unsupported = False
        options.verbose = False
        
        result = oniris.Simplifier.simplify(model, options)
        
        self.assertTrue(result.success, f"Simplification failed: {result.error_msg}")
        self.assertGreaterEqual(result.num_changes, 0)
        
        # Verify simplified model is still valid
        valid, error_msg = model.validate()
        self.assertTrue(valid, f"Simplified model invalid: {error_msg}")
        
        return model
    
    def test_save_and_reload(self, model: oniris.Model, name: str) -> oniris.Model:
        """Test saving and reloading a model.
        
        Args:
            model: Model to save
            name: Base name for saved file
            
        Returns:
            Reloaded model
        """
        output_path = Path(self.temp_dir.name) / f"{name}_test.onnx"
        
        success = oniris.save_model(model, str(output_path))
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        
        # Reload
        reloaded = oniris.load_model(str(output_path))
        self.assertIsNotNone(reloaded)
        
        # Verify basic properties match
        self.assertEqual(model.get_ir_version(), reloaded.get_ir_version())
        
        return reloaded


# =============================================================================
# Synthetic Model Tests
# =============================================================================

class TestSyntheticModels(ModelTestBase):
    """Tests using programmatically created models."""
    
    def test_simple_conv_relu(self):
        """Test a simple Conv -> ReLU network."""
        model = oniris.Model(8)
        model.set_producer_name("oniris_test")
        
        opset = oniris.OpsetImport()
        opset.domain = ""
        opset.version = 13
        model.add_opset_import(opset)
        
        graph = model.create_graph("test_conv_relu")
        
        # Inputs
        input_info = oniris.ValueInfo()
        input_info.name = "input"
        input_info.shape = oniris.Shape([1, 3, 32, 32])
        input_info.dtype = oniris.DataType.FLOAT32
        graph.add_input(input_info)
        
        weight_info = oniris.ValueInfo()
        weight_info.name = "weight"
        weight_info.shape = oniris.Shape([16, 3, 3, 3])
        weight_info.dtype = oniris.DataType.FLOAT32
        graph.set_value_info("weight", weight_info)
        
        # Conv
        conv = graph.create_node("Conv", "conv1")
        conv.add_input("input")
        conv.add_input("weight")
        conv.add_output("conv_out")
        conv.set_attribute_ints("kernel_shape", [3, 3])
        conv.set_attribute_ints("pads", [1, 1, 1, 1])
        
        # ReLU
        relu = graph.create_node("Relu", "relu1")
        relu.add_input("conv_out")
        relu.add_output("output")
        
        # Output
        output_info = oniris.ValueInfo()
        output_info.name = "output"
        graph.add_output(output_info)
        
        # Test shape inference
        self.test_shape_inference(model)
        
        # Verify shapes
        graph = model.get_graph()
        conv_out_info = graph.get_value_info("conv_out")
        self.assertIsNotNone(conv_out_info)
        self.assertEqual(conv_out_info.shape.get_dim(0).get_static_value(), 1)
        self.assertEqual(conv_out_info.shape.get_dim(1).get_static_value(), 16)
        self.assertEqual(conv_out_info.shape.get_dim(2).get_static_value(), 32)
        self.assertEqual(conv_out_info.shape.get_dim(3).get_static_value(), 32)
    
    def test_identity_elimination(self):
        """Test that identity nodes are eliminated."""
        model = oniris.Model(8)
        
        opset = oniris.OpsetImport()
        opset.version = 13
        model.add_opset_import(opset)
        
        graph = model.create_graph("test_identity")
        
        input_info = oniris.ValueInfo()
        input_info.name = "input"
        graph.add_input(input_info)
        
        # Identity -> Identity chain
        id1 = graph.create_node("Identity", "id1")
        id1.add_input("input")
        id1.add_output("mid1")
        
        id2 = graph.create_node("Identity", "id2")
        id2.add_input("mid1")
        id2.add_output("output")
        
        output_info = oniris.ValueInfo()
        output_info.name = "output"
        graph.add_output(output_info)
        
        initial_nodes = len(graph.get_nodes())
        self.assertEqual(initial_nodes, 2)
        
        # Simplify
        self.test_simplification(model)
        
        # Both identities should be eliminated
        graph = model.get_graph()
        self.assertEqual(len(graph.get_nodes()), 0)
    
    def test_dynamic_shapes(self):
        """Test handling of dynamic shapes."""
        model = oniris.Model(8)
        
        opset = oniris.OpsetImport()
        opset.version = 13
        model.add_opset_import(opset)
        
        graph = model.create_graph("test_dynamic")
        
        # Input with dynamic batch dimension
        input_info = oniris.ValueInfo()
        input_info.name = "input"
        input_info.shape = oniris.Shape([oniris.Dimension("batch"), 3, 224, 224])
        input_info.dtype = oniris.DataType.FLOAT32
        graph.add_input(input_info)
        
        # Relu
        relu = graph.create_node("Relu", "relu1")
        relu.add_input("input")
        relu.add_output("output")
        
        output_info = oniris.ValueInfo()
        output_info.name = "output"
        graph.add_output(output_info)
        
        # Shape inference
        self.test_shape_inference(model)
        
        # Check that dynamic dimension is preserved
        graph = model.get_graph()
        output_vi = graph.get_value_info("output")
        self.assertIsNotNone(output_vi)
        self.assertTrue(output_vi.shape.get_dim(0).is_dynamic())
        self.assertEqual(output_vi.shape.get_dim(1).get_static_value(), 3)
    
    def test_unsupported_op_handling(self):
        """Test that unsupported ops are handled gracefully."""
        model = oniris.Model(8)
        
        opset = oniris.OpsetImport()
        opset.version = 13
        model.add_opset_import(opset)
        
        graph = model.create_graph("test_unsupported")
        
        input_info = oniris.ValueInfo()
        input_info.name = "input"
        graph.add_input(input_info)
        
        # Add a fake unsupported op
        fake_op = graph.create_node("FakeCustomOp", "fake1")
        fake_op.add_input("input")
        fake_op.add_output("output")
        
        output_info = oniris.ValueInfo()
        output_info.name = "output"
        graph.add_output(output_info)
        
        # Should not fail on unsupported op
        options = oniris.SimplifyOptions()
        options.fail_on_unsupported = False
        
        result = oniris.Simplifier.simplify(model, options)
        self.assertTrue(result.success)


# =============================================================================
# ONNX Model Zoo Tests
# =============================================================================

@unittest.skipIf(
    os.environ.get("ONIRIS_SKIP_DOWNLOAD_TESTS", "0") == "1",
    "Skipping download tests"
)
class TestONNXModelZoo(ModelTestBase):
    """Tests using models from ONNX Model Zoo."""
    
    def _test_model_from_zoo(self, model_name: str):
        """Test a model from ONNX Model Zoo.
        
        Args:
            model_name: Name of the model to test
        """
        config = ONNX_MODEL_ZOO_MODELS.get(model_name)
        if config is None:
            self.skipTest(f"Unknown model: {model_name}")
        
        try:
            model_path = download_model(config["url"], str(self.cache_dir))
        except Exception as e:
            self.skipTest(f"Failed to download {model_name}: {e}")
        
        # Load and verify
        model = self.load_and_verify(model_path)
        
        # Print model info
        print(f"\nTesting {model_name}:")
        oniris.print_model_summary(model)
        
        # Test shape inference
        self.test_shape_inference(model)
        
        # Test simplification
        simplified = self.test_simplification(model)
        
        # Test save and reload
        self.test_save_and_reload(simplified, model_name)
    
    def test_resnet50(self):
        """Test ResNet50."""
        self._test_model_from_zoo("resnet50")
    
    def test_mobilenetv2(self):
        """Test MobileNetV2."""
        self._test_model_from_zoo("mobilenetv2")
    
    def test_squeezenet(self):
        """Test SqueezeNet."""
        self._test_model_from_zoo("squeezenet")


# =============================================================================
# Custom Handler Registration Tests
# =============================================================================

class TestCustomHandlers(ModelTestBase):
    """Tests for custom shape inference handlers."""
    
    def test_custom_handler_registration(self):
        """Test registering a custom shape inference handler."""
        
        def my_custom_infer(ctx):
            # Custom op: output shape is input shape with first dim doubled
            from oniris import InferenceResult, Dimension, Shape
            
            input_shape = ctx.input_shapes[0]
            output_shape = Shape(list(input_shape.get_dims()))
            
            if not output_shape.get_dim(0).is_dynamic():
                new_dim = output_shape.get_dim(0).get_static_value() * 2
                output_shape.get_dim(0).set_static_value(new_dim)
            
            return InferenceResult.Success([output_shape])
        
        # Register custom handler
        oniris.register_custom_shape_inference("MyCustomOp", my_custom_infer)
        
        # Create model with custom op
        model = oniris.Model(8)
        
        opset = oniris.OpsetImport()
        opset.version = 13
        model.add_opset_import(opset)
        
        graph = model.create_graph("test_custom")
        
        input_info = oniris.ValueInfo()
        input_info.name = "input"
        input_info.shape = oniris.Shape([4, 5])
        graph.add_input(input_info)
        
        custom_op = graph.create_node("MyCustomOp", "custom1")
        custom_op.add_input("input")
        custom_op.add_output("output")
        
        output_info = oniris.ValueInfo()
        output_info.name = "output"
        graph.add_output(output_info)
        
        # Run inference
        engine = oniris.ShapeInferenceEngine.get_instance()
        engine.infer_model(model, fail_on_unknown=False)
        
        # Check output shape
        output_vi = graph.get_value_info("output")
        self.assertIsNotNone(output_vi)
        self.assertEqual(output_vi.shape.get_dim(0).get_static_value(), 8)
        self.assertEqual(output_vi.shape.get_dim(1).get_static_value(), 5)
    
    def test_list_supported_ops(self):
        """Test listing supported operators."""
        engine = oniris.ShapeInferenceEngine.get_instance()
        ops = engine.get_supported_ops()
        
        # Check that common ops are supported
        self.assertIn("Conv", ops)
        self.assertIn("Relu", ops)
        self.assertIn("MatMul", ops)
        self.assertIn("Reshape", ops)
        self.assertIn("Transpose", ops)


# =============================================================================
# Command Line Interface Tests
# =============================================================================

class TestCLI(unittest.TestCase):
    """Tests for command line interface."""
    
    def test_simplify_function(self):
        """Test the simplify convenience function."""
        # Create a simple test model
        model = oniris.Model(8)
        
        opset = oniris.OpsetImport()
        opset.version = 13
        model.add_opset_import(opset)
        
        graph = model.create_graph("test_cli")
        
        input_info = oniris.ValueInfo()
        input_info.name = "input"
        graph.add_input(input_info)
        
        id_node = graph.create_node("Identity", "id1")
        id_node.add_input("input")
        id_node.add_output("output")
        
        output_info = oniris.ValueInfo()
        output_info.name = "output"
        graph.add_output(output_info)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.onnx"
            output_path = Path(tmpdir) / "output.onnx"
            
            oniris.save_model(model, str(input_path))
            
            # Simplify using convenience function
            result = oniris.simplify(str(input_path), str(output_path))
            
            self.assertTrue(result.success)
            self.assertTrue(output_path.exists())
            
            # Verify simplified model
            simplified = oniris.load_model(str(output_path))
            self.assertIsNotNone(simplified)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
