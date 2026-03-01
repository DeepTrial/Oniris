"""
Test script for Oniris ONNX Model Compiler
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import oniris

# Try to import onnx
try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnx package not installed. Some tests will be skipped.")


def create_test_conv_relu_model():
    """Create a simple Conv + ReLU model for testing."""
    if not HAS_ONNX:
        return None
    
    # Create input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
    
    # Create weights
    weight_data = np.random.randn(16, 3, 3, 3).astype(np.float32)
    weight_tensor = numpy_helper.from_array(weight_data, 'weight')
    
    bias_data = np.random.randn(16).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias_data, 'bias')
    
    # Create Conv node
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight', 'bias'],
        outputs=['conv_out'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        name='conv1'
    )
    
    # Create ReLU node
    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['output'],
        name='relu1'
    )
    
    # Create output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
    
    # Create graph
    graph = helper.make_graph(
        [conv_node, relu_node],
        'test_conv_relu',
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='oniris_test')
    model.opset_import[0].version = 13
    
    return model


def test_basic_compiler():
    """Test basic compiler functionality."""
    print("=" * 60)
    print("Test 1: Basic Compiler Creation")
    print("=" * 60)
    
    compiler = oniris.Compiler()
    assert compiler.pattern_count == 0
    
    # Add patterns
    success = compiler.add_pattern("ConvRelu", """
        Conv(?, c0)
        Relu(c0, ?)
    """)
    assert success
    assert compiler.pattern_count == 1
    
    print(f"✓ Compiler created with {compiler.pattern_count} pattern")
    print(f"  Pattern names: {compiler.get_pattern_names()}")


def test_common_patterns():
    """Test common patterns."""
    print("\n" + "=" * 60)
    print("Test 2: Common Patterns")
    print("=" * 60)
    
    patterns = oniris.get_common_patterns()
    assert len(patterns) > 0
    
    print(f"✓ Got {len(patterns)} common patterns:")
    for p in patterns:
        print(f"  - {p.name}")
    
    # Add to compiler
    compiler = oniris.Compiler()
    compiler.add_patterns(patterns)
    assert compiler.pattern_count == len(patterns)
    print(f"✓ All patterns added to compiler")


def test_compile_onnx_model():
    """Test compiling an ONNX model."""
    if not HAS_ONNX:
        print("\n" + "=" * 60)
        print("Test 3: Compile ONNX Model - SKIPPED (onnx not installed)")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("Test 3: Compile ONNX Model")
    print("=" * 60)
    
    # Create test model
    onnx_model = create_test_conv_relu_model()
    assert onnx_model is not None
    
    # Save to file
    test_input_path = '/tmp/test_conv_relu.onnx'
    onnx.save(onnx_model, test_input_path)
    print(f"✓ Created test model: {test_input_path}")
    
    # Compile with Oniris
    compiler = oniris.Compiler()
    compiler.add_patterns(oniris.get_common_patterns())
    
    options = oniris.CompilerOptions()
    options.verbose = True
    options.enable_optimization = True
    options.enable_shape_inference = True
    options.enable_pattern_matching = True
    
    result = compiler.compile_file(test_input_path, options=options)
    
    print(f"\n✓ Compilation result:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_ms:.2f} ms")
    
    print(f"\n✓ Model info:")
    print(f"  Nodes: {result.model_info.num_nodes}")
    print(f"  Inputs: {result.model_info.num_inputs}")
    print(f"  Outputs: {result.model_info.num_outputs}")
    print(f"  Op types: {result.model_info.op_types_used}")
    
    print(f"\n✓ Optimization:")
    print(f"  Success: {result.optimization_stats.success}")
    print(f"  Changes: {result.optimization_stats.num_changes}")
    print(f"  Iterations: {result.optimization_stats.num_iterations}")
    
    print(f"\n✓ Shape inference:")
    print(f"  Success: {result.shape_inference_stats.success}")
    print(f"  Processed: {result.shape_inference_stats.num_nodes_processed}")
    print(f"  Failed: {result.shape_inference_stats.num_nodes_failed}")
    
    print(f"\n✓ Pattern matching:")
    print(f"  Total patterns: {result.pattern_matching_summary.total_patterns}")
    print(f"  Patterns with matches: {result.pattern_matching_summary.patterns_with_matches}")
    print(f"  Total matches: {result.pattern_matching_summary.total_matches}")
    
    for pattern, count in result.pattern_matching_summary.match_counts.items():
        if count > 0:
            print(f"    {pattern}: {count} matches")
    
    # Test JSON output
    json_str = result.to_json(pretty=True)
    assert len(json_str) > 0
    
    # Parse JSON
    data = json.loads(json_str)
    assert data['success'] == result.success
    assert data['model_info']['num_nodes'] == result.model_info.num_nodes
    
    print(f"\n✓ JSON output validated (length: {len(json_str)} chars)")
    
    # Save JSON
    json_path = '/tmp/compilation_result.json'
    result.save_json(json_path, pretty=True)
    print(f"✓ JSON saved to: {json_path}")
    
    # Cleanup
    try:
        os.remove(test_input_path)
        os.remove(json_path)
    except:
        pass


def test_pattern_matching_details():
    """Test detailed pattern matching results."""
    if not HAS_ONNX:
        print("\n" + "=" * 60)
        print("Test 4: Pattern Matching Details - SKIPPED (onnx not installed)")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("Test 4: Pattern Matching Details")
    print("=" * 60)
    
    # Create test model
    onnx_model = create_test_conv_relu_model()
    test_input_path = '/tmp/test_conv_relu.onnx'
    onnx.save(onnx_model, test_input_path)
    
    # Load with onnx bridge
    model = oniris.load_onnx_model(test_input_path)
    print(f"✓ Model loaded via onnx bridge")
    print(f"  Nodes: {model.get_graph().get_nodes().__len__()}")
    
    # Create compiler and add patterns
    compiler = oniris.Compiler()
    compiler.add_pattern("ConvRelu", """
        Conv(?, c0)
        Relu(c0, ?)
    """)
    
    # Run pattern matching only
    summary = compiler.run_pattern_matching(model, oniris.PatternMatchType.ALL)
    
    print(f"\n✓ Pattern matching summary:")
    print(f"  Total patterns: {summary.total_patterns}")
    print(f"  Total matches: {summary.total_matches}")
    
    # Check detailed results
    if summary.total_matches > 0:
        for pattern_name, matches in summary.pattern_results.items():
            print(f"\n  Pattern '{pattern_name}':")
            for match in matches:
                print(f"    Match {match.match_id}:")
                for node in match.nodes:
                    print(f"      - {node.node_name} ({node.op_type})")
                    if node.input_shapes:
                        print(f"        Input shapes: {node.input_shapes}")
                    if node.output_shapes:
                        print(f"        Output shapes: {node.output_shapes}")
    
    # Cleanup
    try:
        os.remove(test_input_path)
    except:
        pass


def test_json_structure():
    """Test JSON output structure."""
    print("\n" + "=" * 60)
    print("Test 5: JSON Output Structure")
    print("=" * 60)
    
    # Create a simple model programmatically
    model = oniris.Model()
    graph = model.create_graph("test")
    
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([1, 3, 32, 32])
    input_info.dtype = oniris.DataType.FLOAT
    graph.add_input(input_info)
    
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("W")
    conv.add_input("B")
    conv.add_output("conv_out")
    conv.set_attribute_ints("kernel_shape", [3, 3])
    
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    # Compile
    compiler = oniris.Compiler()
    compiler.add_pattern("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)")
    
    result = compiler.compile_model(model)
    
    # Get JSON
    json_str = result.to_json(pretty=True)
    data = json.loads(json_str)
    
    # Verify structure
    required_keys = [
        'success', 'input_path', 'output_path', 'timing',
        'model_info', 'optimization', 'shape_inference', 'pattern_matching'
    ]
    
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
    
    print("✓ JSON structure validated")
    print(f"  Keys: {list(data.keys())}")
    
    # Verify model_info
    model_info_keys = ['producer_name', 'ir_version', 'num_nodes', 'op_types_used', 'op_type_counts']
    for key in model_info_keys:
        assert key in data['model_info'], f"Missing model_info key: {key}"
    
    print("✓ Model info structure validated")
    
    # Verify pattern_matching
    pattern_keys = ['total_patterns', 'patterns_with_matches', 'total_matches', 'match_counts', 'results']
    for key in pattern_keys:
        assert key in data['pattern_matching'], f"Missing pattern_matching key: {key}"
    
    print("✓ Pattern matching structure validated")


def test_convenience_function():
    """Test convenience compile_model function."""
    print("\n" + "=" * 60)
    print("Test 6: Convenience Function")
    print("=" * 60)
    
    if HAS_ONNX:
        # Create and save a test model
        onnx_model = create_test_conv_relu_model()
        test_path = '/tmp/test_conv.onnx'
        onnx.save(onnx_model, test_path)
        
        # Use convenience function
        patterns = [oniris.PatternDefinition("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)")]
        options = oniris.CompilerOptions()
        options.verbose = False
        
        result = oniris.compile_model(test_path, "", patterns, options)
        
        print(f"✓ Convenience function worked")
        print(f"  Success: {result.success}")
        print(f"  Pattern matches: {result.pattern_matching_summary.total_matches}")
        
        # Cleanup
        try:
            os.remove(test_path)
        except:
            pass
    else:
        print("  SKIPPED (onnx not installed)")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Oniris ONNX Model Compiler - Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_compiler,
        test_common_patterns,
        test_compile_onnx_model,
        test_pattern_matching_details,
        test_json_structure,
        test_convenience_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
