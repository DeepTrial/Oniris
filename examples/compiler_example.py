#!/usr/bin/env python3
"""
ONNX Model Compiler - Full Example

This example demonstrates the complete compilation pipeline:
1. Create/load an ONNX model
2. Define custom patterns for matching
3. Run the compiler with optimization, shape inference, and pattern matching
4. Analyze the JSON output

Usage:
    python compiler_full_example.py
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import oniris

# Try to import onnx
try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnx package not installed. Install with: pip install onnx")


def create_sample_model():
    """Create a sample model with Conv + ReLU pattern for demonstration."""
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
        'sample_model',
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='oniris_example')
    model.opset_import[0].version = 13
    
    return model


def main():
    print("=" * 70)
    print("Oniris ONNX Model Compiler - Full Example")
    print("=" * 70)
    
    if not HAS_ONNX:
        print("\nError: onnx package is required for this example")
        print("Install with: pip install onnx")
        return 1
    
    # Step 1: Create a sample model
    print("\n[Step 1] Creating sample model...")
    onnx_model = create_sample_model()
    if onnx_model is None:
        print("Failed to create model")
        return 1
    
    # Save the model
    input_path = '/tmp/compiler_example_input.onnx'
    onnx.save(onnx_model, input_path)
    print(f"  ✓ Model saved to: {input_path}")
    print(f"  ✓ Model has {len(onnx_model.graph.node)} nodes")
    
    # Step 2: Create compiler and define patterns
    print("\n[Step 2] Setting up compiler with patterns...")
    compiler = oniris.Compiler()
    
    # Add custom patterns
    patterns = [
        # Conv + ReLU fusion pattern
        oniris.PatternDefinition("ConvRelu", """
            Conv(?, c0)
            Relu(c0, ?)
        """),
        
        # Conv + BatchNorm + ReLU pattern
        oniris.PatternDefinition("ConvBnRelu", """
            Conv(?, c0)
            BatchNormalization(c0, bn0)
            Relu(bn0, ?)
        """),
        
        # Residual connection
        oniris.PatternDefinition("Residual", """
            Conv(?, c0)
            Add([c0, ?], ?)
        """),
    ]
    
    compiler.add_patterns(patterns)
    print(f"  ✓ Added {len(patterns)} custom patterns")
    
    # Also add common patterns
    common_patterns = oniris.get_common_patterns()
    compiler.add_patterns(common_patterns)
    print(f"  ✓ Added {len(common_patterns)} common patterns")
    
    # Step 3: Configure compiler options
    print("\n[Step 3] Configuring compiler options...")
    options = oniris.CompilerOptions()
    options.enable_optimization = True
    options.enable_shape_inference = True
    options.enable_pattern_matching = True
    options.verbose = True
    options.save_optimized_model = True
    options.save_json_result = True
    options.json_output_path = '/tmp/compiler_example_result.json'
    
    print("  ✓ Options configured:")
    print(f"    - Optimization: {options.enable_optimization}")
    print(f"    - Shape inference: {options.enable_shape_inference}")
    print(f"    - Pattern matching: {options.enable_pattern_matching}")
    print(f"    - JSON output: {options.json_output_path}")
    
    # Step 4: Run compilation
    print("\n[Step 4] Running compilation pipeline...")
    output_path = '/tmp/compiler_example_output.onnx'
    result = compiler.compile_file(input_path, output_path, options)
    
    if not result.success:
        print(f"  ✗ Compilation failed: {result.error_msg}")
        return 1
    
    print(f"  ✓ Compilation successful!")
    print(f"    - Duration: {result.duration_ms:.2f} ms")
    
    # Step 5: Analyze results
    print("\n[Step 5] Analyzing compilation results...")
    
    # Model info
    print("\n  Model Information:")
    print(f"    - Producer: {result.model_info.producer_name}")
    print(f"    - IR Version: {result.model_info.ir_version}")
    print(f"    - Nodes: {result.model_info.num_nodes}")
    print(f"    - Inputs: {result.model_info.num_inputs}")
    print(f"    - Outputs: {result.model_info.num_outputs}")
    print(f"    - Op types: {', '.join(result.model_info.op_types_used)}")
    
    # Optimization stats
    print("\n  Optimization:")
    print(f"    - Success: {result.optimization_stats.success}")
    print(f"    - Changes: {result.optimization_stats.num_changes}")
    print(f"    - Iterations: {result.optimization_stats.num_iterations}")
    if result.optimization_stats.pass_stats:
        print(f"    - Pass statistics:")
        for pass_name, count in result.optimization_stats.pass_stats.items():
            print(f"      - {pass_name}: {count}")
    
    # Shape inference stats
    print("\n  Shape Inference:")
    print(f"    - Success: {result.shape_inference_stats.success}")
    print(f"    - Nodes processed: {result.shape_inference_stats.num_nodes_processed}")
    print(f"    - Nodes failed: {result.shape_inference_stats.num_nodes_failed}")
    if result.shape_inference_stats.failed_nodes:
        print(f"    - Failed nodes: {', '.join(result.shape_inference_stats.failed_nodes)}")
    
    # Pattern matching results
    print("\n  Pattern Matching:")
    print(f"    - Total patterns: {result.pattern_matching_summary.total_patterns}")
    print(f"    - Patterns with matches: {result.pattern_matching_summary.patterns_with_matches}")
    print(f"    - Total matches: {result.pattern_matching_summary.total_matches}")
    
    if result.pattern_matching_summary.total_matches > 0:
        print(f"    - Match details:")
        for pattern_name, count in result.pattern_matching_summary.match_counts.items():
            if count > 0:
                print(f"      - {pattern_name}: {count} matches")
        
        # Show detailed match information
        print(f"\n    - Detailed matches:")
        for pattern_name, matches in result.pattern_matching_summary.pattern_results.items():
            if matches:
                print(f"\n      Pattern '{pattern_name}':")
                for match in matches:
                    print(f"        Match {match.match_id}:")
                    for node in match.nodes:
                        print(f"          - {node.node_name} ({node.op_type})")
                        if node.input_shapes:
                            shapes_str = ', '.join(str(s) for s in node.input_shapes)
                            print(f"            Inputs: {shapes_str}")
                        if node.output_shapes:
                            shapes_str = ', '.join(str(s) for s in node.output_shapes)
                            print(f"            Outputs: {shapes_str}")
    
    # Step 6: JSON output analysis
    print("\n[Step 6] JSON Output Analysis...")
    
    # Get JSON string
    json_str = result.to_json(pretty=True)
    print(f"  ✓ JSON output size: {len(json_str)} characters")
    
    # Save to file
    json_path = '/tmp/compiler_example_result.json'
    result.save_json(json_path, pretty=True)
    print(f"  ✓ JSON saved to: {json_path}")
    
    # Parse and analyze JSON
    data = json.loads(json_str)
    
    print("\n  JSON Structure:")
    print(f"    - Top-level keys: {list(data.keys())}")
    print(f"    - Model info keys: {list(data['model_info'].keys())}")
    print(f"    - Optimization keys: {list(data['optimization'].keys())}")
    print(f"    - Pattern matching keys: {list(data['pattern_matching'].keys())}")
    
    # Step 7: Advanced - Custom analysis
    print("\n[Step 7] Advanced Analysis...")
    
    # Extract all Conv layers
    conv_nodes = []
    if 'pattern_matching' in data and 'results' in data['pattern_matching']:
        for pattern_name, matches in data['pattern_matching']['results'].items():
            for match in matches:
                for node in match['nodes']:
                    if node['op_type'] == 'Conv':
                        conv_nodes.append({
                            'name': node['name'],
                            'pattern': pattern_name,
                            'input_shapes': node['input_shapes'],
                            'output_shapes': node['output_shapes']
                        })
    
    if conv_nodes:
        print(f"  ✓ Found {len(conv_nodes)} Conv nodes in patterns:")
        for conv in conv_nodes:
            print(f"    - {conv['name']} (in pattern '{conv['pattern']}')")
    
    # Step 8: Cleanup
    print("\n[Step 8] Cleanup...")
    try:
        os.remove(input_path)
        os.remove(output_path)
        os.remove(json_path)
        print("  ✓ Temporary files removed")
    except:
        pass
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
