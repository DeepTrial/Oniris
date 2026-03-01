#!/usr/bin/env python3
"""
Complete Demo: YAML Pattern Files + Compiler

Demonstrates the complete workflow:
1. Load Patterns from YAML files
2. Manage Patterns (enable/disable/priority)
3. Apply to compiler
4. Compile model and get results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import oniris


def step1_load_patterns():
    """Step 1: Load Patterns from YAML files"""
    print("=" * 70)
    print("Step 1: Load from YAML Pattern files")
    print("=" * 70)
    
    pm = oniris.PatternManager()
    pattern_dir = os.path.join(os.path.dirname(__file__), 'patterns')
    
    # Load all YAML pattern files
    print(f"\n[1.1] Loading Pattern files from {pattern_dir}...")
    
    files = ['fusion_patterns.yaml', 'optimization_patterns.yaml']
    for filename in files:
        filepath = os.path.join(pattern_dir, filename)
        count = oniris.import_yaml_patterns(pm, filepath)
        print(f"      {filename}: {count} patterns")
    
    print(f"\n[1.2] Total loaded {pm.get_pattern_count()} patterns")
    return pm


def step2_manage_patterns(pm):
    """Step 2: Manage Patterns"""
    print("\n" + "=" * 70)
    print("Step 2: Manage Patterns")
    print("=" * 70)
    
    # Display current status
    print("\n[2.1] Current Patterns:")
    for name in pm.get_pattern_names():
        p = pm.get_pattern(name)
        enabled = "✓" if p.metadata.enabled else "✗"
        cat = str(p.metadata.category).split('.')[-1]
        print(f"      {enabled} {name} (priority: {p.metadata.priority}, cat: {cat})")
    
    # Disable some patterns
    print("\n[2.2] Disable Optimization patterns...")
    pm.set_category_enabled(oniris.PatternCategory.OPTIMIZATION, False)
    
    # Adjust priority
    print("\n[2.3] Set ConvRelu priority to highest...")
    pm.set_pattern_priority("ConvRelu", 999)
    
    # Display updated status
    print("\n[2.4] Updated status:")
    enabled_count = pm.get_enabled_pattern_count()
    total_count = pm.get_pattern_count()
    print(f"      Enabled Patterns: {enabled_count}/{total_count}")
    
    for name in pm.get_pattern_names():
        if pm.is_pattern_enabled(name):
            p = pm.get_pattern(name)
            print(f"      ✓ {name} (priority: {p.metadata.priority})")


def step3_apply_to_compiler(pm):
    """Step 3: Apply to compiler"""
    print("\n" + "=" * 70)
    print("Step 3: Apply to Compiler")
    print("=" * 70)
    
    print("\n[3.1] Create compiler from PatternManager...")
    compiler = pm.create_compiler()
    print(f"      Compiler contains {compiler.get_pattern_count()} patterns")
    
    print("\n[3.2] Patterns used by compiler:")
    for name in compiler.get_pattern_names():
        print(f"      - {name}")
    
    return compiler


def step4_compile_model(compiler):
    """Step 4: Compile model"""
    print("\n" + "=" * 70)
    print("Step 4: Compile Model")
    print("=" * 70)
    
    # Create a simple test model
    print("\n[4.1] Creating test model (Conv + ReLU)...")
    model = oniris.Model()
    graph = model.create_graph("test")
    
    # Add input
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([1, 3, 32, 32])
    input_info.dtype = oniris.DataType.FLOAT
    graph.add_input(input_info)
    
    # Add Conv node
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    conv.set_attribute_ints("kernel_shape", [3, 3])
    conv.set_attribute_ints("pads", [1, 1, 1, 1])
    conv.set_attribute_ints("strides", [1, 1])
    
    # Add ReLU node
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    # Add output
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    print("      Model created")
    
    # Configure compile options
    print("\n[4.2] Configure compile options...")
    options = oniris.CompilerOptions()
    options.verbose = False
    options.enable_optimization = True
    options.enable_shape_inference = True
    options.enable_pattern_matching = True
    options.pattern_match_before_opt = True  # Match before optimization
    
    # Compile
    print("\n[4.3] Compiling model...")
    result = compiler.compile_model(model, options)
    
    print(f"      Compile success: {result.success}")
    print(f"      Duration: {result.duration_ms:.2f} ms")
    print(f"      Optimization changes: {result.optimization_stats.num_changes}")
    print(f"      Pattern matches: {result.pattern_matching_summary.total_matches}")
    
    return result


def step5_analyze_results(result):
    """Step 5: Analyze results"""
    print("\n" + "=" * 70)
    print("Step 5: Analyze Compile Results")
    print("=" * 70)
    
    # Pattern match details
    if result.pattern_matching_summary.total_matches > 0:
        print("\n[5.1] Pattern match details:")
        for pattern_name, matches in result.pattern_matching_summary.pattern_results.items():
            print(f"\n      Pattern: {pattern_name}")
            print(f"      Matches: {len(matches)}")
            for match in matches:
                print(f"\n      Match {match.match_id}:")
                for node in match.nodes:
                    print(f"        - {node.node_name} ({node.op_type})")
    else:
        print("\n[5.1] No Pattern matches")
    
    # Export JSON
    print("\n[5.2] Compile result JSON:")
    json_str = result.to_json(pretty=True)
    
    # Parse and display key information
    data = json.loads(json_str)
    summary = {
        "success": data.get("success"),
        "duration_ms": data.get("timing", {}).get("duration_ms"),
        "model_info": {
            "num_nodes": data.get("model_info", {}).get("num_nodes"),
            "op_types": data.get("model_info", {}).get("op_types_used")
        },
        "optimization": {
            "num_changes": data.get("optimization", {}).get("num_changes")
        },
        "pattern_matching": {
            "total_matches": data.get("pattern_matching", {}).get("total_matches")
        }
    }
    print(json.dumps(summary, indent=4))


def main():
    """Run complete demo"""
    print("\n" + "=" * 70)
    print("Complete Demo: YAML Pattern Files + Compiler")
    print("=" * 70)
    
    try:
        # Step 1: Load patterns
        pm = step1_load_patterns()
        
        # Step 2: Manage patterns
        step2_manage_patterns(pm)
        
        # Step 3: Apply to compiler
        compiler = step3_apply_to_compiler(pm)
        
        # Step 4: Compile model
        result = step4_compile_model(compiler)
        
        # Step 5: Analyze results
        step5_analyze_results(result)
        
        print("\n" + "=" * 70)
        print("Demo completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
