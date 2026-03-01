#!/usr/bin/env python3
"""
Complete Workflow Example

This example demonstrates the complete workflow:
1. Setup PatternManager with custom patterns
2. Load and organize patterns by category
3. Export pattern collection for team sharing
4. Use patterns with ModelCompiler
5. Compile model and analyze results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import oniris


def step1_setup_pattern_library():
    """Step 1: Setup personal pattern library."""
    print("=" * 70)
    print("Step 1: Setup Pattern Library")
    print("=" * 70)
    
    # Create pattern manager
    pm = oniris.PatternManager()
    
    # Import built-in patterns
    print("\n[1.1] Importing built-in patterns...")
    pm.import_patterns(oniris.get_fusion_patterns())
    pm.import_patterns(oniris.get_optimization_patterns())
    print(f"      Imported {pm.get_pattern_count()} patterns")
    
    # Register custom patterns
    print("\n[1.2] Registering custom patterns...")
    pm.register_pattern(
        "CustomSwish",
        "Conv(?, c0)\nSigmoid(c0, s0)\nMul([s0, c0], ?)",
        oniris.PatternCategory.CUSTOM,
        "Custom Swish activation"
    )
    pm.register_pattern(
        "CustomResidual",
        "Conv(?, c0)\nAdd([c0, ?], ?)",
        oniris.PatternCategory.CUSTOM,
        "Residual connection"
    )
    print(f"      Total patterns: {pm.get_pattern_count()}")
    
    # Set priorities
    print("\n[1.3] Setting pattern priorities...")
    pm.set_pattern_priority("ConvRelu", 100)  # High priority
    pm.set_pattern_priority("CustomSwish", 50)  # Medium priority
    
    return pm


def step2_organize_patterns(pm):
    """Step 2: Organize patterns by category."""
    print("\n" + "=" * 70)
    print("Step 2: Organize Patterns")
    print("=" * 70)
    
    # Get statistics
    stats = pm.get_statistics()
    print(f"\n[2.1] Pattern Statistics:")
    print(f"      Total: {stats.total_patterns}")
    print(f"      Enabled: {stats.enabled_patterns}")
    print(f"      Valid: {stats.valid_patterns}")
    
    print(f"\n[2.2] By Category:")
    for cat, count in stats.category_counts.items():
        print(f"      {cat}: {count}")
    
    # Disable optimization patterns (we only want fusion for this demo)
    print("\n[2.3] Disabling OPTIMIZATION category...")
    pm.set_category_enabled(oniris.PatternCategory.OPTIMIZATION, False)
    print(f"      Enabled patterns: {pm.get_enabled_pattern_count()}")


def step3_export_patterns(pm):
    """Step 3: Export pattern collection."""
    print("\n" + "=" * 70)
    print("Step 3: Export Pattern Collection")
    print("=" * 70)
    
    # Export to JSON
    print("\n[3.1] Exporting to JSON...")
    json_str = pm.export_to_json(pretty=True)
    print(f"      JSON size: {len(json_str)} characters")
    
    # Save to file
    output_path = "/tmp/my_pattern_collection.json"
    pm.export_to_file(output_path, pretty=True)
    print(f"      Saved to: {output_path}")
    
    # Show preview
    print("\n[3.2] JSON Preview:")
    data = json.loads(json_str)
    preview = {
        "name": data.get("name", "patterns"),
        "pattern_count": len(data.get("patterns", [])),
        "categories": list(set(p.get("category", "unknown") 
                              for p in data.get("patterns", [])))
    }
    print(json.dumps(preview, indent=4))
    
    return output_path


def step4_import_and_use(json_path):
    """Step 4: Import patterns and use with compiler."""
    print("\n" + "=" * 70)
    print("Step 4: Import and Use Patterns")
    print("=" * 70)
    
    # Create new pattern manager and import
    print("\n[4.1] Creating new PatternManager and importing...")
    pm = oniris.PatternManager()
    count = pm.import_patterns_from_file(json_path)
    print(f"      Imported {count} patterns")
    print(f"      Current patterns: {pm.get_pattern_names()}")
    
    # Create compiler with patterns
    print("\n[4.2] Creating ModelCompiler with patterns...")
    compiler = pm.create_compiler()
    print(f"      Compiler has {compiler.get_pattern_count()} patterns")
    
    return compiler


def step5_compile_with_patterns(compiler):
    """Step 5: Compile a model with the patterns."""
    print("\n" + "=" * 70)
    print("Step 5: Compile Model")
    print("=" * 70)
    
    # Create a simple test model
    print("\n[5.1] Creating test model...")
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
    conv.add_output("conv_out")
    conv.set_attribute_ints("kernel_shape", [3, 3])
    
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    print("      Created Conv+ReLU model")
    
    # Configure options
    print("\n[5.2] Configuring compilation options...")
    options = oniris.CompilerOptions()
    options.verbose = False
    options.enable_optimization = True
    options.enable_shape_inference = True
    options.enable_pattern_matching = True
    options.pattern_match_before_opt = True  # Match before fusion
    
    # Compile
    print("\n[5.3] Compiling model...")
    result = compiler.compile_model(model, options)
    
    print(f"      Compilation success: {result.success}")
    print(f"      Duration: {result.duration_ms:.2f} ms")
    print(f"      Optimization changes: {result.optimization_stats.num_changes}")
    print(f"      Pattern matches: {result.pattern_matching_summary.total_matches}")
    
    # Show pattern matching results
    if result.pattern_matching_summary.total_matches > 0:
        print("\n[5.4] Pattern Matches:")
        for pattern_name, matches in result.pattern_matching_summary.pattern_results.items():
            print(f"      {pattern_name}: {len(matches)} match(es)")
            for match in matches:
                print(f"        Match {match.match_id}:")
                for node in match.nodes:
                    print(f"          - {node.node_name} ({node.op_type})")
    
    return result


def step6_analyze_results(result):
    """Step 6: Analyze compilation results."""
    print("\n" + "=" * 70)
    print("Step 6: Analyze Results")
    print("=" * 70)
    
    # Export to JSON
    print("\n[6.1] Exporting results to JSON...")
    json_str = result.to_json(pretty=True)
    print(f"      Result JSON size: {len(json_str)} characters")
    
    # Parse and analyze
    print("\n[6.2] Result Summary:")
    data = json.loads(json_str)
    summary = {
        "success": data.get("success"),
        "model_info": {
            "num_nodes": data.get("model_info", {}).get("num_nodes"),
            "op_types": data.get("model_info", {}).get("op_types_used", [])
        },
        "optimization": {
            "num_changes": data.get("optimization", {}).get("num_changes")
        },
        "pattern_matching": {
            "total_matches": data.get("pattern_matching", {}).get("total_matches"),
            "patterns_with_matches": data.get("pattern_matching", {}).get("patterns_with_matches")
        }
    }
    print(json.dumps(summary, indent=4))


def main():
    """Run complete workflow."""
    print("\n" + "=" * 70)
    print("Complete Workflow: PatternManager + Compiler")
    print("=" * 70)
    
    try:
        # Step 1: Setup pattern library
        pm = step1_setup_pattern_library()
        
        # Step 2: Organize patterns
        step2_organize_patterns(pm)
        
        # Step 3: Export patterns
        json_path = step3_export_patterns(pm)
        
        # Step 4: Import and create compiler
        compiler = step4_import_and_use(json_path)
        
        # Step 5: Compile model
        result = step5_compile_with_patterns(compiler)
        
        # Step 6: Analyze results
        step6_analyze_results(result)
        
        # Cleanup
        try:
            os.remove(json_path)
        except:
            pass
        
        print("\n" + "=" * 70)
        print("Complete workflow finished successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
