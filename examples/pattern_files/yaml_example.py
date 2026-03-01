#!/usr/bin/env python3
"""
YAML Pattern File Usage Example

Demonstrates how to use YAML format pattern files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import oniris


def example_basic_yaml_loading():
    """Example 1: Basic YAML file loading"""
    print("=" * 70)
    print("Example 1: Load Patterns from YAML file")
    print("=" * 70)
    
    # Create Pattern Manager
    pm = oniris.PatternManager()
    
    # Load YAML file
    yaml_file = os.path.join(os.path.dirname(__file__), 'patterns', 'fusion_patterns.yaml')
    print(f"\n[1] Loading YAML file: {yaml_file}")
    
    try:
        patterns = oniris.load_yaml_patterns(yaml_file)
        print(f"    Successfully loaded {len(patterns)} patterns")
        
        # Register to Pattern Manager
        for p in patterns:
            pm.register_pattern(p)
        
        print(f"\n[2] Patterns in Pattern Manager:")
        for name in pm.get_pattern_names():
            pattern = pm.get_pattern(name)
            print(f"    - {name}")
            print(f"      Description: {pattern.metadata.description}")
            print(f"      Priority: {pattern.metadata.priority}")
            
    except Exception as e:
        print(f"    Error: {e}")


def example_yaml_loader():
    """Example 2: Using YamlPatternLoader"""
    print("\n" + "=" * 70)
    print("Example 2: Using YamlPatternLoader")
    print("=" * 70)
    
    # Use YamlPatternLoader
    loader = oniris.YamlPatternLoader()
    
    pattern_dir = os.path.join(os.path.dirname(__file__), 'patterns')
    print(f"\n[1] Loading all YAML files from directory: {pattern_dir}")
    
    total = loader.load_all(pattern_dir)
    print(f"\n[2] Total loaded {total} patterns")
    
    pm = loader.get_manager()
    print(f"\n[3] Pattern Manager statistics:")
    print(f"    Total: {pm.get_pattern_count()}")
    
    # Display by category
    for cat in [oniris.PatternCategory.FUSION, 
                oniris.PatternCategory.OPTIMIZATION,
                oniris.PatternCategory.CUSTOM]:
        patterns = pm.get_patterns_by_category(cat)
        if patterns:
            print(f"\n    {cat}: {len(patterns)}")
            for p in patterns:
                print(f"      - {p.metadata.name}")


def example_yaml_to_compiler():
    """Example 3: YAML Patterns + Compiler"""
    print("\n" + "=" * 70)
    print("Example 3: YAML Patterns + Compiler")
    print("=" * 70)
    
    # Create Pattern Manager
    pm = oniris.PatternManager()
    
    # Import from YAML
    yaml_file = os.path.join(os.path.dirname(__file__), 'patterns', 'fusion_patterns.yaml')
    count = oniris.import_yaml_patterns(pm, yaml_file)
    print(f"\n[1] Imported {count} patterns from YAML")
    
    # Create compiler
    compiler = pm.create_compiler()
    print(f"[2] Created compiler with {compiler.get_pattern_count()} patterns")
    
    # Create simple model
    print("\n[3] Creating test model...")
    model = oniris.Model()
    graph = model.create_graph("test")
    
    input_info = oniris.ValueInfo()
    input_info.name = "input"
    input_info.shape = oniris.Shape([1, 3, 32, 32])
    input_info.dtype = oniris.DataType.FLOAT
    graph.add_input(input_info)
    
    conv = graph.create_node("Conv", "conv1")
    conv.add_input("input")
    conv.add_input("weight")
    conv.add_output("conv_out")
    
    relu = graph.create_node("Relu", "relu1")
    relu.add_input("conv_out")
    relu.add_output("output")
    
    output_info = oniris.ValueInfo()
    output_info.name = "output"
    graph.add_output(output_info)
    
    # Compile
    print("\n[4] Compiling model...")
    options = oniris.CompilerOptions()
    options.pattern_match_before_opt = True
    
    result = compiler.compile_model(model, options)
    print(f"    Pattern matches: {result.pattern_matching_summary.total_matches}")
    
    if result.pattern_matching_summary.total_matches > 0:
        for name, matches in result.pattern_matching_summary.pattern_results.items():
            print(f"    - {name}: {len(matches)} matches")


def example_yaml_format():
    """Example 4: Show YAML format"""
    print("\n" + "=" * 70)
    print("Example 4: YAML Format Description")
    print("=" * 70)
    
    yaml_content = '''
# YAML Pattern file example
patterns:
  - name: ConvRelu           # Pattern name (required)
    pattern: |               # Pattern definition (required), use | for multi-line string
      Conv(?, c0)
      Relu(c0, ?)
    desc: Conv + ReLU fusion  # Description (optional)
    cat: fusion              # Category (optional): fusion/optimization/quantization/custom
    priority: 100            # Priority (optional): higher number = higher priority

  - name: ConvBn
    pattern: |
      Conv(?, c0)
      BatchNormalization(c0, ?)
    desc: Conv + BN fusion
    cat: fusion
    priority: 90
'''
    
    print("\nYAML format features:")
    print("  1. Use indentation to represent hierarchy")
    print("  2. Use | for multi-line strings (pattern definition)")
    print("  3. Simplified field names: name, pattern, desc, cat, priority")
    print("  4. No need for author, version and other metadata")
    print("  5. Supports comments (starting with #)")
    
    print("\nExample content:")
    print(yaml_content)


def main():
    """Run all examples"""
    examples = [
        example_basic_yaml_loading,
        example_yaml_loader,
        example_yaml_to_compiler,
        example_yaml_format,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
