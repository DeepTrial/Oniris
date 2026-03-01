#!/usr/bin/env python3
"""
Pattern File Management Example (YAML Version)

Demonstrates how to load and manage Patterns from YAML files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import oniris


def get_pattern_files_dir():
    """Get default pattern files directory."""
    return os.path.join(os.path.dirname(__file__), 'patterns')


def example_load_single_file():
    """Example 1: Load Pattern from single YAML file"""
    print("=" * 70)
    print("Example 1: Load Pattern from single YAML file")
    print("=" * 70)
    
    pm = oniris.PatternManager()
    
    # Load YAML file
    pattern_file = os.path.join(get_pattern_files_dir(), 'fusion_patterns.yaml')
    print(f"\n[1] Loading from file: {pattern_file}")
    
    count = oniris.import_yaml_patterns(pm, pattern_file)
    print(f"    Successfully loaded {count} Patterns")
    
    # Display loaded Patterns
    print(f"\n[2] Loaded Patterns:")
    for name in pm.get_pattern_names():
        pattern = pm.get_pattern(name)
        print(f"    - {name}: {pattern.metadata.description}")
        print(f"      Category: {pattern.metadata.category}, Priority: {pattern.metadata.priority}")
    
    # Display statistics
    print(f"\n[3] Pattern Statistics:")
    stats = pm.get_statistics()
    print(f"    Total: {stats.total_patterns}")
    print(f"    Enabled: {stats.enabled_patterns}")
    print(f"    Valid: {stats.valid_patterns}")


def example_load_multiple_files():
    """Example 2: Batch load Patterns from multiple files"""
    print("\n" + "=" * 70)
    print("Example 2: Batch load multiple Pattern files")
    print("=" * 70)
    
    pm = oniris.PatternManager()
    pattern_dir = get_pattern_files_dir()
    
    print(f"\n[1] Scanning directory: {pattern_dir}")
    
    # Use YamlPatternLoader
    loader = oniris.YamlPatternLoader(pm)
    total = loader.load_all(pattern_dir)
    
    print(f"\n[2] Total loaded: {total} Patterns")
    print(f"    Current Pattern total: {pm.get_pattern_count()}")
    
    # Display by category
    print(f"\n[3] By Category:")
    for category in [oniris.PatternCategory.FUSION, 
                     oniris.PatternCategory.OPTIMIZATION,
                     oniris.PatternCategory.CUSTOM]:
        patterns = pm.get_patterns_by_category(category)
        if patterns:
            cat_name = str(category).split('.')[-1]
            print(f"    {cat_name}: {len(patterns)}")
            for p in patterns:
                print(f"      - {p.metadata.name}")


def example_selective_loading():
    """Example 3: Selectively load Patterns of specific categories"""
    print("\n" + "=" * 70)
    print("Example 3: Selective Pattern loading")
    print("=" * 70)
    
    pm = oniris.PatternManager()
    
    # Only load fusion Patterns
    print("\n[1] Only load Fusion Patterns:")
    fusion_file = os.path.join(get_pattern_files_dir(), 'fusion_patterns.yaml')
    oniris.import_yaml_patterns(pm, fusion_file)
    
    # Disable some Patterns
    print("\n[2] Disable ConvBnRelu Pattern:")
    pm.set_pattern_enabled("ConvBnRelu", False)
    
    # Display enabled Patterns
    print(f"\n[3] Enabled Patterns ({pm.get_enabled_pattern_count()}/{pm.get_pattern_count()}):")
    for name in pm.get_pattern_names():
        if pm.is_pattern_enabled(name):
            print(f"    ✓ {name}")
        else:
            print(f"    ✗ {name} (disabled)")


def example_create_and_save():
    """Example 4: Create and save Pattern"""
    print("\n" + "=" * 70)
    print("Example 4: Create and Export Pattern")
    print("=" * 70)
    
    pm = oniris.PatternManager()
    
    # Register some Patterns
    print("\n[1] Create new Patterns:")
    pm.register_pattern(
        "MyConvRelu",
        "Conv(?, c0)\nRelu(c0, ?)",
        oniris.PatternCategory.FUSION,
        "My custom Conv+ReLU"
    )
    pm.register_pattern(
        "MyCustomPattern",
        "Conv(?, c0)\nSigmoid(c0, s0)\nMul([s0, c0], ?)",
        oniris.PatternCategory.CUSTOM,
        "Custom Swish"
    )
    
    # Set priority
    pm.set_pattern_priority("MyConvRelu", 100)
    pm.set_pattern_priority("MyCustomPattern", 50)
    
    print(f"    Created {pm.get_pattern_count()} Patterns")
    
    # Display info
    print(f"\n[2] Pattern Info:")
    for name in pm.get_pattern_names():
        p = pm.get_pattern(name)
        print(f"    - {name}")
        print(f"      Description: {p.metadata.description}")
        print(f"      Priority: {p.metadata.priority}")


def example_compiler_integration():
    """Example 5: Integration with compiler"""
    print("\n" + "=" * 70)
    print("Example 5: Pattern + Compiler Integration")
    print("=" * 70)
    
    # Create Pattern Manager and load files
    print("\n[1] Load Pattern from YAML:")
    pm = oniris.PatternManager()
    
    pattern_files = ['fusion_patterns.yaml', 'custom_patterns.yaml']
    for filename in pattern_files:
        filepath = os.path.join(get_pattern_files_dir(), filename)
        count = oniris.import_yaml_patterns(pm, filepath)
        print(f"    {filename}: {count} patterns")
    
    print(f"\n    Total: {pm.get_pattern_count()} patterns")
    
    # Create compiler
    print("\n[2] Create ModelCompiler:")
    compiler = pm.create_compiler()
    print(f"    Compiler contains {compiler.get_pattern_count()} Patterns")
    
    # Display Pattern names
    print(f"\n[3] Patterns used by compiler:")
    for name in compiler.get_pattern_names():
        print(f"    - {name}")


def main():
    """Run all examples"""
    examples = [
        example_load_single_file,
        example_load_multiple_files,
        example_selective_loading,
        example_create_and_save,
        example_compiler_integration,
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
