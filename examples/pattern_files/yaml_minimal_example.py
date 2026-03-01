#!/usr/bin/env python3
"""
YAML Pattern Minimal Example
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import oniris

# 1. Load YAML Patterns
pm = oniris.PatternManager()
yaml_file = os.path.join(os.path.dirname(__file__), 'patterns', 'fusion_patterns.yaml')
count = oniris.import_yaml_patterns(pm, yaml_file)
print(f"Loaded {count} patterns from YAML")

# 2. Display loaded patterns
print("\nPatterns:")
for name in pm.get_pattern_names():
    p = pm.get_pattern(name)
    print(f"  - {name}: {p.metadata.description} (priority: {p.metadata.priority})")

# 3. Apply to compiler
compiler = pm.create_compiler()
print(f"\nCompiler has {compiler.get_pattern_count()} patterns")
