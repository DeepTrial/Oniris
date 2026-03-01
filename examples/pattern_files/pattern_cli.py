#!/usr/bin/env python3
"""
Pattern File Management Tool (YAML Version)

Provides command line interface to manage Pattern files:
- List Pattern files
- View Pattern content
- Validate Pattern files
- Import Pattern to compiler
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import oniris


def get_patterns_dir():
    """Get default pattern files directory."""
    return os.path.join(os.path.dirname(__file__), 'patterns')


def list_patterns(args):
    """List all Pattern files."""
    pattern_dir = args.dir or get_patterns_dir()
    
    if not os.path.exists(pattern_dir):
        print(f"Error: Directory does not exist: {pattern_dir}")
        return 1
    
    print(f"\nPattern files directory: {pattern_dir}\n")
    print("-" * 70)
    
    for filename in sorted(os.listdir(pattern_dir)):
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join(pattern_dir, filename)
            try:
                patterns = oniris.load_yaml_patterns(filepath)
                
                # Count categories
                categories = {}
                for p in patterns:
                    cat = str(p.metadata.category).split('.')[-1].lower()
                    categories[cat] = categories.get(cat, 0) + 1
                
                print(f"\nFile: {filename}")
                print(f"  Pattern count: {len(patterns)}")
                if categories:
                    print(f"  Category distribution: {categories}")
                    
            except Exception as e:
                print(f"\nFile: {filename}")
                print(f"  Error: {e}")
    
    print("\n" + "-" * 70)
    return 0


def show_patterns(args):
    """Show detailed content of Pattern file."""
    pattern_dir = args.dir or get_patterns_dir()
    
    # Auto-append .yaml extension
    filepath = os.path.join(pattern_dir, args.file)
    if not os.path.exists(filepath):
        filepath = os.path.join(pattern_dir, args.file + '.yaml')
    
    if not os.path.exists(filepath):
        print(f"Error: File does not exist: {args.file}")
        return 1
    
    try:
        patterns = oniris.load_yaml_patterns(filepath)
        
        print(f"\n{'='*70}")
        print(f"Pattern file: {os.path.basename(filepath)}")
        print(f"{'='*70}")
        
        print(f"\nPatterns ({len(patterns)}):")
        print("-" * 70)
        
        for i, p in enumerate(patterns, 1):
            print(f"\n  [{i}] {p.metadata.name}")
            if p.metadata.description:
                print(f"      Description: {p.metadata.description}")
            print(f"      Category: {p.metadata.category}")
            print(f"      Priority: {p.metadata.priority}")
            
            pattern_str = p.definition.pattern_string
            print(f"      Pattern definition:")
            for line in pattern_str.split('\n'):
                if line.strip():
                    print(f"        {line}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"Error: Cannot parse file: {e}")
        return 1
    
    return 0


def validate_patterns(args):
    """Validate Pattern files."""
    pattern_dir = args.dir or get_patterns_dir()
    
    files_to_validate = []
    if args.file:
        filepath = os.path.join(pattern_dir, args.file)
        if not os.path.exists(filepath):
            filepath = os.path.join(pattern_dir, args.file + '.yaml')
        
        if os.path.exists(filepath):
            files_to_validate.append((args.file, filepath))
        else:
            print(f"Error: File does not exist: {args.file}")
            return 1
    else:
        # Validate all files
        for filename in os.listdir(pattern_dir):
            if filename.endswith(('.yaml', '.yml')):
                filepath = os.path.join(pattern_dir, filename)
                files_to_validate.append((filename, filepath))
    
    print(f"\nValidating {len(files_to_validate)} Pattern files...\n")
    print("-" * 70)
    
    total_patterns = 0
    valid_patterns = 0
    invalid_patterns = 0
    
    for filename, filepath in files_to_validate:
        print(f"\nFile: {filename}")
        
        try:
            patterns = oniris.load_yaml_patterns(filepath)
            print(f"  Contains {len(patterns)} patterns")
            
            for p in patterns:
                total_patterns += 1
                name = p.metadata.name or 'Unnamed'
                
                if p.is_valid():
                    print(f"    ✓ {name}")
                    valid_patterns += 1
                else:
                    print(f"    ✗ {name}: Invalid Pattern syntax")
                    invalid_patterns += 1
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "-" * 70)
    print(f"\nValidation results:")
    print(f"  Total Patterns: {total_patterns}")
    print(f"  Valid: {valid_patterns}")
    print(f"  Invalid: {invalid_patterns}")
    
    return 0 if invalid_patterns == 0 else 1


def test_patterns(args):
    """Test Pattern file (load into PatternManager)."""
    pattern_dir = args.dir or get_patterns_dir()
    
    filepath = os.path.join(pattern_dir, args.file)
    if not os.path.exists(filepath):
        filepath = os.path.join(pattern_dir, args.file + '.yaml')
    
    if not os.path.exists(filepath):
        print(f"Error: File does not exist: {args.file}")
        return 1
    
    print(f"\nTesting load Pattern file: {os.path.basename(filepath)}\n")
    print("-" * 70)
    
    pm = oniris.PatternManager()
    
    try:
        count = oniris.import_yaml_patterns(pm, filepath)
        print(f"\n✓ Successfully loaded {count} patterns")
        
        if count > 0:
            print(f"\nLoaded patterns:")
            for name in pm.get_pattern_names():
                pattern = pm.get_pattern(name)
                print(f"  - {name}")
                print(f"    Category: {pattern.metadata.category}")
                print(f"    Priority: {pattern.metadata.priority}")
                
                # Validate pattern syntax
                if pattern.is_valid():
                    print(f"    Syntax: ✓ Valid")
                else:
                    print(f"    Syntax: ✗ Invalid")
        
        # Display statistics
        stats = pm.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total: {stats.total_patterns}")
        print(f"  Enabled: {stats.enabled_patterns}")
        print(f"  Valid: {stats.valid_patterns}")
        
    except Exception as e:
        print(f"\n✗ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "-" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Pattern File Management Tool (YAML)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s list                          # List all pattern files
  %(prog)s show fusion                   # Show fusion.yaml content
  %(prog)s validate                      # Validate all files
  %(prog)s test fusion                   # Test loading fusion.yaml
        '''
    )
    
    parser.add_argument('-d', '--dir', help='Pattern files directory (default: patterns/)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List all Pattern files')
    list_parser.set_defaults(func=list_patterns)
    
    # show command
    show_parser = subparsers.add_parser('show', help='Show Pattern file content')
    show_parser.add_argument('file', help='Pattern filename (without .yaml extension)')
    show_parser.set_defaults(func=show_patterns)
    
    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate Pattern files')
    validate_parser.add_argument('file', nargs='?', help='Specific file (default: all)')
    validate_parser.set_defaults(func=validate_patterns)
    
    # test command
    test_parser = subparsers.add_parser('test', help='Test loading Pattern file')
    test_parser.add_argument('file', help='Pattern filename (without .yaml extension)')
    test_parser.set_defaults(func=test_patterns)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
