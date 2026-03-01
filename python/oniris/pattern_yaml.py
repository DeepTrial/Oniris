"""
YAML Pattern File Support

Load patterns from YAML files with simplified format.

Example YAML format:
    patterns:
      - name: ConvRelu
        pattern: |
          Conv(?, c0)
          Relu(c0, ?)
        desc: Conv + ReLU fusion
        cat: fusion
        priority: 100
"""

from typing import List, Optional
import os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from . import PatternManager, ManagedPattern, PatternCategory


def load_yaml_patterns(filepath: str) -> List[ManagedPattern]:
    """
    Load patterns from a YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        List of ManagedPattern objects
        
    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If file doesn't exist
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    patterns = []
    for p in data.get('patterns', []):
        pattern = _convert_to_managed_pattern(p)
        if pattern:
            patterns.append(pattern)
    
    return patterns


def import_yaml_patterns(pm: PatternManager, filepath: str) -> int:
    """
    Import YAML patterns into a PatternManager.
    
    This is the main API for loading patterns from YAML files.
    
    Args:
        pm: PatternManager instance
        filepath: Path to YAML file
        
    Returns:
        Number of patterns imported
    """
    patterns = load_yaml_patterns(filepath)
    
    count = 0
    for p in patterns:
        if pm.register_pattern(p):
            count += 1
    
    return count


def _convert_to_managed_pattern(data: dict) -> Optional[ManagedPattern]:
    """Convert YAML dict to ManagedPattern."""
    name = data.get('name')
    pattern_str = data.get('pattern')
    
    if not name or not pattern_str:
        return None
    
    cat_str = data.get('cat', 'custom')
    category = _parse_category(cat_str)
    
    mp = ManagedPattern(name, pattern_str.strip(), category)
    mp.metadata.description = data.get('desc', '')
    mp.metadata.priority = data.get('priority', 0)
    mp.metadata.enabled = data.get('enabled', True)
    
    return mp


def _parse_category(cat_str: str) -> PatternCategory:
    """Parse category string to enum."""
    mapping = {
        'fusion': PatternCategory.FUSION,
        'optimization': PatternCategory.OPTIMIZATION,
        'quantization': PatternCategory.QUANTIZATION,
        'custom': PatternCategory.CUSTOM,
        'analysis': PatternCategory.ANALYSIS,
    }
    return mapping.get(cat_str.lower(), PatternCategory.CUSTOM)


# Deprecated: Use import_yaml_patterns instead
class YamlPatternLoader:
    """
    Helper class to load YAML patterns into PatternManager.
    
    Deprecated: Use import_yaml_patterns() function instead.
    """
    
    def __init__(self, pattern_manager: Optional[PatternManager] = None):
        self.pm = pattern_manager or PatternManager()
    
    def load(self, filepath: str) -> int:
        """Load patterns from YAML file."""
        return import_yaml_patterns(self.pm, filepath)
    
    def load_all(self, directory: str) -> int:
        """Load all YAML files from a directory."""
        total = 0
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                filepath = os.path.join(directory, filename)
                count = self.load(filepath)
                total += count
                print(f"  {filename}: {count} patterns")
        
        return total
    
    def get_manager(self) -> PatternManager:
        """Get the PatternManager."""
        return self.pm
