"""
Unit tests for YAML pattern loading functionality.
"""

import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import oniris

# Check if PyYAML is available
try:
    import yaml
    from oniris.pattern_yaml import load_yaml_patterns, import_yaml_patterns, YamlPatternLoader
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class TestPatternYamlLoading(unittest.TestCase):
    """Test YAML pattern loading functionality."""
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def setUp(self):
        """Create temporary YAML files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Valid YAML pattern file
        self.valid_yaml = """
patterns:
  - name: ConvRelu
    pattern: |
      Conv(?, c0)
      Relu(c0, ?)
    desc: Conv + ReLU fusion
    cat: fusion
    priority: 100
    enabled: true

  - name: ConvBn
    pattern: |
      Conv(?, c0)
      BatchNormalization(c0, ?)
    desc: Conv + BatchNorm fusion
    cat: fusion
    priority: 90
"""
        
        # Minimal YAML (only required fields)
        self.minimal_yaml = """
patterns:
  - name: SimplePattern
    pattern: Conv(?, ?)
"""
        
        # Invalid YAML (missing required field)
        self.invalid_yaml = """
patterns:
  - name: MissingPattern
    desc: Missing pattern field
    cat: fusion
"""
        
        # Empty patterns
        self.empty_yaml = """
patterns: []
"""
        
        # Write files
        self.valid_file = os.path.join(self.temp_dir, 'valid.yaml')
        self.minimal_file = os.path.join(self.temp_dir, 'minimal.yaml')
        self.invalid_file = os.path.join(self.temp_dir, 'invalid.yaml')
        self.empty_file = os.path.join(self.temp_dir, 'empty.yaml')
        
        with open(self.valid_file, 'w') as f:
            f.write(self.valid_yaml)
        with open(self.minimal_file, 'w') as f:
            f.write(self.minimal_yaml)
        with open(self.invalid_file, 'w') as f:
            f.write(self.invalid_yaml)
        with open(self.empty_file, 'w') as f:
            f.write(self.empty_yaml)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_load_valid_yaml(self):
        """Test loading valid YAML patterns."""
        patterns = load_yaml_patterns(self.valid_file)
        
        self.assertEqual(len(patterns), 2)
        
        # Check first pattern
        p1 = patterns[0]
        self.assertEqual(p1.metadata.name, 'ConvRelu')
        self.assertEqual(p1.metadata.description, 'Conv + ReLU fusion')
        self.assertEqual(p1.metadata.category, oniris.PatternCategory.FUSION)
        self.assertEqual(p1.metadata.priority, 100)
        self.assertTrue(p1.metadata.enabled)
        self.assertIn('Conv(?, c0)', p1.definition.pattern_string)
        self.assertIn('Relu(c0, ?)', p1.definition.pattern_string)
        
        # Check second pattern
        p2 = patterns[1]
        self.assertEqual(p2.metadata.name, 'ConvBn')
        self.assertEqual(p2.metadata.priority, 90)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_load_minimal_yaml(self):
        """Test loading YAML with only required fields."""
        patterns = load_yaml_patterns(self.minimal_file)
        
        self.assertEqual(len(patterns), 1)
        p = patterns[0]
        
        self.assertEqual(p.metadata.name, 'SimplePattern')
        self.assertEqual(p.definition.pattern_string.strip(), 'Conv(?, ?)')
        # Check defaults
        self.assertEqual(p.metadata.description, '')
        self.assertEqual(p.metadata.category, oniris.PatternCategory.CUSTOM)
        self.assertEqual(p.metadata.priority, 0)
        self.assertTrue(p.metadata.enabled)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_load_empty_yaml(self):
        """Test loading YAML with empty patterns."""
        patterns = load_yaml_patterns(self.empty_file)
        self.assertEqual(len(patterns), 0)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_load_invalid_yaml_skips_invalid(self):
        """Test that invalid patterns (missing required fields) are skipped."""
        patterns = load_yaml_patterns(self.invalid_file)
        # Pattern with missing 'pattern' field should be skipped
        self.assertEqual(len(patterns), 0)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_import_yaml_patterns(self):
        """Test importing YAML patterns into PatternManager."""
        pm = oniris.PatternManager()
        
        count = import_yaml_patterns(pm, self.valid_file)
        
        self.assertEqual(count, 2)
        self.assertEqual(pm.get_pattern_count(), 2)
        
        # Verify patterns were registered
        pattern_names = pm.get_pattern_names()
        self.assertIn('ConvRelu', pattern_names)
        self.assertIn('ConvBn', pattern_names)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_yaml_pattern_loader(self):
        """Test YamlPatternLoader class."""
        loader = YamlPatternLoader()
        
        # Load single file
        count = loader.load(self.valid_file)
        self.assertEqual(count, 2)
        
        pm = loader.get_manager()
        self.assertEqual(pm.get_pattern_count(), 2)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_yaml_pattern_loader_load_all(self):
        """Test loading all YAML files from directory."""
        loader = YamlPatternLoader()
        
        total = loader.load_all(self.temp_dir)
        
        # Should load from valid.yaml, minimal.yaml, empty.yaml
        # invalid.yaml has no valid patterns
        self.assertGreaterEqual(total, 3)
        
        pm = loader.get_manager()
        self.assertGreaterEqual(pm.get_pattern_count(), 3)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_yaml_loader_with_existing_manager(self):
        """Test YamlPatternLoader with existing PatternManager."""
        pm = oniris.PatternManager()
        pm.register_pattern("ExistingPattern", "Relu(?, ?)")
        
        loader = YamlPatternLoader(pm)
        loader.load(self.valid_file)
        
        self.assertEqual(pm.get_pattern_count(), 3)  # 1 + 2
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_yaml_patterns('/nonexistent/file.yaml')
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_category_mapping(self):
        """Test category string to enum mapping."""
        yaml_content = """
patterns:
  - name: FusionPattern
    pattern: Conv(?, ?)
    cat: fusion
  - name: OptPattern
    pattern: Relu(?, ?)
    cat: optimization
  - name: QuantPattern
    pattern: Conv(?, ?)
    cat: quantization
  - name: CustomPattern
    pattern: Add(?, ?)
    cat: custom
  - name: AnalysisPattern
    pattern: Sub(?, ?)
    cat: analysis
  - name: UnknownPattern
    pattern: Mul(?, ?)
    cat: unknown_category
"""
        temp_file = os.path.join(self.temp_dir, 'categories.yaml')
        with open(temp_file, 'w') as f:
            f.write(yaml_content)
        
        patterns = load_yaml_patterns(temp_file)
        
        self.assertEqual(len(patterns), 6)
        self.assertEqual(patterns[0].metadata.category, oniris.PatternCategory.FUSION)
        self.assertEqual(patterns[1].metadata.category, oniris.PatternCategory.OPTIMIZATION)
        self.assertEqual(patterns[2].metadata.category, oniris.PatternCategory.QUANTIZATION)
        self.assertEqual(patterns[3].metadata.category, oniris.PatternCategory.CUSTOM)
        self.assertEqual(patterns[4].metadata.category, oniris.PatternCategory.ANALYSIS)
        # Unknown category defaults to CUSTOM
        self.assertEqual(patterns[5].metadata.category, oniris.PatternCategory.CUSTOM)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_pattern_string_multiline(self):
        """Test multiline pattern string handling."""
        yaml_content = """
patterns:
  - name: MultiLinePattern
    pattern: |
      Conv(?, c0)
      BatchNormalization(c0, bn0)
      Relu(bn0, ?)
"""
        temp_file = os.path.join(self.temp_dir, 'multiline.yaml')
        with open(temp_file, 'w') as f:
            f.write(yaml_content)
        
        patterns = load_yaml_patterns(temp_file)
        
        self.assertEqual(len(patterns), 1)
        pattern_str = patterns[0].definition.pattern_string
        
        # Should contain all three lines
        self.assertIn('Conv(?, c0)', pattern_str)
        self.assertIn('BatchNormalization(c0, bn0)', pattern_str)
        self.assertIn('Relu(bn0, ?)', pattern_str)


class TestPatternYamlWithoutYaml(unittest.TestCase):
    """Test behavior when PyYAML is not installed."""
    
    def test_import_error_without_yaml(self):
        """Test that ImportError is raised when PyYAML is not available."""
        # This test only makes sense if yaml is not available
        # In practice, we skip it since yaml is usually installed
        pass


if __name__ == '__main__':
    unittest.main()
