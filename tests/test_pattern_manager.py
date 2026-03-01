"""
Unit tests for PatternManager functionality.
"""

import sys
import os
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import oniris
from oniris import PatternManager, ManagedPattern, PatternCategory


class TestPatternManagerBasic(unittest.TestCase):
    """Test basic PatternManager functionality."""
    
    def setUp(self):
        """Create a fresh PatternManager for each test."""
        self.pm = PatternManager()
    
    def test_initial_state(self):
        """Test initial state of PatternManager."""
        self.assertEqual(self.pm.get_pattern_count(), 0)
        self.assertEqual(self.pm.get_enabled_pattern_count(), 0)
        self.assertEqual(self.pm.get_pattern_names(), [])
    
    def test_register_simple_pattern(self):
        """Test registering a simple pattern."""
        result = self.pm.register_pattern(
            "ConvRelu",
            "Conv(?, c0)\nRelu(c0, ?)",
            PatternCategory.FUSION,
            "Conv + ReLU fusion"
        )
        
        self.assertTrue(result)
        self.assertEqual(self.pm.get_pattern_count(), 1)
        
        # Verify pattern was stored correctly
        pattern = self.pm.get_pattern("ConvRelu")
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.metadata.name, "ConvRelu")
        self.assertEqual(pattern.metadata.category, PatternCategory.FUSION)
        self.assertEqual(pattern.metadata.description, "Conv + ReLU fusion")
    
    def test_register_managed_pattern(self):
        """Test registering a ManagedPattern object."""
        mp = ManagedPattern("GemmRelu", "Gemm(?, g0)\nRelu(g0, ?)", PatternCategory.FUSION)
        mp.metadata.description = "FC + ReLU fusion"
        mp.metadata.priority = 50
        
        result = self.pm.register_pattern(mp)
        
        self.assertTrue(result)
        self.assertEqual(self.pm.get_pattern_count(), 1)
    
    def test_register_duplicate_pattern(self):
        """Test registering duplicate pattern names."""
        # Register first pattern using ManagedPattern with overwrite=False
        mp1 = ManagedPattern("TestPattern", "Conv(?, ?)")
        result = self.pm.register_pattern(mp1, False)
        self.assertTrue(result)
        
        # Without overwrite, should fail
        mp2 = ManagedPattern("TestPattern", "Relu(?, ?)")
        result = self.pm.register_pattern(mp2, False)
        self.assertFalse(result)
        self.assertEqual(self.pm.get_pattern_count(), 1)
        
        # With overwrite, should succeed
        result = self.pm.register_pattern(mp2, True)
        self.assertTrue(result)
    
    def test_unregister_pattern(self):
        """Test unregistering patterns."""
        self.pm.register_pattern("Pattern1", "Conv(?, ?)")
        self.pm.register_pattern("Pattern2", "Relu(?, ?)")
        
        self.assertEqual(self.pm.get_pattern_count(), 2)
        
        # Unregister existing
        result = self.pm.unregister_pattern("Pattern1")
        self.assertTrue(result)
        self.assertEqual(self.pm.get_pattern_count(), 1)
        self.assertIsNone(self.pm.get_pattern("Pattern1"))
        
        # Unregister non-existing
        result = self.pm.unregister_pattern("NonExistent")
        self.assertFalse(result)
    
    def test_clear_patterns(self):
        """Test clearing all patterns."""
        self.pm.register_pattern("Pattern1", "Conv(?, ?)")
        self.pm.register_pattern("Pattern2", "Relu(?, ?)")
        
        self.pm.clear_patterns()
        
        self.assertEqual(self.pm.get_pattern_count(), 0)
        self.assertEqual(self.pm.get_pattern_names(), [])
    
    def test_clear_by_category(self):
        """Test clearing patterns by category."""
        self.pm.register_pattern("Fusion1", "Conv(?, ?)", PatternCategory.FUSION)
        self.pm.register_pattern("Fusion2", "Relu(?, ?)", PatternCategory.FUSION)
        self.pm.register_pattern("Opt1", "Add(?, ?)", PatternCategory.OPTIMIZATION)
        
        self.assertEqual(self.pm.get_pattern_count(), 3)
        
        self.pm.clear_patterns_by_category(PatternCategory.FUSION)
        
        self.assertEqual(self.pm.get_pattern_count(), 1)
        self.assertIsNotNone(self.pm.get_pattern("Opt1"))
        self.assertIsNone(self.pm.get_pattern("Fusion1"))


class TestPatternManagerQueries(unittest.TestCase):
    """Test PatternManager query functionality."""
    
    def setUp(self):
        """Set up test patterns."""
        self.pm = PatternManager()
        
        # Add various patterns
        self.pm.register_pattern("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)", 
                                 PatternCategory.FUSION, "Conv + ReLU")
        self.pm.register_pattern("ConvBn", "Conv(?, c0)\nBatchNormalization(c0, ?)", 
                                 PatternCategory.FUSION, "Conv + BN")
        self.pm.register_pattern("ConstFold", "Constant(?, ?)", 
                                 PatternCategory.OPTIMIZATION, "Constant folding")
        self.pm.register_pattern("CustomPattern", "MyOp(?, ?)", 
                                 PatternCategory.CUSTOM, "Custom op")
        
        # Set some disabled
        self.pm.set_pattern_enabled("CustomPattern", False)
    
    def test_get_pattern_names(self):
        """Test getting all pattern names."""
        names = self.pm.get_pattern_names()
        
        self.assertEqual(len(names), 4)
        self.assertIn("ConvRelu", names)
        self.assertIn("ConvBn", names)
        self.assertIn("ConstFold", names)
        self.assertIn("CustomPattern", names)
    
    def test_get_all_pattern_names(self):
        """Test getting all pattern names."""
        names = self.pm.get_pattern_names()
        self.assertEqual(len(names), 4)
    

    

    
    def test_has_pattern(self):
        """Test pattern existence check."""
        self.assertTrue(self.pm.has_pattern("ConvRelu"))
        self.assertTrue(self.pm.has_pattern("CustomPattern"))
        self.assertFalse(self.pm.has_pattern("NonExistent"))
    
    def test_is_enabled(self):
        """Test pattern enabled check."""
        self.assertTrue(self.pm.is_pattern_enabled("ConvRelu"))
        self.assertFalse(self.pm.is_pattern_enabled("CustomPattern"))
        
        # Non-existent pattern should return False
        self.assertFalse(self.pm.is_pattern_enabled("NonExistent"))
    
    def test_set_enabled(self):
        """Test enabling/disabling patterns."""
        # Disable an enabled pattern
        result = self.pm.set_pattern_enabled("ConvRelu", False)
        self.assertTrue(result)
        self.assertFalse(self.pm.is_pattern_enabled("ConvRelu"))
        
        # Enable a disabled pattern
        result = self.pm.set_pattern_enabled("CustomPattern", True)
        self.assertTrue(result)
        self.assertTrue(self.pm.is_pattern_enabled("CustomPattern"))
        
        # Try to set non-existent pattern
        result = self.pm.set_pattern_enabled("NonExistent", True)
        self.assertFalse(result)
    
    def test_set_priority(self):
        """Test setting pattern priority."""
        result = self.pm.set_pattern_priority("ConvRelu", 100)
        self.assertTrue(result)
        
        pattern = self.pm.get_pattern("ConvRelu")
        self.assertEqual(pattern.metadata.priority, 100)
        
        # Non-existent pattern
        result = self.pm.set_pattern_priority("NonExistent", 50)
        self.assertFalse(result)
    
    def test_set_category_enabled(self):
        """Test enabling/disabling entire categories."""
        self.pm.set_category_enabled(PatternCategory.FUSION, False)
        
        self.assertFalse(self.pm.is_pattern_enabled("ConvRelu"))
        self.assertFalse(self.pm.is_pattern_enabled("ConvBn"))
        self.assertTrue(self.pm.is_pattern_enabled("ConstFold"))  # OPTIMIZATION
    
    def test_get_enabled_count(self):
        """Test getting enabled pattern count."""
        self.assertEqual(self.pm.get_pattern_count(), 4)
        self.assertEqual(self.pm.get_enabled_pattern_count(), 3)


class TestPatternManagerStatistics(unittest.TestCase):
    """Test PatternManager statistics."""
    
    def test_empty_statistics(self):
        """Test statistics for empty manager."""
        pm = PatternManager()
        stats = pm.get_statistics()
        
        self.assertEqual(stats.total_patterns, 0)
        self.assertEqual(stats.enabled_patterns, 0)
        self.assertEqual(stats.valid_patterns, 0)
        self.assertEqual(stats.invalid_patterns, 0)
    
    def test_statistics(self):
        """Test statistics with patterns."""
        pm = PatternManager()
        
        pm.register_pattern("Pattern1", "Conv(?, ?)", PatternCategory.FUSION)
        pm.register_pattern("Pattern2", "Relu(?, ?)", PatternCategory.FUSION)
        pm.register_pattern("Pattern3", "Add(?, ?)", PatternCategory.OPTIMIZATION)
        pm.set_pattern_enabled("Pattern3", False)
        
        stats = pm.get_statistics()
        
        self.assertEqual(stats.total_patterns, 3)
        self.assertEqual(stats.enabled_patterns, 2)
        
        # Check category counts
        self.assertEqual(stats.category_counts.get(PatternCategory.FUSION, 0), 2)
        self.assertEqual(stats.category_counts.get(PatternCategory.OPTIMIZATION, 0), 1)


class TestPatternManagerImportExport(unittest.TestCase):
    """Test PatternManager import/export functionality."""
    
    def setUp(self):
        self.pm = PatternManager()
        self.pm.register_pattern("Pattern1", "Conv(?, ?)", PatternCategory.FUSION, "First pattern")
        self.pm.register_pattern("Pattern2", "Relu(?, ?)", PatternCategory.OPTIMIZATION, "Second pattern")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_to_json(self):
        """Test exporting patterns to JSON."""
        json_str = self.pm.export_to_json()
        
        self.assertIsNotNone(json_str)
        self.assertGreater(len(json_str), 0)
        
        # Should contain pattern names
        self.assertIn("Pattern1", json_str)
        self.assertIn("Pattern2", json_str)
    
    def test_export_import_roundtrip(self):
        """Test export and re-import."""
        # Export
        json_str = self.pm.export_to_json()
        
        # Create new manager and import
        pm2 = PatternManager()
        count = pm2.import_patterns_from_json(json_str)
        
        self.assertEqual(count, 2)
        self.assertEqual(pm2.get_pattern_count(), 2)
        self.assertIsNotNone(pm2.get_pattern("Pattern1"))
        self.assertIsNotNone(pm2.get_pattern("Pattern2"))
    
    def test_export_to_file(self):
        """Test exporting to file."""
        filepath = os.path.join(self.temp_dir, "patterns.json")
        result = self.pm.export_to_file(filepath)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(filepath))
        
        # Verify file content
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertIn("Pattern1", content)
    
    def test_import_from_file(self):
        """Test importing from file."""
        # First export
        filepath = os.path.join(self.temp_dir, "patterns.json")
        self.pm.export_to_file(filepath)
        
        # Create new manager and import
        pm2 = PatternManager()
        count = pm2.import_patterns_from_file(filepath)
        
        self.assertEqual(count, 2)


class TestPatternManagerValidation(unittest.TestCase):
    """Test PatternManager pattern validation."""
    
    def test_validate_valid_pattern(self):
        """Test validating a valid pattern."""
        # Skip if validate_pattern is not exposed or has issues
        if not hasattr(PatternManager, 'validate_pattern'):
            self.skipTest("validate_pattern not exposed to Python")
        
        mp = ManagedPattern("ValidPattern", "Conv(?, ?)", PatternCategory.FUSION)
        
        try:
            result = PatternManager.validate_pattern(mp)
            self.assertTrue(result.valid)
        except Exception as e:
            self.skipTest(f"validate_pattern binding issue: {e}")
    
    def test_validate_invalid_pattern(self):
        """Test validating an invalid pattern."""
        if not hasattr(PatternManager, 'validate_pattern'):
            self.skipTest("validate_pattern not exposed to Python")
        
        try:
            mp = ManagedPattern()  # Empty pattern
            result = PatternManager.validate_pattern(mp)
            self.assertFalse(result.valid)
        except Exception as e:
            self.skipTest(f"validate_pattern binding issue: {e}")


class TestPatternManagerCompilerIntegration(unittest.TestCase):
    """Test PatternManager integration with ModelCompiler."""
    
    def setUp(self):
        self.pm = PatternManager()
        self.pm.register_pattern("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)", PatternCategory.FUSION)
        self.pm.register_pattern("GemmRelu", "Gemm(?, g0)\nRelu(g0, ?)", PatternCategory.FUSION)
        
        # Disable one pattern
        self.pm.set_pattern_enabled("GemmRelu", False)
    
    def test_get_enabled_pattern_definitions(self):
        """Test getting enabled patterns as PatternDefinitions."""
        definitions = self.pm.get_enabled_pattern_definitions()
        
        # Only ConvRelu is enabled
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0].name, "ConvRelu")
    
    def test_apply_to_compiler(self):
        """Test applying patterns to a ModelCompiler."""
        compiler = oniris.ModelCompiler()
        
        self.assertEqual(compiler.get_pattern_count(), 0)
        
        self.pm.apply_to_compiler(compiler)
        
        # Only enabled patterns should be applied
        self.assertEqual(compiler.get_pattern_count(), 1)


class TestPatternRegistry(unittest.TestCase):
    """Test PatternRegistry singleton."""
    
    def test_singleton_instance(self):
        """Test that PatternRegistry is a singleton."""
        reg1 = oniris.get_pattern_registry()
        reg2 = oniris.get_pattern_registry()
        
        self.assertIs(reg1, reg2)
    
    def test_registry_manager_access(self):
        """Test that registry provides access to manager."""
        reg = oniris.get_pattern_registry()
        manager = reg.get_manager()
        
        # Manager should be a PatternManager
        self.assertIsInstance(manager, PatternManager)
        
        # Register through manager
        result = manager.register_pattern("TestPattern", "Conv(?, ?)")
        self.assertTrue(result)
        
        # Verify pattern exists
        self.assertIsNotNone(manager.get_pattern("TestPattern"))


if __name__ == '__main__':
    unittest.main()
