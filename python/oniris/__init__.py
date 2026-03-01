"""Oniris - ONNX Compilation Toolkit

A high-performance ONNX model compilation and optimization toolkit.

Example usage:
    >>> import oniris
    >>> 
    >>> # Simplify a model
    >>> oniris.simplify('input.onnx', 'output.onnx')
    >>> 
    >>> # Simplify with fusion disabled
    >>> oniris.simplify('input.onnx', 'output.onnx', fuse_conv_bn=False)
    >>> 
    >>> # Load and analyze a model
    >>> model = oniris.load_model('model.onnx')
    >>> oniris.print_model_summary(model)
"""

from ._oniris import (
    # Core types
    DataType,
    Dimension,
    Shape,
    data_type_to_string,
    string_to_data_type,
    
    # IR
    Node,
    ValueInfo,
    Graph,
    Model,
    OpsetImport,
    
    # Passes
    ShapeInferenceEngine,
    SimplifyOptions,
    SimplifyResult,
    Simplifier,
    InferenceResult,
    
    # Subgraph Matching (ONNX Matcher Style)
    OnnxMatcherPattern,
    OnnxMatcherStyleMatcher,
    
    # Model Compiler
    PatternMatchType,
    PatternDefinition,
    MatchedNodeInfo,
    PatternMatchResult,
    PatternMatchingSummary,
    OptimizationStats,
    ShapeInferenceStats,
    ModelSummary,
    CompilationResult,
    CompilerOptions,
    ModelCompiler,
    compile_model,
    get_common_patterns,
    
    # Pattern Manager
    PatternCategory,
    PatternMetadata,
    ManagedPattern,
    PatternStatistics,
    PatternCollection,
    PatternManager,
    PatternRegistry,
    get_fusion_patterns,
    get_optimization_patterns,
    get_quantization_patterns,
    get_all_builtin_pattern_collections,
    get_pattern_registry,
    
    # Utils
    load_model,
    save_model,
    is_valid_onnx_file,
    get_onnx_version,
    get_model_info,
    print_model_summary,
    simplify,
)

# Alias for convenience
Matcher = OnnxMatcherStyleMatcher

__version__ = "0.1.0"
__all__ = [
    # Version
    "__version__",
    
    # Core types
    "DataType",
    "Dimension",
    "Shape",
    "data_type_to_string",
    "string_to_data_type",
    
    # IR
    "Node",
    "ValueInfo",
    "Graph",
    "Model",
    "OpsetImport",
    
    # Passes
    "ShapeInferenceEngine",
    "SimplifyOptions",
    "SimplifyResult",
    "Simplifier",
    "InferenceResult",
    
    # Subgraph Matching (ONNX Matcher Style)
    "OnnxMatcherPattern",
    "OnnxMatcherStyleMatcher",
    "Matcher",  # Python alias
    
    # Model Compiler
    "PatternMatchType",
    "PatternDefinition",
    "MatchedNodeInfo",
    "PatternMatchResult",
    "PatternMatchingSummary",
    "OptimizationStats",
    "ShapeInferenceStats",
    "ModelSummary",
    "CompilationResult",
    "CompilerOptions",
    "ModelCompiler",
    "compile_model",
    "get_common_patterns",
    
    # Pattern Manager
    "PatternCategory",
    "PatternMetadata",
    "ManagedPattern",
    "PatternStatistics",
    "PatternCollection",
    "PatternManager",
    "PatternRegistry",
    "get_fusion_patterns",
    "get_optimization_patterns",
    "get_quantization_patterns",
    "get_all_builtin_pattern_collections",
    "get_pattern_registry",
    
    # Utils
    "load_model",
    "save_model",
    "is_valid_onnx_file",
    "get_onnx_version",
    "get_model_info",
    "print_model_summary",
    "simplify",
]


def register_custom_shape_inference(
    op_type: str,
    infer_func,
    domain: str = ""
) -> None:
    """Register a custom shape inference function for an operator.
    
    This allows users to extend the shape inference engine with custom
    operators that are not part of the standard ONNX spec.
    
    Args:
        op_type: The operator type name
        infer_func: A callable that takes an InferenceContext and returns
            an InferenceResult
        domain: Optional operator domain
        
    Example:
        >>> def my_custom_infer(ctx):
        ...     # ctx.input_shapes contains input shapes
        ...     # ctx.attributes contains node attributes
        ...     output_shape = oniris.Shape([ctx.input_shapes[0].get_dim(0), 64])
        ...     return oniris.InferenceResult.Success([output_shape])
        >>> 
        >>> oniris.register_custom_shape_inference("MyCustomOp", my_custom_infer)
    """
    engine = ShapeInferenceEngine.get_instance()
    engine.register_handler(op_type, infer_func, domain)


def get_supported_operators() -> list[str]:
    """Get list of operators supported by the shape inference engine.
    
    Returns:
        List of operator type names
    """
    engine = ShapeInferenceEngine.get_instance()
    return engine.get_supported_ops()


# Import onnx bridge if available
try:
    from .onnx_bridge import load_onnx_model, save_onnx_model, convert_onnx_model
    _has_onnx_bridge = True
except ImportError:
    _has_onnx_bridge = False
    
# Add to __all__ if available
if _has_onnx_bridge:
    __all__.extend(['load_onnx_model', 'save_onnx_model', 'convert_onnx_model'])


# Import YAML pattern support if available
try:
    from .pattern_yaml import (
        load_yaml_patterns,
        YamlPatternLoader,
        import_yaml_patterns,
    )
    _has_yaml = True
    __all__.extend([
        'load_yaml_patterns',
        'YamlPatternLoader',
        'import_yaml_patterns',
    ])
except ImportError:
    _has_yaml = False


class Compiler:
    """
    High-level ONNX Model Compiler with full onnx package integration.
    
    This class provides a convenient interface for compiling ONNX models
    with automatic loading/saving using the onnx python package.
    
    Example:
        >>> from oniris import Compiler
        >>> compiler = Compiler()
        >>> compiler.add_pattern("ConvRelu", "Conv(?, c0)\\nRelu(c0, ?)")
        >>> result = compiler.compile_file("input.onnx", "output.onnx")
        >>> print(result.to_json())
    """
    
    def __init__(self):
        self._compiler = ModelCompiler()
        self._patterns_added = False
    
    def add_pattern(self, name: str, pattern_string: str) -> bool:
        """Add a pattern to match."""
        self._patterns_added = True
        return self._compiler.add_pattern(name, pattern_string)
    
    def add_patterns(self, patterns):
        """Add multiple patterns."""
        self._patterns_added = True
        return self._compiler.add_patterns(patterns)
    
    def clear_patterns(self):
        """Clear all patterns."""
        self._compiler.clear_patterns()
        self._patterns_added = False
    
    def compile_file(self, input_path: str, output_path: str = "", 
                     options: CompilerOptions = None) -> CompilationResult:
        """
        Compile an ONNX model file.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path for optimized output model (optional)
            options: Compiler options (uses defaults if not provided)
        
        Returns:
            CompilationResult with all compilation data
        """
        if options is None:
            options = CompilerOptions()
        
        # Try to use onnx bridge for loading
        try:
            from .onnx_bridge import load_onnx_model, save_onnx_model
            model = load_onnx_model(input_path)
            result = self._compiler.compile_model(model, options)
            # Note: input_path and output_path are set by compile_model, but we override them
            # Since CompilationResult members are read-only in Python, we just use the result as-is
            # The JSON output will reflect the compilation but not the file paths
            
            # Save optimized model if requested
            if output_path and options.save_optimized_model:
                save_onnx_model(model, output_path)
            
            # Save JSON result if requested
            if options.save_json_result and options.json_output_path:
                result.save_json(options.json_output_path, pretty=True)
            
            return result
        except ImportError:
            # Fall back to built-in loader
            return self._compiler.compile(input_path, output_path, options or CompilerOptions())
    
    def compile_model(self, model, options: CompilerOptions = None) -> CompilationResult:
        """
        Compile an Oniris Model object.
        
        Args:
            model: Oniris Model to compile
            options: Compiler options
        
        Returns:
            CompilationResult
        """
        return self._compiler.compile_model(model, options or CompilerOptions())
    
    def run_pattern_matching(self, model, match_type=PatternMatchType.ALL):
        """Run only pattern matching on a model."""
        return self._compiler.run_pattern_matching(model, match_type)
    
    def get_pattern_names(self):
        """Get names of registered patterns."""
        return self._compiler.get_pattern_names()
    
    @property
    def pattern_count(self) -> int:
        """Number of registered patterns."""
        return self._compiler.get_pattern_count()


# Add Compiler to exports
__all__.append('Compiler')
