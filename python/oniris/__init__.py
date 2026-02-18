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
