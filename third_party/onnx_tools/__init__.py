"""
ONNX Model Modification Tools

A set of utilities for modifying ONNX models, including:
- Tensor shape modification
- Weight/initializer replacement
- Node removal
- Node/tensor renaming
- Generic layer addition (all ONNX + Microsoft ops)

Usage:
    # Generic layer addition
    from third_party.onnx_tools import add_layer
    
    model = add_layer(model, "Conv", "input", "conv_out", name="conv1",
                      kernel_size=3, in_channels=3, out_channels=64)
    
    # Microsoft domain ops
    model = add_layer(model, "FusedConv", "input", "out", 
                      domain="com.microsoft", activation="Relu",
                      kernel_size=3, in_channels=3, out_channels=64)
"""

# Core model modification functions
from .model_modifier import (
    # Tensor operations
    modify_tensor_shape,
    get_tensor_shape,
    set_tensor_shape,
    
    # Initializer operations
    replace_initializer,
    replace_initializer_from_file,
    
    # Node operations
    remove_node,
    insert_node,
    find_node_by_name,
    find_nodes_by_op,
    
    # Rename operations
    rename_node,
    rename_tensor,
    
    # Model modifier class
    ModelModifier,
)

# Generic layer addition API (NEW)
from .layer_builder import (
    # Main generic API
    add_layer,
    
    # Convenience functions
    add_conv,
    add_linear,
    add_activation,
    add_norm,
    add_pooling,
    add_dropout,
    add_shape_manipulation,
    
    # Microsoft domain ops
    add_fused_conv,
    add_fused_gemm,
    add_attention,
    
    # Schemas (for advanced use)
    ONNX_OP_SCHEMAS,
    MICROSOFT_OP_SCHEMAS,
    ALL_OP_SCHEMAS,
    get_op_schema,
)

__version__ = "0.2.0"

__all__ = [
    # Tensor operations
    "modify_tensor_shape",
    "get_tensor_shape",
    "set_tensor_shape",
    
    # Initializer operations
    "replace_initializer",
    "replace_initializer_from_file",
    
    # Node operations
    "remove_node",
    "insert_node",
    "find_node_by_name",
    "find_nodes_by_op",
    
    # Rename operations
    "rename_node",
    "rename_tensor",
    
    # Generic layer addition (NEW)
    "add_layer",
    "add_conv",
    "add_linear",
    "add_activation",
    "add_norm",
    "add_pooling",
    "add_dropout",
    "add_shape_manipulation",
    "add_fused_conv",
    "add_fused_gemm",
    "add_attention",
    
    # Schemas
    "ONNX_OP_SCHEMAS",
    "MICROSOFT_OP_SCHEMAS",
    "ALL_OP_SCHEMAS",
    "get_op_schema",
    
    # Class
    "ModelModifier",
]
