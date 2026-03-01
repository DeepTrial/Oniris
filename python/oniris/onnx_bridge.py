"""
ONNX Bridge - Bridge between onnx python package and Oniris IR

This module provides functions to convert between onnx.ModelProto and Oniris Model.
"""

from typing import Optional, Dict, Any, List
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Import from the C++ extension
from ._oniris import (
    Model, Graph, Node, ValueInfo, Shape, Dimension, DataType,
    OpsetImport
)


def onnx_dtype_to_oniris(dtype: int) -> DataType:
    """Convert ONNX data type to Oniris DataType."""
    mapping = {
        0: DataType.UNDEFINED,
        1: DataType.FLOAT,
        2: DataType.UINT8,
        3: DataType.INT8,
        4: DataType.UINT16,
        5: DataType.INT16,
        6: DataType.INT32,
        7: DataType.INT64,
        8: DataType.STRING,
        9: DataType.BOOL,
        10: DataType.FLOAT16,
        11: DataType.DOUBLE,
        12: DataType.UINT32,
        13: DataType.UINT64,
        14: DataType.COMPLEX64,
        15: DataType.COMPLEX128,
        16: DataType.BFLOAT16,
    }
    return mapping.get(dtype, DataType.UNDEFINED)


def _oniris_dtype_to_onnx(dtype: DataType) -> int:
    """Convert Oniris DataType to ONNX data type."""
    mapping = {
        DataType.UNDEFINED: 0,
        DataType.FLOAT: 1,
        DataType.UINT8: 2,
        DataType.INT8: 3,
        DataType.UINT16: 4,
        DataType.INT16: 5,
        DataType.INT32: 6,
        DataType.INT64: 7,
        DataType.STRING: 8,
        DataType.BOOL: 9,
        DataType.FLOAT16: 10,
        DataType.DOUBLE: 11,
        DataType.UINT32: 12,
        DataType.UINT64: 13,
        DataType.COMPLEX64: 14,
        DataType.COMPLEX128: 15,
        DataType.BFLOAT16: 16,
    }
    return mapping.get(dtype, 0)


def oniris_dtype_to_numpy(dtype: DataType) -> np.dtype:
    """Convert Oniris DataType to numpy dtype."""
    mapping = {
        DataType.FLOAT: np.float32,
        DataType.UINT8: np.uint8,
        DataType.INT8: np.int8,
        DataType.UINT16: np.uint16,
        DataType.INT16: np.int16,
        DataType.INT32: np.int32,
        DataType.INT64: np.int64,
        DataType.BOOL: np.bool_,
        DataType.FLOAT16: np.float16,
        DataType.DOUBLE: np.float64,
        DataType.UINT32: np.uint32,
        DataType.UINT64: np.uint64,
        DataType.BFLOAT16: np.float32,  # Use float32 as proxy
    }
    return mapping.get(dtype, np.float32)


def convert_onnx_value_info(onnx_value_info) -> ValueInfo:
    """Convert onnx.ValueInfoProto to Oniris ValueInfo."""
    value_info = ValueInfo()
    value_info.name = onnx_value_info.name
    
    if onnx_value_info.type and onnx_value_info.type.tensor_type:
        tensor_type = onnx_value_info.type.tensor_type
        
        # Convert data type
        value_info.dtype = onnx_dtype_to_oniris(tensor_type.elem_type)
        
        # Convert shape
        if tensor_type.shape:
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value > 0:
                    dims.append(Dimension(dim.dim_value))
                elif dim.dim_param:
                    dims.append(Dimension(dim.dim_param))
                else:
                    dims.append(Dimension())  # Dynamic
            value_info.shape = Shape(dims)
    
    return value_info


def convert_onnx_attribute(onnx_attr):
    """Convert onnx AttributeProto to Python value."""
    attr_type = onnx_attr.type
    
    if attr_type == onnx.AttributeProto.INT:
        return onnx_attr.i
    elif attr_type == onnx.AttributeProto.INTS:
        return list(onnx_attr.ints)
    elif attr_type == onnx.AttributeProto.FLOAT:
        return onnx_attr.f
    elif attr_type == onnx.AttributeProto.FLOATS:
        return list(onnx_attr.floats)
    elif attr_type == onnx.AttributeProto.STRING:
        return onnx_attr.s.decode('utf-8') if isinstance(onnx_attr.s, bytes) else onnx_attr.s
    elif attr_type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') if isinstance(s, bytes) else s for s in onnx_attr.strings]
    elif attr_type == onnx.AttributeProto.TENSOR:
        # Convert tensor to numpy array
        return numpy_helper.to_array(onnx_attr.t)
    else:
        return None


def convert_onnx_node(onnx_node) -> Node:
    """Convert onnx.NodeProto to Oniris Node."""
    node = Node(onnx_node.op_type, onnx_node.name)
    node.set_domain(onnx_node.domain)
    
    # Add inputs
    for input_name in onnx_node.input:
        node.add_input(input_name)
    
    # Add outputs
    for output_name in onnx_node.output:
        node.add_output(output_name)
    
    # Add attributes
    for attr in onnx_node.attribute:
        value = convert_onnx_attribute(attr)
        if value is not None:
            if isinstance(value, int):
                node.set_attribute_int(attr.name, value)
            elif isinstance(value, float):
                node.set_attribute_float(attr.name, value)
            elif isinstance(value, str):
                node.set_attribute_string(attr.name, value)
            elif isinstance(value, list):
                if value and isinstance(value[0], int):
                    node.set_attribute_ints(attr.name, value)
                elif value and isinstance(value[0], float):
                    node.set_attribute_floats(attr.name, value)
    
    return node


def convert_onnx_graph(onnx_graph) -> Graph:
    """Convert onnx.GraphProto to Oniris Graph."""
    graph = Graph(onnx_graph.name)
    
    # Add initializers as value info with shape
    for init in onnx_graph.initializer:
        value_info = ValueInfo()
        value_info.name = init.name
        value_info.dtype = onnx_dtype_to_oniris(init.data_type)
        
        # Convert shape
        dims = [Dimension(int(d)) for d in init.dims]
        value_info.shape = Shape(dims)
        
        # Store shape in graph's value info
        graph.set_value_info(init.name, value_info)
    
    # Add inputs
    for input_info in onnx_graph.input:
        value_info = convert_onnx_value_info(input_info)
        graph.add_input(value_info)
        # Also set value info
        graph.set_value_info(value_info.name, value_info)
    
    # Add outputs
    for output_info in onnx_graph.output:
        value_info = convert_onnx_value_info(output_info)
        graph.add_output(value_info)
    
    # Add value_info (intermediate tensors with shapes)
    for value_info_proto in onnx_graph.value_info:
        value_info = convert_onnx_value_info(value_info_proto)
        graph.set_value_info(value_info.name, value_info)
    
    # Add nodes
    for onnx_node in onnx_graph.node:
        node = convert_onnx_node(onnx_node)
        graph.add_node(node)
    
    return graph


def load_onnx_model(path: str) -> Optional[Model]:
    """Load an ONNX model from file using onnx python package and convert to Oniris Model."""
    if not HAS_ONNX:
        raise ImportError("onnx package is required. Install with: pip install onnx")
    
    onnx_model = onnx.load(path)
    return convert_onnx_model(onnx_model)


def convert_onnx_model(onnx_model) -> Model:
    """Convert onnx.ModelProto to Oniris Model."""
    if not HAS_ONNX:
        raise ImportError("onnx package is required. Install with: pip install onnx")
    
    model = Model(onnx_model.ir_version)
    model.set_producer_name(onnx_model.producer_name)
    model.set_producer_version(onnx_model.producer_version)
    model.set_domain(onnx_model.domain)
    model.set_model_version(onnx_model.model_version)
    model.set_doc_string(onnx_model.doc_string)
    
    # Add opset imports
    for opset in onnx_model.opset_import:
        opset_import = OpsetImport()
        opset_import.domain = opset.domain
        opset_import.version = opset.version
        model.add_opset_import(opset_import)
    
    # Convert graph
    if onnx_model.graph:
        graph = convert_onnx_graph(onnx_model.graph)
        model.set_graph(graph)
    
    return model


def save_onnx_model(model: Model, path: str):
    """Save Oniris Model to ONNX file using onnx python package."""
    if not HAS_ONNX:
        raise ImportError("onnx package is required. Install with: pip install onnx")
    
    onnx_model = convert_to_onnx(model)
    onnx.save(onnx_model, path)


def convert_to_onnx(model: Model):
    """Convert Oniris Model to onnx.ModelProto."""
    if not HAS_ONNX:
        raise ImportError("onnx package is required. Install with: pip install onnx")
    
    onnx_model = onnx.ModelProto()
    onnx_model.ir_version = int(model.get_ir_version())
    onnx_model.producer_name = model.get_producer_name()
    onnx_model.producer_version = model.get_producer_version()
    onnx_model.domain = model.get_domain()
    onnx_model.model_version = int(model.get_model_version())
    onnx_model.doc_string = model.get_doc_string()
    
    # Add opset imports
    for opset in model.get_opset_imports():
        onnx_opset = onnx_model.opset_import.add()
        onnx_opset.domain = opset.domain
        onnx_opset.version = int(opset.version)
    
    # Convert graph
    graph = model.get_graph()
    if graph:
        onnx_model.graph.CopyFrom(convert_graph_to_onnx(graph))
    
    return onnx_model


def convert_graph_to_onnx(graph: Graph):
    """Convert Oniris Graph to onnx.GraphProto."""
    if not HAS_ONNX:
        raise ImportError("onnx package is required")
    
    onnx_graph = onnx.GraphProto()
    onnx_graph.name = graph.get_name()
    
    # Convert nodes
    for node in graph.get_nodes():
        onnx_node = onnx_graph.node.add()
        onnx_node.op_type = node.get_op_type()
        onnx_node.name = node.get_name()
        onnx_node.domain = node.get_domain()
        onnx_node.input.extend(node.get_inputs())
        onnx_node.output.extend(node.get_outputs())
        
        # Convert attributes - try common attribute names
        # This is simplified - full implementation would track all attributes
        common_attrs = ['kernel_shape', 'pads', 'strides', 'dilations', 'group', 
                       'axis', 'axes', 'keepdims', 'perm', 'shape', 'to',
                       'alpha', 'beta', 'transA', 'transB', 'broadcast']
        for attr_name in common_attrs:
            try:
                if node.has_attribute(attr_name):
                    attr_val = node.get_attribute(attr_name)
                    onnx_attr = onnx_node.attribute.add()
                    onnx_attr.name = attr_name
                    
                    if isinstance(attr_val, int):
                        onnx_attr.type = onnx.AttributeProto.INT
                        onnx_attr.i = attr_val
                    elif isinstance(attr_val, float):
                        onnx_attr.type = onnx.AttributeProto.FLOAT
                        onnx_attr.f = attr_val
                    elif isinstance(attr_val, str):
                        onnx_attr.type = onnx.AttributeProto.STRING
                        onnx_attr.s = attr_val.encode('utf-8')
                    elif isinstance(attr_val, list):
                        if attr_val and isinstance(attr_val[0], int):
                            onnx_attr.type = onnx.AttributeProto.INTS
                            onnx_attr.ints.extend(attr_val)
                        elif attr_val and isinstance(attr_val[0], float):
                            onnx_attr.type = onnx.AttributeProto.FLOATS
                            onnx_attr.floats.extend(attr_val)
                        elif attr_val and isinstance(attr_val[0], str):
                            onnx_attr.type = onnx.AttributeProto.STRINGS
                            onnx_attr.strings.extend([s.encode('utf-8') for s in attr_val])
            except:
                pass
    
    # Convert inputs (ValueInfo objects)
    for value_info in graph.get_inputs():
        onnx_input = onnx_graph.input.add()
        onnx_input.name = value_info.name
        # Convert dtype
        onnx_input.type.tensor_type.elem_type = _oniris_dtype_to_onnx(value_info.dtype)
        # Convert shape
        if value_info.shape.num_dims() > 0:
            for dim in value_info.shape.get_dims():
                onnx_dim = onnx_input.type.tensor_type.shape.dim.add()
                if dim.is_dynamic():
                    if dim.get_symbolic_name():
                        onnx_dim.dim_param = dim.get_symbolic_name()
                    # else: leave empty for unknown dim
                else:
                    onnx_dim.dim_value = dim.get_static_value()
    
    # Convert outputs (ValueInfo objects)
    for value_info in graph.get_outputs():
        onnx_output = onnx_graph.output.add()
        onnx_output.name = value_info.name
        onnx_output.type.tensor_type.elem_type = _oniris_dtype_to_onnx(value_info.dtype)
        # Convert shape
        if value_info.shape.num_dims() > 0:
            for dim in value_info.shape.get_dims():
                onnx_dim = onnx_output.type.tensor_type.shape.dim.add()
                if dim.is_dynamic():
                    if dim.get_symbolic_name():
                        onnx_dim.dim_param = dim.get_symbolic_name()
                else:
                    onnx_dim.dim_value = dim.get_static_value()
    
    return onnx_graph
