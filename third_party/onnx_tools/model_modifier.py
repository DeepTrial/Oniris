"""
ONNX Model Modification Utilities

Provides functions to modify ONNX models including:
- Modify tensor shapes
- Replace initializers with numpy data
- Remove nodes
- Rename nodes and tensors
"""

import onnx
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional


def modify_tensor_shape(model: onnx.ModelProto, tensor_name: str, new_shape: List[int]) -> onnx.ModelProto:
    """
    Modify the shape of a tensor in the model.
    
    Args:
        model: ONNX model
        tensor_name: Name of the tensor to modify
        new_shape: New shape for the tensor
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = modify_tensor_shape(model, "input", [1, 3, 224, 224])
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Try to find and modify in inputs
    for input_proto in model.graph.input:
        if input_proto.name == tensor_name:
            _set_tensor_shape(input_proto, new_shape)
            return model
    
    # Try to find and modify in outputs
    for output_proto in model.graph.output:
        if output_proto.name == tensor_name:
            _set_tensor_shape(output_proto, new_shape)
            return model
    
    # Try to find and modify in value_info
    for value_info in model.graph.value_info:
        if value_info.name == tensor_name:
            _set_tensor_shape(value_info, new_shape)
            return model
    
    # Try to find and modify in initializers
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            # For initializers, we need to reshape the data
            old_shape = list(initializer.dims)
            if np.prod(old_shape) != np.prod(new_shape):
                raise ValueError(
                    f"Cannot reshape initializer '{tensor_name}' from {old_shape} to {new_shape}: "
                    f"size mismatch ({np.prod(old_shape)} vs {np.prod(new_shape)})"
                )
            del initializer.dims[:]
            initializer.dims.extend(new_shape)
            return model
    
    raise ValueError(f"Tensor '{tensor_name}' not found in model")


def replace_initializer(
    model: onnx.ModelProto, 
    initializer_name: str, 
    numpy_array: np.ndarray,
    name: Optional[str] = None
) -> onnx.ModelProto:
    """
    Replace an initializer with data from a numpy array.
    
    Args:
        model: ONNX model
        initializer_name: Name of the initializer to replace
        numpy_array: New data as numpy array
        name: Optional new name for the initializer
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> new_weights = np.load("new_weights.npy")
        >>> model = replace_initializer(model, "conv1.weight", new_weights)
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Find and remove old initializer
    old_initializer = None
    for i, init in enumerate(model.graph.initializer):
        if init.name == initializer_name:
            old_initializer = init
            del model.graph.initializer[i]
            break
    
    if old_initializer is None:
        raise ValueError(f"Initializer '{initializer_name}' not found in model")
    
    # Create new initializer
    new_name = name if name is not None else initializer_name
    new_initializer = numpy_to_initializer(new_name, numpy_array)
    
    model.graph.initializer.append(new_initializer)
    
    # Update references if name changed
    if name is not None and name != initializer_name:
        model = rename_tensor(model, initializer_name, name)
    
    return model


def replace_initializer_from_file(
    model: onnx.ModelProto,
    initializer_name: str,
    numpy_file: Union[str, Path],
    name: Optional[str] = None
) -> onnx.ModelProto:
    """
    Replace an initializer with data from a numpy file.
    
    Args:
        model: ONNX model
        initializer_name: Name of the initializer to replace
        numpy_file: Path to .npy file
        name: Optional new name for the initializer
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = replace_initializer_from_file(model, "conv1.weight", "new_weights.npy")
        >>> onnx.save(model, "modified.onnx")
    """
    numpy_array = np.load(numpy_file)
    return replace_initializer(model, initializer_name, numpy_array, name)


def remove_node(
    model: onnx.ModelProto, 
    node_name: str,
    reconnect_inputs: bool = True
) -> onnx.ModelProto:
    """
    Remove a node from the model.
    
    Args:
        model: ONNX model
        node_name: Name of the node to remove
        reconnect_inputs: If True, connect the node's inputs to its outputs
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = remove_node(model, "dropout_1")
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Find the node
    node_to_remove = None
    node_index = -1
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            node_to_remove = node
            node_index = i
            break
    
    if node_to_remove is None:
        raise ValueError(f"Node '{node_name}' not found in model")
    
    # Get inputs and outputs
    inputs = list(node_to_remove.input)
    outputs = list(node_to_remove.output)
    
    # Remove the node
    del model.graph.node[node_index]
    
    # Reconnect if requested and possible
    if reconnect_inputs and len(inputs) > 0 and len(outputs) > 0:
        # Simple case: connect first input to all outputs
        # More complex cases may require custom handling
        _reconnect_tensors(model, inputs[0], outputs)
    
    return model


def rename_node(model: onnx.ModelProto, old_name: str, new_name: str) -> onnx.ModelProto:
    """
    Rename a node in the model.
    
    Args:
        model: ONNX model
        old_name: Current name of the node
        new_name: New name for the node
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = rename_node(model, "conv1", "feature_conv1")
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Find and rename the node
    for node in model.graph.node:
        if node.name == old_name:
            node.name = new_name
            return model
    
    raise ValueError(f"Node '{old_name}' not found in model")


def rename_tensor(
    model: onnx.ModelProto, 
    old_name: str, 
    new_name: str
) -> onnx.ModelProto:
    """
    Rename a tensor (value) in the model.
    Updates all references to the tensor.
    
    Args:
        model: ONNX model
        old_name: Current name of the tensor
        new_name: New name for the tensor
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = rename_tensor(model, "input_0", "image_input")
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    found = False
    
    # Update in inputs
    for input_proto in model.graph.input:
        if input_proto.name == old_name:
            input_proto.name = new_name
            found = True
    
    # Update in outputs
    for output_proto in model.graph.output:
        if output_proto.name == old_name:
            output_proto.name = new_name
            found = True
    
    # Update in value_info
    for value_info in model.graph.value_info:
        if value_info.name == old_name:
            value_info.name = new_name
            found = True
    
    # Update in initializers
    for initializer in model.graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name
            found = True
    
    # Update in nodes (inputs and outputs)
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old_name:
                node.input[i] = new_name
                found = True
        for i, name in enumerate(node.output):
            if name == old_name:
                node.output[i] = new_name
                found = True
    
    if not found:
        raise ValueError(f"Tensor '{old_name}' not found in model")
    
    return model


def get_tensor_shape(model: onnx.ModelProto, tensor_name: str) -> List[int]:
    """
    Get the shape of a tensor.
    
    Args:
        model: ONNX model
        tensor_name: Name of the tensor
        
    Returns:
        Shape as a list of integers
    """
    # Search in inputs
    for tensor in model.graph.input:
        if tensor.name == tensor_name:
            return _get_tensor_shape(tensor)
    
    # Search in outputs
    for tensor in model.graph.output:
        if tensor.name == tensor_name:
            return _get_tensor_shape(tensor)
    
    # Search in value_info
    for tensor in model.graph.value_info:
        if tensor.name == tensor_name:
            return _get_tensor_shape(tensor)
    
    # Search in initializers
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)
    
    raise ValueError(f"Tensor '{tensor_name}' not found")


def set_tensor_shape(model: onnx.ModelProto, tensor_name: str, new_shape: List[int]) -> onnx.ModelProto:
    """Alias for modify_tensor_shape."""
    return modify_tensor_shape(model, tensor_name, new_shape)


def insert_node(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    before: Optional[str] = None,
    after: Optional[str] = None
) -> onnx.ModelProto:
    """
    Insert a node into the graph.
    
    Args:
        model: ONNX model
        node: Node to insert
        before: Insert before this node name
        after: Insert after this node name
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    if before is None and after is None:
        # Append to end
        model.graph.node.append(node)
    elif before is not None:
        # Insert before specified node
        for i, n in enumerate(model.graph.node):
            if n.name == before:
                model.graph.node.insert(i, node)
                break
    elif after is not None:
        # Insert after specified node
        for i, n in enumerate(model.graph.node):
            if n.name == after:
                model.graph.node.insert(i + 1, node)
                break
    
    return model


def find_node_by_name(model: onnx.ModelProto, node_name: str) -> Optional[onnx.NodeProto]:
    """
    Find a node by name.
    
    Args:
        model: ONNX model
        node_name: Name of the node to find
        
    Returns:
        NodeProto if found, None otherwise
    """
    for node in model.graph.node:
        if node.name == node_name:
            return node
    return None


def find_nodes_by_op(model: onnx.ModelProto, op_type: str) -> List[onnx.NodeProto]:
    """
    Find all nodes with a specific op type.
    
    Args:
        model: ONNX model
        op_type: Operator type to search for
        
    Returns:
        List of matching nodes
    """
    return [node for node in model.graph.node if node.op_type == op_type]


def _get_tensor_shape(tensor_proto):
    """Get shape from tensor proto."""
    shape = []
    for dim in tensor_proto.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(-1)  # Dynamic dimension
        else:
            shape.append(-1)  # Unknown
    return shape


class ModelModifier:
    """
    A class for chaining multiple model modifications.
    
    Example:
        >>> modifier = ModelModifier("model.onnx")
        >>> modifier.modify_tensor_shape("input", [1, 3, 224, 224])
        >>> modifier.replace_initializer_from_file("conv1.weight", "new_weights.npy")
        >>> modifier.remove_node("dropout_1")
        >>> modifier.rename_tensor("input_0", "image")
        >>> modifier.save("modified.onnx")
    """
    
    def __init__(self, model: Union[str, Path, onnx.ModelProto]):
        """
        Initialize with a model path or ModelProto.
        
        Args:
            model: Path to ONNX model or ModelProto object
        """
        if isinstance(model, (str, Path)):
            self.model = onnx.load(model)
        else:
            self.model = copy_model(model)
    
    def modify_tensor_shape(self, tensor_name: str, new_shape: List[int]) -> 'ModelModifier':
        """Modify tensor shape (chainable)."""
        self.model = modify_tensor_shape(self.model, tensor_name, new_shape)
        return self
    
    def replace_initializer(
        self, 
        initializer_name: str, 
        numpy_array: np.ndarray,
        name: Optional[str] = None
    ) -> 'ModelModifier':
        """Replace initializer (chainable)."""
        self.model = replace_initializer(self.model, initializer_name, numpy_array, name)
        return self
    
    def replace_initializer_from_file(
        self,
        initializer_name: str,
        numpy_file: Union[str, Path],
        name: Optional[str] = None
    ) -> 'ModelModifier':
        """Replace initializer from file (chainable)."""
        self.model = replace_initializer_from_file(self.model, initializer_name, numpy_file, name)
        return self
    
    def remove_node(self, node_name: str, reconnect_inputs: bool = True) -> 'ModelModifier':
        """Remove node (chainable)."""
        self.model = remove_node(self.model, node_name, reconnect_inputs)
        return self
    
    def rename_node(self, old_name: str, new_name: str) -> 'ModelModifier':
        """Rename node (chainable)."""
        self.model = rename_node(self.model, old_name, new_name)
        return self
    
    def rename_tensor(self, old_name: str, new_name: str) -> 'ModelModifier':
        """Rename tensor (chainable)."""
        self.model = rename_tensor(self.model, old_name, new_name)
        return self
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the modified model."""
        onnx.save(self.model, path)
    
    def get_model(self) -> onnx.ModelProto:
        """Get the modified model."""
        return self.model


# Helper functions

def copy_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Create a deep copy of the model."""
    return onnx.ModelProto.FromString(model.SerializeToString())


def _set_tensor_shape(tensor_proto, shape: List[int]):
    """Set the shape of a tensor proto."""
    # Clear existing shape
    while len(tensor_proto.type.tensor_type.shape.dim) > 0:
        tensor_proto.type.tensor_type.shape.dim.pop()
    
    # Add new dimensions
    for dim in shape:
        dim_proto = tensor_proto.type.tensor_type.shape.dim.add()
        if dim is None or dim == -1:
            dim_proto.dim_param = "dynamic"
        else:
            dim_proto.dim_value = dim


def _reconnect_tensors(model: onnx.ModelProto, input_name: str, output_names: List[str]):
    """Reconnect tensors by replacing output references with input."""
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in output_names:
                node.input[i] = input_name


def numpy_to_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    """Convert numpy array to ONNX initializer."""
    # Map numpy dtypes to ONNX types
    dtype_map = {
        np.float32: onnx.TensorProto.FLOAT,
        np.float64: onnx.TensorProto.DOUBLE,
        np.int8: onnx.TensorProto.INT8,
        np.int16: onnx.TensorProto.INT16,
        np.int32: onnx.TensorProto.INT32,
        np.int64: onnx.TensorProto.INT64,
        np.uint8: onnx.TensorProto.UINT8,
        np.uint16: onnx.TensorProto.UINT16,
        np.uint32: onnx.TensorProto.UINT32,
        np.uint64: onnx.TensorProto.UINT64,
        np.bool_: onnx.TensorProto.BOOL,
        np.float16: onnx.TensorProto.FLOAT16,
    }
    
    onnx_dtype = dtype_map.get(array.dtype.type)
    if onnx_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {array.dtype}")
    
    # Ensure contiguous array
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    
    return onnx.helper.make_tensor(
        name=name,
        data_type=onnx_dtype,
        dims=array.shape,
        vals=array.flatten().tolist()
    )


# =============================================================================
# Layer Addition Functions
# =============================================================================

def add_conv2d(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
    initializer_type: np.dtype = np.float32
) -> onnx.ModelProto:
    """
    Add a Conv2d layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Conv node
        input_name: Name of input tensor
        output_name: Name of output tensor
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (int or tuple)
        stride: Stride (int or tuple)
        padding: Padding (int or 4-element tuple for asymmetric)
        dilation: Dilation (int or tuple)
        groups: Number of groups for grouped convolution
        bias: Whether to include bias
        weight_name: Name for weight initializer (default: {name}_weight)
        bias_name: Name for bias initializer (default: {name}_bias)
        initializer_type: Data type for initializers
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = add_conv2d(model, "conv1", "input", "conv_out", 
        ...                    in_channels=3, out_channels=64, kernel_size=7, 
        ...                    stride=2, padding=3)
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Normalize parameters
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    
    # Default names
    w_name = weight_name or f"{name}_W"
    b_name = bias_name or f"{name}_B" if bias else None
    
    # Create weight initializer (shape: [out_channels, in_channels/groups, kH, kW])
    weight_shape = [out_channels, in_channels // groups, kernel_size[0], kernel_size[1]]
    weight_data = np.random.randn(*weight_shape).astype(initializer_type) * 0.01
    weight_init = numpy_to_initializer(w_name, weight_data)
    model.graph.initializer.append(weight_init)
    
    # Create node inputs
    node_inputs = [input_name, w_name]
    
    # Create bias initializer if needed
    if bias:
        bias_data = np.zeros(out_channels, dtype=initializer_type)
        bias_init = numpy_to_initializer(b_name, bias_data)
        model.graph.initializer.append(bias_init)
        node_inputs.append(b_name)
    
    # Create Conv node
    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=node_inputs,
        outputs=[output_name],
        name=name,
        kernel_shape=list(kernel_size),
        strides=list(stride),
        pads=list(padding),
        dilations=list(dilation),
        group=groups
    )
    
    model.graph.node.append(conv_node)
    
    # Add value_info for output shape inference hint
    output_shape = _compute_conv_output_shape(
        None, in_channels, out_channels, kernel_size, 
        stride, padding, dilation
    )
    value_info = onnx.helper.make_tensor_value_info(
        output_name, 
        onnx.TensorProto.FLOAT if initializer_type == np.float32 else onnx.TensorProto.DOUBLE,
        output_shape
    )
    model.graph.value_info.append(value_info)
    
    return model


def add_linear(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
    initializer_type: np.dtype = np.float32
) -> onnx.ModelProto:
    """
    Add a Linear (Fully Connected / Gemm) layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Gemm node
        input_name: Name of input tensor
        output_name: Name of output tensor
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias
        weight_name: Name for weight initializer (default: {name}_W)
        bias_name: Name for bias initializer (default: {name}_B)
        initializer_type: Data type for initializers
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = add_linear(model, "fc1", "flatten", "fc_out",
        ...                    in_features=512, out_features=10)
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Default names
    w_name = weight_name or f"{name}_W"
    b_name = bias_name or f"{name}_B" if bias else None
    
    # Create weight initializer (shape: [out_features, in_features] for Gemm)
    weight_shape = [out_features, in_features]
    weight_data = np.random.randn(*weight_shape).astype(initializer_type) * 0.01
    weight_init = numpy_to_initializer(w_name, weight_data)
    model.graph.initializer.append(weight_init)
    
    # Create node inputs
    node_inputs = [input_name, w_name]
    
    # Create bias initializer if needed
    if bias:
        bias_data = np.zeros(out_features, dtype=initializer_type)
        bias_init = numpy_to_initializer(b_name, bias_data)
        model.graph.initializer.append(bias_init)
        node_inputs.append(b_name)
    
    # Create Gemm node (General Matrix Multiplication)
    gemm_node = onnx.helper.make_node(
        "Gemm",
        inputs=node_inputs,
        outputs=[output_name],
        name=name,
        transB=1  # Weight is stored as [out, in], need transpose for Gemm
    )
    
    model.graph.node.append(gemm_node)
    
    # Add value_info
    value_info = onnx.helper.make_tensor_value_info(
        output_name,
        onnx.TensorProto.FLOAT if initializer_type == np.float32 else onnx.TensorProto.DOUBLE,
        [None, out_features]  # Batch dimension can be dynamic
    )
    model.graph.value_info.append(value_info)
    
    return model


def add_activation(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    activation: str = "relu"
) -> onnx.ModelProto:
    """
    Add an activation layer to the model.
    
    Supported activations: relu, sigmoid, tanh, leakyrelu, elu, selu,
                           softmax, logsoftmax, clip, hardsigmoid
    
    Args:
        model: ONNX model
        name: Name for the activation node
        input_name: Name of input tensor
        output_name: Name of output tensor
        activation: Type of activation function
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = add_activation(model, "relu1", "conv_out", "relu_out", "relu")
        >>> model = add_activation(model, "sig1", "fc_out", "prob", "sigmoid")
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    activation = activation.lower()
    
    # Map activation names to ONNX op types and attributes
    op_map = {
        "relu": ("Relu", {}),
        "sigmoid": ("Sigmoid", {}),
        "tanh": ("Tanh", {}),
        "leakyrelu": ("LeakyRelu", {"alpha": 0.01}),
        "elu": ("Elu", {"alpha": 1.0}),
        "selu": ("Selu", {"alpha": 1.67326, "gamma": 1.0507}),
        "softmax": ("Softmax", {"axis": -1}),
        "logsoftmax": ("LogSoftmax", {"axis": -1}),
        "hardsigmoid": ("HardSigmoid", {"alpha": 0.2, "beta": 0.5}),
    }
    
    if activation not in op_map:
        raise ValueError(f"Unsupported activation: {activation}. "
                        f"Supported: {list(op_map.keys())}")
    
    op_type, attrs = op_map[activation]
    
    # Create node
    node = onnx.helper.make_node(
        op_type,
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        **attrs
    )
    
    model.graph.node.append(node)
    
    return model


def add_batchnorm2d(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    num_features: int,
    epsilon: float = 1e-5,
    momentum: float = 0.9,
    scale_name: Optional[str] = None,
    bias_name: Optional[str] = None,
    mean_name: Optional[str] = None,
    var_name: Optional[str] = None,
    initializer_type: np.dtype = np.float32
) -> onnx.ModelProto:
    """
    Add a BatchNorm2d layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the BatchNormalization node
        input_name: Name of input tensor
        output_name: Name of output tensor
        num_features: Number of features/channels
        epsilon: Small constant for numerical stability
        momentum: Momentum for running statistics
        scale_name: Name for scale initializer (default: {name}_scale)
        bias_name: Name for bias initializer (default: {name}_bias)
        mean_name: Name for mean initializer (default: {name}_mean)
        var_name: Name for var initializer (default: {name}_var)
        initializer_type: Data type for initializers
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = add_batchnorm2d(model, "bn1", "conv_out", "bn_out", num_features=64)
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Default names
    s_name = scale_name or f"{name}_scale"
    b_name = bias_name or f"{name}_bias"
    m_name = mean_name or f"{name}_mean"
    v_name = var_name or f"{name}_var"
    
    # Create initializers
    # Scale (gamma): initialized to 1
    scale_data = np.ones(num_features, dtype=initializer_type)
    scale_init = numpy_to_initializer(s_name, scale_data)
    model.graph.initializer.append(scale_init)
    
    # Bias (beta): initialized to 0
    bias_data = np.zeros(num_features, dtype=initializer_type)
    bias_init = numpy_to_initializer(b_name, bias_data)
    model.graph.initializer.append(bias_init)
    
    # Running mean: initialized to 0
    mean_data = np.zeros(num_features, dtype=initializer_type)
    mean_init = numpy_to_initializer(m_name, mean_data)
    model.graph.initializer.append(mean_init)
    
    # Running var: initialized to 1
    var_data = np.ones(num_features, dtype=initializer_type)
    var_init = numpy_to_initializer(v_name, var_data)
    model.graph.initializer.append(var_init)
    
    # Create BatchNormalization node
    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=[input_name, s_name, b_name, m_name, v_name],
        outputs=[output_name],
        name=name,
        epsilon=epsilon,
        momentum=momentum
    )
    
    model.graph.node.append(bn_node)
    
    return model


def add_pooling(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    pool_type: str = "max",
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int, int, int]] = 0,
    ceil_mode: bool = False
) -> onnx.ModelProto:
    """
    Add a pooling layer (MaxPool or AveragePool) to the model.
    
    Args:
        model: ONNX model
        name: Name for the pooling node
        input_name: Name of input tensor
        output_name: Name of output tensor
        pool_type: "max" or "avg"
        kernel_size: Kernel size (int or tuple)
        stride: Stride (int or tuple, defaults to kernel_size)
        padding: Padding (int or 4-element tuple)
        ceil_mode: Whether to use ceiling mode
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = add_pooling(model, "pool1", "relu_out", "pool_out",
        ...                     pool_type="max", kernel_size=2, stride=2)
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # Normalize parameters
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    
    # Determine op type
    pool_type = pool_type.lower()
    if pool_type in ("max", "maxpool", "max_pool"):
        op_type = "MaxPool"
    elif pool_type in ("avg", "average", "avgpool", "averagepool", "global"):
        op_type = "AveragePool"
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")
    
    # Create node
    pool_node = onnx.helper.make_node(
        op_type,
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        kernel_shape=list(kernel_size),
        strides=list(stride),
        pads=list(padding),
        ceil_mode=1 if ceil_mode else 0
    )
    
    model.graph.node.append(pool_node)
    
    return model


def add_global_average_pool(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str
) -> onnx.ModelProto:
    """
    Add a GlobalAveragePool layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the node
        input_name: Name of input tensor
        output_name: Name of output tensor
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    gap_node = onnx.helper.make_node(
        "GlobalAveragePool",
        inputs=[input_name],
        outputs=[output_name],
        name=name
    )
    
    model.graph.node.append(gap_node)
    
    return model


def add_dropout(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    ratio: float = 0.5,
    training_mode: bool = False
) -> onnx.ModelProto:
    """
    Add a Dropout layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Dropout node
        input_name: Name of input tensor
        output_name: Name of output tensor
        ratio: Dropout ratio (probability of dropping)
        training_mode: Whether in training mode
        
    Returns:
        Modified model
        
    Example:
        >>> model = onnx.load("model.onnx")
        >>> model = add_dropout(model, "drop1", "fc_out", "drop_out", ratio=0.5)
        >>> onnx.save(model, "modified.onnx")
    """
    model = copy_model(model)
    
    # ONNX Dropout takes ratio as an input in newer opsets, but as attribute in older
    # We'll create a constant tensor for ratio
    ratio_name = f"{name}_ratio"
    ratio_data = np.array([ratio], dtype=np.float32)
    ratio_init = numpy_to_initializer(ratio_name, ratio_data)
    model.graph.initializer.append(ratio_init)
    
    # Create outputs (output + optional mask)
    if training_mode:
        mask_name = f"{output_name}_mask"
        outputs = [output_name, mask_name]
    else:
        outputs = [output_name]
    
    dropout_node = onnx.helper.make_node(
        "Dropout",
        inputs=[input_name, ratio_name],
        outputs=outputs,
        name=name
    )
    
    model.graph.node.append(dropout_node)
    
    return model


def add_flatten(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    axis: int = 1
) -> onnx.ModelProto:
    """
    Add a Flatten layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Flatten node
        input_name: Name of input tensor
        output_name: Name of output tensor
        axis: Axis to flatten from
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    flatten_node = onnx.helper.make_node(
        "Flatten",
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        axis=axis
    )
    
    model.graph.node.append(flatten_node)
    
    return model


def add_concat(
    model: onnx.ModelProto,
    name: str,
    input_names: List[str],
    output_name: str,
    axis: int = 0
) -> onnx.ModelProto:
    """
    Add a Concat layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Concat node
        input_names: List of input tensor names
        output_name: Name of output tensor
        axis: Axis to concatenate along
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=input_names,
        outputs=[output_name],
        name=name,
        axis=axis
    )
    
    model.graph.node.append(concat_node)
    
    return model


def add_reshape(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    shape: List[int],
    allowzero: bool = False
) -> onnx.ModelProto:
    """
    Add a Reshape layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Reshape node
        input_name: Name of input tensor
        output_name: Name of output tensor
        shape: Target shape (use 0 for copying dimension, -1 for inferred)
        allowzero: Whether to allow zeros in shape
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    # Shape is a constant input
    shape_name = f"{name}_shape"
    shape_data = np.array(shape, dtype=np.int64)
    shape_init = numpy_to_initializer(shape_name, shape_data)
    model.graph.initializer.append(shape_init)
    
    reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=[input_name, shape_name],
        outputs=[output_name],
        name=name,
        allowzero=1 if allowzero else 0
    )
    
    model.graph.node.append(reshape_node)
    
    return model


def add_transpose(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    perm: List[int]
) -> onnx.ModelProto:
    """
    Add a Transpose layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Transpose node
        input_name: Name of input tensor
        output_name: Name of output tensor
        perm: Permutation of dimensions
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        perm=perm
    )
    
    model.graph.node.append(transpose_node)
    
    return model


def add_elementwise(
    model: onnx.ModelProto,
    name: str,
    input_names: List[str],
    output_name: str,
    op_type: str
) -> onnx.ModelProto:
    """
    Add an element-wise operation layer (Add, Mul, Sub, Div).
    
    Args:
        model: ONNX model
        name: Name for the node
        input_names: List of input tensor names (2 for binary ops)
        output_name: Name of output tensor
        op_type: Operation type ("Add", "Mul", "Sub", "Div")
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    valid_ops = ("Add", "Mul", "Sub", "Div", "Pow", "Max", "Min")
    if op_type not in valid_ops:
        raise ValueError(f"Unsupported op_type: {op_type}. Valid: {valid_ops}")
    
    node = onnx.helper.make_node(
        op_type,
        inputs=input_names,
        outputs=[output_name],
        name=name
    )
    
    model.graph.node.append(node)
    
    return model


def add_softmax(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    axis: int = -1
) -> onnx.ModelProto:
    """
    Add a Softmax layer to the model.
    
    Args:
        model: ONNX model
        name: Name for the Softmax node
        input_name: Name of input tensor
        output_name: Name of output tensor
        axis: Axis to apply softmax
        
    Returns:
        Modified model
    """
    model = copy_model(model)
    
    softmax_node = onnx.helper.make_node(
        "Softmax",
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        axis=axis
    )
    
    model.graph.node.append(softmax_node)
    
    return model


# =============================================================================
# Helper Functions for Layer Addition
# =============================================================================

def _compute_conv_output_shape(
    input_shape, in_channels, out_channels, kernel_size, stride, padding, dilation
):
    """Compute output shape hint for Conv layer."""
    # Return shape with dynamic batch and computed spatial dims
    # Actual shape inference will be done by ONNX
    return [1, out_channels, 32, 32]  # Placeholder, actual shape depends on input


# Add methods to ModelModifier class for chain support
def _add_conv2d_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int, int, int]] = 0,
    **kwargs
) -> 'ModelModifier':
    """Chainable wrapper for add_conv2d."""
    self.model = add_conv2d(
        self.model, name, input_name, output_name,
        in_channels, out_channels, kernel_size, stride, padding,
        **kwargs
    )
    return self


def _add_linear_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    **kwargs
) -> 'ModelModifier':
    """Chainable wrapper for add_linear."""
    self.model = add_linear(
        self.model, name, input_name, output_name,
        in_features, out_features, bias, **kwargs
    )
    return self


def _add_activation_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    activation: str = "relu"
) -> 'ModelModifier':
    """Chainable wrapper for add_activation."""
    self.model = add_activation(self.model, name, input_name, output_name, activation)
    return self


def _add_batchnorm2d_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    num_features: int,
    **kwargs
) -> 'ModelModifier':
    """Chainable wrapper for add_batchnorm2d."""
    self.model = add_batchnorm2d(
        self.model, name, input_name, output_name, num_features, **kwargs
    )
    return self


def _add_pooling_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    pool_type: str = "max",
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    **kwargs
) -> 'ModelModifier':
    """Chainable wrapper for add_pooling."""
    self.model = add_pooling(
        self.model, name, input_name, output_name,
        pool_type, kernel_size, stride=stride, **kwargs
    )
    return self


def _add_dropout_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    ratio: float = 0.5
) -> 'ModelModifier':
    """Chainable wrapper for add_dropout."""
    self.model = add_dropout(self.model, name, input_name, output_name, ratio)
    return self


def _add_flatten_chain(
    self,
    name: str,
    input_name: str,
    output_name: str,
    axis: int = 1
) -> 'ModelModifier':
    """Chainable wrapper for add_flatten."""
    self.model = add_flatten(self.model, name, input_name, output_name, axis)
    return self


# Attach chain methods to ModelModifier
ModelModifier.add_conv2d = _add_conv2d_chain
ModelModifier.add_linear = _add_linear_chain
ModelModifier.add_activation = _add_activation_chain
ModelModifier.add_batchnorm2d = _add_batchnorm2d_chain
ModelModifier.add_pooling = _add_pooling_chain
ModelModifier.add_dropout = _add_dropout_chain
ModelModifier.add_flatten = _add_flatten_chain
