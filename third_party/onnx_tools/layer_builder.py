"""
Generic Layer Builder for ONNX Model Modification.

Provides a unified API for adding any ONNX operator (standard and Microsoft domain).
"""

import onnx
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any, Callable


# =============================================================================
# Operator Schema Definitions
# =============================================================================

# Standard ONNX ops with their common attributes and inputs/outputs
ONNX_OP_SCHEMAS = {
    # Convolution
    "Conv": {
        "inputs": ["X", "W", "B"],  # B is optional
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "group": {"type": "int", "default": 1},
            "auto_pad": {"type": "string", "default": "NOTSET"},
        },
        "initializer_builder": "conv",
    },
    "ConvTranspose": {
        "inputs": ["X", "W", "B"],
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "group": {"type": "int", "default": 1},
            "output_padding": {"type": "ints", "default": [0, 0]},
            "output_shape": {"type": "ints", "default": None},
            "auto_pad": {"type": "string", "default": "NOTSET"},
        },
        "initializer_builder": "conv_transpose",
    },
    "ConvInteger": {
        "inputs": ["x", "w", "x_zero_point", "w_zero_point"],
        "outputs": ["y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "group": {"type": "int", "default": 1},
        },
    },
    # Linear / MatMul
    "Gemm": {
        "inputs": ["A", "B", "C"],  # C is optional
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.0},
            "beta": {"type": "float", "default": 1.0},
            "transA": {"type": "int", "default": 0},
            "transB": {"type": "int", "default": 0},
        },
        "initializer_builder": "gemm",
    },
    "MatMul": {
        "inputs": ["A", "B"],
        "outputs": ["Y"],
        "attributes": {},
    },
    "MatMulInteger": {
        "inputs": ["A", "B", "a_zero_point", "b_zero_point"],
        "outputs": ["Y"],
        "attributes": {},
    },
    # Activations
    "Relu": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Sigmoid": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Tanh": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Softmax": {
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": -1},
        },
    },
    "LogSoftmax": {
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": -1},
        },
    },
    "LeakyRelu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 0.01},
        },
    },
    "Elu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.0},
        },
    },
    "Selu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.67326},
            "gamma": {"type": "float", "default": 1.0507},
        },
    },
    "PRelu": {
        "inputs": ["X", "slope"],
        "outputs": ["Y"],
        "attributes": {},
        "initializer_builder": "prelu",
    },
    "Gelu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {},
    },
    "HardSigmoid": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 0.2},
            "beta": {"type": "float", "default": 0.5},
        },
    },
    "HardSwish": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Softplus": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Softsign": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Clip": {
        "inputs": ["input", "min", "max"],
        "outputs": ["output"],
        "attributes": {},
    },
    "Celu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.0},
        },
    },
    "Mish": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "ThresholdedRelu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.0},
        },
    },
    # Normalization
    "BatchNormalization": {
        "inputs": ["X", "scale", "B", "input_mean", "input_var"],
        "outputs": ["Y", "running_mean", "running_var", "saved_mean", "saved_var"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-5},
            "momentum": {"type": "float", "default": 0.9},
            "training_mode": {"type": "int", "default": 0},
        },
        "initializer_builder": "batchnorm",
    },
    "InstanceNormalization": {
        "inputs": ["input", "scale", "B"],
        "outputs": ["output"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-5},
        },
        "initializer_builder": "instance_norm",
    },
    "LayerNormalization": {
        "inputs": ["X", "Scale", "B"],
        "outputs": ["Y", "Mean", "InvStdDev"],
        "attributes": {
            "axis": {"type": "int", "default": -1},
            "epsilon": {"type": "float", "default": 1e-5},
            "stash_type": {"type": "int", "default": 1},
        },
        "initializer_builder": "layer_norm",
    },
    "GroupNormalization": {
        "inputs": ["X", "scale", "bias"],
        "outputs": ["Y"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-5},
            "num_groups": {"type": "int", "required": True},
        },
        "initializer_builder": "group_norm",
    },
    # Pooling
    "MaxPool": {
        "inputs": ["X"],
        "outputs": ["Y", "Indices"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": None},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "ceil_mode": {"type": "int", "default": 0},
            "storage_order": {"type": "int", "default": 0},
            "auto_pad": {"type": "string", "default": "NOTSET"},
        },
    },
    "AveragePool": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": None},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "ceil_mode": {"type": "int", "default": 0},
            "count_include_pad": {"type": "int", "default": 0},
            "auto_pad": {"type": "string", "default": "NOTSET"},
        },
    },
    "GlobalMaxPool": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "GlobalAveragePool": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "LpPool": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": None},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "p": {"type": "int", "default": 2},
        },
    },
    "MaxUnpool": {
        "inputs": ["X", "I", "output_shape"],
        "outputs": ["output"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": None},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
        },
    },
    "RoiAlign": {
        "inputs": ["X", "rois", "batch_indices"],
        "outputs": ["Y"],
        "attributes": {
            "mode": {"type": "string", "default": "avg"},
            "output_height": {"type": "int", "default": 1},
            "output_width": {"type": "int", "default": 1},
            "sampling_ratio": {"type": "int", "default": 0},
            "spatial_scale": {"type": "float", "default": 1.0},
        },
    },
    # Dropout
    "Dropout": {
        "inputs": ["data", "ratio", "training_mode"],
        "outputs": ["output", "mask"],
        "attributes": {},
    },
    # Shape manipulation
    "Flatten": {
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": 1},
        },
    },
    "Reshape": {
        "inputs": ["data", "shape"],
        "outputs": ["reshaped"],
        "attributes": {
            "allowzero": {"type": "int", "default": 0},
        },
    },
    "Transpose": {
        "inputs": ["data"],
        "outputs": ["transposed"],
        "attributes": {
            "perm": {"type": "ints", "default": None},
        },
    },
    "Squeeze": {
        "inputs": ["data", "axes"],
        "outputs": ["squeezed"],
        "attributes": {},
    },
    "Unsqueeze": {
        "inputs": ["data", "axes"],
        "outputs": ["expanded"],
        "attributes": {},
    },
    "Expand": {
        "inputs": ["input", "shape"],
        "outputs": ["output"],
        "attributes": {},
    },
    "Concat": {
        "inputs": None,  # Variable number of inputs
        "outputs": ["concat_result"],
        "attributes": {
            "axis": {"type": "int", "required": True},
        },
    },
    "Split": {
        "inputs": ["input", "split"],
        "outputs": None,  # Variable number of outputs
        "attributes": {
            "axis": {"type": "int", "default": 0},
            "num_outputs": {"type": "int", "default": None},
        },
    },
    "Tile": {
        "inputs": ["input", "repeats"],
        "outputs": ["output"],
        "attributes": {},
    },
    "Pad": {
        "inputs": ["data", "pads", "constant_value", "axes"],
        "outputs": ["output"],
        "attributes": {
            "mode": {"type": "string", "default": "constant"},
        },
    },
    # Element-wise operations
    "Add": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Sub": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Mul": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Div": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Pow": {"inputs": ["X", "Y"], "outputs": ["Z"], "attributes": {}},
    "Max": {"inputs": None, "outputs": ["max"], "attributes": {}},
    "Min": {"inputs": None, "outputs": ["min"], "attributes": {}},
    "Mean": {"inputs": None, "outputs": ["mean"], "attributes": {}},
    "Sum": {"inputs": None, "outputs": ["sum"], "attributes": {}},
    "Abs": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Neg": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Floor": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Ceil": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Round": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Sqrt": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Reciprocal": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Exp": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Log": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Sin": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Cos": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Tan": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Asin": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Acos": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Atan": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Sinh": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Cosh": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Asinh": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Acosh": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Atanh": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Erf": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "Sign": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "IsNaN": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    "IsInf": {"inputs": ["X"], "outputs": ["Y"], "attributes": {
        "detect_negative": {"type": "int", "default": 1},
        "detect_positive": {"type": "int", "default": 1},
    }},
    "Where": {"inputs": ["condition", "X", "Y"], "outputs": ["output"], "attributes": {}},
    # Comparison
    "Equal": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Greater": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "GreaterOrEqual": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Less": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "LessOrEqual": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Not": {"inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    # Logical
    "And": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Or": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    "Xor": {"inputs": ["A", "B"], "outputs": ["C"], "attributes": {}},
    # Reduction
    "ReduceMean": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceSum": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceMax": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceMin": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceProd": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceL1": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceL2": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceLogSum": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceLogSumExp": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ReduceSumSquare": {
        "inputs": ["data", "axes"],
        "outputs": ["reduced"],
        "attributes": {
            "keepdims": {"type": "int", "default": 1},
            "noop_with_empty_axes": {"type": "int", "default": 0},
        },
    },
    "ArgMax": {
        "inputs": ["data"],
        "outputs": ["reduced"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
            "keepdims": {"type": "int", "default": 1},
            "select_last_index": {"type": "int", "default": 0},
        },
    },
    "ArgMin": {
        "inputs": ["data"],
        "outputs": ["reduced"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
            "keepdims": {"type": "int", "default": 1},
            "select_last_index": {"type": "int", "default": 0},
        },
    },
    # Broadcasting and indexing
    "Gather": {
        "inputs": ["data", "indices"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
        },
    },
    "GatherElements": {
        "inputs": ["data", "indices"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
        },
    },
    "GatherND": {
        "inputs": ["data", "indices"],
        "outputs": ["output"],
        "attributes": {
            "batch_dims": {"type": "int", "default": 0},
        },
    },
    "Scatter": {
        "inputs": ["data", "indices", "updates"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
            "reduction": {"type": "string", "default": "none"},
        },
    },
    "ScatterElements": {
        "inputs": ["data", "indices", "updates"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
            "reduction": {"type": "string", "default": "none"},
        },
    },
    "ScatterND": {
        "inputs": ["data", "indices", "updates"],
        "outputs": ["output"],
        "attributes": {
            "reduction": {"type": "string", "default": "none"},
        },
    },
    "Slice": {
        "inputs": ["data", "starts", "ends", "axes", "steps"],
        "outputs": ["output"],
        "attributes": {},
    },
    "OneHot": {
        "inputs": ["indices", "depth", "values"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": -1},
        },
    },
    # Shape operations
    "Shape": {
        "inputs": ["data"],
        "outputs": ["shape"],
        "attributes": {
            "start": {"type": "int", "default": None},
            "end": {"type": "int", "default": None},
        },
    },
    "Size": {"inputs": ["data"], "outputs": ["size"], "attributes": {}},
    "Rank": {"inputs": ["data"], "outputs": ["rank"], "attributes": {}},
    # Casting
    "Cast": {
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": {
            "to": {"type": "int", "required": True},  # TensorProto.DataType
            "saturate": {"type": "int", "default": 1},
        },
    },
    "CastLike": {
        "inputs": ["input", "target_type"],
        "outputs": ["output"],
        "attributes": {
            "saturate": {"type": "int", "default": 1},
        },
    },
    # Identity
    "Identity": {"inputs": ["input"], "outputs": ["output"], "attributes": {}},
    # Constant
    "Constant": {
        "inputs": [],
        "outputs": ["output"],
        "attributes": {
            "value": {"type": "tensor", "default": None},
            "sparse_value": {"type": "sparse_tensor", "default": None},
            "value_float": {"type": "float", "default": None},
            "value_floats": {"type": "floats", "default": None},
            "value_int": {"type": "int", "default": None},
            "value_ints": {"type": "ints", "default": None},
            "value_string": {"type": "string", "default": None},
            "value_strings": {"type": "strings", "default": None},
        },
    },
    # Reshape variants
    "Flatten": {
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": 1},
        },
    },
    "Compress": {
        "inputs": ["input", "condition"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "default": None},
        },
    },
    "ReverseSequence": {
        "inputs": ["input", "sequence_lens"],
        "outputs": ["Y"],
        "attributes": {
            "batch_axis": {"type": "int", "default": 1},
            "time_axis": {"type": "int", "default": 0},
        },
    },
    "Unique": {
        "inputs": ["X"],
        "outputs": ["Y", "indices", "inverse_indices", "counts"],
        "attributes": {
            "axis": {"type": "int", "default": None},
            "sorted": {"type": "int", "default": 1},
        },
    },
    # Grid operations
    "GridSample": {
        "inputs": ["X", "Grid"],
        "outputs": ["Y"],
        "attributes": {
            "align_corners": {"type": "int", "default": 0},
            "mode": {"type": "string", "default": "bilinear"},
            "padding_mode": {"type": "string", "default": "zeros"},
        },
    },
    # Optional - for sequence and map types
    "SequenceConstruct": {"inputs": None, "outputs": ["output_sequence"], "attributes": {}},
    "SequenceEmpty": {"inputs": [], "outputs": ["output_sequence"], "attributes": {}},
    "SequenceLength": {"inputs": ["input_sequence"], "outputs": ["length"], "attributes": {}},
    "ConcatFromSequence": {
        "inputs": ["input_sequence"],
        "outputs": ["output"],
        "attributes": {
            "axis": {"type": "int", "required": True},
            "new_axis": {"type": "int", "default": 0},
        },
    },
    "SplitToSequence": {
        "inputs": ["input", "split"],
        "outputs": ["output_sequence"],
        "attributes": {
            "axis": {"type": "int", "default": 0},
            "keepdims": {"type": "int", "default": 1},
        },
    },
}

# Microsoft domain operators
MICROSOFT_OP_SCHEMAS = {
    # Fused Convolutions
    "com.microsoft.FusedConv": {
        "inputs": ["X", "W", "B", "Z"],
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "group": {"type": "int", "default": 1},
            "activation": {"type": "string", "default": ""},
            "activation_params": {"type": "floats", "default": []},
        },
        "initializer_builder": "fused_conv",
    },
    "com.microsoft.FusedGemm": {
        "inputs": ["A", "B", "C"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.0},
            "beta": {"type": "float", "default": 1.0},
            "transA": {"type": "int", "default": 0},
            "transB": {"type": "int", "default": 0},
            "activation": {"type": "string", "default": ""},
            "activation_params": {"type": "floats", "default": []},
        },
        "initializer_builder": "fused_gemm",
    },
    "com.microsoft.FusedMatMul": {
        "inputs": ["A", "B", "C"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.0},
            "transA": {"type": "int", "default": 0},
            "transB": {"type": "int", "default": 0},
            "activation": {"type": "string", "default": ""},
        },
    },
    # QLinear operations (Quantized)
    "com.microsoft.QLinearConv": {
        "inputs": ["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point", "B"],
        "outputs": ["y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "group": {"type": "int", "default": 1},
        },
    },
    "com.microsoft.QLinearMatMul": {
        "inputs": ["a", "a_scale", "a_zero_point", "b", "b_scale", "b_zero_point", "y_scale", "y_zero_point"],
        "outputs": ["y"],
        "attributes": {},
    },
    "com.microsoft.QLinearAdd": {
        "inputs": ["A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point", "C_scale", "C_zero_point"],
        "outputs": ["C"],
        "attributes": {},
    },
    "com.microsoft.QLinearSigmoid": {
        "inputs": ["X", "X_scale", "X_zero_point", "Y_scale", "Y_zero_point"],
        "outputs": ["Y"],
        "attributes": {},
    },
    "com.microsoft.QLinearLeakyRelu": {
        "inputs": ["X", "X_scale", "X_zero_point", "Y_scale", "Y_zero_point"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 0.01},
        },
    },
    # Attention and Transformers
    "com.microsoft.Attention": {
        "inputs": ["input", "weight", "bias", "mask_index", "past", "relative_position_bias"],
        "outputs": ["output", "present"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
            "unidirectional": {"type": "int", "default": 0},
            "past_present_share_buffer": {"type": "int", "default": 0},
            "qkv_hidden_sizes": {"type": "ints", "default": []},
        },
    },
    "com.microsoft.MultiHeadAttention": {
        "inputs": ["query", "key", "value", "bias", "key_padding_mask", "attention_bias", "past_key", "past_value"],
        "outputs": ["output", "present_key", "present_value"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
        },
    },
    "com.microsoft.SkipLayerNormalization": {
        "inputs": ["input", "skip", "gamma", "beta", "bias"],
        "outputs": ["output", "mean", "inv_std_var"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-12},
            "simplified": {"type": "int", "default": 0},
        },
    },
    "com.microsoft.SkipSimplifiedLayerNormalization": {
        "inputs": ["input", "skip", "gamma", "bias"],
        "outputs": ["output", "mean", "inv_std_var"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-12},
        },
    },
    "com.microsoft.EmbeddingLayerNormalization": {
        "inputs": ["input_ids", "word_embedding", "position_embedding", "segment_embedding", 
                   "gamma", "beta", "mask", "position_ids"],
        "outputs": ["output", "mask_index"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-12},
            "mask_index_type": {"type": "int", "default": 0},
        },
    },
    "com.microsoft.BiasGelu": {
        "inputs": ["A", "B"],
        "outputs": ["C"],
        "attributes": {},
    },
    "com.microsoft.FastGelu": {
        "inputs": ["X", "bias"],
        "outputs": ["Y"],
        "attributes": {},
    },
    # Grouped operations
    "com.microsoft.GroupNorm": {
        "inputs": ["X", "gamma", "beta"],
        "outputs": ["Y"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-5},
            "num_groups": {"type": "int", "required": True},
            "channels_last": {"type": "int", "default": 1},
        },
    },
    "com.microsoft.GroupQueryAttention": {
        "inputs": ["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length", "cos_cache", "sin_cache"],
        "outputs": ["output", "present_key", "present_value"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
            "kv_num_heads": {"type": "int", "required": True},
            "local_window_size": {"type": "int", "default": -1},
            "scale": {"type": "float", "default": None},
            "do_rotary": {"type": "int", "default": 0},
        },
    },
    # NHWC operations
    "com.microsoft.NhwcMaxPool": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "ceil_mode": {"type": "int", "default": 0},
            "storage_order": {"type": "int", "default": 0},
        },
    },
    "com.microsoft.NhwcConv": {
        "inputs": ["X", "W", "B"],
        "outputs": ["Y"],
        "attributes": {
            "kernel_shape": {"type": "ints", "required": True},
            "strides": {"type": "ints", "default": [1, 1]},
            "pads": {"type": "ints", "default": [0, 0, 0, 0]},
            "dilations": {"type": "ints", "default": [1, 1]},
            "group": {"type": "int", "default": 1},
            "activation": {"type": "string", "default": ""},
        },
    },
    # Sampling operations
    "com.microsoft.SampleOp": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "sample_size": {"type": "int", "required": True},
            "seed": {"type": "int", "default": 0},
        },
    },
    # Decoder masked multihead attention
    "com.microsoft.DecoderMaskedSelfAttention": {
        "inputs": ["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length"],
        "outputs": ["output", "present_key", "present_value", "qk_v"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
            "scale": {"type": "float", "default": None},
        },
    },
    "com.microsoft.DecoderMaskedMultiHeadAttention": {
        "inputs": ["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length"],
        "outputs": ["output", "present_key", "present_value"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
            "scale": {"type": "float", "default": None},
        },
    },
    # Rotary Embedding
    "com.microsoft.RotaryEmbedding": {
        "inputs": ["input", "position_ids", "cos_cache", "sin_cache"],
        "outputs": ["output"],
        "attributes": {
            "interleaved": {"type": "int", "default": 0},
            "num_heads": {"type": "int", "default": 0},
            "rotary_embedding_dim": {"type": "int", "default": 0},
            "scale": {"type": "float", "default": 1.0},
        },
    },
    # Remove Padding
    "com.microsoft.RemovePadding": {
        "inputs": ["input", "sequence_lengths"],
        "outputs": ["output", "cumulated_seq_len", "max_seq_len", "token_offset"],
        "attributes": {},
    },
    "com.microsoft.RestorePadding": {
        "inputs": ["input", "cumulated_seq_len", "token_offset"],
        "outputs": ["output"],
        "attributes": {},
    },
    # Packed operations
    "com.microsoft.PackedAttention": {
        "inputs": ["query", "key", "value", "packed_qkv", "token_offset", "cumulative_sequence_length", "max_seq_len"],
        "outputs": ["output"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
        },
    },
    "com.microsoft.PackedMultiHeadAttention": {
        "inputs": ["query", "key", "value", "packed_qkv", "token_offset", "cumulative_sequence_length", "max_seq_len"],
        "outputs": ["output"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
        },
    },
    # Cross attention
    "com.microsoft.CrossAttention": {
        "inputs": ["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length"],
        "outputs": ["output", "present_key", "present_value"],
        "attributes": {
            "num_heads": {"type": "int", "required": True},
        },
    },
    # Other Microsoft ops
    "com.microsoft.MatMulInteger16": {
        "inputs": ["A", "B"],
        "outputs": ["Y"],
        "attributes": {},
    },
    "com.microsoft.MatMulIntegerToFloat": {
        "inputs": ["A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point", "bias"],
        "outputs": ["Y"],
        "attributes": {
            "producer": {"type": "string", "default": ""},
        },
    },
    "com.microsoft.DynamicQuantizeLSTM": {
        "inputs": ["X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P", "W_scale", "W_zero_point", "R_scale", "R_zero_point"],
        "outputs": ["Y", "Y_h", "Y_c"],
        "attributes": {
            "hidden_size": {"type": "int", "required": True},
            "direction": {"type": "string", "default": "forward"},
        },
    },
    "com.microsoft.DynamicQuantizeMatMul": {
        "inputs": ["A", "B", "b_scale", "b_zero_point", "bias"],
        "outputs": ["Y"],
        "attributes": {},
    },
    "com.microsoft.MulInteger": {
        "inputs": ["A", "B"],
        "outputs": ["C"],
        "attributes": {},
    },
    "com.microsoft.Shlu": {
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": {},
    },
    "com.microsoft.QuickGelu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "alpha": {"type": "float", "default": 1.702},
        },
    },
    "com.microsoft.SimplifiedLayerNormalization": {
        "inputs": ["X", "scale"],
        "outputs": ["Y", "inv_std_var"],
        "attributes": {
            "epsilon": {"type": "float", "default": 1e-5},
            "axis": {"type": "int", "default": -1},
        },
    },
    "com.microsoft.Inverse": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {},
    },
    "com.microsoft.Trilu": {
        "inputs": ["input", "k"],
        "outputs": ["output"],
        "attributes": {
            "upper": {"type": "int", "default": 1},
        },
    },
    "com.microsoft.Gelu": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {},
    },
}

# Combine all schemas
ALL_OP_SCHEMAS = {**ONNX_OP_SCHEMAS, **MICROSOFT_OP_SCHEMAS}


def get_op_schema(op_type: str, domain: str = "") -> Optional[Dict]:
    """Get operator schema by op_type and domain."""
    full_name = f"{domain}.{op_type}" if domain else op_type
    return ALL_OP_SCHEMAS.get(full_name)


# =============================================================================
# Utility Functions
# =============================================================================

def numpy_to_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    """Convert numpy array to ONNX initializer."""
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
    
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    
    return onnx.helper.make_tensor(
        name=name,
        data_type=onnx_dtype,
        dims=array.shape,
        vals=array.flatten().tolist()
    )


def copy_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Create a deep copy of the model."""
    return onnx.ModelProto.FromString(model.SerializeToString())


def normalize_ints(value, expected_dims=2):
    """Normalize int or tuple to list of ints."""
    if isinstance(value, int):
        return [value] * expected_dims
    return list(value)


# =============================================================================
# Initializer Builders
# =============================================================================

def build_conv_initializers(
    model: onnx.ModelProto,
    node_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: List[int],
    groups: int = 1,
    bias: bool = True,
    dtype: np.dtype = np.float32,
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Build initializers for Conv layer.
    
    Returns:
        Tuple of (initializer_tensor_names, input_names_to_add)
    """
    w_name = weight_name or f"{node_name}_W"
    b_name = bias_name or f"{node_name}_B" if bias else None
    
    # Weight shape: [out_channels, in_channels/groups, *kernel_size]
    weight_shape = [out_channels, in_channels // groups] + list(kernel_size)
    weight_data = np.random.randn(*weight_shape).astype(dtype) * 0.01
    weight_init = numpy_to_initializer(w_name, weight_data)
    model.graph.initializer.append(weight_init)
    
    input_names = [w_name]
    init_names = [w_name]
    
    if bias:
        bias_data = np.zeros(out_channels, dtype=dtype)
        bias_init = numpy_to_initializer(b_name, bias_data)
        model.graph.initializer.append(bias_init)
        input_names.append(b_name)
        init_names.append(b_name)
    
    return init_names, input_names


def build_gemm_initializers(
    model: onnx.ModelProto,
    node_name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    dtype: np.dtype = np.float32,
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Build initializers for Gemm layer."""
    w_name = weight_name or f"{node_name}_W"
    b_name = bias_name or f"{node_name}_B" if bias else None
    
    # Weight shape: [out_features, in_features] for transB=1
    weight_shape = [out_features, in_features]
    weight_data = np.random.randn(*weight_shape).astype(dtype) * 0.01
    weight_init = numpy_to_initializer(w_name, weight_data)
    model.graph.initializer.append(weight_init)
    
    input_names = [w_name]
    init_names = [w_name]
    
    if bias:
        bias_data = np.zeros(out_features, dtype=dtype)
        bias_init = numpy_to_initializer(b_name, bias_data)
        model.graph.initializer.append(bias_init)
        input_names.append(b_name)
        init_names.append(b_name)
    
    return init_names, input_names


def build_batchnorm_initializers(
    model: onnx.ModelProto,
    node_name: str,
    num_features: int,
    dtype: np.dtype = np.float32,
) -> Tuple[List[str], List[str]]:
    """Build initializers for BatchNormalization layer."""
    s_name = f"{node_name}_scale"
    b_name = f"{node_name}_bias"
    m_name = f"{node_name}_mean"
    v_name = f"{node_name}_var"
    
    # Scale (gamma): ones
    scale_data = np.ones(num_features, dtype=dtype)
    model.graph.initializer.append(numpy_to_initializer(s_name, scale_data))
    
    # Bias (beta): zeros
    bias_data = np.zeros(num_features, dtype=dtype)
    model.graph.initializer.append(numpy_to_initializer(b_name, bias_data))
    
    # Mean: zeros
    mean_data = np.zeros(num_features, dtype=dtype)
    model.graph.initializer.append(numpy_to_initializer(m_name, mean_data))
    
    # Var: ones
    var_data = np.ones(num_features, dtype=dtype)
    model.graph.initializer.append(numpy_to_initializer(v_name, var_data))
    
    init_names = [s_name, b_name, m_name, v_name]
    input_names = init_names
    
    return init_names, input_names


# =============================================================================
# Main Generic Layer Addition API
# =============================================================================

def add_layer(
    model: onnx.ModelProto,
    op_type: str,
    inputs: Union[str, List[str], Dict[str, Union[str, Dict]]],
    outputs: Union[str, List[str]],
    name: Optional[str] = None,
    domain: str = "",
    attributes: Optional[Dict[str, Any]] = None,
    initializers: Optional[Dict[str, np.ndarray]] = None,
    **kwargs
) -> onnx.ModelProto:
    """
    Generic API to add any ONNX operator to a model.
    
    This is the most flexible way to add layers. You can specify inputs as:
    - Simple string (single input)
    - List of strings (multiple inputs, all from other tensors)
    - Dict mapping formal names to input specs (allows specifying initializers)
    
    Args:
        model: ONNX model
        op_type: Operator type (e.g., "Conv", "Relu", "MatMul", "FusedConv")
        inputs: Input specification - can be:
            - String: Single input tensor name
            - List[str]: Multiple input tensor names (all from other tensors)
            - Dict: Maps formal input names to specs:
                - "input_name": use tensor from another node
                - {"name": "init_name", "data": np.array(...)}: use initializer with data
                - {"name": "init_name", "shape": [...], "type": np.float32}: auto-generate initializer
        outputs: Output tensor name(s) - string for single, list for multiple
        name: Node name (optional, defaults to f"{op_type}_{random_id}")
        domain: Operator domain (empty for ONNX standard, "com.microsoft" for MS ops)
        attributes: Dict of attributes for the operator
        initializers: (Legacy) Dict mapping initializer names to numpy arrays
        **kwargs: Additional layer-specific parameters (see below)
    
    Layer-specific kwargs (legacy, for backward compatibility):
        Conv/ConvTranspose:
            - kernel_size, stride, padding, dilation, groups
            - in_channels, out_channels, bias
            - weight_name, bias_name, initializer_type
        
        Gemm/MatMul:
            - in_features, out_features, bias
            - weight_name, bias_name, initializer_type
            - transA, transB, alpha, beta
    
    Returns:
        Modified ONNX model
    
    Examples:
        # Simple - single input from another tensor
        model = add_layer(model, "Relu", "input", "output", name="relu1")
        
        # Multiple inputs from other tensors
        model = add_layer(model, "Add", ["a", "b"], "output", name="add1")
        
        # Conv with auto-generated initializers (legacy API)
        model = add_layer(model, "Conv", "input", "conv_out",
                          name="conv1",
                          kernel_size=3, in_channels=3, out_channels=64)
        
        # Conv with explicit initializer specs (NEW API)
        model = add_layer(model, "Conv",
                          inputs={
                              "X": "input_data",  # from another node
                              "W": {"name": "weight", "data": weight_array},
                              "B": {"name": "bias", "shape": [64], "type": np.float32}
                          },
                          outputs="conv_out",
                          name="conv1",
                          attributes={"kernel_shape": [3, 3], "strides": [1, 1]})
        
        # Gemm with mixed inputs
        model = add_layer(model, "Gemm",
                          inputs={
                              "A": "flatten",
                              "B": {"name": "weight", "shape": [512, 10]},
                              "C": {"name": "bias", "shape": [10]}
                          },
                          outputs="fc_out",
                          attributes={"transB": 1})
        
        # Elementwise with constant
        model = add_layer(model, "Mul",
                          inputs={
                              "A": "input",
                              "B": {"name": "scale", "data": np.array([0.5])}
                          },
                          outputs="scaled")
    """
    model = copy_model(model)
    
    # Generate default name if not provided
    if name is None:
        import random
        name = f"{op_type}_{random.randint(1000, 9999)}"
    
    # Parse inputs specification
    input_specs = _parse_inputs(inputs)
    
    # Normalize outputs to list
    if isinstance(outputs, str):
        outputs = [outputs]
    
    # Build final input names list and create initializers
    final_inputs = []
    
    for formal_name, spec in input_specs.items():
        if isinstance(spec, str):
            # Simple tensor name from another node
            final_inputs.append(spec)
        elif isinstance(spec, dict):
            # Initializer spec
            init_name = spec.get("name", f"{name}_{formal_name}")
            final_inputs.append(init_name)
            
            if "data" in spec:
                # Use provided data
                init_data = spec["data"]
                if not isinstance(init_data, np.ndarray):
                    init_data = np.array(init_data)
            elif "shape" in spec:
                # Auto-generate with given shape
                shape = spec["shape"]
                dtype = spec.get("type", np.float32)
                # Initialize with small random values for weights, zeros for bias
                if formal_name in ("B", "bias", "beta"):
                    init_data = np.zeros(shape, dtype=dtype)
                else:
                    init_data = np.random.randn(*shape).astype(dtype) * 0.01
            else:
                raise ValueError(f"Initializer spec must have 'data' or 'shape': {spec}")
            
            # Create and add initializer
            init = numpy_to_initializer(init_name, init_data)
            model.graph.initializer.append(init)
    
    # Handle legacy initializers parameter
    if initializers:
        for init_name, init_data in initializers.items():
            if init_name not in final_inputs:
                init = numpy_to_initializer(init_name, init_data)
                model.graph.initializer.append(init)
                final_inputs.append(init_name)
    
    # Handle legacy layer-specific kwargs (for backward compatibility)
    attributes = attributes or {}
    
    # Legacy: Conv with in_channels/out_channels
    if op_type in ("Conv", "ConvTranspose", "FusedConv") and "in_channels" in kwargs:
        in_ch = kwargs.pop("in_channels")
        out_ch = kwargs.pop("out_channels")
        kernel = normalize_ints(kwargs.pop("kernel_size", 3), 2)
        groups = kwargs.pop("groups", 1)
        bias = kwargs.pop("bias", True)
        dtype = kwargs.pop("initializer_type", np.float32)
        w_name = kwargs.pop("weight_name", None) or f"{name}_W"
        b_name = kwargs.pop("bias_name", None) or f"{name}_B"
        
        weight_shape = [out_ch, in_ch // groups] + list(kernel)
        weight_data = np.random.randn(*weight_shape).astype(dtype) * 0.01
        model.graph.initializer.append(numpy_to_initializer(w_name, weight_data))
        
        # Add to inputs if not already specified
        if w_name not in final_inputs:
            final_inputs.append(w_name)
        
        if bias:
            bias_data = np.zeros(out_ch, dtype=dtype)
            model.graph.initializer.append(numpy_to_initializer(b_name, bias_data))
            if b_name not in final_inputs:
                final_inputs.append(b_name)
        
        # Set attributes
        stride = normalize_ints(kwargs.pop("stride", 1), 2)
        padding = normalize_ints(kwargs.pop("padding", 0), 4)
        dilation = normalize_ints(kwargs.pop("dilation", 1), 2)
        
        attributes.setdefault("kernel_shape", kernel)
        attributes.setdefault("strides", stride)
        attributes.setdefault("pads", padding)
        attributes.setdefault("dilations", dilation)
        attributes.setdefault("group", groups)
        
        if op_type == "FusedConv":
            if "activation" in kwargs:
                attributes.setdefault("activation", kwargs.pop("activation"))
            if "activation_params" in kwargs:
                attributes.setdefault("activation_params", kwargs.pop("activation_params"))
    
    # Legacy: Gemm with in_features/out_features
    elif op_type == "Gemm" and "in_features" in kwargs:
        in_feat = kwargs.pop("in_features")
        out_feat = kwargs.pop("out_features")
        bias = kwargs.pop("bias", True)
        dtype = kwargs.pop("initializer_type", np.float32)
        w_name = kwargs.pop("weight_name", None) or f"{name}_W"
        b_name = kwargs.pop("bias_name", None) or f"{name}_B"
        
        # Weight shape: [out_features, in_features] for transB=1
        weight_shape = [out_feat, in_feat]
        weight_data = np.random.randn(*weight_shape).astype(dtype) * 0.01
        model.graph.initializer.append(numpy_to_initializer(w_name, weight_data))
        
        if w_name not in final_inputs:
            final_inputs.append(w_name)
        
        if bias:
            bias_data = np.zeros(out_feat, dtype=dtype)
            model.graph.initializer.append(numpy_to_initializer(b_name, bias_data))
            if b_name not in final_inputs:
                final_inputs.append(b_name)
        
        attributes.setdefault("transB", 1)
    
    # Legacy: BatchNormalization with num_features
    elif op_type == "BatchNormalization" and "num_features" in kwargs:
        num_feat = kwargs.pop("num_features")
        dtype = kwargs.pop("initializer_type", np.float32)
        
        s_name = f"{name}_scale"
        b_name = f"{name}_bias"
        m_name = f"{name}_mean"
        v_name = f"{name}_var"
        
        for init_name, init_shape, init_val in [
            (s_name, [num_feat], 1.0),
            (b_name, [num_feat], 0.0),
            (m_name, [num_feat], 0.0),
            (v_name, [num_feat], 1.0),
        ]:
            init_data = np.full(init_shape, init_val, dtype=dtype)
            model.graph.initializer.append(numpy_to_initializer(init_name, init_data))
            if init_name not in final_inputs:
                final_inputs.append(init_name)
        
        attributes.setdefault("epsilon", kwargs.pop("epsilon", 1e-5))
        attributes.setdefault("momentum", kwargs.pop("momentum", 0.9))
    
    # Get operator schema for default attributes
    full_op_type = f"{domain}.{op_type}" if domain else op_type
    schema = ALL_OP_SCHEMAS.get(full_op_type)
    
    # Build attributes from schema defaults if not provided
    if schema:
        for attr_name, attr_info in schema.get("attributes", {}).items():
            if attr_name not in attributes:
                default = attr_info.get("default")
                if default is not None:
                    if isinstance(default, (list, tuple)) and len(default) == 0:
                        continue
                    attributes[attr_name] = default
    
    # Filter out empty lists
    attributes = {k: v for k, v in attributes.items() 
                  if not (isinstance(v, (list, tuple)) and len(v) == 0)}
    
    # Create ONNX node
    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=final_inputs,
        outputs=list(outputs),
        name=name,
        domain=domain,
        **attributes
    )
    
    model.graph.node.append(node)
    
    return model


def _parse_inputs(inputs: Union[str, List[str], Dict[str, Union[str, Dict]]]) -> Dict[str, Union[str, Dict]]:
    """
    Parse inputs specification into a standardized dict format.
    
    Returns:
        Dict mapping formal input names to specs:
        - "input_name" -> str: tensor from another node
        - "input_name" -> {"name": ..., "data": ...} or {"name": ..., "shape": ...}: initializer
    """
    if isinstance(inputs, str):
        # Single input - use "X" as formal name for most ops
        return {"X": inputs}
    
    if isinstance(inputs, (list, tuple)):
        # List of input names - need to map to formal names from schema
        # For simplicity, use common conventions: X, W, B for Conv; A, B, C for Gemm
        result = {}
        common_formals = ["X", "W", "B", "Z", "A", "C"]
        for i, inp in enumerate(inputs):
            if i < len(common_formals):
                result[common_formals[i]] = inp
            else:
                result[f"input_{i}"] = inp
        return result
    
    if isinstance(inputs, dict):
        # Already in spec format
        return inputs
    
    raise ValueError(f"Invalid inputs format: {type(inputs)}")


# =============================================================================
# Convenience Functions for Common Patterns
# =============================================================================

def add_conv(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    **kwargs
) -> onnx.ModelProto:
    """
    Convenience function to add a Conv layer.
    
    This is equivalent to:
        add_layer(model, "Conv", input_name, output_name, name=name,
                  in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, ...)
    """
    return add_layer(
        model, "Conv", input_name, output_name, name=name,
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=bias,
        **kwargs
    )


def add_linear(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    **kwargs
) -> onnx.ModelProto:
    """
    Convenience function to add a Linear (Gemm) layer.
    """
    return add_layer(
        model, "Gemm", input_name, output_name, name=name,
        in_features=in_features, out_features=out_features,
        bias=bias, **kwargs
    )


def add_activation(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    activation: str,
    **kwargs
) -> onnx.ModelProto:
    """
    Convenience function to add an activation layer.
    
    Supported: Relu, Sigmoid, Tanh, LeakyRelu, Elu, Selu, Softmax, 
               Gelu, HardSigmoid, etc.
    """
    # Map common aliases
    alias_map = {
        "relu": "Relu",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "leaky_relu": "LeakyRelu",
        "leakyrelu": "LeakyRelu",
        "elu": "Elu",
        "selu": "Selu",
        "softmax": "Softmax",
        "gelu": "Gelu",
        "hardsigmoid": "HardSigmoid",
        "hardswish": "HardSwish",
        "mish": "Mish",
    }
    
    op_type = alias_map.get(activation.lower(), activation)
    
    # Build attributes for activations that need them
    attributes = {}
    if op_type == "LeakyRelu" and "alpha" in kwargs:
        attributes["alpha"] = kwargs.pop("alpha")
    if op_type == "Elu" and "alpha" in kwargs:
        attributes["alpha"] = kwargs.pop("alpha")
    if op_type == "Selu":
        if "alpha" in kwargs:
            attributes["alpha"] = kwargs.pop("alpha")
        if "gamma" in kwargs:
            attributes["gamma"] = kwargs.pop("gamma")
    if op_type in ("Softmax", "LogSoftmax") and "axis" in kwargs:
        attributes["axis"] = kwargs.pop("axis")
    
    return add_layer(
        model, op_type, input_name, output_name, name=name,
        attributes=attributes if attributes else None,
        **kwargs
    )


def add_norm(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    norm_type: str,
    num_features: int,
    **kwargs
) -> onnx.ModelProto:
    """
    Convenience function to add normalization layers.
    
    norm_type: "batchnorm", "instancenorm", "layernorm", "groupnorm"
    """
    type_map = {
        "batchnorm": "BatchNormalization",
        "batch_norm": "BatchNormalization",
        "instancenorm": "InstanceNormalization",
        "instance_norm": "InstanceNormalization",
        "layernorm": "LayerNormalization",
        "layer_norm": "LayerNormalization",
        "groupnorm": "GroupNormalization",
        "group_norm": "GroupNormalization",
    }
    
    op_type = type_map.get(norm_type.lower(), norm_type)
    
    attributes = {}
    if "epsilon" in kwargs:
        attributes["epsilon"] = kwargs.pop("epsilon")
    if op_type == "GroupNormalization" and "num_groups" in kwargs:
        attributes["num_groups"] = kwargs.pop("num_groups")
    if op_type == "LayerNormalization" and "axis" in kwargs:
        attributes["axis"] = kwargs.pop("axis")
    
    return add_layer(
        model, op_type, input_name, output_name, name=name,
        num_features=num_features,
        attributes=attributes if attributes else None,
        **kwargs
    )


def add_pooling(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    pool_type: str = "max",
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, ...]] = 0,
    **kwargs
) -> onnx.ModelProto:
    """
    Convenience function to add pooling layers.
    
    pool_type: "max", "avg", "global_max", "global_avg", "lp"
    """
    type_map = {
        "max": "MaxPool",
        "avg": "AveragePool",
        "average": "AveragePool",
        "global_max": "GlobalMaxPool",
        "global_avg": "GlobalAveragePool",
        "global_average": "GlobalAveragePool",
        "lp": "LpPool",
    }
    
    op_type = type_map.get(pool_type.lower(), pool_type)
    
    # Global pooling doesn't need kernel_size
    if "Global" in op_type:
        return add_layer(
            model, op_type, input_name, output_name, name=name,
            **kwargs
        )
    
    # Regular pooling
    kernel = normalize_ints(kernel_size, 2)
    stride = normalize_ints(stride if stride is not None else kernel_size, 2)
    
    attributes = {
        "kernel_shape": kernel,
        "strides": stride,
    }
    
    if padding != 0:
        if isinstance(padding, int):
            padding = [padding] * 4
        attributes["pads"] = list(padding)
    
    if "ceil_mode" in kwargs:
        attributes["ceil_mode"] = int(kwargs.pop("ceil_mode"))
    if "count_include_pad" in kwargs:
        attributes["count_include_pad"] = int(kwargs.pop("count_include_pad"))
    if op_type == "LpPool" and "p" in kwargs:
        attributes["p"] = kwargs.pop("p")
    
    return add_layer(
        model, op_type, input_name, output_name, name=name,
        attributes=attributes,
        **kwargs
    )


def add_dropout(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    ratio: float = 0.5,
    **kwargs
) -> onnx.ModelProto:
    """Add Dropout layer."""
    ratio_name = f"{name}_ratio"
    ratio_data = np.array([ratio], dtype=np.float32)
    
    return add_layer(
        model, "Dropout", input_name, output_name, name=name,
        initializers={ratio_name: ratio_data},
        **kwargs
    )


def add_shape_manipulation(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    op_type: str,
    **kwargs
) -> onnx.ModelProto:
    """
    Add shape manipulation layers.
    
    op_type: "Flatten", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Expand"
    """
    attributes = {}
    
    if op_type == "Flatten":
        attributes["axis"] = kwargs.pop("axis", 1)
    
    elif op_type == "Reshape":
        shape = kwargs.pop("shape")
        shape_name = f"{name}_shape"
        # Use new API: dict input with initializer spec
        return add_layer(
            model, "Reshape",
            inputs={
                "data": input_name,
                "shape": {"name": shape_name, "data": np.array(shape, dtype=np.int64)}
            },
            outputs=output_name,
            name=name,
            attributes={"allowzero": kwargs.pop("allowzero", 0)},
            **kwargs
        )
    
    elif op_type == "Transpose":
        if "perm" in kwargs:
            attributes["perm"] = kwargs.pop("perm")
    
    elif op_type == "Squeeze":
        if "axes" in kwargs:
            axes_name = f"{name}_axes"
            return add_layer(
                model, "Squeeze",
                inputs={
                    "data": input_name,
                    "axes": {"name": axes_name, "data": np.array(kwargs.pop("axes"), dtype=np.int64)}
                },
                outputs=output_name,
                name=name,
                **kwargs
            )
    
    elif op_type == "Unsqueeze":
        axes = kwargs.pop("axes")
        axes_name = f"{name}_axes"
        return add_layer(
            model, "Unsqueeze",
            inputs={
                "data": input_name,
                "axes": {"name": axes_name, "data": np.array(axes, dtype=np.int64)}
            },
            outputs=output_name,
            name=name,
            **kwargs
        )
    
    return add_layer(
        model, op_type, input_name, output_name, name=name,
        attributes=attributes if attributes else None,
        **kwargs
    )


def add_fused_conv(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    activation: str = "",
    domain: str = "com.microsoft",
    **kwargs
) -> onnx.ModelProto:
    """
    Add Microsoft FusedConv layer (Conv + Activation fused).
    
    activation: "Relu", "LeakyRelu", "Tanh", "Sigmoid", etc.
    """
    return add_layer(
        model, "FusedConv", input_name, output_name,
        name=name, domain=domain,
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size,
        activation=activation,
        **kwargs
    )


def add_fused_gemm(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    in_features: int,
    out_features: int,
    activation: str = "",
    domain: str = "com.microsoft",
    **kwargs
) -> onnx.ModelProto:
    """
    Add Microsoft FusedGemm layer (Gemm + Activation fused).
    """
    return add_layer(
        model, "FusedGemm", input_name, output_name,
        name=name, domain=domain,
        in_features=in_features, out_features=out_features,
        activation=activation,
        **kwargs
    )


def add_attention(
    model: onnx.ModelProto,
    name: str,
    input_name: str,
    output_name: str,
    num_heads: int,
    domain: str = "com.microsoft",
    **kwargs
) -> onnx.ModelProto:
    """
    Add Microsoft Attention layer.
    """
    attributes = {"num_heads": num_heads}
    
    if "unidirectional" in kwargs:
        attributes["unidirectional"] = kwargs.pop("unidirectional")
    
    return add_layer(
        model, "Attention", input_name, output_name,
        name=name, domain=domain,
        attributes=attributes,
        **kwargs
    )


# =============================================================================
# ModelModifier Integration
# =============================================================================

def _add_layer_chain(
    self,
    op_type: str,
    inputs: Union[str, List[str]],
    outputs: Union[str, List[str]],
    name: Optional[str] = None,
    domain: str = "",
    **kwargs
) -> Any:
    """Chainable wrapper for add_layer."""
    from .model_modifier import copy_model
    
    self.model = add_layer(self.model, op_type, inputs, outputs, name, domain, **kwargs)
    return self


# Import this at the end to avoid circular imports
def _attach_to_model_modifier():
    """Attach add_layer method to ModelModifier class."""
    from .model_modifier import ModelModifier
    ModelModifier.add_layer = _add_layer_chain


_attach_to_model_modifier()
