#!/usr/bin/env python3
"""
ONNX Model Modification CLI Tool

Usage:
    python -m third_party.onnx_tools.cli <command> [options]

Commands:
    modify-shape    Modify tensor shape
    replace-weight  Replace initializer with numpy data
    remove-node     Remove a node
    rename-node     Rename a node
    rename-tensor   Rename a tensor
    inspect         Inspect model structure

Examples:
    python -m third_party.onnx_tools.cli modify-shape model.onnx input "1,3,320,320" -o output.onnx
    python -m third_party.onnx_tools.cli replace-weight model.onnx conv1.weight new_weights.npy -o output.onnx
    python -m third_party.onnx_tools.cli remove-node model.onnx dropout_1 -o output.onnx
    python -m third_party.onnx_tools.cli rename-tensor model.onnx input_0 image_input -o output.onnx
    python -m third_party.onnx_tools.cli inspect model.onnx
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import onnx
from third_party.onnx_tools import (
    modify_tensor_shape,
    replace_initializer,
    replace_initializer_from_file,
    remove_node,
    rename_node,
    rename_tensor,
    ModelModifier,
)


def cmd_modify_shape(args):
    """Modify tensor shape."""
    shape = [int(x) if x not in ("-1", "None", "") else None for x in args.shape.split(",")]
    
    model = onnx.load(args.input)
    model = modify_tensor_shape(model, args.tensor, shape)
    onnx.save(model, args.output)
    
    print(f"✓ Modified tensor '{args.tensor}' shape to {shape}")
    print(f"  Saved to: {args.output}")


def cmd_replace_weight(args):
    """Replace initializer with numpy data."""
    model = onnx.load(args.input)
    model = replace_initializer_from_file(model, args.initializer, args.numpy_file, args.name)
    onnx.save(model, args.output)
    
    new_name = args.name if args.name else args.initializer
    print(f"✓ Replaced initializer '{args.initializer}' with data from {args.numpy_file}")
    print(f"  New name: {new_name}")
    print(f"  Saved to: {args.output}")


def cmd_remove_node(args):
    """Remove a node."""
    model = onnx.load(args.input)
    model = remove_node(model, args.node, reconnect_inputs=not args.no_reconnect)
    onnx.save(model, args.output)
    
    print(f"✓ Removed node '{args.node}'")
    print(f"  Reconnect inputs: {not args.no_reconnect}")
    print(f"  Saved to: {args.output}")


def cmd_rename_node(args):
    """Rename a node."""
    model = onnx.load(args.input)
    model = rename_node(model, args.old_name, args.new_name)
    onnx.save(model, args.output)
    
    print(f"✓ Renamed node '{args.old_name}' -> '{args.new_name}'")
    print(f"  Saved to: {args.output}")


def cmd_rename_tensor(args):
    """Rename a tensor."""
    model = onnx.load(args.input)
    model = rename_tensor(model, args.old_name, args.new_name)
    onnx.save(model, args.output)
    
    print(f"✓ Renamed tensor '{args.old_name}' -> '{args.new_name}'")
    print(f"  Saved to: {args.output}")


def cmd_inspect(args):
    """Inspect model structure."""
    model = onnx.load(args.input)
    
    print(f"\n{'='*60}")
    print(f"Model: {args.input}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"{'='*60}")
    
    print(f"\n📥 Inputs ({len(model.graph.input)}):")
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(str(d.dim_value))
            elif d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append("?")
        print(f"  {inp.name}: [{', '.join(shape)}]")
    
    print(f"\n📤 Outputs ({len(model.graph.output)}):")
    for out in model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(str(d.dim_value))
            elif d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append("?")
        print(f"  {out.name}: [{', '.join(shape)}]")
    
    print(f"\n🗂️  Initializers ({len(model.graph.initializer)}):")
    for init in model.graph.initializer[:args.limit]:
        dtype_name = onnx.TensorProto.DataType.Name(init.data_type)
        print(f"  {init.name}: {list(init.dims)} ({dtype_name})")
    if len(model.graph.initializer) > args.limit:
        print(f"  ... and {len(model.graph.initializer) - args.limit} more")
    
    print(f"\n🔧 Nodes ({len(model.graph.node)}):")
    for node in model.graph.node[:args.limit]:
        inputs = ", ".join(node.input) if node.input else "none"
        outputs = ", ".join(node.output) if node.output else "none"
        print(f"  {node.name or '(unnamed)'}: {node.op_type}")
        print(f"    inputs: {inputs}")
        print(f"    outputs: {outputs}")
    if len(model.graph.node) > args.limit:
        print(f"  ... and {len(model.graph.node) - args.limit} more")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Model Modification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Modify input shape
  %(prog)s modify-shape model.onnx input "1,3,320,320" -o out.onnx
  
  # Replace weights
  %(prog)s replace-weight model.onnx conv1.weight weights.npy -o out.onnx
  
  # Remove dropout
  %(prog)s remove-node model.onnx dropout_1 -o out.onnx
  
  # Rename tensor
  %(prog)s rename-tensor model.onnx input_0 image -o out.onnx
  
  # Inspect model
  %(prog)s inspect model.onnx
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # modify-shape
    p = subparsers.add_parser("modify-shape", help="Modify tensor shape")
    p.add_argument("input", help="Input ONNX model")
    p.add_argument("tensor", help="Tensor name")
    p.add_argument("shape", help="New shape (comma-separated, e.g., '1,3,224,224' or '1,3,-1,-1' for dynamic)")
    p.add_argument("-o", "--output", required=True, help="Output ONNX model")
    p.set_defaults(func=cmd_modify_shape)
    
    # replace-weight
    p = subparsers.add_parser("replace-weight", help="Replace initializer with numpy data")
    p.add_argument("input", help="Input ONNX model")
    p.add_argument("initializer", help="Initializer name")
    p.add_argument("numpy_file", help="Path to .npy file")
    p.add_argument("-n", "--name", help="New name for the initializer (optional)")
    p.add_argument("-o", "--output", required=True, help="Output ONNX model")
    p.set_defaults(func=cmd_replace_weight)
    
    # remove-node
    p = subparsers.add_parser("remove-node", help="Remove a node")
    p.add_argument("input", help="Input ONNX model")
    p.add_argument("node", help="Node name")
    p.add_argument("--no-reconnect", action="store_true", help="Don't reconnect inputs to outputs")
    p.add_argument("-o", "--output", required=True, help="Output ONNX model")
    p.set_defaults(func=cmd_remove_node)
    
    # rename-node
    p = subparsers.add_parser("rename-node", help="Rename a node")
    p.add_argument("input", help="Input ONNX model")
    p.add_argument("old_name", help="Current node name")
    p.add_argument("new_name", help="New node name")
    p.add_argument("-o", "--output", required=True, help="Output ONNX model")
    p.set_defaults(func=cmd_rename_node)
    
    # rename-tensor
    p = subparsers.add_parser("rename-tensor", help="Rename a tensor")
    p.add_argument("input", help="Input ONNX model")
    p.add_argument("old_name", help="Current tensor name")
    p.add_argument("new_name", help="New tensor name")
    p.add_argument("-o", "--output", required=True, help="Output ONNX model")
    p.set_defaults(func=cmd_rename_tensor)
    
    # inspect
    p = subparsers.add_parser("inspect", help="Inspect model structure")
    p.add_argument("input", help="Input ONNX model")
    p.add_argument("-l", "--limit", type=int, default=10, help="Limit for initializers and nodes (default: 10)")
    p.set_defaults(func=cmd_inspect)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
