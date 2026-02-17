"""
Oniris Web Visualizer - Backend API

A web-based ONNX model visualizer with integrated shape inference,
simplification, and model editing capabilities.
"""

import os
import sys
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# Add project root to path (3 levels up: backend -> web -> third_party -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import onnx
import numpy as np

# Import Oniris core functionality
import oniris
# Import spatial index (handle both module and direct import paths)
try:
    from backend.spatial_index import ViewportManager, SpatialIndex
except ImportError:
    from spatial_index import ViewportManager, SpatialIndex
from third_party.onnx_tools import (
    add_layer, add_conv, add_linear, add_activation, add_norm, add_pooling,
    modify_tensor_shape, rename_tensor, rename_node, remove_node,
    replace_initializer, ModelModifier,
    ALL_OP_SCHEMAS, get_op_schema,
)

app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../frontend'
)
CORS(app)

# Increase max content length for large models (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Global model cache
model_cache: Dict[str, Any] = {}


@dataclass
class ModelInfo:
    """Model metadata"""
    ir_version: int
    opset_version: int
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    doc_string: str
    num_nodes: int
    num_initializers: int
    num_inputs: int
    num_outputs: int
    num_value_info: int
    file_size: int = 0
    file_name: str = ""
    num_edges: int = 0


@dataclass
class NodeData:
    """Node representation for visualization"""
    id: str
    op_type: str
    name: str
    domain: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    position: Dict[str, float]  # x, y for layout
    shape_info: Dict[str, List[int]]  # tensor name -> shape


def parse_attribute(attr):
    """Parse ONNX attribute to Python value"""
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings]
    elif attr.type == onnx.AttributeProto.TENSOR:
        return onnx.numpy_helper.to_array(attr.t).tolist()
    elif attr.type == onnx.AttributeProto.GRAPH:
        return "<graph>"
    else:
        return str(attr)


def get_tensor_shape_safe(tensor_type):
    """Safely extract shape from tensor type"""
    shape = []
    if tensor_type.HasField('tensor_type'):
        for dim in tensor_type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            elif dim.HasField('dim_param'):
                shape.append(dim.dim_param)  # dynamic dimension
            else:
                shape.append(-1)  # unknown
    return shape


# ONNX TensorProto data type mapping (name -> raw value)
DTYPE_TO_RAW = {
    'UNDEFINED': 0,
    'FLOAT': 1,
    'UINT8': 2,
    'INT8': 3,
    'UINT16': 4,
    'INT16': 5,
    'INT32': 6,
    'INT64': 7,
    'STRING': 8,
    'BOOL': 9,
    'FLOAT16': 10,
    'DOUBLE': 11,
    'UINT32': 12,
    'UINT64': 13,
    'COMPLEX64': 14,
    'COMPLEX128': 15,
    'BFLOAT16': 16,
    # FP8 types (ONNX 1.14+)
    'FLOAT8E4M3FN': 17,
    'FLOAT8E4M3FNUZ': 18,
    'FLOAT8E5M2': 19,
    'FLOAT8E5M2FNUZ': 20,
    # INT4 types (ONNX 1.16+)
    'UINT4': 21,
    'INT4': 22,
    # INT2 types (ONNX 1.20+)
    'UINT2': 23,
    'INT2': 24,
}


def get_dtype_info(elem_type):
    """Get dtype name and raw value from element type"""
    if elem_type == 0:
        return 'UNDEFINED', 0
    try:
        name = onnx.TensorProto.DataType.Name(elem_type)
        return name, DTYPE_TO_RAW.get(name, elem_type)
    except:
        return 'unknown', elem_type


def model_to_graph_data(model: onnx.ModelProto) -> Dict[str, Any]:
    """Convert ONNX model to graph visualization data"""
    graph = model.graph
    
    nodes = []
    edges = []
    tensors = {}  # tensor name -> info
    
    # Build tensor info map
    for input_tensor in graph.input:
        dtype_name, dtype_raw = get_dtype_info(input_tensor.type.tensor_type.elem_type) if input_tensor.type.HasField('tensor_type') else ('unknown', 0)
        tensors[input_tensor.name] = {
            'name': input_tensor.name,
            'type': 'input',
            'shape': get_tensor_shape_safe(input_tensor.type),
            'dtype': dtype_name,
            'dtype_raw': dtype_raw
        }
    
    for output_tensor in graph.output:
        dtype_name, dtype_raw = get_dtype_info(output_tensor.type.tensor_type.elem_type) if output_tensor.type.HasField('tensor_type') else ('unknown', 0)
        tensors[output_tensor.name] = {
            'name': output_tensor.name,
            'type': 'output',
            'shape': get_tensor_shape_safe(output_tensor.type),
            'dtype': dtype_name,
            'dtype_raw': dtype_raw
        }
    
    for value_info in graph.value_info:
        dtype_name, dtype_raw = get_dtype_info(value_info.type.tensor_type.elem_type) if value_info.type.HasField('tensor_type') else ('unknown', 0)
        tensors[value_info.name] = {
            'name': value_info.name,
            'type': 'intermediate',
            'shape': get_tensor_shape_safe(value_info.type),
            'dtype': dtype_name,
            'dtype_raw': dtype_raw
        }
    
    for init in graph.initializer:
        dtype_name, dtype_raw = get_dtype_info(init.data_type)
        tensor_info = tensors.get(init.name, {
            'name': init.name,
            'type': 'initializer',
            'shape': list(init.dims),
            'dtype': dtype_name,
            'dtype_raw': dtype_raw
        })
        # Ensure type is initializer
        tensor_info['type'] = 'initializer'
        tensor_info['dtype'] = dtype_name
        tensor_info['dtype_raw'] = dtype_raw
        tensor_info['shape'] = list(init.dims)
        
        # Extract values for small initializers only (<= 100 elements) to avoid huge responses
        total_elements = int(np.prod(init.dims)) if init.dims else 0
        if total_elements <= 100:
            try:
                arr = onnx.numpy_helper.to_array(init)
                tensor_info['values'] = arr.flatten().tolist()
            except Exception:
                pass
        tensors[init.name] = tensor_info
    
    # Infer dtypes for intermediate tensors from their producer nodes
    for node in graph.node:
        # Try to infer output dtype from input dtypes
        input_dtype_raw = None
        for inp in node.input:
            if inp in tensors and tensors[inp].get('dtype_raw') is not None:
                input_dtype_raw = tensors[inp]['dtype_raw']
                break
        
        # Use most common input dtype for outputs, default to FLOAT (1)
        inferred_dtype_raw = input_dtype_raw if input_dtype_raw is not None else 1
        inferred_dtype_name = DTYPE_TO_RAW.get(inferred_dtype_raw, 'unknown')
        
        for out in node.output:
            if out not in tensors:
                tensors[out] = {
                    'name': out,
                    'type': 'intermediate',
                    'shape': [],  # Will be filled from shape_info
                    'dtype': inferred_dtype_name,
                    'dtype_raw': inferred_dtype_raw
                }
    
    # Pre-compute output -> producer mapping for efficient edge building
    output_to_producer = {}
    for idx, node in enumerate(graph.node):
        for out in node.output:
            output_to_producer[out] = idx
    
    # Pre-compute input -> consumer mapping
    input_to_consumers = {}
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            if inp not in input_to_consumers:
                input_to_consumers[inp] = []
            input_to_consumers[inp].append(idx)
    
    # Build nodes
    for idx, node in enumerate(graph.node):
        node_id = f"node_{idx}"
        
        # Get shape info for this node's outputs
        shape_info = {}
        for out in node.output:
            if out in tensors:
                shape_info[out] = tensors[out]['shape']
        
        # Limit attributes for large models to reduce payload size
        attributes = {}
        if len(graph.node) > 1000:
            # For very large models, only keep essential attributes
            for attr in node.attribute:
                if attr.name in ('kernel_shape', 'strides', 'pads', 'dilations', 
                                'group', 'axis', 'perm', 'shape', 'to', 'alpha'):
                    try:
                        attributes[attr.name] = parse_attribute(attr)
                    except:
                        pass
        else:
            attributes = {attr.name: parse_attribute(attr) for attr in node.attribute}
        
        node_data = {
            'id': node_id,
            'op_type': node.op_type,
            'name': node.name or f"{node.op_type}_{idx}",
            'domain': node.domain,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'attributes': attributes,
            'shape_info': shape_info
        }
        nodes.append(node_data)
        
        # Build edges using pre-computed mapping (O(n) instead of O(n^2))
        for out_idx, output in enumerate(node.output):
            consumers = input_to_consumers.get(output, [])
            for target_idx in consumers:
                target_node = graph.node[target_idx]
                try:
                    target_input_idx = list(target_node.input).index(output)
                except ValueError:
                    target_input_idx = 0
                
                target_id = f"node_{target_idx}"
                edges.append({
                    'id': f"edge_{idx}_{target_idx}_{output}",
                    'source': node_id,
                    'target': target_id,
                    'tensor': output,
                    'shape': tensors.get(output, {}).get('shape', []),
                    'source_output_idx': out_idx,
                    'target_input_idx': target_input_idx
                })
    
    # Add virtual nodes for model inputs and outputs
    node_idx = len(nodes)
    
    # Model inputs - create virtual input nodes
    for input_tensor in graph.input:
        input_name = input_tensor.name
        tensor_info = tensors.get(input_name, {})
        
        # Create virtual input node
        input_node_id = f"input_{input_name}"
        input_node = {
            'id': input_node_id,
            'op_type': 'Input',
            'name': input_name,
            'domain': '',
            'inputs': [],
            'outputs': [input_name],
            'attributes': {},
            'shape_info': {input_name: tensor_info.get('shape', [])},
            'is_virtual': True,
            'virtual_type': 'input'
        }
        nodes.append(input_node)
        
        # Create edges from input node to consumers
        consumers = input_to_consumers.get(input_name, [])
        for target_idx in consumers:
            target_node = graph.node[target_idx]
            try:
                target_input_idx = list(target_node.input).index(input_name)
            except ValueError:
                target_input_idx = 0
            
            target_id = f"node_{target_idx}"
            edges.append({
                'id': f"edge_input_{input_name}_{target_idx}",
                'source': input_node_id,
                'target': target_id,
                'tensor': input_name,
                'shape': tensor_info.get('shape', []),
                'source_output_idx': 0,
                'target_input_idx': target_input_idx
            })
    
    # Model outputs - create virtual output nodes
    for output_tensor in graph.output:
        output_name = output_tensor.name
        tensor_info = tensors.get(output_name, {})
        
        # Find producer of this output
        producer_idx = output_to_producer.get(output_name)
        
        # Create virtual output node
        output_node_id = f"output_{output_name}"
        output_node = {
            'id': output_node_id,
            'op_type': 'Output',
            'name': output_name,
            'domain': '',
            'inputs': [output_name],
            'outputs': [],
            'attributes': {},
            'shape_info': {output_name: tensor_info.get('shape', [])},
            'is_virtual': True,
            'virtual_type': 'output'
        }
        nodes.append(output_node)
        
        # Create edge from producer to output node
        if producer_idx is not None:
            source_id = f"node_{producer_idx}"
            source_node = graph.node[producer_idx]
            try:
                source_output_idx = list(source_node.output).index(output_name)
            except ValueError:
                source_output_idx = 0
            
            edges.append({
                'id': f"edge_output_{producer_idx}_{output_name}",
                'source': source_id,
                'target': output_node_id,
                'tensor': output_name,
                'shape': tensor_info.get('shape', []),
                'source_output_idx': source_output_idx,
                'target_input_idx': 0
            })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'tensors': tensors,
        'ir_version': model.ir_version,
        'opset': {op.domain: op.version for op in model.opset_import}
    }


def compute_layout(graph_data: Dict[str, Any], layout_type: str = 'hierarchical') -> Dict[str, Any]:
    """
    Compute node positions for visualization.
    
    Supports: 'hierarchical', 'force', 'grid'
    """
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    
    if layout_type == 'hierarchical':
        result = compute_hierarchical_layout(nodes, edges)
    elif layout_type == 'force':
        result = compute_force_layout(nodes, edges)
    else:
        result = compute_grid_layout(nodes, edges)
    
    # Preserve additional fields from original graph_data
    result['tensors'] = graph_data.get('tensors', {})
    result['ir_version'] = graph_data.get('ir_version', 0)
    result['opset'] = graph_data.get('opset', {})
    
    return result


def compute_hierarchical_layout(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    """
    Netron-style hierarchical layout with orthogonal edge routing.
    Nodes are arranged in vertical layers, edges use Manhattan routing with proper spacing.
    """
    node_map = {n['id']: n for n in nodes}
    
    # Build adjacency
    incoming = {n['id']: [] for n in nodes}
    outgoing = {n['id']: [] for n in nodes}
    for edge in edges:
        outgoing[edge['source']].append(edge['target'])
        incoming[edge['target']].append(edge['source'])
    
    # Identify virtual nodes
    virtual_inputs = [n['id'] for n in nodes if n.get('virtual_type') == 'input']
    virtual_outputs = [n['id'] for n in nodes if n.get('virtual_type') == 'output']
    regular_nodes = [n['id'] for n in nodes if not n.get('is_virtual')]
    
    # Compute levels using topological sort
    levels = {}
    visited = set()
    
    def compute_level(node_id):
        if node_id in levels:
            return levels[node_id]
        if node_id in visited:
            return 0
        visited.add(node_id)
        
        if not incoming[node_id]:
            levels[node_id] = 0
        else:
            max_parent_level = max(compute_level(p) for p in incoming[node_id])
            levels[node_id] = max_parent_level + 1
        return levels[node_id]
    
    for node in nodes:
        compute_level(node['id'])
    
    # Adjust levels: input at 0, regular at 1+, output at max+2
    max_regular_level = max((levels[nid] for nid in regular_nodes), default=0)
    for node_id in virtual_inputs:
        levels[node_id] = 0
    for node_id in regular_nodes:
        levels[node_id] = levels[node_id] + 1
    for node_id in virtual_outputs:
        levels[node_id] = max_regular_level + 2
    
    # Group nodes by level
    level_groups = {}
    for node_id, level in levels.items():
        level_groups.setdefault(level, []).append(node_id)
    
    # Layout constants (Netron-style)
    # Netron uses dagre with specific settings:
    # - nodesep: 20 (horizontal separation)
    # - ranksep: 80 (vertical separation between ranks)
    # - Node sizes are measured dynamically, but defaults are:
    NODE_WIDTH = 160     # Default node width
    NODE_HEIGHT = 40     # Header-only nodes are compact
    LEVEL_SPACING = 80   # dagre ranksep
    NODE_SPACING = 20    # dagre nodesep
    EDGE_LANE_SPACING = 10
    
    # Assign positions - minimize edge crossings
    positions = {}
    
    # Process levels from top to bottom
    sorted_levels = sorted(level_groups.keys())
    
    for level_idx, level in enumerate(sorted_levels):
        node_ids = level_groups[level]
        
        # Sort nodes to minimize crossings
        # Place nodes with connections to previous level near their parents
        if level_idx > 0:
            prev_level_nodes = level_groups.get(sorted_levels[level_idx - 1], [])
            
            def sort_key(nid):
                node = node_map[nid]
                # Get average x of parents
                parents = incoming[nid]
                if parents:
                    avg_parent_x = sum(positions[p]['x'] for p in parents if p in positions) / len(parents)
                    return (0, avg_parent_x, node['name'])
                return (1, 0, node['name'])
            
            node_ids.sort(key=sort_key)
        else:
            # First level - sort by name
            node_ids.sort(key=lambda nid: (node_map[nid].get('virtual_type', ''), node_map[nid]['name']))
        
        # Update level_groups with sorted order
        level_groups[level] = node_ids
        
        # Calculate positions
        total_width = len(node_ids) * NODE_WIDTH + (len(node_ids) - 1) * NODE_SPACING
        start_x = -total_width / 2
        
        for idx, node_id in enumerate(node_ids):
            positions[node_id] = {
                'x': start_x + idx * (NODE_WIDTH + NODE_SPACING) + NODE_WIDTH / 2,
                'y': level_idx * LEVEL_SPACING
            }
    
    # Calculate edge routing info to avoid overlaps
    # Group edges by source level and track lane assignments
    edge_lanes = {}  # (source_level, target_level) -> list of edge indices
    
    for i, edge in enumerate(edges):
        source_level = levels.get(edge['source'], 0)
        target_level = levels.get(edge['target'], 0)
        key = (min(source_level, target_level), max(source_level, target_level))
        if key not in edge_lanes:
            edge_lanes[key] = []
        edge_lanes[key].append(i)
    
    # Assign lane positions to edges
    for edge in edges:
        source_level = levels.get(edge['source'], 0)
        target_level = levels.get(edge['target'], 0)
        key = (min(source_level, target_level), max(source_level, target_level))
        
        # Find index within this lane group
        lane_group = edge_lanes.get(key, [])
        if len(lane_group) > 1:
            lane_idx = lane_group.index(edges.index(edge))
            # Store lane info for frontend routing
            edge['_lane_idx'] = lane_idx
            edge['_total_lanes'] = len(lane_group)
            edge['_lane_offset'] = (lane_idx - (len(lane_group) - 1) / 2) * EDGE_LANE_SPACING
        else:
            edge['_lane_idx'] = 0
            edge['_total_lanes'] = 1
            edge['_lane_offset'] = 0
    
    # Update nodes with positions and dimensions
    for node in nodes:
        node['position'] = positions.get(node['id'], {'x': 0, 'y': 0})
        node['_width'] = NODE_WIDTH
        node['_height'] = NODE_HEIGHT
    
    return {'nodes': nodes, 'edges': edges}


def compute_force_layout(nodes: List[Dict], edges: List[Dict], iterations: int = 100) -> Dict[str, Any]:
    """Force-directed layout for complex graphs"""
    import random
    import math
    
    # Initialize random positions
    positions = {}
    for node in nodes:
        positions[node['id']] = {
            'x': random.uniform(-500, 500),
            'y': random.uniform(-500, 500)
        }
    
    # Simple force simulation
    for _ in range(iterations):
        # Repulsive forces
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                p1 = positions[n1['id']]
                p2 = positions[n2['id']]
                dx = p1['x'] - p2['x']
                dy = p1['y'] - p2['y']
                dist = math.sqrt(dx*dx + dy*dy) + 0.01
                force = 5000 / (dist * dist)
                fx = force * dx / dist
                fy = force * dy / dist
                p1['x'] += fx
                p1['y'] += fy
                p2['x'] -= fx
                p2['y'] -= fy
        
        # Attractive forces for edges
        for edge in edges:
            p1 = positions[edge['source']]
            p2 = positions[edge['target']]
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            dist = math.sqrt(dx*dx + dy*dy) + 0.01
            force = dist * dist / 5000
            fx = force * dx / dist
            fy = force * dy / dist
            p1['x'] += fx
            p1['y'] += fy
            p2['x'] -= fx
            p2['y'] -= fy
    
    for node in nodes:
        node['position'] = positions[node['id']]
    
    return {'nodes': nodes, 'edges': edges}


def compute_grid_layout(nodes: List[Dict], edges: List[Dict] = None) -> Dict[str, Any]:
    """Simple grid layout"""
    cols = int(len(nodes) ** 0.5) + 1
    spacing = 200
    
    for idx, node in enumerate(nodes):
        row = idx // cols
        col = idx % cols
        node['position'] = {
            'x': col * spacing,
            'y': row * spacing
        }
    
    return {'nodes': nodes, 'edges': edges or []}


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    """Main visualization page"""
    return render_template('index.html')


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NaN and Infinity"""
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return 'NaN'
            if math.isinf(obj):
                return 'Infinity' if obj > 0 else '-Infinity'
        return super().default(obj)


@app.route('/api/model/upload', methods=['POST'])
def upload_model():
    """Upload and parse ONNX model"""
    import traceback
    import tempfile
    import os
    
    print(f"[{datetime.now()}] Upload request received")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Check file extension
    if not file.filename.endswith('.onnx'):
        return jsonify({'error': 'File must be an .onnx file'}), 400
    
    temp_path = None
    
    try:
        # For large models, save to temp file instead of memory
        file_size = 0
        
        # Try to get file size from request
        try:
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
        except:
            pass
        
        print(f"[{datetime.now()}] File size: {file_size} bytes")
        
        # If file is large (>50MB), use temp file
        if file_size > 50 * 1024 * 1024:
            print(f"Large model detected ({file_size / 1024 / 1024:.1f}MB), using temp file")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as tmp:
                file.save(tmp)
                temp_path = tmp.name
            
            # Load from temp file
            model = onnx.load(temp_path)
        else:
            # Read smaller models into memory
            model_bytes = file.read()
            file_size = len(model_bytes)
            model = onnx.load_model_from_string(model_bytes)
        
        print(f"[{datetime.now()}] Model loaded successfully")
        
        # Validate model
        try:
            onnx.checker.check_model(model)
        except Exception as validation_error:
            print(f"Model validation warning: {validation_error}")
            # Continue anyway, as some models may have minor issues
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Compute layout immediately for visualization
        print(f"[{datetime.now()}] Computing hierarchical layout...")
        graph_data = compute_layout(graph_data, 'hierarchical')
        print(f"[{datetime.now()}] Layout complete: {len(graph_data['nodes'])} nodes with positions")
        
        # Create viewport manager for large models
        viewport_manager = None
        if len(graph_data['nodes']) > 500:
            print(f"[{datetime.now()}] Large model ({len(graph_data['nodes'])} nodes), creating viewport manager...")
            viewport_manager = ViewportManager(graph_data)
            print(f"[{datetime.now()}] Viewport manager created")
        
        # Cache model (store path for large models)
        cache_entry = {
            'model': model,
            'filename': file.filename,
            'is_large': file_size > 50 * 1024 * 1024,
            'viewport_manager': viewport_manager,
            'full_graph_data': graph_data if len(graph_data['nodes']) <= 500 else None
        }
        
        if temp_path:
            cache_entry['temp_path'] = temp_path
        else:
            cache_entry['original_bytes'] = model_bytes
            
        model_cache[session_id] = cache_entry
        
        # Get model info
        info = ModelInfo(
            ir_version=model.ir_version,
            opset_version=model.opset_import[0].version if model.opset_import else 0,
            producer_name=model.producer_name,
            producer_version=model.producer_version,
            domain=model.domain,
            model_version=model.model_version,
            doc_string=model.doc_string,
            num_nodes=len(graph.node),
            num_initializers=len(graph.initializer),
            num_inputs=len(graph.input),
            num_outputs=len(graph.output),
            num_value_info=len(graph.value_info),
            file_size=file_size,
            file_name=file.filename,
            num_edges=len(graph_data['edges'])
        )
        
        # Build response - use simple jsonify which is more reliable
        print(f"[{datetime.now()}] Building response...")
        
        # Simplify graph_data to reduce size if needed
        if len(graph_data.get('nodes', [])) > 100:
            print(f"[{datetime.now()}] Large model detected, keeping essential data only")
        
        response_data = {
            'success': True,
            'session_id': session_id,
            'info': {
                'ir_version': model.ir_version,
                'opset_version': model.opset_import[0].version if model.opset_import else 0,
                'producer_name': model.producer_name,
                'producer_version': model.producer_version,
                'domain': model.domain,
                'model_version': model.model_version,
                'doc_string': model.doc_string,
                'num_nodes': len(graph.node),
                'num_initializers': len(graph.initializer),
                'num_inputs': len(graph.input),
                'num_outputs': len(graph.output),
                'num_value_info': len(graph.value_info),
                'file_size': file_size,
                'file_name': file.filename,
                'num_edges': len(graph_data['edges'])
            },
            'graph': graph_data
        }
        
        print(f"[{datetime.now()}] Serializing to JSON...")
        try:
            json_bytes = json.dumps(response_data, cls=JSONEncoder).encode('utf-8')
            print(f"[{datetime.now()}] JSON bytes: {len(json_bytes)}")
        except Exception as json_err:
            print(f"[{datetime.now()}] JSON serialization failed: {json_err}")
            raise
        
        print(f"[{datetime.now()}] Creating response object...")
        from flask import Response
        response = Response(
            json_bytes,
            status=200,
            content_type='application/json; charset=utf-8'
        )
        response.headers['Content-Length'] = len(json_bytes)
        print(f"[{datetime.now()}] Response ready, returning...")
        return response
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error uploading model: {error_msg}")
        print(error_trace)
        
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Return more detailed error message
        return jsonify({
            'error': error_msg,
            'type': type(e).__name__,
            'traceback': error_trace if app.debug else None
        }), 500


@app.route('/api/model/<session_id>/layout', methods=['POST'])
def compute_layout_endpoint(session_id):
    """Compute layout for the graph"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json() or {}
    layout_type = data.get('layout', 'hierarchical')
    
    try:
        model = model_cache[session_id]['model']
        graph_data = model_to_graph_data(model)
        
        # Compute layout
        layout_data = compute_layout(graph_data, layout_type)
        
        return jsonify({
            'success': True,
            'layout': layout_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/shape_inference', methods=['POST'])
def run_shape_inference(session_id):
    """Run shape inference on the model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        model = model_cache[session_id]['model']
        
        # Run shape inference
        inferred_model = onnx.shape_inference.infer_shapes(model)
        model_cache[session_id]['model'] = inferred_model
        
        # Return updated graph
        graph_data = model_to_graph_data(inferred_model)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': 'Shape inference completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/simplify', methods=['POST'])
def simplify_model(session_id):
    """Simplify the model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        model = model_cache[session_id]['model']
        
        # Use Oniris simplifier if available, otherwise use onnxsim
        try:
            from third_party.onnx_tools.simplify import simplify_model as onnxsim
            simplified, check = onnxsim(model)
            if not check:
                return jsonify({'error': 'Simplified model check failed'}), 500
            model_cache[session_id]['model'] = simplified
        except ImportError:
            # Fallback: try onnx-simplifier
            try:
                import onnxsim
                simplified, check = onnxsim.simplify(model)
                if not check:
                    return jsonify({'error': 'Simplified model check failed'}), 500
                model_cache[session_id]['model'] = simplified
            except ImportError:
                return jsonify({'error': 'onnx-simplifier not installed'}), 500
        
        # Return updated graph
        graph_data = model_to_graph_data(model_cache[session_id]['model'])
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': 'Model simplified successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/viewport', methods=['POST'])
def get_viewport_nodes(session_id):
    """
    Get nodes and edges within a viewport.
    For large models, only returns visible nodes to improve performance.
    """
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        data = request.get_json() or {}
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 1000)
        height = data.get('height', 800)
        
        cache_entry = model_cache[session_id]
        viewport_manager = cache_entry.get('viewport_manager')
        
        if viewport_manager:
            # Large model: use viewport manager to get visible nodes
            result = viewport_manager.update_viewport(x, y, width, height)
            return jsonify({
                'success': True,
                'viewport': result['viewport'],
                'bounds': result['bounds'],
                'total_nodes': result['total_nodes'],
                'total_edges': result['total_edges'],
                'visible_nodes': result['visible_nodes'],
                'visible_edges': result['visible_edges'],
                'graph': {
                    'nodes': result['nodes'],
                    'edges': result['edges'],
                    'tensors': result['tensors']
                }
            })
        else:
            # Small model: return full graph
            graph_data = cache_entry.get('full_graph_data')
            if not graph_data:
                # Fallback: regenerate graph data
                model = cache_entry['model']
                graph_data = model_to_graph_data(model)
                graph_data = compute_layout(graph_data, 'hierarchical')
            
            return jsonify({
                'success': True,
                'viewport': {'x': x, 'y': y, 'width': width, 'height': height},
                'total_nodes': len(graph_data['nodes']),
                'total_edges': len(graph_data['edges']),
                'visible_nodes': len(graph_data['nodes']),
                'visible_edges': len(graph_data['edges']),
                'graph': graph_data
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/export', methods=['GET'])
def export_model(session_id):
    """Export modified model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        model = model_cache[session_id]['model']
        filename = model_cache[session_id].get('filename', 'model.onnx')
        
        # Save to temp file
        temp_dir = tempfile.gettempdir()
        export_path = os.path.join(temp_dir, f"{session_id}_{filename}")
        onnx.save(model, export_path)
        
        return send_from_directory(temp_dir, f"{session_id}_{filename}",
                                   as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ops/schemas', methods=['GET'])
def get_op_schemas():
    """Get all operator schemas"""
    schemas = []
    for name, schema in ALL_OP_SCHEMAS.items():
        schemas.append({
            'name': name,
            'inputs': schema.get('inputs', []),
            'outputs': schema.get('outputs', []),
            'attributes': list(schema.get('attributes', {}).keys())
        })
    return jsonify({'schemas': schemas})


@app.route('/api/ops/<op_name>/schema', methods=['GET'])
def get_single_op_schema(op_name):
    """Get schema for specific operator"""
    domain = request.args.get('domain', '')
    schema = get_op_schema(op_name, domain)
    if schema:
        return jsonify(schema)
    return jsonify({'error': 'Schema not found'}), 404


@app.route('/api/model/<session_id>/add_layer', methods=['POST'])
def api_add_layer(session_id):
    """Add a layer to the model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        model = model_cache[session_id]['model']
        
        op_type = data.get('op_type')
        inputs = data.get('inputs', {})
        outputs = data.get('outputs', [])
        name = data.get('name')
        domain = data.get('domain', '')
        attributes = data.get('attributes', {})
        
        # Convert inputs format if needed
        if isinstance(inputs, dict):
            # Convert frontend format to backend format
            formatted_inputs = {}
            for key, val in inputs.items():
                if isinstance(val, str):
                    formatted_inputs[key] = val
                elif isinstance(val, dict):
                    if 'data' in val:
                        formatted_inputs[key] = {
                            'name': val.get('name', f"{name}_{key}"),
                            'data': np.array(val['data'])
                        }
                    elif 'shape' in val:
                        formatted_inputs[key] = {
                            'name': val.get('name', f"{name}_{key}"),
                            'shape': val['shape'],
                            'type': getattr(np, val.get('type', 'float32'))
                        }
            inputs = formatted_inputs
        
        # Add layer
        modified = add_layer(
            model, op_type, inputs, outputs,
            name=name, domain=domain, attributes=attributes
        )
        
        model_cache[session_id]['model'] = modified
        
        # Return updated graph
        graph_data = model_to_graph_data(modified)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Layer {name} ({op_type}) added successfully'
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/model/<session_id>/modify_shape', methods=['POST'])
def api_modify_shape(session_id):
    """Modify tensor shape"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    tensor_name = data.get('tensor_name')
    new_shape = data.get('new_shape')
    
    try:
        model = model_cache[session_id]['model']
        modified = modify_tensor_shape(model, tensor_name, new_shape)
        model_cache[session_id]['model'] = modified
        
        graph_data = model_to_graph_data(modified)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Shape of {tensor_name} modified to {new_shape}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/modify_dtype', methods=['POST'])
def api_modify_dtype(session_id):
    """Modify tensor data type"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    tensor_name = data.get('tensor_name')
    new_dtype = data.get('new_dtype')
    
    if not tensor_name or new_dtype is None:
        return jsonify({'error': 'tensor_name and new_dtype are required'}), 400
    
    try:
        model = model_cache[session_id]['model']
        graph = model.graph
        
        # Find the tensor in value_info, input, or output
        target_tensor = None
        tensor_list = None
        
        # Check inputs
        for inp in graph.input:
            if inp.name == tensor_name:
                target_tensor = inp
                tensor_list = 'input'
                break
        
        # Check outputs
        if not target_tensor:
            for out in graph.output:
                if out.name == tensor_name:
                    target_tensor = out
                    tensor_list = 'output'
                    break
        
        # Check value_info
        if not target_tensor:
            for vi in graph.value_info:
                if vi.name == tensor_name:
                    target_tensor = vi
                    tensor_list = 'value_info'
                    break
        
        if not target_tensor:
            return jsonify({'error': f'Tensor {tensor_name} not found'}), 404
        
        # Convert dtype name to raw value
        if isinstance(new_dtype, str):
            dtype_raw = DTYPE_TO_RAW.get(new_dtype.upper(), 1)
        else:
            dtype_raw = int(new_dtype)
        
        # Update the tensor type
        if target_tensor.type.HasField('tensor_type'):
            target_tensor.type.tensor_type.elem_type = dtype_raw
        
        # Update model in cache
        model_cache[session_id]['model'] = model
        
        graph_data = model_to_graph_data(model)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Dtype of {tensor_name} modified to {new_dtype}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/remove_node', methods=['POST'])
def api_remove_node(session_id):
    """Remove a node from the model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    node_name = data.get('node_name')
    
    try:
        model = model_cache[session_id]['model']
        modified = remove_node(model, node_name, reconnect_inputs=True)
        model_cache[session_id]['model'] = modified
        
        graph_data = model_to_graph_data(modified)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Node {node_name} removed'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/graph', methods=['GET'])
def get_graph(session_id):
    """Get current graph data"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        model = model_cache[session_id]['model']
        graph_data = model_to_graph_data(model)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/rename_node', methods=['POST'])
def api_rename_node(session_id):
    """Rename a node in the model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    node_id = data.get('node_id')
    new_name = data.get('new_name')
    
    if not node_id or not new_name:
        return jsonify({'error': 'node_id and new_name required'}), 400
    
    try:
        model = model_cache[session_id]['model']
        
        # Find the actual node name from the id
        target_node = None
        for node in model.graph.node:
            node_id_str = f"node_{list(model.graph.node).index(node)}"
            if node_id_str == node_id or node.name == node_id:
                target_node = node
                break
        
        if not target_node:
            return jsonify({'error': 'Node not found'}), 404
        
        modified = rename_node(model, target_node.name, new_name)
        model_cache[session_id]['model'] = modified
        
        graph_data = model_to_graph_data(modified)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Node renamed to {new_name}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/rename_tensor', methods=['POST'])
def api_rename_tensor(session_id):
    """Rename a tensor in the model"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    
    if not old_name or not new_name:
        return jsonify({'error': 'old_name and new_name required'}), 400
    
    try:
        model = model_cache[session_id]['model']
        modified = rename_tensor(model, old_name, new_name)
        model_cache[session_id]['model'] = modified
        
        graph_data = model_to_graph_data(modified)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Tensor renamed from {old_name} to {new_name}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<session_id>/replace_initializer', methods=['POST'])
def api_replace_initializer(session_id):
    """Replace an initializer with numpy data"""
    if session_id not in model_cache:
        return jsonify({'error': 'Session not found'}), 404
    
    tensor_name = request.form.get('tensor_name')
    numpy_file = request.files.get('numpy_data')
    
    if not tensor_name or not numpy_file:
        return jsonify({'error': 'tensor_name and numpy_data required'}), 400
    
    try:
        import io
        # Read numpy data
        numpy_bytes = numpy_file.read()
        
        # Load numpy array
        import numpy as np
        np_array = np.load(io.BytesIO(numpy_bytes))
        
        model = model_cache[session_id]['model']
        modified = replace_initializer(model, tensor_name, np_array)
        model_cache[session_id]['model'] = modified
        
        graph_data = model_to_graph_data(modified)
        layout_data = compute_layout(graph_data, 'hierarchical')
        
        return jsonify({
            'success': True,
            'graph': layout_data,
            'message': f'Initializer {tensor_name} replaced'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
