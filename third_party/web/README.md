# Oniris Web Visualizer

A web-based ONNX model visualizer with integrated shape inference, simplification, and model editing capabilities.

## Features

### Visualization (Direct Netron Code Integration)
- 🎨 **Uses Actual Netron Code** - Directly uses Netron's grapher.js
- 📐 **Netron's Graph Class** - `grapher.Graph` with dagre layout
- 🎯 **Netron's Node/Edge Classes** - `grapher.Node`, `grapher.Edge`
- 🔧 **Exact Rendering** - Same code paths as Netron
- 🔍 **Netron Interactions** - Same pan/zoom behavior

### Recent Fixes
- Fixed dagre.js module loading (removed ES6 exports)
- Fixed static file serving (Flask static_url_path)
- Fixed render order (build -> measure -> layout -> update)

### Model Operations
- 📐 **Shape Inference** - Infer tensor shapes throughout the graph
- ✨ **Model Simplification** - Simplify models by removing redundant operations
- ➕ **Add Layers** - Add new operators with visual editor
- ➖ **Remove Nodes** - Delete nodes with automatic connection rewiring
- 📝 **Modify Shapes** - Edit tensor shapes interactively
- 💾 **Export Models** - Download modified models

### Performance
- ⚡ **Optimized for Large Models** - Viewport-based rendering for LLM-scale models (>500 nodes)
- 🚀 **Efficient SVG Rendering** - Hardware-accelerated graphics
- 📊 **Lazy Loading** - Only render visible nodes in large graphs

## Quick Start

### Prerequisites

The web visualizer requires Python 3.8+ with the following packages:
- flask >= 2.3.0
- flask-cors >= 4.0.0
- onnx >= 1.15.0
- numpy >= 1.24.0

### Using the Convenience Script (Recommended)

From the project root directory:

```bash
# Install dependencies first
pip install -r third_party/web/requirements.txt

# Start development server
./scripts/start_web.sh

# If automatic dependency installation fails, install manually:
pip install flask flask-cors onnx numpy
./scripts/start_web.sh --skip-deps

# Production mode
./scripts/start_web.sh --prod

# Custom port
./scripts/start_web.sh --port 8080

# Show all options
./scripts/start_web.sh --help
```

### Manual Start

```bash
cd third_party/web
pip install -r requirements.txt
python start_server.py
```

### Troubleshooting

**Flask not found error:**
```bash
# Install manually
pip install flask flask-cors

# Then run with --skip-deps
./scripts/start_web.sh --skip-deps
```

**Port already in use:**
```bash
# Use a different port
./scripts/start_web.sh --port 8080
```

**Model upload fails / web freezes:**

If you're uploading large models from ONNX Model Zoo:

1. **Check model size** - Models larger than 500MB may fail (Flask limit)
   ```bash
   # Check file size
   ls -lh model.onnx
   ```

2. **Use temp file mode** - Large models (>50MB) are automatically saved to temp files

3. **Check browser console** - Press F12 to see detailed error messages

4. **Test with smaller model first** - Try a small model (e.g., MNIST) to verify setup

**Common errors:**

- `413 Request Entity Too Large` - Model exceeds 500MB limit
- `Connection timeout` - Model is too large for your network
- `Memory error` - Server ran out of memory processing the model

**Solutions:**

```bash
# For very large models, try:
1. Use the Python API directly instead of web interface
2. Simplify the model first: python -c "import oniris; oniris.simplify('big.onnx', 'small.onnx')"
3. Increase system memory or use a machine with more RAM
```

### 3. Open Browser

Navigate to `http://localhost:5000`

### 4. Upload Model

Click "Upload Model" and select an `.onnx` file

## Architecture

```
third_party/web/
├── backend/
│   └── app.py           # Flask API server
├── frontend/
│   ├── css/
│   │   └── style.css    # UI styles
│   └── js/
│       └── app.js       # Frontend application
├── templates/
│   └── index.html       # Main page
├── requirements.txt     # Python dependencies
└── start_server.py      # Server startup script
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main visualization page |
| `/api/model/upload` | POST | Upload ONNX model |
| `/api/model/<id>/layout` | POST | Compute graph layout |
| `/api/model/<id>/shape_inference` | POST | Run shape inference |
| `/api/model/<id>/simplify` | POST | Simplify model |
| `/api/model/<id>/export` | GET | Export modified model |
| `/api/model/<id>/add_layer` | POST | Add a layer |
| `/api/model/<id>/remove_node` | POST | Remove a node |
| `/api/ops/schemas` | GET | List operator schemas |

## Usage Examples

### Visualize a Model

1. Click "Upload Model" button
2. Select your `.onnx` file
3. The graph will be displayed with nodes and edges

### Run Shape Inference

1. Click "Shape Inference" in the left sidebar
2. Tensor shapes will be updated throughout the graph

### Add a Layer

1. Click "Add Layer" button
2. Select operator type (e.g., Conv, Relu)
3. Configure inputs/outputs
4. Click "Add Layer"

### Remove a Node

1. Click on a node in the graph
2. Click "Remove Node" in the properties panel
3. The node will be removed and connections rewired

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Delete` | Remove selected node |
| `Ctrl + 0` | Fit to screen |
| `Ctrl + +` | Zoom in |
| `Ctrl + -` | Zoom out |

## Supported Operators

The visualizer supports all 164 ONNX and Microsoft domain operators including:

- **Neural Network**: Conv, Gemm, BatchNormalization, etc.
- **Activations**: Relu, Sigmoid, Tanh, Gelu, etc.
- **Pooling**: MaxPool, AveragePool, GlobalPool, etc.
- **Shape**: Flatten, Reshape, Transpose, etc.
- **Math**: Add, Mul, MatMul, etc.
- **Microsoft Fused**: FusedConv, FusedGemm, Attention, etc.

## Development

### Frontend Development

The frontend uses vanilla JavaScript with SVG for rendering:

```javascript
// Add a layer programmatically
api.addLayer(sessionId, {
    op_type: 'Conv',
    inputs: { X: 'input', W: { shape: [64, 3, 3, 3] }, B: { shape: [64] } },
    outputs: ['output'],
    attributes: { kernel_shape: [3, 3] }
});
```

### Backend Development

The backend is a Flask server with REST API:

```python
from third_party.web.backend.app import app, model_to_graph_data

# Convert model to graph data
graph_data = model_to_graph_data(model)
```

## Netron-Style Features

This visualizer replicates Netron's visualization approach:

### Node Styling
- **Color-coded headers** by operator category:
  - 🔵 Layer operations (Conv, Gemm, etc.)
  - 🔴 Activation functions (ReLU, Sigmoid, etc.)
  - 🟢 Pooling operations (MaxPool, AveragePool, etc.)
  - 🟣 Normalization (BatchNorm, LayerNorm, etc.)
  - 🟡 Shape operations (Reshape, Transpose, etc.)
  - ⚪ Inputs/Outputs

- **Compact header-only design** - Clean, minimal node appearance
- **Rounded corners** - 5px border radius

### Edge Styling
- **Cubic bezier curves** - Smooth S-curves between nodes
- **Arrow markers** - Clear direction indicators
- **Hit testing** - Wide invisible hit area for easy selection
- **Shape labels** - Tensor dimensions displayed along edges

### Interactions
- **Mouse wheel** - Pan vertically
- **Ctrl/Cmd + wheel** - Zoom
- **Click + drag** - Pan
- **Click node/edge** - Select and view properties

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This is part of the Oniris project.
