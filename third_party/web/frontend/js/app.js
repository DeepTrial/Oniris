/**
 * Oniris Web Visualizer - Direct Netron Integration
 * 
 * Uses Netron's grapher.js directly for exact rendering
 */

// =============================================================================
// State
// =============================================================================

const state = {
    sessionId: null,
    modelInfo: null,
    graph: null,
    selectedNode: null,
    selectedEdge: null,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    showMinimap: true,
    minimapScale: 1,
    minimapOffset: { x: 0, y: 0 },
    isMinimapDragging: false,
    minimapDragStart: { x: 0, y: 0 },
    // Viewport management for large models
    isLargeModel: false,
    viewport: { x: 0, y: 0, width: 1000, height: 800 },
    graphBounds: null,
    totalNodes: 0,
    totalEdges: 0,
    visibleNodes: new Set(),  // Track currently visible node IDs
    lastViewportUpdate: 0,
    viewportUpdateThrottle: 100  // ms
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    graphContainer: document.getElementById('graphContainer'),
    graphSvg: document.getElementById('graphSvg'),
    zoomLayer: document.getElementById('zoomLayer'),
    nodesLayer: document.getElementById('nodesLayer'),
    edgesLayer: document.getElementById('edgesLayer'),
    minimap: document.getElementById('minimap'),
    minimapSvg: document.getElementById('minimapSvg'),
    minimapLayer: document.getElementById('minimapLayer'),
    minimapViewport: document.getElementById('minimapViewport'),
    zoomLevel: document.getElementById('zoomLevel'),
    modelInfo: document.getElementById('modelInfo'),
    layerList: document.getElementById('layerList'),
    propertiesPanel: document.getElementById('propertiesPanel'),
    statusText: document.getElementById('statusText'),
    nodeCount: document.getElementById('nodeCount'),
    emptyState: document.getElementById('emptyState'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toastContainer: document.getElementById('toastContainer'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    exportBtn: document.getElementById('exportBtn'),
    zoomIn: document.getElementById('zoomIn'),
    zoomOut: document.getElementById('zoomOut'),
    fitToScreen: document.getElementById('fitToScreen'),
    resetZoom: document.getElementById('resetZoom'),
    toggleMinimap: document.getElementById('toggleMinimap'),
    layerFilter: document.getElementById('layerFilter'),
    opTypeFilter: document.getElementById('opTypeFilter'),
    shapeInferenceBtn: document.getElementById('shapeInferenceBtn'),
    simplifyBtn: document.getElementById('simplifyBtn'),
    addLayerBtn: document.getElementById('addLayerBtn'),
    addLayerModal: document.getElementById('addLayerModal'),
    addLayerForm: document.getElementById('addLayerForm'),
    confirmAddLayer: document.getElementById('confirmAddLayer'),
    layerOpType: document.getElementById('layerOpType'),
    layerName: document.getElementById('layerName'),
    layerDomain: document.getElementById('layerDomain'),
    layerAttributes: document.getElementById('layerAttributes')
};

// =============================================================================
// API
// =============================================================================

async function uploadModel(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    console.log('[Upload] Starting upload...');
    const response = await fetch('/api/model/upload', { method: 'POST', body: formData });
    console.log('[Upload] Response received, status:', response.status);
    
    if (!response.ok) {
        const text = await response.text();
        console.error('[Upload] HTTP error:', response.status, text);
        throw new Error(`HTTP ${response.status}: ${text}`);
    }
    
    // Read response as text first
    const responseText = await response.text();
    console.log('[Upload] Response text length:', responseText.length);
    console.log('[Upload] Response text (first 200 chars):', responseText.substring(0, 200));
    
    // Parse JSON manually
    try {
        const result = JSON.parse(responseText);
        console.log('[Upload] JSON parsed successfully');
        return result;
    } catch (parseError) {
        console.error('[Upload] JSON parse error:', parseError);
        console.error('[Upload] Full response:', responseText);
        throw new Error('Failed to parse server response: ' + parseError.message);
    }
}

async function apiRequest(endpoint, options = {}) {
    const response = await fetch(endpoint, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `HTTP ${response.status}`);
    }
    return response.json();
}

async function updateNodeName(sessionId, nodeId, newName) {
    return apiRequest(`/api/model/${sessionId}/rename_node`, {
        method: 'POST',
        body: JSON.stringify({ node_id: nodeId, new_name: newName })
    });
}

async function updateTensorName(sessionId, oldName, newName) {
    return apiRequest(`/api/model/${sessionId}/rename_tensor`, {
        method: 'POST',
        body: JSON.stringify({ old_name: oldName, new_name: newName })
    });
}

async function replaceInitializer(sessionId, tensorName, numpyData) {
    const formData = new FormData();
    formData.append('tensor_name', tensorName);
    formData.append('numpy_data', numpyData);
    const response = await fetch(`/api/model/${sessionId}/replace_initializer`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
    return response.json();
}

// =============================================================================
// Viewport Management for Large Models
// =============================================================================

/**
 * Load viewport for current view (convenience function)
 */
async function loadViewportForCurrentView() {
    const container = elements.graphContainer;
    const rect = container.getBoundingClientRect();
    const scale = state.zoom;
    const x = (-state.pan.x) / scale;
    const y = (-state.pan.y) / scale;
    const width = rect.width / scale;
    const height = rect.height / scale;
    
    return loadViewportNodes(x, y, width, height);
}

/**
 * Load nodes for current viewport (for large models)
 */
async function loadViewportNodes(x, y, width, height) {
    if (!state.sessionId || !state.isLargeModel) return null;
    
    try {
        const result = await apiRequest(`/api/model/${state.sessionId}/viewport`, {
            method: 'POST',
            body: JSON.stringify({ x, y, width, height })
        });
        
        if (result.success) {
            // Update visible nodes tracking
            state.visibleNodes = new Set(result.graph.nodes.map(n => n.id));
            state.viewport = result.viewport;
            state.graphBounds = result.bounds;
            state.totalNodes = result.total_nodes;
            state.totalEdges = result.total_edges;
            
            // Update graph with new visible nodes
            state.graph = result.graph;
            
            return result;
        }
    } catch (error) {
        console.error('Failed to load viewport nodes:', error);
    }
    return null;
}

/**
 * Update viewport based on current pan/zoom
 * Throttled to avoid excessive API calls
 */
function updateViewport() {
    const now = Date.now();
    if (now - state.lastViewportUpdate < state.viewportUpdateThrottle) return;
    state.lastViewportUpdate = now;
    
    if (!state.isLargeModel) return;
    
    // Calculate viewport in graph coordinates
    const container = elements.graphContainer;
    const rect = container.getBoundingClientRect();
    
    // Convert screen coordinates to graph coordinates
    const scale = state.zoom;
    const x = (-state.pan.x) / scale;
    const y = (-state.pan.y) / scale;
    const width = rect.width / scale;
    const height = rect.height / scale;
    
    // Load viewport nodes
    loadViewportNodes(x, y, width, height).then(result => {
        if (result) {
            // Re-render only visible nodes
            renderVisibleNodes();
            updateStatusBar();
        }
    });
}

/**
 * Render only visible nodes (for large models)
 */
function renderVisibleNodes() {
    if (!netronGraph || !state.graph) return;
    
    // Clear current nodes
    netronGraph.clear();
    
    // Add visible nodes only
    for (const node of state.graph.nodes) {
        netronGraph.addNode(
            node.id,
            node.op_type,
            node.name || `${node.op_type}_${node.id}`,
            node.inputs,
            node.outputs,
            node.attributes || {}
        );
    }
    
    // Set node positions
    for (const node of state.graph.nodes) {
        if (node.x !== undefined && node.y !== undefined) {
            netronGraph.setNodePosition(node.id, node.x, node.y);
        }
    }
    
    // Update visualization
    netronGraph.update();
}

/**
 * Update status bar with viewport info
 */
function updateStatusBar() {
    if (state.isLargeModel) {
        const visible = state.visibleNodes.size;
        const total = state.totalNodes;
        elements.nodeCount.textContent = `${visible}/${total} nodes (viewport)`;
    } else {
        elements.nodeCount.textContent = `${state.graph.nodes.length} nodes, ${state.graph.edges.length} edges`;
    }
}

// =============================================================================
// UI Helpers
// =============================================================================

function showLoading(text = 'Processing...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle';
    toast.innerHTML = `<i class="fas fa-${icon}"></i><span>${message}</span>`;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => { toast.style.animation = 'slideIn 0.3s ease reverse'; setTimeout(() => toast.remove(), 300); }, 3000);
}

function updateStatus(text) {
    elements.statusText.textContent = text;
}

// =============================================================================
// Text Utilities
// =============================================================================

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    const begin = text.substring(0, Math.floor(maxLength / 2) - 1);
    const end = text.substring(text.length - Math.floor(maxLength / 2) + 2);
    return `${begin}\u2026${end}`;
}

function sanitizeClassName(name) {
    return (name || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
}

// =============================================================================
// Netron Graph Wrapper
// =============================================================================

class NetronGraphWrapper {
    constructor() {
        this.graph = new grapher.Graph(false);
        this.nodeMap = new Map();
        this.edgeMap = new Map();
    }

    clear() {
        this.graph = new grapher.Graph(false);
        this.nodeMap.clear();
        this.edgeMap.clear();
    }

    addNode(nodeData) {
        const node = new grapher.Node();
        node.name = nodeData.id;
        node.id = nodeData.id;
        node.data = nodeData; // Store original data for selection
        
        // Header with operation type
        const header = node.header();
        const category = this._getCategory(nodeData.op_type);
        const styles = category ? ['node-item-type', `node-item-type-${category}`] : ['node-item-type'];
        const displayName = nodeData.op_type || 'Unknown';
        
        const titleEntry = header.add(null, styles, displayName, nodeData.name || '');
        titleEntry.on('click', () => selectNode(nodeData));
        
        // Add argument list if there are attributes to show
        if (nodeData.attributes && Object.keys(nodeData.attributes).length > 0) {
            const list = node.list();
            for (const [key, value] of Object.entries(nodeData.attributes)) {
                let valueStr = JSON.stringify(value);
                if (valueStr.length > 15) valueStr = valueStr.substring(0, 12) + '\u2026';
                const arg = list.argument(key, valueStr);
                arg.separator = ' = ';
                list.add(arg);
            }
        }
        
        this.graph.setNode(node);
        this.nodeMap.set(nodeData.id, { grapherNode: node, data: nodeData });
        return node;
    }

    addEdge(edgeData) {
        const sourceInfo = this.nodeMap.get(edgeData.source);
        const targetInfo = this.nodeMap.get(edgeData.target);
        
        if (!sourceInfo || !targetInfo) {
            console.warn(`Skipping edge ${edgeData.id}: source=${edgeData.source} target=${edgeData.target} not found`);
            return;
        }
        
        const sourceNode = sourceInfo.grapherNode;
        const targetNode = targetInfo.grapherNode;
        
        const edge = new grapher.Edge(sourceNode, targetNode);
        edge.from = sourceNode;
        edge.to = targetNode;
        edge.data = edgeData;
        edge.tensorName = edgeData.tensor;
        
        edge.v = edgeData.source;
        edge.w = edgeData.target;
        
        // Set label with tensor shape only if shape exists
        if (edgeData.shape && edgeData.shape.length > 0) {
            edge.label = this._formatShape(edgeData.shape);
        }
        
        this.graph.setEdge(edge);
        this.edgeMap.set(`${edgeData.source}:${edgeData.target}`, { grapherEdge: edge, data: edgeData });
    }
    
    _formatShape(shape) {
        if (!Array.isArray(shape) || shape.length === 0) return '?';
        // Format shape like [1,3,224,224] or [B,C,H,W] for dynamic dims
        const dims = shape.map(d => {
            if (typeof d === 'string') return d;  // dynamic dimension like 'batch'
            if (d === null || d === undefined) return '?';
            return d;
        });
        return '[' + dims.join(',') + ']';
    }

    async layout() {
        await this.graph.layout();
    }

    prepareContainer() {
        const origin = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        origin.setAttribute('id', 'origin');
        
        while (elements.zoomLayer.firstChild) {
            elements.zoomLayer.removeChild(elements.zoomLayer.firstChild);
        }
        elements.zoomLayer.appendChild(origin);
    }

    build(svgDocument) {
        this.graph.build(svgDocument);
        
        // Add click handlers to nodes
        for (const [id, nodeInfo] of this.nodeMap) {
            const nodeEl = nodeInfo.grapherNode.element;
            if (nodeEl) {
                nodeEl.style.cursor = 'pointer';
                nodeEl.addEventListener('click', (e) => {
                    e.stopPropagation();
                    selectNode(nodeInfo.data);
                });
            }
        }
    }

    update() {
        this.graph.update();
    }

    _getCategory(opType) {
        const categories = {
            'Conv': 'layer', 'ConvTranspose': 'layer', 'Gemm': 'layer', 'MatMul': 'layer',
            'MaxPool': 'pool', 'AveragePool': 'pool', 'GlobalAveragePool': 'pool',
            'BatchNormalization': 'normalization', 'LayerNormalization': 'normalization',
            'Relu': 'activation', 'Sigmoid': 'activation', 'Tanh': 'activation', 'Softmax': 'activation',
            'Gelu': 'activation', 'LeakyRelu': 'activation',
            'Reshape': 'shape', 'Transpose': 'shape', 'Squeeze': 'shape', 'Unsqueeze': 'shape',
            'Constant': 'constant',
            'Input': 'input', 'Output': 'output'
        };
        return categories[opType] || '';
    }
}

// =============================================================================
// Graph Rendering
// =============================================================================

let netronGraph = null;

async function renderGraph() {
    if (!state.graph || !state.graph.nodes.length) {
        elements.emptyState.style.display = 'block';
        return;
    }
    
    elements.emptyState.style.display = 'none';
    showLoading(state.isLargeModel ? 'Rendering viewport...' : 'Rendering graph...');
    
    try {
        if (typeof dagre === 'undefined') throw new Error('dagre library not loaded');
        if (typeof grapher === 'undefined') throw new Error('grapher library not loaded');
        
        netronGraph = new NetronGraphWrapper();
        
        for (const node of state.graph.nodes) {
            netronGraph.addNode(node);
        }
        
        for (const edge of state.graph.edges) {
            netronGraph.addEdge(edge);
        }
        
        netronGraph.prepareContainer();
        netronGraph.build(document);
        
        if (document.fonts && document.fonts.ready) await document.fonts.ready;
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        
        await netronGraph.graph.measure();
        
        // For viewport mode, use pre-computed positions from backend
        // For normal mode, run layout algorithm
        if (state.isLargeModel && state.graph.nodes[0]?.x !== undefined) {
            console.log('[Viewport] Using pre-computed node positions');
            // Set positions from backend data
            for (const node of state.graph.nodes) {
                if (node.x !== undefined && node.y !== undefined) {
                    netronGraph.setNodePosition(node.id, node.x, node.y);
                }
            }
            // Build edge paths
            netronGraph.update();
        } else {
            // Normal layout
            await netronGraph.layout();
            netronGraph.update();
        }
        
        // Update status bar with appropriate message
        if (state.isLargeModel) {
            updateStatusBar();
        } else {
            elements.nodeCount.textContent = `${state.graph.nodes.length} nodes, ${state.graph.edges.length} edges`;
        }
        
        renderLayerList();
        renderModelInfo(state.modelInfo);
        renderMinimap();
        
        // Set zoom to 100% (no auto fit)
        resetZoom100();
        
    } catch (err) {
        console.error('Render error:', err);
        showToast(`Failed to render: ${err.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function clearGraph() {
    elements.nodesLayer.innerHTML = '';
    elements.edgesLayer.innerHTML = '';
    state.graph = null;
    state.selectedNode = null;
    elements.emptyState.style.display = 'block';
}

// =============================================================================
// Selection & Properties
// =============================================================================

function selectNode(node) {
    if (!node) {
        // Clear previous selection
        document.querySelectorAll('.node.select').forEach(el => el.classList.remove('select'));
        state.selectedNode = null;
        renderProperties(null);
        return;
    }
    
    state.selectedNode = node;
    renderProperties(node);
    
    // Highlight in layer list
    document.querySelectorAll('.layer-item').forEach(el => {
        el.classList.toggle('selected', el.dataset.id === node.id);
    });
    
    // Scroll and center on node (this will also handle highlighting)
    scrollToNode(node);
}

function scrollToNode(node) {
    if (!node) return;
    
    // Get the grapher node for accurate position
    const nodeInfo = netronGraph?.nodeMap?.get(node.id);
    if (!nodeInfo || !nodeInfo.grapherNode) {
        console.warn("Grapher node not found for:", node.id);
        return;
    }
    
    const grapherNode = nodeInfo.grapherNode;
    // Use grapherNode actual rendered position (center point)
    const nodeX = grapherNode.x || 0;
    const nodeY = grapherNode.y || 0;
    
    const container = elements.graphContainer;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    // Calculate the target pan to center the node
    state.pan.x = containerWidth / 2 - nodeX * state.zoom;
    state.pan.y = containerHeight / 2 - nodeY * state.zoom;
    
    updateTransform();
    
    // Highlight the node in the graph after transform is applied
    requestAnimationFrame(() => {
        highlightNodeInGraph(node.id);
    });
}

function highlightNodeInGraph(nodeId) {
    // Clear previous selection
    document.querySelectorAll('.node.select').forEach(el => el.classList.remove('select'));
    
    if (!netronGraph) {
        console.warn('netronGraph not available');
        return;
    }
    
    // Use grapher.Node's select method if available
    const nodeInfo = netronGraph.nodeMap?.get(nodeId);
    if (!nodeInfo) {
        console.warn('Node not found in nodeMap:', nodeId);
        return;
    }
    
    const grapherNode = nodeInfo.grapherNode;
    if (!grapherNode) {
        console.warn('No grapherNode for node:', nodeId);
        return;
    }
    
    // Try using the grapher's native select method first
    if (typeof grapherNode.select === 'function') {
        const result = grapherNode.select();
        console.log('select() called, returned:', result);
    } else if (grapherNode.element) {
        // Fallback: directly add class to the element
        grapherNode.element.classList.add('select');
        console.log('Added select class directly to element');
    } else {
        console.warn('grapherNode has no select method or element');
    }
}

// Global state for expanded initializers
const expandedInitializers = new Set();

function renderProperties(node) {
    if (!node) {
        elements.propertiesPanel.innerHTML = '<p class="placeholder">Select a node to view properties</p>';
        elements.propertiesPanel.classList.add('empty');
        return;
    }
    
    elements.propertiesPanel.classList.remove('empty');
    
    const isVirtual = node.is_virtual || false;
    const canEdit = !isVirtual;
    const tensors = state.graph?.tensors || {};
    
    let html = '';
    
    // ===== Node Properties Section =====
    html += '<div class="property-section">';
    html += '<div class="property-section-title">Node Properties</div>';
    
    // Type
    html += '<div class="property-item">';
    html += '<div class="property-item-name">type</div>';
    html += `<div class="property-item-value">${node.op_type}</div>`;
    html += '</div>';
    
    // Name (editable)
    html += '<div class="property-item">';
    html += '<div class="property-item-name">name</div>';
    html += '<div class="property-item-value editable">';
    html += `<input type="text" id="nodeNameInput" value="${node.name || ''}" ${canEdit ? '' : 'disabled'} placeholder="Node name">`;
    html += '</div>';
    html += '</div>';
    
    if (node.domain) {
        html += '<div class="property-item">';
        html += '<div class="property-item-name">domain</div>';
        html += `<div class="property-item-value">${node.domain}</div>`;
        html += '</div>';
    }
    
    html += '</div>'; // End Node Properties
    
    // ===== Attributes Section =====
    if (node.attributes && Object.keys(node.attributes).length > 0) {
        html += '<div class="property-section">';
        html += '<div class="property-section-title">Attributes</div>';
        
        for (const [key, value] of Object.entries(node.attributes)) {
            html += '<div class="property-item">';
            html += `<div class="property-item-name">${key}</div>`;
            html += `<div class="property-item-value attribute-value">${JSON.stringify(value)}</div>`;
            html += '</div>';
        }
        
        html += '</div>';
    }
    
    // ===== Inputs Section =====
    if (node.inputs && node.inputs.length > 0) {
        html += '<div class="property-section">';
        html += '<div class="property-section-title">Inputs</div>';
        
        for (let i = 0; i < node.inputs.length; i++) {
            const inputName = node.inputs[i];
            const tensorInfo = tensors[inputName];
            html += renderTensorProperty(inputName, i, tensorInfo, false);
        }
        
        html += '</div>';
    }
    
    // ===== Outputs Section =====
    if (node.outputs && node.outputs.length > 0) {
        html += '<div class="property-section">';
        html += '<div class="property-section-title">Outputs</div>';
        
        for (let i = 0; i < node.outputs.length; i++) {
            const outputName = node.outputs[i];
            const tensorInfo = tensors[outputName];
            html += renderTensorProperty(outputName, i, tensorInfo, true);
        }
        
        html += '</div>';
    }
    
    // ===== Actions =====
    html += '<div class="property-actions">';
    html += '<button class="btn btn-primary" onclick="saveNodeChanges()"><i class="fas fa-save"></i> Save Changes</button>';
    html += '</div>';
    
    elements.propertiesPanel.innerHTML = html;
}

// Helper to render tensor property (Netron style)
function renderTensorProperty(name, index, tensorInfo, isOutput) {
    const isInitializer = tensorInfo && tensorInfo.type === 'initializer';
    const hasValues = tensorInfo && Array.isArray(tensorInfo.values) && tensorInfo.values.length > 0;
    const dtype = tensorInfo?.dtype || 'unknown';
    const shape = tensorInfo?.shape || [];
    const shapeStr = shape.length > 0 ? shape.join(', ') : '';
    
    let html = '<div class="property-list-item">';
    
    // Tensor name (bold, like Netron)
    html += `<div class="property-list-item-header">${name}</div>`;
    
    // Name input (editable)
    html += '<div class="property-item">';
    html += '<div class="property-item-name">name</div>';
    html += '<div class="property-item-value editable">';
    html += `<input type="text" class="tensor-name-input" data-original="${name}" value="${name}">`;
    html += '</div>';
    html += '</div>';
    
    if (tensorInfo) {
        // Data type (dropdown)
        html += '<div class="property-item">';
        html += '<div class="property-item-name">type</div>';
        html += '<div class="property-item-value editable">';
        html += `<select data-tensor="${name}" onchange="handleDtypeChange(this)">`;
        for (const dt of ONNX_DTYPE_OPTIONS) {
            const selected = dt === dtype ? 'selected' : '';
            html += `<option value="${dt}" ${selected}>${dt}</option>`;
        }
        html += '</select>';
        html += '</div>';
        html += '</div>';
        
        // Shape (input)
        html += '<div class="property-item">';
        html += '<div class="property-item-name">shape</div>';
        html += '<div class="property-item-value editable">';
        html += `<input type="text" class="shape-input" data-tensor="${name}" value="[${shapeStr}]" placeholder="[dim1, dim2, ...]" onchange="handleShapeChange(this)">`;
        html += '</div>';
        html += '</div>';
        
        // Initializer expander (if applicable)
        if (isInitializer) {
            const expanderId = `expander-${name}`;
            const isExpanded = expandedInitializers.has(expanderId);
            
            html += '<div class="property-item">';
            html += '<div class="property-item-name">category</div>';
            html += '<div class="property-item-value">';
            html += `<span class="initializer-category">Initializer</span>`;
            
            // Replace button
            if (!isOutput) {
                html += `<button class="btn-icon-small" onclick="replaceInitializerData('${name}')" title="Replace data"><i class="fas fa-file-import"></i></button>`;
            }
            
            html += '</div>';
            html += '</div>';
            
            // Expandable data section - always show the header
            html += `<div class="initializer-header" onclick="toggleInitializer('${expanderId}')">`;
            html += `<span class="initializer-expand-icon${isExpanded ? ' expanded' : ''}">${isExpanded ? '▼' : '▶'}</span>`;
            html += `<span>tensor:${dtype}[${shapeStr}]</span>`;
            html += '</div>';
            
            // Show data if expanded
            if (isExpanded) {
                html += `<div id="${expanderId}" class="initializer-data">`;
                if (hasValues) {
                    html += `<pre>${formatTensorData(tensorInfo.values, shape)}</pre>`;
                } else {
                    html += `<pre style="color: #888;">No data available (large tensor or empty)</pre>`;
                }
                html += '</div>';
            }
        }
    }
    
    html += '</div>';
    return html;
}

// Toggle initializer expand
window.toggleInitializer = function(expanderId) {
    if (expandedInitializers.has(expanderId)) {
        expandedInitializers.delete(expanderId);
    } else {
        expandedInitializers.add(expanderId);
    }
    // Re-render to show/hide data
    if (state.selectedNode) {
        renderProperties(state.selectedNode);
    }
};

// Format tensor data for display (multi-dimensional)
function formatTensorData(values, shape) {
    if (!values || !shape) return '[]';
    
    const totalElements = shape.reduce((a, b) => a * b, 1);
    if (values.length !== totalElements) {
        // Flat display if shape doesn't match
        return JSON.stringify(values.slice(0, 20)) + (values.length > 20 ? '...' : '');
    }
    
    // Reshape and format
    function formatRecursive(data, dims, offset) {
        if (dims.length === 0) return String(data[offset]);
        if (dims.length === 1) {
            const row = [];
            for (let i = 0; i < dims[0] && offset + i < data.length; i++) {
                const val = data[offset + i];
                row.push(typeof val === 'number' ? val.toFixed(6) : String(val));
            }
            return '[' + row.join(', ') + ']';
        }
        
        const dimSize = dims[0];
        const subDimSize = dims.slice(1).reduce((a, b) => a * b, 1);
        const rows = [];
        for (let i = 0; i < dimSize; i++) {
            rows.push(formatRecursive(data, dims.slice(1), offset + i * subDimSize));
        }
        return '[\n' + rows.join(',\n') + '\n]';
    }
    
    return formatRecursive(values, shape, 0);
}

// ONNX Data Types
const ONNX_DTYPE_OPTIONS = [
    // Standard types
    'FLOAT', 'UINT8', 'INT8', 'UINT16', 'INT16', 
    'INT32', 'INT64', 'STRING', 'BOOL', 'FLOAT16',
    'DOUBLE', 'UINT32', 'UINT64', 'COMPLEX64', 
    'COMPLEX128', 'BFLOAT16',
    // FP8 types (ONNX 1.14+)
    'FLOAT8E4M3FN', 'FLOAT8E4M3FNUZ', 'FLOAT8E5M2', 'FLOAT8E5M2FNUZ',
    // INT4 types (ONNX 1.16+)
    'UINT4', 'INT4',
    // INT2 types (ONNX 1.20+)
    'UINT2', 'INT2'
];

// Format initializer values for display
function formatInitializerValues(values, dtype) {
    if (!values || !Array.isArray(values)) {
        return 'N/A';
    }
    
    const totalCount = values.length;
    if (totalCount === 0) {
        return '[]';
    }
    
    // Format a single value based on dtype
    const isFloat = dtype && (dtype.includes('FLOAT') || dtype.includes('DOUBLE'));
    const formatValue = (v) => {
        if (typeof v === 'number') {
            return isFloat ? v.toFixed(4) : v;
        }
        return String(v);
    };
    
    // For small arrays, show all values
    if (totalCount <= 20) {
        return values.map(formatValue).join(', ');
    }
    
    // For large arrays, show head and tail
    const headCount = 8;
    const tailCount = 8;
    
    const head = values.slice(0, headCount).map(formatValue);
    const tail = values.slice(-tailCount).map(formatValue);
    
    return `${head.join(', ')}, ..., ${tail.join(', ')} [${totalCount} total]`;
}

// Generate dtype select HTML
function generateDtypeSelect(currentDtype, tensorName, onChangeHandler) {
    const options = ONNX_DTYPE_OPTIONS.map(dt => 
        `<option value="${dt}" ${dt === currentDtype ? 'selected' : ''}>${dt}</option>`
    ).join('');
    
    return `<select class="dtype-select form-control" data-tensor="${tensorName}" onchange="${onChangeHandler}(this)">
        ${options}
    </select>`;
}

// Handle dtype change
window.handleDtypeChange = async function(selectEl) {
    if (!state.sessionId) {
        showToast('No model loaded', 'error');
        return;
    }
    
    const tensorName = selectEl.dataset.tensor;
    const newDtype = selectEl.value;
    
    showLoading('Updating data type...');
    try {
        const result = await apiRequest(`/api/model/${state.sessionId}/modify_dtype`, {
            method: 'POST',
            body: JSON.stringify({
                tensor_name: tensorName,
                new_dtype: newDtype
            })
        });
        
        state.graph = result.graph;
        await renderGraph();
        showToast(`Data type updated to ${newDtype}`, 'success');
    } catch (error) {
        showToast(`Failed to update dtype: ${error.message}`, 'error');
        // Reset select to previous value
        const tensorInfo = state.graph.tensors?.[tensorName];
        if (tensorInfo) {
            selectEl.value = tensorInfo.dtype;
        }
    } finally {
        hideLoading();
    }
};

// Handle shape change
window.handleShapeChange = async function(inputEl) {
    if (!state.sessionId) {
        showToast('No model loaded', 'error');
        return;
    }
    
    const tensorName = inputEl.dataset.tensor;
    const shapeStr = inputEl.value.trim();
    
    // Parse shape string [dim1, dim2, ...]
    let newShape;
    try {
        if (shapeStr.startsWith('[') && shapeStr.endsWith(']')) {
            const inner = shapeStr.slice(1, -1).trim();
            newShape = inner ? inner.split(',').map(s => {
                const trimmed = s.trim();
                if (trimmed === '' || trimmed.toLowerCase() === 'none') return null;
                const num = parseInt(trimmed);
                return isNaN(num) ? trimmed : num;
            }) : [];
        } else {
            throw new Error('Invalid shape format');
        }
    } catch (e) {
        showToast('Invalid shape format. Use [dim1, dim2, ...]', 'error');
        return;
    }
    
    showLoading('Updating shape...');
    try {
        const result = await apiRequest(`/api/model/${state.sessionId}/modify_shape`, {
            method: 'POST',
            body: JSON.stringify({
                tensor_name: tensorName,
                new_shape: newShape
            })
        });
        
        state.graph = result.graph;
        await renderGraph();
        showToast('Shape updated', 'success');
    } catch (error) {
        showToast(`Failed to update shape: ${error.message}`, 'error');
        // Reset input to previous value
        const tensorInfo = state.graph.tensors?.[tensorName];
        if (tensorInfo?.shape) {
            inputEl.value = `[${tensorInfo.shape.join(', ')}]`;
        }
    } finally {
        hideLoading();
    }
};

// Global function for initializer replacement
window.replaceInitializerData = function(tensorName) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.npy,.npz';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        showLoading('Uploading numpy data...');
        try {
            const result = await replaceInitializer(state.sessionId, tensorName, file);
            showToast('Initializer replaced successfully', 'success');
            // Refresh graph
            state.graph = result.graph;
            await renderGraph();
        } catch (error) {
            showToast(`Failed: ${error.message}`, 'error');
        } finally {
            hideLoading();
        }
    };
    input.click();
};

// Global function to save node changes
window.saveNodeChanges = async function() {
    if (!state.selectedNode || !state.sessionId) return;
    
    const nodeNameInput = document.getElementById('nodeNameInput');
    const tensorInputs = document.querySelectorAll('.tensor-name-input');
    
    showLoading('Saving changes...');
    try {
        // Update node name if changed
        if (nodeNameInput && nodeNameInput.value !== state.selectedNode.name) {
            await updateNodeName(state.sessionId, state.selectedNode.id, nodeNameInput.value);
        }
        
        // Update tensor names if changed
        for (const input of tensorInputs) {
            const originalName = input.dataset.original;
            const newName = input.value;
            if (newName !== originalName) {
                await updateTensorName(state.sessionId, originalName, newName);
            }
        }
        
        showToast('Changes saved', 'success');
        // Refresh to show updates
        await refreshGraph();
    } catch (error) {
        showToast(`Failed to save: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
};

async function refreshGraph() {
    if (!state.sessionId) return;
    // Reload model data
    const result = await apiRequest(`/api/model/${state.sessionId}/graph`);
    state.graph = result.graph;
    await renderGraph();
}

// =============================================================================
// Layer List
// =============================================================================

function renderLayerList() {
    if (!state.graph) return;
    
    const filter = elements.layerFilter?.value?.toLowerCase() || '';
    const opFilter = elements.opTypeFilter?.value || '';
    
    // Get unique op types for filter
    const opTypes = [...new Set(state.graph.nodes.map(n => n.op_type))].sort();
    if (elements.opTypeFilter && elements.opTypeFilter.options.length <= 1) {
        opTypes.forEach(op => {
            const option = document.createElement('option');
            option.value = op;
            option.textContent = op;
            elements.opTypeFilter.appendChild(option);
        });
    }
    
    const filtered = state.graph.nodes.filter(node => {
        const matchesName = (node.name || node.id).toLowerCase().includes(filter);
        const matchesOp = !opFilter || node.op_type === opFilter;
        return matchesName && matchesOp;
    });
    
    elements.layerList.innerHTML = filtered.map(node => `
        <div class="layer-item ${state.selectedNode?.id === node.id ? 'selected' : ''}" 
             data-id="${node.id}" onclick="window.selectNodeById('${node.id}')">
            <span class="op-type">${node.op_type}</span>
            <span class="layer-name">${truncateText(node.name || node.id, 20)}</span>
        </div>
    `).join('');
}

window.selectNodeById = function(nodeId) {
    const node = state.graph.nodes.find(n => n.id === nodeId);
    if (node) selectNode(node);
};

// =============================================================================
// Model Info
// =============================================================================

function renderModelInfo(info) {
    if (!info) {
        elements.modelInfo.innerHTML = '<p class="placeholder">Upload a model</p>';
        return;
    }
    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    elements.modelInfo.innerHTML = `
        <div class="info-row"><span class="info-label">File</span><span class="info-value">${info.file_name}</span></div>
        <div class="info-row"><span class="info-label">Size</span><span class="info-value">${formatBytes(info.file_size)}</span></div>
        <div class="info-row"><span class="info-label">Nodes</span><span class="info-value">${info.num_nodes}</span></div>
        <div class="info-row"><span class="info-label">Edges</span><span class="info-value">${info.num_edges}</span></div>
    `;
}

// =============================================================================
// Viewport / Navigation
// =============================================================================

function updateTransform() {
    elements.zoomLayer.setAttribute('transform', 
        `translate(${state.pan.x}, ${state.pan.y}) scale(${state.zoom})`);
    elements.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
    updateMinimapViewport();
    
    // Update viewport for large models
    if (state.isLargeModel) {
        updateViewport();
    }
}

function zoomIn() {
    state.zoom = Math.min(state.zoom * 1.1, 2);
    updateTransform();
}

function zoomOut() {
    state.zoom = Math.max(state.zoom / 1.1, 0.1);
    updateTransform();
}

function resetZoom100() {
    state.zoom = 1.0;
    // Center the graph at 100% zoom
    if (netronGraph) {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        
        for (const [id, nodeInfo] of netronGraph.nodeMap) {
            const node = nodeInfo.grapherNode;
            const x = node.x || 0;
            const y = node.y || 0;
            const w = node.width || 100;
            const h = node.height || 40;
            minX = Math.min(minX, x - w / 2);
            minY = Math.min(minY, y - h / 2);
            maxX = Math.max(maxX, x + w / 2);
            maxY = Math.max(maxY, y + h / 2);
        }
        
        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;
        const containerWidth = elements.graphContainer.clientWidth;
        const containerHeight = elements.graphContainer.clientHeight;
        
        // Center the graph
        state.pan.x = (containerWidth - graphWidth) / 2 - minX;
        state.pan.y = (containerHeight - graphHeight) / 2 - minY;
    }
    updateTransform();
}

function fitToScreen() {
    if (!netronGraph) return;
    
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    for (const [id, nodeInfo] of netronGraph.nodeMap) {
        const node = nodeInfo.grapherNode;
        const x = node.x || 0;
        const y = node.y || 0;
        const w = node.width || 100;
        const h = node.height || 40;
        minX = Math.min(minX, x - w / 2);
        minY = Math.min(minY, y - h / 2);
        maxX = Math.max(maxX, x + w / 2);
        maxY = Math.max(maxY, y + h / 2);
    }
    
    const padding = 50;
    minX -= padding; minY -= padding;
    maxX += padding; maxY += padding;
    
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    const containerWidth = elements.graphContainer.clientWidth;
    const containerHeight = elements.graphContainer.clientHeight;
    
    state.zoom = Math.min(containerWidth / graphWidth, containerHeight / graphHeight, 1);
    state.pan.x = (containerWidth - graphWidth * state.zoom) / 2 - minX * state.zoom;
    state.pan.y = (containerHeight - graphHeight * state.zoom) / 2 - minY * state.zoom;
    
    updateTransform();
}

function handleWheel(e) {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        state.zoom = Math.max(0.1, Math.min(2, state.zoom * delta));
        updateTransform();
    } else {
        state.pan.x -= e.deltaX;
        state.pan.y -= e.deltaY;
        updateTransform();
    }
}

function handleMouseDown(e) {
    if (e.button === 0) {
        state.isDragging = true;
        state.dragStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
        elements.graphContainer.style.cursor = 'grabbing';
    }
}

function handleMouseMove(e) {
    if (state.isDragging) {
        state.pan.x = e.clientX - state.dragStart.x;
        state.pan.y = e.clientY - state.dragStart.y;
        updateTransform();
    }
}

function handleMouseUp() {
    state.isDragging = false;
    elements.graphContainer.style.cursor = 'grab';
}

// =============================================================================
// Minimap
// =============================================================================

function toggleMinimap() {
    state.showMinimap = !state.showMinimap;
    elements.minimap.classList.toggle('hidden', !state.showMinimap);
    elements.toggleMinimap?.classList.toggle('active', state.showMinimap);
    if (state.showMinimap) renderMinimap();
}

function renderMinimap() {
    if (!netronGraph || !state.showMinimap) return;
    
    elements.minimapLayer.innerHTML = '';
    
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    for (const [id, nodeInfo] of netronGraph.nodeMap) {
        const node = nodeInfo.grapherNode;
        const x = node.x || 0;
        const y = node.y || 0;
        const w = node.width || 100;
        const h = node.height || 40;
        minX = Math.min(minX, x - w / 2);
        minY = Math.min(minY, y - h / 2);
        maxX = Math.max(maxX, x + w / 2);
        maxY = Math.max(maxY, y + h / 2);
    }
    
    const padding = 20;
    minX -= padding; minY -= padding;
    maxX += padding; maxY += padding;
    
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    const minimapWidth = elements.minimap.clientWidth;
    const minimapHeight = elements.minimap.clientHeight;
    
    const scale = Math.min(minimapWidth / graphWidth, minimapHeight / graphHeight);
    const offsetX = (minimapWidth - graphWidth * scale) / 2;
    const offsetY = (minimapHeight - graphHeight * scale) / 2;
    
    // Render nodes
    for (const [id, nodeInfo] of netronGraph.nodeMap) {
        const node = nodeInfo.grapherNode;
        const x = (node.x || 0) - (node.width || 100) / 2;
        const y = (node.y || 0) - (node.height || 40) / 2;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', offsetX + (x - minX) * scale);
        rect.setAttribute('y', offsetY + (y - minY) * scale);
        rect.setAttribute('width', (node.width || 100) * scale);
        rect.setAttribute('height', (node.height || 40) * scale);
        rect.setAttribute('fill', getNodeColor(nodeInfo.data.op_type));
        rect.setAttribute('rx', 2);
        elements.minimapLayer.appendChild(rect);
    }
    
    state.minimapScale = scale;
    state.minimapOffset = { x: offsetX - minX * scale, y: offsetY - minY * scale };
    updateMinimapViewport();
}

function getNodeColor(opType) {
    if (opType === 'Input') return '#4CAF50';
    if (opType === 'Output') return '#FF5722';
    const categories = {
        'layer': '#3b82f6', 'activation': '#ef4444', 'pool': '#10b981',
        'normalization': '#8b5cf6', 'shape': '#f59e0b', 'constant': '#6b7280',
        'input': '#4CAF50', 'output': '#FF5722'
    };
    return categories['layer'] || '#3b82f6';
}

function updateMinimapViewport() {
    if (!state.showMinimap) return;
    const containerWidth = elements.graphContainer.clientWidth;
    const containerHeight = elements.graphContainer.clientHeight;
    const viewX = -state.pan.x / state.zoom;
    const viewY = -state.pan.y / state.zoom;
    const viewW = containerWidth / state.zoom;
    const viewH = containerHeight / state.zoom;
    
    elements.minimapViewport.setAttribute('x', viewX * state.minimapScale + state.minimapOffset.x);
    elements.minimapViewport.setAttribute('y', viewY * state.minimapScale + state.minimapOffset.y);
    elements.minimapViewport.setAttribute('width', viewW * state.minimapScale);
    elements.minimapViewport.setAttribute('height', viewH * state.minimapScale);
}

// =============================================================================
// Event Handlers
// =============================================================================

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.endsWith('.onnx')) {
        showToast('Please select an .onnx file', 'error');
        return;
    }
    
    showLoading('Loading model...');
    try {
        const result = await uploadModel(file);
        if (!result || !result.graph) {
            throw new Error('Invalid server response');
        }
        state.sessionId = result.session_id;
        state.modelInfo = result.info;
        state.graph = result.graph;
        
        // Check if this is a large model that needs viewport management
        const nodeCount = result.info?.num_nodes || result.graph.nodes.length;
        state.isLargeModel = nodeCount > 500;
        state.totalNodes = nodeCount;
        state.totalEdges = result.info?.num_edges || result.graph.edges.length;
        
        if (state.isLargeModel) {
            console.log(`[Viewport] Large model detected (${nodeCount} nodes), enabling viewport mode`);
            showToast(`Large model: ${nodeCount} nodes. Viewport mode enabled.`, 'info');
            
            // For large models, we need to load initial viewport
            // Reset pan/zoom to origin
            state.pan = { x: 0, y: 0 };
            state.zoom = 1.0;
            updateTransform();
            
            // Load initial viewport
            await loadViewportForCurrentView();
        }
        
        await renderGraph();
        updateToolButtonsState();
        showToast('Model loaded', 'success');
    } catch (error) {
        console.error('Upload error:', error);
        showToast(`Failed: ${error.message || 'Unknown error'}`, 'error');
    } finally {
        hideLoading();
        elements.fileInput.value = '';
    }
}

function initEventListeners() {
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileUpload);
    
    elements.zoomIn.addEventListener('click', zoomIn);
    elements.zoomOut.addEventListener('click', zoomOut);
    elements.fitToScreen.addEventListener('click', fitToScreen);
    if (elements.resetZoom) elements.resetZoom.addEventListener('click', resetZoom100);
    if (elements.toggleMinimap) elements.toggleMinimap.addEventListener('click', toggleMinimap);
    
    // Tool buttons
    if (elements.shapeInferenceBtn) elements.shapeInferenceBtn.addEventListener('click', handleShapeInference);
    if (elements.simplifyBtn) elements.simplifyBtn.addEventListener('click', handleSimplify);
    if (elements.addLayerBtn) elements.addLayerBtn.addEventListener('click', handleAddLayer);
    
    // Modal buttons
    if (elements.confirmAddLayer) elements.confirmAddLayer.addEventListener('click', confirmAddLayerHandler);
    
    // Close modal when clicking outside or on close button
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    });
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', () => {
            const modal = btn.closest('.modal');
            if (modal) modal.classList.remove('active');
        });
    });
    
    elements.graphContainer.addEventListener('wheel', handleWheel, { passive: false });
    elements.graphContainer.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    
    // Minimap drag
    if (elements.minimap) {
        elements.minimap.addEventListener('mousedown', handleMinimapMouseDown);
        elements.minimap.addEventListener('mousemove', handleMinimapMouseMove);
        elements.minimap.addEventListener('mouseup', handleMinimapMouseUp);
        elements.minimap.addEventListener('mouseleave', handleMinimapMouseUp);
    }
    
    if (elements.layerFilter) elements.layerFilter.addEventListener('input', renderLayerList);
    if (elements.opTypeFilter) elements.opTypeFilter.addEventListener('change', renderLayerList);
    
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === '0') {
            e.preventDefault();
            fitToScreen();
        }
    });
}

function init() {
    console.log('Oniris with Netron Grapher initializing...');
    
    if (typeof dagre === 'undefined') {
        console.error('ERROR: dagre library not loaded!');
        showToast('Failed to load dagre library', 'error');
    } else {
        console.log('dagre library loaded');
    }
    
    if (typeof grapher === 'undefined') {
        console.error('ERROR: grapher library not loaded!');
        showToast('Failed to load grapher library', 'error');
    } else {
        console.log('grapher library loaded');
    }
    
    initEventListeners();
    elements.emptyState.style.display = 'block';
    
    // Show minimap by default
    if (state.showMinimap) {
        elements.minimap?.classList.remove('hidden');
        elements.toggleMinimap?.classList.add('active');
    }
    
    updateStatus('Ready');
}

// =============================================================================
// Tool Button Handlers
// =============================================================================

async function handleShapeInference() {
    if (!state.sessionId) {
        showToast('No model loaded', 'error');
        return;
    }
    showLoading('Running shape inference...');
    try {
        const result = await apiRequest(`/api/model/${state.sessionId}/shape_inference`, { method: 'POST' });
        state.graph = result.graph;
        await renderGraph();
        showToast('Shape inference completed', 'success');
    } catch (error) {
        showToast(`Shape inference failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handleSimplify() {
    if (!state.sessionId) {
        showToast('No model loaded', 'error');
        return;
    }
    showLoading('Simplifying model...');
    try {
        const result = await apiRequest(`/api/model/${state.sessionId}/simplify`, { method: 'POST' });
        state.graph = result.graph;
        await renderGraph();
        showToast('Model simplified', 'success');
    } catch (error) {
        showToast(`Simplify failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function handleAddLayer() {
    if (!state.sessionId) {
        showToast('No model loaded', 'error');
        return;
    }
    openModal('addLayerModal');
}

async function confirmAddLayerHandler() {
    if (!state.sessionId) return;
    
    const opType = elements.layerOpType?.value;
    const name = elements.layerName?.value;
    const domain = elements.layerDomain?.value;
    const attributesStr = elements.layerAttributes?.value;
    
    if (!opType) {
        showToast('Please select an operator type', 'error');
        return;
    }
    
    let attributes = {};
    if (attributesStr) {
        try {
            attributes = JSON.parse(attributesStr);
        } catch (e) {
            showToast('Invalid JSON in attributes field', 'error');
            return;
        }
    }
    
    showLoading('Adding layer...');
    try {
        const result = await apiRequest(`/api/model/${state.sessionId}/add_layer`, {
            method: 'POST',
            body: JSON.stringify({
                op_type: opType,
                name: name,
                domain: domain,
                attributes: attributes
            })
        });
        
        state.graph = result.graph;
        await renderGraph();
        closeModal('addLayerModal');
        showToast('Layer added successfully', 'success');
        
        // Reset form
        if (elements.addLayerForm) elements.addLayerForm.reset();
    } catch (error) {
        showToast(`Failed to add layer: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

function updateToolButtonsState() {
    const hasModel = !!state.sessionId;
    if (elements.shapeInferenceBtn) elements.shapeInferenceBtn.disabled = !hasModel;
    if (elements.simplifyBtn) elements.simplifyBtn.disabled = !hasModel;
    if (elements.addLayerBtn) elements.addLayerBtn.disabled = !hasModel;
}

// =============================================================================
// Minimap Drag Handlers
// =============================================================================

function handleMinimapMouseDown(e) {
    if (!state.graph) return;
    e.preventDefault();
    
    const rect = elements.minimap.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    state.isMinimapDragging = true;
    state.minimapDragStart = { x, y };
    
    // If clicked on viewport, start drag; otherwise jump to position
    const viewportX = parseFloat(elements.minimapViewport.getAttribute('x')) || 0;
    const viewportY = parseFloat(elements.minimapViewport.getAttribute('y')) || 0;
    const viewportW = parseFloat(elements.minimapViewport.getAttribute('width')) || 0;
    const viewportH = parseFloat(elements.minimapViewport.getAttribute('height')) || 0;
    
    if (x >= viewportX && x <= viewportX + viewportW &&
        y >= viewportY && y <= viewportY + viewportH) {
        // Clicked on viewport - will drag
    } else {
        // Clicked elsewhere - jump to center at this position
        jumpToMinimapPosition(x, y);
    }
}

function handleMinimapMouseMove(e) {
    if (!state.isMinimapDragging || !state.graph) return;
    
    const rect = elements.minimap.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dx = x - state.minimapDragStart.x;
    const dy = y - state.minimapDragStart.y;
    
    // Convert minimap delta to graph coordinates
    const graphDx = -dx / state.minimapScale * state.zoom;
    const graphDy = -dy / state.minimapScale * state.zoom;
    
    state.pan.x += graphDx;
    state.pan.y += graphDy;
    
    state.minimapDragStart = { x, y };
    updateTransform();
}

function handleMinimapMouseUp(e) {
    state.isMinimapDragging = false;
}

function jumpToMinimapPosition(minimapX, minimapY) {
    const container = elements.graphContainer;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    // Convert minimap position to graph coordinates
    const graphX = (minimapX - state.minimapOffset.x) / state.minimapScale;
    const graphY = (minimapY - state.minimapOffset.y) / state.minimapScale;
    
    // Center the viewport on this position
    state.pan.x = containerWidth / 2 - graphX * state.zoom;
    state.pan.y = containerHeight / 2 - graphY * state.zoom;
    
    updateTransform();
}

document.addEventListener('DOMContentLoaded', init);
