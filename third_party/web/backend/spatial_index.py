"""
Spatial Index for Viewport-based Node Loading
Supports large models (4GB+) by only loading visible nodes
"""

import math
from typing import List, Dict, Any, Tuple, Optional


class QuadTreeNode:
    """QuadTree node for spatial indexing"""
    
    def __init__(self, x: float, y: float, width: float, height: float, max_items: int = 10):
        self.bounds = (x, y, width, height)  # x, y, width, height
        self.max_items = max_items
        self.items: List[Dict[str, Any]] = []  # Nodes in this quadrant
        self.children: Optional[List['QuadTreeNode']] = None  # Sub-quadrants
        self.is_leaf = True
    
    def insert(self, item: Dict[str, Any]) -> bool:
        """Insert a node into the quadtree"""
        x, y, w, h = self.bounds
        item_x = item.get('x', 0)
        item_y = item.get('y', 0)
        
        # Check if item is within bounds
        if not (x <= item_x < x + w and y <= item_y < y + h):
            return False
        
        # If leaf and not full, add directly
        if self.is_leaf and len(self.items) < self.max_items:
            self.items.append(item)
            return True
        
        # Split if leaf and full
        if self.is_leaf:
            self._split()
        
        # Insert into children
        for child in self.children:
            if child.insert(item):
                return True
        
        return False
    
    def _split(self):
        """Split this node into 4 children"""
        x, y, w, h = self.bounds
        half_w = w / 2
        half_h = h / 2
        
        self.children = [
            QuadTreeNode(x, y, half_w, half_h, self.max_items),           # NW
            QuadTreeNode(x + half_w, y, half_w, half_h, self.max_items),  # NE
            QuadTreeNode(x, y + half_h, half_w, half_h, self.max_items),  # SW
            QuadTreeNode(x + half_w, y + half_h, half_w, half_h, self.max_items)  # SE
        ]
        
        # Redistribute items
        for item in self.items:
            for child in self.children:
                if child.insert(item):
                    break
        
        self.items = []
        self.is_leaf = False
    
    def query(self, query_x: float, query_y: float, query_w: float, query_h: float) -> List[Dict[str, Any]]:
        """Query nodes within a rectangular viewport"""
        results = []
        x, y, w, h = self.bounds
        
        # Check if query intersects with this node
        if not self._intersects(query_x, query_y, query_w, query_h):
            return results
        
        # Add items from this node
        if self.is_leaf:
            for item in self.items:
                item_x = item.get('x', 0)
                item_y = item.get('y', 0)
                if query_x <= item_x < query_x + query_w and query_y <= item_y < query_y + query_h:
                    results.append(item)
        else:
            # Query children
            for child in self.children:
                results.extend(child.query(query_x, query_y, query_w, query_h))
        
        return results
    
    def _intersects(self, qx: float, qy: float, qw: float, qh: float) -> bool:
        """Check if query rectangle intersects with this node"""
        x, y, w, h = self.bounds
        return not (qx + qw < x or qx > x + w or qy + qh < y or qy > y + h)


class SpatialIndex:
    """Spatial index for efficient viewport queries"""
    
    def __init__(self, nodes: List[Dict[str, Any]], bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Build spatial index from nodes
        
        Args:
            nodes: List of nodes with 'x', 'y' positions
            bounds: (x, y, width, height) of the entire graph area
        """
        if not nodes:
            self.quadtree = None
            return
        
        # Compute bounds if not provided
        if bounds is None:
            xs = [n.get('x', 0) for n in nodes if 'x' in n]
            ys = [n.get('y', 0) for n in nodes if 'y' in n]
            
            if not xs or not ys:
                self.quadtree = None
                return
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Add padding
            padding = max(max_x - min_x, max_y - min_y) * 0.1
            bounds = (
                min_x - padding,
                min_y - padding,
                max_x - min_x + 2 * padding,
                max_y - min_y + 2 * padding
            )
        
        self.bounds = bounds
        self.quadtree = QuadTreeNode(bounds[0], bounds[1], bounds[2], bounds[3])
        
        # Insert all nodes
        for node in nodes:
            self.quadtree.insert(node)
    
    def query_viewport(self, x: float, y: float, width: float, height: float, padding: float = 100) -> List[Dict[str, Any]]:
        """
        Query nodes within viewport with padding for smooth scrolling
        
        Args:
            x, y: Viewport top-left position
            width, height: Viewport dimensions
            padding: Extra padding to load (for smooth scrolling)
        
        Returns:
            List of nodes within the viewport
        """
        if self.quadtree is None:
            return []
        
        # Add padding
        query_x = x - padding
        query_y = y - padding
        query_w = width + 2 * padding
        query_h = height + 2 * padding
        
        return self.quadtree.query(query_x, query_y, query_w, query_h)
    
    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the total bounds of the graph"""
        return self.bounds


class ViewportManager:
    """Manages viewport state and node visibility"""
    
    def __init__(self, graph_data: Dict[str, Any]):
        """
        Initialize viewport manager
        
        Args:
            graph_data: Graph data with 'nodes' and 'edges'
        """
        self.all_nodes = graph_data.get('nodes', [])
        self.all_edges = graph_data.get('edges', [])
        self.tensors = graph_data.get('tensors', {})
        
        # Build node ID to index mapping
        self.node_id_to_idx = {n['id']: i for i, n in enumerate(self.all_nodes)}
        
        # Build spatial index (only if we have positions)
        nodes_with_pos = [n for n in self.all_nodes if 'x' in n and 'y' in n]
        self.spatial_index = SpatialIndex(nodes_with_pos) if nodes_with_pos else None
        
        # Current viewport
        self.viewport = {'x': 0, 'y': 0, 'width': 1000, 'height': 800}
        
        # Visible nodes cache
        self.visible_node_ids: set = set()
        self.visible_nodes: List[Dict[str, Any]] = []
        self.visible_edges: List[Dict[str, Any]] = []
    
    def update_viewport(self, x: float, y: float, width: float, height: float) -> Dict[str, Any]:
        """
        Update viewport and return visible nodes/edges
        
        Returns:
            Dict with 'nodes', 'edges', 'total_nodes', 'total_edges'
        """
        self.viewport = {'x': x, 'y': y, 'width': width, 'height': height}
        
        if not self.spatial_index:
            # No spatial index, return all nodes (fallback)
            return {
                'nodes': self.all_nodes,
                'edges': self.all_edges,
                'tensors': self.tensors,
                'total_nodes': len(self.all_nodes),
                'total_edges': len(self.all_edges),
                'viewport': self.viewport,
                'bounds': None
            }
        
        # Query visible nodes
        visible_nodes = self.spatial_index.query_viewport(x, y, width, height)
        visible_node_ids = {n['id'] for n in visible_nodes}
        
        # Filter edges to only those with both endpoints visible
        visible_edges = []
        for edge in self.all_edges:
            if edge.get('source') in visible_node_ids and edge.get('target') in visible_node_ids:
                visible_edges.append(edge)
        
        self.visible_node_ids = visible_node_ids
        self.visible_nodes = visible_nodes
        self.visible_edges = visible_edges
        
        return {
            'nodes': visible_nodes,
            'edges': visible_edges,
            'tensors': self.tensors,
            'total_nodes': len(self.all_nodes),
            'total_edges': len(self.all_edges),
            'visible_nodes': len(visible_nodes),
            'visible_edges': len(visible_edges),
            'viewport': self.viewport,
            'bounds': self.spatial_index.get_bounds()
        }
    
    def get_all_nodes_count(self) -> int:
        return len(self.all_nodes)
    
    def get_visible_nodes_count(self) -> int:
        return len(self.visible_nodes)
