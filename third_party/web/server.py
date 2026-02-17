#!/usr/bin/env python3
"""
Simple HTTP server for Oniris
"""
import http.server
import socketserver
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app import model_to_graph_data, compute_layout
import onnx

PORT = 5000  # Use port 5006
socketserver.TCPServer.allow_reuse_address = True
model_cache = {}

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{datetime.now()}] {format % args}")
    
    def do_GET(self):
        static_dir = Path(__file__).parent / 'frontend'
        template_dir = Path(__file__).parent / 'templates'
        
        if self.path.startswith('/frontend/'):
            file_path = static_dir / self.path[10:]
            if file_path.exists():
                self.send_file(file_path)
                return
        
        if self.path == '/' or self.path == '/index.html':
            index_path = template_dir / 'index.html'
            if index_path.exists():
                self.send_file(index_path, 'text/html')
                return
        
        self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/model/upload':
            self.handle_upload()
        elif self.path.startswith('/api/model/') and self.path.endswith('/shape_inference'):
            self.handle_shape_inference()
        elif self.path.startswith('/api/model/') and self.path.endswith('/simplify'):
            self.handle_simplify()
        else:
            self.send_error(404)
    
    def handle_upload(self):
        try:
            print(f"[{datetime.now()}] Upload started")
            
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json({'error': 'Empty body'}, 400)
                return
            
            # Read the multipart body
            body = self.rfile.read(content_length)
            
            # Parse boundary
            content_type = self.headers.get('Content-Type', '')
            boundary = content_type.split('boundary=')[1].split(';')[0].strip('"')
            
            # Find file data by locating the file input in the multipart body
            boundary_bytes = boundary.encode()
            file_data = None
            
            # Find the file input section
            file_pos = body.find(b'name="file"')
            if file_pos >= 0:
                # Find the Content-Type header for this file
                content_type_pos = body.find(b'Content-Type:', file_pos)
                if content_type_pos >= 0:
                    # Find the end of headers (double CRLF or LF)
                    header_end = body.find(b'\r\n\r\n', content_type_pos)
                    offset = 4
                    if header_end < 0:
                        header_end = body.find(b'\n\n', content_type_pos)
                        offset = 2
                    
                    if header_end >= 0:
                        file_start = header_end + offset
                        # Find the next boundary (boundary line starts with --)
                        boundary_line = b'--' + boundary_bytes
                        next_boundary = body.find(b'\r\n' + boundary_line, file_start)
                        if next_boundary < 0:
                            next_boundary = body.find(b'\n' + boundary_line, file_start)
                        if next_boundary < 0:
                            next_boundary = body.find(boundary_line, file_start)
                        
                        if next_boundary > file_start:
                            file_data = body[file_start:next_boundary]
                            # Remove trailing \r\n or \n (multipart delimiter before boundary)
                            if file_data.endswith(b'\r\n'):
                                file_data = file_data[:-2]
                            elif file_data.endswith(b'\n'):
                                file_data = file_data[:-1]
            
            if not file_data:
                self.send_json({'error': 'No file'}, 400)
                return
            
            print(f"[{datetime.now()}] Loading ONNX model ({len(file_data)} bytes)")
            
            # Load model
            try:
                model = onnx.load_model_from_string(file_data)
            except Exception as e:
                self.send_json({'error': f'Invalid ONNX: {e}'}, 400)
                return
            
            # Process
            import uuid
            session_id = str(uuid.uuid4())[:8]
            model_cache[session_id] = {'model': model}
            
            graph_data = model_to_graph_data(model)
            graph_data = compute_layout(graph_data, 'hierarchical')
            
            response = {
                'success': True,
                'session_id': session_id,
                'info': {
                    'file_name': 'model.onnx',
                    'file_size': len(file_data),
                    'num_nodes': len(model.graph.node),
                    'num_edges': len(graph_data['edges'])
                },
                'graph': graph_data
            }
            
            print(f"[{datetime.now()}] Sending response ({len(json.dumps(response))} bytes)")
            self.send_json(response, 200)
            
        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def send_file(self, path, content_type='application/octet-stream'):
        ext = path.suffix.lower()
        types = {'.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript'}
        content_type = types.get(ext, content_type)
        
        with open(path, 'rb') as f:
            data = f.read()
        
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)
    
    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
    
    def handle_shape_inference(self):
        try:
            # Extract session_id from path: /api/model/{session_id}/shape_inference
            parts = self.path.split('/')
            if len(parts) < 5:
                self.send_json({'error': 'Invalid path'}, 400)
                return
            session_id = parts[3]
            
            if session_id not in model_cache:
                self.send_json({'error': 'Session not found'}, 404)
                return
            
            model = model_cache[session_id]['model']
            
            # Run shape inference
            try:
                inferred_model = onnx.shape_inference.infer_shapes(model)
                model_cache[session_id]['model'] = inferred_model
            except Exception as e:
                self.send_json({'error': f'Shape inference failed: {e}'}, 500)
                return
            
            graph_data = model_to_graph_data(inferred_model)
            graph_data = compute_layout(graph_data, 'hierarchical')
            
            self.send_json({
                'success': True,
                'graph': graph_data,
                'message': 'Shape inference completed successfully'
            })
        except Exception as e:
            print(f"[{datetime.now()}] Error in shape inference: {e}")
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def handle_simplify(self):
        try:
            # Extract session_id from path: /api/model/{session_id}/simplify
            parts = self.path.split('/')
            if len(parts) < 5:
                self.send_json({'error': 'Invalid path'}, 400)
                return
            session_id = parts[3]
            
            if session_id not in model_cache:
                self.send_json({'error': 'Session not found'}, 404)
                return
            
            model = model_cache[session_id]['model']
            
            # Try to simplify using onnxsim
            try:
                import onnxsim
                simplified, check = onnxsim.simplify(model)
                if not check:
                    self.send_json({'error': 'Simplified model check failed'}, 500)
                    return
                model_cache[session_id]['model'] = simplified
            except ImportError:
                self.send_json({'error': 'onnx-simplifier not installed. Run: pip install onnxsim'}, 500)
                return
            except Exception as e:
                self.send_json({'error': f'Simplification failed: {e}'}, 500)
                return
            
            graph_data = model_to_graph_data(simplified)
            graph_data = compute_layout(graph_data, 'hierarchical')
            
            self.send_json({
                'success': True,
                'graph': graph_data,
                'message': 'Model simplified successfully'
            })
        except Exception as e:
            print(f"[{datetime.now()}] Error in simplify: {e}")
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server running on http://localhost:{PORT}")
        httpd.serve_forever()
