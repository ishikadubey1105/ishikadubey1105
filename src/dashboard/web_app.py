"""
Web Dashboard for Smart Traffic Light System
Real-time monitoring and control interface
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import time
import logging
from typing import Dict, Any
import threading
import json

from ..core.camera_processor import CameraProcessor

class TrafficDashboard:
    """
    Web dashboard for monitoring and controlling the smart traffic light system
    """
    
    def __init__(self, camera_processor: CameraProcessor, 
                 host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """
        Initialize web dashboard
        
        Args:
            camera_processor: Camera processor instance
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.camera_processor = camera_processor
        self.host = host
        self.port = port
        self.debug = debug
        
        self.logger = logging.getLogger(__name__)
        
        # Create Flask app
        self.app = Flask(__name__, 
                        template_folder='../../templates',
                        static_folder='../../static')
        self.app.config['SECRET_KEY'] = 'smart_traffic_system_2024'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.connected_clients = 0
        self.last_frame_data = {}
        self.last_analysis_data = {}
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        self._setup_callbacks()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status"""
            try:
                status = self.camera_processor.get_system_status()
                return jsonify({
                    'success': True,
                    'data': status,
                    'timestamp': time.time()
                })
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/traffic_lights')
        def get_traffic_lights():
            """Get current traffic light states"""
            try:
                lights = self.camera_processor.get_current_traffic_lights()
                return jsonify({
                    'success': True,
                    'data': lights,
                    'timestamp': time.time()
                })
            except Exception as e:
                self.logger.error(f"Error getting traffic lights: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/emergency', methods=['POST'])
        def set_emergency():
            """Set emergency mode"""
            try:
                data = request.get_json()
                direction = data.get('direction')
                
                if data.get('clear', False):
                    self.camera_processor.clear_emergency_mode()
                    return jsonify({
                        'success': True,
                        'message': 'Emergency mode cleared'
                    })
                else:
                    self.camera_processor.set_emergency_mode(direction)
                    return jsonify({
                        'success': True,
                        'message': f'Emergency mode set for {direction or "all directions"}'
                    })
                    
            except Exception as e:
                self.logger.error(f"Error setting emergency mode: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/analytics')
        def get_analytics():
            """Get traffic analytics data"""
            try:
                # Get recent analysis data
                analytics = {
                    'traffic_analysis': self.last_analysis_data.get('traffic_analysis', {}),
                    'controller_status': self.last_analysis_data.get('controller_status', {}),
                    'system_stats': self.camera_processor.get_system_status(),
                    'timestamp': time.time()
                }
                
                return jsonify({
                    'success': True,
                    'data': analytics
                })
                
            except Exception as e:
                self.logger.error(f"Error getting analytics: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def _setup_socket_handlers(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.connected_clients += 1
            self.logger.info(f"Client connected. Total clients: {self.connected_clients}")
            
            # Send initial data
            emit('system_status', {
                'status': self.camera_processor.get_system_status(),
                'timestamp': time.time()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.connected_clients = max(0, self.connected_clients - 1)
            self.logger.info(f"Client disconnected. Total clients: {self.connected_clients}")
        
        @self.socketio.on('request_frame')
        def handle_frame_request(data):
            """Handle frame request from client"""
            try:
                camera_id = data.get('camera_id', 'camera_0')
                
                if camera_id in self.last_frame_data:
                    frame_data = self.last_frame_data[camera_id]
                    
                    # Encode frame as base64
                    frame_b64 = self._encode_frame(frame_data['frame'])
                    
                    emit('frame_data', {
                        'camera_id': camera_id,
                        'frame': frame_b64,
                        'detections': len(frame_data.get('detections', [])),
                        'timestamp': frame_data.get('timestamp', time.time())
                    })
                    
            except Exception as e:
                self.logger.error(f"Error handling frame request: {e}")
        
        @self.socketio.on('emergency_control')
        def handle_emergency(data):
            """Handle emergency control"""
            try:
                action = data.get('action')
                direction = data.get('direction')
                
                if action == 'set':
                    self.camera_processor.set_emergency_mode(direction)
                    emit('emergency_response', {
                        'success': True,
                        'message': f'Emergency mode activated for {direction or "all directions"}'
                    })
                elif action == 'clear':
                    self.camera_processor.clear_emergency_mode()
                    emit('emergency_response', {
                        'success': True,
                        'message': 'Emergency mode cleared'
                    })
                    
            except Exception as e:
                self.logger.error(f"Error handling emergency control: {e}")
                emit('emergency_response', {
                    'success': False,
                    'error': str(e)
                })
    
    def _setup_callbacks(self):
        """Setup callbacks for camera processor"""
        
        def frame_callback(processed_frames):
            """Handle processed frames"""
            self.last_frame_data = processed_frames
            
            if self.connected_clients > 0:
                # Broadcast frame updates to connected clients
                for camera_id, frame_data in processed_frames.items():
                    frame_b64 = self._encode_frame(frame_data['frame'])
                    
                    self.socketio.emit('frame_update', {
                        'camera_id': camera_id,
                        'frame': frame_b64,
                        'detections': len(frame_data.get('detections', [])),
                        'processing_time': frame_data.get('processing_time', 0),
                        'timestamp': frame_data.get('timestamp', time.time())
                    })
        
        def data_callback(analysis_data):
            """Handle analysis data"""
            self.last_analysis_data = analysis_data
            
            if self.connected_clients > 0:
                # Broadcast analysis updates
                self.socketio.emit('analysis_update', {
                    'traffic_analysis': self._serialize_traffic_data(
                        analysis_data.get('traffic_analysis', {})
                    ),
                    'controller_status': analysis_data.get('controller_status', {}),
                    'total_detections': len(analysis_data.get('detections', [])),
                    'timestamp': analysis_data.get('timestamp', time.time())
                })
        
        # Register callbacks
        self.camera_processor.add_frame_callback(frame_callback)
        self.camera_processor.add_data_callback(data_callback)
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 string"""
        try:
            # Resize frame for web display (optional)
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert to base64
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_b64}"
            
        except Exception as e:
            self.logger.error(f"Error encoding frame: {e}")
            return ""
    
    def _serialize_traffic_data(self, traffic_data: Dict) -> Dict:
        """Serialize traffic data for JSON transmission"""
        serialized = {}
        
        for direction, data in traffic_data.items():
            serialized[direction] = {
                'direction': data.direction,
                'vehicle_count': data.vehicle_count,
                'density_level': data.density_level.value,
                'avg_vehicle_size': data.avg_vehicle_size,
                'vehicle_types': data.vehicle_types,
                'confidence_score': data.confidence_score,
                'timestamp': data.timestamp
            }
        
        return serialized
    
    def start_dashboard(self):
        """Start the web dashboard"""
        try:
            self.logger.info(f"Starting traffic dashboard on {self.host}:{self.port}")
            self.socketio.run(self.app, 
                            host=self.host, 
                            port=self.port, 
                            debug=self.debug,
                            allow_unsafe_werkzeug=True)
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
            raise
    
    def stop_dashboard(self):
        """Stop the web dashboard"""
        # SocketIO doesn't have a direct stop method, but we can shutdown the server
        self.logger.info("Stopping traffic dashboard")
        # In a production environment, you'd want to implement proper shutdown