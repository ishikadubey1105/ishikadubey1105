"""
Real-time Camera Feed Processor
Processes live camera feeds and coordinates traffic analysis and control
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import List, Dict, Optional, Callable, Any
from collections import deque
import queue
from dataclasses import dataclass

from .vehicle_detector import VehicleDetector, Detection
from .traffic_analyzer import TrafficAnalyzer, TrafficData
from .traffic_controller import TrafficLightController

@dataclass
class CameraSource:
    """Camera source configuration"""
    source_id: str
    source_path: Any  # Can be int (device), str (file/URL)
    name: str
    active: bool = True
    fps: float = 30.0
    resolution: tuple = (1280, 720)

@dataclass
class ProcessingStats:
    """Processing statistics"""
    frames_processed: int = 0
    detections_made: int = 0
    avg_fps: float = 0.0
    avg_processing_time: float = 0.0
    last_update_time: float = 0.0

class CameraProcessor:
    """
    Real-time camera feed processor that orchestrates the entire traffic system
    """
    
    def __init__(self, 
                 config: Dict,
                 detector: Optional[VehicleDetector] = None,
                 analyzer: Optional[TrafficAnalyzer] = None,
                 controller: Optional[TrafficLightController] = None):
        """
        Initialize camera processor
        
        Args:
            config: Configuration dictionary
            detector: Vehicle detector instance
            analyzer: Traffic analyzer instance
            controller: Traffic light controller instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector = detector or self._create_detector()
        self.analyzer = analyzer or self._create_analyzer()
        self.controller = controller or self._create_controller()
        
        # Camera sources
        self.camera_sources = {}
        self.video_captures = {}
        self._initialize_cameras()
        
        # Processing state
        self.running = False
        self.processing_threads = {}
        self.frame_queues = {}
        self.result_queues = {}
        
        # Statistics and monitoring
        self.stats = ProcessingStats()
        self.performance_history = deque(maxlen=100)
        
        # Frame callbacks for external consumers (like web dashboard)
        self.frame_callbacks = []
        self.data_callbacks = []
        
        # Threading coordination
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
    
    def _create_detector(self) -> VehicleDetector:
        """Create vehicle detector from config"""
        model_config = self.config.get('model', {})
        return VehicleDetector(
            model_path=model_config.get('name', 'yolov8n.pt'),
            confidence=model_config.get('confidence', 0.5),
            iou_threshold=model_config.get('iou_threshold', 0.45),
            device=model_config.get('device', 'cpu'),
            vehicle_classes=self.config.get('traffic_analysis', {}).get('vehicle_classes', [2, 3, 5, 7])
        )
    
    def _create_analyzer(self) -> TrafficAnalyzer:
        """Create traffic analyzer from config"""
        analysis_config = self.config.get('traffic_analysis', {})
        camera_config = self.config.get('camera', {})
        
        return TrafficAnalyzer(
            roi_config=analysis_config.get('roi', {}),
            density_thresholds=analysis_config.get('density_thresholds', {}),
            analysis_window=analysis_config.get('analysis_window', 10.0),
            frame_dimensions=(
                camera_config.get('resolution', {}).get('width', 1280),
                camera_config.get('resolution', {}).get('height', 720)
            )
        )
    
    def _create_controller(self) -> TrafficLightController:
        """Create traffic controller from config"""
        controller_config = self.config.get('traffic_light', {})
        analysis_config = self.config.get('traffic_analysis', {})
        
        directions = list(analysis_config.get('roi', {}).keys())
        
        return TrafficLightController(
            directions=directions,
            default_timing=controller_config.get('default_timing', {}),
            timing_ranges=controller_config.get('timing_ranges', {}),
            min_cycle_time=controller_config.get('min_cycle_time', 60.0),
            max_cycle_time=controller_config.get('max_cycle_time', 180.0)
        )
    
    def _initialize_cameras(self) -> None:
        """Initialize camera sources"""
        camera_config = self.config.get('camera', {})
        sources = camera_config.get('sources', [0])
        
        for i, source in enumerate(sources):
            source_id = f"camera_{i}"
            name = f"Camera {i+1}"
            
            self.camera_sources[source_id] = CameraSource(
                source_id=source_id,
                source_path=source,
                name=name,
                fps=camera_config.get('fps', 30.0),
                resolution=(
                    camera_config.get('resolution', {}).get('width', 1280),
                    camera_config.get('resolution', {}).get('height', 720)
                )
            )
            
            # Initialize frame and result queues
            self.frame_queues[source_id] = queue.Queue(maxsize=5)
            self.result_queues[source_id] = queue.Queue(maxsize=10)
        
        self.logger.info(f"Initialized {len(self.camera_sources)} camera sources")
    
    def start_processing(self) -> bool:
        """Start camera processing"""
        if self.running:
            self.logger.warning("Processing already running")
            return False
        
        try:
            # Open camera captures
            for source_id, camera_source in self.camera_sources.items():
                if camera_source.active:
                    cap = cv2.VideoCapture(camera_source.source_path)
                    if cap.isOpened():
                        # Set resolution if possible
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_source.resolution[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_source.resolution[1])
                        cap.set(cv2.CAP_PROP_FPS, camera_source.fps)
                        
                        self.video_captures[source_id] = cap
                        self.logger.info(f"Opened camera: {camera_source.name}")
                    else:
                        self.logger.error(f"Failed to open camera: {camera_source.name}")
                        continue
            
            if not self.video_captures:
                self.logger.error("No cameras could be opened")
                return False
            
            # Start processing threads
            self.running = True
            self.stop_event.clear()
            
            # Start frame capture threads
            for source_id in self.video_captures.keys():
                thread = threading.Thread(
                    target=self._capture_frames,
                    args=(source_id,),
                    daemon=True,
                    name=f"capture_{source_id}"
                )
                thread.start()
                self.processing_threads[f"capture_{source_id}"] = thread
            
            # Start processing thread
            process_thread = threading.Thread(
                target=self._process_frames,
                daemon=True,
                name="frame_processor"
            )
            process_thread.start()
            self.processing_threads["processor"] = process_thread
            
            # Start traffic controller
            self.controller.start_control_loop()
            
            # Start statistics thread
            stats_thread = threading.Thread(
                target=self._update_statistics,
                daemon=True,
                name="statistics"
            )
            stats_thread.start()
            self.processing_threads["statistics"] = stats_thread
            
            self.logger.info("Camera processing started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self.stop_processing()
            return False
    
    def stop_processing(self) -> None:
        """Stop camera processing"""
        if not self.running:
            return
        
        self.logger.info("Stopping camera processing...")
        
        # Signal stop
        self.running = False
        self.stop_event.set()
        
        # Stop traffic controller
        self.controller.stop_control_loop()
        
        # Wait for threads to finish
        for thread_name, thread in self.processing_threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread_name} did not stop gracefully")
        
        # Close video captures
        for source_id, cap in self.video_captures.items():
            cap.release()
        
        # Clear data structures
        self.video_captures.clear()
        self.processing_threads.clear()
        
        # Clear queues
        for frame_queue in self.frame_queues.values():
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
        
        for result_queue in self.result_queues.values():
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
        
        self.logger.info("Camera processing stopped")
    
    def _capture_frames(self, source_id: str) -> None:
        """Capture frames from a specific camera"""
        cap = self.video_captures.get(source_id)
        if not cap:
            return
        
        frame_queue = self.frame_queues[source_id]
        camera_source = self.camera_sources[source_id]
        
        self.logger.info(f"Started frame capture for {camera_source.name}")
        
        while self.running and not self.stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    # Handle end of video file or camera disconnection
                    if isinstance(camera_source.source_path, str):
                        # For video files, restart from beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # For cameras, try to reconnect
                        self.logger.warning(f"Camera {camera_source.name} disconnected")
                        time.sleep(1.0)
                        continue
                
                # Add frame to queue (non-blocking)
                try:
                    frame_data = {
                        'source_id': source_id,
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_number': cap.get(cv2.CAP_PROP_POS_FRAMES)
                    }
                    frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                # Control frame rate
                time.sleep(1.0 / camera_source.fps)
                
            except Exception as e:
                self.logger.error(f"Error capturing frames from {camera_source.name}: {e}")
                time.sleep(1.0)
        
        self.logger.info(f"Stopped frame capture for {camera_source.name}")
    
    def _process_frames(self) -> None:
        """Process frames from all cameras"""
        self.logger.info("Started frame processing")
        
        while self.running and not self.stop_event.is_set():
            try:
                # Collect frames from all active cameras
                current_frames = {}
                
                for source_id, frame_queue in self.frame_queues.items():
                    try:
                        frame_data = frame_queue.get(timeout=0.1)
                        current_frames[source_id] = frame_data
                    except queue.Empty:
                        continue
                
                if not current_frames:
                    continue
                
                # Process each frame
                all_detections = []
                processed_frames = {}
                
                for source_id, frame_data in current_frames.items():
                    start_time = time.time()
                    
                    # Detect vehicles
                    detections = self.detector.detect_vehicles(frame_data['frame'])
                    all_detections.extend(detections)
                    
                    # Draw detections on frame
                    roi_boxes = self.analyzer.get_roi_boxes()
                    processed_frame = self.detector.draw_detections(
                        frame_data['frame'], detections, roi_boxes
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Store processed frame data
                    processed_frames[source_id] = {
                        'frame': processed_frame,
                        'detections': detections,
                        'processing_time': processing_time,
                        'timestamp': frame_data['timestamp']
                    }
                    
                    # Add to result queue
                    try:
                        self.result_queues[source_id].put_nowait({
                            'source_id': source_id,
                            'frame': processed_frame,
                            'detections': detections,
                            'timestamp': frame_data['timestamp']
                        })
                    except queue.Full:
                        # Remove oldest result if queue is full
                        try:
                            self.result_queues[source_id].get_nowait()
                            self.result_queues[source_id].put_nowait({
                                'source_id': source_id,
                                'frame': processed_frame,
                                'detections': detections,
                                'timestamp': frame_data['timestamp']
                            })
                        except queue.Empty:
                            pass
                
                # Analyze traffic across all cameras
                if all_detections:
                    traffic_analysis = self.analyzer.analyze_traffic(all_detections)
                    
                    # Update traffic controller
                    self.controller.update_traffic_data(traffic_analysis)
                    
                    # Call data callbacks
                    for callback in self.data_callbacks:
                        try:
                            callback({
                                'traffic_analysis': traffic_analysis,
                                'controller_status': self.controller.get_current_states(),
                                'detections': all_detections,
                                'timestamp': time.time()
                            })
                        except Exception as e:
                            self.logger.error(f"Error in data callback: {e}")
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(processed_frames)
                    except Exception as e:
                        self.logger.error(f"Error in frame callback: {e}")
                
                # Update statistics
                with self.lock:
                    self.stats.frames_processed += len(current_frames)
                    self.stats.detections_made += len(all_detections)
                
            except Exception as e:
                self.logger.error(f"Error in frame processing: {e}")
                time.sleep(0.1)
        
        self.logger.info("Stopped frame processing")
    
    def _update_statistics(self) -> None:
        """Update processing statistics"""
        last_frames = 0
        last_time = time.time()
        
        while self.running and not self.stop_event.is_set():
            try:
                time.sleep(1.0)  # Update every second
                
                current_time = time.time()
                
                with self.lock:
                    # Calculate FPS
                    frames_diff = self.stats.frames_processed - last_frames
                    time_diff = current_time - last_time
                    
                    if time_diff > 0:
                        current_fps = frames_diff / time_diff
                        self.stats.avg_fps = (self.stats.avg_fps * 0.9 + current_fps * 0.1)
                    
                    self.stats.last_update_time = current_time
                    
                    # Store performance data
                    self.performance_history.append({
                        'timestamp': current_time,
                        'fps': self.stats.avg_fps,
                        'frames_processed': self.stats.frames_processed,
                        'detections_made': self.stats.detections_made
                    })
                
                last_frames = self.stats.frames_processed
                last_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error updating statistics: {e}")
    
    def get_latest_results(self, source_id: str = None) -> Dict:
        """Get latest processing results"""
        if source_id:
            # Get results for specific camera
            if source_id in self.result_queues:
                try:
                    return self.result_queues[source_id].get_nowait()
                except queue.Empty:
                    return {}
            return {}
        else:
            # Get results for all cameras
            results = {}
            for sid, result_queue in self.result_queues.items():
                try:
                    results[sid] = result_queue.get_nowait()
                except queue.Empty:
                    continue
            return results
    
    def add_frame_callback(self, callback: Callable) -> None:
        """Add callback for processed frames"""
        self.frame_callbacks.append(callback)
    
    def add_data_callback(self, callback: Callable) -> None:
        """Add callback for analysis data"""
        self.data_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable) -> None:
        """Remove frame callback"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def remove_data_callback(self, callback: Callable) -> None:
        """Remove data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        with self.lock:
            return {
                'processing': {
                    'running': self.running,
                    'active_cameras': len([s for s in self.camera_sources.values() if s.active]),
                    'total_cameras': len(self.camera_sources),
                    'fps': self.stats.avg_fps,
                    'frames_processed': self.stats.frames_processed,
                    'detections_made': self.stats.detections_made
                },
                'traffic_controller': self.controller.get_system_status(),
                'traffic_analyzer': self.analyzer.get_system_stats(),
                'cameras': {
                    sid: {
                        'name': camera.name,
                        'active': camera.active,
                        'source': str(camera.source_path)
                    }
                    for sid, camera in self.camera_sources.items()
                }
            }
    
    def get_current_traffic_lights(self) -> Dict:
        """Get current traffic light states"""
        return self.controller.get_current_states()
    
    def set_emergency_mode(self, direction: str = None) -> None:
        """Set emergency mode"""
        self.controller.set_emergency_mode(direction)
    
    def clear_emergency_mode(self) -> None:
        """Clear emergency mode"""
        self.controller.clear_emergency_mode()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_processing()