#!/usr/bin/env python3
"""
Smart Traffic Light System - Main Application
Real-time traffic management using AI and computer vision
"""

import os
import sys
import argparse
import logging
import signal
import threading
import time
import yaml
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.camera_processor import CameraProcessor
from src.dashboard.web_app import TrafficDashboard

class SmartTrafficSystem:
    """
    Main application class for the Smart Traffic Light System
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Smart Traffic System
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.camera_processor = None
        self.dashboard = None
        self.dashboard_thread = None
        
        # System state
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("Smart Traffic Light System initialized")
    
    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Configuration file {self.config_path} not found. Using defaults.")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'camera': {
                'sources': [0],
                'resolution': {'width': 1280, 'height': 720},
                'fps': 30
            },
            'model': {
                'name': 'yolov8n.pt',
                'confidence': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu'
            },
            'traffic_analysis': {
                'vehicle_classes': [2, 3, 5, 7],
                'density_thresholds': {'low': 5, 'medium': 15, 'high': 30},
                'analysis_window': 10.0,
                'roi': {
                    'north': [0.1, 0.1, 0.9, 0.4],
                    'south': [0.1, 0.6, 0.9, 0.9],
                    'east': [0.6, 0.1, 0.9, 0.9],
                    'west': [0.1, 0.1, 0.4, 0.9]
                }
            },
            'traffic_light': {
                'default_timing': {'green': 30, 'yellow': 5, 'red': 35},
                'timing_ranges': {
                    'green_min': 15, 'green_max': 60, 'yellow': 5,
                    'red_min': 10, 'red_max': 45
                },
                'min_cycle_time': 60,
                'max_cycle_time': 180
            },
            'dashboard': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/traffic_system.log'
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_file = log_config.get('file', 'logs/traffic_system.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_components(self):
        """Initialize system components"""
        try:
            # Initialize camera processor
            self.logger.info("Initializing camera processor...")
            self.camera_processor = CameraProcessor(self.config)
            
            # Initialize dashboard
            self.logger.info("Initializing web dashboard...")
            dashboard_config = self.config.get('dashboard', {})
            self.dashboard = TrafficDashboard(
                camera_processor=self.camera_processor,
                host=dashboard_config.get('host', '0.0.0.0'),
                port=dashboard_config.get('port', 5000),
                debug=dashboard_config.get('debug', False)
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start_system(self):
        """Start the traffic system"""
        try:
            self.logger.info("Starting Smart Traffic Light System...")
            
            # Initialize components
            self.initialize_components()
            
            # Start camera processing
            if not self.camera_processor.start_processing():
                raise RuntimeError("Failed to start camera processing")
            
            # Start dashboard in separate thread
            self.dashboard_thread = threading.Thread(
                target=self.dashboard.start_dashboard,
                daemon=True,
                name="dashboard_thread"
            )
            self.dashboard_thread.start()
            
            self.running = True
            self.logger.info("Smart Traffic Light System started successfully")
            
            # Print system information
            self.print_system_info()
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop_system()
            raise
    
    def stop_system(self):
        """Stop the traffic system"""
        if not self.running:
            return
        
        self.logger.info("Stopping Smart Traffic Light System...")
        self.running = False
        self.shutdown_event.set()
        
        # Stop camera processor
        if self.camera_processor:
            self.camera_processor.stop_processing()
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop_dashboard()
        
        # Wait for dashboard thread
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5.0)
        
        self.logger.info("Smart Traffic Light System stopped")
    
    def run(self):
        """Main run loop"""
        try:
            self.start_system()
            
            # Keep the main thread alive
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1.0)
                
                # Perform periodic maintenance
                self.perform_maintenance()
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.stop_system()
    
    def perform_maintenance(self):
        """Perform periodic system maintenance"""
        # This could include:
        # - Checking system health
        # - Cleaning up resources
        # - Logging statistics
        # - Restarting failed components
        pass
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        self.running = False
        self.shutdown_event.set()
    
    def print_system_info(self):
        """Print system information"""
        dashboard_config = self.config.get('dashboard', {})
        host = dashboard_config.get('host', '0.0.0.0')
        port = dashboard_config.get('port', 5000)
        
        print("\n" + "="*60)
        print("🚦 SMART TRAFFIC LIGHT SYSTEM")
        print("="*60)
        print(f"📊 Dashboard: http://{host}:{port}")
        print(f"📹 Cameras: {len(self.config.get('camera', {}).get('sources', []))}")
        print(f"🛣️  Directions: {len(self.config.get('traffic_analysis', {}).get('roi', {}))}")
        print(f"🤖 AI Model: {self.config.get('model', {}).get('name', 'yolov8n.pt')}")
        print("="*60)
        print("System is running. Press Ctrl+C to stop.")
        print("="*60 + "\n")

def create_sample_video():
    """Create a sample traffic video for testing"""
    import cv2
    import numpy as np
    
    # Create sample videos directory
    os.makedirs('sample_videos', exist_ok=True)
    
    # Video parameters
    width, height = 1280, 720
    fps = 30
    duration = 30  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_videos/traffic1.mp4', fourcc, fps, (width, height))
    
    for frame_num in range(fps * duration):
        # Create a simple animated background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Draw road markings
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
        cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
        # Draw moving rectangles to simulate vehicles
        for i in range(3):
            x = (frame_num * 5 + i * 200) % (width + 100) - 50
            y = height//2 + 50 + i * 30
            cv2.rectangle(frame, (x, y), (x+60, y+30), (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    print("Sample video created: sample_videos/traffic1.mp4")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Traffic Light System')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample video for testing')
    parser.add_argument('--simulate', action='store_true',
                       help='Run in simulation mode with sample data')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_video()
        return
    
    # Modify config for simulation mode
    if args.simulate:
        print("Running in simulation mode...")
        # This would modify the config to use sample videos instead of cameras
    
    try:
        # Create and run the traffic system
        system = SmartTrafficSystem(args.config)
        system.run()
        
    except Exception as e:
        print(f"Failed to start system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()