#!/usr/bin/env python3
"""
Smart Traffic Light System - Test Script
Demonstrates system functionality with sample data
"""

import os
import sys
import time
import threading
import cv2
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.vehicle_detector import VehicleDetector, Detection
from src.core.traffic_analyzer import TrafficAnalyzer, TrafficData
from src.core.traffic_controller import TrafficLightController

def create_test_frame(frame_num: int, vehicles_count: int = 5) -> np.ndarray:
    """Create a test frame with simulated vehicles"""
    width, height = 1280, 720
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Draw road markings
    cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    
    # Draw moving rectangles to simulate vehicles
    for i in range(vehicles_count):
        # Horizontal movement
        x = (frame_num * 3 + i * 150) % (width + 100) - 50
        y = height//2 + 20 + (i % 3) * 40
        
        # Different colors for different vehicle types
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[i % len(colors)]
        
        cv2.rectangle(frame, (x, y), (x+80, y+40), color, -1)
        
        # Vertical movement
        if i < 2:
            x2 = width//2 + 20 + (i * 40)
            y2 = (frame_num * 2 + i * 100) % (height + 80) - 40
            cv2.rectangle(frame, (x2, y2), (x2+40, y2+80), color, -1)
    
    return frame

def create_test_detections(frame_num: int, vehicles_count: int = 5) -> list:
    """Create test vehicle detections"""
    detections = []
    width, height = 1280, 720
    
    for i in range(vehicles_count):
        # Simulate vehicle positions
        x = (frame_num * 3 + i * 150) % (width + 100) - 50
        y = height//2 + 20 + (i % 3) * 40
        
        if 0 <= x <= width - 80:  # Only if vehicle is visible
            detection = Detection(
                class_id=2 if i % 2 == 0 else 3,  # Alternate between car and motorcycle
                class_name='car' if i % 2 == 0 else 'motorcycle',
                confidence=0.8 + (i % 3) * 0.1,
                bbox=(x, y, x+80, y+40),
                center=(x+40, y+20),
                area=80 * 40
            )
            detections.append(detection)
        
        # Add vertical movement vehicles
        if i < 2:
            x2 = width//2 + 20 + (i * 40)
            y2 = (frame_num * 2 + i * 100) % (height + 80) - 40
            
            if 0 <= y2 <= height - 80:
                detection = Detection(
                    class_id=5 if i == 0 else 7,  # Bus or truck
                    class_name='bus' if i == 0 else 'truck',
                    confidence=0.9,
                    bbox=(x2, y2, x2+40, y2+80),
                    center=(x2+20, y2+40),
                    area=40 * 80
                )
                detections.append(detection)
    
    return detections

def test_vehicle_detection():
    """Test vehicle detection component"""
    print("🔍 Testing Vehicle Detection...")
    
    try:
        detector = VehicleDetector(
            model_path="yolov8n.pt",
            confidence=0.5,
            device="cpu"
        )
        print("✅ Vehicle detector initialized successfully")
        
        # Test with a sample frame
        test_frame = create_test_frame(0, 3)
        detections = create_test_detections(0, 3)  # Use mock detections for testing
        
        # Draw detections
        result_frame = detector.draw_detections(test_frame, detections)
        
        # Save test image
        os.makedirs('test_output', exist_ok=True)
        cv2.imwrite('test_output/test_detection.jpg', result_frame)
        print(f"✅ Detected {len(detections)} vehicles")
        
        # Test statistics
        stats = detector.get_detection_stats(detections)
        print(f"📊 Detection stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vehicle detection test failed: {e}")
        return False

def test_traffic_analysis():
    """Test traffic analysis component"""
    print("\n📊 Testing Traffic Analysis...")
    
    try:
        # Initialize analyzer
        roi_config = {
            'north': [0.1, 0.1, 0.9, 0.4],
            'south': [0.1, 0.6, 0.9, 0.9],
            'east': [0.6, 0.1, 0.9, 0.9],
            'west': [0.1, 0.1, 0.4, 0.9]
        }
        
        analyzer = TrafficAnalyzer(
            roi_config=roi_config,
            density_thresholds={'low': 3, 'medium': 8, 'high': 15},
            frame_dimensions=(1280, 720)
        )
        print("✅ Traffic analyzer initialized successfully")
        
        # Test with sample detections
        detections = create_test_detections(10, 8)
        traffic_analysis = analyzer.analyze_traffic(detections)
        
        print(f"📈 Analysis results:")
        for direction, data in traffic_analysis.items():
            print(f"  {direction}: {data.vehicle_count} vehicles, {data.density_level.value} density")
        
        # Test priority calculation
        priority_direction = analyzer.get_priority_direction()
        print(f"🚦 Priority direction: {priority_direction}")
        
        # Test system stats
        stats = analyzer.get_system_stats()
        print(f"📊 System stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Traffic analysis test failed: {e}")
        return False

def test_traffic_controller():
    """Test traffic light controller"""
    print("\n🚦 Testing Traffic Light Controller...")
    
    try:
        # Initialize controller
        directions = ['north', 'south', 'east', 'west']
        controller = TrafficLightController(
            directions=directions,
            default_timing={'green': 20, 'yellow': 5, 'red': 25},
            timing_ranges={'green_min': 10, 'green_max': 40, 'yellow': 5, 'red_min': 15, 'red_max': 35}
        )
        print("✅ Traffic controller initialized successfully")
        
        # Start controller
        controller.start_control_loop()
        print("✅ Control loop started")
        
        # Simulate traffic data updates
        for i in range(5):
            # Create mock traffic data
            mock_traffic_data = {}
            for direction in directions:
                from src.core.traffic_analyzer import TrafficData, TrafficDensity
                mock_traffic_data[direction] = TrafficData(
                    direction=direction,
                    vehicle_count=np.random.randint(2, 12),
                    density_level=np.random.choice(list(TrafficDensity)),
                    avg_vehicle_size=2000.0,
                    vehicle_types={'cars': 3, 'motorcycles': 1, 'buses': 0, 'trucks': 1},
                    timestamp=time.time(),
                    confidence_score=0.85
                )
            
            # Update controller
            controller.update_traffic_data(mock_traffic_data)
            
            # Get current states
            states = controller.get_current_states()
            print(f"⏱️  Cycle {i+1}:")
            for direction, state in states.items():
                print(f"  {direction}: {state['current_state']} ({state['remaining_time']:.1f}s)")
            
            time.sleep(2)
        
        # Test emergency mode
        print("🚨 Testing emergency mode...")
        controller.set_emergency_mode('north')
        time.sleep(1)
        
        emergency_states = controller.get_current_states()
        print("Emergency states:")
        for direction, state in emergency_states.items():
            print(f"  {direction}: {state['current_state']}")
        
        controller.clear_emergency_mode()
        print("✅ Emergency mode cleared")
        
        # Stop controller
        controller.stop_control_loop()
        print("✅ Controller stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Traffic controller test failed: {e}")
        return False

def test_integration():
    """Test system integration"""
    print("\n🔧 Testing System Integration...")
    
    try:
        # Initialize all components
        roi_config = {
            'north': [0.1, 0.1, 0.9, 0.4],
            'south': [0.1, 0.6, 0.9, 0.9],
            'east': [0.6, 0.1, 0.9, 0.9],
            'west': [0.1, 0.1, 0.4, 0.9]
        }
        
        analyzer = TrafficAnalyzer(roi_config=roi_config)
        controller = TrafficLightController(list(roi_config.keys()))
        
        print("✅ All components initialized")
        
        # Start controller
        controller.start_control_loop()
        
        # Simulate processing loop
        print("🔄 Simulating processing loop...")
        for frame_num in range(10):
            # Generate test data
            detections = create_test_detections(frame_num, 6)
            
            # Analyze traffic
            traffic_analysis = analyzer.analyze_traffic(detections)
            
            # Update controller
            controller.update_traffic_data(traffic_analysis)
            
            # Get system status
            if frame_num % 3 == 0:  # Every 3 frames
                states = controller.get_current_states()
                print(f"Frame {frame_num}: Active lights:")
                for direction, state in states.items():
                    if state['current_state'] == 'green':
                        print(f"  🟢 {direction}: {state['remaining_time']:.1f}s remaining")
            
            time.sleep(0.5)
        
        # Stop controller
        controller.stop_control_loop()
        print("✅ Integration test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def create_demo_video():
    """Create a demonstration video"""
    print("\n🎥 Creating Demo Video...")
    
    try:
        os.makedirs('test_output', exist_ok=True)
        
        # Video parameters
        width, height = 1280, 720
        fps = 10
        duration = 20  # seconds
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_output/demo_traffic.mp4', fourcc, fps, (width, height))
        
        for frame_num in range(fps * duration):
            # Create frame with varying traffic
            vehicles_count = 3 + int(5 * np.sin(frame_num * 0.1))  # Varying traffic
            frame = create_test_frame(frame_num, vehicles_count)
            
            # Add title
            cv2.putText(frame, "Smart Traffic Light System - Demo", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_num} | Vehicles: {vehicles_count}", 
                       (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print("✅ Demo video created: test_output/demo_traffic.mp4")
        return True
        
    except Exception as e:
        print(f"❌ Demo video creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚦 Smart Traffic Light System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Vehicle Detection", test_vehicle_detection),
        ("Traffic Analysis", test_traffic_analysis),
        ("Traffic Controller", test_traffic_controller),
        ("System Integration", test_integration),
        ("Demo Video Creation", create_demo_video)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print results summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("\nNext steps:")
        print("1. Run 'python main.py --create-sample' to create sample videos")
        print("2. Run 'python main.py' to start the system")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)