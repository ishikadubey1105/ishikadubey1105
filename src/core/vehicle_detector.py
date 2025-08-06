"""
Vehicle Detection Module using YOLOv8
Detects vehicles in real-time from camera feeds
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class Detection:
    """Data class for vehicle detection results"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: float

class VehicleDetector:
    """
    Vehicle detector using YOLOv8 model
    Detects cars, motorcycles, buses, and trucks in video frames
    """
    
    # COCO dataset class names
    CLASS_NAMES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
    }
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 confidence: float = 0.5, 
                 iou_threshold: float = 0.45,
                 device: str = "cpu",
                 vehicle_classes: List[int] = None):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu' or 'cuda')
            vehicle_classes: List of class IDs to detect (default: cars, motorcycles, buses, trucks)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info(f"Loaded YOLO model: {self.model_path} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence, iou=self.iou_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data
                        class_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        bbox = box.xyxy.cpu().numpy()[0].astype(int)
                        
                        # Filter for vehicle classes only
                        if class_id in self.vehicle_classes:
                            x1, y1, x2, y2 = bbox
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            area = (x2 - x1) * (y2 - y1)
                            
                            detection = Detection(
                                class_id=class_id,
                                class_name=self.CLASS_NAMES.get(class_id, 'unknown'),
                                confidence=confidence,
                                bbox=(x1, y1, x2, y2),
                                center=center,
                                area=area
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during vehicle detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                       roi_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            roi_boxes: Optional ROI boxes to draw
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Draw ROI boxes if provided
        if roi_boxes:
            for direction, (x1, y1, x2, y2) in roi_boxes.items():
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(result_frame, direction.upper(), (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on vehicle type
            color_map = {
                2: (0, 255, 0),    # car - green
                3: (255, 0, 0),    # motorcycle - blue
                5: (0, 0, 255),    # bus - red
                7: (255, 255, 0)   # truck - cyan
            }
            color = color_map.get(detection.class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_frame
    
    def filter_detections_by_roi(self, detections: List[Detection], 
                                roi: Tuple[int, int, int, int]) -> List[Detection]:
        """
        Filter detections that fall within a region of interest
        
        Args:
            detections: List of detections
            roi: ROI as (x1, y1, x2, y2)
            
        Returns:
            Filtered detections
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        filtered_detections = []
        
        for detection in detections:
            center_x, center_y = detection.center
            
            # Check if detection center is within ROI
            if (roi_x1 <= center_x <= roi_x2 and 
                roi_y1 <= center_y <= roi_y2):
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def get_detection_stats(self, detections: List[Detection]) -> Dict:
        """
        Get statistics about detections
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total_vehicles': 0,
                'cars': 0,
                'motorcycles': 0,
                'buses': 0,
                'trucks': 0,
                'avg_confidence': 0.0
            }
        
        stats = {
            'total_vehicles': len(detections),
            'cars': len([d for d in detections if d.class_id == 2]),
            'motorcycles': len([d for d in detections if d.class_id == 3]),
            'buses': len([d for d in detections if d.class_id == 5]),
            'trucks': len([d for d in detections if d.class_id == 7]),
            'avg_confidence': np.mean([d.confidence for d in detections])
        }
        
        return stats