"""
Traffic Density Analyzer
Analyzes traffic density and patterns from vehicle detections
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import logging
from dataclasses import dataclass, field
from enum import Enum

from .vehicle_detector import Detection

class TrafficDensity(Enum):
    """Traffic density levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TrafficData:
    """Data class for traffic analysis results"""
    direction: str
    vehicle_count: int
    density_level: TrafficDensity
    avg_vehicle_size: float
    vehicle_types: Dict[str, int]
    timestamp: float
    confidence_score: float

@dataclass
class DirectionAnalysis:
    """Analysis data for a specific direction"""
    direction: str
    roi: Tuple[int, int, int, int]
    vehicle_history: deque = field(default_factory=lambda: deque(maxlen=100))
    density_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_analysis_time: float = 0.0
    total_vehicles_seen: int = 0

class TrafficAnalyzer:
    """
    Analyzes traffic density and patterns from vehicle detections
    Provides intelligent insights for traffic light control
    """
    
    def __init__(self, 
                 roi_config: Dict[str, List[float]],
                 density_thresholds: Dict[str, int] = None,
                 analysis_window: float = 10.0,
                 frame_dimensions: Tuple[int, int] = (1280, 720)):
        """
        Initialize traffic analyzer
        
        Args:
            roi_config: ROI configuration for each direction (as percentages)
            density_thresholds: Thresholds for traffic density levels
            analysis_window: Time window for analysis in seconds
            frame_dimensions: Frame width and height
        """
        self.roi_config = roi_config
        self.density_thresholds = density_thresholds or {
            'low': 5, 'medium': 15, 'high': 30, 'critical': 50
        }
        self.analysis_window = analysis_window
        self.frame_width, self.frame_height = frame_dimensions
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize direction analyses
        self.directions = {}
        self._initialize_directions()
        
        # Historical data for trend analysis
        self.global_traffic_history = deque(maxlen=1000)
        self.peak_hours = {}
        
    def _initialize_directions(self) -> None:
        """Initialize direction analysis objects"""
        for direction, roi_percent in self.roi_config.items():
            # Convert percentage ROI to pixel coordinates
            x1 = int(roi_percent[0] * self.frame_width)
            y1 = int(roi_percent[1] * self.frame_height)
            x2 = int(roi_percent[2] * self.frame_width)
            y2 = int(roi_percent[3] * self.frame_height)
            
            roi = (x1, y1, x2, y2)
            
            self.directions[direction] = DirectionAnalysis(
                direction=direction,
                roi=roi
            )
            
            self.logger.info(f"Initialized {direction} direction with ROI: {roi}")
    
    def analyze_traffic(self, detections: List[Detection]) -> Dict[str, TrafficData]:
        """
        Analyze traffic for all directions
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            Dictionary mapping direction to traffic analysis
        """
        current_time = time.time()
        results = {}
        
        for direction, analysis in self.directions.items():
            # Filter detections for this direction's ROI
            direction_detections = self._filter_detections_by_roi(
                detections, analysis.roi
            )
            
            # Analyze traffic for this direction
            traffic_data = self._analyze_direction_traffic(
                direction, direction_detections, current_time
            )
            
            results[direction] = traffic_data
            
            # Update historical data
            analysis.vehicle_history.append({
                'timestamp': current_time,
                'count': traffic_data.vehicle_count,
                'detections': direction_detections
            })
            
            analysis.density_history.append({
                'timestamp': current_time,
                'density': traffic_data.density_level
            })
            
            analysis.total_vehicles_seen += traffic_data.vehicle_count
            analysis.last_analysis_time = current_time
        
        # Update global traffic history
        self.global_traffic_history.append({
            'timestamp': current_time,
            'directions': results.copy()
        })
        
        return results
    
    def _filter_detections_by_roi(self, detections: List[Detection], 
                                 roi: Tuple[int, int, int, int]) -> List[Detection]:
        """Filter detections within ROI"""
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        filtered = []
        
        for detection in detections:
            center_x, center_y = detection.center
            
            if (roi_x1 <= center_x <= roi_x2 and 
                roi_y1 <= center_y <= roi_y2):
                filtered.append(detection)
        
        return filtered
    
    def _analyze_direction_traffic(self, direction: str, 
                                  detections: List[Detection],
                                  current_time: float) -> TrafficData:
        """Analyze traffic for a specific direction"""
        vehicle_count = len(detections)
        
        # Calculate vehicle types distribution
        vehicle_types = {
            'cars': len([d for d in detections if d.class_id == 2]),
            'motorcycles': len([d for d in detections if d.class_id == 3]),
            'buses': len([d for d in detections if d.class_id == 5]),
            'trucks': len([d for d in detections if d.class_id == 7])
        }
        
        # Calculate average vehicle size (area)
        avg_vehicle_size = np.mean([d.area for d in detections]) if detections else 0.0
        
        # Calculate average confidence
        confidence_score = np.mean([d.confidence for d in detections]) if detections else 0.0
        
        # Determine density level
        density_level = self._calculate_density_level(vehicle_count, direction)
        
        return TrafficData(
            direction=direction,
            vehicle_count=vehicle_count,
            density_level=density_level,
            avg_vehicle_size=avg_vehicle_size,
            vehicle_types=vehicle_types,
            timestamp=current_time,
            confidence_score=confidence_score
        )
    
    def _calculate_density_level(self, vehicle_count: int, direction: str) -> TrafficDensity:
        """Calculate traffic density level based on vehicle count"""
        # Apply direction-specific adjustments if needed
        adjusted_thresholds = self.density_thresholds.copy()
        
        # Some directions might have different capacity
        if direction in ['north', 'south']:  # Main roads might have higher capacity
            for key in adjusted_thresholds:
                adjusted_thresholds[key] = int(adjusted_thresholds[key] * 1.2)
        
        if vehicle_count >= adjusted_thresholds.get('critical', 50):
            return TrafficDensity.CRITICAL
        elif vehicle_count >= adjusted_thresholds.get('high', 30):
            return TrafficDensity.HIGH
        elif vehicle_count >= adjusted_thresholds.get('medium', 15):
            return TrafficDensity.MEDIUM
        else:
            return TrafficDensity.LOW
    
    def get_traffic_trends(self, direction: str, 
                          time_window: float = 300.0) -> Dict:
        """
        Get traffic trends for a direction over a time window
        
        Args:
            direction: Direction to analyze
            time_window: Time window in seconds
            
        Returns:
            Dictionary with trend analysis
        """
        if direction not in self.directions:
            return {}
        
        analysis = self.directions[direction]
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter recent history
        recent_data = [
            entry for entry in analysis.vehicle_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_data:
            return {
                'trend': 'stable',
                'avg_vehicles': 0,
                'peak_count': 0,
                'trend_score': 0.0
            }
        
        # Calculate trend metrics
        vehicle_counts = [entry['count'] for entry in recent_data]
        timestamps = [entry['timestamp'] for entry in recent_data]
        
        avg_vehicles = np.mean(vehicle_counts)
        peak_count = max(vehicle_counts)
        
        # Calculate trend using linear regression
        if len(vehicle_counts) > 1:
            trend_score = np.polyfit(timestamps, vehicle_counts, 1)[0]
            
            if trend_score > 0.1:
                trend = 'increasing'
            elif trend_score < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            trend_score = 0.0
        
        return {
            'trend': trend,
            'avg_vehicles': avg_vehicles,
            'peak_count': peak_count,
            'trend_score': trend_score,
            'data_points': len(recent_data)
        }
    
    def get_priority_direction(self) -> Optional[str]:
        """
        Get the direction with highest priority for green light
        
        Returns:
            Direction name or None
        """
        if not self.global_traffic_history:
            return None
        
        # Get latest traffic data
        latest_data = self.global_traffic_history[-1]['directions']
        
        # Calculate priority scores
        priority_scores = {}
        
        for direction, traffic_data in latest_data.items():
            score = 0
            
            # Base score from vehicle count
            score += traffic_data.vehicle_count * 1.0
            
            # Density level multiplier
            density_multipliers = {
                TrafficDensity.LOW: 1.0,
                TrafficDensity.MEDIUM: 2.0,
                TrafficDensity.HIGH: 4.0,
                TrafficDensity.CRITICAL: 8.0
            }
            score *= density_multipliers.get(traffic_data.density_level, 1.0)
            
            # Heavy vehicle bonus (buses and trucks need more time)
            heavy_vehicles = traffic_data.vehicle_types.get('buses', 0) + \
                           traffic_data.vehicle_types.get('trucks', 0)
            score += heavy_vehicles * 2.0
            
            # Trend bonus (increasing traffic gets priority)
            trends = self.get_traffic_trends(direction, 60.0)  # 1-minute window
            if trends.get('trend') == 'increasing':
                score *= 1.5
            
            priority_scores[direction] = score
        
        # Return direction with highest priority
        if priority_scores:
            return max(priority_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def get_system_stats(self) -> Dict:
        """Get overall system statistics"""
        current_time = time.time()
        total_vehicles = sum(
            analysis.total_vehicles_seen 
            for analysis in self.directions.values()
        )
        
        # Calculate average density across all directions
        if self.global_traffic_history:
            latest_data = self.global_traffic_history[-1]['directions']
            avg_density = np.mean([
                data.vehicle_count for data in latest_data.values()
            ])
            
            # Get system-wide density level
            system_density = self._calculate_density_level(int(avg_density), 'system')
        else:
            avg_density = 0
            system_density = TrafficDensity.LOW
        
        return {
            'total_vehicles_detected': total_vehicles,
            'active_directions': len(self.directions),
            'average_density': avg_density,
            'system_density_level': system_density.value,
            'analysis_uptime': current_time - (
                min(analysis.last_analysis_time for analysis in self.directions.values())
                if self.directions else current_time
            ),
            'data_points_collected': len(self.global_traffic_history)
        }
    
    def get_roi_boxes(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Get ROI boxes for visualization"""
        return {
            direction: analysis.roi 
            for direction, analysis in self.directions.items()
        }