"""
Intelligent Traffic Light Controller
Dynamically adjusts traffic light timing based on real-time traffic analysis
"""

import time
import logging
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
import threading
from collections import deque

from .traffic_analyzer import TrafficData, TrafficDensity

class LightState(Enum):
    """Traffic light states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"

class ControlMode(Enum):
    """Traffic control modes"""
    FIXED = "fixed"          # Fixed timing
    ADAPTIVE = "adaptive"    # Adaptive based on traffic
    EMERGENCY = "emergency"  # Emergency override

@dataclass
class LightTiming:
    """Traffic light timing configuration"""
    green_duration: float
    yellow_duration: float
    red_duration: float
    min_green: float = 15.0
    max_green: float = 60.0

@dataclass
class DirectionState:
    """State for a traffic direction"""
    direction: str
    current_state: LightState
    state_start_time: float
    remaining_time: float
    priority_score: float = 0.0
    last_green_time: float = 0.0

class TrafficLightController:
    """
    Intelligent traffic light controller that adapts timing based on traffic density
    """
    
    def __init__(self, 
                 directions: List[str],
                 default_timing: Dict[str, float] = None,
                 timing_ranges: Dict[str, float] = None,
                 min_cycle_time: float = 60.0,
                 max_cycle_time: float = 180.0):
        """
        Initialize traffic light controller
        
        Args:
            directions: List of traffic directions
            default_timing: Default timing configuration
            timing_ranges: Min/max timing ranges
            min_cycle_time: Minimum cycle time
            max_cycle_time: Maximum cycle time
        """
        self.directions = directions
        self.default_timing = default_timing or {
            'green': 30.0, 'yellow': 5.0, 'red': 35.0
        }
        self.timing_ranges = timing_ranges or {
            'green_min': 15.0, 'green_max': 60.0,
            'yellow': 5.0, 'red_min': 10.0, 'red_max': 45.0
        }
        self.min_cycle_time = min_cycle_time
        self.max_cycle_time = max_cycle_time
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize direction states
        self.direction_states = {}
        self._initialize_directions()
        
        # Control state
        self.control_mode = ControlMode.ADAPTIVE
        self.current_active_direction = None
        self.cycle_start_time = time.time()
        self.emergency_override = False
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.timing_adjustments = deque(maxlen=50)
        
        # Threading for continuous operation
        self.running = False
        self.control_thread = None
        self.lock = threading.Lock()
    
    def _initialize_directions(self) -> None:
        """Initialize direction states"""
        current_time = time.time()
        
        for i, direction in enumerate(self.directions):
            # Start first direction with green, others with red
            if i == 0:
                state = LightState.GREEN
                remaining = self.default_timing['green']
                self.current_active_direction = direction
            else:
                state = LightState.RED
                remaining = self.default_timing['red']
            
            self.direction_states[direction] = DirectionState(
                direction=direction,
                current_state=state,
                state_start_time=current_time,
                remaining_time=remaining,
                last_green_time=current_time if i == 0 else 0.0
            )
    
    def start_control_loop(self) -> None:
        """Start the traffic control loop"""
        if not self.running:
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            self.logger.info("Traffic light controller started")
    
    def stop_control_loop(self) -> None:
        """Stop the traffic control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        self.logger.info("Traffic light controller stopped")
    
    def _control_loop(self) -> None:
        """Main control loop"""
        while self.running:
            try:
                with self.lock:
                    self._update_states()
                    self._check_transitions()
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
                time.sleep(1.0)
    
    def _update_states(self) -> None:
        """Update timing for all directions"""
        current_time = time.time()
        
        for direction_state in self.direction_states.values():
            elapsed = current_time - direction_state.state_start_time
            direction_state.remaining_time = max(0, 
                direction_state.remaining_time - elapsed + 
                (current_time - direction_state.state_start_time)
            )
            direction_state.state_start_time = current_time
    
    def _check_transitions(self) -> None:
        """Check if any lights need to transition"""
        current_time = time.time()
        
        if self.current_active_direction is None:
            return
        
        active_state = self.direction_states[self.current_active_direction]
        
        # Check if current state has expired
        if active_state.remaining_time <= 0:
            self._transition_lights()
    
    def _transition_lights(self) -> None:
        """Transition traffic lights to next state"""
        if self.current_active_direction is None:
            return
        
        current_time = time.time()
        active_state = self.direction_states[self.current_active_direction]
        
        if active_state.current_state == LightState.GREEN:
            # Green -> Yellow
            self._set_light_state(self.current_active_direction, LightState.YELLOW,
                                self.timing_ranges['yellow'])
            
        elif active_state.current_state == LightState.YELLOW:
            # Yellow -> Red, and start next direction
            self._set_light_state(self.current_active_direction, LightState.RED,
                                self._calculate_red_duration(self.current_active_direction))
            
            # Find next direction to activate
            next_direction = self._get_next_active_direction()
            if next_direction:
                self._activate_direction(next_direction)
    
    def _set_light_state(self, direction: str, state: LightState, duration: float) -> None:
        """Set light state for a direction"""
        if direction not in self.direction_states:
            return
        
        current_time = time.time()
        direction_state = self.direction_states[direction]
        
        direction_state.current_state = state
        direction_state.state_start_time = current_time
        direction_state.remaining_time = duration
        
        if state == LightState.GREEN:
            direction_state.last_green_time = current_time
        
        self.logger.info(f"{direction} light: {state.value} for {duration:.1f}s")
    
    def _activate_direction(self, direction: str) -> None:
        """Activate green light for a direction"""
        if direction not in self.direction_states:
            return
        
        # Calculate green duration based on traffic
        green_duration = self._calculate_green_duration(direction)
        
        self._set_light_state(direction, LightState.GREEN, green_duration)
        self.current_active_direction = direction
        
        # Set all other directions to red
        for other_direction in self.directions:
            if other_direction != direction:
                red_duration = self._calculate_red_duration(other_direction)
                self._set_light_state(other_direction, LightState.RED, red_duration)
    
    def _get_next_active_direction(self) -> Optional[str]:
        """Get the next direction that should have green light"""
        if self.control_mode == ControlMode.FIXED:
            return self._get_next_direction_fixed()
        elif self.control_mode == ControlMode.ADAPTIVE:
            return self._get_next_direction_adaptive()
        else:
            return self._get_next_direction_fixed()
    
    def _get_next_direction_fixed(self) -> Optional[str]:
        """Get next direction using fixed rotation"""
        if self.current_active_direction is None:
            return self.directions[0] if self.directions else None
        
        try:
            current_index = self.directions.index(self.current_active_direction)
            next_index = (current_index + 1) % len(self.directions)
            return self.directions[next_index]
        except ValueError:
            return self.directions[0] if self.directions else None
    
    def _get_next_direction_adaptive(self) -> Optional[str]:
        """Get next direction using adaptive logic"""
        # Find direction with highest priority that hasn't had green recently
        current_time = time.time()
        min_wait_time = 30.0  # Minimum time between green lights for same direction
        
        candidates = []
        for direction, state in self.direction_states.items():
            if direction == self.current_active_direction:
                continue
            
            # Check if direction has waited long enough
            time_since_green = current_time - state.last_green_time
            if state.last_green_time == 0 or time_since_green >= min_wait_time:
                candidates.append((direction, state.priority_score))
        
        if candidates:
            # Sort by priority score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Fallback to fixed rotation if no candidates
        return self._get_next_direction_fixed()
    
    def _calculate_green_duration(self, direction: str) -> float:
        """Calculate green light duration based on traffic conditions"""
        base_duration = self.default_timing['green']
        
        if direction not in self.direction_states:
            return base_duration
        
        direction_state = self.direction_states[direction]
        priority_score = direction_state.priority_score
        
        # Adjust duration based on priority score
        if priority_score > 0:
            # Higher priority gets longer green time
            multiplier = min(2.0, 1.0 + (priority_score / 100.0))
            duration = base_duration * multiplier
        else:
            duration = base_duration
        
        # Apply min/max constraints
        duration = max(self.timing_ranges['green_min'], 
                      min(self.timing_ranges['green_max'], duration))
        
        return duration
    
    def _calculate_red_duration(self, direction: str) -> float:
        """Calculate red light duration"""
        # Red duration should accommodate other directions' green+yellow time
        total_other_time = 0
        
        for other_direction in self.directions:
            if other_direction != direction:
                green_time = self._calculate_green_duration(other_direction)
                yellow_time = self.timing_ranges['yellow']
                total_other_time += green_time + yellow_time
        
        # Add some buffer time
        red_duration = total_other_time + 5.0
        
        # Apply constraints
        red_duration = max(self.timing_ranges['red_min'],
                          min(self.timing_ranges['red_max'], red_duration))
        
        return red_duration
    
    def update_traffic_data(self, traffic_data: Dict[str, TrafficData]) -> None:
        """Update traffic data and recalculate priorities"""
        with self.lock:
            current_time = time.time()
            
            for direction, data in traffic_data.items():
                if direction in self.direction_states:
                    # Calculate priority score based on traffic density and other factors
                    priority_score = self._calculate_priority_score(data)
                    self.direction_states[direction].priority_score = priority_score
            
            # Log traffic update
            self.logger.debug(f"Updated traffic data for {len(traffic_data)} directions")
    
    def _calculate_priority_score(self, traffic_data: TrafficData) -> float:
        """Calculate priority score for a direction based on traffic data"""
        score = 0.0
        
        # Base score from vehicle count
        score += traffic_data.vehicle_count * 2.0
        
        # Density level multiplier
        density_multipliers = {
            TrafficDensity.LOW: 1.0,
            TrafficDensity.MEDIUM: 2.5,
            TrafficDensity.HIGH: 5.0,
            TrafficDensity.CRITICAL: 10.0
        }
        score *= density_multipliers.get(traffic_data.density_level, 1.0)
        
        # Heavy vehicle bonus
        heavy_vehicles = (traffic_data.vehicle_types.get('buses', 0) + 
                         traffic_data.vehicle_types.get('trucks', 0))
        score += heavy_vehicles * 5.0
        
        # Confidence factor
        score *= traffic_data.confidence_score
        
        return score
    
    def set_emergency_mode(self, emergency_direction: Optional[str] = None) -> None:
        """Set emergency mode for priority vehicle passage"""
        with self.lock:
            self.control_mode = ControlMode.EMERGENCY
            self.emergency_override = True
            
            if emergency_direction and emergency_direction in self.direction_states:
                # Immediately switch to emergency direction
                self._activate_direction(emergency_direction)
                self.logger.warning(f"Emergency mode activated for {emergency_direction}")
            else:
                self.logger.warning("Emergency mode activated - all lights red")
                # Set all lights to red for safety
                for direction in self.directions:
                    self._set_light_state(direction, LightState.RED, 60.0)
    
    def clear_emergency_mode(self) -> None:
        """Clear emergency mode and return to adaptive control"""
        with self.lock:
            self.emergency_override = False
            self.control_mode = ControlMode.ADAPTIVE
            self.logger.info("Emergency mode cleared - returning to adaptive control")
    
    def get_current_states(self) -> Dict[str, Dict]:
        """Get current state of all traffic lights"""
        with self.lock:
            states = {}
            current_time = time.time()
            
            for direction, state in self.direction_states.items():
                states[direction] = {
                    'current_state': state.current_state.value,
                    'remaining_time': max(0, state.remaining_time),
                    'priority_score': state.priority_score,
                    'last_green_time': state.last_green_time,
                    'time_since_green': current_time - state.last_green_time if state.last_green_time > 0 else None
                }
            
            return states
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        with self.lock:
            current_time = time.time()
            cycle_time = current_time - self.cycle_start_time
            
            return {
                'control_mode': self.control_mode.value,
                'emergency_override': self.emergency_override,
                'current_active_direction': self.current_active_direction,
                'cycle_time': cycle_time,
                'directions_count': len(self.directions),
                'running': self.running,
                'performance_data_points': len(self.performance_history)
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.performance_history:
            return {
                'avg_cycle_time': 0,
                'avg_wait_time': 0,
                'efficiency_score': 0,
                'adaptations_made': 0
            }
        
        recent_data = list(self.performance_history)[-20:]  # Last 20 cycles
        
        avg_cycle_time = sum(data.get('cycle_time', 0) for data in recent_data) / len(recent_data)
        avg_wait_time = sum(data.get('avg_wait_time', 0) for data in recent_data) / len(recent_data)
        
        # Simple efficiency score based on cycle time vs traffic demand
        efficiency_score = min(100, max(0, 100 - (avg_cycle_time - self.min_cycle_time) * 2))
        
        return {
            'avg_cycle_time': avg_cycle_time,
            'avg_wait_time': avg_wait_time,
            'efficiency_score': efficiency_score,
            'adaptations_made': len(self.timing_adjustments),
            'data_points': len(recent_data)
        }