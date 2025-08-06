// Smart Traffic Light System Dashboard JavaScript

class TrafficDashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.emergencyMode = false;
        this.updateIntervals = {};
        
        this.init();
    }
    
    init() {
        this.initializeSocket();
        this.setupEventListeners();
        this.startPeriodicUpdates();
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('system_status', (data) => {
            this.updateSystemStatus(data.status);
        });
        
        this.socket.on('frame_update', (data) => {
            this.updateCameraFeed(data);
        });
        
        this.socket.on('analysis_update', (data) => {
            this.updateTrafficAnalytics(data);
            this.updateTrafficLights(data.controller_status);
        });
        
        this.socket.on('emergency_response', (data) => {
            this.handleEmergencyResponse(data);
        });
    }
    
    setupEventListeners() {
        // Request initial frame for each camera
        this.requestCameraFrames();
        
        // Setup periodic frame requests
        setInterval(() => {
            if (this.isConnected) {
                this.requestCameraFrames();
            }
        }, 1000); // Request frames every second
    }
    
    startPeriodicUpdates() {
        // Update system status every 5 seconds
        this.updateIntervals.status = setInterval(() => {
            this.fetchSystemStatus();
        }, 5000);
        
        // Update traffic lights every 2 seconds
        this.updateIntervals.lights = setInterval(() => {
            this.fetchTrafficLights();
        }, 2000);
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.className = 'badge bg-success me-3';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
        } else {
            statusElement.className = 'badge bg-danger me-3 connection-lost';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
        }
    }
    
    updateSystemStatus(status) {
        // Update system status indicators
        const systemStatusEl = document.getElementById('system-status');
        const activeCamerasEl = document.getElementById('active-cameras');
        const processingFpsEl = document.getElementById('processing-fps');
        const totalDetectionsEl = document.getElementById('total-detections');
        
        if (status.processing) {
            const processing = status.processing;
            
            // System status
            if (processing.running) {
                systemStatusEl.innerHTML = '<i class="fas fa-circle text-success"></i><span>Running</span>';
            } else {
                systemStatusEl.innerHTML = '<i class="fas fa-circle text-danger"></i><span>Stopped</span>';
            }
            
            // Active cameras
            activeCamerasEl.textContent = processing.active_cameras || 0;
            
            // Processing FPS
            processingFpsEl.textContent = (processing.fps || 0).toFixed(1);
            
            // Total detections
            totalDetectionsEl.textContent = processing.detections_made || 0;
        }
    }
    
    updateCameraFeed(data) {
        const feedsContainer = document.getElementById('camera-feeds');
        
        // Create or update camera feed element
        let feedElement = document.getElementById(`feed-${data.camera_id}`);
        if (!feedElement) {
            feedElement = this.createCameraFeedElement(data.camera_id);
            feedsContainer.appendChild(feedElement);
        }
        
        // Update image
        const imgElement = feedElement.querySelector('img');
        if (imgElement && data.frame) {
            imgElement.src = data.frame;
        }
        
        // Update detection count
        const countElement = feedElement.querySelector('.detection-count');
        if (countElement) {
            countElement.textContent = `${data.detections || 0} vehicles`;
        }
        
        // Update timestamp
        const overlayElement = feedElement.querySelector('.camera-overlay');
        if (overlayElement) {
            const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
            overlayElement.textContent = `${data.camera_id} - ${timestamp}`;
        }
    }
    
    createCameraFeedElement(cameraId) {
        const feedElement = document.createElement('div');
        feedElement.id = `feed-${cameraId}`;
        feedElement.className = 'col-md-6 camera-feed';
        feedElement.innerHTML = `
            <div class="position-relative">
                <img src="" alt="Camera Feed" class="img-fluid">
                <div class="camera-overlay">${cameraId}</div>
                <div class="detection-count">0 vehicles</div>
            </div>
        `;
        return feedElement;
    }
    
    updateTrafficLights(controllerStatus) {
        const container = document.getElementById('traffic-lights-container');
        
        if (!controllerStatus) return;
        
        // Clear existing lights
        container.innerHTML = '';
        
        // Create traffic light for each direction
        Object.keys(controllerStatus).forEach(direction => {
            const lightData = controllerStatus[direction];
            const lightElement = this.createTrafficLightElement(direction, lightData);
            container.appendChild(lightElement);
        });
    }
    
    createTrafficLightElement(direction, lightData) {
        const colElement = document.createElement('div');
        colElement.className = 'col-md-3';
        
        const currentState = lightData.current_state;
        const remainingTime = Math.max(0, lightData.remaining_time || 0);
        const priorityScore = lightData.priority_score || 0;
        
        colElement.innerHTML = `
            <div class="traffic-light ${this.emergencyMode ? 'emergency-active' : ''}">
                <h6>${direction.toUpperCase()}</h6>
                <div class="light-indicator ${currentState === 'red' ? 'light-red' : 'light-off'}"></div>
                <div class="light-indicator ${currentState === 'yellow' ? 'light-yellow' : 'light-off'}"></div>
                <div class="light-indicator ${currentState === 'green' ? 'light-green' : 'light-off'}"></div>
                <div class="light-timer">${Math.ceil(remainingTime)}s</div>
                <div class="priority-score">Priority: ${priorityScore.toFixed(1)}</div>
            </div>
        `;
        
        return colElement;
    }
    
    updateTrafficAnalytics(data) {
        const container = document.getElementById('analytics-container');
        
        if (!data.traffic_analysis) return;
        
        // Clear existing analytics
        container.innerHTML = '';
        
        // Create analytics for each direction
        Object.keys(data.traffic_analysis).forEach(direction => {
            const trafficData = data.traffic_analysis[direction];
            const analyticsElement = this.createAnalyticsElement(direction, trafficData);
            container.appendChild(analyticsElement);
        });
        
        // Add summary analytics
        const summaryElement = this.createSummaryAnalytics(data);
        container.appendChild(summaryElement);
    }
    
    createAnalyticsElement(direction, trafficData) {
        const element = document.createElement('div');
        element.className = 'analytics-item';
        
        const densityClass = `density-${trafficData.density_level}`;
        const vehicleTypes = trafficData.vehicle_types || {};
        
        element.innerHTML = `
            <div class="analytics-header">${direction} Direction</div>
            <div class="analytics-value">${trafficData.vehicle_count}</div>
            <div class="mb-2">
                <span class="density-indicator ${densityClass}">${trafficData.density_level}</span>
            </div>
            <div class="vehicle-types">
                <div class="vehicle-type">
                    <i class="fas fa-car"></i>
                    <div class="count">${vehicleTypes.cars || 0}</div>
                </div>
                <div class="vehicle-type">
                    <i class="fas fa-motorcycle"></i>
                    <div class="count">${vehicleTypes.motorcycles || 0}</div>
                </div>
                <div class="vehicle-type">
                    <i class="fas fa-bus"></i>
                    <div class="count">${vehicleTypes.buses || 0}</div>
                </div>
                <div class="vehicle-type">
                    <i class="fas fa-truck"></i>
                    <div class="count">${vehicleTypes.trucks || 0}</div>
                </div>
            </div>
            <div class="mt-2">
                <small class="text-muted">
                    Confidence: ${(trafficData.confidence_score * 100).toFixed(1)}%
                </small>
            </div>
        `;
        
        return element;
    }
    
    createSummaryAnalytics(data) {
        const element = document.createElement('div');
        element.className = 'analytics-item';
        element.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        element.style.color = 'white';
        
        const totalVehicles = data.total_detections || 0;
        const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
        
        element.innerHTML = `
            <div class="analytics-header" style="color: white;">System Summary</div>
            <div class="analytics-value" style="color: white;">${totalVehicles}</div>
            <div>Total Vehicles Detected</div>
            <div class="mt-2">
                <small>Last Update: ${timestamp}</small>
            </div>
        `;
        
        return element;
    }
    
    requestCameraFrames() {
        // Request frames for all cameras
        ['camera_0', 'camera_1', 'camera_2'].forEach(cameraId => {
            this.socket.emit('request_frame', { camera_id: cameraId });
        });
    }
    
    fetchSystemStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.updateSystemStatus(data.data);
                }
            })
            .catch(error => console.error('Error fetching system status:', error));
    }
    
    fetchTrafficLights() {
        fetch('/api/traffic_lights')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.updateTrafficLights(data.data);
                }
            })
            .catch(error => console.error('Error fetching traffic lights:', error));
    }
    
    handleEmergencyResponse(data) {
        if (data.success) {
            this.showAlert(data.message, 'success');
        } else {
            this.showAlert(data.error || 'Emergency operation failed', 'danger');
        }
    }
    
    showAlert(message, type) {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show alert-custom" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Insert alert at the top of the container
        const container = document.querySelector('.container-fluid');
        const alertElement = document.createElement('div');
        alertElement.innerHTML = alertHtml;
        container.insertBefore(alertElement.firstElementChild, container.firstElementChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
}

// Global functions for button handlers
function setEmergencyMode() {
    const direction = document.getElementById('emergency-direction').value;
    
    fetch('/api/emergency', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            direction: direction || null
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            dashboard.emergencyMode = true;
            dashboard.showAlert(data.message, 'warning');
            
            // Show emergency modal
            const modal = new bootstrap.Modal(document.getElementById('emergencyModal'));
            modal.show();
        } else {
            dashboard.showAlert(data.error || 'Failed to activate emergency mode', 'danger');
        }
    })
    .catch(error => {
        console.error('Error setting emergency mode:', error);
        dashboard.showAlert('Network error occurred', 'danger');
    });
}

function clearEmergencyMode() {
    fetch('/api/emergency', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            clear: true
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            dashboard.emergencyMode = false;
            dashboard.showAlert(data.message, 'success');
            
            // Hide emergency modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('emergencyModal'));
            if (modal) {
                modal.hide();
            }
        } else {
            dashboard.showAlert(data.error || 'Failed to clear emergency mode', 'danger');
        }
    })
    .catch(error => {
        console.error('Error clearing emergency mode:', error);
        dashboard.showAlert('Network error occurred', 'danger');
    });
}

function toggleEmergency() {
    if (dashboard.emergencyMode) {
        clearEmergencyMode();
    } else {
        setEmergencyMode();
    }
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new TrafficDashboard();
});