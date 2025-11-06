# GEOVINS: Geographic Visual-Inertial Navigation System

A comprehensive state estimation framework that combines Multi-State Constraint Kalman Filter (MSCKF) with geographic constraints for robust 6-DOF pose estimation.

## Features

- **MSCKF Implementation**: Full implementation of Multi-State Constraint Kalman Filter for visual-inertial odometry
- **Geographic Integration**: Seamless integration of GPS/GNSS measurements with visual-inertial navigation
- **Modular Architecture**: Clean, extensible design with separate modules for sensors, features, filters, and geographic processing
- **Multiple Coordinate Frames**: Support for ECEF, ENU, and local frames with automatic transformations
- **Robust Feature Tracking**: Advanced feature detection, tracking, and matching with RANSAC outlier rejection
- **Adaptive Noise Estimation**: Dynamic adjustment of measurement noise based on sensor quality
- **RTK Support**: High-precision positioning with RTK-GPS corrections

## System Architecture

```
geovins_state_estimation/
├── src/
│   ├── core/           # State definitions, coordinate frames, data types
│   ├── sensors/        # IMU and camera models
│   ├── features/       # Feature detection, tracking, and matching
│   ├── filter/         # EKF and MSCKF implementations
│   └── geographic/     # GPS processing and geographic features
├── config/             # Configuration files
├── tests/              # Unit tests
└── examples/           # Usage examples
```

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy, SciPy

### Basic Installation

```bash
pip install -e .
```

### With Computer Vision Support

```bash
pip install -e ".[cv]"
```

### With Visualization

```bash
pip install -e ".[viz]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic MSCKF Example

```python
import numpy as np
from src.core.state import create_initial_state
from src.core.types import IMUMeasurement, Quaternion, Pose
from src.sensors.imu_model import IMUModel
from src.sensors.camera_model import PinholeCameraModel
from src.filter.msckf import MSCKF

# Create initial state
initial_state = create_initial_state(
    position=np.zeros(3),
    velocity=np.zeros(3),
    orientation=Quaternion.identity()
)

# Initialize IMU model
imu_model = IMUModel(
    gyro_noise_density=1.6968e-04,
    accel_noise_density=2.0000e-03
)

# Initialize camera model
camera_model = PinholeCameraModel(
    width=640, height=480,
    fx=458.654, fy=457.296,
    cx=367.215, cy=248.375
)

# Create camera-IMU extrinsics
T_cam_imu = Pose(
    position=np.zeros(3),
    orientation=Quaternion.identity()
)

# Initialize MSCKF
msckf = MSCKF(
    initial_state=initial_state,
    imu_model=imu_model,
    camera_model=camera_model,
    T_cam_imu=T_cam_imu
)

# Process IMU measurements
imu_measurements = [
    IMUMeasurement(
        timestamp=0.005 * i,
        angular_velocity=np.array([0.01, 0.0, 0.0]),
        linear_acceleration=np.array([0.0, 0.0, 9.81])
    )
    for i in range(100)
]

msckf.propagate_imu(imu_measurements)

# Get current state
state = msckf.get_state()
print(f"Position: {state.imu_state.position}")
print(f"Velocity: {state.imu_state.velocity}")
```

### 2. With Geographic Constraints

```python
from src.geographic.measurements import GPSMeasurement, GPSProcessor
from src.geographic.geo_features import GeoConstraint

# Set reference location
ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 10.0  # San Francisco

# Initialize GPS processor
gps_processor = GPSProcessor(ref_lat, ref_lon, ref_alt)

# Initialize state with geographic reference
initial_state.set_geo_reference(ref_lat, ref_lon, ref_alt)

# Process GPS measurement
gps_meas = GPSMeasurement(
    timestamp=1.0,
    latitude=37.7750,
    longitude=-122.4195,
    altitude=12.0,
    horizontal_accuracy=5.0
)

# Compute measurement model
H, R, residual = gps_processor.compute_measurement_model(state, gps_meas)

# Apply update (simplified)
# In practice, this would be integrated into MSCKF update cycle
```

### 3. Feature Tracking

```python
from src.features.tracker import FeatureTracker
from src.core.types import Feature

# Initialize tracker
tracker = FeatureTracker(
    max_track_length=20,
    min_track_length=3,
    max_features=200
)

# Add new features
features = [(100.5, 200.3), (150.2, 180.7), (220.1, 250.9)]
new_features = tracker.add_features(features, camera_state_id=0)

# Track features to next frame
curr_observations = [
    (0, 102.1, 201.5),  # (feature_id, u, v)
    (1, 151.8, 182.1),
    (2, 221.5, 252.3)
]

tracked, lost = tracker.track_features(new_features, curr_observations, camera_state_id=1)

print(f"Tracked: {len(tracked)}, Lost: {len(lost)}")
```

## Configuration

Edit `config/params.yaml` to customize system parameters:

```yaml
# IMU noise parameters
imu:
  gyro_noise_density: 1.6968e-04
  accel_noise_density: 2.0000e-03

# Camera parameters
camera:
  width: 640
  height: 480
  fx: 458.654
  fy: 457.296

# MSCKF parameters
msckf:
  max_camera_states: 20
  feature_noise: 1.0

# Geographic parameters
geographic:
  ref_latitude: 0.0      # SET YOUR REFERENCE
  ref_longitude: 0.0     # SET YOUR REFERENCE
  ref_altitude: 0.0      # SET YOUR REFERENCE
  use_gps: true
```

## Mathematical Background

### MSCKF State Vector

The MSCKF maintains a state vector consisting of:

1. **IMU State** (15 DOF):
   - Orientation (3): Error-state quaternion
   - Position (3): Position in world frame
   - Velocity (3): Velocity in world frame
   - Gyro bias (3): Gyroscope bias
   - Accel bias (3): Accelerometer bias

2. **Camera States** (6 DOF each):
   - Orientation (3): Camera orientation error
   - Position (3): Camera position

### Measurement Update

When a feature is observed from multiple camera poses, MSCKF computes:

1. **Triangulation**: Estimate 3D position of feature
2. **Residual**: Reprojection error for each observation
3. **Jacobian**: How residual changes with state
4. **Null-space projection**: Eliminate feature depth parameter
5. **EKF Update**: Apply Kalman update to state and covariance

### Geographic Integration

GPS measurements provide absolute position constraints:

```
z_GPS = p + v    (measurement = position + noise)
```

This is integrated into the MSCKF framework through standard EKF updates.

## Testing

Run unit tests:

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=src tests/
```

## Performance Considerations

- **Computational Complexity**: O(n²) for n camera states in sliding window
- **Memory Usage**: Covariance matrix grows with number of camera states
- **Recommended Settings**:
  - Max camera states: 15-25
  - Feature tracking: 100-250 features
  - IMU rate: 100-200 Hz
  - Camera rate: 10-30 Hz

## Areas for Further Development

The current implementation provides a solid foundation but would benefit from:

1. **Integration with OpenCV**:
   - Replace placeholder feature detection with cv2.goodFeaturesToTrack or FAST
   - Use cv2.calcOpticalFlowPyrLK for feature tracking
   - Implement BRIEF/ORB descriptors for loop closure

2. **Loop Closure Detection**:
   - Visual vocabulary (Bag-of-Words)
   - Place recognition
   - Pose graph optimization

3. **Advanced IMU Integration**:
   - Ceres-based batch optimization
   - IMU preintegration for efficiency
   - Online calibration of IMU-camera extrinsics

4. **Visualization**:
   - Real-time trajectory plotting
   - 3D feature visualization
   - Uncertainty ellipsoids

5. **ROS Integration**:
   - ROS2 nodes for real-time operation
   - Standard message types (sensor_msgs, nav_msgs)
   - RVIZ visualization

6. **Dataset Support**:
   - EuRoC MAV dataset loader
   - KITTI dataset loader
   - Custom rosbag parser

7. **Performance Optimization**:
   - Sparse matrix operations
   - Multi-threading for feature tracking
   - GPU acceleration for vision pipeline

8. **Robustness Improvements**:
   - Better outlier rejection (M-estimators)
   - Degeneracy detection
   - Automatic reset on failure

## Contributing

Contributions are welcome! Areas of interest:

- [ ] OpenCV integration for feature detection/tracking
- [ ] ROS2 wrapper
- [ ] Visualization tools
- [ ] Dataset loaders
- [ ] Performance benchmarks
- [ ] Documentation improvements

## References

1. Mourikis, A. I., & Roumeliotis, S. I. (2007). "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"
2. Sun, K., Mohta, K., et al. (2018). "Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight"
3. Geneva, P., et al. (2020). "OpenVINS: A Research Platform for Visual-Inertial Estimation"

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{geovins2024,
  title={GEOVINS: Geographic Visual-Inertial Navigation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/geovins}
}
```

## Contact

For questions or support, please open an issue on GitHub.
