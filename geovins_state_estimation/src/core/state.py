"""
State definitions for MSCKF-based GEOVINS.

The state vector consists of:
1. IMU state (error state): orientation, position, velocity, gyro bias, accel bias
2. Camera pose states (sliding window)
3. Geographic reference state (optional)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .types import Quaternion, Pose


@dataclass
class IMUState:
    """
    IMU state representation.

    State vector (21D in error-state formulation):
    - Orientation error: 3D (small angle approximation)
    - Position: 3D
    - Velocity: 3D
    - Gyroscope bias: 3D
    - Accelerometer bias: 3D
    """
    # Nominal state
    orientation: Quaternion  # Rotation from world to IMU
    position: np.ndarray  # Position in world frame (3,)
    velocity: np.ndarray  # Velocity in world frame (3,)
    gyro_bias: np.ndarray  # Gyroscope bias (3,)
    accel_bias: np.ndarray  # Accelerometer bias (3,)

    # State ID
    state_id: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.gyro_bias = np.array(self.gyro_bias, dtype=np.float64)
        self.accel_bias = np.array(self.accel_bias, dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from world to IMU."""
        return self.orientation.to_rotation_matrix()

    def clone(self) -> 'IMUState':
        """Create a deep copy of the state."""
        return IMUState(
            orientation=Quaternion(
                self.orientation.w,
                self.orientation.x,
                self.orientation.y,
                self.orientation.z
            ),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias=self.accel_bias.copy(),
            state_id=self.state_id,
            timestamp=self.timestamp
        )


@dataclass
class CameraState:
    """
    Camera pose state in the sliding window.

    Represents a camera pose at a specific timestamp.
    """
    state_id: int
    timestamp: float
    orientation: Quaternion  # Rotation from world to camera
    position: np.ndarray  # Position in world frame (3,)

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from world to camera."""
        return self.orientation.to_rotation_matrix()

    def to_pose(self) -> Pose:
        """Convert to Pose object."""
        return Pose(position=self.position, orientation=self.orientation)


@dataclass
class GeoRefState:
    """
    Geographic reference state.

    Stores the reference location for ENU frame and ECEF transformations.
    """
    ref_latitude: float  # degrees
    ref_longitude: float  # degrees
    ref_altitude: float  # meters
    ref_ecef: np.ndarray = field(init=False)  # ECEF position (3,)

    def __post_init__(self):
        """Compute ECEF coordinates from LLA."""
        from .frames import ECEFFrame
        self.ref_ecef = ECEFFrame.lla_to_ecef(
            self.ref_latitude,
            self.ref_longitude,
            self.ref_altitude
        )


class MSCKFState:
    """
    Complete MSCKF state including IMU state and camera poses.

    State vector structure:
    [IMU state (15D), Camera states (6D each)]

    Error state is 15D + 6D * num_cameras
    """

    def __init__(self,
                 imu_state: IMUState,
                 max_camera_states: int = 20):
        """
        Initialize MSCKF state.

        Args:
            imu_state: Initial IMU state
            max_camera_states: Maximum number of camera poses in sliding window
        """
        self.imu_state = imu_state
        self.camera_states: Dict[int, CameraState] = {}  # state_id -> CameraState
        self.max_camera_states = max_camera_states

        # Geographic reference (optional)
        self.geo_ref: Optional[GeoRefState] = None

        # State counter
        self._state_counter = 0

    @property
    def num_camera_states(self) -> int:
        """Number of camera states in sliding window."""
        return len(self.camera_states)

    @property
    def state_dim(self) -> int:
        """Dimension of error state vector."""
        return 15 + 6 * self.num_camera_states  # 15 for IMU, 6 for each camera

    def add_camera_state(self, camera_state: CameraState):
        """
        Add a camera state to the sliding window.

        Args:
            camera_state: Camera state to add
        """
        if self.num_camera_states >= self.max_camera_states:
            # Remove oldest camera state
            oldest_id = min(self.camera_states.keys())
            del self.camera_states[oldest_id]

        self.camera_states[camera_state.state_id] = camera_state

    def remove_camera_state(self, state_id: int):
        """
        Remove a camera state from sliding window.

        Args:
            state_id: ID of camera state to remove
        """
        if state_id in self.camera_states:
            del self.camera_states[state_id]

    def get_camera_state(self, state_id: int) -> Optional[CameraState]:
        """
        Get camera state by ID.

        Args:
            state_id: Camera state ID

        Returns:
            Camera state or None if not found
        """
        return self.camera_states.get(state_id)

    def get_sorted_camera_states(self) -> List[CameraState]:
        """
        Get camera states sorted by state ID.

        Returns:
            List of camera states
        """
        return [self.camera_states[sid] for sid in sorted(self.camera_states.keys())]

    def set_geo_reference(self, latitude: float, longitude: float, altitude: float):
        """
        Set geographic reference location.

        Args:
            latitude: Reference latitude in degrees
            longitude: Reference longitude in degrees
            altitude: Reference altitude in meters
        """
        self.geo_ref = GeoRefState(latitude, longitude, altitude)

    def clone(self) -> 'MSCKFState':
        """Create a deep copy of the state."""
        new_state = MSCKFState(
            imu_state=self.imu_state.clone(),
            max_camera_states=self.max_camera_states
        )

        # Copy camera states
        for state_id, cam_state in self.camera_states.items():
            new_state.camera_states[state_id] = CameraState(
                state_id=cam_state.state_id,
                timestamp=cam_state.timestamp,
                orientation=Quaternion(
                    cam_state.orientation.w,
                    cam_state.orientation.x,
                    cam_state.orientation.y,
                    cam_state.orientation.z
                ),
                position=cam_state.position.copy()
            )

        # Copy geo reference
        if self.geo_ref is not None:
            new_state.geo_ref = GeoRefState(
                ref_latitude=self.geo_ref.ref_latitude,
                ref_longitude=self.geo_ref.ref_longitude,
                ref_altitude=self.geo_ref.ref_altitude
            )

        new_state._state_counter = self._state_counter

        return new_state

    def to_vector(self) -> np.ndarray:
        """
        Convert state to vector representation (for error state).

        Returns:
            State vector (error state formulation)
        """
        # IMU error state: [delta_theta (3), p (3), v (3), bg (3), ba (3)]
        imu_vector = np.concatenate([
            np.zeros(3),  # orientation error (will be updated during EKF)
            self.imu_state.position,
            self.imu_state.velocity,
            self.imu_state.gyro_bias,
            self.imu_state.accel_bias
        ])

        # Camera states: [delta_theta (3), p (3)] for each camera
        camera_vectors = []
        for state_id in sorted(self.camera_states.keys()):
            cam_state = self.camera_states[state_id]
            camera_vectors.append(np.zeros(3))  # orientation error
            camera_vectors.append(cam_state.position)

        if camera_vectors:
            return np.concatenate([imu_vector] + camera_vectors)
        else:
            return imu_vector

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MSCKFState(\n"
            f"  IMU pos: {self.imu_state.position}\n"
            f"  IMU vel: {self.imu_state.velocity}\n"
            f"  Camera states: {self.num_camera_states}\n"
            f"  State dim: {self.state_dim}\n"
            f")"
        )


def create_initial_state(
    position: np.ndarray = np.zeros(3),
    velocity: np.ndarray = np.zeros(3),
    orientation: Optional[Quaternion] = None,
    gyro_bias: np.ndarray = np.zeros(3),
    accel_bias: np.ndarray = np.zeros(3),
    timestamp: float = 0.0
) -> MSCKFState:
    """
    Create an initial MSCKF state.

    Args:
        position: Initial position in world frame
        velocity: Initial velocity in world frame
        orientation: Initial orientation (default: identity)
        gyro_bias: Initial gyroscope bias
        accel_bias: Initial accelerometer bias
        timestamp: Initial timestamp

    Returns:
        Initial MSCKF state
    """
    if orientation is None:
        orientation = Quaternion.identity()

    imu_state = IMUState(
        orientation=orientation,
        position=position,
        velocity=velocity,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        state_id=0,
        timestamp=timestamp
    )

    return MSCKFState(imu_state=imu_state)
