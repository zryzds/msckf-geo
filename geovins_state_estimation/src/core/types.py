"""
Data type definitions for GEOVINS state estimation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class FeatureStatus(Enum):
    """Status of a tracked feature."""
    TRACKED = 0
    LOST = 1
    MARGINALIZED = 2
    READY_FOR_UPDATE = 3


@dataclass
class IMUMeasurement:
    """IMU measurement data."""
    timestamp: float
    angular_velocity: np.ndarray  # rad/s (3,)
    linear_acceleration: np.ndarray  # m/s^2 (3,)

    def __post_init__(self):
        self.angular_velocity = np.array(self.angular_velocity, dtype=np.float64)
        self.linear_acceleration = np.array(self.linear_acceleration, dtype=np.float64)


@dataclass
class CameraMeasurement:
    """Camera measurement data."""
    timestamp: float
    image_id: int
    features: List['Feature']  # List of detected features


@dataclass
class Feature:
    """Visual feature representation."""
    feature_id: int
    u: float  # normalized image x coordinate
    v: float  # normalized image y coordinate
    camera_id: int = 0

    # Optional 3D position if triangulated
    position_w: Optional[np.ndarray] = None  # 3D position in world frame

    # Tracking information
    observations: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (camera_state_id, uv)
    status: FeatureStatus = FeatureStatus.TRACKED

    def add_observation(self, camera_state_id: int, uv: np.ndarray):
        """Add an observation of this feature."""
        self.observations.append((camera_state_id, uv))

    @property
    def num_observations(self) -> int:
        """Number of times this feature has been observed."""
        return len(self.observations)


@dataclass
class GeoFeature:
    """Geographic feature (e.g., GPS landmark, map point)."""
    feature_id: int
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters

    # Position in ECEF (Earth-Centered Earth-Fixed) frame
    position_ecef: Optional[np.ndarray] = None

    # Covariance
    covariance: Optional[np.ndarray] = None  # 3x3

    # Uncertainty
    horizontal_accuracy: float = 5.0  # meters
    vertical_accuracy: float = 10.0  # meters

    def __post_init__(self):
        if self.position_ecef is None:
            # Convert lat/lon/alt to ECEF
            self.position_ecef = self._lla_to_ecef(
                self.latitude, self.longitude, self.altitude
            )

    @staticmethod
    def _lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert Latitude, Longitude, Altitude to ECEF coordinates."""
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis (m)
        f = 1 / 298.257223563  # flattening
        e2 = 2 * f - f * f  # first eccentricity squared

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Radius of curvature in prime vertical
        N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - e2) + alt) * sin_lat

        return np.array([x, y, z])


@dataclass
class GeoMeasurement:
    """Geographic measurement (e.g., GPS reading)."""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float

    # Accuracy estimates
    horizontal_accuracy: float = 5.0
    vertical_accuracy: float = 10.0

    # Covariance in ECEF frame
    covariance: Optional[np.ndarray] = None  # 3x3

    def to_ecef(self) -> np.ndarray:
        """Convert to ECEF coordinates."""
        return GeoFeature._lla_to_ecef(self.latitude, self.longitude, self.altitude)


@dataclass
class Quaternion:
    """Quaternion representation for rotation (w, x, y, z)."""
    w: float
    x: float
    y: float
    z: float

    def __post_init__(self):
        # Normalize
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> 'Quaternion':
        """Create quaternion from rotation matrix."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return Quaternion(w, x, y, z)

    @staticmethod
    def identity() -> 'Quaternion':
        """Return identity quaternion."""
        return Quaternion(1.0, 0.0, 0.0, 0.0)


@dataclass
class Pose:
    """SE(3) pose representation."""
    position: np.ndarray  # (3,) translation
    orientation: Quaternion  # rotation as quaternion

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.orientation.to_rotation_matrix()
        T[:3, 3] = self.position
        return T

    def inverse(self) -> 'Pose':
        """Compute inverse pose."""
        R = self.orientation.to_rotation_matrix()
        R_inv = R.T
        t_inv = -R_inv @ self.position

        return Pose(
            position=t_inv,
            orientation=Quaternion.from_rotation_matrix(R_inv)
        )
