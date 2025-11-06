"""
Geographic measurements module.

Handles GPS and other geographic sensor measurements.
"""
import numpy as np
from typing import Optional, Tuple, List
from ..core.types import GeoMeasurement
from ..core.state import MSCKFState
from ..core.frames import ECEFFrame, ENUFrame
from dataclasses import dataclass


@dataclass
class GPSMeasurement:
    """GPS measurement data."""
    timestamp: float
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters

    # Accuracy estimates
    horizontal_accuracy: float = 5.0  # meters
    vertical_accuracy: float = 10.0  # meters

    # Number of satellites
    num_satellites: int = 0

    # Velocity (if available)
    velocity_east: Optional[float] = None  # m/s
    velocity_north: Optional[float] = None  # m/s
    velocity_up: Optional[float] = None  # m/s


class GPSProcessor:
    """
    GPS measurement processor.

    Converts GPS measurements to local ENU frame and computes measurement models.
    """

    def __init__(self,
                 ref_latitude: float,
                 ref_longitude: float,
                 ref_altitude: float,
                 use_adaptive_noise: bool = True):
        """
        Initialize GPS processor.

        Args:
            ref_latitude: Reference latitude for ENU frame
            ref_longitude: Reference longitude for ENU frame
            ref_altitude: Reference altitude for ENU frame
            use_adaptive_noise: Adaptively adjust noise based on GPS quality
        """
        self.enu_frame = ENUFrame(ref_latitude, ref_longitude, ref_altitude)
        self.use_adaptive_noise = use_adaptive_noise

        # Minimum satellite threshold for quality
        self.min_satellites = 4

        # Base noise parameters
        self.base_position_noise = 5.0  # meters
        self.base_altitude_noise = 10.0  # meters

    def process_measurement(self,
                          gps_measurement: GPSMeasurement) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process GPS measurement to ENU coordinates.

        Args:
            gps_measurement: GPS measurement

        Returns:
            (position_enu, covariance) Position in ENU frame and covariance (3x3)
        """
        # Convert to ENU
        position_enu = self.enu_frame.lla_to_enu(
            gps_measurement.latitude,
            gps_measurement.longitude,
            gps_measurement.altitude
        )

        # Compute measurement noise covariance
        if self.use_adaptive_noise:
            horizontal_noise = self._compute_adaptive_noise(gps_measurement)
            vertical_noise = gps_measurement.vertical_accuracy
        else:
            horizontal_noise = self.base_position_noise
            vertical_noise = self.base_altitude_noise

        covariance = np.diag([
            horizontal_noise**2,
            horizontal_noise**2,
            vertical_noise**2
        ])

        return position_enu, covariance

    def compute_measurement_model(self,
                                 state: MSCKFState,
                                 gps_measurement: GPSMeasurement) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute measurement model for GPS update.

        Args:
            state: Current MSCKF state
            gps_measurement: GPS measurement

        Returns:
            (H, R, residual) Measurement Jacobian, noise covariance, and residual
        """
        # Process measurement
        measured_position, R = self.process_measurement(gps_measurement)

        # Current position estimate
        estimated_position = state.imu_state.position

        # Residual
        residual = measured_position - estimated_position

        # Measurement Jacobian
        H = np.zeros((3, state.state_dim))
        H[:, 3:6] = np.eye(3)  # Position component of state

        return H, R, residual

    def compute_velocity_measurement_model(self,
                                          state: MSCKFState,
                                          gps_measurement: GPSMeasurement) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute measurement model for GPS velocity.

        Args:
            state: Current MSCKF state
            gps_measurement: GPS measurement with velocity

        Returns:
            (H, R, residual) or None if velocity not available
        """
        if (gps_measurement.velocity_east is None or
            gps_measurement.velocity_north is None or
            gps_measurement.velocity_up is None):
            return None

        # Measured velocity in ENU
        measured_velocity = np.array([
            gps_measurement.velocity_east,
            gps_measurement.velocity_north,
            gps_measurement.velocity_up
        ])

        # Current velocity estimate
        estimated_velocity = state.imu_state.velocity

        # Residual
        residual = measured_velocity - estimated_velocity

        # Measurement Jacobian
        H = np.zeros((3, state.state_dim))
        H[:, 6:9] = np.eye(3)  # Velocity component of state

        # Velocity noise (typically 0.1 m/s for good GPS)
        velocity_noise = 0.1
        R = velocity_noise**2 * np.eye(3)

        return H, R, residual

    def _compute_adaptive_noise(self, gps_measurement: GPSMeasurement) -> float:
        """
        Compute adaptive noise based on GPS quality indicators.

        Args:
            gps_measurement: GPS measurement

        Returns:
            Horizontal noise standard deviation (meters)
        """
        # Start with reported accuracy
        noise = gps_measurement.horizontal_accuracy

        # Adjust based on number of satellites
        if gps_measurement.num_satellites < self.min_satellites:
            # Poor satellite coverage, increase noise
            noise *= 2.0
        elif gps_measurement.num_satellites >= 8:
            # Good satellite coverage, reduce noise
            noise *= 0.8

        # Enforce minimum noise
        noise = max(noise, 1.0)

        return noise

    def check_measurement_quality(self, gps_measurement: GPSMeasurement) -> bool:
        """
        Check if GPS measurement meets quality criteria.

        Args:
            gps_measurement: GPS measurement

        Returns:
            True if measurement is of acceptable quality
        """
        # Check number of satellites
        if gps_measurement.num_satellites > 0 and gps_measurement.num_satellites < self.min_satellites:
            return False

        # Check accuracy
        if gps_measurement.horizontal_accuracy > 50.0:  # More than 50m error
            return False

        if gps_measurement.vertical_accuracy > 100.0:  # More than 100m error
            return False

        return True


class RTKProcessor:
    """
    RTK (Real-Time Kinematic) GPS processor.

    Provides centimeter-level accuracy with differential corrections.
    """

    def __init__(self,
                 ref_latitude: float,
                 ref_longitude: float,
                 ref_altitude: float):
        """
        Initialize RTK processor.

        Args:
            ref_latitude: Reference latitude
            ref_longitude: Reference longitude
            ref_altitude: Reference altitude
        """
        self.enu_frame = ENUFrame(ref_latitude, ref_longitude, ref_altitude)

        # RTK noise is much lower (centimeter level)
        self.horizontal_noise = 0.02  # 2 cm
        self.vertical_noise = 0.05  # 5 cm

    def process_rtk_measurement(self,
                               gps_measurement: GPSMeasurement,
                               fix_type: str = "fixed") -> Tuple[np.ndarray, np.ndarray]:
        """
        Process RTK GPS measurement.

        Args:
            gps_measurement: GPS measurement
            fix_type: RTK fix type ("fixed", "float", "single")

        Returns:
            (position_enu, covariance)
        """
        # Convert to ENU
        position_enu = self.enu_frame.lla_to_enu(
            gps_measurement.latitude,
            gps_measurement.longitude,
            gps_measurement.altitude
        )

        # Adjust noise based on fix type
        if fix_type == "fixed":
            h_noise = self.horizontal_noise
            v_noise = self.vertical_noise
        elif fix_type == "float":
            h_noise = 0.1  # 10 cm
            v_noise = 0.2  # 20 cm
        else:  # single
            h_noise = 5.0
            v_noise = 10.0

        covariance = np.diag([h_noise**2, h_noise**2, v_noise**2])

        return position_enu, covariance


class GeodeticUtils:
    """
    Utility functions for geodetic calculations.
    """

    @staticmethod
    def haversine_distance(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Compute great-circle distance between two points.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters

        lat1_rad = np.deg2rad(lat1)
        lat2_rad = np.deg2rad(lat2)
        dlat = np.deg2rad(lat2 - lat1)
        dlon = np.deg2rad(lon2 - lon1)

        a = (np.sin(dlat/2)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    @staticmethod
    def compute_bearing(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
        """
        Compute bearing from point 1 to point 2.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Bearing in degrees (0-360)
        """
        lat1_rad = np.deg2rad(lat1)
        lat2_rad = np.deg2rad(lat2)
        dlon = np.deg2rad(lon2 - lon1)

        y = np.sin(dlon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) -
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))

        bearing = np.rad2deg(np.arctan2(y, x))
        return (bearing + 360) % 360

    @staticmethod
    def destination_point(lat: float, lon: float,
                         bearing: float, distance: float) -> Tuple[float, float]:
        """
        Compute destination point given start, bearing, and distance.

        Args:
            lat, lon: Start point (degrees)
            bearing: Bearing in degrees
            distance: Distance in meters

        Returns:
            (latitude, longitude) of destination
        """
        R = 6371000  # Earth radius in meters

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        bearing_rad = np.deg2rad(bearing)

        lat2 = np.arcsin(
            np.sin(lat_rad) * np.cos(distance/R) +
            np.cos(lat_rad) * np.sin(distance/R) * np.cos(bearing_rad)
        )

        lon2 = lon_rad + np.arctan2(
            np.sin(bearing_rad) * np.sin(distance/R) * np.cos(lat_rad),
            np.cos(distance/R) - np.sin(lat_rad) * np.sin(lat2)
        )

        return np.rad2deg(lat2), np.rad2deg(lon2)


class GPSOutlierDetector:
    """
    Detector for GPS outliers and jumps.
    """

    def __init__(self,
                 max_velocity: float = 100.0,  # m/s (360 km/h)
                 max_jump: float = 50.0):  # meters
        """
        Initialize outlier detector.

        Args:
            max_velocity: Maximum plausible velocity (m/s)
            max_jump: Maximum position jump between measurements (meters)
        """
        self.max_velocity = max_velocity
        self.max_jump = max_jump

        self.last_position: Optional[np.ndarray] = None
        self.last_timestamp: Optional[float] = None

    def check_measurement(self,
                         gps_measurement: GPSMeasurement,
                         position_enu: np.ndarray) -> bool:
        """
        Check if GPS measurement is an outlier.

        Args:
            gps_measurement: GPS measurement
            position_enu: Position in ENU frame

        Returns:
            True if measurement is valid, False if outlier
        """
        if self.last_position is None:
            # First measurement, accept it
            self.last_position = position_enu
            self.last_timestamp = gps_measurement.timestamp
            return True

        # Compute displacement
        displacement = np.linalg.norm(position_enu - self.last_position)
        dt = gps_measurement.timestamp - self.last_timestamp

        if dt <= 0:
            return False

        # Check velocity
        velocity = displacement / dt
        if velocity > self.max_velocity:
            return False

        # Check jump
        if displacement > self.max_jump:
            return False

        # Update last measurement
        self.last_position = position_enu
        self.last_timestamp = gps_measurement.timestamp

        return True

    def reset(self):
        """Reset detector state."""
        self.last_position = None
        self.last_timestamp = None
