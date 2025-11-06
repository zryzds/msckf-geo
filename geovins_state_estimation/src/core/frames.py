"""
Coordinate frame definitions and transformations for GEOVINS.

Coordinate Frames:
- World (W): Global navigation frame (usually ENU - East-North-Up at initial position)
- IMU (I): IMU body frame
- Camera (C): Camera frame
- ECEF (E): Earth-Centered Earth-Fixed frame
- LLA: Latitude-Longitude-Altitude
"""
import numpy as np
from typing import Optional
from .types import Quaternion, Pose


class CoordinateFrame:
    """Base class for coordinate frames."""

    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """Rotation matrix around x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """Rotation matrix around y-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """Rotation matrix around z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


class ECEFFrame(CoordinateFrame):
    """Earth-Centered Earth-Fixed (ECEF) coordinate frame."""

    # WGS84 ellipsoid parameters
    A = 6378137.0  # semi-major axis (m)
    F = 1 / 298.257223563  # flattening
    E2 = 2 * F - F * F  # first eccentricity squared
    OMEGA_EARTH = 7.2921151467e-5  # Earth rotation rate (rad/s)

    @staticmethod
    def lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Convert Latitude, Longitude, Altitude to ECEF.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters (height above ellipsoid)

        Returns:
            ECEF coordinates [x, y, z] in meters
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Radius of curvature in prime vertical
        N = ECEFFrame.A / np.sqrt(1 - ECEFFrame.E2 * sin_lat * sin_lat)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - ECEFFrame.E2) + alt) * sin_lat

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_lla(ecef: np.ndarray) -> tuple:
        """
        Convert ECEF to Latitude, Longitude, Altitude.

        Args:
            ecef: ECEF coordinates [x, y, z] in meters

        Returns:
            (latitude, longitude, altitude) in degrees, degrees, meters
        """
        x, y, z = ecef

        # Longitude
        lon = np.arctan2(y, x)

        # Latitude and altitude (iterative method)
        p = np.sqrt(x * x + y * y)
        lat = np.arctan2(z, p * (1 - ECEFFrame.E2))

        for _ in range(5):  # Usually converges in 2-3 iterations
            sin_lat = np.sin(lat)
            N = ECEFFrame.A / np.sqrt(1 - ECEFFrame.E2 * sin_lat * sin_lat)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - ECEFFrame.E2 * N / (N + alt)))

        sin_lat = np.sin(lat)
        N = ECEFFrame.A / np.sqrt(1 - ECEFFrame.E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - N

        return np.rad2deg(lat), np.rad2deg(lon), alt


class ENUFrame(CoordinateFrame):
    """East-North-Up (ENU) local tangent plane frame."""

    def __init__(self, ref_lat: float, ref_lon: float, ref_alt: float):
        """
        Initialize ENU frame with reference LLA origin.

        Args:
            ref_lat: Reference latitude in degrees
            ref_lon: Reference longitude in degrees
            ref_alt: Reference altitude in meters
        """
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.ref_ecef = ECEFFrame.lla_to_ecef(ref_lat, ref_lon, ref_alt)

        # Rotation matrix from ECEF to ENU
        lat_rad = np.deg2rad(ref_lat)
        lon_rad = np.deg2rad(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        self.R_ecef_to_enu = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])

    def ecef_to_enu(self, ecef: np.ndarray) -> np.ndarray:
        """
        Convert ECEF coordinates to ENU.

        Args:
            ecef: ECEF coordinates [x, y, z]

        Returns:
            ENU coordinates [east, north, up]
        """
        return self.R_ecef_to_enu @ (ecef - self.ref_ecef)

    def enu_to_ecef(self, enu: np.ndarray) -> np.ndarray:
        """
        Convert ENU coordinates to ECEF.

        Args:
            enu: ENU coordinates [east, north, up]

        Returns:
            ECEF coordinates [x, y, z]
        """
        return self.R_ecef_to_enu.T @ enu + self.ref_ecef

    def lla_to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Convert LLA to ENU relative to reference.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters

        Returns:
            ENU coordinates [east, north, up]
        """
        ecef = ECEFFrame.lla_to_ecef(lat, lon, alt)
        return self.ecef_to_enu(ecef)


class IMUFrame(CoordinateFrame):
    """IMU body frame."""

    @staticmethod
    def gravity_vector(magnitude: float = 9.81) -> np.ndarray:
        """
        Get gravity vector in world frame (pointing down).

        Args:
            magnitude: Gravity magnitude in m/s^2

        Returns:
            Gravity vector [0, 0, -g]
        """
        return np.array([0.0, 0.0, -magnitude])


class CameraFrame(CoordinateFrame):
    """Camera frame."""

    def __init__(self, T_cam_imu: Optional[Pose] = None):
        """
        Initialize camera frame with extrinsic calibration.

        Args:
            T_cam_imu: Transformation from IMU to camera frame
        """
        if T_cam_imu is None:
            # Default: camera frame aligned with IMU
            T_cam_imu = Pose(
                position=np.zeros(3),
                orientation=Quaternion.identity()
            )
        self.T_cam_imu = T_cam_imu

    def imu_to_camera(self, point_imu: np.ndarray) -> np.ndarray:
        """
        Transform point from IMU frame to camera frame.

        Args:
            point_imu: 3D point in IMU frame

        Returns:
            3D point in camera frame
        """
        R = self.T_cam_imu.orientation.to_rotation_matrix()
        t = self.T_cam_imu.position
        return R @ point_imu + t

    def camera_to_imu(self, point_cam: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to IMU frame.

        Args:
            point_cam: 3D point in camera frame

        Returns:
            3D point in IMU frame
        """
        inv_pose = self.T_cam_imu.inverse()
        R = inv_pose.orientation.to_rotation_matrix()
        t = inv_pose.position
        return R @ point_cam + t


class FrameTransforms:
    """Utility class for common frame transformations."""

    @staticmethod
    def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
        w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z

        return Quaternion(
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        )

    @staticmethod
    def rotate_vector(q: Quaternion, v: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion."""
        R = q.to_rotation_matrix()
        return R @ v

    @staticmethod
    def quaternion_to_euler(q: Quaternion) -> np.ndarray:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).

        Returns:
            [roll, pitch, yaw] in radians
        """
        w, x, y, z = q.w, q.x, q.y, q.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        """
        Convert Euler angles to quaternion.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians

        Returns:
            Quaternion
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(w, x, y, z)
