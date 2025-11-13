"""
Camera model for projection and triangulation.

Supports pinhole camera model with distortion.
"""
import numpy as np
from typing import Optional, Tuple, List
from ..core.types import Pose, Quaternion, Feature
from ..core.state import CameraState


class PinholeCameraModel:
    """
    Pinhole camera model with radial-tangential distortion.

    Intrinsic parameters: fx, fy, cx, cy
    Distortion: k1, k2, p1, p2, k3
    """

    def __init__(self,
                 width: int,
                 height: int,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 k1: float = 0.0,
                 k2: float = 0.0,
                 p1: float = 0.0,
                 p2: float = 0.0,
                 k3: float = 0.0):
        """
        Initialize camera model.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            fx: Focal length in x (pixels)
            fy: Focal length in y (pixels)
            cx: Principal point x (pixels)
            cy: Principal point y (pixels)
            k1, k2, k3: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3

        # Intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def project(self, point_camera: np.ndarray, with_distortion: bool = True) -> Optional[np.ndarray]:
        """
        Project 3D point in camera frame to image plane.

        Args:
            point_camera: 3D point in camera frame (3,)
            with_distortion: Apply distortion model

        Returns:
            Pixel coordinates [u, v] or None if behind camera
        """
        X, Y, Z = point_camera

        if Z <= 0:
            return None  # Point behind camera

        # Normalized coordinates
        x = X / Z
        y = Y / Z

        if with_distortion:
            x, y = self._apply_distortion(x, y)

        # Project to pixel coordinates
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        # Check if in image bounds
        if 0 <= u < self.width and 0 <= v < self.height:
            return np.array([u, v])
        else:
            return None

    def unproject(self, pixel: np.ndarray, depth: float = 1.0, undistort: bool = True) -> np.ndarray:
        """
        Unproject pixel to 3D ray in camera frame.

        Args:
            pixel: Pixel coordinates [u, v]
            depth: Depth value (default 1.0 for unit ray)
            undistort: Remove distortion

        Returns:
            3D point in camera frame
        """
        u, v = pixel

        # Normalized coordinates
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        if undistort:
            x, y = self._remove_distortion(x, y)

        # 3D point
        return depth * np.array([x, y, 1.0])

    def _apply_distortion(self, x: float, y: float) -> Tuple[float, float]:
        """Apply radial-tangential distortion."""
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        # Tangential distortion
        x_distorted = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_distorted = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

        return x_distorted, y_distorted

    def _remove_distortion(self, x: float, y: float, max_iter: int = 5) -> Tuple[float, float]:
        """Remove distortion using iterative method."""
        x_undistorted = x
        y_undistorted = y

        for _ in range(max_iter):
            r2 = x_undistorted * x_undistorted + y_undistorted * y_undistorted
            r4 = r2 * r2
            r6 = r4 * r2

            radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

            # Tangential distortion
            dx = 2 * self.p1 * x_undistorted * y_undistorted + \
                 self.p2 * (r2 + 2 * x_undistorted * x_undistorted)
            dy = self.p1 * (r2 + 2 * y_undistorted * y_undistorted) + \
                 2 * self.p2 * x_undistorted * y_undistorted

            x_undistorted = (x - dx) / radial
            y_undistorted = (y - dy) / radial

        return x_undistorted, y_undistorted

    def compute_jacobian(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of projection function.

        Args:
            point_camera: 3D point in camera frame

        Returns:
            Jacobian matrix (2x3)
        """
        X, Y, Z = point_camera
        Z2 = Z * Z

        # Simplified: without distortion
        J = np.array([
            [self.fx / Z, 0, -self.fx * X / Z2],
            [0, self.fy / Z, -self.fy * Y / Z2]
        ])

        return J


class Triangulation:
    """Triangulation methods for 3D reconstruction."""

    @staticmethod
    def triangulate_two_view(
        cam_state_1: CameraState,
        cam_state_2: CameraState,
        point_1: np.ndarray,
        point_2: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from two camera observations.

        Args:
            cam_state_1: First camera state
            cam_state_2: Second camera state
            point_1: Normalized point in camera 1 (u, v)
            point_2: Normalized point in camera 2 (u, v)

        Returns:
            3D point in world frame or None if triangulation fails
        """
        # Get camera poses
        R1 = cam_state_1.rotation_matrix
        t1 = cam_state_1.position
        R2 = cam_state_2.rotation_matrix
        t2 = cam_state_2.position

        # Observation directions in world frame
        # For normalized coordinates, the ray is [u, v, 1] in camera frame
        ray1_cam = np.array([point_1[0], point_1[1], 1.0])
        ray2_cam = np.array([point_2[0], point_2[1], 1.0])

        # Transform to world frame
        # (inverse rotation since R is world-to-camera)
        ray1_world = R1.T @ ray1_cam
        ray2_world = R2.T @ ray2_cam

        # Normalize rays
        ray1_world = ray1_world / np.linalg.norm(ray1_world)
        ray2_world = ray2_world / np.linalg.norm(ray2_world)

        # Solve for point using midpoint method
        # p1 + lambda1 * ray1 = p2 + lambda2 * ray2
        # We want to find the point that minimizes distance between rays

        # Direction from camera 1 to camera 2
        d = t2 - t1

        # Solve for depths using least squares
        A = np.column_stack([ray1_world, -ray2_world])
        try:
            depths, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
            depth1 = depths[0]
        except np.linalg.LinAlgError:
            return None

        # Check if depth is positive
        if depth1 <= 0:
            return None

        # Compute 3D point
        point_world = t1 + depth1 * ray1_world

        return point_world

    @staticmethod
    def triangulate_multi_view(
        camera_states: List[CameraState],
        observations: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from multiple camera observations.

        Args:
            camera_states: List of camera states
            observations: List of normalized observations [(u, v), ...]

        Returns:
            3D point in world frame or None if triangulation fails
        """
        if len(camera_states) < 2 or len(camera_states) != len(observations):
            return None

        # Build linear system: A * p = b
        # For each observation: [I - ray * ray^T] * (p - t) = 0
        A = np.zeros((3 * len(camera_states), 3))
        b = np.zeros(3 * len(camera_states))

        for i, (cam_state, obs) in enumerate(zip(camera_states, observations)):
            R = cam_state.rotation_matrix
            t = cam_state.position

            # Ray in world frame
            ray_cam = np.array([obs[0], obs[1], 1.0])
            ray_world = R.T @ ray_cam
            ray_world = ray_world / np.linalg.norm(ray_world)

            # Build matrix
            I = np.eye(3)
            ray_outer = np.outer(ray_world, ray_world)
            A_i = I - ray_outer
            b_i = A_i @ t

            A[3*i:3*i+3, :] = A_i
            b[3*i:3*i+3] = b_i

        # Solve using least squares
        try:
            point_world, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        # Verify that point is in front of all cameras
        for cam_state in camera_states:
            R = cam_state.rotation_matrix
            t = cam_state.position
            point_cam = R @ (point_world - t)
            if point_cam[2] <= 0:
                return None

        return point_world

    @staticmethod
    def check_triangulation_quality(
        point_world: np.ndarray,
        camera_states: List[CameraState],
        observations: List[np.ndarray],
        max_reprojection_error: float = 3.0
    ) -> bool:
        """
        Check if triangulated point meets quality criteria.

        Args:
            point_world: 3D point in world frame
            camera_states: List of camera states
            observations: List of normalized observations
            max_reprojection_error: Maximum allowed reprojection error (pixels)

        Returns:
            True if triangulation quality is good
        """
        for cam_state, obs in zip(camera_states, observations):
            R = cam_state.rotation_matrix
            t = cam_state.position

            # Project point to camera
            point_cam = R @ (point_world - t)

            if point_cam[2] <= 0:
                return False

            # Compute normalized projection
            u_proj = point_cam[0] / point_cam[2]
            v_proj = point_cam[1] / point_cam[2]

            # Reprojection error
            error = np.sqrt((u_proj - obs[0])**2 + (v_proj - obs[1])**2)

            if error > max_reprojection_error:
                return False

        return True
